"""Hybrid search, RAG chain, StateGraph, and DeepEval evaluation."""

import argparse
import os
import sys
from pathlib import Path


def _format_search_metadata(metadata: dict) -> dict:
    """
    Add display-friendly fields to metadata: document (source path), chunk (index), source_display.
    Uses full source path (e.g. policies/security.txt) for citations when available.
    """
    source = metadata.get("source") or ""
    filename = metadata.get("filename") or (Path(source).name if source else "")
    chunk_idx = metadata.get("chunk_index")
    doc_name = source or filename or "unknown"
    chunk_str = f"Chunk {chunk_idx}" if chunk_idx is not None else "Chunk ?"
    source_display = f"{doc_name} ({chunk_str})"
    return {
        **metadata,
        "document": doc_name,
        "chunk": chunk_idx,
        "source_display": source_display,
    }
from typing import Any, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Sentinel returned when no chunks meet the relevance threshold
NO_RELEVANT_CHUNKS: str = "No relevant information found in the knowledge base."

# User-facing message when DeepEval finds results not relevant
FEEDBACK_REPHRASE_OR_ADD_CHUNKS = (
    "The retrieved information does not appear relevant to your question. "
    "Please try rephrasing your question, or consider adding additional documents "
    "to the knowledge base if you believe the answer should be available."
)


# --- DeepEval env cleanup (must run before any deepeval import) ---
def _clean_empty_azure_env():
    for _key in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"):
        if not (os.environ.get(_key) or "").strip():
            os.environ.pop(_key, None)
    os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
    # Pin .deepeval to project root so notebooks/ doesn't create notebooks/.deepeval
    if "DEEPEVAL_CACHE_FOLDER" not in os.environ:
        _project_root = Path(__file__).resolve().parent.parent
        os.environ["DEEPEVAL_CACHE_FOLDER"] = str(_project_root / ".deepeval")


def hybrid_search(
    query: str,
    top_n: int = 5,
    vector_store: Optional[Any] = None,
    index_name: str = "kb-index",
    score_threshold: Optional[float] = None,
) -> list[dict] | str:
    """
    Execute hybrid search (keyword + vector) with semantic captions in prod.

    - prod: Uses azure-search-documents SDK directly for query_type="semantic",
      parses @search.captions from response.
    - dev: Uses Chroma similarity_search (no semantic captions).

    Args:
        query: Search query.
        top_n: Max number of chunks to return.
        vector_store: Chroma vector store (dev only).
        index_name: Azure Search index name (prod only).
        score_threshold: Min similarity to return chunks. When all results fall
            below this threshold, returns NO_RELEVANT_CHUNKS.
            - dev (Chroma, cosine similarity): Min similarity required
              (1 = identical, 0 = orthogonal). E.g. 0.75 keeps chunks with
              cosine similarity >= 0.75.
            - prod (Azure): Min @search.score. Tune based on your index.

    Returns:
        List of dicts with keys: content, metadata, caption (prod), score.
        Or NO_RELEVANT_CHUNKS when score_threshold is set and no chunks meet it.
    """
    if os.getenv("ENVIRONMENT") == "dev":
        result = _search_dev(query, top_n, vector_store, score_threshold)
    else:
        result = _search_prod(query, top_n, index_name, score_threshold)
    if isinstance(result, list):
        sources = [
            r["metadata"].get("source")
            for r in result
            if r.get("metadata", {}).get("source")
        ]
        sources = list(dict.fromkeys(sources))
        if sources:
            try:
                from src.index import get_catalog_engine, increment_access_count

                engine = get_catalog_engine()
                increment_access_count(engine, sources)
            except Exception:
                pass
    return result


def _search_dev(
    query: str, top_n: int, vector_store: Any, score_threshold: Optional[float] = None
) -> list[dict] | str:
    """Dev: Chroma similarity_search_with_score, converted to cosine similarity (1 = identical)."""
    if vector_store is None:
        from src.embed import get_embeddings
        from src.index import get_vector_store

        embeddings = get_embeddings()
        vector_store = get_vector_store(embeddings)

    docs_and_scores = vector_store.similarity_search_with_score(query, k=top_n)

    # Chroma returns cosine *distance*; convert to similarity: sim = 1 - dist
    docs_and_sims = [(doc, 1.0 - dist) for doc, dist in docs_and_scores]

    if score_threshold is not None:
        filtered = [
            (doc, sim)
            for doc, sim in docs_and_sims
            if sim >= score_threshold
        ]
        if not filtered:
            return NO_RELEVANT_CHUNKS
        docs_and_sims = filtered

    return [
        {
            "content": doc.page_content,
            "metadata": _format_search_metadata(dict(doc.metadata)),
            "caption": None,
            "score": float(sim),
        }
        for doc, sim in docs_and_sims
    ]


def _search_prod(
    query: str,
    top_n: int,
    index_name: str,
    score_threshold: Optional[float] = None,
) -> list[dict] | str:
    """Prod: Azure SearchClient with semantic ranking and captions."""
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.models import VectorizedQuery

    from src.embed import get_embeddings

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_KEY")
    if not endpoint or not key:
        raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY required")

    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(query)

    search_client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(key),
    )

    results = search_client.search(
        search_text=query,
        vector_queries=[
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_n * 2,
                fields="content_vector",
            )
        ],
        query_type="semantic",
        semantic_configuration_name="kb-semantic-config",
        query_caption="extractive",
        query_caption_highlight_enabled=True,
        top=top_n,
        select=["id", "content", "source", "filename", "page", "chunk_index"],
    )

    output = []
    for r in results:
        score = None
        if hasattr(r, "score") and r.score is not None:
            score = float(r.score)
        elif hasattr(r, "get") and r.get("@search.score") is not None:
            score = float(r.get("@search.score"))

        # Only filter when we have a score; if score unavailable, include (fail open)
        if (
            score_threshold is not None
            and score is not None
            and score < score_threshold
        ):
            continue

        caption = None
        if hasattr(r, "captions") and r.captions:
            caption = r.captions[0].text if r.captions else None
        elif hasattr(r, "get") and r.get("@search.captions"):
            captions = r.get("@search.captions", [])
            caption = captions[0].get("text") if captions else None

        content = getattr(r, "content", None) or r.get("content", "")
        metadata = _format_search_metadata({
            "source": getattr(r, "source", None) or r.get("source"),
            "filename": getattr(r, "filename", None) or r.get("filename"),
            "page": getattr(r, "page", None) or r.get("page"),
            "chunk_index": getattr(r, "chunk_index", None) or r.get("chunk_index"),
        })
        output.append(
            {
                "content": content,
                "metadata": metadata,
                "caption": caption,
            }
        )

    if score_threshold is not None and not output:
        return NO_RELEVANT_CHUNKS

    return output


# --- RAG chain (retrieve + generate) ---

def _get_llm():
    """Return chat LLM for generation (dev: OpenAI, prod: Azure OpenAI)."""
    if os.getenv("ENVIRONMENT") == "dev":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("ENVIRONMENT=dev requires OPENAI_API_KEY in .env")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=api_key,
        )
    else:
        from langchain_openai import AzureChatOpenAI

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise ValueError(
                "ENVIRONMENT=prod requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
            )
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-02-15",
            temperature=0,
        )


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's question based only on the "
            "provided context. Cite your sources using the bracketed numbers "
            "(e.g. [1], [2]) that label each context passage. "
            "If the context does not contain relevant information, "
            "say that you do not have enough information to answer.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def _build_context(chunks: list[dict]) -> str:
    """Build context string from retrieved chunks."""
    parts = []
    for i, c in enumerate(chunks, 1):
        content = c.get("content", "")
        parts.append(f"[{i}] {content}")
    return "\n\n".join(parts)


def rag_retrieve_and_generate(
    query: str,
    top_n: int = 5,
    vector_store: Optional[Any] = None,
    index_name: str = "kb-index",
    score_threshold: Optional[float] = None,
) -> tuple[str, list[dict] | str]:
    """
    Retrieve chunks and generate an answer.

    Returns:
        (answer, chunks_or_NO_RELEVANT_CHUNKS)
    """
    chunks = hybrid_search(
        query=query,
        top_n=top_n,
        vector_store=vector_store,
        index_name=index_name,
        score_threshold=score_threshold,
    )

    if chunks == NO_RELEVANT_CHUNKS:
        return (
            "I do not have relevant information to answer your question.",
            NO_RELEVANT_CHUNKS,
        )

    context = _build_context(chunks)
    llm = _get_llm()
    chain = RAG_PROMPT | llm
    response = chain.invoke({"context": context, "question": query})
    answer = response.content if hasattr(response, "content") else str(response)

    return answer, chunks


def query_with_evaluation(
    query: str,
    top_n: int = 5,
    vector_store: Optional[Any] = None,
    index_name: str = "kb-index",
    score_threshold: Optional[float] = None,
    eval_threshold: float = 0.5,
) -> dict:
    """
    Run RAG and evaluate with DeepEval (Claude Haiku 4.5).
    If evaluation fails, return feedback to rephrase or add chunks.

    Returns:
        {
            "answer": str,
            "passed": bool,
            "metrics": {"answer_relevancy": float, "contextual_relevancy": float},
            "feedback": str | None,  # When not passed, suggests rephrase/add chunks
        }
    """
    answer, chunks = rag_retrieve_and_generate(
        query=query,
        top_n=top_n,
        vector_store=vector_store,
        index_name=index_name,
        score_threshold=score_threshold,
    )

    if chunks == NO_RELEVANT_CHUNKS:
        return {
            "answer": answer,
            "passed": False,
            "metrics": {},
            "feedback": FEEDBACK_REPHRASE_OR_ADD_CHUNKS,
            "chunks": [],
        }

    eval_result = _evaluate_rag(query, answer, chunks, eval_threshold)

    if eval_result["passed"]:
        return {
            "answer": answer,
            "passed": True,
            "metrics": eval_result["metrics"],
            "feedback": None,
            "chunks": chunks,
        }

    return {
        "answer": answer,
        "passed": False,
        "metrics": eval_result["metrics"],
        "feedback": FEEDBACK_REPHRASE_OR_ADD_CHUNKS,
        "chunks": chunks,
    }


def _evaluate_rag(
    query: str,
    answer: str,
    chunks: list[dict],
    threshold: float = 0.5,
) -> dict:
    """Evaluate RAG output with DeepEval using Claude Haiku 4.5."""
    _clean_empty_azure_env()
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric
    from deepeval.models import AnthropicModel
    from deepeval.test_case import LLMTestCase

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "DeepEval evaluation requires ANTHROPIC_API_KEY in .env (used for Claude Haiku 4.5)"
        )

    retrieval_context = [c.get("content", "") for c in chunks]

    model = AnthropicModel(
        model="claude-haiku-4-5",
        temperature=0,
        api_key=api_key,
    )

    answer_relevancy = AnswerRelevancyMetric(
        model=model,
        threshold=threshold,
        include_reason=False,
    )
    contextual_relevancy = ContextualRelevancyMetric(
        model=model,
        threshold=threshold,
        include_reason=False,
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=retrieval_context,
    )

    answer_relevancy.measure(test_case)
    contextual_relevancy.measure(test_case)

    ar_score = getattr(answer_relevancy, "score", 0.0) or 0.0
    cr_score = getattr(contextual_relevancy, "score", 0.0) or 0.0

    ar_pass = ar_score >= threshold
    cr_pass = cr_score >= threshold
    passed = ar_pass and cr_pass

    return {
        "passed": passed,
        "metrics": {
            "answer_relevancy": ar_score,
            "contextual_relevancy": cr_score,
        },
    }


# --- RAG StateGraph (DeepEval-driven routing) ---

FEEDBACK_ANSWER_RELEVANCY = (
    "The answer does not appear relevant to your question. "
    "Please try rephrasing your question for better results."
)
FEEDBACK_CONTEXTUAL_RELEVANCY = (
    "The retrieved documents do not appear relevant to your question. "
    "Please try rephrasing your question, or consider adding additional documents "
    "to the knowledge base if you believe the answer should be available."
)
REPHRASE_PROMPT = (
    "Rephrase this question to improve retrieval from a knowledge base. "
    "Output only the rephrased question, nothing else: {query}"
)


class RAGState(TypedDict, total=False):
    """State schema for the RAG graph."""

    query: str
    original_query: str
    rephrased_query: Optional[str]
    top_n: int
    eval_threshold: float
    chunks: list[dict] | str
    answer: str
    eval_metrics: dict
    eval_passed: bool
    retry_count: int
    feedback: Optional[str]
    final_response: Optional[str]
    path_taken: list[str]


def _create_retrieve_node(
    vector_store: Any,
    index_name: str,
    default_top_n: int,
    score_threshold: Optional[float],
):
    def retrieve(state: RAGState) -> dict:
        path = state.get("path_taken", [])
        path.append("retrieve")
        current_top_n = state.get("top_n", default_top_n)
        chunks = hybrid_search(
            query=state["query"],
            top_n=current_top_n,
            vector_store=vector_store,
            index_name=index_name,
            score_threshold=score_threshold,
        )
        return {"chunks": chunks, "path_taken": path}

    return retrieve


def _create_generate_node():
    def generate(state: RAGState) -> dict:
        path = state.get("path_taken", [])
        path.append("generate")
        chunks = state["chunks"]
        if chunks == NO_RELEVANT_CHUNKS:
            return {"path_taken": path}
        context = _build_context(chunks)
        llm = _get_llm()
        chain = RAG_PROMPT | llm
        response = chain.invoke({"context": context, "question": state["query"]})
        answer = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer, "path_taken": path}

    return generate


def _create_evaluate_node(default_eval_threshold: float):
    def evaluate(state: RAGState) -> dict:
        path = state.get("path_taken", [])
        path.append("evaluate")
        chunks = state["chunks"]
        if chunks == NO_RELEVANT_CHUNKS:
            return {"path_taken": path}
        current_threshold = state.get("eval_threshold", default_eval_threshold)
        eval_result = _evaluate_rag(
            state["query"], state["answer"], chunks, threshold=current_threshold
        )
        return {
            "eval_metrics": eval_result["metrics"],
            "eval_passed": eval_result["passed"],
            "path_taken": path,
        }

    return evaluate


def _create_route_after_retrieve():
    def route(state: RAGState) -> Literal["feedback_node", "generate"]:
        if state.get("chunks") == NO_RELEVANT_CHUNKS:
            return "feedback_node"
        return "generate"

    return route


def _create_route_by_eval(max_retries: int, default_eval_threshold: float):
    def route(
        state: RAGState,
    ) -> Literal["success", "rephrase_query", "feedback_node"]:
        if state.get("eval_passed"):
            return "success"
        retry_count = state.get("retry_count", 0)
        if retry_count >= max_retries:
            return "feedback_node"
        # Use the *original* threshold for retry decisions so that relaxing
        # eval_threshold in the rephrase node doesn't prematurely cut off
        # retries.  Retry on either metric failure — rephrasing with a
        # higher top_n can improve both contextual and answer relevancy.
        metrics = state.get("eval_metrics") or {}
        ar = metrics.get("answer_relevancy", 0.0)
        cr = metrics.get("contextual_relevancy", 0.0)
        if cr < default_eval_threshold or ar < default_eval_threshold:
            return "rephrase_query"
        return "feedback_node"

    return route


def _create_rephrase_node(
    default_top_n: int,
    default_eval_threshold: float,
    top_n_max: int,
    eval_threshold_floor: float,
):
    def rephrase(state: RAGState) -> dict:
        path = state.get("path_taken", [])
        path.append("rephrase_query")

        llm = _get_llm()
        prompt = ChatPromptTemplate.from_template(REPHRASE_PROMPT)
        chain = prompt | llm
        response = chain.invoke({"query": state["query"]})
        new_query = (
            response.content.strip()
            if hasattr(response, "content")
            else str(response).strip()
        )

        retry_count = state.get("retry_count", 0) + 1
        current_top_n = state.get("top_n", default_top_n)
        current_threshold = state.get("eval_threshold", default_eval_threshold)

        metrics = state.get("eval_metrics") or {}
        cr = metrics.get("contextual_relevancy", 0.0)
        gap = max(0.0, current_threshold - cr)
        top_n_bump = max(2, round(gap * 10))
        new_top_n = min(current_top_n + top_n_bump, top_n_max)

        new_threshold = max(eval_threshold_floor, current_threshold - 0.05)

        updates: dict = {
            "query": new_query,
            "rephrased_query": new_query,
            "retry_count": retry_count,
            "top_n": new_top_n,
            "eval_threshold": new_threshold,
            "path_taken": path,
        }
        if state.get("original_query") and not state.get("rephrased_query"):
            updates["original_query"] = state["original_query"]
        return updates

    return rephrase


def _create_route_after_rephrase(max_retries: int):
    def route(state: RAGState) -> Literal["retrieve", "feedback_node"]:
        if state.get("retry_count", 0) <= max_retries:
            return "retrieve"
        return "feedback_node"

    return route


def _create_feedback_node(default_eval_threshold: float):
    def feedback(state: RAGState) -> dict:
        path = state.get("path_taken", [])
        path.append("feedback_node")
        chunks = state.get("chunks")
        if chunks == NO_RELEVANT_CHUNKS:
            msg = FEEDBACK_REPHRASE_OR_ADD_CHUNKS
        else:
            metrics = state.get("eval_metrics") or {}
            ar = metrics.get("answer_relevancy", 0.0)
            cr = metrics.get("contextual_relevancy", 0.0)
            if cr < default_eval_threshold and ar >= default_eval_threshold:
                msg = FEEDBACK_CONTEXTUAL_RELEVANCY
            elif ar < default_eval_threshold:
                msg = FEEDBACK_ANSWER_RELEVANCY
            else:
                msg = FEEDBACK_REPHRASE_OR_ADD_CHUNKS
        return {"final_response": msg, "feedback": msg, "path_taken": path}

    return feedback


def _create_success_node():
    def success(state: RAGState) -> dict:
        path = state.get("path_taken", [])
        path.append("success")
        return {"final_response": state.get("answer", ""), "path_taken": path}

    return success


def build_rag_graph(
    vector_store: Any,
    index_name: str = "kb-index",
    top_n: int = 5,
    score_threshold: Optional[float] = None,
    eval_threshold: float = 0.5,
    max_retries: int = 1,
    top_n_max: int = 15,
    eval_threshold_floor: float = 0.3,
):
    """Build and compile the RAG StateGraph."""
    from langgraph.constants import END, START
    from langgraph.graph import StateGraph

    graph = StateGraph(RAGState)
    graph.add_node(
        "retrieve",
        _create_retrieve_node(vector_store, index_name, top_n, score_threshold),
    )
    graph.add_node("generate", _create_generate_node())
    graph.add_node("evaluate", _create_evaluate_node(eval_threshold))
    graph.add_node(
        "rephrase_query",
        _create_rephrase_node(top_n, eval_threshold, top_n_max, eval_threshold_floor),
    )
    graph.add_node("feedback_node", _create_feedback_node(eval_threshold))
    graph.add_node("success", _create_success_node())

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        _create_route_after_retrieve(),
        {"feedback_node": "feedback_node", "generate": "generate"},
    )
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        _create_route_by_eval(max_retries, eval_threshold),
        {
            "success": "success",
            "rephrase_query": "rephrase_query",
            "feedback_node": "feedback_node",
        },
    )
    graph.add_edge("success", END)
    graph.add_conditional_edges(
        "rephrase_query",
        _create_route_after_rephrase(max_retries),
        {"retrieve": "retrieve", "feedback_node": "feedback_node"},
    )
    graph.add_edge("feedback_node", END)

    return graph.compile()


def query_with_graph(
    query: str,
    vector_store: Any,
    index_name: str = "kb-index",
    top_n: int = 5,
    score_threshold: Optional[float] = None,
    eval_threshold: float = 0.5,
    max_retries: int = 1,
    top_n_max: int = 15,
    eval_threshold_floor: float = 0.3,
) -> dict:
    """Run RAG through the StateGraph with DeepEval-driven routing.

    On retry the graph dynamically adjusts top_n (increases to retrieve more
    context) and eval_threshold (relaxes slightly) based on DeepEval scores.
    """
    graph = build_rag_graph(
        vector_store=vector_store,
        index_name=index_name,
        top_n=top_n,
        score_threshold=score_threshold,
        eval_threshold=eval_threshold,
        max_retries=max_retries,
        top_n_max=top_n_max,
        eval_threshold_floor=eval_threshold_floor,
    )
    initial: RAGState = {
        "query": query,
        "original_query": query,
        "rephrased_query": None,
        "top_n": top_n,
        "eval_threshold": eval_threshold,
        "chunks": [],
        "answer": "",
        "eval_metrics": {},
        "eval_passed": False,
        "retry_count": 0,
        "feedback": None,
        "final_response": None,
        "path_taken": [],
    }
    final_state = graph.invoke(initial)
    raw_chunks = final_state.get("chunks", [])
    chunks_out = raw_chunks if isinstance(raw_chunks, list) else []
    return {
        "answer": final_state.get("final_response") or final_state.get("answer", ""),
        "passed": final_state.get("eval_passed", False),
        "metrics": final_state.get("eval_metrics") or {},
        "feedback": final_state.get("feedback"),
        "path_taken": final_state.get("path_taken") or [],
        "original_query": final_state.get("original_query", query),
        "rephrased_query": final_state.get("rephrased_query"),
        "top_n": final_state.get("top_n", top_n),
        "eval_threshold": final_state.get("eval_threshold", eval_threshold),
        "chunks": chunks_out,
    }


def _query_cli() -> int:
    """CLI for querying RAG (run via python -m src.search)."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    from src.embed import get_embeddings
    from src.index import get_vector_store

    parser = argparse.ArgumentParser(description="Query RAG with DeepEval evaluation")
    parser.add_argument("query", help="Your question")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use-graph", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    if args.use_graph:
        result = query_with_graph(
            query=args.query,
            vector_store=vector_store,
            top_n=args.top_n,
            eval_threshold=args.threshold,
            max_retries=args.max_retries,
        )
    else:
        result = query_with_evaluation(
            query=args.query,
            top_n=args.top_n,
            vector_store=vector_store,
            eval_threshold=args.threshold,
        )

    print("\n--- Answer ---")
    print(result["answer"])
    if result.get("rephrased_query"):
        print(f"\n--- Rephrased Query ---")
        print(f"  Original:   {result.get('original_query', 'N/A')}")
        print(f"  Rephrased:  {result['rephrased_query']}")
    if result.get("metrics"):
        print("\n--- Metrics ---")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v:.2f}")
    if "top_n" in result:
        print(f"\n--- Dynamic Parameters ---")
        print(f"  top_n: {result['top_n']}")
        print(f"  eval_threshold: {result.get('eval_threshold', 'N/A')}")
    if result.get("path_taken"):
        print("\n--- Path ---")
        print(" -> ".join(result["path_taken"]))
    if result.get("chunks"):
        print("\n--- Citations ---")
        for i, c in enumerate(result["chunks"], 1):
            meta = c.get("metadata", {})
            print(f"  [{i}] {meta.get('source_display', 'unknown')}")
    if result.get("feedback"):
        print("\n--- Feedback ---")
        print(result["feedback"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(_query_cli())
