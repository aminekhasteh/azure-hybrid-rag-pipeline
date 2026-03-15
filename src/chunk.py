"""Chunk extracted text into retrieval-friendly segments."""

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Split documents into retrieval-friendly chunks.

    Uses RecursiveCharacterTextSplitter with token-aware sizing via tiktoken.
    Preserves metadata (source, filename, page) on each chunk and adds chunk_index
    (1-based) per document for display.

    Args:
        documents: List of LangChain Document objects
        chunk_size: Target chunk size in tokens (default 512)
        chunk_overlap: Overlap between chunks (default 100)

    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name="cl100k_base",
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    # Add chunk_index (1-based) per document for display
    source_count: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        source_count[source] += 1
        chunk.metadata["chunk_index"] = source_count[source]
    return chunks
