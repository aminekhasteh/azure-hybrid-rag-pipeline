# Classifying and Addressing the Diversity of Errors in Retrieval-Augmented Generation Systems

**Authors:** Kin Kwan Leung, Mouloud Belbahri, Yi Sui, Alex Labach, Xueying Zhang, Stephen Rose, Jesse C. Cresswell  
**Affiliation:** Layer 6 AI, Toronto, Canada  
**Date/Source:** arXiv:2510.13975v1 [cs.CL] 15 Oct 2025  
**License:** arXiv.org perpetual non-exclusive license  

---

## Abstract
Retrieval-augmented generation (RAG) is a prevalent approach for building LLM-based question-answering systems that can take advantage of external knowledge databases. Due to the complexity of real-world RAG systems, there are many potential causes for erroneous outputs. Understanding the range of errors that can occur in practice is crucial for robust deployment. We present a new taxonomy of the error types that can occur in realistic RAG systems, examples of each, and practical advice for addressing them. Additionally, we curate a dataset of erroneous RAG responses annotated by error types. We then propose an auto-evaluation method aligned with our taxonomy that can be used in practice to track and address errors during development. Code and data are available at `github.com/layer6ai-labs/rag-error-classification`.

---

## 1. Introduction
Retrieval-augmented generation (RAG) Lewis et al. (2020) has become the dominant paradigm for applying generative large language models (LLMs) in applications where outputs must incorporate knowledge from outside of the model’s training set. This is especially valuable for grounding generation in factual information to reduce fabricated content Maynez et al. (2020); Shuster et al. (2021), and when non-public domain knowledge is required. RAG systems are already widely deployed in real-world applications to provide natural language interfaces to knowledge sources Amugongo et al. (2024), but despite their merits they can still be error-prone in practice Venkit et al. (2024); Magesh et al. (2025); Grant (2024). Due to the greater complexity of RAG pipelines compared to direct LLM generation, these errors are diverse and their causes can be difficult to trace. Deploying a RAG pipeline, especially in critical industries like healthcare, requires understanding the variety of errors that can occur in order to monitor and minimize them.

Existing work on RAG errors has generally not accounted for the complexity of real-world RAG systems and their failure modes. On the data side, widely used benchmark tasks for evaluating RAG systems are often overly simplistic, typically featuring multiple-choice questions Mihaylov et al. (2018); Guinet et al. (2024); Yang et al. (2024) or requiring short factual answers for ease of validation (e.g., Rajpurkar et al. (2016); Yang et al. (2018); Joshi et al. (2017); Kwiatkowski et al. (2019)). In practice, users expect more sophisticated answers that thoroughly explain a topic, but as a result can fail in more subtle ways.

On the model side, real RAG systems use complex multi-step pipelines that go beyond a retriever-generator pair Akkiraju et al. (2024), with additional data processing steps like adaptive chunking and reranking. Prior works that categorize RAG errors have used bare-bones pipelines benchmarked on simple datasets, and as a result have overlooked entire classes of errors, especially those associated with pre-generation steps (Barnett et al., 2024; Venkit et al., 2024). As a result, practitioners relying on these works may be left with an overly optimistic view of their RAG system’s performance, and fail to identify the scope and severity of errors afflicting them. To address these gaps, we use challenging public datasets and build a realistic RAG question-answering (QA) system reflective of those currently used in industry. We perform a deep analysis of RAG failure modes with illustrative examples, and practical advice for mitigating errors.

Finally, we develop auto-evaluation tools to classify error types according to our taxonomy. To validate these systems, we manually annotate errors producing a first-of-its-kind RAG error type dataset. The overall purpose of our taxonomy and auto-evaluation system is for practitioners to be able to identify weak links and common errors in their RAG pipelines. Hence, we provide an end-to-end demonstration of how auto-evaluation can be used by identifying the most common errors produced by our reference pipeline, and implementing targeted improvements.

Our main contributions are:
1. A novel taxonomy of errors made by RAG systems, along with examples of each and recommendations for addressing them;
2. An auto-evaluation system for identifying and classifying errors according to our taxonomy;
3. A dataset of RAG errors annotated by type.

## 2. Related Works
**RAG Error Taxonomies:** Barnett et al. (2024) provide the most comparable taxonomy of RAG errors to ours with seven types associated to select RAG components. Here, we describe a more comprehensive range of RAG errors relating to *every* step of a realistic RAG pipeline, along with specific examples and mitigation strategies, both of which are lacking in the prior work. Venkit et al. (2024) classify errors in public black-box QA systems from the perspectives of human evaluators without examining specific pipeline components or attributing errors to them. Agrawal et al. (2024) focus on RAG systems built on knowledge graphs. Yu et al. (2024a) survey RAG evaluation benchmarks and metrics but do not break down causes of errors beyond the high-level dichotomy of retrieval versus generation. Other works Huang et al. (2025); Magesh et al. (2025) that specifically focus on hallucinations overlook the nuanced ways RAG can fail. Error taxonomies for non-RAG QA settings also exist Rawte et al. (2023); Huang et al. (2025).

**RAG Auto-evaluation:** Simple QA benchmarks can be evaluated using exact match or overlap-based metrics, but realistic questions and answers require more sophisticated evaluations based in natural language understanding. Previous works have applied metrics using LLM-as-a-Judge Zheng et al. (2023) to automatically evaluate RAG outputs, including RAGChecker Ru et al. (2024), ARES Saad-Falcon et al. (2024), and RAGAs Es et al. (2024). These works are each tailored to a specific error classification scheme, with narrower scope than ours. Other prior works have developed benchmark datasets with LLM-based auto-evaluation (e.g., Zhu et al. (2024); Liu et al. (2024)), but do not probe the internals of RAG systems to give detailed information on error causes.

## 3. RAG Implementation
To demonstrate errors that can occur in realistic, production-grade RAG systems, we designed a modular pipeline reflecting common architectural patterns used in industry Bleiweiss (2024); Alexander et al. (2025). We begin with a **chunking** stage, where documents are processed to optimize granularity and relevance. The chunks are indexed using an embedding model, then queried via dense **retrieval**, which returns top candidates based on similarity to the input query. Retrieved chunks are **reranked** using a separate language model given the original query as context, which can better highlight semantic relevance. Finally, the top-ranked chunks are fed to a **generator** LLM, which produces the final answer. This architecture supports component-level variation and allows for in-depth error analysis at each stage of the pipeline. We provide complete details on the options implemented and used at each stage in Appendix A.

**Datasets:** We use the DragonBall dataset, part of the RAGEval framework (Zhu et al., 2024), an evaluation suite designed to rigorously assess RAG systems across diverse domains and scenarios. The dataset provides a broad benchmark for evaluating RAG performance in complex settings, spanning domains such as finance, law, and medicine. We used both the English and Chinese portions of the dataset with 3108 and 3601 questions respectively. We also use the CLAPnq dataset for additional analysis (Rosenthal et al., 2025), which contains high-quality reference answers and passages for 4946 natural questions relating to Wikipedia documents.

## 4. Error Classification
We present a practically grounded taxonomy of errors that can be attributed to different parts of a RAG system. Errors are grouped by the pipeline stage that caused them: chunking, retrieval, reranking, or generation. For each stage and error type, we describe the nature of the error, explain why it occurs in RAG systems, provide real examples from our RAG implementation, and finally give advice for how to reduce their occurrence.

Error types are not mutually exclusive; in practice multiple error types often co-occur, as errors early in the pipeline beget later ones. Additionally, we do not claim that our taxonomy is exhaustive, since variations in RAG architecture will lead to different error types. Our taxonomy focuses on errors that occur within RAG systems, and hence we exclude failures caused by adversarial inputs, faults in the corpus, or similar anomalies.

### 4.1 Chunking

* **E1 Overchunking:** Documents are split into excessively small or disjointed segments, causing incomplete coverage of topics. Individual chunks are fragmented or ambiguous. Errors cascade downstream when search fails to retrieve consecutive chunks.
    * *Query:* What platform did Sunrise Holidays introduce in April 2019?
    * *Chunk Excerpt:* [Excluded from chunk: In April 2019, Sunrise Holidays introduced an *online booking platform*, which greatly improved its competitiveness.] The launch of this user-friendly platform attracted more customers […]
    * *Response:* The platform introduced by Sunrise Holidays in April 2019 is not specified.
    * *Ground Truth:* An online booking platform.
    * *Cause:* The retrieved chunks come from the correct document, but the description of it as an *online booking* platform is cut off.

* **E2 Underchunking:** Chunks are too large, covering multiple topics with mixed content. Irrelevant information dilutes keywords or phrases, lowering retrieval scores on the correct chunks. Chunks provided to the generator contain extraneous content that can confuse the model.
    * *Query:* What system does CleanCo Housekeeping Services have in place to safeguard assets and ensure financial accuracy?
    * *Chunk Excerpt:* […] To mitigate risks such as increasing competition, regulatory changes, and economic uncertainties, CleanCo plans to implement *risk management* strategies through diversification and continuous monitoring. […]
    * *Response:* CleanCo has a system in place to safeguard assets and ensure financial accuracy through its *risk management* framework.
    * *Ground Truth:* Unable to answer.
    * *Cause:* Retrieved chunks contain so much tangential information that the generator uses unrelated information to answer instead of abstaining.

* **E3 Context Mismatch:** Chunks split text at arbitrary points, breaking contextual links by separating definitions from the information they support. This ambiguity causes failed retrieval downstream where keywords are missing.
    * *Query:* Why is *the Philippines* typhoon prone?
    * *Chunk A (Retrieved):* *The Philippines’* evident risk to natural disasters is due to its location. […]
    * *Chunk B (Not Retrieved):* In addition, *the country* faces the Pacific Ocean where 60% of the world’s typhoons are made. […]
    * *Response:* Due to its geographical location, climate, and topography.
    * *Ground Truth:* It faces the Pacific Ocean where 60% of the world’s typhoons are made.
    * *Cause:* Chunk B mentions “the country” but not *the Philippines* by name, which led to the retriever incorrectly assigning it low relevance.

**Improvement:** Straightforward heuristics include adjusting chunk size, where larger chunks help reduce **E1**, smaller chunks help reduce **E2**, while adding small overlaps mitigates **E3** by preserving continuity Safjan (2023). However, these approaches require careful tuning and are often insufficient on their own. Adaptive chunking strategies handle real-world variability more effectively. Structure-aware chunking preserves document logic by splitting along units like paragraphs or section headers. This works especially well for corpora with consistent structure, such as financial reports Yepes et al. (2024). Semantic chunking segments text based on meaning, such as by detecting topic shifts using cosine distance between sentence embeddings Qu et al. (2025) or spikes in perplexity Zhao et al. (2024). Hybrid strategies combine structural and semantic cues to balance coherence and topical focus. For example, S2 Chunking Verma (2025) integrates layout with embeddings and achieves strong results on document QA.

### 4.2 Retrieval

* **E4 Missed Retrieval:** Relevant chunks are not retrieved, leading the generator to give incomplete answers, fabricate information to fill gaps, or abstain unnecessarily. Misses can occur when high-similarity chunks only address part of the question.
    * *Query:* What were the large-scale financing activities undertaken by *Vanguard Media Group* in 2019 and *Energex Corp.* in 2021?
    * *Response:* *Vanguard Media Group’s* large-scale financing activities in 2019 included the acquisition of Silver Screen Studios for $200 million […]
    * *Ground Truth:* *Vanguard Media Group* raised $100 million in capital in 2019, while *Energex Corp.* conducted large-scale financing in 2021.
    * *Cause:* The query asks about two entities, but all top-ranked chunks relate to only one.

* **E5 Low Relevance:** Retrieved chunks are only loosely related to the query. Causes include an unbalanced corpus with many chunks relating to one topic, and vocabulary mismatch where the query uses different terms than the corpus.
    * *Query:* In which year is the Corporate Governance Report for Green Fields Agriculture Co.?
    * *Response:* The Corporate Governance Report does not provide a specific year, but it appears to cover events and initiatives from *2018* and 2021.
    * *Ground Truth:* 2018.
    * *Cause:* Many chunks from the Report were assigned high similarity and retrieved, but did not relate to the Report’s publication year. The generator latches onto years discussed in the content.

* **E6 Semantic Drift:** Retrieved documents match keywords, not the query’s intent, due to reliance on keyword matching rather than semantic relevance.
    * *Query:* According to the judgment of Hartford, Ashland, Court and Glenwood, Quailwood, Court, what sentences were given to the defendants *M. Cooper* and *Y. Nelson*?
    * *Response:* *M. Cooper* was sentenced to four years imprisonment.
    * *Ground Truth:* *M. Cooper* was sentenced to four years imprisonment; *Y. Nelson* was sentenced to three years imprisonment.
    * *Cause:* The retrieved chunks contain information about both defendants’ cases, but don’t include the sentence for *Y. Nelson*.

**Improvements:** Retrieval systems present the widest diversity of approaches out of the RAG stages, and hence can be tuned in many ways. First, query rewriting via keyword expansion, paraphrasing, or semantic reformulation can improve recall and precision by reducing vocabulary mismatch Ma et al. (2023). Second, hybrid retrieval that combines sparse (e.g., BM25 Robertson et al. (1994)) and dense (embedding-based) methods can improve robustness by capturing both lexical and semantic matches Ni et al. (2022). Third, top-k retrieval is often insufficient when both **E4** and **E5** are frequent. Adaptive thresholds can better balance recall and precision Sun et al. (2025). Fourth, domain-specific or fine-tuned embedding models can significantly boost retrieval accuracy and reduce hallucinations Databricks (2025). Finally, metadata filtering enhances precision by using structured fields (e.g., source or section) to prioritize relevant content Poliakov and Shvai (2025).

### 4.3 Reranking

* **E7 Low Recall:** Although the necessary chunks are retrieved, they are reranked too low and not forwarded to the generation model. Low recall produces false negatives, where relevant data is available but unintentionally ignored.
    * *Query:* According to Greenfield, Fairview, Court and Vandalia, Bayside, Court, whose sentencing time is longer, H. Price or G. Torres?
    * *Response:* G. Torres’ sentencing time is longer, four years, compared to *H. Price*, whose sentence is not mentioned in the provided context.
    * *Ground Truth:* *H. Price* was sentenced to eight years of imprisonment, while G. Torres was sentenced to four years imprisonment.
    * *Cause:* The chunk with H. Price’s sentencing was retrieved, but was ranked below the reranker threshold and not passed to the generator.

* **E8 Low Precision:** Irrelevant chunks are ranked highly and forwarded to the generation model, leading to the generation model being confused by noise. Low precision leads to false positives, where the generator is overly confident in noisy context.
    * *Query:* On which stock exchange is Retail Emporium publicly traded?
    * *Response:* The *New York* Stock Exchange.
    * *Ground Truth:* Unable to answer.
    * *Cause:* Highly ranked but irrelevant chunks reference *New York*, leading the model to make an inappropriate conclusion.

**Improvements:** Modern RAG pipelines increasingly include a reranking stage to reorder retrieved chunks based on semantic relevance Bleiweiss (2024). The two dominant approaches are cross-encoders, which jointly encode query-chunk pairs Déjean et al. (2024), and LLM-based rerankers, which use prompted models to assess candidates Sun et al. (2023). Rerankers may still miss key context (**E7**) or over-rank irrelevant chunks (**E8**) when domain relevance is poorly understood. Fine-tuning on in-domain data is the most effective way to improve performance Moreira et al. (2024); Aarsen (2025); Yu et al. (2024b), and even small amounts of labeled data can yield significant gains Aarsen (2025). More specifically, incorporating hard negatives, which are plausible but non-relevant passages, further helps the model reduce false positives (**E8**) Yu et al. (2024b).

### 4.4 Generation

* **E9 Abstention Failure:** The model should have abstained, but instead answers incorrectly. Abstention is appropriate when the answer is not available in the corpus, or when the query is ambiguous.
    * *Query:* How much did Green Fields Co. raise through large-scale financing in December 2018?
    * *Response:* Green Fields Co. did not raise any funds in December 2018.
    * *Ground Truth:* Unable to answer
    * *Cause:* The retrieved context contains information about 2018, but not about fundraising. Hence, the model assumes fundraising did not occur, but this was not stated in the corpus.

* **E10 Fabricated Content:** Although the query is answerable, the response includes unverifiable information not grounded in the retrieved chunks nor supported by external evidence.
    * *Query:* According to the hospitalization records of Southport General Hospital, summarize the past medical history of K. Mendoza.
    * *Response:* K. Mendoza has a history of *Parkinson’s disease*, with symptoms including tremors, bradykinesia, and postural instability.
    * *Ground Truth:* K. Mendoza was *generally healthy* prior to the onset of his symptoms.
    * *Cause:* The model fabricates a history of *Parkinson’s disease* which is not in the context.

* **E11 Parametric Overreliance:** The LLM relies on its internal (parametric) knowledge rather than retrieved documents. Internal knowledge may be factual, as opposed to fabricated, but irrelevant.
    * *Query:* What was the total amount of liabilities of Artistic Creations at the end of 2019?
    * *Response:* The total amount of liabilities of Artistic Creations at the end of 2019 is $150M - $20M (acquisition) - $50M (financing) - $10M (investment) + $15M (restructuring) = $85M.
    * *Ground Truth:* Unable to answer.
    * *Cause:* The model does a calculation based on parametric financial knowledge instead of acknowledging that it has insufficient information.

* **E12 Incomplete Answer:** The response is from the corpus and correct, but misses critical details.
    * *Query:* How did the dividend in July 2017 reflect AMS Inc.’s *strong financial performance*?
    * *Response:* The dividend in July 2017 reflected AMS Inc.’s *strong financial performance*, which was driven by market demand and changes in product prices, resulting in an operating income of $450 million and a net profit of $60 million.
    * *Ground Truth:* The dividend demonstrated the company’s *ability to generate sufficient profits and maintain a healthy cash flow*.
    * *Cause:* The response affirms the basis of the question, but does not analyze the situation.

* **E13 Misinterpretation:** The generator misuses or misrepresents retrieved content.
    * *Query:* According to the judgments of Trenton, Springfield and Hamilton, Harrison Courts what sentences were given to *Q. Ruiz* and *M. Ward*?
    * *Response:* *M. Ward* was sentenced to four years imprisonment, and *Q. Ruiz* has the right to appeal this judgment within 30 days.
    * *Ground Truth:* *M. Ward* was sentenced to four years of imprisonment, and *Q. Ruiz* was sentenced to 5 years of imprisonment.
    * *Cause:* The model misinterprets facts about two defendants as being related and incorrectly combines them for the answer.

* **E14 Contextual Misalignment:** The response is factual and comes from related information in the corpus, but does not address the query.
    * *Query:* According to the judgment of Preston, Lancaster, Court, summarize the *facts of M. Harris’s crimes*.
    * *Response:* M. Harris was *found guilty* of homicide and was *sentenced* to life imprisonment.
    * *Ground Truth:* M. Harris committed the crime of intentional homicide by *tampering with the brakes* of her business partner.
    * *Cause:* The response contains factual information, but discusses *sentences* rather than the requested *facts of the crimes*.

* **E15 Chronological Inconsistency:** The response presents events or facts in the wrong temporal order, or confuses past, present, or future timelines.
    * *Query:* Which company, Green Fields or JetWing, changed their board of directors earlier?
    * *Response:* Green Fields made changes to their board of directors earlier, in *December 2021*, while JetWing made changes in *October 2021*.
    * *Ground Truth:* JetWing made changes earlier in *September 2021*, while Green Fields made changes in *December 2021*.
    * *Cause:* The model provides a wrong date, and incorrectly considers Dec. earlier than Oct.

* **E16 Numerical Error:** The response includes incorrect calculations, quantities, or misrepresents numerical data from the retrieved documents.
    * *Query:* Which company reduced its liabilities more, InnovateTech or Sparkling Clean?
    * *Response:* InnovateTech reduced its liabilities by a larger amount, a *total of $50 million*, compared to Sparkling Clean’s £500,000.
    * *Ground Truth:* InnovateTech reduced its liabilities by a larger amount ($30 million) compared to Sparkling Clean (£500,000).
    * *Cause:* The model incorrectly added together two numbers from the context giving $50 million.

**Improvements:** Generation-stage errors can be mitigated through abstention modeling, answer validation, prompt engineering, and tool augmentation. For abstention failures (**E9**), models can explicitly use ambiguity detection Cole et al. (2023); Kim et al. (2023), conformal abstention Yadkori et al. (2024), or fine-tuning approaches like Trust-Align Song et al. (2025). For content fabrication (**E10**), post-generation validation such as Chain-of-Verification Dhuliawala et al. (2024) or critique modules Asai et al. (2023) encourage fact-checking and grounding. Errors related to context misuse (E11-E14) often stem from noisy retrieval; refinement modules can filter context before generation Jin et al. (2025); Chirkova et al. (2025); Wang et al. (2025). Query decomposition Lin et al. (2023) helps reduce **E12** by breaking complex queries into sub-questions, especially for comparisons, causal questions, or multi-step reasoning. For temporal and numerical errors (E15–E16), structured prompting Wei et al. (2022), timeline reasoning Bazaga et al. (2025), or tool-augmented generation with code modules like PAL Gao et al. (2023) can boost reliability.

## 5. RAGEC: RAG Error Classification
To deepen our understanding of the RAG error taxonomy and aid practitioners in applying it, we developed a RAG Error Classification system (RAGEC). In this section we describe its design, curate a dataset with human annotations to validate it, and demonstrate its use via our reference RAG system with standard research datasets.

The primary objective of RAGEC is to identify weak links in the RAG pipeline by classifying the *stage* responsible for errors. More granular error *types*, which can co-occur and overlap, are classified secondarily. Identifying the *first* stage where an error occurred enables the developer to implement targeted improvements on the RAG pipeline.

**Design.** Given a RAG pipeline and dataset, RAGEC consists of 3 steps: answer evaluation, stage classification, and error type classification. For *answer evaluation*, we prompt an LLM on each datapoint individually to determine whether the generated answer is incorrect. Included as context are the original query, ground-truth answer and documents according to the dataset, and the generated answer. This evaluation yields a subset of $N_{err}$ examples identified as incorrect. 

For *stage classification* on an erroneous example, we apply a rules-based approach to cascade over the stages, using specialized information to determine if an error is present. Starting with the generation stage, we check whether the generator had sufficient information to answer the query by computing the chunk-level recall and comparing to a threshold. If so, we conclude the generation stage caused the error. Otherwise, we continue and check if the reranker recall dropped compared to the retrieval recall, indicating that the reranker caused the error. If not, we move to the final comparison between retrieval and chunking. To differentiate these two we adapt the idea from Matton et al. (2025) to extract key concepts in the query. We then prompt the LLM as to whether each concept is included in the ground truth chunks. If query concepts fail to appear in the chunks, it indicates the chunking stage caused the error. Otherwise, by process of elimination, we conclude the retriever was responsible. This design is empirically motivated, and provided the highest alignment with our human-annotated results. Other designs we tested are described in Section B.4.

**Table 1: Stage classification agreement matrix**

| RAGEC Stage Classification | Human Annotation | | | |
| :--- | :--- | :--- | :--- | :--- |
| | **Chunking** | **Retrieval** | **Reranking** | **Generation** |
| **Chunking** | 49 | 13 | 5 | 10 |
| **Retrieval** | 24 | 85 | 12 | 40 |
| **Reranking** | 6 | 8 | 22 | 10 |
| **Generation** | 4 | 21 | 6 | 62 |

For *error type classification*, we take information relevant to the identified stage, and conduct detailed analysis using LLM-as-a-Judge (Zheng et al., 2023). For example, on a chunking error we only provide the query, ground truth answer, and chunks from the ground truth document. Other information, like the retrieved chunks or generated answer are irrelevant for classifying a chunking error type. To account for uncertainty in the classification, each erroneous example is annotated $K$ times (Wang et al., 2023). We then compute the modal vote over error types and metrics that capture the self-consistency of the LLM’s predictions.

Further details on the implementation of RAGEC, including the complete methodologies, inputs used for each step, and LLM prompts, are included in Appendix B.

**Data Curation.** No existing dataset details the types of errors which can occur in RAG systems. Hence, to validate RAGEC, we manually annotated 406 erroneous responses generated by our reference system on the DragonBall dataset as to the responsible stage, and error types present. Annotators used all available context including ground truth answer, generated response, retrieved and reranked chunks, and the corpus. First, annotators selected the earliest stage in the pipeline which demonstrated erroneous behaviour, since errors propagate through RAG systems. Subsequently, one or more error types were selected within that stage. The repo at `github.com/layer6ai-labs/rag-error-classification` contains our annotated data.

**Results & Analysis.** In the first step of RAGEC, *answer evaluation*, 27% (25%) of responses for the DragonBall-EN (CLAPnq) dataset were classified as incorrect, giving $N_{err}=832$ (1222) responses to analyze. Of these, 406 queries were manually annotated, and 377 were confirmed as erroneous, reaching an agreement rate of 92.9%.

For *stage classification*, RAGEC achieves a 57.8% agreement rate with human annotations. Details of the results are shown in Table 1. We find that human annotations more often identify the retrieval stage as causing errors, whereas RAGEC favors blaming the generator. While RAGEC’s agreement rate is not close to perfect, we tested several other methods to engineer the context for LLM-based auto-evaluation that all performed worse (described in Section B.4). Hence, RAG error classification appears to be a challenging problem due to the highly complex set of intermediate outputs produced by RAG pipelines. We encourage the community to further explore this topic by releasing our reference pipeline and annotated error data.

For *error type classification*, we query the LLM $K=10$ times for each datapoint and extract the most and second-most frequently predicted error types. Compared to human-annotated error types, RAGEC achieves an accuracy of 40.3%. Table 2 shows more details for the DragonBall-EN dataset (CLAPnq is shown in Table 5 of Section C.1). These distributions differ, showing that the nature of the dataset can affect the types of errors committed by a given RAG pipeline. This type of information can direct diagnosis and triaging of problems in RAG. For example, CLAPnq exhibited no chunking errors, as its source documents were manually curated by the dataset annotators into self-contained paragraphs designed to answer the corresponding queries. In contrast, DragonBall-EN almost never shows Parametric Overreliance (**E11**) for both English and Chinese. This can be attributed to the dataset’s domain-specific and context-dependent queries, which are unlikely to be covered by the LLM’s parametric knowledge. RAGEC also demonstrates that some error types dominate their stage, like **E3** for chunking, while others like **E5** primarily occur as a secondary error along with another type. Notably, Fabricated Content (**E10**)—often referred to as hallucination—is relatively rare in our analysis. This observation stands in contrast to the focus on hallucination in recent research and public discourse surrounding LLMs Ji et al. (2023). Our findings suggest that, in the context of RAG, other types of errors such as retrieval or chunking issues are more prevalent and may warrant greater attention. Finally, in Table 3 we examine the consistency of classifications via mode frequency, the number of the $K$ runs that agreed with the mode error type. For the majority of the queries, RAGEC is quite consistent in determining error categories. Section C.3 discusses the co-occurrence of the error categories.

**Table 2: Error-type distributions for DragonBall-EN. Mode and 2nd Mode show the original RAG pipeline, while Impr. shows the mode after targeted improvements to the RAG pipeline.**

| Error Classification | Mode | 2nd Mode | Impr. |
| :--- | :--- | :--- | :--- |
| E1 Overchunking | 0 | 19 | 0 |
| E2 Underchunking | 32 | 92 | 10 |
| E3 Context Mismatch | 207 | 32 | 78 |
| **Percentage Chunking** | **29.7%** | **32.2%** | **16.4%** |
| E4 Missed Retrieval | 229 | 27 | 122 |
| E5 Low Relevance | 33 | 107 | 8 |
| E6 Semantic Drift | 0 | 7 | 0 |
| **Percentage Retrieval** | **31.5%** | **31.8%** | **24.3%** |
| E7 Low Recall | 53 | 5 | 28 |
| E8 Low Precision | 6 | 21 | 0 |
| **Percentage Reranking** | **7.1%** | **5.9%** | **5.2%** |
| E9 Abstention Failure | 6 | 0 | 5 |
| E10 Fabricated Content | 34 | 37 | 53 |
| E11 Parametric Overreliance | 4 | 6 | 0 |
| E12 Incomplete Answer | 17 | 22 | 26 |
| E13 Misinterpretation | 178 | 42 | 164 |
| E14 Contextual Misalignment | 14 | 18 | 8 |
| E15 Chronological Inconsistency | 3 | 3 | 13 |
| E16 Numerical Error | 16 | 6 | 19 |
| **Percentage Generation** | **32.7%** | **30.2%** | **53.9%** |
| **Total** | **832** | **444** | **534** |

**Summary.** RAGEC provides an entry point for debugging and improvement of RAG pipelines, which can be complex and opaque to developers. Instead of manually sifting through individual generations, our system automatically captures intermediate information like retrieved and reranked chunks, and distills the information into high-level descriptive statistics. Hence, our focus is not on numerical metrics of performance, but on understanding the types and causes of errors so that developers can intelligently prioritize aspects to improve.

**Using RAGEC to Improve RAG.** We conduct a case study on how RAGEC can be used in practice to improve a RAG pipeline. We focus on DragonBall-EN where 832 errors were originally identified (73.3% correct). RAGEC indicates that chunking, retrieval, and generation all caused roughly the same proportion of errors (Table 2, Mode), and hence are the primary candidates for improvement. Since most chunking errors come from Context Mismatch (**E3**) where contextual links are broken across chunks, we follow the recommendation in Section 4.1 and modify the chunking strategy from fixed-length segmentation to a recursive, sentence-level segmentation of approximately the same size. In addition, RAGEC indicates that most retrieval and reranking errors come from Missed Retrieval (**E4**) and Low Recall (**E7**) which are addressed by increasing the number of chunks retrieved and passed to the generator.

With these improvements, we reran the RAG pipeline and RAGEC. The improved pipeline resulted in fewer errors overall, only 534 (82.8% correct). In Table 2 (column Impr.) we observe that chunking-, retrieval-, and reranker-related errors were significantly reduced, while generator errors stay similar as no intervention was done.

**Table 3: Distribution of mode frequency (K=10).**

| Mode Frequency | DragonBall | CLAPnq |
| :--- | :--- | :--- |
| 3 | 14 | 47 |
| 4 | 32 | 54 |
| 5 | 19 | 167 |
| 6 | 41 | 237 |
| 7 | 74 | 135 |
| 8 | 89 | 149 |
| 9 | 91 | 172 |
| 10 | 388 | 161 |

## 6. Conclusion
Our contributions in this work provide a framework for practitioners working with RAG systems to understand, categorize, track, and fix model errors. Our findings also indicate the wide range of possible avenues for improving the robustness of RAG systems in future works. While our RAG error classification method, RAGEC, achieves a useful level of agreement to human error annotations, it is far from perfect, indicating that error classification remains a challenging problem for LLM-based systems. To promote future research in this area, we release our annotated dataset of error types as a resource for the community.

### Limitations
Our RAG error taxonomy is more expansive than prior works in this vein, but still is not exhaustive of all possible errors that could occur, especially if further stages were to be added to the RAG pipeline. We focused on the most prevalent types observed in practice. 

The RAG pipeline we used for demonstrating auto-evaluation for error classification reflects the general architecture of modern pipelines used in real-world applications. However, it was not highly tuned for the DragonBall and CLAPnq datasets, for example by implementing the many improvements we listed in Section 4. Our aim was to show a starting point, where practitioners can understand errors in their system before making changes.

Additionally, our study focuses on single-turn textual queries. Expanding the evaluation framework to handle multi-turn conversations or multimodal inputs remains an important direction for future work.

Finally, we found RAG error classification according to our taxonomy to be a challenging prediction problem. Part of this stems from the difficulty of collecting labeled data. The error type labels we collected require annotators to be experts in RAG systems so that they comprehend the role and operation of each stage, and can disambiguate the error types. Hence, crowdsourcing the annotation work was not possible, and all annotation was done manually by the authors to ensure the highest quality labels. Still, LLM-as-a-Judge systems were not fully capable of reasoning over all the intermediate information created throughout the RAG pipeline, leading to rather low stage-classification accuracy. We expect this to increase as LLMs become more proficient with multi-step reasoning and assimilating information over long contexts.

---

## References
* Aarsen (2025) Tom Aarsen. 2025. Training and finetuning reranker models with sentence transformers v4.
* Agrawal et al. (2024) Garima Agrawal, Tharindu Kumarage, Zeyad Alghamdi, and Huan Liu. 2024. Mindful-RAG: A Study of Points of Failure in Retrieval Augmented Generation. In 2nd International Conference on Foundation and Large Language Models, pages 607–611.
* Akkiraju et al. (2024) Rama Akkiraju, Anbang Xu, Deepak Bora, Tan Yu, Lu An, Vishal Seth, Aaditya Shukla, Pritam Gundecha, Hridhay Mehta, Ashwin Jha, Prithvi Raj, Abhinav Balasubramanian, Murali Maram, Guru Muthusamy, Shivakesh Reddy Annepally, Sidney Knowles, Min Du, Nick Burnett, Sean Javiya, and 19 others. 2024. Facts about building retrieval augmented generation-based chatbots. arXiv:2407.07858.
* Alexander et al. (2025) John Alexander, Kristine Toliver, and Schofield McLean. 2025. Build advanced retrieval-augmented generation systems.
* Amugongo et al. (2024) Lameck Mbangula Amugongo, Pietro Mascheroni, Steven Geoffrey Brooks, Stefan Doering, and Jan Seidel. 2024. Retrieval augmented generation for large language models in healthcare: A systematic review. Preprints.
* Asai et al. (2023) Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. In The Twelfth International Conference on Learning Representations.
* Barnett et al. (2024) Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, and Mohamed Abdelrazek. 2024. Seven failure points when engineering a retrieval augmented generation system. In Proceedings of the IEEE/ACM 3rd International Conference on AI Engineering-Software Engineering for AI, pages 194–199.
* Bazaga et al. (2025) Adrián Bazaga, Rexhina Blloshmi, Bill Byrne, and Adrià de Gispert. 2025. Learning to reason over time: Timeline self-reflection for improved temporal reasoning in language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics.
* Bleiweiss (2024) Amit Bleiweiss. 2024. Enhancing RAG Pipelines with Re-Ranking.
* Chirkova et al. (2025) Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, and Stéphane Clinchant. 2025. Provence: efficient and robust context pruning for retrieval-augmented generation. In The Thirteenth International Conference on Learning Representations.
* Cole et al. (2023) Jeremy Cole, Michael Zhang, Daniel Gillick, Julian Eisenschlos, Bhuwan Dhingra, and Jacob Eisenstein. 2023. Selectively answering ambiguous questions. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 530–543.
* Databricks (2025) Databricks. 2025. Improving Retrieval and RAG with Embedding Model Fine-Tuning.
* Dhuliawala et al. (2024) Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. 2024. Chain-of-verification reduces hallucination in large language models. In Findings of the Association for Computational Linguistics ACL 2024, pages 3563–3578.
* Déjean et al. (2024) Hervé Déjean, Stéphane Clinchant, and Thibault Formal. 2024. A thorough comparison of cross-encoders and llms for reranking splade. Preprint, arXiv:2403.10407.
* Es et al. (2024) Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. RAGAs: Automated evaluation of retrieval augmented generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations.
* Gao et al. (2023) Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. PAL: Program-aided Language Models. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pages 10764–10799.
* Grant (2024) Nico Grant. 2024. Google’s AI Search errors cause a furor online. The New York Times.
* Guinet et al. (2024) Gauthier Guinet, Behrooz Omidvar-Tehrani, Anoop Deoras, and Laurent Callot. 2024. Automated evaluation of retrieval-augmented language models with task-specific exam generation. In Proceedings of the 41st International Conference on Machine Learning, volume 235.
* Huang et al. (2025) Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. ACM Trans. Inf. Syst., 43(2).
* Ji et al. (2023) Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Comput. Surv., 55(12).
* Jin et al. (2025) Jiajie Jin, Xiaoxi Li, Guanting Dong, Yuyao Zhang, Yutao Zhu, Yongkang Wu, Zhonghua Li, Qi Ye, and Zhicheng Dou. 2025. Hierarchical Document Refinement for Long-context Retrieval-augmented Generation. arXiv:2505.10413.
* Joshi et al. (2017) Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics.
* Kim et al. (2023) Gangwoo Kim, Sungdong Kim, Byeongguk Jeon, Joonsuk Park, and Jaewoo Kang. 2023. Tree of clarifications: Answering ambiguous questions with retrieval-augmented large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 996–1009.
* Kwiatkowski et al. (2019) Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics, 7.
* Lewis et al. (2020) Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459–9474.
* Lin et al. (2023) Kevin Lin, Kyle Lo, Joseph Gonzalez, and Dan Klein. 2023. Decomposing complex queries for tip-of-the-tongue retrieval. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 5521–5533.
* Liu et al. (2024) Jintao Liu, Ruixue Ding, Linhao Zhang, Pengjun Xie, and Fie Huang. 2024. CoFE-RAG: A Comprehensive Full-chain Evaluation Framework for Retrieval-Augmented Generation with Enhanced Data Diversity. arXiv:2410.12248.
* Ma et al. (2023) Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query rewriting in retrieval-augmented large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing.
* Magesh et al. (2025) Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D. Manning, and Daniel E. Ho. 2025. Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools. Journal of Empirical Legal Studies, 22(2):216–242.
* Matton et al. (2025) Katie Matton, Robert Ness, John Guttag, and Emre Kiciman. 2025. Walk the talk? measuring the faithfulness of large language model explanations. In The Thirteenth International Conference on Learning Representations.
* Maynez et al. (2020) Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. On Faithfulness and Factuality in Abstractive Summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1906–1919.
* Mihaylov et al. (2018) Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.
* Moreira et al. (2024) Gabriel de Souza P Moreira, Ronay Ak, Benedikt Schifferer, Mengyao Xu, Radek Osmulski, and Even Oldridge. 2024. Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG. arXiv:2409.07691.
* Ni et al. (2022) Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, and Yinfei Yang. 2022. Large dual encoders are generalizable retrievers. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing.
* Poliakov and Shvai (2025) Mykhailo Poliakov and Nadiya Shvai. 2025. Multi-Meta-RAG: Improving RAG for Multi-hop Queries Using Database Filtering with LLM-Extracted Metadata. In Information and Communication Technologies in Education, Research, and Industrial Applications.
* Qu et al. (2025) Renyi Qu, Ruixuan Tu, and Forrest Sheng Bao. 2025. Is semantic chunking worth the computational cost? In Findings of the Association for Computational Linguistics: NAACL 2025.
* Rajpurkar et al. (2016) Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392.
* Rawte et al. (2023) Vipula Rawte, Swagata Chakraborty, Agnibh Pathak, Anubhav Sarkar, S.M Towhidul Islam Tonmoy, Aman Chadha, Amit Sheth, and Amitava Das. 2023. The Troubling Emergence of Hallucination in Large Language Models - An Extensive Definition, Quantification, and Prescriptive Remediations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2541–2573.
* Robertson et al. (1994) Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford. 1994. Okapi at TREC-3. In Proceedings of the Third Text REtrieval Conference, pages 109–126.
* Rosenthal et al. (2025) Sara Rosenthal, Avirup Sil, Radu Florian, and Salim Roukos. 2025. CLAPnq: Cohesive Long-form Answers from Passages in Natural Questions for RAG systems. Transactions of the Association for Computational Linguistics, 13:53–72.
* Ru et al. (2024) Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang, Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang, and Zheng Zhang. 2024. RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation. In Advances in Neural Information Processing Systems, volume 37, pages 21999–22027.
* Saad-Falcon et al. (2024) Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. 2024. ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics, pages 338–354.
* Safjan (2023) Krystian Safjan. 2023. From Fixed-Size to NLP Chunking – A Deep Dive into Text Chunking Techniques.
* Shuster et al. (2021) Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021. Retrieval Augmentation Reduces Hallucination in Conversation. In Findings of the Association for Computational Linguistics: EMNLP 2021.
* Song et al. (2025) Maojia Song, Shang Hong Sim, Rishabh Bhardwaj, Hai Leong Chieu, Navonil Majumder, and Soujanya Poria. 2025. Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse. In The Thirteenth International Conference on Learning Representations.
* Sun et al. (2025) Jiashuo Sun, Xianrui Zhong, Sizhe Zhou, and Jiawei Han. 2025. DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation. arXiv:2505.07233.
* Sun et al. (2023) Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 14918–14937.
* Venkit et al. (2024) Pranav Narayanan Venkit, Philippe Laban, Yilun Zhou, Yixin Mao, and Chien-Sheng Wu. 2024. Search Engines in an AI Era: The False Promise of Factual and Verifiable Source-Cited Responses. arXiv:2410.22349.
* Verma (2025) Prashant Verma. 2025. S2 Chunking: A Hybrid Framework for Document Segmentation Through Integrated Spatial and Semantic Analysis. arXiv:2501.05485.
* Wang et al. (2023) Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.
* Wang et al. (2025) Zilong Wang, Zifeng Wang, Long Le, Steven Zheng, Swaroop Mishra, Vincent Perot, Yuwei Zhang, Anush Mattapalli, Ankur Taly, Jingbo Shang, Chen-Yu Lee, and Tomas Pfister. 2025. Speculative RAG: Enhancing retrieval augmented generation through drafting. In The Thirteenth International Conference on Learning Representations.
* Wei et al. (2022) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837.
* Yadkori et al. (2024) Yasin Abbasi Yadkori, Ilja Kuzborskij, David Stutz, András György, Adam Fisch, Arnaud Doucet, Iuliya Beloshapka, Wei-Hung Weng, Yao-Yuan Yang, Csaba Szepesvári, Ali Taylan Cemgil, and Nenad Tomasev. 2024. Mitigating LLM Hallucinations via Conformal Abstention. arXiv:2405.01563.
* Yang et al. (2024) Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Daniel Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong, Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan, Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang, and 8 others. 2024. CRAG - Comprehensive RAG Benchmark. In Advances in Neural Information Processing Systems, volume 37, pages 10470–10490.
* Yang et al. (2018) Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380.
* Yepes et al. (2024) Antonio Jimeno Yepes, Yao You, Jan Milczek, Sebastian Laverde, and Renyu Li. 2024. Financial Report Chunking for Effective Retrieval Augmented Generation. arXiv:2402.05131.
* Yu et al. (2024a) Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu. 2024a. Evaluation of retrieval-augmented generation: A survey. In CCF Conference on Big Data, pages 102–120. Springer.
* Yu et al. (2024b) Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, and Bryan Catanzaro. 2024b. RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs. In Advances in Neural Information Processing Systems, volume 37.
* Zhao et al. (2024) Jihao Zhao, Zhiyuan Ji, Yuchen Feng, Pengnian Qi, Simin Niu, Bo Tang, Feiyu Xiong, and Zhiyu Li. 2024. Meta-Chunking: Learning Efficient Text Segmentation via Logical Perception. arXiv:2410.12788.
* Zheng et al. (2023) Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36.
* Zhu et al. (2024) Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan, Zhenghao Liu, Shi Yu, Ruobing Wang, Shuo Wang, Yishan Li, Nan Zhang, Xu Han, Zhiyuan Liu, and Maosong Sun. 2024. RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework. arXiv:2408.01262.

---

## Appendix A: RAG Pipeline and Dataset Details

This section outlines the configurable components of our RAG pipeline and provides details on the specific configurations used to obtain the experimental results presented in the main paper. In Table 4, we list important hyperparameters used on the DragonBall and CLAPnq datasets. Note that the DragonBall dataset is publicly available under a CC BY-NC-SA 4.0 license. The CLAPnq dataset is publicly available under an Apache 2.0 license.

**Chunker.** We implemented multiple chunking strategies, including fixed-length chunking (with and without overlap) and recursive chunking at the sentence level. These methods enable flexible preprocessing of documents depending on the context granularity needed. When available, our implementation also handles pre-defined semantic chunks provided by the dataset. For instance, the CLAPnq dataset provides pre-chunked documents aligned with natural discourse boundaries, which we use directly in our experiments. For the DragonBall dataset, we follow the original paper’s best configuration and apply fixed-length chunking with a window size of 128 tokens and an overlap of 25 tokens (Zhu et al., 2024).

**Retrieval.** For dense retrieval, we used models from the Hugging Face repository (e.g., `gte-large-en-v1.5`) to generate embeddings of the queries and chunks. Retrieval computes vector similarity (using cosine similarity) between the query and document chunks and selects the top-k most relevant candidates. The value of k is treated as a hyperparameter. We fixed k=8 and k=5 for the DragonBall and CLAPnq datasets respectively.

**Reranker.** For the reranking stage, we use specialized LLM reranker models fine-tuned for ranking tasks (e.g., `rank-zephyr-7b-v1-full`), also sourced from Hugging Face. Given a set of retrieved candidate chunks and the original query, this model assigns a relevance score to each chunk. We then select the top k′ candidates (with k′<k from retrieval) to ensure that only the most relevant context is passed to the generator. The reranking step plays a critical role in filtering noisy or marginally relevant content returned by dense retrieval. This two-stage process is meant to enhance retrieval precision, thereby improving final answer quality. We fixed k′=5 and k′=3 for the DragonBall and CLAPnq datasets respectively. The system prompt is reproduced in Listing 1.

```text
"""
System:
You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.
User:
I will provide you with {{K}} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {{query}}
[1] {{chunk1}}
[2] {{chunk2}}
...
[K] {{chunkK}}

Search Query: {{query}}.
Rank the {{K}} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [2] > [1]. Only respond with the ranking results, do not say any word or explain.
Assistant:
"""