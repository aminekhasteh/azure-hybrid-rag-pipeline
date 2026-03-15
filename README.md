# Azure ETL & Retrieval - RAG Pipeline

An Azure-native retrieval-augmented search pipeline for a knowledge base containing product manuals, troubleshooting guides, and policy documents.

## Architecture

- **Blob Storage** → **Document Intelligence** → **Text Splitter** → **Azure OpenAI Embeddings** → **Azure AI Search** (with Semantic Ranking)
- **Document Catalog**: SQLite (dev) or Azure SQL (prod) tracks ingested documents for incremental updates, delete-by-document support, and access counts (incremented when documents are retrieved via search).

## Environment & Testing Strategy

To ensure a smooth developer experience while maintaining an enterprise-ready architecture, this pipeline implements an environment toggle (`ENVIRONMENT=dev|prod`).

**Development Mode (dev)**: Utilizes Azurite for local Azure Blob Storage emulation, standard OpenAI API for embeddings, Chroma vector store (cosine similarity), and SQLite document catalog. Local data is stored in hidden folders: `.azurite-data/` (created when Azurite starts), `.sql-data/`, `.chroma-data/`, and `.deepeval/` (DeepEval cache). The demo notebook uses `--local` and loads from `data/`—no Azurite required.

**Production Mode (prod)**: Fully connects to the Azure AI Foundry ecosystem, utilizing Azure Blob Storage, Azure AI Document Intelligence, Azure OpenAI, Azure AI Search, and Azure SQL.

The default submission is configured for **prod**. Please update the .env file with your Azure credentials to execute the end-to-end cloud pipeline.

## Quick Start

1. **Create virtual environment**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure credentials (for prod) or dev settings
   ```

4. **Run ingestion**
   ```bash
   # From blob (prod or Azurite): python -m src.ingest
   # From local data/ (dev):     python -m src.ingest --local
   python -m src.ingest
   ```

5. **Run search** (or use [notebooks/demo.ipynb](notebooks/demo.ipynb))
   ```bash
   python -c "from src.search import hybrid_search; print(hybrid_search('how to troubleshoot', 5))"
   ```

6. **Query RAG with DeepEval** (requires `ANTHROPIC_API_KEY` for Claude Haiku 4.5)
   ```bash
   python -m src.search "What is RAG?"
   # With StateGraph (dynamic retry on contextual failure):
   python -m src.search "What is RAG?" --use-graph
   ```

## StateGraph Dynamic Parameters

When using `--use-graph`, the RAG StateGraph dynamically adjusts retrieval parameters on retry based on DeepEval evaluation scores:

- **`top_n` (chunk count)**: Increases based on the gap between the contextual relevancy score and the threshold. A larger gap triggers a bigger bump, capped at `top_n_max` (default 15).
- **`eval_threshold`**: Relaxes by 0.05 per retry, floored at `eval_threshold_floor` (default 0.3) to prevent over-relaxation.
- **Query rephrasing**: The original query is preserved and the rephrased version is surfaced in the output alongside the parameter adjustments.

## Ingest Options

| Option | Description |
|--------|-------------|
| `--local` | Load from local `data/` instead of blob |
| `--incremental` | Skip documents unchanged since last ingest (by content hash) |
| `--delete-source PATH` | Remove a document from the catalog and vector store |
| `--upload` | Upload local `data/` to blob (Azurite or Azure), then exit |

## Chunk Lookup (Catalog → Vector Store)

The document catalog maps each document to its chunk IDs in the vector store. To look up the vector-store ID for a specific chunk by document source and chunk number (1-based), run SQL against `.sql-data/document_catalog.db`:

```sql
SELECT
    source,
    filename,
    chunk_count,
    json_extract(chunk_ids, '$[' || (10 - 1) || ']') AS chunk_id
FROM document_catalog
WHERE source = 'policies/security.txt';
```

Replace `10` with your chunk number and `'policies/security.txt'` with your document source. The `chunk_ids` column is a JSON array; `json_extract(..., '$[N]')` uses 0-based indexing, so use `(chunk_number - 1)`.

## Prerequisites

- Python 3.12+
- Azure subscription (for prod): Blob Storage, Document Intelligence, OpenAI, AI Search
- For dev: Azurite (optional), OpenAI API key (~$5)

To run the Azure SQL production ledger, ensure Microsoft ODBC Driver 18 is installed on your system and run `pip install pyodbc`.

Note: For brevity in this assignment, chunk IDs are serialized as JSON. In a true production schema, this would be normalized into a one-to-many document_chunks table to allow faster indexed lookups.

## Local Testing with Azurite

The demo notebook uses `--local` and loads from `data/`—no Azurite needed. To test blob extraction (load from Azure Blob / Azurite instead of local files):

1. **Install Azurite**: See [docs/README.md#installing-azurite](docs/README.md#installing-azurite) (VS Code extension, npm, or Docker).
2. **Start Azurite** (e.g. via VS Code "Azurite: Start" or `azurite --silent --location ./.azurite-data`). This creates `.azurite-data/`.
3. **Set `.env`**: `ENVIRONMENT=dev` and the Azurite connection string (see `.env.example`).
4. **Upload data**: `python -m src.ingest --upload` (uploads `data/` to Azurite).
5. **Run ingest** (without `--local`): `python -m src.ingest` (uses Azurite as blob source).

## Documentation

See [docs/README.md](docs/README.md) for detailed setup, architecture diagram, assumptions, and known limitations.
