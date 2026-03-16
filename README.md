# Azure Hybrid RAG Pipeline

An Azure-native retrieval-augmented search pipeline for a knowledge base containing product manuals, troubleshooting guides, and policy documents.

## Architecture

- **Blob Storage** → **Document Intelligence** (PDF OCR) → **Text Splitter** → **Azure OpenAI Embeddings** → **Azure AI Search** (with Semantic Ranking)
- **Document Catalog**: SQLite (dev) or Azure SQL (prod) tracks ingested documents for incremental updates, delete-by-document support, and access counts.
- **Evaluation**: DeepEval-driven LangGraph StateGraph with dynamic retry on either metric failure (contextual or answer relevancy), query rephrasing, and parameter adjustment.

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit with your credentials
python -m src.ingest --local
python -m src.search "How to troubleshoot a RAG pipeline?"
```

See [docs/README.md](docs/README.md) for full setup, architecture diagram, environment configuration, Azurite instructions, and known limitations.
