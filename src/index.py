"""Create vector store and upload chunks. Dev/prod router."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def get_vector_store(embeddings: Embeddings, index_name: str = "kb-index"):
    """
    Return vector store based on ENVIRONMENT.

    - dev: Chroma in-memory
    - prod: AzureSearch (requires pre-created index with semantic config)
    """
    if os.getenv("ENVIRONMENT") == "dev":
        from langchain_chroma import Chroma

        # Persist to disk so notebook/search can reuse after ingest
        # Resolve relative to project root; use hidden .chroma-data to match .sql-data, .azurite-data
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", ".chroma-data")
        if not os.path.isabs(persist_dir):
            project_root = Path(__file__).resolve().parent.parent
            persist_dir = str(project_root / persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        return Chroma(
            embedding_function=embeddings,
            collection_name="kb_docs",
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
    else:
        from langchain_community.vectorstores import AzureSearch

        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        key = os.getenv("AZURE_SEARCH_KEY")
        if not endpoint or not key:
            raise ValueError(
                "ENVIRONMENT=prod requires AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY in .env"
            )
        return AzureSearch(
            azure_search_endpoint=endpoint,
            azure_search_key=key,
            index_name=index_name,
            embedding_function=embeddings,
        )


def create_index(index_name: str, embedding_dim: int = 1536) -> None:
    """
    Create Azure AI Search index with semantic configuration (prod only).

    Call this before upload_documents when using a new index.
    """
    if os.getenv("ENVIRONMENT") == "dev":
        return  # Chroma doesn't need index creation

    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
        HnswAlgorithmConfiguration,
    )

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_KEY")
    if not endpoint or not key:
        raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY required")

    client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=True,
            vector_search_dimensions=embedding_dim,
            vector_search_profile_name="myHnswProfile",
        ),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int64, filterable=True, retrievable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int64, filterable=True, retrievable=True),
    ]

    semantic_config = SemanticConfiguration(
        name="kb-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")],
        ),
    )

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
        profiles=[
            VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="hnsw-algo")
        ],
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        semantic_search=SemanticSearch(
            configurations=[semantic_config],
            default_configuration_name="kb-semantic-config",
        ),
        vector_search=vector_search,
    )

    try:
        client.create_or_update_index(index)
    except Exception as e:
        # Index may already exist
        if "already exists" not in str(e).lower():
            raise


def upload_documents(
    index_name: str,
    chunks: list[Document],
    embeddings: Embeddings,
) -> list[str]:
    """
    Upload chunked documents to the vector store.

    Returns list of document IDs.
    """
    vector_store = get_vector_store(embeddings, index_name)
    ids = vector_store.add_documents(chunks)
    return ids if isinstance(ids, list) else [str(i) for i in ids]


# --- Document Catalog (Metadata Ledger) ---


def get_catalog_engine():
    """
    Return SQLAlchemy engine for document catalog.

    - dev: SQLite at .sql-data/document_catalog.db (or SQL_DATA_DIR env)
    - prod: Azure SQL from AZURE_SQL_CONNECTION_STRING
    """
    from sqlalchemy import create_engine
    from urllib.parse import quote_plus

    if os.getenv("ENVIRONMENT") == "dev":
        # Store in .sql-data/ alongside .azurite-data (project root, hidden)
        sql_dir = os.getenv("SQL_DATA_DIR", ".sql-data")
        if not os.path.isabs(sql_dir):
            project_root = Path(__file__).resolve().parent.parent
            sql_dir = str(project_root / sql_dir)
        os.makedirs(sql_dir, exist_ok=True)
        db_path = os.path.join(sql_dir, "document_catalog.db")
        return create_engine(f"sqlite:///{db_path}")
    else:
        conn_str = os.getenv("AZURE_SQL_CONNECTION_STRING")
        if not conn_str:
            raise ValueError(
                "ENVIRONMENT=prod requires AZURE_SQL_CONNECTION_STRING in .env"
            )
        try:
            import pyodbc  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Azure SQL document catalog requires pyodbc. "
                "Ensure Microsoft ODBC Driver 18 is installed on your system, "
                "then run: pip install pyodbc"
            ) from e
        url = f"mssql+pyodbc://?odbc_connect={quote_plus(conn_str)}"
        return create_engine(url)


def _migrate_add_access_count(engine) -> None:
    """Add access_count column to document_catalog if it does not exist (for existing DBs)."""
    from sqlalchemy import text

    dialect = engine.dialect.name
    with engine.connect() as conn:
        if dialect == "sqlite":
            result = conn.execute(text("PRAGMA table_info(document_catalog)"))
            columns = [row[1] for row in result.fetchall()]
        else:
            result = conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_NAME = 'document_catalog' AND COLUMN_NAME = 'access_count'"
                )
            )
            columns = ["access_count"] if result.fetchone() else []
        if "access_count" not in columns:
            if dialect == "sqlite":
                conn.execute(
                    text("ALTER TABLE document_catalog ADD COLUMN access_count INTEGER DEFAULT 0")
                )
            else:
                conn.execute(
                    text("ALTER TABLE document_catalog ADD access_count INT NOT NULL DEFAULT 0")
                )
            conn.commit()


def init_catalog(engine) -> None:
    """Create document_catalog table if not exists, and migrate access_count if missing."""
    from sqlalchemy import MetaData, Table, Column, String, Integer, DateTime, Text

    metadata = MetaData()
    # Use DateTime for both; SQLite and Azure SQL handle it
    # Note: For brevity in this assignment, chunk IDs are serialized as JSON. In a true
    # production schema, this would be normalized into a one-to-many document_chunks
    # table to allow faster indexed lookups.
    document_catalog = Table(
        "document_catalog",
        metadata,
        Column("source", String(1024), primary_key=True),
        Column("filename", String(512), nullable=False),
        Column("content_hash", String(64), nullable=False),
        Column("ingested_at", DateTime, nullable=False),
        Column("chunk_count", Integer, nullable=False),
        Column("chunk_ids", Text, nullable=False),
        Column("access_count", Integer, nullable=False, server_default="0"),
    )
    metadata.create_all(engine)
    _migrate_add_access_count(engine)


def load_catalog(engine) -> dict[str, dict[str, Any]]:
    """
    Load document catalog from database.

    Returns dict: {source: {filename, content_hash, ingested_at, chunk_count, chunk_ids: list, access_count: int}}
    """
    from sqlalchemy import text

    init_catalog(engine)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM document_catalog"))
        rows = result.fetchall()
    catalog: dict[str, dict[str, Any]] = {}
    for row in rows:
        source = row[0]
        chunk_ids_raw = row[5]
        chunk_ids = json.loads(chunk_ids_raw) if chunk_ids_raw else []
        access_count = int(row[6]) if len(row) > 6 else 0
        catalog[source] = {
            "filename": row[1],
            "content_hash": row[2],
            "ingested_at": row[3],
            "chunk_count": row[4],
            "chunk_ids": chunk_ids,
            "access_count": access_count,
        }
    return catalog


def add_document_record(
    engine,
    source: str,
    filename: str,
    content_hash: str,
    chunk_ids: list[str],
) -> None:
    """Insert or replace a document record in the catalog."""
    from sqlalchemy import text

    init_catalog(engine)
    now = datetime.now(timezone.utc)
    chunk_ids_json = json.dumps(chunk_ids)
    params = {
        "source": source,
        "filename": filename,
        "content_hash": content_hash,
        "ingested_at": now,
        "chunk_count": len(chunk_ids),
        "chunk_ids": chunk_ids_json,
    }
    with engine.connect() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(
                text(
                    """
                INSERT INTO document_catalog (source, filename, content_hash, ingested_at, chunk_count, chunk_ids, access_count)
                VALUES (:source, :filename, :content_hash, :ingested_at, :chunk_count, :chunk_ids, 0)
                ON CONFLICT(source) DO UPDATE SET
                    filename = :filename,
                    content_hash = :content_hash,
                    ingested_at = :ingested_at,
                    chunk_count = :chunk_count,
                    chunk_ids = :chunk_ids
                """
                ),
                params,
            )
        else:
            conn.execute(
                text(
                    """
                MERGE document_catalog AS target
                USING (SELECT :source AS source) AS src
                ON target.source = src.source
                WHEN MATCHED THEN UPDATE SET
                    filename = :filename,
                    content_hash = :content_hash,
                    ingested_at = :ingested_at,
                    chunk_count = :chunk_count,
                    chunk_ids = :chunk_ids
                WHEN NOT MATCHED THEN INSERT
                    (source, filename, content_hash, ingested_at, chunk_count, chunk_ids, access_count)
                    VALUES (:source, :filename, :content_hash, :ingested_at, :chunk_count, :chunk_ids, 0);
                """
                ),
                params,
            )
        conn.commit()


def increment_access_count(engine, sources: list[str]) -> None:
    """Increment access_count for each source in the document catalog."""
    if not sources:
        return
    from sqlalchemy import text

    unique_sources = list(dict.fromkeys(sources))  # preserve order, dedupe
    placeholders = ", ".join(f":s{i}" for i in range(len(unique_sources)))
    params = {f"s{i}": s for i, s in enumerate(unique_sources)}
    with engine.connect() as conn:
        conn.execute(
            text(f"UPDATE document_catalog SET access_count = access_count + 1 WHERE source IN ({placeholders})"),
            params,
        )
        conn.commit()


def get_documents_to_skip(
    catalog: dict[str, dict[str, Any]], documents: list[Document]
) -> set[str]:
    """
    Return set of source paths to skip (unchanged documents).

    A document is skipped if it exists in catalog with same content_hash.
    """
    to_skip: set[str] = set()
    seen_sources: set[str] = set()
    for doc in documents:
        source = doc.metadata.get("source")
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)
        if source in catalog:
            doc_hash = doc.metadata.get("content_hash")
            if doc_hash and catalog[source]["content_hash"] == doc_hash:
                to_skip.add(source)
    return to_skip


def delete_chunks_by_source(
    index_name: str,
    source: str,
    chunk_ids: list[str],
    embeddings=None,
) -> None:
    """Remove chunks for a document from the vector store."""
    if os.getenv("ENVIRONMENT") == "dev":
        from src.embed import get_embeddings

        emb = embeddings or get_embeddings()
        vector_store = get_vector_store(emb, index_name)
        vector_store.delete(ids=chunk_ids)
    else:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        key = os.getenv("AZURE_SEARCH_KEY")
        if not endpoint or not key:
            raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY required")
        client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(key),
        )
        docs_to_delete = [{"id": cid} for cid in chunk_ids]
        client.delete_documents(documents=docs_to_delete)


def remove_document_from_catalog(engine, source: str) -> None:
    """Delete a document record from the catalog."""
    from sqlalchemy import text

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM document_catalog WHERE source = :source"), {"source": source})
        conn.commit()
