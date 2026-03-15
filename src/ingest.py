"""Orchestrate the full RAG pipeline and blob upload."""

import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.extract import load_from_blob, load_from_local
from src.chunk import chunk_documents
from src.embed import get_embeddings
from src.index import (
    add_document_record,
    create_index,
    delete_chunks_by_source,
    get_catalog_engine,
    get_documents_to_skip,
    init_catalog,
    load_catalog,
    remove_document_from_catalog,
    upload_documents,
)


def upload_to_blob(
    connection_string: Optional[str] = None,
    data_dir: Optional[Path] = None,
    container_name: str = "documents",
) -> None:
    """
    Upload local data/ folder to Azure Blob Storage (or Azurite).

    Creates container and uploads data/manuals/, data/troubleshooting/, data/policies/.
    """
    from azure.storage.blob import BlobServiceClient

    conn_str = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        print("Error: AZURE_STORAGE_CONNECTION_STRING not set in .env")
        print("For Azurite, use:")
        print(
            'AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"'
        )
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent
    data_path = data_dir or project_root / "data"
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        sys.exit(1)

    blob_service = BlobServiceClient.from_connection_string(
        conn_str, api_version="2019-12-12"
    )
    try:
        container_client = blob_service.create_container(container_name)
        print(f"Created container: {container_name}")
    except Exception as e:
        if "already exists" in str(e).lower() or "ContainerAlreadyExists" in str(e):
            container_client = blob_service.get_container_client(container_name)
            print(f"Using existing container: {container_name}")
        else:
            raise

    def upload_folder(local_path: Path, prefix: str) -> None:
        for f in local_path.rglob("*"):
            if f.is_file():
                rel = f.relative_to(local_path)
                blob_name = f"{prefix}{rel}".replace("\\", "/")
                print(f"  Uploading {blob_name}")
                with open(f, "rb") as fp:
                    container_client.upload_blob(blob_name, fp, overwrite=True)

    for subdir, prefix in [
        ("manuals", "manuals/"),
        ("troubleshooting", "troubleshooting/"),
        ("policies", "policies/"),
    ]:
        folder = data_path / subdir
        if folder.exists():
            print(f"\nUploading {subdir}/...")
            upload_folder(folder, prefix)

    print("\nDone. Run ingest with: python -m src.ingest --container documents")


def run_ingest(
    container_name: str = "documents",
    index_name: str = "kb-index",
    prefix: Optional[str] = None,
    connection_string: Optional[str] = None,
    use_local: bool = False,
    local_path: str = "data",
    incremental: bool = False,
    delete_source: Optional[str] = None,
) -> None:
    """
    Run the full ingestion pipeline: extract -> chunk -> embed -> index.

    Args:
        container_name: Azure Blob container name (or Azurite container)
        index_name: Azure AI Search index name (prod) or Chroma collection (dev)
        prefix: Optional blob prefix (e.g. "manuals/")
        connection_string: Azure Storage connection string (default: from env)
        use_local: If True, load from local data/ instead of blob (for tests)
        local_path: Local path when use_local=True
        incremental: If True, skip documents unchanged since last ingest (by content_hash)
        delete_source: If set, delete this document from catalog and vector store, then exit
    """
    engine = get_catalog_engine()
    init_catalog(engine)

    if delete_source:
        _run_delete_source(engine, index_name, delete_source)
        return

    connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    # Load documents
    if use_local:
        print("Loading from local data")
        documents = load_from_local(local_path)
    else:
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING required for blob loading")
        print(f"Loading from blob container: {container_name}")
        documents = load_from_blob(connection_string, container_name, prefix)

    if not documents:
        print("No documents found. Exiting.")
        return

    print(f"Loaded {len(documents)} documents")

    # Incremental: filter out unchanged documents
    if incremental:
        catalog = load_catalog(engine)
        to_skip = get_documents_to_skip(catalog, documents)
        if to_skip:
            documents = [d for d in documents if d.metadata.get("source") not in to_skip]
            print(f"Skipping {len(to_skip)} unchanged documents, processing {len(documents)}")
        if not documents:
            print("All documents unchanged. Exiting.")
            return

    # Chunk
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Embeddings and index
    embeddings = get_embeddings()

    env = os.getenv("ENVIRONMENT", "prod")
    if env == "prod":
        create_index(index_name, embedding_dim=1536)
        print(f"Index {index_name} ready")

    ids = upload_documents(index_name, chunks, embeddings)
    print(f"Uploaded {len(ids)} chunks to index")

    # Update document catalog
    from collections import defaultdict

    source_to_ids: dict[str, list[str]] = defaultdict(list)
    source_to_meta: dict[str, dict] = {}
    for chunk, cid in zip(chunks, ids):
        source = chunk.metadata.get("source")
        if source:
            source_to_ids[source].append(str(cid))
            if source not in source_to_meta:
                source_to_meta[source] = {
                    "filename": chunk.metadata.get("filename", ""),
                    "content_hash": chunk.metadata.get("content_hash", ""),
                }
    for source, chunk_ids in source_to_ids.items():
        meta = source_to_meta.get(source, {})
        add_document_record(
            engine,
            source=source,
            filename=meta.get("filename", ""),
            content_hash=meta.get("content_hash", ""),
            chunk_ids=chunk_ids,
        )
    print(f"Updated catalog for {len(source_to_ids)} documents")


def _run_delete_source(engine, index_name: str, source: str) -> None:
    """Delete a document from the catalog and vector store."""
    catalog = load_catalog(engine)
    if source not in catalog:
        print(f"Document not in catalog: {source}")
        return
    chunk_ids = catalog[source]["chunk_ids"]
    delete_chunks_by_source(index_name, source, chunk_ids)
    remove_document_from_catalog(engine, source)
    print(f"Deleted {source} ({len(chunk_ids)} chunks)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--container", default="documents", help="Blob container name")
    parser.add_argument("--index", default="kb-index", help="Index name")
    parser.add_argument("--prefix", default=None, help="Blob prefix")
    parser.add_argument("--local", action="store_true", help="Use local data/ instead of blob")
    parser.add_argument("--path", default="data", help="Local path when --local")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload local data/ to blob (Azurite or Azure) then exit",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip documents unchanged since last ingest (by content_hash)",
    )
    parser.add_argument(
        "--delete-source",
        metavar="PATH",
        default=None,
        help="Delete document from catalog and vector store, then exit",
    )
    args = parser.parse_args()

    if args.upload:
        upload_to_blob(container_name=args.container)
    else:
        run_ingest(
            container_name=args.container,
            index_name=args.index,
            prefix=args.prefix,
            use_local=args.local,
            local_path=args.path,
            incremental=args.incremental,
            delete_source=args.delete_source,
        )
