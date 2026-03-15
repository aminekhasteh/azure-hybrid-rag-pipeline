"""Extract text and metadata from Azure Blob Storage documents."""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from azure.storage.blob import BlobServiceClient


def _compute_content_hash(content: bytes) -> str:
    """Compute SHA-256 hash of raw content (hex string)."""
    return hashlib.sha256(content).hexdigest()

# Lazy imports for optional dependencies
def _get_document_intelligence_loader():
    from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
    return AzureAIDocumentIntelligenceLoader

def _get_pypdf_loader():
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader


def load_from_blob(
    connection_string: str,
    container_name: str,
    prefix: Optional[str] = None,
) -> list[Document]:
    """
    Load documents from Azure Blob Storage (or Azurite when using dev connection string).

    For PDFs: uses Azure AI Document Intelligence in prod (OCR for scanned);
    falls back to PyPDFLoader in dev or when Document Intelligence creds not configured.
    For MD/TXT: parses blob content directly.

    Args:
        connection_string: Azure Storage connection string (or Azurite for dev)
        container_name: Blob container name
        prefix: Optional blob prefix (e.g. "manuals/", "troubleshooting/", "policies/")

    Returns:
        List of LangChain Document objects with page_content and metadata
    """
    # Use older API version for Azurite compatibility (SDK default 2026-02-06 not supported)
    blob_service = BlobServiceClient.from_connection_string(
        connection_string, api_version="2019-12-12"
    )
    container_client = blob_service.get_container_client(container_name)

    documents: list[Document] = []
    doc_intel_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    doc_intel_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    use_document_intelligence = bool(doc_intel_endpoint and doc_intel_key)

    # List blobs (optionally filtered by prefix)
    blob_list = container_client.list_blobs(name_starts_with=prefix or "")

    for blob in blob_list:
        if not blob.name or blob.name.endswith("/"):
            continue

        ext = Path(blob.name).suffix.lower()
        blob_client = container_client.get_blob_client(blob.name)

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            blob_bytes = blob_client.download_blob().readall()
            content_hash = _compute_content_hash(blob_bytes)
            with open(tmp_path, "wb") as f:
                f.write(blob_bytes)

            if ext == ".pdf":
                try:
                    if use_document_intelligence:
                        AzureAIDocumentIntelligenceLoader = _get_document_intelligence_loader()
                        loader = AzureAIDocumentIntelligenceLoader(
                            api_endpoint=doc_intel_endpoint,
                            api_key=doc_intel_key,
                            file_path=tmp_path,
                            api_model="prebuilt-layout",
                        )
                        docs = loader.load()
                    else:
                        PyPDFLoader = _get_pypdf_loader()
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        
                    for doc in docs:
                        doc.metadata["source"] = blob.name
                        doc.metadata["filename"] = Path(blob.name).name
                        doc.metadata["content_hash"] = content_hash
                    documents.extend(docs)
                    
                except Exception as e:
                    print(f"⚠️ Failed to parse {blob.name}: {e}. Skipping...")
                    continue # Skip to the next blob instead of crashing the pipeline
            elif ext in (".md", ".txt"):
                with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": blob.name,
                            "filename": Path(blob.name).name,
                            "page": 0,
                            "content_hash": content_hash,
                        },
                    )
                )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return documents


def _load_file_with_hash(
    file_path: Path,
    root: Path,
    loader_cls,
    loader_kwargs: dict | None = None,
) -> list[Document]:
    """Load a single file, compute content hash, add to metadata. Returns list of docs."""
    loader_kwargs = loader_kwargs or {}
    with open(file_path, "rb") as f:
        content_hash = _compute_content_hash(f.read())
    loader = loader_cls(str(file_path), **loader_kwargs)
    docs = loader.load()
    # Normalize source to path relative to root (e.g. manuals/foo.pdf)
    rel = file_path.relative_to(root)
    source = str(rel).replace("\\", "/")
    filename = file_path.name
    for doc in docs:
        doc.metadata["source"] = source
        doc.metadata["filename"] = filename
        doc.metadata["content_hash"] = content_hash
    return docs


def load_from_local(path: str) -> list[Document]:
    """
    Load documents from local directory (for unit tests without Azure).

    Expects structure: path/manuals/, path/troubleshooting/, path/policies/

    Args:
        path: Root path (e.g. "data/")

    Returns:
        List of LangChain Document objects with source, filename, content_hash
    """
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    documents: list[Document] = []
    root = Path(path)

    if not root.exists():
        return documents

    # PDFs and TXT from manuals/
    manuals_dir = root / "manuals"
    if manuals_dir.exists():
        for f in manuals_dir.rglob("*.pdf"):
            if f.is_file():
                documents.extend(
                    _load_file_with_hash(f, root, PyPDFLoader)
                )
        for f in manuals_dir.rglob("*.txt"):
            if f.is_file():
                documents.extend(
                    _load_file_with_hash(
                        f, root, TextLoader,
                        {"encoding": "utf-8"},
                    )
                )

    # Markdown from troubleshooting/
    troubleshooting_dir = root / "troubleshooting"
    if troubleshooting_dir.exists():
        for f in troubleshooting_dir.rglob("*.md"):
            if f.is_file():
                documents.extend(
                    _load_file_with_hash(
                        f, root, TextLoader,
                        {"encoding": "utf-8"},
                    )
                )

    # TXT from policies/
    policies_dir = root / "policies"
    if policies_dir.exists():
        for f in policies_dir.rglob("*.txt"):
            if f.is_file():
                documents.extend(
                    _load_file_with_hash(
                        f, root, TextLoader,
                        {"encoding": "utf-8"},
                    )
                )

    return documents
