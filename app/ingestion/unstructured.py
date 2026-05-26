"""Load unstructured audit documents (PDF / DOCX / TXT / MD) into ChromaDB."""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)

DATA_DIR = Path(settings.data_dir) / "northstar_robotics_audit_dataset"

UNSTRUCTURED_FILES: list[tuple[str, str, str]] = [
    # (filename, loader_type, doc_type) — doc_type powers the richer routing layer's
    # vector_filter so policy questions don't compete with the 134-page IAASB handbook.
    ("IAASB-2023-2024-Handbook-Volume-2.pdf", "pdf", "standard"),
    ("revenue_recognition_policy.pdf", "pdf", "policy"),
    ("Retail-Lease-Agreement-Acme.pdf", "pdf", "contract"),
    ("audit_planning_memo.docx", "docx", "memo"),
    ("audit_procedures_revenue_and_expenses.docx", "docx", "procedure"),
    ("client_provided_evidence_notes.txt", "text", "evidence"),
    ("travel_expense_workpaper.md", "text", "workpaper"),
]

_embeddings: HuggingFaceEmbeddings | None = None
_client: chromadb.api.ClientAPI | None = None


def get_chroma_client() -> chromadb.api.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _embeddings


def _load_file(path: Path, file_type: str):
    if file_type == "pdf":
        loader = PyPDFLoader(str(path))
    elif file_type == "docx":
        loader = Docx2txtLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


def ingest_unstructured() -> None:
    client = get_chroma_client()

    try:
        existing = client.get_collection(settings.collection_name)
        if existing.count() > 0:
            logger.info(
                "Vector store already has %d chunks — skipping ingestion.", existing.count()
            )
            return
    except Exception:
        pass

    logger.info("Ingesting unstructured documents into ChromaDB…")

    documents = []
    for filename, ftype, doc_type in UNSTRUCTURED_FILES:
        path = DATA_DIR / filename
        if not path.exists():
            logger.warning("  missing file: %s", filename)
            continue
        try:
            docs = _load_file(path, ftype)
            for doc in docs:
                doc.metadata["source"] = filename
                doc.metadata["doc_type"] = doc_type
            documents.extend(docs)
            logger.info("  loaded %d pages from %s [%s]", len(docs), filename, doc_type)
        except Exception as exc:
            logger.error("  failed to load %s: %s", filename, exc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(chunks))

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        client=client,
        collection_name=settings.collection_name,
    )
    logger.info("Stored %d chunks in ChromaDB collection '%s'.", len(chunks), settings.collection_name)
