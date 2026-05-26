"""Vector retriever: similarity search over ChromaDB."""

from __future__ import annotations

import logging

from langchain_chroma import Chroma

from app.config import settings
from app.ingestion.unstructured import get_chroma_client, get_embeddings

logger = logging.getLogger(__name__)

_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            client=get_chroma_client(),
            collection_name=settings.collection_name,
            embedding_function=get_embeddings(),
        )
    return _vectorstore


def retrieve_unstructured(question: str, where: dict | None = None) -> dict:
    """Returns dict with: context, citations, sources (filenames), chunks (with scores).

    `where` is an optional Chroma metadata filter, e.g. `{"doc_type": "policy"}` or
    `{"doc_type": {"$in": ["policy", "standard"]}}`. When provided, the candidate
    pool is narrowed BEFORE similarity ranking — this is how the richer routing layer
    fixes the IAASB-dominates problem (policy questions don't compete against the
    134-page handbook because the handbook is structurally excluded).
    """
    vs = _get_vectorstore()
    if where:
        pairs = vs.similarity_search_with_score(question, k=settings.top_k, filter=where)
        logger.info("Vector retrieval with filter %s → %d chunks", where, len(pairs))
    else:
        pairs = vs.similarity_search_with_score(question, k=settings.top_k)

    context_parts: list[str] = []
    citations: list[dict] = []
    chunks: list[dict] = []
    sources: set[str] = set()

    for i, (doc, distance) in enumerate(pairs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_display = (page + 1) if isinstance(page, int) else None

        sources.add(source)

        header = f"[{i}] Source: {source}" + (f", page {page_display}" if page_display else "")
        context_parts.append(f"{header}\n{doc.page_content}")

        excerpt = doc.page_content[:180].replace("\n", " ").strip()
        if len(doc.page_content) > 180:
            excerpt += "…"

        # ChromaDB returns L2 distance; convert to a similarity-like score (smaller distance = higher similarity).
        similarity = round(1.0 / (1.0 + float(distance)), 4)

        citations.append({
            "type": "document",
            "source": source,
            "page": page_display,
            "excerpt": excerpt,
            "similarity": similarity,
        })
        chunks.append({
            "source": source,
            "page": page_display,
            "similarity": similarity,
            "content": doc.page_content,
        })

    return {
        "context": "\n\n".join(context_parts),
        "citations": citations,
        "sources": sorted(sources),
        "chunks": chunks,
    }
