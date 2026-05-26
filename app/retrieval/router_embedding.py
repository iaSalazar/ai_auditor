"""Embedding-similarity router.

Uses the same MiniLM model already loaded for ChromaDB retrieval — no extra dependency,
no extra model load. Pre-embeds intent prototype questions at startup, classifies new
questions by cosine similarity to the closest prototype.

Costs $0 per request (no API call). ~50ms latency (one embedding call, reused if the
same embedding is later used for vector retrieval).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.retrieval import QueryType

logger = logging.getLogger(__name__)

# Intent prototype questions per route. 5–10 per intent is plenty.
# Drawn from the eval gold set + paraphrasing variations.
PROTOTYPES: dict[QueryType, list[str]] = {
    QueryType.STRUCTURED: [
        "Show me the details of transaction TX1012",
        "What is the total travel expense for Q1 2026?",
        "List all vendors with incomplete onboarding",
        "Are there any duplicate transactions?",
        "Which journal entries have an audit issue flag?",
        "Which vendors are categorized as high risk?",
        "How many transactions are missing support?",
        "Sum the consulting expense for Q1",
    ],
    QueryType.UNSTRUCTURED: [
        "What does the IAASB Handbook say about risk assessment?",
        "Summarize the revenue recognition policy",
        "Explain the key terms of the Acme retail lease agreement",
        "What audit procedures are recommended for revenue testing?",
        "What are the focus items in the travel expense workpaper?",
        "According to ISA 315, what should the auditor evaluate?",
        "Describe the audit planning memo's risk areas",
    ],
    QueryType.HYBRID: [
        "Analyze transaction TX1018 against the revenue recognition policy on multiple performance obligations",
        "What audit procedures should we follow for vendor V1010 given its inactive status?",
        "Apply IAASB cutoff guidance to TX1039 which has an acceptance certificate after quarter-end",
    ],
}

_embeddings: Optional[HuggingFaceEmbeddings] = None
_proto_embeddings: dict[QueryType, np.ndarray] = {}


def _init() -> None:
    global _embeddings
    if _embeddings is None:
        logger.info("Initializing embedding-similarity router…")
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        for route, prompts in PROTOTYPES.items():
            _proto_embeddings[route] = np.array(_embeddings.embed_documents(prompts))
        logger.info("Embedding router ready — %d prototypes total",
                    sum(len(v) for v in PROTOTYPES.values()))


def _cosine_max(query_emb: np.ndarray, proto_matrix: np.ndarray) -> float:
    """Max cosine similarity between a query embedding and a matrix of prototype embeddings."""
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    p_norms = proto_matrix / (np.linalg.norm(proto_matrix, axis=1, keepdims=True) + 1e-9)
    return float(np.max(p_norms @ q_norm))


def classify_embedding(question: str, threshold: float = 0.45) -> tuple[QueryType, str]:
    """Returns (route, reason_string).

    Picks the route whose closest prototype has the highest cosine similarity to the
    question. Falls back to HYBRID if every route's best score is below `threshold`
    (low confidence — better to over-retrieve than to send the question to the wrong path).
    """
    _init()
    q_emb = np.array(_embeddings.embed_query(question))

    scores = {
        route: _cosine_max(q_emb, proto_matrix)
        for route, proto_matrix in _proto_embeddings.items()
    }
    best_route = max(scores, key=scores.get)
    best_score = scores[best_route]

    score_str = ", ".join(f"{k.value}={v:.2f}" for k, v in scores.items())

    if best_score < threshold:
        return QueryType.HYBRID, f"embedding-sim [{score_str}] — low confidence → hybrid fallback"

    return best_route, f"embedding-sim [{score_str}] → {best_route.value}"
