"""Deterministic evaluation metrics: citation accuracy, context precision/recall, entity coverage."""

from __future__ import annotations

import re

CITATION_PATTERNS = [
    re.compile(r"\bTX\d+\b"),
    re.compile(r"\bJE\d+\b"),
    re.compile(r"\bV\d{4}\b"),
    re.compile(r"\bDOC-[A-Z]+-\d+\b"),
    re.compile(r"\baccount\s*\d{3,4}\b", re.IGNORECASE),
    re.compile(r"\b[\w\-]+\.(?:pdf|docx|txt|md|csv|xlsx|json)\b", re.IGNORECASE),
]


def extract_citations(text: str) -> list[str]:
    found: set[str] = set()
    for pattern in CITATION_PATTERNS:
        found.update(pattern.findall(text))
    return sorted(found)


def citation_accuracy(answer: str, context: str) -> dict:
    """Fraction of cited entities in the answer that actually appear in the retrieved context."""
    citations = extract_citations(answer)
    if not citations:
        return {"score": None, "total": 0, "supported": 0, "unsupported": []}

    ctx_lower = context.lower()
    supported = [c for c in citations if c.lower() in ctx_lower]
    unsupported = [c for c in citations if c not in supported]
    return {
        "score": round(len(supported) / len(citations), 3),
        "total": len(citations),
        "supported": len(supported),
        "unsupported": unsupported,
    }


def context_precision(retrieved: list[str], expected: list[str]) -> dict:
    """Fraction of retrieved sources that are in the expected set."""
    if not retrieved:
        return {"score": 0.0, "retrieved": 0, "relevant": 0}
    expected_set = {s.lower() for s in expected}
    relevant = sum(1 for s in retrieved if s.lower() in expected_set)
    return {
        "score": round(relevant / len(retrieved), 3),
        "retrieved": len(retrieved),
        "relevant": relevant,
    }


def context_recall(retrieved: list[str], expected: list[str]) -> dict:
    """Fraction of expected sources that were actually retrieved."""
    if not expected:
        return {"score": None, "expected": 0, "found": 0}
    retrieved_set = {s.lower() for s in retrieved}
    found = sum(1 for s in expected if s.lower() in retrieved_set)
    return {
        "score": round(found / len(expected), 3),
        "expected": len(expected),
        "found": found,
    }


def entity_coverage(answer: str, expected_entities: list[str]) -> dict:
    """Fraction of expected entities mentioned anywhere in the answer."""
    if not expected_entities:
        return {"score": None, "expected": 0, "found": 0, "missing": []}
    ans_lower = answer.lower()
    hits = [e for e in expected_entities if e.lower() in ans_lower]
    missing = [e for e in expected_entities if e not in hits]
    return {
        "score": round(len(hits) / len(expected_entities), 3),
        "expected": len(expected_entities),
        "found": len(hits),
        "hits": hits,
        "missing": missing,
    }
