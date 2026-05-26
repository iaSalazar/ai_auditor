"""Regex-based router — the original.

Cheap, deterministic, transparent. Brittle to phrasing.

Emits both the original 3 fallback labels (structured/unstructured/hybrid) AND
specific categories when the question contains strong doc-type or entity-type
signals. The richer specific labels carry retrieval scope (doc_type filters etc.)
defined in `app.retrieval.routes.ROUTE_REGISTRY`, so even the cheap regex path
benefits from the IAASB-dominates fix when the question is obvious.
"""

from __future__ import annotations

import re

from app.retrieval import QueryType

STRUCTURED_PATTERNS = [
    r"\bTX\d+\b",
    r"\bJE\d+\b",
    r"\bV\d{4}\b",
    r"\bDOC-[A-Z]+-\d+\b",
    r"\baccount\s*\d{3,4}\b",
    r"\b(total|sum|count|average|how\s+many|amount|balance|debit|credit)\b",
    r"\b(transactions?|journal\s+entr(?:y|ies)|vendors?|trial\s+balance)\b",
    r"\b(support\s+status|missing\s+support|audit\s+issue|flagged|duplicates?|risk|inactive|onboarding)\b",
]

UNSTRUCTURED_PATTERNS = [
    r"\bIAASB\b",
    r"\bISA\s+\d+\b",
    r"\b(polic(?:y|ies)|standards?|procedures?|guidance|handbook|memos?|workpapers?|agreements?)\b",
    r"\b(revenue\s+recognition|lease\s+agreement|planning|performance\s+obligations?)\b",
    r"\b(what\s+does|explain|describe|define|according\s+to|summari[sz]e)\b",
]


# Specific-category signals. Each tuple = (regex, route, label_for_reason).
# Order matters — first match wins. Hybrid (mixed signal) patterns at the top.
SPECIFIC_PATTERNS: list[tuple[str, QueryType, str]] = [
    # ---- hybrid (entity + doc-type signal together) — must be checked first ----
    (
        r"\b(TX\d+|JE\d+|V\d{4}|account\s*\d{3,4}).*(polic(?:y|ies)|standards?|IAASB|ISA\b|recognition|cutoff|procedures?)\b",
        QueryType.HYBRID_TX_COMPLIANCE,
        "entity + policy/standard signal",
    ),
    (
        r"\b(TX\d+|JE\d+|V\d{4}).*(workpaper|evidence|support|contract|lease|agreement)\b",
        QueryType.HYBRID_TX_EVIDENCE,
        "entity + workpaper/evidence signal",
    ),
    # ---- specific unstructured ----
    (
        r"\b(IAASB|ISA\s+\d+|handbook|international\s+standards?\s+on\s+auditing)\b",
        QueryType.STANDARDS_LOOKUP,
        "external standards signal",
    ),
    (
        r"\b(polic(?:y|ies)|recognition\s+policy|company\s+policy|internal\s+policy)\b",
        QueryType.POLICY_LOOKUP,
        "company policy signal",
    ),
    (
        r"\b(planning\s+memo|audit\s+procedures?|risk\s+area|procedures?)\b",
        QueryType.PROCEDURE_LOOKUP,
        "procedure/planning signal",
    ),
    (
        r"\b(lease\s+agreement|workpapers?|evidence\s+notes?|contract|client[-\s]provided|client\s+evidence)\b",
        QueryType.EVIDENCE_LOOKUP,
        "evidence/contract signal",
    ),
    # ---- specific structured ----
    (
        r"\b(vendors?|suppliers?|vendor\s+master|onboarding|inactive)\b",
        QueryType.VENDOR_LOOKUP,
        "vendor signal",
    ),
    (
        r"\b(journal\s+entr(?:y|ies)|JE\d+|debit|credit|posting)\b",
        QueryType.JOURNAL_ENTRY_QUERY,
        "journal entry signal",
    ),
    (
        r"\b(trial\s+balance|account\s+balance|GL\s+balance|chart\s+of\s+accounts)\b",
        QueryType.BALANCE_QUERY,
        "trial balance signal",
    ),
    (
        r"\b(TX\d+|transactions?|missing\s+support|support\s+status|duplicates?|amounts?)\b",
        QueryType.TRANSACTION_QUERY,
        "transaction signal",
    ),
]


def classify_regex(question: str) -> tuple[QueryType, str]:
    """Returns (route, reason_string)."""
    # First try the specific-category patterns (richer routing layer)
    for pattern, route, label in SPECIFIC_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            return route, f"specific match [{label}] → {route.value}"

    # Fall back to the original 3-route classification
    structured_hits = [
        m.group() for p in STRUCTURED_PATTERNS for m in [re.search(p, question, re.IGNORECASE)] if m
    ]
    unstructured_hits = [
        m.group() for p in UNSTRUCTURED_PATTERNS for m in [re.search(p, question, re.IGNORECASE)] if m
    ]

    parts: list[str] = []
    if structured_hits:
        parts.append(f"structured signals: {structured_hits}")
    if unstructured_hits:
        parts.append(f"unstructured signals: {unstructured_hits}")
    reason = "; ".join(parts) if parts else "no specific signals matched"

    if structured_hits and unstructured_hits:
        return QueryType.HYBRID, reason
    if structured_hits:
        return QueryType.STRUCTURED, reason
    if unstructured_hits:
        return QueryType.UNSTRUCTURED, reason
    return QueryType.UNSTRUCTURED, reason
