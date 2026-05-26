"""Shared types for retrieval and routing."""

from enum import Enum


class QueryType(str, Enum):
    """Where the query gets routed to retrieve evidence from.

    The original 3 are the *fallback layer* — used when the LLM router can't
    confidently pick a more specific category. Regex and embedding routers only
    emit the original 3; specific categories below are exclusive to the LLM router.
    Each route's retrieval scope (which SQL tables, which Chroma `where` filter)
    is defined in `app.retrieval.routes.ROUTE_REGISTRY`.
    """
    # Fallback layer (original 3)
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    HYBRID = "hybrid"

    # Specific structured categories
    TRANSACTION_QUERY = "transaction_query"
    JOURNAL_ENTRY_QUERY = "journal_entry_query"
    VENDOR_LOOKUP = "vendor_lookup"
    BALANCE_QUERY = "balance_query"

    # Specific unstructured categories (each carries a doc_type filter)
    POLICY_LOOKUP = "policy_lookup"
    STANDARDS_LOOKUP = "standards_lookup"
    PROCEDURE_LOOKUP = "procedure_lookup"
    EVIDENCE_LOOKUP = "evidence_lookup"

    # Specific hybrid categories
    HYBRID_TX_EVIDENCE = "hybrid_tx_evidence"
    HYBRID_TX_COMPLIANCE = "hybrid_tx_compliance"


class RouterMode(str, Enum):
    """Which router *strategy* to use to make the QueryType decision."""
    REGEX = "regex"
    EMBEDDING = "embedding"
    LLM = "llm"
    HYBRID = "hybrid"        # cascade: regex → embedding → llm
    COMPARE = "compare"      # run all three, majority vote
