"""Route registry — maps QueryType labels to retrieval scope.

Each route declares:
  - `is_structured`: should the SQL retriever run?
  - `is_unstructured`: should the vector retriever run?
  - `vector_filter`: optional Chroma `where` filter on chunk metadata. Narrows
    the candidate pool BEFORE similarity scoring — this is the fix for the
    "IAASB-dominates" problem (policy questions don't compete with the 134-page
    handbook because the handbook is structurally excluded from the pool).
  - `sql_table_hint`: optional list of SQL tables to nudge the SQL generator toward.
    Not enforced; just a prompt hint when the route is narrow.
  - `description`: free-text used by the LLM router prompt to teach the categories.

The original 3 routes (STRUCTURED, UNSTRUCTURED, HYBRID) remain as **fallback layer**:
when the LLM router can't confidently pick a specific category, it falls back to
one of these. Existing regex and embedding routers operate ONLY on the original 3
— the richer categories are exclusive to the LLM router (where semantic
classification work is appropriate).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval import QueryType


@dataclass(frozen=True)
class RouteScope:
    is_structured: bool
    is_unstructured: bool
    vector_filter: dict | None
    sql_table_hint: list[str] | None
    description: str


ROUTE_REGISTRY: dict[QueryType, RouteScope] = {
    # ---- Original 3 routes (fallback layer) ----
    QueryType.STRUCTURED: RouteScope(
        is_structured=True,
        is_unstructured=False,
        vector_filter=None,
        sql_table_hint=None,
        description="General SQL question that doesn't fit a more specific structured category — use when intent is clearly numeric/aggregate/lookup but the exact table is ambiguous.",
    ),
    QueryType.UNSTRUCTURED: RouteScope(
        is_structured=False,
        is_unstructured=True,
        vector_filter=None,
        sql_table_hint=None,
        description="General document question that doesn't fit a more specific document category — use when the question references narrative content but the doc type is ambiguous.",
    ),
    QueryType.HYBRID: RouteScope(
        is_structured=True,
        is_unstructured=True,
        vector_filter=None,
        sql_table_hint=None,
        description="Question genuinely needs BOTH structured records AND narrative context, and neither side fits a more specific hybrid category.",
    ),

    # ---- Specific structured categories ----
    QueryType.TRANSACTION_QUERY: RouteScope(
        is_structured=True,
        is_unstructured=False,
        vector_filter=None,
        sql_table_hint=["transactions", "support_mapping"],
        description="Questions about specific transactions, transaction lookups, missing support, totals/sums of transaction amounts.",
    ),
    QueryType.JOURNAL_ENTRY_QUERY: RouteScope(
        is_structured=True,
        is_unstructured=False,
        vector_filter=None,
        sql_table_hint=["journal_entries"],
        description="Questions about journal entries, debits/credits, audit issue flags on journal entries.",
    ),
    QueryType.VENDOR_LOOKUP: RouteScope(
        is_structured=True,
        is_unstructured=False,
        vector_filter=None,
        sql_table_hint=["vendors", "transactions"],
        description="Questions about vendors/suppliers: onboarding status, risk level, vendor metadata, transactions linked to a vendor.",
    ),
    QueryType.BALANCE_QUERY: RouteScope(
        is_structured=True,
        is_unstructured=False,
        vector_filter=None,
        sql_table_hint=["trial_balance"],
        description="Trial balance lookups, account-level balances, period balances.",
    ),

    # ---- Specific unstructured categories ----
    QueryType.POLICY_LOOKUP: RouteScope(
        is_structured=False,
        is_unstructured=True,
        vector_filter={"doc_type": "policy"},
        sql_table_hint=None,
        description="Questions about the company's internal accounting/recognition POLICIES (e.g. revenue recognition policy). NOT external auditing standards.",
    ),
    QueryType.STANDARDS_LOOKUP: RouteScope(
        is_structured=False,
        is_unstructured=True,
        vector_filter={"doc_type": "standard"},
        sql_table_hint=None,
        description="Questions about external auditing standards (IAASB, ISA), what the standard says or requires.",
    ),
    QueryType.PROCEDURE_LOOKUP: RouteScope(
        is_structured=False,
        is_unstructured=True,
        vector_filter={"doc_type": {"$in": ["procedure", "memo"]}},
        sql_table_hint=None,
        description="Questions about audit procedures we plan to perform, planning memos, risk-area procedures.",
    ),
    QueryType.EVIDENCE_LOOKUP: RouteScope(
        is_structured=False,
        is_unstructured=True,
        vector_filter={"doc_type": {"$in": ["workpaper", "evidence", "contract"]}},
        sql_table_hint=None,
        description="Questions about workpapers, evidence notes, contracts, client-provided documentation.",
    ),

    # ---- Specific hybrid categories ----
    QueryType.HYBRID_TX_EVIDENCE: RouteScope(
        is_structured=True,
        is_unstructured=True,
        vector_filter={"doc_type": {"$in": ["workpaper", "evidence", "contract"]}},
        sql_table_hint=["transactions", "support_mapping"],
        description="Questions that need BOTH transaction records AND the evidence/workpapers backing them.",
    ),
    QueryType.HYBRID_TX_COMPLIANCE: RouteScope(
        is_structured=True,
        is_unstructured=True,
        vector_filter={"doc_type": {"$in": ["policy", "standard", "procedure"]}},
        sql_table_hint=["transactions", "journal_entries"],
        description="Questions that need BOTH a specific transaction/JE AND the policy/standard/procedure that governs it (e.g. 'analyze TX1018 against revenue recognition policy').",
    ),
}


def get_scope(route: QueryType) -> RouteScope:
    """Look up the retrieval scope for a route. Falls back to HYBRID if missing."""
    return ROUTE_REGISTRY.get(route, ROUTE_REGISTRY[QueryType.HYBRID])
