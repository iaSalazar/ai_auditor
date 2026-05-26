"""Router orchestrator — picks which router strategy to use, optionally compares all three."""

from __future__ import annotations

import logging
from collections import Counter

from app.config import settings
from app.retrieval import QueryType, RouterMode
from app.retrieval.router_embedding import classify_embedding
from app.retrieval.router_llm import classify_llm
from app.retrieval.router_regex import classify_regex

logger = logging.getLogger(__name__)


# Map each specific category to its fallback "family" — used to normalize votes
# in compare mode so that regex's "structured" and llm's "transaction_query" count
# as the same vote (both want SQL).
_FAMILY: dict[QueryType, QueryType] = {
    # structured family
    QueryType.TRANSACTION_QUERY: QueryType.STRUCTURED,
    QueryType.JOURNAL_ENTRY_QUERY: QueryType.STRUCTURED,
    QueryType.VENDOR_LOOKUP: QueryType.STRUCTURED,
    QueryType.BALANCE_QUERY: QueryType.STRUCTURED,
    QueryType.STRUCTURED: QueryType.STRUCTURED,
    # unstructured family
    QueryType.POLICY_LOOKUP: QueryType.UNSTRUCTURED,
    QueryType.STANDARDS_LOOKUP: QueryType.UNSTRUCTURED,
    QueryType.PROCEDURE_LOOKUP: QueryType.UNSTRUCTURED,
    QueryType.EVIDENCE_LOOKUP: QueryType.UNSTRUCTURED,
    QueryType.UNSTRUCTURED: QueryType.UNSTRUCTURED,
    # hybrid family
    QueryType.HYBRID_TX_EVIDENCE: QueryType.HYBRID,
    QueryType.HYBRID_TX_COMPLIANCE: QueryType.HYBRID,
    QueryType.HYBRID: QueryType.HYBRID,
}


def _family_of(route: QueryType) -> QueryType:
    return _FAMILY.get(route, QueryType.HYBRID)


def _hybrid_cascade(question: str) -> tuple[QueryType, str]:
    """Cheapest path that works: regex → embedding → llm.

    Regex+embedding only emit the 3 fallback labels (structured/unstructured/hybrid).
    If the cascade exits at one of those, retrieval still works — just with the
    broader fallback scope. To get a SPECIFIC category (e.g. policy_lookup with its
    doc_type filter) the cascade has to reach the LLM stage. That's an intentional
    cost/specificity trade-off: questions that need richer routing are also the
    ones worth paying ~$0.0005 to classify properly.
    """
    route, reason = classify_regex(question)
    if "no specific signals" not in reason:
        return route, f"[regex] {reason}"

    route, reason = classify_embedding(question)
    if "low confidence" not in reason:
        return route, f"[regex miss → embedding] {reason}"

    route, reason = classify_llm(question)
    return route, f"[regex miss → embedding low conf → llm] {reason}"


def _compare_all(question: str) -> tuple[QueryType, str, dict]:
    """Run all three routers, return majority vote + full breakdown.

    Voting is done on the route FAMILY (structured/unstructured/hybrid) since
    regex and embedding only emit the 3 fallback labels while LLM can emit
    specific categories. Once the family majority is determined, if the LLM
    voted in the winning family, we use its SPECIFIC choice (better retrieval
    scope). Otherwise we use the family label.
    """
    regex_route, regex_reason = classify_regex(question)
    emb_route, emb_reason = classify_embedding(question)
    llm_route, llm_reason = classify_llm(question)

    family_votes = [_family_of(regex_route), _family_of(emb_route), _family_of(llm_route)]
    counts = Counter(family_votes)
    top_family, top_count = counts.most_common(1)[0]

    llm_family = _family_of(llm_route)

    if top_count >= 2:
        if llm_family == top_family:
            # LLM agrees with majority → use its specific category (richer retrieval)
            chosen = llm_route
            verdict = f"majority {top_count}/3 on family={top_family.value}, LLM specific → {chosen.value}"
        else:
            # Majority disagrees with LLM → use the family label as the safer choice
            chosen = top_family
            verdict = f"majority {top_count}/3 on family={top_family.value}, LLM dissented → {chosen.value}"
    else:
        # 3-way family disagreement → trust the LLM's specific choice
        chosen = llm_route
        verdict = f"3-way family disagreement, LLM tiebreaker → {chosen.value}"

    comparison = {
        "regex": {"route": regex_route.value, "reason": regex_reason},
        "embedding": {"route": emb_route.value, "reason": emb_reason},
        "llm": {"route": llm_route.value, "reason": llm_reason},
        "agreement": top_count == 3 and llm_family == top_family,
        "verdict": verdict,
    }

    reason = f"compare mode | regex={regex_route.value} | embedding={emb_route.value} | llm={llm_route.value} | {verdict}"
    return chosen, reason, comparison


def classify_query(
    question: str,
    mode: RouterMode | None = None,
) -> tuple[QueryType, str, dict | None]:
    """Returns (route, reason, comparison_dict_or_None).

    Comparison dict is non-None only when mode == COMPARE.
    """
    mode = mode or RouterMode(settings.router_mode)

    if mode == RouterMode.REGEX:
        route, reason = classify_regex(question)
        return route, reason, None
    if mode == RouterMode.EMBEDDING:
        route, reason = classify_embedding(question)
        return route, reason, None
    if mode == RouterMode.LLM:
        route, reason = classify_llm(question)
        return route, reason, None
    if mode == RouterMode.HYBRID:
        route, reason = _hybrid_cascade(question)
        return route, reason, None
    if mode == RouterMode.COMPARE:
        return _compare_all(question)

    # Defensive fallback
    logger.warning("Unknown router mode %r, falling back to regex", mode)
    route, reason = classify_regex(question)
    return route, reason, None
