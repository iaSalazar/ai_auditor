"""LLM-based router using Haiku (cheap and fast).

Zero-shot classification via a few-shot prompt. No training data needed.
Costs ~$0.0005 per request. ~600ms latency.
"""

from __future__ import annotations

import json
import logging
import re

from anthropic import Anthropic

from app.config import settings
from app.retrieval import QueryType
from app.retrieval.routes import ROUTE_REGISTRY

logger = logging.getLogger(__name__)

_client = Anthropic(api_key=settings.anthropic_api_key)


def _build_route_catalog() -> str:
    """Render the route registry as a bulleted catalog for the prompt."""
    lines = []
    for route, scope in ROUTE_REGISTRY.items():
        lines.append(f"- {route.value}: {scope.description}")
    return "\n".join(lines)


ROUTER_PROMPT = """You classify audit questions into the MOST SPECIFIC retrieval route that applies.

Route catalog (specific routes preferred; the original 3 are FALLBACKS for ambiguous cases):
{catalog}

Selection rules:
1. Prefer the most SPECIFIC route. Only fall back to structured/unstructured/hybrid when no specific category clearly fits.
2. Match by INTENT, not just keywords. "Suppliers" → vendor_lookup. "What does ISA say" → standards_lookup. "What does our policy say" → policy_lookup.
3. If the question needs both records AND interpretation against a policy/standard, prefer hybrid_tx_compliance.
4. If the question needs both records AND the evidence/workpapers, prefer hybrid_tx_evidence.

Examples:
Q: Which transactions are missing support documentation?
A: {{"route":"transaction_query","reason":"transactions filtered by support_status"}}

Q: Summarize the revenue recognition policy
A: {{"route":"policy_lookup","reason":"company internal policy lookup"}}

Q: What does IAASB say about audit risk assessment?
A: {{"route":"standards_lookup","reason":"external auditing standard reference"}}

Q: Which vendors are flagged as inactive pending review?
A: {{"route":"vendor_lookup","reason":"vendor metadata/status lookup"}}

Q: Analyze TX1018 in the context of the revenue recognition policy
A: {{"route":"hybrid_tx_compliance","reason":"transaction record + policy interpretation"}}

Q: What workpaper evidence supports TX1058?
A: {{"route":"hybrid_tx_evidence","reason":"transaction record + supporting evidence document"}}

Q: Tell me about the lease terms
A: {{"route":"evidence_lookup","reason":"contract document lookup"}}

Q: What audit procedures should we run on travel expenses?
A: {{"route":"procedure_lookup","reason":"audit procedure / planning memo content"}}

Q: {question}
A:"""


def _format_prompt(question: str) -> str:
    return ROUTER_PROMPT.format(
        catalog=_build_route_catalog(),
        question=question,
    )


def classify_llm(question: str) -> tuple[QueryType, str]:
    """Returns (route, reason_string)."""
    try:
        msg = _client.messages.create(
            model=settings.judge_model,  # Haiku — cheap + fast
            max_tokens=200,
            messages=[{"role": "user", "content": _format_prompt(question)}],
        )
        text = msg.content[0].text
    except Exception as exc:
        logger.error("LLM router call failed: %s", exc)
        return QueryType.UNSTRUCTURED, f"LLM router error: {exc}"

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning("LLM router: no JSON in response: %s", text[:200])
        return QueryType.UNSTRUCTURED, f"LLM router parse error: {text[:100]}"

    try:
        parsed = json.loads(match.group())
        route_str = parsed.get("route", "").lower()
        reason = parsed.get("reason", "")
        route = QueryType(route_str)
        return route, f"LLM(Haiku): {reason}"
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("LLM router parse failed: %s", exc)
        return QueryType.UNSTRUCTURED, f"LLM router parse error: {exc}"
