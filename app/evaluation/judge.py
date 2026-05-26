"""LLM-as-judge metrics: faithfulness, answer relevance, refusal correctness.

Uses a small/fast model (configured as `judge_model`) to keep eval cheap.
"""

from __future__ import annotations

import json
import logging
import re

from anthropic import Anthropic

from app.config import settings

logger = logging.getLogger(__name__)

_client = Anthropic(api_key=settings.anthropic_api_key)


FAITHFULNESS_PROMPT = """You evaluate AI audit assistants.

Decompose the ANSWER into distinct factual claims. For each claim, judge whether it is fully
supported by the CONTEXT (verbatim or paraphrased). Numbers, IDs, and names must match.

CONTEXT:
{context}

ANSWER:
{answer}

Output STRICT JSON only, nothing else:
{{
  "claims": [{{"claim": "<text>", "supported": true|false}}],
  "score": <fraction of supported claims, 0.0-1.0>
}}
"""

RELEVANCE_PROMPT = """Rate how directly the ANSWER addresses the QUESTION on a 0.0-1.0 scale:
  1.0  directly and completely answers
  0.5  partially addresses
  0.0  does not address the question

QUESTION: {question}

ANSWER: {answer}

Output STRICT JSON only:
{{"score": <0.0-1.0>, "reasoning": "<one short sentence>"}}
"""

REFUSAL_PROMPT = """An audit assistant should refuse to fabricate answers. Refusal is correct when EITHER:
  (a) the QUESTION is outside the audit/accounting/finance domain, OR
  (b) the QUESTION is in-domain but the underlying data does not contain what was asked
      (e.g. asking for a quarter/account/entity that isn't loaded). In this case a correct
      response says "no matching records / no data for that period" and does NOT invent
      a specific number, dollar amount, or count.

A WRONG response either answers the off-topic question, OR fabricates a specific value
when the data didn't contain it.

QUESTION: {question}

ANSWER: {answer}

Output STRICT JSON only:
{{"refused": true|false, "reasoning": "<one short sentence>"}}
"""


def _extract_json(text: str) -> dict:
    """Robust JSON extraction — tolerates surrounding text or code fences."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"error": "no JSON object found", "raw": text[:300]}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        return {"error": f"JSON parse failed: {exc}", "raw": text[:300]}


def _judge(prompt: str, max_tokens: int = 800) -> dict:
    try:
        msg = _client.messages.create(
            model=settings.judge_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_json(msg.content[0].text)
    except Exception as exc:
        logger.exception("Judge call failed")
        return {"error": str(exc)}


def judge_faithfulness(answer: str, context: str) -> dict:
    return _judge(FAITHFULNESS_PROMPT.format(context=context[:10000], answer=answer), max_tokens=2000)


def judge_relevance(question: str, answer: str) -> dict:
    return _judge(RELEVANCE_PROMPT.format(question=question, answer=answer), max_tokens=300)


def judge_refusal(question: str, answer: str) -> dict:
    return _judge(REFUSAL_PROMPT.format(question=question, answer=answer), max_tokens=200)
