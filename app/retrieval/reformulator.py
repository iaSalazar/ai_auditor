"""Question reformulation — rewrites follow-up questions as standalone.

Before routing, this resolves coreferences ('it', 'that one'), pronouns ('they',
'their'), and ellipsis ('and the policy implications?') by feeding conversation
history to a cheap Haiku call.

Both the router and the retrievers operate on the *rewritten* version so a
follow-up like "tell me more about it" doesn't route blind. The original question
is still shown to the answer LLM via the user prompt — the model sees both the
prior conversation and the user's actual phrasing.

Cost: ~$0.0005 per turn that has history. Latency: ~600ms. Skipped entirely on
the first turn (no history).
"""

from __future__ import annotations

import logging
from typing import Iterable

from anthropic import Anthropic

from app.config import settings

logger = logging.getLogger(__name__)

_client = Anthropic(api_key=settings.anthropic_api_key)

REFORMULATE_PROMPT = """Given a conversation history and the user's LATEST question,
rewrite the latest question so it can be understood WITHOUT the prior context.

Rules:
- Resolve pronouns ('it', 'they', 'their', 'that one', 'those', 'this') by replacing them with the specific entity from history.
- Resolve ellipsis (e.g. "and the policy implications?" → "What are the policy implications of <prior topic>?").
- If the latest question is ALREADY standalone (no pronouns, no implicit context), return it VERBATIM with no changes.
- Do NOT answer the question, add commentary, or change its intent.
- Return ONLY the rewritten question. No preamble. No quotes. No explanation.

CONVERSATION HISTORY:
{history}

LATEST QUESTION: {question}

REWRITTEN QUESTION:"""


def _format_history(history: Iterable[dict]) -> str:
    """Format the last few turns as a compact transcript for the reformulator."""
    lines = []
    for msg in list(history)[-4:]:  # last 4 messages = last 2 (user, assistant) pairs
        role = msg.get("role", "?").upper()
        content = str(msg.get("content", ""))[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def reformulate_question(question: str, history: list[dict]) -> tuple[str, str]:
    """Returns (rewritten_question, reason).

    If history is empty, returns the original question unchanged (no Haiku call).
    If the Haiku call fails for any reason, falls back to the original question
    so the chain never breaks on reformulation errors.
    """
    if not history:
        return question, "no history — original kept"

    history_text = _format_history(history)

    try:
        response = _client.messages.create(
            model=settings.judge_model,  # Haiku — cheap and fast
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": REFORMULATE_PROMPT.format(
                        history=history_text,
                        question=question,
                    ),
                }
            ],
        )
        rewritten = response.content[0].text.strip().strip('"').strip("'").strip()

        if not rewritten:
            return question, "empty reformulation — original kept"

        if rewritten.lower() == question.lower():
            return question, "already standalone"

        return rewritten, f"reformulated → '{rewritten[:80]}'"
    except Exception as exc:
        logger.warning("Reformulation failed (%s) — using original question", exc)
        return question, f"reformulation error: {exc}"
