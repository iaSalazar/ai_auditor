"""Question decomposer — splits compound questions into atomic sub-questions.

When `decompose=true` is passed on /ask, the question is first sent to Haiku to
split into a JSON list of atomic sub-questions. Each sub-question then runs
through the existing chain (router → retrieve → answer-as-sub-context). A final
Sonnet synthesizer call stitches the sub-answers into a unified response with
grouped citations.

Trade-offs:
  - Cost: +1 Haiku call ($0.0005) and +1 Sonnet synthesizer call (~$0.01) per
    compound question. Atomic questions skip both (decomposer returns 1 element).
  - Latency: +600 ms (Haiku) + 5-7 s (Sonnet synthesizer) for compound.
  - Quality: surgical retrieval per sub-question vs. one diluted retrieval on
    the original compound question. Especially helps when the sub-questions
    would route to different specific categories (e.g. transaction_query +
    policy_lookup) under the richer routing dictionary.

Opt-in via the `decompose` flag (UI sidebar toggle or per-request) — kept opt-in
so the demo's golden path stays single-pass and predictable.
"""

from __future__ import annotations

import json
import logging
import re

from anthropic import Anthropic

from app.config import settings

logger = logging.getLogger(__name__)

_client = Anthropic(api_key=settings.anthropic_api_key)


DECOMPOSE_PROMPT = """You break compound audit questions into atomic sub-questions.

Rules:
- If the input is ALREADY a single atomic question, return a JSON list with ONE element (the original verbatim).
- If the input contains MULTIPLE distinct intents (e.g. joined by "and", "also", semicolons; OR clearly separable analytical steps), split into 2-4 atomic sub-questions.
- Each sub-question must be self-contained and answerable independently.
- Preserve specific entity references (transaction IDs like TX1012, vendor IDs like V1010, account numbers) in the relevant sub-question.
- Do NOT invent sub-questions. Do NOT change the meaning. Do NOT answer.

Return STRICT JSON only — a list of strings. No preamble, no code fences:

["sub-question 1", "sub-question 2"]

Examples:

Question: What is the total travel expense for Q1 2026 and what does IAASB say about audit risk assessment?
Output: ["What is the total travel expense for Q1 2026?", "What does IAASB say about audit risk assessment?"]

Question: Tell me about TX1018.
Output: ["Tell me about TX1018."]

Question: Show me TX1018 details and explain the revenue recognition policy that applies to it
Output: ["Show me TX1018 details.", "What does the revenue recognition policy say about transactions like TX1018 (multiple performance obligations)?"]

Question: {question}
Output:"""


SYNTHESIZE_PROMPT = """You are stitching together answers to atomic sub-questions into ONE unified response for the auditor.

Original compound question: {original}

Sub-questions and their grounded answers:
{sub_block}

Rules:
1. Write a single coherent answer that addresses the ORIGINAL compound question, drawing only from the sub-answers above.
2. Preserve ALL inline citations from the sub-answers ([TX1012], [policy.pdf p.2], etc.) exactly as written.
3. Group the response by sub-topic for readability. Use headings if it helps.
4. Do NOT add new facts, claims, or citations beyond what the sub-answers contain.
5. If a sub-answer is a refusal or "no data", carry that forward honestly — do not paper over it.
"""


def decompose_question(question: str) -> tuple[list[str], str]:
    """Returns (sub_questions, reason). On any failure, returns [question] unchanged."""
    try:
        msg = _client.messages.create(
            model=settings.judge_model,  # Haiku — cheap and fast
            max_tokens=400,
            messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(question=question)}],
        )
        text = msg.content[0].text.strip()
    except Exception as exc:
        logger.warning("Decomposer call failed (%s) — treating as atomic", exc)
        return [question], f"decomposer error: {exc}"

    # Strip code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, list) or not parsed:
            return [question], "decomposer returned empty/invalid list — treating as atomic"
        sub_qs = [str(q).strip() for q in parsed if str(q).strip()]
        if not sub_qs:
            return [question], "decomposer returned no non-empty sub-questions — treating as atomic"
        if len(sub_qs) == 1:
            return sub_qs, "single atomic sub-question — no decomposition"
        return sub_qs, f"decomposed into {len(sub_qs)} sub-questions"
    except json.JSONDecodeError as exc:
        logger.warning("Decomposer JSON parse failed (%s) — treating as atomic", exc)
        return [question], f"decomposer JSON parse error: {exc}"


def synthesize_answers(
    original_question: str,
    sub_results: list[dict],
) -> tuple[str, int, int]:
    """Stitch sub-answers into one. Returns (unified_answer, input_tokens, output_tokens)."""
    sub_block_lines: list[str] = []
    for i, sub in enumerate(sub_results, 1):
        sub_block_lines.append(f"--- SUB-QUESTION {i}: {sub['question']}")
        sub_block_lines.append(sub["answer"])
        sub_block_lines.append("")

    sub_block = "\n".join(sub_block_lines)

    response = _client.messages.create(
        model=settings.model,  # Sonnet for the final synthesis
        max_tokens=1536,
        system="You synthesize multiple sub-answers into one coherent audit response, preserving all citations.",
        messages=[
            {
                "role": "user",
                "content": SYNTHESIZE_PROMPT.format(original=original_question, sub_block=sub_block),
            }
        ],
    )

    return (
        response.content[0].text,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )
