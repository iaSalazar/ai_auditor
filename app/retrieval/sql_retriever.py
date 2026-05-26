"""Text-to-SQL retriever over the SQLite audit warehouse."""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

from anthropic import Anthropic

from app.config import settings
from app.ingestion.structured import get_schema_description, get_value_hints

logger = logging.getLogger(__name__)

_client = Anthropic(api_key=settings.anthropic_api_key)

SQL_PROMPT = """You are a SQL expert helping audit a company's financial records (Northstar Robotics).

SQLite schema:
{schema}

KEY VALUE REFERENCE — these are the EXACT values present in the data right now (auto-discovered):
{value_hints}
  vendors.risk    free-text — match with LOWER(risk) LIKE '%high%' etc.

Guidelines:
- Return ONLY a single SQL query. No markdown, no commentary.
- Use SELECT only. Never write DDL/DML.
- Use ONLY the values listed above when filtering. If the user asks about a value
  not in the reference (e.g. a quarter or account that isn't loaded), still write
  the SQL using their requested value — the empty result will be handled honestly downstream.
- Add LIMIT 50 unless the user explicitly asks for an aggregate (SUM/COUNT/AVG).
- Match string filters case-insensitively (LOWER(col) LIKE LOWER('%...%')) when the user's term might not be exact.
- Common joins:
    transactions.transaction_id  = support_mapping.transaction_id
    transactions.vendor_id       = vendors.vendor_id
    transactions.related_journal_entry_id = journal_entries.journal_entry_id

Question: {question}
"""


def _strip_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip().rstrip(";")


def _extract_tables(sql: str) -> list[str]:
    return sorted(set(t.lower() for t in re.findall(r"(?:FROM|JOIN)\s+(\w+)", sql, re.IGNORECASE)))


def _generate_sql(question: str) -> str:
    prompt = SQL_PROMPT.format(
        schema=get_schema_description(),
        value_hints=get_value_hints(),
        question=question,
    )
    msg = _client.messages.create(
        model=settings.model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    sql = _strip_fence(msg.content[0].text)
    logger.info("Generated SQL: %s", sql)
    return sql


def _execute(sql: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def retrieve_structured(question: str) -> dict:
    """Returns dict with: context, citations, sources (table names), sql, row_count, error."""
    try:
        sql = _generate_sql(question)
    except Exception as exc:
        logger.exception("SQL generation failed")
        return {
            "context": f"(SQL generation failed: {exc})",
            "citations": [],
            "sources": [],
            "sql": None,
            "row_count": 0,
            "error": str(exc),
        }

    if not sql.lower().lstrip().startswith("select"):
        logger.warning("Refusing non-SELECT statement: %s", sql)
        return {
            "context": "(Only SELECT statements are permitted.)",
            "citations": [],
            "sources": [],
            "sql": sql,
            "row_count": 0,
            "error": "non-SELECT refused",
        }

    tables = _extract_tables(sql)

    try:
        rows = _execute(sql)
    except sqlite3.Error as exc:
        logger.error("SQL execution error: %s", exc)
        return {
            "context": f"(SQL execution error: {exc}\nQuery: {sql})",
            "citations": [{"type": "sql", "query": sql, "error": str(exc)}],
            "sources": tables,
            "sql": sql,
            "row_count": 0,
            "error": str(exc),
        }

    if not rows:
        # Explicit zero-row signal — keep Sonnet from fabricating numbers when
        # the query ran fine but the data simply doesn't contain what was asked.
        context = (
            f"SQL executed:\n{sql}\n\n"
            f"Result: 0 rows. The query ran successfully but no records match the filter. "
            f"Treat this as authoritative — the data does NOT contain what was asked for. "
            f"Respond honestly that no matching records exist; do not estimate or extrapolate."
        )
    else:
        context = f"SQL executed:\n{sql}\n\nResult rows ({len(rows)}):\n"
        for row in rows[:25]:
            context += str(row) + "\n"

    citations: list[dict] = [{"type": "sql", "query": sql, "row_count": len(rows)}]
    seen: set[str] = set()
    for row in rows[:25]:
        for key in ("transaction_id", "journal_entry_id", "vendor_id", "document_id"):
            value = row.get(key)
            if value and value not in seen:
                seen.add(value)
                citations.append({"type": "record", "id": value, "field": key})

    return {
        "context": context,
        "citations": citations,
        "sources": tables,
        "sql": sql,
        "row_count": len(rows),
        "error": None,
    }
