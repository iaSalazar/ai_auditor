"""Load structured audit files (CSV / JSON / XLSX) into a local SQLite database."""

import json
import logging
import os
import sqlite3
from pathlib import Path

import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

DATA_DIR = Path(settings.data_dir) / "northstar_robotics_audit_dataset"


def _already_ingested(conn: sqlite3.Connection) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'"
    )
    return cursor.fetchone() is not None


def ingest_structured() -> None:
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    try:
        if _already_ingested(conn):
            logger.info("Structured data already loaded — skipping ingestion.")
            return

        logger.info("Ingesting structured audit data into SQLite…")
        _load_transactions(conn)
        _load_journal_entries(conn)
        _load_support_mapping(conn)
        _load_vendors(conn)
        _load_trial_balance(conn)
        conn.commit()
        logger.info("Structured ingestion complete.")
    finally:
        conn.close()


def _load_transactions(conn: sqlite3.Connection) -> None:
    df = pd.read_csv(DATA_DIR / "financial_transactions.csv")
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    logger.info("  transactions: %d rows", len(df))


def _load_journal_entries(conn: sqlite3.Connection) -> None:
    df = pd.read_csv(DATA_DIR / "journal_entries.csv")
    df.to_sql("journal_entries", conn, if_exists="replace", index=False)
    logger.info("  journal_entries: %d rows", len(df))


def _load_support_mapping(conn: sqlite3.Connection) -> None:
    df = pd.read_csv(DATA_DIR / "audit_support_mapping.csv")
    df.to_sql("support_mapping", conn, if_exists="replace", index=False)
    logger.info("  support_mapping: %d rows", len(df))


def _load_vendors(conn: sqlite3.Connection) -> None:
    with open(DATA_DIR / "vendor_master.json") as f:
        data = json.load(f)
    vendors = data["vendors"] if isinstance(data, dict) else data
    df = pd.DataFrame(vendors)
    df.to_sql("vendors", conn, if_exists="replace", index=False)
    logger.info("  vendors: %d rows", len(df))


def _load_trial_balance(conn: sqlite3.Connection) -> None:
    df = pd.read_excel(DATA_DIR / "trial_balance.xlsx")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df.to_sql("trial_balance", conn, if_exists="replace", index=False)
    logger.info("  trial_balance: %d rows", len(df))


def get_schema_description() -> str:
    """Return a human-readable schema description for the SQL retriever prompt."""
    conn = sqlite3.connect(settings.db_path)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        lines = []
        for table in tables:
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            col_str = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            lines.append(f"{table}({col_str})")
        return "\n".join(lines)
    finally:
        conn.close()


# Bounded-cardinality columns whose distinct values are safe to inline in the prompt.
# Format: (table, column, max_cardinality, optional_label_column).
# Skip free-text or high-cardinality columns (transaction_id, names, risk free-text).
_HINT_COLUMNS: list[tuple[str, str, int, str | None]] = [
    ("transactions", "quarter", 20, None),
    ("transactions", "account_number", 50, "account_name"),
    ("transactions", "transaction_type", 20, None),
    ("transactions", "support_status", 20, None),
    ("vendors", "status", 20, None),
    ("vendors", "category", 30, None),
]


def get_value_hints() -> str:
    """Auto-discover the distinct values present in bounded-cardinality columns.

    Drives the 'KEY VALUE REFERENCE' block of the SQL prompt — replaces hand-written
    hints so the model always sees the exact literals in the current data (e.g.
    'Q1-2026' with a hyphen, only the quarters actually loaded).
    """
    conn = sqlite3.connect(settings.db_path)
    try:
        existing_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        lines: list[str] = []
        for table, column, max_card, label_col in _HINT_COLUMNS:
            if table not in existing_tables:
                continue
            try:
                if label_col:
                    rows = conn.execute(
                        f"SELECT DISTINCT {column}, {label_col} FROM {table} "
                        f"ORDER BY {column} LIMIT {max_card + 1}"
                    ).fetchall()
                    if len(rows) > max_card:
                        continue  # too many — would blow prompt budget
                    lines.append(f"  {table}.{column} values:")
                    for value, label in rows:
                        lines.append(f"    {value} = {label}")
                else:
                    rows = conn.execute(
                        f"SELECT DISTINCT {column} FROM {table} "
                        f"ORDER BY {column} LIMIT {max_card + 1}"
                    ).fetchall()
                    if len(rows) > max_card:
                        continue
                    values = [r[0] for r in rows]
                    lines.append(f"  {table}.{column} values: {values}")
            except sqlite3.Error as exc:
                logger.warning("Could not auto-discover %s.%s: %s", table, column, exc)
        return "\n".join(lines) if lines else "  (no bounded-cardinality columns discovered)"
    finally:
        conn.close()
