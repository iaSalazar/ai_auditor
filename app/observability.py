import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("audit.requests")


def log_request(question: str, result: dict) -> None:
    sql = result.get("sql") or {}
    chunks = result.get("chunks") or []
    avg_similarity = (
        round(sum(c.get("similarity", 0) for c in chunks) / len(chunks), 3) if chunks else None
    )
    comparison = result.get("router_comparison") or {}
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_id": result.get("conversation_id"),
        "turn_index": result.get("turn_index"),
        "question": question,
        "query_type": result.get("query_type"),
        "routing_reason": result.get("routing_reason"),
        "router_mode": result.get("router_mode"),
        "router_agreement": comparison.get("agreement"),
        "router_picks": (
            {k: v["route"] for k, v in comparison.items() if isinstance(v, dict) and "route" in v}
            if comparison else None
        ),
        "retrieved_sources": result.get("retrieved_sources", []),
        "sql_tables": sql.get("tables"),
        "sql_row_count": sql.get("row_count"),
        "chunk_count": len(chunks),
        "avg_similarity": avg_similarity,
        "citation_count": len(result.get("citations", [])),
        "latency_ms": result.get("latency_ms"),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
    }
    logger.info(json.dumps(record))
