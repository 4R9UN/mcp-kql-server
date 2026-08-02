"""Bounded formatting for Kusto DataFrame results returned through MCP."""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .constants import LIMITS, SPECIAL_TOKENS
from .utils import ErrorHandler

logger = logging.getLogger(__name__)


def limit_result_rows(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, int, bool]:
    """Return a bounded DataFrame while preserving the original row count."""
    total_row_count = len(dataframe)
    max_result_rows = max(
        1,
        int(os.environ.get("MCP_KQL_MAX_RESULT_ROWS", LIMITS["max_result_rows"])),
    )
    truncated = total_row_count > max_result_rows
    if truncated:
        dataframe = dataframe.head(max_result_rows).copy()
    return dataframe, total_row_count, truncated


def dataframe_to_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame rows to JSON-safe records."""
    try:
        records = []
        for _, row in dataframe.iterrows():
            record = {}
            for column, value in row.items():
                if pd.isna(value):
                    record[column] = None
                elif hasattr(value, "isoformat"):
                    record[column] = value.isoformat()
                elif hasattr(value, "strftime"):
                    record[column] = value.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(value, type):
                    record[column] = value.__name__
                elif hasattr(value, "item"):
                    record[column] = value.item()
                else:
                    record[column] = value
            records.append(record)
        return records
    except (ValueError, TypeError, AttributeError) as error:
        logger.warning("DataFrame conversion failed: %s", error)
        return dataframe.astype(str).to_dict("records")


def build_json_result(
    dataframe: pd.DataFrame,
    total_row_count: int,
    truncated: bool,
    requested_auth_method: str,
    active_auth_method: Optional[str],
    repair_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the stable JSON result contract for a bounded DataFrame."""
    columns = dataframe.columns.tolist()
    result = {
        "success": True,
        "row_count": len(dataframe),
        "total_row_count": total_row_count,
        "truncated": truncated,
        "columns": columns,
        "column_tokens": [f"{SPECIAL_TOKENS['COLUMN']}:{column}" for column in columns[:15]],
        "data": dataframe_to_records(dataframe),
        "result_context": (
            f"{SPECIAL_TOKENS['RESULT_START']}Returned {len(dataframe)} rows "
            f"with {len(columns)} columns{SPECIAL_TOKENS['RESULT_END']}"
        ),
        "requested_auth_method": requested_auth_method,
        "active_auth_method": active_auth_method,
    }
    if truncated:
        result["truncation_reason"] = "row_limit"
    if repair_metadata:
        result.update({
            "repair_applied": True,
            "replacements": repair_metadata["replacements"],
            "schema_grounded_table": repair_metadata["table_name"],
            "preferred_time_column": repair_metadata["preferred_time_column"],
        })
    return result


def serialize_bounded_json_result(result: Dict[str, Any]) -> str:
    """Serialize a result while retaining the largest row prefix within budget."""
    max_bytes = max(
        1,
        int(os.environ.get("MCP_KQL_MAX_RESULT_BYTES", LIMITS["max_result_bytes"])),
    )
    rendered = ErrorHandler.safe_json_dumps(result, indent=2)
    if len(rendered.encode("utf-8")) <= max_bytes:
        return rendered

    rows = result.get("data")
    if not isinstance(rows, list):
        return ErrorHandler.safe_json_dumps({
            "success": False,
            "error": "Response exceeded the configured MCP byte limit",
            "max_result_bytes": max_bytes,
        }, indent=2)

    best_count = 0
    low, high = 0, len(rows)
    while low <= high:
        candidate_count = (low + high) // 2
        candidate = {
            **result,
            "data": rows[:candidate_count],
            "row_count": candidate_count,
            "truncated": True,
            "truncation_reason": "response_size",
        }
        candidate_json = ErrorHandler.safe_json_dumps(candidate, indent=2)
        if len(candidate_json.encode("utf-8")) <= max_bytes:
            best_count = candidate_count
            low = candidate_count + 1
        else:
            high = candidate_count - 1

    result.update({
        "data": rows[:best_count],
        "row_count": best_count,
        "truncated": True,
        "truncation_reason": "response_size",
    })
    return ErrorHandler.safe_json_dumps(result, indent=2)