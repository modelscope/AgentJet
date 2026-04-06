# -*- coding: utf-8 -*-
"""Handle new user requests and track query history for diversity awareness."""

from typing import List, Dict
from loguru import logger
from ajet.schema.task import Task

# Rolling buffer of recent queries — used to detect repeated / near-duplicate
# questions so the system can log warnings.  The response-level diversity
# signal lives in on_compute_relative_reward._response_history.
_query_history: List[Dict] = []
QUERY_HISTORY_MAX = 100


def get_query_history() -> List[Dict]:
    """Return the current query history (read-only copy)."""
    return list(_query_history)


async def on_user_submit_new_requests(request_id: str, task: Task) -> None:
    """
    Store user request metadata when submitted.

    This populates a lightweight in-process history so that:
    1. The /requests endpoint can expose recent queries for debugging.
    2. We can detect if the same question keeps appearing, which signals
       a data distribution issue upstream rather than a model problem.
    """
    entry = {
        "request_id": request_id,
        "task_id": task.task_id,
        "query": task.main_query,
    }
    _query_history.append(entry)

    # Trim oldest entries
    while len(_query_history) > QUERY_HISTORY_MAX:
        _query_history.pop(0)

    logger.info(
        f"[on_user_submit] request_id={request_id} "
        f"query_len={len(task.main_query)} "
        f"history_size={len(_query_history)}"
    )
