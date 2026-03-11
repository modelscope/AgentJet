# -*- coding: utf-8 -*-
"""Handle new user requests."""

from ajet.schema.task import Task

async def on_user_submit_new_requests(request_id: str, task: Task) -> None:
    """Store user request when submitted."""
    pass  # No special processing needed for this use case
