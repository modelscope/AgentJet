import copy
from typing import Dict, List

from loguru import logger


def log_empty_content_messages(messages: List[Dict], episode_uuid: str = "") -> None:
    """Scan an OpenAI-compatible message list and log an error for any message
    whose content is empty/None and which carries no tool_calls.
    """
    for idx, m in enumerate(messages or []):
        content = m.get("content")
        tool_calls = m.get("tool_calls") or []
        if content in (None, "") and not tool_calls:
            logger.error(
                f"[{episode_uuid}] Empty content in inbound message "
                f"index={idx} role={m.get('role')} tool_call_id={m.get('tool_call_id')!r} "
                f"content={content!r} tool_calls={tool_calls}"
            )


# apply chat_template to a message, and then convert back to message
def convert_tool_to_user_message(tool_message, tokenizer, format="qwen"):
    assert format == "qwen"

    if tool_message["role"] == "user":
        return tool_message
    elif tool_message["role"] == "tool" and len(tool_message["tool_calls"]) > 0:
        assert len(tool_message["tool_calls"]) == 1
        return {
            "role": "user",
            "content": str(tool_message["tool_calls"][0]["result"]),
        }


def remove_fields(d: Dict, fields: List[str]) -> Dict:
    d = copy.deepcopy(d)
    for field in fields:
        d.pop(field.strip(), None)
    return d
