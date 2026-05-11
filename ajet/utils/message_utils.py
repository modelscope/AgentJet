import copy
from typing import Dict, List

from loguru import logger

_TOKEN_OVERFLOW_SIGNATURE = "Exceeded max model context length. token_overflow"

def is_token_overflow_message(content) -> bool:
    """Return True if `content` represents the AgentJet token-overflow output
    (prompt would exceed max_model_len). Accepts a raw string, a message dict
    with a "content" field, bytes, or None. Match is substring-based so the
    signal survives whitespace, the "AgentJet:" prefix being stripped, or the
    content being embedded in a larger blob.
    """
    if content is None:
        return False
    if isinstance(content, dict):
        content = content.get("content")
        if not isinstance(content, str):
            return False
    elif isinstance(content, (bytes, bytearray)):
        try:
            content = content.decode("utf-8", errors="ignore")
        except Exception:
            return False
    elif not isinstance(content, str):
        return False
    return _TOKEN_OVERFLOW_SIGNATURE in content


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
