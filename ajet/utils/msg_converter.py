"""
Message format conversion utilities

Provides bidirectional conversion between OpenAI format and AgentScope format.
Unified for both train and val phases.

## OpenAI format examples:
- Assistant with tool_calls:
  {"role": "assistant", "content": "...", "tool_calls": [{"id": "call_xxx", "type": "function", "function": {"name": "...", "arguments": "..."}}]}
- Tool result:
  {"role": "tool", "content": "...", "tool_call_id": "call_xxx"}
- Normal message:
  {"role": "user/assistant/system", "content": "..."}

## AgentScope format examples:
- Assistant with tool_calls:
  {"role": "assistant", "content": [{"type": "text", "text": "..."}, {"type": "tool_use", "id": "call_xxx", "name": "...", "input": {...}}]}
- Tool result:
  {"role": "user", "content": [{"type": "tool_result", "id": "call_xxx", "output": "..."}]}
- Normal message:
  {"role": "user/assistant/system", "content": "..."}
"""

import json
from typing import List, Dict, Any, Union



# =============================================================================
# ExtendedMessage -> OpenAI conversion (backward compatible functions)
# =============================================================================

def convert_ext_msg_to_openai_format(ext_msg: Any) -> Dict[str, Any]:
    """
    Convert a single ExtendedMessage or dict to OpenAI format message.

    Args:
        ext_msg: ExtendedMessage object or dict

    Returns:
        Message dict in OpenAI format
    """
    # Helper function: get attribute value
    def get_attr(obj, attr_name, default=None):
        if hasattr(obj, attr_name):
            return getattr(obj, attr_name)
        elif isinstance(obj, dict):
            return obj.get(attr_name, default)
        return default

    # Check if there are tool_calls (assistant initiates tool call)
    tool_calls = get_attr(ext_msg, 'tool_calls')
    has_tool_calls = bool(tool_calls)

    # Check if there's tool_call_id (tool return result)
    tool_call_id = get_attr(ext_msg, 'tool_call_id')
    has_tool_call_id = bool(tool_call_id)

    # Get basic attributes
    role = get_attr(ext_msg, 'role', 'user')
    content = get_attr(ext_msg, 'content', '')

    if has_tool_calls:
        # Assistant message contains tool_calls -> keep OpenAI format
        msg_dict = {
            "role": "assistant",
            "content": content if content else "",
            "tool_calls": tool_calls
        }
    elif has_tool_call_id:
        # Tool return result -> use OpenAI format (role: "tool")
        msg_dict = {
            "role": "tool",
            "content": content if content else "",
            "tool_call_id": tool_call_id
        }
    else:
        # Normal message, keep original format
        msg_dict = {
            "role": role,
            "content": content if content else ""
        }

    return msg_dict


def convert_grouped_steps_to_openai_format(timelines: List[List[Any]]) -> List[List[Dict[str, Any]]]:
    """
    Convert timelines (multi-turn conversation steps) to OpenAI format.

    Args:
        timelines: List of List of ExtendedMessage or dict

    Returns:
        Trajectory data in OpenAI format (List of List of dict)
    """
    formatted_traj = []
    for context in timelines:
        step_msgs = []
        for ext_msg in context:
            msg_dict = convert_ext_msg_to_openai_format(ext_msg)
            step_msgs.append(msg_dict)
        formatted_traj.append(step_msgs)
    return formatted_traj
