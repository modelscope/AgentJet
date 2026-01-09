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
# OpenAI -> AgentScope conversion
# =============================================================================

def openai_to_agentscope_single(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single OpenAI format message to AgentScope format.
    
    Args:
        msg: Message dict in OpenAI format
        
    Returns:
        Message dict in AgentScope format
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")
    tool_calls = msg.get("tool_calls", [])
    tool_call_id = msg.get("tool_call_id", "")
    
    if tool_calls:
        # Assistant message contains tool_calls -> convert to ToolUseBlock format
        content_blocks = []
        # If there's text content, add TextBlock first
        if content:
            content_blocks.append({"type": "text", "text": content})
        # Convert each tool_call to ToolUseBlock
        for tc in tool_calls:
            func_info = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
            tool_use_block = {
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": func_info.get("name", ""),
                "input": func_info.get("arguments", "{}")
            }
            # Try to parse arguments as dict
            if isinstance(tool_use_block["input"], str):
                try:
                    tool_use_block["input"] = json.loads(tool_use_block["input"])
                except:
                    pass
            content_blocks.append(tool_use_block)
        return {
            "name": "assistant",
            "role": "assistant",
            "content": content_blocks
        }
    
    elif role == "tool" and tool_call_id:
        # Tool return result -> convert to ToolResultBlock format
        tool_result_block = {
            "type": "tool_result",
            "id": tool_call_id,
            "output": content
        }
        return {
            "name": "tool",
            "role": "user",  # tool_result in AgentScope is treated as user message
            "content": [tool_result_block]
        }
    
    else:
        # Normal message, keep original format
        return {
            "name": role,
            "role": role,
            "content": content
        }


def openai_to_agentscope(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI format message list to AgentScope format.
    
    Args:
        messages: Message list in OpenAI format
        
    Returns:
        Message list in AgentScope format
    """
    return [openai_to_agentscope_single(msg) for msg in messages]


def openai_to_agentscope_grouped(grouped_steps: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    """
    Convert grouped_steps (multi-turn conversation steps) from OpenAI format to AgentScope format.
    
    Args:
        grouped_steps: List of List of dict in OpenAI format
        
    Returns:
        Trajectory data in AgentScope format
    """
    return [[openai_to_agentscope_single(msg) for msg in step] for step in grouped_steps]


# =============================================================================
# AgentScope -> OpenAI conversion
# =============================================================================

def agentscope_to_openai_single(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single AgentScope format message to OpenAI format.
    
    Args:
        msg: Message dict in AgentScope format
        
    Returns:
        Message dict in OpenAI format
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")
    
    # If content is string, return directly
    if isinstance(content, str):
        return {
            "role": role,
            "content": content
        }
    
    # If content is list (AgentScope block format)
    if isinstance(content, list):
        text_parts = []
        tool_calls = []
        tool_call_id = ""
        tool_output = ""
        is_tool_result = False
        
        for item in content:
            if not isinstance(item, dict):
                continue
            
            item_type = item.get("type", "")
            
            if item_type == "text":
                # TextBlock
                text_parts.append(item.get("text", ""))
            
            elif item_type == "tool_use":
                # ToolUseBlock -> tool_calls
                arguments = item.get("input", {})
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_calls.append({
                    "id": item.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": arguments
                    }
                })
            
            elif item_type == "tool_result":
                # ToolResultBlock -> tool response
                is_tool_result = True
                tool_call_id = item.get("id", "")
                output = item.get("output", "")
                if isinstance(output, str):
                    tool_output += output
                else:
                    tool_output += str(output)
        
        # Build OpenAI format based on parsing result
        if is_tool_result and tool_call_id:
            return {
                "role": "tool",
                "content": tool_output,
                "tool_call_id": tool_call_id
            }
        elif tool_calls:
            result = {
                "role": "assistant",
                "content": "".join(text_parts) if text_parts else "",
                "tool_calls": tool_calls
            }
            return result
        else:
            return {
                "role": role,
                "content": "".join(text_parts) if text_parts else ""
            }
    
    # Otherwise, return as is
    return {
        "role": role,
        "content": str(content) if content else ""
    }


def agentscope_to_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert AgentScope format message list to OpenAI format.
    
    Args:
        messages: Message list in AgentScope format
        
    Returns:
        Message list in OpenAI format
    """
    return [agentscope_to_openai_single(msg) for msg in messages]


def agentscope_to_openai_grouped(grouped_steps: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    """
    Convert grouped_steps (multi-turn conversation steps) from AgentScope format to OpenAI format.
    
    Args:
        grouped_steps: List of List of dict in AgentScope format
        
    Returns:
        Trajectory data in OpenAI format
    """
    return [[agentscope_to_openai_single(msg) for msg in step] for step in grouped_steps]


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


def convert_grouped_steps_to_openai_format(grouped_steps: List[List[Any]]) -> List[List[Dict[str, Any]]]:
    """
    Convert grouped_steps (multi-turn conversation steps) to OpenAI format.
    
    Args:
        grouped_steps: List of List of ExtendedMessage or dict
        
    Returns:
        Trajectory data in OpenAI format (List of List of dict)
    """
    formatted_traj = []
    for context in grouped_steps:
        step_msgs = []
        for ext_msg in context:
            msg_dict = convert_ext_msg_to_openai_format(ext_msg)
            step_msgs.append(msg_dict)
        formatted_traj.append(step_msgs)
    return formatted_traj


def convert_flat_messages_to_openai_format(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert flat message list to OpenAI format.
    
    Args:
        messages: List of ExtendedMessage or dict
        
    Returns:
        Message list in OpenAI format (List of dict)
    """
    return [convert_ext_msg_to_openai_format(msg) for msg in messages]
