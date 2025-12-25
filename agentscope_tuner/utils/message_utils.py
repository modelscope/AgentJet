import copy
from typing import Dict, List


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
