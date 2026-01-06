import copy
import json
from typing import Dict, List


def cleanup_messages(messages: List[Dict]) -> List[Dict]:
    "A temperary fix for tool_calls being str instead of dict"
    messages_copied = copy.deepcopy(messages)
    for m in messages_copied:
        if "tool_calls" not in m:
            continue
        for t in m["tool_calls"]:
            if "function" not in t or "arguments" not in t["function"]:
                continue
            if isinstance(t["function"]["arguments"], str):
                try:
                    t["function"]["arguments"] = json.loads(t["function"]["arguments"])
                except Exception:
                    pass
    return messages_copied


def ajet_apply_chat_template(
    tokenizer,
    conversation,
    tools,
    add_generation_prompt: bool = False,
    tokenize: bool = True,
):
    conversation = cleanup_messages(conversation)
    if tools:
        return tokenizer.apply_chat_template(
            conversation,
            tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
        )
    else:
        return tokenizer.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
