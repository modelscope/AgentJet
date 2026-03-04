import copy
import json
import threading
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

# Cache storage
_cache = {}
_cache_lock = threading.Lock()


def ajet_apply_chat_template(
    tokenizer,
    conversation,
    tools,
    add_generation_prompt: bool = False,
    tokenize: bool = True,
):
    conversation = cleanup_messages(conversation)

    # Create cache key by hashing all inputs
    cache_key = (
        id(tokenizer),
        hash(json.dumps(conversation, sort_keys=True)),
        hash(json.dumps(tools, sort_keys=True)) if tools else 0,
        add_generation_prompt,
        tokenize,
    )

    # Check cache with thread safety
    with _cache_lock:
        if cache_key in _cache:
            return _cache[cache_key]

    # Compute result (time consuming) - outside lock to avoid blocking other threads
    if tools:
        result = tokenizer.apply_chat_template(
            conversation,
            tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
        )
    else:
        result = tokenizer.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )

    # Store in cache with thread safety (implement LRU eviction if cache gets too large)
    with _cache_lock:
        if len(_cache) >= 1024:
            # Remove oldest item (first inserted)
            try:
                _cache.pop(next(iter(_cache)))
            except KeyError:
                # Cache was modified by another thread, which is fine
                pass

        _cache[cache_key] = result

    return result
