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


# Cache for (sep, fold) per tokenizer id (one entry per tokenizer, never
# evicted — tiny). The values depend only on the tokenizer, not on tools
# (verified across Qwen2.5 / Qwen3 / Qwen3.6 text + VL).
_tool_sep_cache: "dict[int, tuple[str, bool]]" = {}


def derive_tool_sep_and_fold(tokenizer) -> "tuple[str, bool]":
    """Derive, in ONE chat-template render, both:

    1. ``sep`` — the text the template inserts BETWEEN two consecutive ``tool``
       messages. When the template FOLDS a run of tool turns into one
       ``<|im_start|>user`` segment (Qwen3 text), this is the in-segment
       separator (joining merged tool contents with it makes ONE merged tool
       message render identically to the folded multi-tool block). When the
       template renders each tool as its OWN segment (Qwen2.5-VL), this is the
       end-of-segment + next-segment-open text.

    2. ``fold`` — whether the template folds consecutive tool messages into one
       ``<|im_start|>user`` segment (True) or renders each as its own segment
       (False). Callers merge consecutive tool messages only when fold=True.

    Both are detected by rendering ``[user, assistant+tool_call, tool, tool]``
    with unique placeholder content and inspecting the text between the
    placeholders: if no ``<|im_start|>`` lies between them they share a segment
    (fold=True) and the text between is the in-segment separator; otherwise
    fold=False and the text between is the inter-segment separator. Cached
    per-tokenizer (id), thread-safe. Adapts to any chat template.
    """
    key = id(tokenizer)
    with _cache_lock:
        cached = _tool_sep_cache.get(key)
    if cached is not None:
        return cached

    PH_A = "\x00TOOLSEP_A\x00"
    PH_B = "\x00TOOLSEP_B\x00"
    convo = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "content": PH_A, "tool_call_id": "c1"},
        {"role": "tool", "content": PH_B, "tool_call_id": "c1"},
    ]
    try:
        text = ajet_apply_chat_template(
            tokenizer=tokenizer, conversation=convo, tools=None,
            add_generation_prompt=False, tokenize=False,
        )
        a = text.find(PH_A)
        b = text.find(PH_B)
        if a == -1 or b == -1 or b < a:
            sep, fold = "\n", False
        else:
            sep = text[a + len(PH_A): b]
            fold = "<|im_start|>" not in text[a + len(PH_A): b]
    except Exception:
        sep, fold = "\n", False

    with _cache_lock:
        _tool_sep_cache[key] = (sep, fold)
    return sep, fold


def flush_pending_tool_run(
    timeline: list,
    pending_tool_run: list,
    tokenizer,
    processor,
    tools: list,
    sep: str,
    fold: bool,
) -> list:
    """Append the buffered run of consecutive ``role == "tool"`` messages to
    ``timeline`` as ExtendedMessage(s), keeping the timeline 1:1 aligned with
    the chat template's ``<|im_start|>`` segments:

    - If ``fold`` is True (Qwen3 text: consecutive tool turns fold into ONE
      ``<|im_start|>user`` segment), the whole run is MERGED into a single
      ExtendedMessage whose content is the per-tool contents joined by ``sep``
      (so the merged message renders identically to the folded segment).
    - If ``fold`` is False (Qwen2.5-VL: each tool is its own segment), each
      tool message becomes its own ExtendedMessage (else the timeline would be
      shorter than the segment count).

    ``pending_tool_run`` is a list of dicts: {content, tool_call_id,
    first_index}. Returns the (now emptied) pending list for caller
    reassignment; also mutates ``timeline`` in place.
    """
    from ajet.schema.extended_msg import ExtendedMessage  # local import avoids cycle

    if not pending_tool_run:
        return pending_tool_run

    if not fold or len(pending_tool_run) == 1:
        # Emit one ExtendedMessage per tool message.
        for t in pending_tool_run:
            timeline.append(ExtendedMessage(
                author="env",
                role="tool",
                content=t["content"],
                tokenizer=tokenizer,
                tools=tools,
                tool_calls=[],
                tool_call_id=t["tool_call_id"] or "",
                token_generator="manual",
                name="",
                first_message=(t["first_index"] == 0),
                images=None,
                processor=processor,
            ))
    else:
        contents = [t["content"] for t in pending_tool_run]
        tcids = [t["tool_call_id"] for t in pending_tool_run]
        first_idx = pending_tool_run[0]["first_index"]
        merged_content = sep.join(contents)
        timeline.append(ExtendedMessage(
            author="env",
            role="tool",
            content=merged_content,
            tokenizer=tokenizer,
            tools=tools,
            tool_calls=[],
            tool_call_id=tcids[0] if tcids and tcids[0] else "",
            token_generator="manual",
            name="",
            first_message=(first_idx == 0),
            images=None,
            processor=processor,
        ))
    pending_tool_run.clear()
    return pending_tool_run
