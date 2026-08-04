# flake8: noqa
"""
Regression tests for the consecutive-tool-merge fix, using a REAL production
crash scene.

Background
----------
patch_prompt_tokens (the "retokenization drift" safety net) assumes a **1:1
correspondence** between the timeline messages (previous_ext_context) and the
<|im_start|> segments of vLLM's prompt. But the chat template folds a run of
consecutive `tool` messages into ONE `<|im_start|>user` segment, so if
step_spawn_timeline kept each tool message separate, the timeline was longer
than the segment count and ensure_retokenization_perfect_match did
``prompt_text_split[j]`` with ``j`` out of range -> IndexError, crashing the
rollout worker.

The crash scene was captured on a real benchmark_math run: a ReAct agent hit
max iterations and _summarizing fired, producing a 24-message conversation
(1 system + 2 user + 2 assistant + 19 tool) that the Qwen3 template renders
as only 7 <|im_start|> segments. The 24-message list is checked in at
tests/data/patch_prompt_scene_24msgs.json.

The FIX (in step_spawn_timeline): merge consecutive `role == "tool"` messages
into ONE ExtendedMessage, joining their contents with the template's own
inter-tool separator (derived dynamically via _derive_tool_sep) so the merged
single tool message renders identically to the folded segment. After the fix,
len(timeline) == segment count, and patch_prompt_tokens no longer crashes.

Tests:
  - test_crash_scene_is_24_messages_7_segments: the scene really is 24 msgs / 7 segs.
  - test_slicer_handles_crash_scene_no_crash: the slicer never crashed on this
    scene; concat(token_arr) == full render.
  - test_step_spawn_timeline_matches_tokenize_segments: AFTER the fix,
    len(timeline) == segment count (7 == 7) and roles line up per-segment.
  - test_patch_prompt_tokens_no_longer_crashes: AFTER the fix, patch_prompt_tokens
    completes without IndexError and history concat stays a prefix of vLLM's prompt.
"""

import json
import os

import pytest

from ajet.schema.extended_msg import ExtendedMessage
from ajet.context_tracker.multiagent_tracking import MultiAgentContextTracker


_QWEN_TEXT = "Qwen3-8B"
_MODEL_CACHE = "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen"
_SCENE_JSON = os.path.join(
    os.path.dirname(__file__), "data", "patch_prompt_scene_24msgs.json"
)


def _tokenizer_path():
    p = os.path.join(_MODEL_CACHE, _QWEN_TEXT)
    return p if os.path.isdir(p) else None


pytestmark = pytest.mark.skipif(
    _tokenizer_path() is None or not os.path.exists(_SCENE_JSON),
    reason="Qwen3-8B tokenizer or the crash-scene json is not available.",
)


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_tokenizer_path(), trust_remote_code=True)


def _build_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": "Execute python code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "timeout": {"default": 300, "type": "number"},
                    },
                    "required": ["code"],
                },
            },
        }
    ]


def _make_tracker(tokenizer):
    """Tracker stub with only the attributes patch_prompt_tokens /
    tokenize_and_slice_timeline / step_spawn_timeline / to_role_content need."""
    from ajet.utils.tokenizer import derive_tool_sep_and_fold
    tr = MultiAgentContextTracker.__new__(MultiAgentContextTracker)
    tr.tokenizer = tokenizer
    tr.processor = None
    tr._im_start_token_id = tokenizer.encode("<|im_start|>")[0]
    # step_spawn_timeline reads these (normally set in __init__).
    tr.tool_res_sep, tr.tool_res_fold = derive_tool_sep_and_fold(tokenizer)

    # generation_prompt: the delta between rendering [user, assistant] with and
    # without add_generation_prompt (mirrors get_generation_prompt_token).
    from ajet.utils.tokenizer import ajet_apply_chat_template
    dummy = [{"role": "user", "content": "dummy"},
             {"role": "assistant", "content": "dummy text"}]
    frag_from = ajet_apply_chat_template(
        tokenizer=tokenizer, conversation=dummy, tokenize=False, tools=[],
        add_generation_prompt=False)
    frag_to = ajet_apply_chat_template(
        tokenizer=tokenizer, conversation=dummy, tokenize=False, tools=[],
        add_generation_prompt=True)
    ids_from = tokenizer(frag_from, add_special_tokens=False)["input_ids"]
    ids_to = tokenizer(frag_to, add_special_tokens=False)["input_ids"]
    tr.generation_prompt_token = ids_to[len(ids_from):]
    tr.generation_prompt = tokenizer.decode(tr.generation_prompt_token)

    # config stub: patch_prompt_tokens reads fix_retokenization_drift.
    class _Ctx:
        class ajet:
            class context_tracker:
                fix_retokenization_drift = True
    tr.config = _Ctx
    tr.episode_uuid = "test-episode"
    tr.task_id = "test-task"
    return tr


def _build_timeline(tracker, messages, tools):
    """Build a timeline the way step_spawn_timeline does (manual, empty token_arr)."""
    timeline = []
    for i, m in enumerate(messages):
        timeline.append(ExtendedMessage(
            author="initialization" if m["role"] == "system" else "env",
            role=m["role"],
            content=m.get("content", "") or "",
            tokenizer=tracker.tokenizer,
            tools=tools,
            tool_calls=m.get("tool_calls", []),
            tool_call_id=m.get("tool_call_id", ""),
            token_generator="manual",
            first_message=(i == 0),
        ))
    return timeline


def _full_render_ids(tokenizer, messages, tools, add_generation_prompt):
    from ajet.utils.tokenizer import ajet_apply_chat_template
    text = ajet_apply_chat_template(
        tokenizer=tokenizer, conversation=messages, tools=tools,
        add_generation_prompt=add_generation_prompt, tokenize=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _load_scene():
    with open(_SCENE_JSON, encoding="utf-8") as f:
        return json.load(f)


def test_crash_scene_is_24_messages_7_segments():
    """Sanity: the checked-in scene really is the 24-message / 7-segment case.
    If this ever changes, the scene file was regenerated and the assertions
    below must be revisited."""
    messages = _load_scene()
    tokenizer = _load_tokenizer()
    tools = _build_tools()
    text = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=False, tokenize=False)
    segs = [s for s in text.split("<|im_start|>") if s.strip()]
    assert len(messages) == 24, f"scene has {len(messages)} messages, expected 24"
    assert len(segs) == 7, f"scene renders to {len(segs)} segments, expected 7"
    from collections import Counter
    roles = Counter(m["role"] for m in messages)
    assert roles["tool"] == 19, f"expected 19 tool messages, got {roles['tool']}"


def test_slicer_handles_crash_scene_no_crash():
    """The slicer (tokenize_and_slice_timeline) must NOT crash on the 24->7
    scene and must keep concat(token_arr) == the full render."""
    messages = _load_scene()
    tokenizer = _load_tokenizer()
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    timeline = _build_timeline(tr, messages, tools)
    tr.tokenize_and_slice_timeline(timeline, tools)  # must not raise

    full_ids = _full_render_ids(tokenizer, messages, tools, add_generation_prompt=False)
    concat = [t for m in timeline for t in m.token_arr]
    assert concat == full_ids, (
        f"slicer concat != full render on crash scene "
        f"(concat={len(concat)} full={len(full_ids)})"
    )


def test_patch_prompt_tokens_no_longer_crashes():
    """After step_spawn_timeline merges consecutive tool messages, the timeline
    length (7) matches the prompt segment count (7), so patch_prompt_tokens
    no longer raises IndexError. It must complete and the timeline's
    concat(token_arr) must still equal vLLM's prompt (the core invariant)."""
    messages = _load_scene()
    tokenizer = _load_tokenizer()
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    timeline = tr.step_spawn_timeline(messages, tools, disable_toolcalls=False)
    tr.tokenize_and_slice_timeline(timeline, tools)

    prompt_ids = _full_render_ids(
        tokenizer, messages, tools, add_generation_prompt=True)
    prompt_text = tokenizer.decode(prompt_ids)

    # Must not raise.
    tr.patch_prompt_tokens(
        prompt_text=prompt_text,
        prompt_token_ids=prompt_ids,
        previous_ext_context=timeline,
    )
    # After patching, the history concat must still be a prefix of the vLLM
    # prompt (the property patch_prompt_tokens is meant to preserve).
    concat = [t for m in timeline for t in m.token_arr]
    assert prompt_ids[:len(concat)] == concat, (
        f"after patch_prompt_tokens, history concat is not a prefix of vLLM prompt "
        f"(concat={len(concat)} prompt={len(prompt_ids)})"
    )


def test_step_spawn_timeline_matches_tokenize_segments():
    """After the fix, step_spawn_timeline merges consecutive `tool` messages
    into one ExtendedMessage, so the timeline length now EQUALS the number of
    <|im_start|> segments the chat template renders (7 == 7), and the role
    sequence lines up segment-by-segment. This is what makes patch_prompt_tokens
    safe (its 1:1 message-to-segment assumption holds)."""
    messages = _load_scene()
    tokenizer = _load_tokenizer()
    tr = _make_tracker(tokenizer)
    tools = _build_tools()

    timeline = tr.step_spawn_timeline(messages, tools, disable_toolcalls=False)
    text = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=False, tokenize=False)
    n_segments = len([s for s in text.split("<|im_start|>") if s.strip()])

    assert len(timeline) == n_segments, (
        f"timeline length {len(timeline)} != segment count {n_segments} — "
        f"step_spawn_timeline did not merge consecutive tool messages correctly"
    )
    # 24 raw messages -> 7 timeline messages (1 system + 2 user + 2 assistant
    # + 2 merged tool + 1 summarizing user). Wait: 2 tool because the 19 tool
    # messages are split into 2 runs by the empty-content assistant at msg[4].
    assert len(timeline) == 7, f"expected 7 timeline messages, got {len(timeline)}"
    # Each timeline message maps 1:1 to a rendered segment. Note the chat
    # template renders `tool` messages inside a `<|im_start|>user` block, so a
    # timeline `tool` message corresponds to a `user` segment header — that's
    # expected, not a mismatch.
    seg_headers = [s.split("\n", 1)[0]
                   for s in [x for x in text.split("<|im_start|>") if x.strip()]]
    assert len(seg_headers) == len(timeline), (
        f"segment count {len(seg_headers)} != timeline {len(timeline)}"
    )
    role_to_header = {"system": "system", "user": "user", "assistant": "assistant",
                      "tool": "user"}  # tool messages render as <|im_start|>user
    for i, m in enumerate(timeline):
        assert role_to_header[m.role] == seg_headers[i], (
            f"msg {i}: timeline role {m.role!r} -> expected header "
            f"{role_to_header[m.role]!r}, got {seg_headers[i]!r}"
        )
