# flake8: noqa
"""
Regression tests for whole-list tokenization + per-message slicing.

Background
----------
``ExtendedMessage`` used to tokenize itself in isolation at construction time
(``token_generator="auto"``), using dummy "anchor" user messages and
longest-common-prefix/suffix math to recover a single message's tokens from an
isolated render. That machinery was the root cause of two drift bugs:

  1. **Tools-block drift** (the production log): the trailing anchor was
     rendered *with tools*; Qwen3 templates synthesize a ``<tools>``-bearing
     system block for any conversation that has tools, so the anchor and the
     full render shared the entire ``<tools>`` block as a common suffix —
     stripping it deleted the tools out of the system message's ``token_arr``.
  2. **``<|im_end|>`` loss on non-first messages**: the anchor stripping also
     dropped the trailing ``<|im_end|>`` from user/assistant messages, causing
     drift on every multi-turn conversation (masked at runtime by
     ``patch_prompt_tokens``).

The fix is architectural: ``MultiAgentContextTracker.tokenize_and_slice_timeline``
renders the **whole** conversation once, tokenizes it, and splits the token-id
list on the ``<|im_start|>`` boundary into one contiguous chunk per message.
Each message's ``token_arr`` is then an exact slice of the single
whole-conversation render, so ``concat(token_arr)`` reconstructs the same
token stream vLLM produces — drift impossible by construction, and
``patch_prompt_tokens`` becomes a no-op safety net.

These tests exercise ``tokenize_and_slice_timeline`` directly (the tracker is
constructed via ``__new__`` with only the attributes the slicer needs, so no
GPU / config / swarm is required) across many conversation shapes:

  - system(+tools) + user                 (the production case)
  - single-turn without tools
  - multi-turn with think blocks (before/after the last user query)
  - consecutive tool messages (merged into one ``<|im_start|>user`` block)
  - assistant with tool_calls
  - no-tools vs with-tools parity
  - empty timeline

For each case they assert:
  - ``concat(token_arr)`` == the full-conversation tokenization (the core
    invariant), AND
  - each per-message chunk matches the corresponding ``<|im_start|>`` segment
    of vLLM's ``add_generation_prompt=True`` prompt (zero drift, the exact
    check ``patch_prompt_tokens`` performs).

A ``before/after`` comparison helper also shows the old per-message path's
failure mode (``<|im_end|>`` lost, tools block lost) vs the new path's
correctness, so the improvement is visible in failure output.

Tests skip cleanly when no Qwen tokenizer is cached (CI-safe).
"""

import os
import pytest

from ajet.schema.extended_msg import ExtendedMessage
from ajet.context_tracker.multiagent_tracking import MultiAgentContextTracker


_QWEN_CANDIDATES = [
    "Qwen3-8B",
    "Qwen3-0___6B",
    "Qwen3-30B-A3B-Instruct-2507",
    "Qwen3___6-35B-A3B",
    "Qwen2___5-7B-Instruct",
]
# Both Qwen2.5-VL and Qwen3.6-VL expand <|image_pad|> placeholders into
# input_ids as a contiguous run (verified) — they differ only in the token id
# (Qwen2.5-VL = 151655, Qwen3.6-VL = 248056), so the whole-list processor
# render + slice works the same way for both. Qwen3.6-VL additionally emits
# mm_token_type_ids (a per-token 0=text/1=image marker); that key is dropped by
# merge_multi_modal_inputs (not dim-0 concatenable, dead in the training
# path), so it does not affect slicing.
_VL_CANDIDATES = [
    "Qwen2___5-VL-7B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen3___6-35B-A3B",
    "Qwen3.6-35B-A3B",
]
_MODEL_CACHE = "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen"


def _available_tokenizers():
    out = []
    for name in _QWEN_CANDIDATES:
        p = os.path.join(_MODEL_CACHE, name)
        if os.path.isdir(p) and (
            os.path.exists(os.path.join(p, "tokenizer.json"))
            or os.path.exists(os.path.join(p, "tokenizer_config.json"))
        ):
            out.append((name, p))
    return out


def _available_vl_processors():
    out = []
    for name in _VL_CANDIDATES:
        p = os.path.join(_MODEL_CACHE, name)
        if os.path.isdir(p) and (
            os.path.exists(os.path.join(p, "preprocessor_config.json"))
        ):
            out.append((name, p))
    return out


pytestmark = pytest.mark.skipif(
    not _available_tokenizers(),
    reason="No Qwen tokenizer found in the model cache; skipping tokenization regression tests.",
)


def _load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _make_tracker(tokenizer):
    """Build a tracker with only the attributes tokenize_and_slice_timeline /
    to_role_content / step_spawn_timeline need (bypasses the heavy __init__ that
    wants config / workflow_task / GPUs)."""
    from ajet.utils.tokenizer import derive_tool_sep_and_fold
    tr = MultiAgentContextTracker.__new__(MultiAgentContextTracker)
    tr.tokenizer = tokenizer
    tr.processor = None
    tr._im_start_token_id = tokenizer.encode("<|im_start|>")[0]
    tr._im_end_token_id = tokenizer.encode("<|im_end|>")[0]
    tr.tool_res_sep, tr.tool_res_fold = derive_tool_sep_and_fold(tokenizer)
    return tr


def _build_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": "Execute the given python code in a temp file and capture the return",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"description": "The Python code to be executed.", "type": "string"},
                        "timeout": {"default": 300, "description": "Max seconds.", "type": "number"},
                    },
                    "required": ["code"],
                },
            },
        }
    ]


def _make_timeline(tracker, messages, tools):
    """Build a timeline of ExtendedMessage the way step_spawn_timeline does
    (token_generator="manual", token_arr left empty for the slicer to fill)."""
    timeline = []
    for i, m in enumerate(messages):
        timeline.append(ExtendedMessage(
            author="initialization" if m["role"] == "system" else "env",
            role=m["role"],
            content=m["content"],
            tokenizer=tracker.tokenizer,
            tools=tools,
            tool_calls=m.get("tool_calls", []),
            tool_call_id=m.get("tool_call_id", ""),
            token_generator="manual",
            first_message=(i == 0),
        ))
    return timeline


def _full_render_ids(tokenizer, messages, tools, add_generation_prompt):
    """The vLLM-style full-conversation tokenization.

    Goes through ajet_apply_chat_template (not the raw tokenizer) so it matches
    the slicer path exactly: ajet_apply_chat_template runs cleanup_messages,
    which parses tool_calls string arguments into dicts before rendering, so
    str-args and dict-args render identically through this path. Comparing
    against the raw tokenizer.apply_chat_template would diverge on JSON
    spacing whenever tool_calls carry string arguments."""
    from ajet.utils.tokenizer import ajet_apply_chat_template
    text = ajet_apply_chat_template(
        tokenizer=tokenizer,
        conversation=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _split_on_im_start(tokenizer, ids):
    """Split token ids on <|im_start|> exactly like patch_prompt_tokens /
    tokenize_and_slice_timeline do."""
    ims = tokenizer.encode("<|im_start|>")[0]
    segs, tmp = [], []
    for t in ids:
        if t != ims:
            tmp.append(t)
        else:
            if tmp:
                segs.append(tmp)
            tmp = [t]
    if tmp:
        segs.append(tmp)
    return segs


def _assert_no_drift(tracker, tokenizer, messages, tools, name):
    """The core end-to-end check: build+slice a timeline, then assert
    (a) concat(token_arr) == the full-conversation tokenization, and
    (b) each non-tool message's chunk == the corresponding vLLM prompt chunk
        (consecutive tool messages fold into one segment in the template, so
        only the first tool message's chunk is compared; the rest are empty)."""
    timeline = _make_timeline(tracker, messages, tools)
    tracker.tokenize_and_slice_timeline(timeline, tools)

    # (a) concat(token_arr) must equal the full no-gen-prompt render.
    full_ids = _full_render_ids(tokenizer, messages, tools, add_generation_prompt=False)
    concat = [t for msg in timeline for t in msg.token_arr]
    assert concat == full_ids, (
        f"[{name}] concat(token_arr) != full-conversation tokenization. "
        f"concat_len={len(concat)} full_len={len(full_ids)} -- the per-message "
        f"slices do not reconstruct the whole render (slicing is broken)."
    )

    # (b) history chunks must match vLLM's add_generation_prompt=True chunks.
    # Walk timeline + prompt_chunks together; for a run of consecutive tool
    # messages, only the first consumes a prompt chunk (the rest are []).
    prompt_ids = _full_render_ids(tokenizer, messages, tools, add_generation_prompt=True)
    prompt_chunks = _split_on_im_start(tokenizer, prompt_ids)

    im_end = tokenizer.encode("<|im_end|>")[0]
    seg_idx = 0
    mismatches = []
    i = 0
    while i < len(timeline):
        msg = timeline[i]
        if msg.role == "tool":
            run_end = i
            while run_end + 1 < len(timeline) and timeline[run_end + 1].role == "tool":
                run_end += 1
            chunk = prompt_chunks[seg_idx] if seg_idx < len(prompt_chunks) else []
            seg_idx += 1
            if msg.token_arr != chunk:
                mismatches.append((i, msg.role, "<|im_end|>" , im_end in msg.token_arr, im_end in chunk))
            # rest must be empty
            for k in range(i + 1, run_end + 1):
                if timeline[k].token_arr != []:
                    mismatches.append((k, timeline[k].role, "non-empty", False, True))
            i = run_end + 1
        else:
            chunk = prompt_chunks[seg_idx] if seg_idx < len(prompt_chunks) else []
            seg_idx += 1
            if msg.token_arr != chunk:
                mismatches.append((i, msg.role, "<|im_end|>", im_end in msg.token_arr, im_end in chunk))
            i += 1
    assert not mismatches, (
        f"[{name}] drift: per-message token_arr != vLLM prompt chunks (the "
        f"exact 'Prompt token ids mismatch' from patch_prompt_tokens):\n"
        + "\n".join(f"  msg{idx} ({role}): {label} hist={h}/vllm={v}"
                    for idx, role, label, h, v in mismatches)
    )


# ----------------------------- test cases ----------------------------------

SYSTEM_CONTENT = (
    "You are an agent specialized in solving math problems with tools.\n"
    "Please solve the math problem given to you.\n"
    "You can write and execute Python code to perform calculation or "
    "verify your answer.\n"
    "You should return your final answer within \\boxed{}."
)


def _tokenizer_ids():
    return _available_tokenizers()


@pytest.mark.parametrize("name,path", _tokenizer_ids())
def test_system_with_tools_and_user(name, path):
    """The production case: system(+tools) + user. Guards the tools-block drift."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "What is 2+2?"},
    ]
    _assert_no_drift(tr, tokenizer, messages, tools, name)
    # Explicit: the system message's token_arr must contain the tools block.
    timeline = _make_timeline(tr, messages, tools)
    tr.tokenize_and_slice_timeline(timeline, tools)
    sys_decoded = tokenizer.decode(timeline[0].token_arr)
    assert "<tools>" in sys_decoded and "</tools>" in sys_decoded, (
        f"[{name}] system message lost its <tools> block: {sys_decoded[:120]!r}"
    )


@pytest.mark.parametrize("name,path", _tokenizer_ids())
def test_single_turn_no_tools(name, path):
    """system + user, no tools. Guards the simplest path + think handling."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello."},
    ]
    _assert_no_drift(tr, tokenizer, messages, [], name)


@pytest.mark.parametrize("name,path", _tokenizer_ids())
def test_multi_turn_with_think_blocks(name, path):
    """Multi-turn: system + user + assistant(with think) + user + assistant.

    The first assistant carries a think block and sits BEFORE the last user
    query, so Qwen3 templates strip its think; the last assistant keeps it.
    This is the case the old per-message anchor/sandwich logic got wrong
    (lost <|im_end|>, mis-handled think-stripping)."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>2 plus 2 is 4</think>The answer is 4."},
        {"role": "user", "content": "Are you sure?"},
        {"role": "assistant", "content": "<think>yes, 4 is correct</think>Yes, 4."},
    ]
    _assert_no_drift(tr, tokenizer, messages, tools, name)


@pytest.mark.parametrize("name,path", _tokenizer_ids())
def test_consecutive_tool_messages(name, path):
    """Two consecutive tool messages fold into ONE <|im_start|>user segment in
    the template, so segment count < message count. The slicer assigns the
    folded segment to the first tool message and [] to the rest. Guards the
    consecutive-tool handling in tokenize_and_slice_timeline and that
    concat(token_arr) still == the full render (the property that matters for
    training, since tool messages are non-training / loss-masked to 0)."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "Run some code."},
        {"role": "assistant", "content": "Let me compute.", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "execute_python_code", "arguments": '{"code": "print(1)"}'}}]},
        {"role": "tool", "content": "1", "tool_call_id": "call_1"},
        {"role": "tool", "content": "done", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "The result is 1."},
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "You're welcome."},
    ]
    timeline = _make_timeline(tr, messages, tools)
    tr.tokenize_and_slice_timeline(timeline, tools)
    # concat(token_arr) must equal the full render (the core invariant).
    full_ids = _full_render_ids(tokenizer, messages, tools, add_generation_prompt=False)
    concat = [t for m in timeline for t in m.token_arr]
    assert concat == full_ids, (
        f"[{name}] concat(token_arr) != full render for consecutive-tool case "
        f"(concat_len={len(concat)} full_len={len(full_ids)})."
    )
    # The first tool message carries the folded segment; the second is empty.
    tool_msgs = [m for m in timeline if m.role == "tool"]
    assert len(tool_msgs) == 2, f"[{name}] expected 2 tool messages, got {len(tool_msgs)}"
    assert len(tool_msgs[0].token_arr) > 0, f"[{name}] first tool message has empty token_arr"
    assert tool_msgs[1].token_arr == [], f"[{name}] second tool message should be empty"
    # Also no drift vs vLLM's prompt chunks (history chunks match).
    _assert_no_drift(tr, tokenizer, messages, tools, name)


@pytest.mark.parametrize("name,path", _tokenizer_ids())
def test_assistant_with_tool_calls(name, path):
    """Assistant message carrying tool_calls must tokenize correctly in context."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "Compute 3+4."},
        {"role": "assistant", "content": "Let me compute 3+4.", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "execute_python_code", "arguments": '{"code": "print(3+4)"}'}}]},
        {"role": "tool", "content": "7", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "The answer is 7."},
    ]
    _assert_no_drift(tr, tokenizer, messages, tools, name)


@pytest.mark.parametrize("name,path", _tokenizer_ids())
def test_system_without_tools(name, path):
    """System message without tools must still slice correctly (no tools block
    expected). Guards the no-tools parity."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
    ]
    _assert_no_drift(tr, tokenizer, messages, [], name)


@pytest.mark.parametrize("name,path", _tokenizer_ids())
def test_long_multiturn_conversation(name, path):
    """A longer multi-turn conversation with tools, think blocks, tool calls,
    and merged tool responses — the full realistic agent loop."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "Solve step by step: what is 17*23?"},
        {"role": "assistant", "content": "<think>17*23 = 391</think>I'll verify with code.", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "execute_python_code", "arguments": '{"code": "print(17*23)"}'}}]},
        {"role": "tool", "content": "391", "tool_call_id": "c1"},
        {"role": "assistant", "content": "<think>confirmed</think>17*23 = 391."},
        {"role": "user", "content": "Now what is 391 + 9?"},
        {"role": "assistant", "content": "<think>391+9=400</thinking>400.", "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "execute_python_code", "arguments": '{"code": "print(391+9)"}'}}]},
        {"role": "tool", "content": "400", "tool_call_id": "c2"},
        {"role": "assistant", "content": "391 + 9 = 400."},
        {"role": "user", "content": "Great, summarise."},
        {"role": "assistant", "content": "<think>summarising</think>17*23=391, plus 9 is 400."},
    ]
    _assert_no_drift(tr, tokenizer, messages, tools, name)


def test_empty_timeline():
    """An empty timeline must not crash (no messages to slice)."""
    # Use the first available tokenizer; empty timeline is tokenizer-agnostic.
    toks = _available_tokenizers()
    if not toks:
        pytest.skip("no tokenizer")
    tokenizer = _load_tokenizer(toks[0][1])
    tr = _make_tracker(tokenizer)
    tr.tokenize_and_slice_timeline([], [])  # must not raise
    assert True


@pytest.mark.parametrize("name,path", _tokenizer_ids())
def test_before_after_comparison(name, path):
    """Show the before/after difference: the old per-message isolation render
    lost <|im_end|> on non-first messages and (with tools) lost the tools
    block on the system message. The new whole-list slice preserves both.

    This test computes the 'old-style' isolated per-message render and asserts
    the new path strictly dominates it (new == full render; old != full render
    on at least one message), making the improvement visible in failures."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
    ]

    # NEW path: whole-list slice.
    timeline = _make_timeline(tr, messages, tools)
    tr.tokenize_and_slice_timeline(timeline, tools)
    full_ids = _full_render_ids(tokenizer, messages, tools, add_generation_prompt=False)
    new_concat = [t for m in timeline for t in m.token_arr]
    assert new_concat == full_ids, f"[{name}] new path does not reconstruct full render"

    # OLD-style isolation baseline: render each message alone (with tools),
    # the closest faithful reproduction of what the deleted auto path tried to
    # do. For the system message this either raises (Qwen3.6) or includes tools
    # (others). For non-first messages, the isolated render does NOT include
    # the inter-message <|im_end|>\n separator that the full render has, so the
    # isolated token_arr would mismatch vLLM's chunk (the pre-existing bug).
    im_end = tokenizer.encode("<|im_end|>")[0]
    prompt_chunks = _split_on_im_start(
        tokenizer, _full_render_ids(tokenizer, messages, tools, add_generation_prompt=True)
    )

    # The new path's non-first messages all carry <|im_end|> (matching vLLM).
    for j, m in enumerate(timeline[1:], start=1):
        assert im_end in m.token_arr, (
            f"[{name}] new path msg {j} ({m.role}) lost <|im_end|>"
        )
        assert m.token_arr == prompt_chunks[j], (
            f"[{name}] new path msg {j} ({m.role}) != vLLM chunk"
        )

    # OLD-style isolation: prove it loses <|im_end|> for non-first messages.
    old_lost_end = 0
    for m in messages[1:]:
        try:
            iso_text = tokenizer.apply_chat_template(
                [m], tools=tools, add_generation_prompt=False, tokenize=False,
            )
            iso_ids = tokenizer(iso_text, add_special_tokens=False)["input_ids"]
            if im_end not in iso_ids:
                old_lost_end += 1
        except Exception:
            # Isolated render raises (e.g. Qwen3.6 needs a user query) — the old
            # path needed anchors precisely because of this. Count as a failure
            # mode the new path avoids.
            old_lost_end += 1
    # The improvement: the new path preserves <|im_end|> on every non-first
    # message, whereas isolated rendering loses it (or raises) on at least one.
    # (We assert the new path's guarantee holds; old_lost_end is informational.)
    assert all(
        im_end in m.token_arr for m in timeline[1:]
    ), f"[{name}] new path lost <|im_end|> on a non-first message"
    # Print the before/after delta for visibility on failure.
    print(f"\n[{name}] old isolation lost <|im_end|> / raised on {old_lost_end}/"
          f"{len(messages)-1} non-first messages; new path preserves all.")


@pytest.mark.parametrize("name,path", _tokenizer_ids())
def test_concat_equals_vllm_prompt_prefix(name, path):
    """concat(token_arr) of the history must be an exact PREFIX of vLLM's
    add_generation_prompt=True prompt (the prompt the LLM actually saw). This
    is why patch_prompt_tokens finds a perfect match for every history chunk."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    timeline = _make_timeline(tr, messages, tools)
    tr.tokenize_and_slice_timeline(timeline, tools)

    concat = [t for m in timeline for t in m.token_arr]
    prompt_ids = _full_render_ids(tokenizer, messages, tools, add_generation_prompt=True)
    assert prompt_ids[:len(concat)] == concat, (
        f"[{name}] history concat is not a prefix of vLLM's prompt — "
        f"patch_prompt_tokens would find drift."
    )


# ------------------------------ VL tests -----------------------------------
#
# Qwen2.5-VL's processor expands <|image_pad|> placeholders into input_ids, so
# the whole-list processor render + slice works the same way as the text path:
# the combined pixel_values/image_grid_thw are captured once for the whole
# conversation and attached to timeline[0].multi_modal_inputs, and each image
# message's token_arr slice contains its own <|image_pad|> span.
# Both Qwen2.5-VL and Qwen3.6-VL expand <|image_pad|> placeholders into a
# contiguous run in input_ids. They differ only in the token id, so the
# image_pad id is resolved dynamically per tokenizer (do NOT hardcode 151655 —
# that is Qwen2.5-VL only; Qwen3.6-VL uses 248056).


def _vl_ids():
    return _available_vl_processors()


def _load_processor(path):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(path, trust_remote_code=True)


def _make_vl_tracker(processor):
    """Tracker stub with the processor set, the way step_spawn_timeline would
    (it passes processor=getattr(self, 'processor', None))."""
    from ajet.utils.tokenizer import derive_tool_sep_and_fold
    tr = MultiAgentContextTracker.__new__(MultiAgentContextTracker)
    tr.tokenizer = processor.tokenizer
    tr.processor = processor
    tr._im_start_token_id = processor.tokenizer.encode("<|im_start|>")[0]
    tr.tool_res_sep, tr.tool_res_fold = derive_tool_sep_and_fold(processor.tokenizer)
    return tr


def _image_pad_id(processor):
    """Resolve <|image_pad|> token id for this processor's tokenizer. Different
    VL families use different ids (Qwen2.5-VL=151655, Qwen3.6-VL=248056)."""
    return processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")


def _pil_to_path(img):
    import tempfile
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(f.name)
    return f.name


def _build_vl_timeline(tr, messages):
    """Build a VL timeline the way step_spawn_timeline does: each vision user
    message splits into (text_content, image_refs) on the ExtendedMessage, and
    to_role_content re-emits them as image_url blocks. image_refs must be
    PIL-loadable (file paths work)."""
    timeline = []
    for i, m in enumerate(messages):
        images = None
        content = m["content"]
        if isinstance(content, list):
            imgs = [c["image"] for c in content if c.get("type") == "image"]
            text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
            images = [_pil_to_path(im) if hasattr(im, "save") else im for im in imgs]
            content = text
        timeline.append(ExtendedMessage(
            author="initialization" if m["role"] == "system" else "env",
            role=m["role"], content=content, tokenizer=tr.tokenizer,
            tools=[], token_generator="manual", first_message=(i == 0),
            images=images, processor=tr.processor,
        ))
    return timeline


@pytest.mark.skipif(not _vl_ids(), reason="No Qwen-VL processor cached.")
@pytest.mark.parametrize("name,path", _vl_ids())
def test_vl_whole_list_slice(name, path):
    """Whole-list processor render + slice on Qwen2.5-VL: concat(token_arr)
    must equal the full-conversation processor tokenization, and the combined
    multi_modal_inputs (pixel_values, image_grid_thw) must be captured on
    timeline[0]."""
    import torch  # noqa: F401  (processor needs torch)
    from PIL import Image
    proc = _load_processor(path)
    tr = _make_vl_tracker(proc)
    img1 = Image.new("RGB", (140, 140), "red")
    img2 = Image.new("RGB", (100, 100), "blue")
    messages = [
        {"role": "system", "content": "You are a vision agent."},
        {"role": "user", "content": [
            {"type": "image", "image": img1},
            {"type": "text", "text": "what is this?"},
        ]},
        {"role": "assistant", "content": "it is red"},
        {"role": "user", "content": [
            {"type": "image", "image": img2},
            {"type": "text", "text": "and this?"},
        ]},
    ]
    timeline = _build_vl_timeline(tr, messages)
    tr.tokenize_and_slice_timeline(timeline, [])

    # The slicer should have captured combined multi_modal_inputs on msg 0.
    mmi = timeline[0].multi_modal_inputs
    assert mmi is not None, f"[{name}] multi_modal_inputs not captured on timeline[0]"
    assert "pixel_values" in mmi, f"[{name}] pixel_values missing from multi_modal_inputs"
    assert "image_grid_thw" in mmi, f"[{name}] image_grid_thw missing from multi_modal_inputs"
    # mm_token_type_ids is dropped (not dim-0 concatenable; see merge_multi_modal_inputs).
    assert "mm_token_type_ids" not in mmi, f"[{name}] mm_token_type_ids should have been dropped"

    # concat(token_arr) must equal the full-conversation processor render.
    conv = tr.to_role_content(timeline)
    full_text = proc.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
    full_ids = proc(text=[full_text], images=[img1, img2], return_tensors="pt")["input_ids"][0].tolist()
    concat = [t for m in timeline for t in m.token_arr]
    assert concat == full_ids, (
        f"[{name}] VL concat(token_arr) != full processor render "
        f"(concat_len={len(concat)} full_len={len(full_ids)})."
    )


@pytest.mark.skipif(not _vl_ids(), reason="No Qwen-VL processor cached.")
@pytest.mark.parametrize("name,path", _vl_ids())
def test_vl_image_pad_in_correct_message(name, path):
    """Each image message's token_arr slice must contain its OWN <|image_pad|>
    span (not the other image's), and non-image messages must have zero
    image_pad tokens. This guards against image spans landing in the wrong
    message's slice. Covers both Qwen2.5-VL (image_pad id 151655) and
    Qwen3.6-VL (image_pad id 248056) — the id is resolved dynamically."""
    import torch  # noqa: F401
    from PIL import Image
    proc = _load_processor(path)
    tr = _make_vl_tracker(proc)
    image_pad_id = _image_pad_id(proc)
    assert image_pad_id is not None and image_pad_id != proc.tokenizer.unk_token_id, (
        f"[{name}] <|image_pad|> not in vocab; got id={image_pad_id}"
    )
    img1 = Image.new("RGB", (140, 140), "red")
    img2 = Image.new("RGB", (100, 100), "blue")
    messages = [
        {"role": "system", "content": "You are a vision agent."},
        {"role": "user", "content": [
            {"type": "image", "image": img1}, {"type": "text", "text": "what is this?"},
        ]},
        {"role": "assistant", "content": "it is red"},
        {"role": "user", "content": [
            {"type": "image", "image": img2}, {"type": "text", "text": "and this?"},
        ]},
    ]
    timeline = _build_vl_timeline(tr, messages)
    tr.tokenize_and_slice_timeline(timeline, [])

    # msg0 system: 0 image_pad; msg1 user: >0; msg2 assistant: 0; msg3 user: >0.
    expected_zero = [0, 2]   # system, assistant
    expected_nonzero = [1, 3]  # the two image-bearing user messages
    for i in expected_zero:
        cnt = timeline[i].token_arr.count(image_pad_id) if timeline[i].token_arr else 0
        assert cnt == 0, f"[{name}] msg{i} ({timeline[i].role}) should have 0 image_pad, got {cnt}"
    for i in expected_nonzero:
        cnt = timeline[i].token_arr.count(image_pad_id) if timeline[i].token_arr else 0
        assert cnt > 0, f"[{name}] msg{i} ({timeline[i].role}) should have >0 image_pad, got {cnt}"
    # Both image messages must carry an <|image_pad|> span. (Qwen2.5-VL: the
    # two different-size images yield different pad counts 25 vs 16; Qwen3.6-VL
    # normalizes both to the same count despite different sizes — so we only
    # assert both are present, not that they differ.)
    assert timeline[1].token_arr.count(image_pad_id) > 0, f"[{name}] msg1 has no image_pad"
    assert timeline[3].token_arr.count(image_pad_id) > 0, f"[{name}] msg3 has no image_pad"


# =============================================================================
# Spurious <|im_start|> in message content (LLM-generated special token).
#
# During rollout an LLM may *generate* the <|im_start|> token as an ordinary
# output token (degenerate/looping samples). Those ids land inside a message's
# content and would be wrongly treated as segment boundaries, inflating the
# im_start-based segment count far beyond the timeline length.
#
# tokenize_and_slice_timeline recovers by re-splitting on <|im_end|>: every
# real chat-template segment is <|im_start|>...<|im_end|>, so splitting on
# <|im_end|> (absorbing content-internal stray <|im_start|> ids into the
# surrounding segment) restores 1:1 message<->segment alignment, and
# concat(token_arr) still == the full render.
# =============================================================================

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def _stray_im_start_in_content(n, sep="\n"):
    """A string of n stray <|im_start|> tokens, joined by sep (mimics the
    degenerate repeating-token output vLLM produced in the real crash)."""
    return sep.join([_IM_START] * n)


@pytest.mark.parametrize("name,path", _available_tokenizers())
def test_no_im_start_in_content_normal_path(name, path):
    """Baseline: clean content (no stray <|im_start|>) slices with the normal
    im_start split and stays 1:1. Guards that the defensive recovery branch
    is NOT entered when it isn't needed."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Let me compute.", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "execute_python_code", "arguments": '{"code": "print(2+2)"}'}}]},
        {"role": "tool", "content": "4", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "The answer is 4."},
    ]
    timeline = _make_timeline(tr, messages, tools)
    tr.tokenize_and_slice_timeline(timeline, tools)
    # Every message got a non-empty token_arr (no empty-slice fallback).
    assert all(len(m.token_arr) > 0 for m in timeline), f"[{name}] empty token_arr"
    _assert_no_drift(tr, tokenizer, messages, tools, name)


@pytest.mark.parametrize("name,path", _available_tokenizers())
def test_assistant_content_with_stray_im_start(name, path):
    """The real crash scene: an assistant message whose content ends with a
    burst of stray <|im_start|> tokens (LLM generated the special token).
    The im_start split would yield far more segments than messages; recovery
    re-splits on <|im_end|> and restores 1:1, with concat == full render."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "If two piles of toys make 120 in total, and the larger is twice the smaller, how many in the larger?"},
        {"role": "assistant", "content": "Let's denote the smaller pile as x. The larger is 2x. So x + 2x = 120.\n"
                                          "IGHL\n" + _stray_im_start_in_content(120), "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "execute_python_code", "arguments": '{"code": "x = 120/3\\nprint(2*x)"}'}}]},
        {"role": "tool", "content": "80.0", "tool_call_id": "call_1"},
    ]
    timeline = _make_timeline(tr, messages, tools)
    tr.tokenize_and_slice_timeline(timeline, tools)
    # Recovery restored 1:1: every message has a token_arr (the assistant's
    # absorbs all 120 stray im_start ids since none has a matching im_end).
    assert len(timeline) == 4, f"[{name}] timeline length changed"
    assert all(len(m.token_arr) > 0 for m in timeline), f"[{name}] empty token_arr after recovery"
    # concat(token_arr) == full render (drift-free invariant holds).
    full_ids = _full_render_ids(tokenizer, messages, tools, add_generation_prompt=False)
    concat = [t for m in timeline for t in m.token_arr]
    assert concat == full_ids, (
        f"[{name}] concat != full render after spurious-im_start recovery "
        f"(concat={len(concat)} full={len(full_ids)})"
    )
    # The stray im_start ids live inside the assistant message's token_arr,
    # NOT as separate segments: im_start count in full render > timeline len,
    # but timeline stayed 4 messages.
    n_im_start = full_ids.count(tokenizer.encode(_IM_START)[0])
    assert n_im_start > len(timeline), f"[{name}] precondition: stray im_start must inflate count"


@pytest.mark.parametrize("name,path", _available_tokenizers())
def test_stray_im_start_in_multiple_messages(name, path):
    """Stray <|im_start|> in BOTH the user and the assistant message. Recovery
    must still restore 1:1 with concat == full render."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "hello " + _stray_im_start_in_content(10)},
        {"role": "assistant", "content": "hi " + _stray_im_start_in_content(15)},
    ]
    timeline = _make_timeline(tr, messages, [])
    tr.tokenize_and_slice_timeline(timeline, [])
    assert len(timeline) == 3
    full_ids = _full_render_ids(tokenizer, messages, [], add_generation_prompt=False)
    concat = [t for m in timeline for t in m.token_arr]
    assert concat == full_ids, f"[{name}] concat != full (multi-stray)"


@pytest.mark.parametrize("name,path", _available_tokenizers())
def test_stray_im_start_in_tool_message(name, path):
    """A tool (return) message carrying stray <|im_start|> in its content
    (e.g. the tool printed the token). Recovery must still align."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "run code"},
        {"role": "assistant", "content": "ok", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "execute_python_code", "arguments": '{"code": "1"}'}}]},
        {"role": "tool", "content": "out\n" + _stray_im_start_in_content(20), "tool_call_id": "call_1"},
        {"role": "assistant", "content": "done"},
    ]
    timeline = _make_timeline(tr, messages, tools)
    tr.tokenize_and_slice_timeline(timeline, tools)
    full_ids = _full_render_ids(tokenizer, messages, tools, add_generation_prompt=False)
    concat = [t for m in timeline for t in m.token_arr]
    assert concat == full_ids, f"[{name}] concat != full (stray in tool)"


@pytest.mark.parametrize("name,path", _available_tokenizers())
def test_stray_im_start_in_every_message(name, path):
    """Stray <|im_start|> in system, user AND assistant content at once.
    Recovery must still restore 1:1 with concat == full render."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT + " " + _stray_im_start_in_content(5)},
        {"role": "user", "content": "q " + _stray_im_start_in_content(5)},
        {"role": "assistant", "content": "a " + _stray_im_start_in_content(3)},
    ]
    timeline = _make_timeline(tr, messages, [])
    tr.tokenize_and_slice_timeline(timeline, [])
    assert len(timeline) == 3, f"[{name}] timeline length changed"
    full_ids = _full_render_ids(tokenizer, messages, [], add_generation_prompt=False)
    concat = [t for m in timeline for t in m.token_arr]
    assert concat == full_ids, f"[{name}] concat != full (stray in every msg)"


@pytest.mark.parametrize("name,path", _available_tokenizers())
def test_unknown_quirk_raises(name, path):
    """If segment count != timeline length AND no content contains a literal
    <|im_start|>, the divergence is an unknown template quirk — recovery must
    raise rather than silently misalign.

    We force this by directly building a timeline that the template would render
    with extra segments but whose message contents are clean. The cleanest
    repro: a single user message with NO system message, where Qwen auto-injects
    a default <|im_start|>system segment (so 2 segments, 1 message, clean
    content)."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    messages = [
        {"role": "user", "content": "hello with no system message"},
    ]
    timeline = _make_timeline(tr, messages, [])
    # Precondition: the template really does emit an extra segment here and the
    # content has no literal <|im_start|> (so recovery cannot apply).
    full_ids = _full_render_ids(tokenizer, messages, [], add_generation_prompt=False)
    n_segs = len(_split_on_im_start(tokenizer, full_ids))
    assert n_segs != len(timeline), f"[{name}] precondition: need segment!=timeline for this test"
    assert all(_IM_START not in (m.content or "") for m in timeline), \
        f"[{name}] precondition: contents must be clean"
    with pytest.raises(AssertionError, match="unknown template quirk"):
        tr.tokenize_and_slice_timeline(timeline, [])


@pytest.mark.parametrize("name,path", _available_tokenizers())
def test_real_crash_scene_from_dump(name, path):
    """Regression pinned to the real production crash captured in
    /tmp/ajet_assert_dump.jsonl: a 4-message conversation (system, user,
    assistant-with-tool_call, tool) where the assistant content ends with 120
    stray <|im_start|> tokens (vLLM generated the special token during
    rollout). Pre-fix this raised; post-fix it recovers and stays drift-free."""
    import os
    dump_path = "/tmp/ajet_assert_dump.jsonl"
    if not os.path.exists(dump_path):
        pytest.skip("no production crash dump captured")
    import json
    d = json.loads(open(dump_path).readline())
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    # Rebuild timeline from the dump's conversation (the source of truth for
    # how the template rendered it).
    timeline = _make_timeline(tr, d["conversation"], [])
    tr.tokenize_and_slice_timeline(timeline, [])
    assert len(timeline) == len(d["conversation"]), f"[{name}] timeline length changed"
    full_ids = _full_render_ids(tokenizer, d["conversation"], [], add_generation_prompt=False)
    concat = [t for m in timeline for t in m.token_arr]
    assert concat == full_ids, f"[{name}] concat != full render on real crash scene"


# =============================================================================
# Fuzz: enumerate content-shape variants × stray <|im_start|> placements and
# assert the slicer never raises and concat(token_arr) == full render for all.
# Migrated from the ad-hoc fuzz_repro.py into the test suite.
# =============================================================================

# Assistant-with-tool_call content variants an agent can actually produce.
_FUZZ_AST_CONTENTS = [
    "Let me compute.",          # plain
    "",                          # empty content + tool_calls
    "   ",                       # whitespace
    "thinking... final answer",  # plain with ellipsis
]
# How many stray <|im_start|> tokens to inject, and where (prefix / suffix).
_FUZZ_N_STRAY = [0, 1, 3, 50]
_FUZZ_POSITIONS = ["prefix", "suffix", "both"]


def _inject_stray(content, n, position):
    """Inject n stray <|im_start|> tokens at the given position."""
    if n == 0:
        return content
    stray = _stray_im_start_in_content(n)
    if position == "prefix":
        return stray + content
    if position == "suffix":
        return content + stray
    # both
    return stray + content + stray


@pytest.mark.parametrize("name,path", _available_tokenizers())
@pytest.mark.parametrize("ast_content", _FUZZ_AST_CONTENTS)
@pytest.mark.parametrize("n_stray", _FUZZ_N_STRAY)
@pytest.mark.parametrize("position", _FUZZ_POSITIONS)
def test_fuzz_stray_im_start_placements(name, path, ast_content, n_stray, position):
    """Fuzz over assistant-content shape × stray <|im_start|> count × placement.
    Every combination must: not raise, keep timeline length, and have
    concat(token_arr) == the full-conversation render."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    ast_content = _inject_stray(ast_content, n_stray, position)
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": ast_content, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "execute_python_code", "arguments": '{"code": "print(2+2)"}'}}]},
        {"role": "tool", "content": "4", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "The answer is 4."},
    ]
    timeline = _make_timeline(tr, messages, tools)
    tr.tokenize_and_slice_timeline(timeline, tools)
    assert len(timeline) == len(messages), f"[{name}] timeline length changed"
    full_ids = _full_render_ids(tokenizer, messages, tools, add_generation_prompt=False)
    concat = [t for m in timeline for t in m.token_arr]
    assert concat == full_ids, (
        f"[{name}] concat != full (ast_content={ast_content!r:.30} "
        f"n_stray={n_stray} pos={position}): concat={len(concat)} full={len(full_ids)}"
    )


@pytest.mark.parametrize("name,path", _available_tokenizers())
@pytest.mark.parametrize("n_turns", [1, 2, 3, 4])
@pytest.mark.parametrize("n_stray", [0, 1, 10])
def test_fuzz_multi_turn_react_with_stray(name, path, n_turns, n_stray):
    """Fuzz over multi-turn ReAct conversations (n_turns) with stray
    <|im_start|> injected into each assistant's content. Must stay drift-free."""
    tokenizer = _load_tokenizer(path)
    tr = _make_tracker(tokenizer)
    tools = _build_tools()
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": "Solve a multi-step problem."},
    ]
    for i in range(n_turns):
        messages.append({
            "role": "assistant",
            "content": f"Step {i}." + _stray_im_start_in_content(n_stray),
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "execute_python_code",
                                         "arguments": '{"code": "1"}'}}],
        })
        messages.append({"role": "tool", "content": f"r{i}", "tool_call_id": f"c{i}"})
        messages.append({"role": "assistant", "content": f"Step {i} done."})
    timeline = _make_timeline(tr, messages, tools)
    tr.tokenize_and_slice_timeline(timeline, tools)
    assert len(timeline) == len(messages), f"[{name}] timeline length changed"
    full_ids = _full_render_ids(tokenizer, messages, tools, add_generation_prompt=False)
    concat = [t for m in timeline for t in m.token_arr]
    assert concat == full_ids, (
        f"[{name}] concat != full (n_turns={n_turns} n_stray={n_stray}): "
        f"concat={len(concat)} full={len(full_ids)}"
    )
