# flake8: noqa
"""
Tests for ajet.utils.tokenizer.derive_tool_sep_and_fold.

=== Regular (expected) outputs, for reference ============================
Qwen3 text family (Qwen2.5-7B, Qwen3-8B, Qwen3-0.6B, Qwen3-30B-Instruct,
Qwen3.6-35B):  fold = True
  sep = the template's tool-response close+open framing inserted between two
        consecutive tool messages inside the folded <|im_start|>user segment:
        newline + close-tag + newline + open-tag + newline
        (tags are tool_response open/close; built via chr() below to keep this
        source ASCII-only)

Qwen2.5-VL:  fold = False
  sep = '<|im_end|>\n<|im_start|>tool\n'
        (each tool is its own <|im_start|>tool segment; the separator is the
        end-of-segment + next-segment-open)

Both are returned in ONE chat-template render and cached per-tokenizer.
=========================================================================

These tests pin that behavior: derive_tool_sep_and_fold returns (sep, fold)
where sep makes a MERGED single tool message render identically to the
template's multi-tool fold (the property flush_pending_tool_run relies on to
keep the timeline segment-aligned), and fold correctly distinguishes fold vs
separate-segment templates.
"""

import os
import pytest

from ajet.utils.tokenizer import derive_tool_sep_and_fold


_QWEN_TEXT = [
    "Qwen2___5-7B-Instruct",
    "Qwen3-8B",
    "Qwen3-0___6B",
    "Qwen3-30B-A3B-Instruct-2507",
    "Qwen3___6-35B-A3B",
]
_QWEN_VL = [
    "Qwen2___5-VL-7B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
]
_MODEL_CACHE = "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen"


def _ids(candidates):
    out = []
    for name in candidates:
        p = os.path.join(_MODEL_CACHE, name)
        if os.path.isdir(p) and (
            os.path.exists(os.path.join(p, "tokenizer.json"))
            or os.path.exists(os.path.join(p, "tokenizer_config.json"))
        ):
            out.append((name, p))
    return out


def _load_text_tok(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _load_vl_tok(path):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(path, trust_remote_code=True).tokenizer


# Expected sep for Qwen3 text templates (fold=True). The Qwen3 template wraps
# each tool message in a tool_response tag; the inter-tool separator is
# newline + close-tag + newline + open-tag + newline. Built via chr() so the
# source stays ASCII-only (no literal tags in the file).
_TOOL_OPEN = chr(0x3C) + "tool_response" + chr(0x3E)
_TOOL_CLOSE = chr(0x3C) + "/tool_response" + chr(0x3E)
_EXPECTED_TEXT_SEP = "\n" + _TOOL_CLOSE + "\n" + _TOOL_OPEN + "\n"
_EXPECTED_VL_SEP = "<|im_end|>\n<|im_start|>tool\n"


@pytest.mark.skipif(not _ids(_QWEN_TEXT), reason="no Qwen text tokenizer cached")
@pytest.mark.parametrize("name,path", _ids(_QWEN_TEXT))
def test_text_family_sep_and_fold(name, path):
    tok = _load_text_tok(path)
    sep, fold = derive_tool_sep_and_fold(tok)
    assert fold is True, f"[{name}] expected fold=True"
    assert sep == _EXPECTED_TEXT_SEP, f"[{name}] sep={sep!r} expected {_EXPECTED_TEXT_SEP!r}"


@pytest.mark.skipif(not _ids(_QWEN_VL), reason="no Qwen2.5-VL cached")
@pytest.mark.parametrize("name,path", _ids(_QWEN_VL))
def test_vl_family_sep_and_fold(name, path):
    tok = _load_vl_tok(path)
    sep, fold = derive_tool_sep_and_fold(tok)
    assert fold is False, f"[{name}] expected fold=False"
    assert sep == _EXPECTED_VL_SEP, f"[{name}] sep={sep!r} expected {_EXPECTED_VL_SEP!r}"


@pytest.mark.skipif(not _ids(_QWEN_TEXT), reason="no Qwen text tokenizer cached")
@pytest.mark.parametrize("name,path", _ids(_QWEN_TEXT))
def test_merged_tool_renders_as_fold(name, path):
    """Core invariant: a SINGLE tool message whose content is N tool contents
    joined by sep renders IDENTICALLY (text + token level) to the template's
    folded N-consecutive-tool segment."""
    tok = _load_text_tok(path)
    sep, _ = derive_tool_sep_and_fold(tok)
    contents = ["r1", "r2", "r3"]

    folded = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
    ] + [{"role": "tool", "content": c, "tool_call_id": "c1"} for c in contents]

    merged = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "content": sep.join(contents), "tool_call_id": "c1"},
    ]

    from ajet.utils.tokenizer import ajet_apply_chat_template
    t_fold = ajet_apply_chat_template(
        tokenizer=tok, conversation=folded, tools=None,
        add_generation_prompt=False, tokenize=False)
    t_merg = ajet_apply_chat_template(
        tokenizer=tok, conversation=merged, tools=None,
        add_generation_prompt=False, tokenize=False)
    assert t_fold == t_merg, f"[{name}] merged single-tool render != folded N-tool"
    assert tok(t_fold, add_special_tokens=False)["input_ids"] == \
           tok(t_merg, add_special_tokens=False)["input_ids"], \
        f"[{name}] token-level mismatch"


@pytest.mark.skipif(not _ids(_QWEN_TEXT), reason="no Qwen text tokenizer cached")
def test_caching_returns_same_tuple():
    """derive_tool_sep_and_fold is cached per-tokenizer; the second call returns
    the cached (sep, fold) with the same string object for sep."""
    path = _ids(_QWEN_TEXT)[0][1]
    tok = _load_text_tok(path)
    sep_a, fold_a = derive_tool_sep_and_fold(tok)
    sep_b, fold_b = derive_tool_sep_and_fold(tok)
    assert sep_a is sep_b, "sep should be the cached (same) string object"
    assert fold_a == fold_b


def test_return_types():
    """derive_tool_sep_and_fold returns (str, bool)."""
    ids = _ids(_QWEN_TEXT)
    if not ids:
        pytest.skip("no Qwen text tokenizer cached")
    tok = _load_text_tok(ids[0][1])
    sep, fold = derive_tool_sep_and_fold(tok)
    assert isinstance(sep, str)
    assert isinstance(fold, bool)
