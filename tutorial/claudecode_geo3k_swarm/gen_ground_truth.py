# -*- coding: utf-8 -*-
"""
Capture the vLLM-side prompt tokenization for each multimodal test case.

For every case in ``multimodal_cases.CASE_NAMES`` this POSTs the case's OpenAI
``messages`` (+ optional ``tools``) to a running ``vllm serve`` endpoint with
the vLLM flag ``"return_token_ids": true`` (the same method that produced the
original ``token_id_io.md``), then writes ``./token_id/<case>.md`` containing the
prompt token-id array, the ``<|image_pad|>`` count, the per-image grid (derived
from counts), the vision-span markers, and the decoded prompt.

These ``*.md`` files are the *ground truth* the live-swarm capture
(``*.debug.md``) is compared against by ``test_multimodal_cases.py``.

Run (vLLM serving a VL model, e.g. Qwen2.5-VL-7B):

    source .venv/bin/activate
    python -m tutorial.claudecode_geo3k_swarm.gen_ground_truth \\
        --base-url http://localhost:8000/v1 --model fill_whatever_model
"""

import os
import sys
import json
import argparse

import requests

from tutorial.claudecode_geo3k_swarm.multimodal_cases import (
    CASE_NAMES,
    EXPECTED_IMAGE_TOKENS,
    build_case_messages,
)

IMAGE_PAD_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOKEN_DIR = os.path.join(REPO_ROOT, "token_id")


def _post_case(base_url: str, api_key: str, model: str, case: str) -> dict:
    spec = build_case_messages(case)
    payload = {
        "model": model,
        "messages": spec["messages"],
        "stream": False,
        "max_tokens": 64,
        "temperature": 1.0,
        # vLLM-specific: surface top-level prompt_token_ids + per-choice token_ids
        "return_token_ids": True,
    }
    if spec["tools"]:
        payload["tools"] = spec["tools"]
        payload["tool_choice"] = "none"

    resp = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Connection": "close"},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_prompt_ids(data: dict) -> list:
    # vLLM returns prompt token ids at the top level when return_token_ids=True
    for key in ("prompt_token_ids", "prompt_tokens_ids"):
        if key in data and data[key]:
            return list(data[key])
    # some builds nest it under usage or choices; fall back to a search
    raise ValueError(
        "prompt_token_ids not found in response — is this vLLM serve with "
        "--return-tokens-as-token-ids / return_token_ids supported?"
    )


def _write_case_md(case: str, prompt_ids: list) -> str:
    n_pad = prompt_ids.count(IMAGE_PAD_ID)
    has_start = VISION_START_ID in prompt_ids
    has_end = VISION_END_ID in prompt_ids
    n_start = prompt_ids.count(VISION_START_ID)
    lines = [
        f"# vLLM-side prompt tokenization — case `{case}`",
        "",
        "Ground truth for the multimodal token-id test. Compare against the "
        f"training-side capture in `token_id/{case}.debug.md`.",
        "",
        f"- image_pad (151655) count: {n_pad} (expected {EXPECTED_IMAGE_TOKENS[case]})",
        f"- vision spans (<|vision_start|>): {n_start}",
        f"- has vision_start: {has_start}, has vision_end: {has_end}",
        f"- prompt length: {len(prompt_ids)}",
        "",
        "## Case A",  # heading kept so the shared md-parser can locate the array
        "",
        f"Input (prompt) token ids — length {len(prompt_ids)}, image_tokens {n_pad}:",
        "",
        "```json",
        json.dumps(prompt_ids),
        "```",
        "",
    ]
    os.makedirs(TOKEN_DIR, exist_ok=True)
    out = os.path.join(TOKEN_DIR, f"{case}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("GEO3K_TEST_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("GEO3K_TEST_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("GEO3K_TEST_MODEL", "fill_whatever_model"))
    parser.add_argument("--case", default="", help="Only this case (default: all).")
    args = parser.parse_args()

    cases = [args.case] if args.case else CASE_NAMES
    all_ok = True
    for case in cases:
        try:
            data = _post_case(args.base_url, args.api_key, args.model, case)
            prompt_ids = _extract_prompt_ids(data)
            out = _write_case_md(case, prompt_ids)
            n_pad = prompt_ids.count(IMAGE_PAD_ID)
            exp = EXPECTED_IMAGE_TOKENS[case]
            status = "OK" if n_pad == exp else "MISMATCH"
            print(f"[{status}] {case}: len={len(prompt_ids)} image_pad={n_pad} (exp {exp}) -> {out}")
            all_ok = all_ok and (n_pad == exp)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {case}: {e}")
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
