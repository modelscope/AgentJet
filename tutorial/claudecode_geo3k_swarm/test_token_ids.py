# -*- coding: utf-8 -*-
"""
End-to-end token-id check for the geo3k multimodal fix.

Purpose
-------
Prove that an image survives the whole swarm pipeline and reaches the
*training* tokenization as real ``<|image_pad|>`` (151655) tokens — not
just the vLLM-side request. It ties together two captures:

  * ``token_id_io.md``       — the vLLM-side tokenization of the geo3k
                               "Find y." request (case A = with image =
                               189 prompt tokens, 54 image tokens).
  * ``token_id_io.debug.md`` — the training-side tokenization dumped at
                               ``ajet/backbone/trainer_verl.py:599`` by
                               ``capture_training_token_ids`` (gated on the
                               ``AJET_CAPTURE_TOKEN_IDS`` env var, set in the
                               swarm-server process).

This client feeds a synthetic task that reproduces case A byte-for-byte
(a 256x168 image + the user text ``"<image>Find y."``, to which the agent
appends the ``\\boxed{}`` suffix), drives the swarm until one training
step captures the batch, then compares the two prompt token arrays.

Only the *prompt* token pattern is compared. The model's answer (the
response tokens) is allowed to differ between runs — as long as the
prompt tokenization (system + vision span + user text + generation
prompt) matches, the multimodal path is proven correct.

Run
---
Server tmux (must export the capture path so the trainer writes it)::

    source .venv/bin/activate
    export AJET_CAPTURE_TOKEN_IDS=$PWD/token_id_io.debug.md
    ajet-swarm start --swarm-port=10086

Client tmux::

    source .venv/bin/activate
    export AJET_SWARM_URL=http://localhost:10086
    export REMOTE_MODEL_PATH=/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-VL-7B-Instruct
    export FORCE_RESTART_SWARM_ENGINE=1
    python -m tutorial.claudecode_geo3k_swarm.test_token_ids

Compare-only (both md files already exist, no swarm needed)::

    python -m tutorial.claudecode_geo3k_swarm.test_token_ids --compare-only
"""

import os
import re
import sys
import json
import time
import argparse
import threading

from ajet.schema.task import Task
from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor

from tutorial.claudecode_geo3k_swarm.geo3k_agent import _execute_agent


IMAGE_PAD_ID = 151655       # Qwen2.5-VL <|image_pad|>
VISION_START_ID = 151652    # <|vision_start|>
VISION_END_ID = 151653      # <|vision_end|>

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOKEN_IO_MD = os.path.join(REPO_ROOT, "token_id_io.md")
DEBUG_MD = os.environ.get("AJET_CAPTURE_TOKEN_IDS", os.path.join(REPO_ROOT, "token_id_io.debug.md"))

AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
REMOTE_MODEL_PATH = os.getenv(
    "REMOTE_MODEL_PATH",
    "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-VL-7B-Instruct",
)
# Small on purpose: one training step should capture quickly.
TEST_BATCH_SIZE = int(os.getenv("TEST_BATCH_SIZE", "8"))
TEST_NUM_REPEAT = int(os.getenv("TEST_NUM_REPEAT", "2"))
TEST_N_GPU = int(os.getenv("TEST_N_GPU", "8"))
MAX_ENV_WORKER = int(os.getenv("MAX_ENV_WORKER", "32"))


# --------------------------- the reproduction task ---------------------------


def _build_case_a_image():
    """256x168 white PNG — same size as token_id_io.md case A (grid 1x12x18,
    54 image-pad tokens)."""
    from PIL import Image
    return Image.new("RGB", (256, 168), "white")


def _build_case_a_task(idx: int) -> Task:
    """A task whose request reproduces token_id_io.md case A exactly.

    The agent (`_execute_agent`) appends the ``\\boxed{}`` suffix, so the
    user text becomes ``"<image>Find y.\\nLet's think step by step and
    output your final answer in \\boxed{}."`` — byte-identical to case A.
    """
    return Task(
        task_id=f"tokentest-{idx}",
        main_query="<image>Find y.",
        metadata={
            "question": "<image>Find y.",
            "image": _build_case_a_image(),
            "ground_truth": "n/a",
        },
    )


# ------------------------------- md parsing ---------------------------------


def _load_case_a_prompt_from_md(path: str):
    """Parse the case-A input (prompt) token id array out of token_id_io.md."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # first json array after the "Case A" heading
    marker = text.find("Case A")
    region = text[marker:] if marker != -1 else text
    m = re.search(r"```json\s*(\[.*?\])\s*```", region, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Could not find a json token array under 'Case A' in {path}")
    return json.loads(m.group(1))


def _load_debug_prompt_from_md(path: str):
    """Parse the training-side prompt token id array out of token_id_io.debug.md."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    marker = text.find("Input (prompt) token ids")
    region = text[marker:] if marker != -1 else text
    m = re.search(r"```json\s*(\[.*?\])\s*```", region, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Could not find the prompt json token array in {path}")
    return json.loads(m.group(1))


# ------------------------------- comparison ---------------------------------


def _pattern_summary(ids):
    """A structural fingerprint of a prompt token array."""
    return {
        "len": len(ids),
        "image_tokens": ids.count(IMAGE_PAD_ID),
        "has_vision_start": VISION_START_ID in ids,
        "has_vision_end": VISION_END_ID in ids,
    }


def compare_prompts(ref, got):
    """Compare training-side prompt (`got`) against vLLM-side case A (`ref`).

    Returns (ok, report_str). Only the prompt pattern matters; response
    tokens are intentionally ignored.
    """
    rs, gs = _pattern_summary(ref), _pattern_summary(got)
    lines = [
        "prompt token pattern comparison (response tokens intentionally ignored)",
        f"  {'field':<16}{'token_id_io.md':>18}{'debug (training)':>20}",
        f"  {'len':<16}{rs['len']:>18}{gs['len']:>20}",
        f"  {'image_tokens':<16}{rs['image_tokens']:>18}{gs['image_tokens']:>20}",
        f"  {'vision_start':<16}{str(rs['has_vision_start']):>18}{str(gs['has_vision_start']):>20}",
        f"  {'vision_end':<16}{str(rs['has_vision_end']):>18}{str(gs['has_vision_end']):>20}",
    ]

    ok = True
    problems = []
    if gs["image_tokens"] == 0:
        ok = False
        problems.append(
            "training prompt has ZERO image tokens — image was dropped before "
            "training (the bug this test guards against)."
        )
    if gs["image_tokens"] != rs["image_tokens"]:
        ok = False
        problems.append(
            f"image token count differs: ref={rs['image_tokens']} got={gs['image_tokens']}."
        )
    if not (gs["has_vision_start"] and gs["has_vision_end"]):
        ok = False
        problems.append("training prompt is missing the vision span markers.")
    if got == ref:
        lines.append("  exact match: YES (training prompt == vLLM-side prompt)")
    else:
        lines.append("  exact match: NO (structural check below decides pass/fail)")
        if gs["len"] != rs["len"]:
            problems.append(f"prompt length differs: ref={rs['len']} got={gs['len']}.")

    if problems:
        lines.append("  ISSUES:")
        lines += [f"    - {p}" for p in problems]
    return ok, "\n".join(lines)


def run_comparison_and_report() -> bool:
    ref = _load_case_a_prompt_from_md(TOKEN_IO_MD)
    got = _load_debug_prompt_from_md(DEBUG_MD)
    ok, report = compare_prompts(ref, got)
    print("=" * 72)
    print(report)
    print("=" * 72)
    print("RESULT:", "PASS ✓ image tokens reach training" if ok else "FAIL ✗")
    return ok


# ------------------------------- swarm driver -------------------------------


def _drive_swarm_until_capture():
    """Submit case-A tasks through the swarm until the trainer captures a batch."""
    assert AJET_SWARM_URL != "http://swarm-server-ip:10086", "Set AJET_SWARM_URL."

    ajet_job = AgentJetJob(
        ensure_new_experiment=True,
        experiment_name="geo3k_tokentest",
        algorithm="grpo",
        logging="tensorboard",
        n_gpu=TEST_N_GPU,
        model=REMOTE_MODEL_PATH,
        batch_size=TEST_BATCH_SIZE,
        num_repeat=TEST_NUM_REPEAT,
        max_env_worker=MAX_ENV_WORKER,
    )

    swarm_worker = SwarmClient(AJET_SWARM_URL)
    swarm_worker.auto_sync_train_config_and_start_engine(
        ajet_job,
        force_restart=os.environ.get("FORCE_RESTART_SWARM_ENGINE", "0") == "1",
    )

    # Remove any stale debug file so we only trust a fresh capture.
    stale_mtime = os.path.getmtime(DEBUG_MD) if os.path.exists(DEBUG_MD) else 0.0

    stop = threading.Event()

    def rollout(task):
        if stop.is_set():
            return
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=240)
        workflow_output = _execute_agent(task, api_baseurl_key)
        swarm_worker.end_episode(task, episode_uuid, workflow_output)

    executor = PeriodicDrainThreadPoolExecutor(
        workers=TEST_NUM_REPEAT * TEST_BATCH_SIZE, auto_retry=True
    )

    print(f"[tokentest] submitting {TEST_BATCH_SIZE} tasks x {TEST_NUM_REPEAT} repeats "
          f"(need {TEST_BATCH_SIZE} finished distinct task_ids to trigger a step)...")
    # Submit a few extra rounds so the batch fills even if some episodes are
    # discarded during a weight-sync boundary.
    for _round in range(3):
        for i in range(TEST_BATCH_SIZE):
            task = _build_case_a_task(i)
            for _ in range(TEST_NUM_REPEAT):
                executor.submit_with_periodic_drain(fn=rollout, task=task)
        # after the first round, wait to see whether the trainer captured
        for _ in range(120):  # up to ~10 min per round
            if os.path.exists(DEBUG_MD) and os.path.getmtime(DEBUG_MD) > stale_mtime:
                stop.set()
                print(f"[tokentest] capture detected at {DEBUG_MD}")
                return True
            time.sleep(5)
    stop.set()
    print("[tokentest] timed out waiting for the trainer to write the capture file.")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip the swarm; just compare token_id_io.md against token_id_io.debug.md.",
    )
    args = parser.parse_args()

    if args.compare_only:
        if not os.path.exists(DEBUG_MD):
            print(f"[tokentest] {DEBUG_MD} not found. Run the swarm path first "
                  f"(export AJET_CAPTURE_TOKEN_IDS on the server).")
            sys.exit(2)
        sys.exit(0 if run_comparison_and_report() else 1)

    captured = _drive_swarm_until_capture()
    if not captured:
        sys.exit(2)
    sys.exit(0 if run_comparison_and_report() else 1)


if __name__ == "__main__":
    main()
