# -*- coding: utf-8 -*-
"""
End-to-end token-id check for the multi-message multimodal fix.

Extends the single-image ``test_token_ids.py`` to four cases (see
``multimodal_cases.py``):

  1. two_images_one_msg      — two images in ONE user turn (positive control;
                               passes even before the fix).
  2. image_text_image_turns  — images interleaved across turns (needs the fix).
  3. pure_text               — zero images.
  4. img_text_img_text_tool  — image / tool_call / tool text / image (needs fix).

It proves each case's images survive the whole swarm pipeline into the
*training* tokenization as real ``<|image_pad|>`` (151655) tokens by comparing:

  * ``token_id/<case>.md``       — the vLLM-side ground truth written by
                                   ``gen_ground_truth.py``.
  * ``token_id/<case>.debug.md`` — the training-side per-case capture written by
                                   ``capture_training_token_ids`` when
                                   ``AJET_CAPTURE_TOKEN_IDS`` names the
                                   ``token_id/`` *directory*.

Only the *prompt* token pattern is compared (length, image-token count, vision
spans); the model's answer may differ between runs.

Run
---
Server tmux::

    source .venv/bin/activate
    export SETUPTOOLS_USE_DISTUTILS=local
    export AJET_CAPTURE_TOKEN_IDS=$PWD/token_id      # a DIRECTORY -> per-case files
    ajet-swarm start --swarm-port=10086

Client tmux::

    source .venv/bin/activate
    export AJET_SWARM_URL=http://localhost:10086
    export REMOTE_MODEL_PATH=/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-VL-7B-Instruct
    export FORCE_RESTART_SWARM_ENGINE=1
    python -m tutorial.claudecode_geo3k_swarm.test_multimodal_cases

Compare-only (both md sets already exist)::

    python -m tutorial.claudecode_geo3k_swarm.test_multimodal_cases --compare-only
"""

import os
import sys
import time
import argparse
import threading

from ajet.copilot.job import AgentJetJob
from ajet.tuner_lib.experimental.swarm_client import SwarmClient
from ajet.utils.thread_executors import PeriodicDrainThreadPoolExecutor

from tutorial.claudecode_geo3k_swarm.multimodal_cases import (
    CASE_NAMES,
    EXPECTED_IMAGE_TOKENS,
    build_case_task,
    _execute_multimodal_case,
)
# reuse the md-parsing + pattern helpers from the single-image test
from tutorial.claudecode_geo3k_swarm.test_token_ids import (
    IMAGE_PAD_ID,
    VISION_START_ID,
    VISION_END_ID,
    _load_case_a_prompt_from_md,
    _load_debug_prompt_from_md,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOKEN_DIR = os.environ.get("AJET_CAPTURE_TOKEN_IDS", os.path.join(REPO_ROOT, "token_id"))

AJET_SWARM_URL = os.getenv("AJET_SWARM_URL", "http://localhost:10086")
REMOTE_MODEL_PATH = os.getenv(
    "REMOTE_MODEL_PATH",
    "/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen/Qwen2___5-VL-7B-Instruct",
)
TEST_BATCH_SIZE = int(os.getenv("TEST_BATCH_SIZE", str(len(CASE_NAMES))))
TEST_NUM_REPEAT = int(os.getenv("TEST_NUM_REPEAT", "2"))
TEST_N_GPU = int(os.getenv("TEST_N_GPU", "8"))
MAX_ENV_WORKER = int(os.getenv("MAX_ENV_WORKER", "32"))


# ------------------------------- comparison ---------------------------------


def _pattern_summary(ids):
    return {
        "len": len(ids),
        "image_tokens": ids.count(IMAGE_PAD_ID),
        "vision_starts": ids.count(VISION_START_ID),
        "vision_ends": ids.count(VISION_END_ID),
    }


def compare_case(case: str) -> tuple:
    """Compare training-side (<case>.debug.md) vs vLLM ground truth (<case>.md).

    Returns (ok, report_str). Only the prompt pattern matters.
    """
    ref_md = os.path.join(TOKEN_DIR, f"{case}.md")
    dbg_md = os.path.join(TOKEN_DIR, f"{case}.debug.md")
    if not os.path.exists(ref_md):
        return False, f"[{case}] missing ground truth {ref_md} (run gen_ground_truth.py)"
    if not os.path.exists(dbg_md):
        return False, f"[{case}] missing training capture {dbg_md} (run the swarm path)"

    ref = _load_case_a_prompt_from_md(ref_md)   # ground-truth uses the "Case A" heading
    got = _load_debug_prompt_from_md(dbg_md)
    rs, gs = _pattern_summary(ref), _pattern_summary(got)
    exp_img = EXPECTED_IMAGE_TOKENS[case]

    problems = []
    if gs["image_tokens"] != exp_img:
        problems.append(f"training image_tokens={gs['image_tokens']} != expected {exp_img}")
    if gs["image_tokens"] != rs["image_tokens"]:
        problems.append(f"image_tokens differ ref={rs['image_tokens']} got={gs['image_tokens']}")
    if gs["len"] != rs["len"]:
        problems.append(f"prompt length differs ref={rs['len']} got={gs['len']}")
    if exp_img > 0:
        # each image contributes one vision span; expect >=1 and matching ref
        if gs["vision_starts"] != rs["vision_starts"] or gs["vision_ends"] != rs["vision_ends"]:
            problems.append(
                f"vision span count differs ref=({rs['vision_starts']},{rs['vision_ends']}) "
                f"got=({gs['vision_starts']},{gs['vision_ends']})"
            )

    ok = not problems
    report = [
        f"[{case}] {'PASS' if ok else 'FAIL'}",
        f"    {'field':<14}{'ground_truth':>14}{'training':>12}",
        f"    {'len':<14}{rs['len']:>14}{gs['len']:>12}",
        f"    {'image_tokens':<14}{rs['image_tokens']:>14}{gs['image_tokens']:>12}  (exp {exp_img})",
        f"    {'vision_start':<14}{rs['vision_starts']:>14}{gs['vision_starts']:>12}",
        f"    {'vision_end':<14}{rs['vision_ends']:>14}{gs['vision_ends']:>12}",
        f"    exact_match: {'YES' if ref == got else 'NO (pattern check decides)'}",
    ]
    for p in problems:
        report.append(f"    ISSUE: {p}")
    return ok, "\n".join(report)


def run_comparison_and_report() -> bool:
    all_ok = True
    print("=" * 72)
    for case in CASE_NAMES:
        ok, report = compare_case(case)
        print(report)
        print("-" * 72)
        all_ok = all_ok and ok
    print("RESULT:", "PASS ✓ all cases' images reach training" if all_ok else "FAIL ✗")
    return all_ok


# ------------------------------- swarm driver -------------------------------


def _newest_mtime() -> float:
    if not os.path.isdir(TOKEN_DIR):
        return 0.0
    mts = [
        os.path.getmtime(os.path.join(TOKEN_DIR, f))
        for f in os.listdir(TOKEN_DIR)
        if f.endswith(".debug.md")
    ]
    return max(mts) if mts else 0.0


def _all_debug_present() -> bool:
    return all(
        os.path.exists(os.path.join(TOKEN_DIR, f"{c}.debug.md")) for c in CASE_NAMES
    )


def _drive_swarm_until_capture() -> bool:
    assert AJET_SWARM_URL != "http://swarm-server-ip:10086", "Set AJET_SWARM_URL."

    ajet_job = AgentJetJob(
        ensure_new_experiment=True,
        experiment_name="geo3k_mm_tokentest",
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

    stale_mtime = _newest_mtime()
    stop = threading.Event()

    def rollout(task):
        if stop.is_set():
            return
        episode_uuid, api_baseurl_key = swarm_worker.begin_episode(discard_episode_timeout=240)
        workflow_output = _execute_multimodal_case(task, api_baseurl_key)
        swarm_worker.end_episode(task, episode_uuid, workflow_output)

    executor = PeriodicDrainThreadPoolExecutor(
        workers=TEST_NUM_REPEAT * TEST_BATCH_SIZE, auto_retry=True
    )

    print(f"[mmtest] submitting {len(CASE_NAMES)} cases x {TEST_NUM_REPEAT} repeats "
          f"(need {TEST_BATCH_SIZE} distinct task_ids to trigger a step)...")
    for _round in range(3):
        for case in CASE_NAMES:
            task = build_case_task(case)
            for _ in range(TEST_NUM_REPEAT):
                executor.submit_with_periodic_drain(fn=rollout, task=task)
        for _ in range(120):  # up to ~10 min per round
            if _all_debug_present() and _newest_mtime() > stale_mtime:
                stop.set()
                print(f"[mmtest] capture detected in {TOKEN_DIR}")
                return True
            time.sleep(5)
    stop.set()
    print("[mmtest] timed out waiting for the trainer to write per-case capture files.")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip the swarm; just compare token_id/<case>.md vs token_id/<case>.debug.md.",
    )
    args = parser.parse_args()

    if args.compare_only:
        sys.exit(0 if run_comparison_and_report() else 1)

    captured = _drive_swarm_until_capture()
    if not captured:
        sys.exit(2)
    sys.exit(0 if run_comparison_and_report() else 1)


if __name__ == "__main__":
    main()
