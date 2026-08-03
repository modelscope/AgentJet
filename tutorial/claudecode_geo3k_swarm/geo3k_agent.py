# -*- coding: utf-8 -*-
"""
Geo3K multimodal agent (backend-agnostic).

This module holds the pure agent logic: build a multimodal chat request
from a geo3k task, call an OpenAI-compatible chat/completions endpoint,
and score the ``\\boxed{}`` answer against the reference.

It has no dependency on the swarm runtime, so it can be tested directly
against any OpenAI-compatible vision endpoint (DashScope or a local vLLM
serving a VL model such as Qwen2.5-VL):

    # DashScope (Aliyun compatible-mode)
    export DASHSCOPE_API_KEY=sk-xxxx
    python -m tutorial.claudecode_geo3k_swarm.geo3k_agent \\
        --backend dashscope --model qwen-vl-max-latest

    # local vLLM OpenAI server
    python -m tutorial.claudecode_geo3k_swarm.geo3k_agent \\
        --backend vllm --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen2.5-VL-7B-Instruct

If ``GEO3K_DATASET_PATH`` points at a real geo3k parquet the harness uses
the first sample from it; otherwise it falls back to a synthetic triangle
task so the multimodal path can still be exercised end to end.
"""

import os
import re
from textwrap import dedent
from typing import Optional

import requests

from ajet.schema.task import Task, WorkflowOutput
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.utils.multimodal import build_multimodal_messages, extract_image


SYSTEM_PROMPT = dedent(
    """A conversation between the User and Assistant. The User asks a question, and the Assistant provides a solution. The Assistant first thinks through the reasoning process internally with self-reflection and consistency check and then gives the final analysis and answer. The reasoning process should be enclosed within <think></think>, followed directly by the final thought and answer, the final answer MUST BE put in \\boxed{}, like this: <think> reasoning process here </think> final thought and \\boxed{answer} here."""
)

ANSWER_RE = re.compile(r"\\boxed\{([^}]*)\}")

# Placeholder model name for the swarm path, where the served model is
# chosen server-side and the value here is ignored.
DEFAULT_MODEL = "fill_whatever_model"


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip().lower()


def _compute_reward(final_answer: str, reference_answer: str) -> float:
    """Extract \\boxed{} answer and compare to ground truth string."""
    m = ANSWER_RE.search(final_answer or "")
    if not m:
        return 0.0
    predicted = _normalize(m.group(1))
    target = _normalize(reference_answer)
    return 1.0 if predicted == target else 0.0


def _execute_agent(
    task: Task,
    api_baseurl_key: OpenaiBaseUrlAndApiKey,
    model: str = DEFAULT_MODEL,
) -> WorkflowOutput:
    base_url, api_key = api_baseurl_key.base_url, api_baseurl_key.api_key

    # Geo3k row layout (from rllm preprocess):
    #   question: str, image/images: list[PIL.Image] or dict with bytes,
    #   answer / ground_truth: str
    meta = task.metadata
    question = meta.get("question") or meta.get("problem") or task.main_query
    if "\\boxed" not in question:
        question = question + "\nLet's think step by step and output your final answer in \\boxed{}."

    reference_answer = meta.get("ground_truth") or meta.get("answer") or ""

    messages = build_multimodal_messages(
        system_prompt=SYSTEM_PROMPT,
        user_text=question,
        image=extract_image(meta),
    )

    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": 2048,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Connection": "close",
        },
        timeout=1260,  # 21 min, must exceed server ZMQ budget (20 min) so it doesn't cut in first
    )
    response.raise_for_status()
    final_answer = response.json()["choices"][0]["message"]["content"]

    raw_reward = _compute_reward(final_answer, reference_answer)
    return WorkflowOutput(
        reward=raw_reward,
        metadata={"final_answer": final_answer, "reference": reference_answer},
    )


def run_agent_and_compute_reward(
    task: Task,
    base_url: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> WorkflowOutput:
    return _execute_agent(
        task, OpenaiBaseUrlAndApiKey(base_url=base_url, api_key=api_key), model=model
    )


# ============================ standalone test ============================


def _load_geo3k_task(dataset_path: str) -> Optional[Task]:
    """Return the first task from a geo3k parquet, or None if unavailable."""
    if not dataset_path or not os.path.exists(dataset_path):
        return None
    from ajet.task_reader import HuggingFaceTaskReader
    from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo

    reader = HuggingFaceTaskReader(
        AjetTaskReader(huggingface_dat_repo=HuggingfaceDatRepo(dataset_path=dataset_path))
    )
    for t in reader.generate_training_tasks():
        return t
    return None


def _build_synthetic_task() -> Task:
    """A tiny multimodal task with a generated right-triangle figure.

    Used when no geo3k dataset is provided so the image -> request ->
    scoring path can still be exercised against a live VL endpoint.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(img)
    d.line([(40, 210), (210, 210)], fill="black", width=3)  # base
    d.line([(40, 210), (40, 40)], fill="black", width=3)     # height
    d.line([(40, 40), (210, 210)], fill="black", width=3)    # hypotenuse
    d.text((115, 214), "6", fill="black")
    d.text((18, 118), "8", fill="black")

    question = "A right triangle has legs of length 6 and 8. What is the length of the hypotenuse?"
    return Task(
        main_query=question,
        metadata={
            "question": question,
            "image": img,
            "ground_truth": "10",
        },
    )


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone test for the geo3k multimodal agent against "
        "DashScope or a vLLM OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--backend",
        choices=["dashscope", "vllm"],
        default=os.getenv("GEO3K_TEST_BACKEND", "dashscope"),
    )
    parser.add_argument("--base-url", default=os.getenv("GEO3K_TEST_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("GEO3K_TEST_API_KEY"))
    parser.add_argument("--model", default=os.getenv("GEO3K_TEST_MODEL"))
    parser.add_argument(
        "--dataset",
        default=os.getenv("GEO3K_DATASET_PATH", ""),
        help="Optional geo3k parquet; if omitted a synthetic task is used.",
    )
    args = parser.parse_args()

    if args.backend == "dashscope":
        base_url = args.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
        model = args.model or "qwen-vl-max-latest"
        assert api_key, "Set DASHSCOPE_API_KEY (or pass --api-key) for the dashscope backend."
    else:  # vllm
        base_url = args.base_url or os.getenv("VLLM_BASE_URL") or "http://localhost:8000/v1"
        api_key = args.api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        model = args.model
        assert model, "Pass --model (or GEO3K_TEST_MODEL) matching your served vLLM model."

    task = _load_geo3k_task(args.dataset)
    if task is None:
        print("[test] no geo3k dataset found; using synthetic right-triangle task.")
        task = _build_synthetic_task()
    else:
        print(f"[test] loaded first task from geo3k dataset: {args.dataset}")

    print(f"[test] backend={args.backend} base_url={base_url} model={model}")
    out = run_agent_and_compute_reward(task, base_url=base_url, api_key=api_key, model=model)

    print("=" * 60)
    print("final_answer:\n", out.metadata.get("final_answer"))
    print("-" * 60)
    print("reference:", out.metadata.get("reference"))
    print("reward:", out.reward)
    print("=" * 60)


if __name__ == "__main__":
    _main()
