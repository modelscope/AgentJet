# -*- coding: utf-8 -*-
"""
AIME Math Agent Test Script - Using DashScope API

This script tests the reward function using DashScope API (qwen3-max)
instead of the swarm server.

Usage:
    export DASHSCOPE_API_KEY="your-api-key"
    python -m tutorial.opencode_build_aime.agent_roll_test

Environment Variables:
    DASHSCOPE_API_KEY: Your DashScope API key (required)
"""

import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo
from ajet.schema.task import Task
from tutorial.opencode_build_aime.agent_run import compute_reward
from ajet.task_reader import HuggingFaceTaskReader
from ajet.default_config.ajet_config_schema import AjetTaskReader, HuggingfaceDatRepo


# ==================== Configuration ====================

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen3-max"

# Dataset paths
LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOCAL_TRAIN_DATASET = os.path.join(LOCAL_DATA_DIR, "dapo-math-17k.parquet")
LOCAL_TEST_DATASET = os.path.join(LOCAL_DATA_DIR, "aime-2024.parquet")

# Test configuration
NUM_TEST_SAMPLES = 10  # Number of samples to test
MAX_WORKERS = 4  # Parallel workers


def call_dashscope_api(messages: list, enable_thinking: bool = True) -> str:
    """
    Call DashScope API with qwen3-max model.

    Args:
        messages: List of chat messages
        enable_thinking: Whether to enable thinking mode (qwen3 feature)

    Returns:
        Model response content
    """
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DASHSCOPE_MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": 8192,
        "temperature": 0.7,
    }

    # Enable thinking for qwen3 models
    if enable_thinking:
        payload["extra_body"] = {"enable_thinking": True}

    response = requests.post(
        f"{DASHSCOPE_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def test_single_task(task: Task, task_idx: int) -> dict:
    """
    Test a single task and compute reward.

    Args:
        task: Task containing the math problem
        task_idx: Index of the task

    Returns:
        dict with test results
    """
    # Extract prompt messages (DAPO format)
    prompt = task.metadata.get("prompt", [])
    messages = []
    query = ""

    if isinstance(prompt, (list, tuple)) and len(prompt) > 0:
        for msg in prompt:
            if isinstance(msg, dict):
                messages.append(msg)
                if msg.get("role") == "user":
                    query = msg.get("content", "")
    else:
        query = task.main_query if task.main_query not in ["Empty", "[not defined]", ""] else str(prompt)
        messages = [{"role": "user", "content": query}]

    # Extract ground truth (DAPO format: reward_model.ground_truth)
    reward_model = task.metadata.get("reward_model", {})
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth", "")
    else:
        ground_truth = task.metadata.get("ground_truth", task.metadata.get("answer", ""))

    # Call DashScope API
    try:
        model_output = call_dashscope_api(messages)
    except Exception as e:
        return {
            "task_idx": task_idx,
            "task_id": task.task_id,
            "error": str(e),
            "reward": -1.0,
            "correct": False,
        }

    # Compute reward
    reward_result = compute_reward(model_output, ground_truth)

    return {
        "task_idx": task_idx,
        "task_id": task.task_id,
        "query": query[:200] + "..." if len(query) > 200 else query,
        "ground_truth": ground_truth,
        "model_output": model_output[:500] + "..." if len(model_output) > 500 else model_output,
        "predicted": reward_result["pred"],
        "reward": reward_result["score"],
        "correct": reward_result["acc"],
        "extraction_method": reward_result["method"],
    }


def main():
    """Main test function."""

    # Validate API key
    if not DASHSCOPE_API_KEY:
        print("[ERROR] DASHSCOPE_API_KEY environment variable not set!")
        print("Please run: export DASHSCOPE_API_KEY='your-api-key'")
        return

    # Check dataset
    dataset_path = LOCAL_TEST_DATASET if os.path.exists(LOCAL_TEST_DATASET) else LOCAL_TRAIN_DATASET
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print("Please run: proxychains python -m tutorial.opencode_build_aime.download_data")
        return

    print("=" * 70)
    print("AIME Math Agent Test - Using DashScope API")
    print("=" * 70)
    print(f"  API Base URL: {DASHSCOPE_BASE_URL}")
    print(f"  Model:        {DASHSCOPE_MODEL}")
    print(f"  Dataset:      {dataset_path}")
    print(f"  Test samples: {NUM_TEST_SAMPLES}")
    print("=" * 70)

    # Load dataset using HuggingFaceTaskReader
    print(f"\n[INFO] Loading dataset...")
    reader_config = AjetTaskReader(
        huggingface_dat_repo=HuggingfaceDatRepo(
            dataset_path=dataset_path
        )
    )
    dataset = HuggingFaceTaskReader(reader_config)

    # Collect test tasks
    tasks = []
    for idx, task in enumerate(dataset.generate_training_tasks()):
        tasks.append((task, idx))
        if len(tasks) >= NUM_TEST_SAMPLES:
            break

    print(f"[INFO] Loaded {len(tasks)} test samples")

    print(f"\n[INFO] Testing {len(tasks)} samples...\n")

    # Run tests in parallel
    results = []
    correct_count = 0
    total_reward = 0.0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(test_single_task, task, idx): idx
            for task, idx in tasks
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            # Print result
            status = "CORRECT" if result.get("correct") else "WRONG"
            print(f"[{result['task_idx'] + 1}/{len(tasks)}] {status}")
            print(f"  Ground truth: {result.get('ground_truth', 'N/A')}")
            print(f"  Predicted:    {result.get('predicted', 'N/A')}")
            print(f"  Method:       {result.get('extraction_method', 'N/A')}")
            print(f"  Reward:       {result.get('reward', 'N/A')}")

            if result.get("error"):
                print(f"  Error:        {result['error']}")

            print()

            if result.get("correct"):
                correct_count += 1
            total_reward += result.get("reward", 0)

    # Print summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"  Total samples:     {len(results)}")
    print(f"  Correct:           {correct_count}")
    print(f"  Accuracy:          {correct_count / len(results) * 100:.2f}%")
    print(f"  Average reward:    {total_reward / len(results):.4f}")
    print("=" * 70)

    # Print detailed results
    print("\nDetailed Results:")
    for result in sorted(results, key=lambda x: x["task_idx"]):
        status = "+" if result.get("correct") else "-"
        print(f"  [{status}] Task {result['task_idx']}: "
              f"GT={result.get('ground_truth', 'N/A')}, "
              f"Pred={result.get('predicted', 'N/A')}, "
              f"Reward={result.get('reward', 'N/A')}")


if __name__ == "__main__":
    main()
