# -*- coding: utf-8 -*-
"""
AIME Math Agent with rStar2-style reward function.

Reference:
- Reward function: rstar2_agent/reward/compute_score.py
- Dataset: BytedTsinghua-SIA/DAPO-Math-17k (train), BytedTsinghua-SIA/AIME-2024 (test)
"""

import requests
from textwrap import dedent
from ajet.schema.task import Task, WorkflowOutput
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey


# ==================== rStar2-style Reward Functions ====================
# Adapted from rstar2_agent/reward/compute_score.py
from tutorial.opencode_build_aime.verl_reward_fn.prime_math import compute_score as prime_compute_score
from tutorial.opencode_build_aime.verl_reward_fn.math_verify import compute_score as math_verify_compute_score


def compute_score(model_output: str, ground_truth: str) -> float:
    """
    Compute reward score for a solution using rStar2-style verification.
    Uses prime_math compute_score first, falls back to math_verify.

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    try:
        prime_score = prime_compute_score(model_output, ground_truth)[0]
        if prime_score:
            return 1.0
    except Exception:
        prime_score = 0.0
    try:
        math_verify_score = math_verify_compute_score(model_output, ground_truth)
        if math_verify_score:
            return 1.0
    except Exception:
        return 0.0
    return 0.0


def compute_reward(solution_str: str, ground_truth: str) -> dict:
    """
    Compute the reward score for a solution.

    Returns:
        dict with keys: score (1.0/0.0), acc (bool), pred (str)
    """
    score = compute_score(solution_str, ground_truth)
    correct = score == 1.0

    return {
        "score": score,
        "acc": correct,
        "pred": "",
    }


# ==================== Agent Execution ====================

SYSTEM_PROMPT = dedent("""
You are an expert mathematician specialized in solving challenging math competition problems.

Instructions:
1. Think through the problem step by step
2. Show your reasoning clearly
3. Put your final numerical answer inside \\boxed{} at the end

For example, if the answer is 42, write: \\boxed{42}
""").strip()


def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey) -> WorkflowOutput:
    """
    Execute the math agent and compute reward.

    Args:
        task: Task containing the math problem
        api_baseurl_key: OpenAI-compatible API credentials

    Returns:
        WorkflowOutput with reward and metadata
    """
    base_url = api_baseurl_key.base_url
    api_key = api_baseurl_key.api_key

    query = task.main_query
    if query in ["Empty", "[not defined]", ""] or not query:
        prompt = task.metadata.get("prompt", [])
        if isinstance(prompt, list) and len(prompt) > 0:
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    query = msg.get("content", "")
                    break
            if not query and len(prompt) > 0:
                last_msg = prompt[-1]
                if isinstance(last_msg, dict):
                    query = last_msg.get("content", "")
                elif isinstance(last_msg, str):
                    query = last_msg
        elif isinstance(prompt, str):
            query = prompt

    ground_truth = task.metadata.get("ground_truth", "")
    if not ground_truth:
        ground_truth = task.metadata.get("answer", "")
    if not ground_truth:
        reward_model = task.metadata.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth", "")

    prompt = task.metadata.get("prompt", [])
    if isinstance(prompt, list) and len(prompt) > 0:
        messages = []
        for msg in prompt:
            if isinstance(msg, dict):
                messages.append(msg)
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": "fill_whatever_model",
                "messages": messages,
                "stream": False,
                "temperature": 1.0,
                "max_tokens": 8192,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Connection": "close",
            },
            timeout=300,
        )
        response.raise_for_status()
        model_output = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return WorkflowOutput(
            reward=0.0,
            metadata={
                "error": str(e),
                "ground_truth": ground_truth,
            }
        )

    reward_result = compute_reward(model_output, ground_truth)

    return WorkflowOutput(
        reward=reward_result["score"],
        metadata={
            "model_output": model_output,
            "ground_truth": ground_truth,
            "predicted": reward_result["pred"],
            "correct": reward_result["acc"],
        }
    )


def run_agent_and_compute_reward(
    task: Task,
    base_url: str,
    api_key: str
) -> WorkflowOutput:
    """
    Convenience wrapper for agent execution.

    Args:
        task: Task containing the math problem
        base_url: OpenAI-compatible API base URL
        api_key: API key

    Returns:
        WorkflowOutput with reward and metadata
    """
    api_baseurl_key = OpenaiBaseUrlAndApiKey(
        base_url=base_url,
        api_key=api_key,
    )
    return execute_agent(task, api_baseurl_key)


if __name__ == "__main__":
    print("Testing reward computation...")
    print("Note: Requires verl submodule to be initialized")
