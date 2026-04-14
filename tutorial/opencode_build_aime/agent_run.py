# -*- coding: utf-8 -*-
"""
AIME Math Agent with DAPO-style reward function.

Reference:
- Reward function: verl/utils/reward_score/math_dapo.py
- Dataset: BytedTsinghua-SIA/DAPO-Math-17k (train), BytedTsinghua-SIA/AIME-2024 (test)
"""

import re
import requests
from textwrap import dedent
from typing import Optional
from ajet.schema.task import Task, WorkflowOutput
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey


# ==================== DAPO-style Reward Functions ====================
# Adapted from verl/utils/reward_score/math_dapo.py

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string."""
    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        return s
    return s[len(left) : -1]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer for comparison."""
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Remove comma in pure numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str,
    gt: str,
    answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> tuple[bool, str]:
    """Check if the solution is correct using Minerva-style extraction."""
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)
    gt = normalize_final_answer(gt)
    return (pred == gt), pred


def is_correct_boxed(solution_str: str, gt: str) -> tuple[bool, str]:
    """Check if the solution is correct using boxed extraction."""
    # Extract from last 300 characters
    solution_str = solution_str[-300:]
    boxed_pred = last_boxed_only_string(solution_str)

    if boxed_pred is None:
        return False, "[NO_BOXED]"

    extracted_pred = remove_boxed(boxed_pred)
    pred = normalize_final_answer(extracted_pred)
    gt = normalize_final_answer(gt)

    return (pred == gt), pred


def compute_reward(solution_str: str, ground_truth: str) -> dict:
    """
    Compute the reward score for a solution using DAPO-style verification.

    Returns:
        dict with keys: score (1.0/-1.0), acc (bool), pred (str), method (str)
    """
    # Limit solution length for efficiency
    solution_str_trimmed = solution_str[-300:]

    # Try boxed extraction first (preferred for math problems)
    correct, pred = is_correct_boxed(solution_str_trimmed, ground_truth)

    if pred == "[NO_BOXED]":
        # Fall back to Minerva-style extraction
        correct, pred = is_correct_minerva(solution_str, ground_truth)
        method = "minerva"
    else:
        method = "boxed"

    reward = 1.0 if correct else 0.0

    return {
        "score": reward,
        "acc": correct,
        "pred": pred,
        "method": method,
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

    # Extract problem and ground truth from task
    # DAPO-Math-17k uses:
    #   - "prompt" column: list of messages in chat format
    #   - "reward_model.ground_truth" or "ground_truth" for answer
    # HuggingFaceTaskReader maps columns to metadata

    # Try to get query from different possible sources
    query = task.main_query
    if query in ["Empty", "[not defined]", ""] or not query:
        # Try to extract from prompt column (DAPO format)
        prompt = task.metadata.get("prompt", [])
        if isinstance(prompt, list) and len(prompt) > 0:
            # Find user message in the chat format
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    query = msg.get("content", "")
                    break
            if not query and len(prompt) > 0:
                # Fallback: use last message's content
                last_msg = prompt[-1]
                if isinstance(last_msg, dict):
                    query = last_msg.get("content", "")
                elif isinstance(last_msg, str):
                    query = last_msg
        elif isinstance(prompt, str):
            query = prompt

    # Try to get ground truth from different possible sources
    ground_truth = task.metadata.get("ground_truth", "")
    if not ground_truth:
        ground_truth = task.metadata.get("answer", "")
    if not ground_truth:
        # DAPO format: reward_model.ground_truth
        reward_model = task.metadata.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth", "")

    # Prepare messages
    # If dataset has original prompt messages, use them directly
    prompt = task.metadata.get("prompt", [])
    if isinstance(prompt, list) and len(prompt) > 0:
        # Use original dataset prompt format (supports "Answer:" format)
        messages = []
        for msg in prompt:
            if isinstance(msg, dict):
                messages.append(msg)
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    else:
        # Fallback to custom format (supports \boxed{} format)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]

    # Call the model
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
        # Return negative reward on error
        return WorkflowOutput(
            reward=-1.0,
            metadata={
                "error": str(e),
                "ground_truth": ground_truth,
            }
        )

    # Compute reward using DAPO-style verification
    reward_result = compute_reward(model_output, ground_truth)

    return WorkflowOutput(
        reward=reward_result["score"],
        metadata={
            "model_output": model_output,
            "ground_truth": ground_truth,
            "predicted": reward_result["pred"],
            "correct": reward_result["acc"],
            "extraction_method": reward_result["method"],
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
    # Simple test
    print("Testing reward computation...")

    # Test boxed extraction
    test_solution = "The answer is \\boxed{42}."
    result = compute_reward(test_solution, "42")
    print(f"Test 1 (boxed): {result}")
    assert result["acc"] == True

    # Test Minerva extraction
    test_solution2 = "After calculation, Answer: 42"
    result2 = compute_reward(test_solution2, "42")
    print(f"Test 2 (minerva): {result2}")
    assert result2["acc"] == True

    # Test normalization
    test_solution3 = "\\boxed{1,000}"
    result3 = compute_reward(test_solution3, "1000")
    print(f"Test 3 (normalization): {result3}")
    assert result3["acc"] == True

    print("All tests passed!")
