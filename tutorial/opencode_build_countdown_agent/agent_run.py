# ------- AI GENERATED --------
# ------- [Read tutorial/opencode_build_countdown_agent.prompt.md] --------

"""
CountDown Number Puzzle Agent

This agent solves countdown number puzzles by finding arithmetic expressions
that reach a target number using a given list of source numbers.
"""

import re
from typing import List, Tuple
from openai import OpenAI

from ajet import WorkflowOutput
from ajet.schema.task import Task
from ajet.utils.retry import retry_with_backoff


def parse_and_verify_solution(
    solution: str, target: int, sources: List[int]
) -> Tuple[bool, float, str]:
    """
    Parse the solution and verify if it's correct.

    Args:
        solution: The solution string from the agent
        target: The target number to reach
        sources: List of source numbers available

    Returns:
        tuple: (is_valid, computed_value, error_message)
    """
    # Extract expression from <answer> tags if present
    match = re.search(r"<answer>(.*?)</answer>", solution, re.DOTALL)
    if match:
        expression = match.group(1).strip()
    else:
        # Try to find any mathematical expression in the solution
        lines = solution.strip().split("\n")
        expression = lines[-1].strip()

    try:
        # Extract all numbers used in the expression
        used_numbers_str = re.findall(r"\b\d+\.?\d*\b", expression)
        used_numbers = [float(n) if "." in n else int(n) for n in used_numbers_str]

        # Check if only source numbers are used (and used only once)
        sources_copy: List[int] = sources.copy()
        for num in used_numbers:
            # Convert to int for comparison if it's a whole number
            check_num = int(num) if isinstance(num, float) and num == int(num) else num
            if isinstance(check_num, int) and check_num in sources_copy:
                sources_copy.remove(check_num)
            else:
                return (
                    False,
                    0,
                    f"Invalid: Number {num} not available or used multiple times",
                )

        # Evaluate the expression safely
        # Only allow basic arithmetic operations
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return False, 0, "Invalid: Expression contains disallowed characters"

        # Evaluate the expression
        result = eval(expression)

        # Check if result matches target (allow small floating point error)
        if abs(result - target) < 0.001:
            return True, result, "Correct!"
        else:
            return (
                False,
                result,
                f"Incorrect: Expression evaluates to {result}, but target is {target}",
            )

    except Exception as e:
        return False, 0, f"Error evaluating expression: {str(e)}"


def _compute_reward(solution: str, target: int, sources: List[int]) -> float:
    """
    Compute reward for the solution.

    Args:
        solution: The solution from the agent
        target: Target number
        sources: Available source numbers

    Returns:
        Reward score (1.0 for correct, 0.0 for incorrect, partial for close)
    """
    is_valid, result, error_msg = parse_and_verify_solution(solution, target, sources)

    if is_valid:
        return 1.0

    # Give partial reward if the result is close to target
    if result > 0:
        diff = abs(result - target)
        if diff < 10:
            # Very close: 0.5 reward
            return 0.5
        elif diff < 50:
            # Somewhat close: 0.2 reward
            return 0.2

    return 0.0


def _execute_agent(task: Task, base_url: str, api_key: str) -> Tuple[str, float]:
    """
    Execute the CountDown agent.

    Args:
        task: Task containing the countdown problem
        base_url: OpenAI API base URL
        api_key: OpenAI API key

    Returns:
        tuple: (solution_text, reward)
    """
    # Extract problem from task
    target = task.metadata["target"]
    sources = task.metadata["sources"]
    description = task.metadata["description"]

    # Prepare system prompt
    system_prompt = """You are an expert at solving CountDown number puzzles.

Given a target number and a list of source numbers, you need to find an arithmetic expression using +, -, *, / operations that reaches the target number.

Rules:
1. Each source number can only be used once
2. You can use +, -, *, / operations
3. You can use parentheses to control order of operations
4. Intermediate results can be any number (including fractions and negatives)
5. The final result must equal the target number

Please think step by step:
1. Analyze the target and available numbers
2. Try different combinations of operations
3. Verify your solution is correct
4. Output your final answer in the format: <answer>expression</answer>

Example:
Target: 100, Numbers: [25, 3, 6, 2]
Solution: <answer>(25 + 3) * 2 + 6</answer>  (This gives: 28 * 2 + 6 = 56 + 6 = 62, which is wrong)
Better solution: <answer>25 * 6 - 3 * 2</answer> (This gives: 150 - 6 = 144, still wrong)
Correct solution: <answer>(25 - 3) * 6 - 2</answer> (This gives: 22 * 6 - 2 = 132 - 2 = 130, still not 100)

Let me try: <answer>25 * (6 - 2) - 3</answer> (This gives: 25 * 4 - 3 = 100 - 3 = 97, close!)
Or: <answer>25 * (6 - 2)</answer> (This gives: 25 * 4 = 100, perfect!)
"""

    # Prepare user query
    user_query = f"{description}\n\nTarget: {target}\nAvailable numbers: {sources}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    # Call the model
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model="agentjet-model",  # This will be the model being trained
        messages=messages,  # type: ignore
        temperature=0.7,  # Some randomness for exploration
    )

    solution = response.choices[0].message.content or ""

    # Compute reward
    reward = _compute_reward(solution, target, sources)

    return solution, reward


@retry_with_backoff(max_retry=3)
def run_agent_and_compute_reward(
    task: Task, base_url: str, api_key: str
) -> WorkflowOutput:
    """
    Main entry point for running the agent and computing reward.

    Args:
        task: The countdown problem task
        base_url: OpenAI API base URL
        api_key: OpenAI API key

    Returns:
        WorkflowOutput containing reward and metadata
    """
    # Execute the agent
    solution, reward = _execute_agent(task, base_url, api_key)

    # Verify the solution for logging
    target = task.metadata["target"]
    sources = task.metadata["sources"]
    is_valid, result, error_msg = parse_and_verify_solution(solution, target, sources)

    # Print results for debugging
    print(f"\n{'=' * 60}")
    print(f"Target: {target}, Numbers: {sources}")
    print(f"Solution: {solution[:200]}...")
    print(f"Valid: {is_valid}, Result: {result}, Message: {error_msg}")
    print(f"Reward: {reward}")
    print(f"{'=' * 60}\n")

    return WorkflowOutput(
        reward=reward,
        metadata={
            "solution": solution,
            "is_valid": is_valid,
            "computed_result": result,
            "verification_message": error_msg,
            "target": target,
            "sources": sources,
        },
    )
