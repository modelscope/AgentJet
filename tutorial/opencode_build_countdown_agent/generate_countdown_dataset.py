# ------- AI GENERATED --------
# ------- [Read tutorial/opencode_build_countdown_agent.prompt.md] --------

"""
Generate mock CountDown number puzzle dataset for testing.

CountDown is a number puzzle game where you need to reach a target number
using basic arithmetic operations (+, -, *, /) on a list of source numbers.
Each source number can only be used once.
"""

import json
import random
import os
from typing import List, Tuple


def generate_simple_countdown_problem() -> Tuple[int, List[int], str]:
    """
    Generate a simple CountDown problem with a known solution.

    Returns:
        tuple: (target_number, source_numbers, solution_expression)
    """
    # Generate source numbers (typically 6 numbers: mix of small and large)
    small_numbers = random.sample(
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10], 4
    )
    large_numbers = random.sample([25, 50, 75, 100], 2)
    source_numbers = small_numbers + large_numbers
    random.shuffle(source_numbers)

    # Create a simple solution by combining random operations
    # Pick 2-4 numbers to construct a target
    num_ops = random.randint(2, 4)
    selected = random.sample(source_numbers, num_ops)

    # Build a simple expression and calculate target
    if num_ops == 2:
        op = random.choice(["+", "-", "*"])
        if op == "+":
            target = selected[0] + selected[1]
            solution = f"{selected[0]} + {selected[1]}"
        elif op == "-":
            target = abs(selected[0] - selected[1])
            solution = f"{max(selected)} - {min(selected)}"
        else:  # '*'
            target = selected[0] * selected[1]
            solution = f"{selected[0]} * {selected[1]}"
    elif num_ops == 3:
        op1, op2 = random.sample(["+", "-", "*"], 2)
        # (a op1 b) op2 c
        if op1 == "+":
            temp = selected[0] + selected[1]
            expr1 = f"({selected[0]} + {selected[1]})"
        elif op1 == "-":
            temp = abs(selected[0] - selected[1])
            expr1 = (
                f"({max(selected[0], selected[1])} - {min(selected[0], selected[1])})"
            )
        else:
            temp = selected[0] * selected[1]
            expr1 = f"({selected[0]} * {selected[1]})"

        if op2 == "+":
            target = temp + selected[2]
            solution = f"{expr1} + {selected[2]}"
        elif op2 == "-":
            target = abs(temp - selected[2])
            solution = f"{max(temp, selected[2])} - {min(temp, selected[2])}".replace(
                str(temp), expr1
            )
        else:
            target = temp * selected[2]
            solution = f"{expr1} * {selected[2]}"
    else:  # num_ops == 4
        # ((a + b) * c) + d
        temp1 = selected[0] + selected[1]
        temp2 = temp1 * selected[2]
        target = temp2 + selected[3]
        solution = f"(({selected[0]} + {selected[1]}) * {selected[2]}) + {selected[3]}"

    # Make sure target is positive and reasonable
    target = abs(target)
    if target == 0:
        target = sum(selected[:2])
        solution = f"{selected[0]} + {selected[1]}"
    if target > 1000:
        target = target % 500 + 100

    return target, source_numbers, solution


def generate_dataset(num_samples: int = 50) -> List[dict]:
    """
    Generate a dataset of CountDown problems.

    Args:
        num_samples: Number of samples to generate

    Returns:
        List of dataset items
    """
    dataset = []

    for i in range(num_samples):
        target, sources, solution = generate_simple_countdown_problem()

        # Format as AgentJet Task structure
        item = {
            "task_id": f"countdown_{i:04d}",
            "main_query": f"Target: {target}, Numbers: {sources}",
            "metadata": {
                "target": target,
                "sources": sources,
                "reference_solution": solution,
                "description": f"Find a way to reach {target} using these numbers: {sources}. You can use +, -, *, / operations. Each number can only be used once.",
            },
        }
        dataset.append(item)

    return dataset


def save_dataset(dataset: List[dict], output_dir: str = "./countdown_dataset"):
    """
    Save dataset to disk in AgentJet compatible format.

    Args:
        dataset: List of dataset items
        output_dir: Directory to save the dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON lines format
    output_file = os.path.join(output_dir, "train.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Dataset saved to {output_file}")
    print(f"Total samples: {len(dataset)}")

    # Save a few examples for inspection
    examples_file = os.path.join(output_dir, "examples.json")
    with open(examples_file, "w", encoding="utf-8") as f:
        json.dump(dataset[:5], f, indent=2, ensure_ascii=False)

    print(f"Examples saved to {examples_file}")


if __name__ == "__main__":
    # Generate 50 training samples
    print("Generating CountDown dataset...")
    dataset = generate_dataset(num_samples=50)

    # Save to disk
    save_dataset(dataset, output_dir="./countdown_dataset")

    # Print a few examples
    print("\n" + "=" * 80)
    print("Sample Problems:")
    print("=" * 80)
    for i, item in enumerate(dataset[:3]):
        print(f"\nProblem {i + 1}:")
        print(f"  Query: {item['main_query']}")
        print(f"  Reference Solution: {item['metadata']['reference_solution']}")
