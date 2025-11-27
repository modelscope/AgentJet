import json
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from astune.schema.task import Task, WorkflowOutput, WorkflowTask
from astune.task_judge.rm_auto_grader_judge import RMAutoGraderJudge
from astune.task_reader.tracing_reader import TracingReader
from astune.utils.config_utils import read_astune_config

# @pytest.fixture
# def config(tmp_path: Path) -> SimpleNamespace:
#     a = SimpleNamespace()
#     a.astune = SimpleNamespace()
#     a.astune.tracing = {
#         "base_url": "./.trash/database.sqlite",
#         "train_output_path": str(tmp_path / "tasks.jsonl"),
#         "filters": [],
#     }
#     return a


# @pytest.fixture
# def config_with_filter(tmp_path: Path) -> SimpleNamespace:
#     a = SimpleNamespace()
#     a.astune = SimpleNamespace()
#     a.astune.tracing = {
#         "base_url": "./.trash/database.sqlite",
#         "train_output_path": str(tmp_path / "tasks.jsonl"),
#         "filters": [
#             {
#                 "type": "llm_evaluate",
#                 "enabled": True,
#                 "params": {
#                     "custom_rubrics": "If the answer claims that it has written the output to a file, consider it an invalid response.",
#                     "temperature": 0.5,
#                     "max_tokens": 2048,
#                     "print_reason": False,
#                 },
#             }
#         ],
#     }
#     return a


def create_math_reference_samples(num_samples: int = 10) -> List[WorkflowTask]:
    """
    Create reference math problem samples for Pointwise rubric generation.

    Each sample contains a single answer with a score label.
    """
    samples = []

    # Simple math problems with answers and scores
    # Format: (query, answer, score)
    problems = [
        ("What is 15 + 27?", "42", 1),
        ("Calculate 8 * 9", "72", 1),
        ("What is 100 - 37?", "63", 1),
        ("Find the value of 144 / 12", "12", 1),
        ("What is 5^3?", "125", 1),
        ("Calculate 23 + 45 - 18", "50", 1),
        ("What is 7 * 8 + 6?", "62", 1),
        ("Find the value of (15 + 5) * 2", "40", 1),
        ("What is 99 - 33 - 22?", "44", 1),
        ("Calculate 16 / 4 + 10", "14", 1),
    ]

    for i in range(min(num_samples, len(problems))):
        query, answer, score = problems[i]

        task = Task(
            main_query=query,
            task_id=f"ref_sample_{i}",
            metadata={"answer": answer, "score": score},  # Pointwise label
        )

        workflow_task = WorkflowTask(
            task_id=f"ref_sample_{i}",
            task=task,
        )

        samples.append(workflow_task)

    return samples


def create_math_test_samples(num_samples: int = 5) -> List[tuple[WorkflowTask, WorkflowOutput]]:
    """Create test samples (task + output pairs) for evaluation."""
    samples = []

    # Test problems with model outputs (some correct, some incorrect)
    test_cases = [
        ("What is 25 + 38?", "63", "63", True),
        ("Calculate 12 * 5", "60", "60", True),
        ("What is 90 - 45?", "45", "45", True),
        ("Find the value of 64 / 8", "8", "7", False),  # Wrong answer
        ("What is 3^4?", "81", "64", False),  # Wrong answer
        ("Calculate 18 + 22", "40", "40", True),
        ("What is 9 * 7?", "63", "56", False),  # Wrong answer
        ("Find the value of (10 + 5) * 3", "45", "45", True),
        ("What is 77 - 33?", "44", "44", True),
        ("Calculate 20 / 5 + 15", "19", "19", True),
        ("What is 6 * 8 - 10?", "38", "40", False),  # Wrong answer
        ("Find the value of 55 + 45", "100", "100", True),
        ("What is 100 - 25 - 25?", "50", "50", True),
        ("Calculate 7^2", "49", "49", True),
        ("What is (20 - 5) / 3?", "5", "7", False),  # Wrong answer
    ]

    for i in range(min(num_samples, len(test_cases))):
        query, reference, model_output, _ = test_cases[i]

        task = Task(main_query=query, task_id=f"test_sample_{i}", metadata={"answer": reference})

        workflow_task = WorkflowTask(
            task_id=f"test_sample_{i}",
            task=task,
        )

        workflow_output = WorkflowOutput(metadata={"final_answer": model_output})

        samples.append((workflow_task, workflow_output))

    return samples


async def test_get_training_tasks_new_file():
    config = read_astune_config("astune/default_config/astune_default.yaml")
    # Step 1: Create reference samples for rubric generation
    print("\nStep 1: Creating reference samples...")
    reference_samples = create_math_reference_samples(num_samples=10)
    print(f"Created {len(reference_samples)} reference samples")

    # Step 2: Initialize judge
    print("\nStep 2: Initializing RMAutoGraderJudge...")
    judge = RMAutoGraderJudge(config)

    # Step 3: Generate rubrics from reference samples
    print("\nStep 3: Generating rubrics from reference samples...")
    print("(This may take a few minutes depending on the number of samples)")
    await judge.generate_rubrics_from_samples(reference_samples)
    print("✓ Rubrics generated successfully!")
    print(f"\nGenerated rubrics:\n{judge.llm_grader.rubrics}\n")

    # Step 4: Evaluate new samples using generated rubrics
    print("\nStep 4: Evaluating new samples...")
    test_samples = create_math_test_samples(num_samples=5)

    for i, (task, output) in enumerate(test_samples, 1):
        print(f"\n--- Test Sample {i} ---")
        print(f"Query: {task.task.main_query}")
        print(f"Answer: {output.metadata['final_answer']}")
        print(f"Reference: {task.task.metadata['answer']}")

        # Use async method directly since we're in async context
        reward = await judge._async_compute_reward(task, output)
        print(f"Result: {reward}")

    print("\n" + "=" * 60)
    print("Example 1 completed!")
    print("=" * 60)


import asyncio

asyncio.run(test_get_training_tasks_new_file())
