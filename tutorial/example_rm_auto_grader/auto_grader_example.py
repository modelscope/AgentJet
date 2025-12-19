"""
Example: Using RM Iterative Rubric Judge with astuner

This example demonstrates how to use the RM Gallery IterativeRubricsGenerator integration
for data-driven evaluation of workflow outputs.

The IterativeRubricsGenerator uses an iterative Propose-Evaluate-Revise loop to
generate high-quality evaluation rubrics from reference samples.

This example shows:
1. Pointwise evaluation mode (scoring individual outputs)
2. Listwise evaluation mode (ranking multiple outputs)
"""

import asyncio
from typing import List

from rm_gallery.core.generator.iterative_rubric.query_rubric_generator import (
    LISTWISE_EVALUATION_TEMPLATE,
    POINTWISE_EVALUATION_TEMPLATE,
)

from astuner.schema.task import Task
from astuner.task_judge.rm_auto_grader_judge import AutoGraderJudge
from astuner.workflow import WorkflowOutput, WorkflowTask

# ============================================
# Example 1: Pre-generated Rubrics Approach
# ============================================


async def example_pregerated_rubrics():
    """
    Example of using RMAutoGraderJudge with iteratively-generated rubrics (Pointwise mode).

    This approach uses the IterativeRubricsGenerator to automatically create
    evaluation rubrics from reference samples using a Propose-Evaluate-Revise loop.
    """
    print("\n\nExample 1: Pointwise Evaluation with Iterative Rubrics")

    # Mock config object
    class MockConfig:
        class Astuner:
            class TaskJudge:
                class RubricsAutoGrader:
                    # Model configuration
                    model_name = "qwen3-32b"

                    # Grader configuration
                    grader_mode = "pointwise"
                    language = "en"
                    min_score = 0
                    max_score = 1

                    # Evaluation prompt template
                    custom_evaluation_prompt = POINTWISE_EVALUATION_TEMPLATE

                    # Advanced configuration (optional)
                    query_specific_generate_number = 1
                    max_epochs = 2
                    max_retries = 3
                    enable_categorization = False

                    # Field mappings
                    query_field = "main_query"
                    answer_field = "final_answer"
                    reference_field = "answer"

                    grader_name = "Math Iterative Rubric Grader"

                rubrics_auto_grader = RubricsAutoGrader()

            task_judge = TaskJudge()
            experiment_dir = "/tmp/rm_grader_example"

        astuner = Astuner()

    config = MockConfig()

    # Step 1: Create reference samples for rubric generation
    reference_samples = create_math_reference_samples(num_samples=10)

    # Step 2: Initialize judge
    judge = AutoGraderJudge(config)

    # Step 3: Generate rubrics from reference samples using iterative refinement
    await judge.generate_rubrics_from_samples(reference_samples)

    # Step 4: Evaluate new samples using generated rubrics
    test_samples = create_math_test_samples(num_samples=5)

    for i, (workflow_task, output) in enumerate(test_samples, 1):
        print(f"\n--- Test Sample {i} ---")
        print(f"Query: {workflow_task.task.main_query}")
        print(f"Answer: {output.metadata['final_answer']}")
        print(f"Reference: {workflow_task.task.metadata['answer']}")

        # Use async method directly since we're in async context
        reward = await judge._async_compute_reward(workflow_task.task, output)
        print(f"Result: {reward}")

    print("Example 1 completed!")


# ============================================
# Example 2: Listwise Mode with Multiple Outputs
# ============================================


async def example_listwise_mode():
    """
    Example of using RMAutoGraderJudge in Listwise mode with iterative rubrics.

    Listwise mode ranks multiple candidate answers for the same query.
    This is useful for:
    - Comparing multiple model outputs
    - Ranking candidate responses by quality
    - Batch evaluation of similar tasks
    """
    print("\n\nExample 2: Listwise Ranking with Iterative Rubrics")

    # Mock config object
    class MockConfig:
        class Astuner:
            class TaskJudge:
                class RubricsAutoGrader:
                    # Model configuration
                    model_name = "qwen3-32b"

                    # Grader configuration - LISTWISE mode
                    grader_mode = "listwise"  # Key difference!
                    language = "en"
                    # Note: min_score/max_score not needed for listwise mode

                    # Evaluation prompt template
                    custom_evaluation_prompt = LISTWISE_EVALUATION_TEMPLATE

                    # Advanced configuration (optional)
                    query_specific_generate_number = 2
                    max_epochs = 2
                    max_retries = 3
                    enable_categorization = False

                    # Field mappings
                    query_field = "main_query"
                    answer_field = "final_answer"
                    reference_field = "answer"

                    grader_name = "Math Listwise Iterative Grader"

                rubrics_auto_grader = RubricsAutoGrader()

            task_judge = TaskJudge()
            experiment_dir = "/tmp/rm_grader_example_listwise"

        astuner = Astuner()

    config = MockConfig()

    # Step 1: Create reference samples with multiple outputs per query
    reference_samples = create_listwise_reference_samples(num_samples=5)

    # Step 2: Initialize judge
    judge = AutoGraderJudge(config)

    # Step 3: Generate ranking rubrics using iterative refinement
    await judge.generate_rubrics_from_samples(reference_samples)

    # Step 4: Evaluate multiple candidate answers for new queries
    test_queries = create_listwise_test_samples(num_queries=3)

    for i, (workflow_task, candidate_outputs) in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"Query {i}: {workflow_task.task.main_query}")
        print(f"{'='*50}")
        print(f"Evaluating {len(candidate_outputs)} candidates...")

        # Evaluate all candidates together (pass list for listwise mode)
        grader_rank_result = await judge._async_compute_reward(
            workflow_task.task, candidate_outputs
        )

        if grader_rank_result and hasattr(grader_rank_result, "rank"):
            ranks = grader_rank_result.rank
            reason = grader_rank_result.reason

            print(f"\nGrader reasoning: {reason}")

            results = []
            for j, (output, rank) in enumerate(zip(candidate_outputs, ranks), 1):
                results.append((j, output.metadata["final_answer"], rank))

            # Sort by rank (ascending, rank 1 is best)
            results.sort(key=lambda x: x[2])

            print("\nRanking (best to worst):")
            for display_rank, (idx, answer, model_rank) in enumerate(results, 1):
                print(f"  {display_rank}. Candidate {idx}: '{answer}' (Model Rank: {model_rank})")
        else:
            print("No results returned from evaluation")

    print("\n" + "=" * 60)
    print("Example 2 completed!")
    print("=" * 60)


# ============================================
# Helper Functions
# ============================================


def create_math_reference_samples(num_samples: int = 10) -> List[Task]:
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

        samples.append(task)

    return samples


def create_listwise_reference_samples(num_samples: int = 5) -> List[Task]:
    """
    Create reference samples for Listwise mode rubric generation.

    Each sample should contain multiple outputs with different quality levels.
    This helps the model learn to distinguish between good and bad answers.
    """
    samples = []

    # Math problems with multiple candidate answers and their quality rankings
    # Lower rank = better quality (rank 1 is best)
    problems = [
        (
            "What is 10 + 15?",
            [
                ("25", 1),  # Perfect answer
                ("Twenty-five", 2),  # Correct but different format
                ("24", 3),  # Close but wrong
                ("30", 4),  # Wrong
            ],
        ),
        (
            "Calculate 6 * 7",
            [
                ("42", 1),  # Perfect
                ("6*7=42", 2),  # Correct with work shown
                ("43", 3),  # Off by one
                ("36", 4),  # Wrong (6*6)
            ],
        ),
        (
            "What is 50 - 18?",
            [
                ("32", 1),  # Perfect
                ("50-18=32", 2),  # Correct with work
                ("33", 3),  # Close
                ("42", 4),  # Wrong
            ],
        ),
        (
            "Find 12 / 4",
            [
                ("3", 1),  # Perfect
                ("3.0", 2),  # Correct, decimal format
                ("4", 3),  # Wrong
                ("2", 4),  # Very wrong
            ],
        ),
        (
            "What is 2^5?",
            [
                ("32", 1),  # Perfect
                ("2*2*2*2*2=32", 2),  # Correct with work
                ("16", 3),  # 2^4, common mistake
                ("10", 4),  # 2*5, wrong operation
            ],
        ),
    ]

    for i in range(min(num_samples, len(problems))):
        query, candidates = problems[i]

        # Create task with metadata containing all candidates and their ranks
        task = Task(
            main_query=query,
            task_id=f"listwise_ref_{i}",
            metadata={"candidates": [{"answer": ans, "rank": rank} for ans, rank in candidates]},
        )

        samples.append(task)

    return samples


def create_listwise_test_samples(
    num_queries: int = 3,
) -> List[tuple[WorkflowTask, List[WorkflowOutput]]]:
    """
    Create test queries with multiple candidate outputs for Listwise evaluation.

    Returns:
        List of (query_task, list_of_candidate_outputs) tuples
    """
    test_data = []

    # Test queries with multiple candidate answers
    queries = [
        (
            "What is 45 + 37?",
            [
                "82",  # Correct
                "45+37=82",  # Correct with work
                "83",  # Off by one
                "72",  # Wrong
                "Eighty-two",  # Correct, word form
            ],
        ),
        (
            "Calculate 9 * 8",
            [
                "72",  # Correct
                "73",  # Close
                "81",  # 9*9, common mistake
                "9*8=72",  # Correct with work
                "64",  # 8*8, wrong
            ],
        ),
        (
            "What is 100 - 45?",
            [
                "55",  # Correct
                "100-45=55",  # Correct with work
                "65",  # Wrong
                "Fifty-five",  # Correct, word form
                "54",  # Off by one
            ],
        ),
    ]

    for i in range(min(num_queries, len(queries))):
        query, candidates = queries[i]

        # Create task
        task = Task(
            main_query=query,
            task_id=f"listwise_test_{i}",
            metadata={},  # No reference needed for evaluation
        )

        workflow_task = WorkflowTask(
            task_id=f"listwise_test_{i}",
            task=task,
        )

        # Create output for each candidate
        candidate_outputs = []
        for j, candidate_answer in enumerate(candidates):
            output = WorkflowOutput(metadata={"final_answer": candidate_answer})
            candidate_outputs.append(output)

        test_data.append((workflow_task, candidate_outputs))

    return test_data


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


# ============================================
# Main Entry Point
# ============================================


async def main():
    """Run all examples."""

    # Run examples
    try:
        await example_pregerated_rubrics()
        await example_listwise_mode()

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
