"""
Example: Using RM Auto Grader Judge with astune

This example demonstrates how to use the RM Gallery AutoGrader integration
for data-driven evaluation of workflow outputs.

The example shows two approaches:
1. Pre-generating rubrics from reference samples
2. Online rubric generation during training (batch mode)
"""

import asyncio
from typing import List

from astune.workflow import Workflow, WorkflowTask, WorkflowOutput
from astune.schema.task import Task
from astune.task_judge.rm_auto_grader_judge import RMAutoGraderJudge

# ============================================
# Example 1: Pre-generated Rubrics Approach
# ============================================

async def example_pregerated_rubrics():
    """
    Example of using RMAutoGraderJudge with pre-generated rubrics.
    
    This approach is suitable when you have a separate set of reference samples
    and want to generate rubrics once before training begins.
    """
    print("=" * 60)
    print("Example 1: Pre-generated Rubrics Approach")
    print("=" * 60)
    
    # Mock config object
    class MockConfig:
        class Astune:
            class TaskJudge:
                # Model configuration
                model_name = "qwen3-32b"
                
                # Grader configuration
                grader_mode = "pointwise"
                language = "en"
                min_score = 0
                max_score = 1
                success_threshold = 0.7
                
                # Rubric generation configuration
                sampling_mode = "all_samples"
                generate_number = 1
                max_epochs = 2
                max_retries = 3
                aggregation_mode = "keep_all"
                
                # Field mappings
                query_field = "main_query"
                answer_field = "final_answer"
                reference_field = "answer"
                
                grader_name = "Math Auto Grader"
            
            task_judge = TaskJudge()
        
        astune = Astune()
    
    config = MockConfig()
    
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


# ============================================
# Example 2: Listwise Mode with Multiple Outputs
# ============================================

async def example_listwise_mode():
    """
    Example of using RMAutoGraderJudge in Listwise mode.
    
    Listwise mode ranks multiple candidate answers for the same query.
    This is useful for:
    - Comparing multiple model outputs
    - Ranking candidate responses by quality
    - Batch evaluation of similar tasks
    """
    print("\n\n" + "=" * 60)
    print("Example 2: Listwise Mode with Multiple Outputs")
    print("=" * 60)
    
    # Mock config object
    class MockConfig:
        class Astune:
            class TaskJudge:
                # Model configuration
                model_name = "qwen3-32b"
                
                # Grader configuration - LISTWISE mode
                grader_mode = "listwise"  # Key difference!
                language = "en"
                min_score = 0
                max_score = 1
                success_threshold = 0.7
                
                # Rubric generation configuration
                sampling_mode = "all_samples"
                generate_number = 2
                max_epochs = 2
                max_retries = 3
                aggregation_mode = "keep_all"
                
                # Field mappings
                query_field = "main_query"
                answer_field = "final_answer"
                reference_field = "answer"
                
                grader_name = "Math Listwise Grader"
            
            task_judge = TaskJudge()
        
        astune = Astune()
    
    config = MockConfig()
    
    # Step 1: Create reference samples with multiple outputs per query
    print("\nStep 1: Creating reference samples with multiple outputs...")
    reference_samples = create_listwise_reference_samples(num_samples=5)
    print(f"Created {len(reference_samples)} reference samples")
    print("Each sample contains multiple candidate answers with rankings")
    
    # Step 2: Initialize judge
    print("\nStep 2: Initializing RMAutoGraderJudge in Listwise mode...")
    judge = RMAutoGraderJudge(config)
    
    # Step 3: Generate rubrics from reference samples
    print("\nStep 3: Generating rubrics from reference samples...")
    print("(Listwise mode learns to rank multiple outputs)")
    await judge.generate_rubrics_from_samples(reference_samples)
    print("✓ Rubrics generated successfully!")
    print(f"\nGenerated rubrics:\n{judge.llm_grader.rubrics}\n")
    
    # Step 4: Evaluate multiple candidate answers for new queries
    print("\nStep 4: Evaluating multiple candidates for each query...")
    test_queries = create_listwise_test_samples(num_queries=3)
    
    for i, (query_task, candidate_outputs) in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"Query {i}: {query_task.task.main_query}")
        print(f"{'='*50}")
        print(f"Evaluating {len(candidate_outputs)} candidates...")
        
        # Evaluate all candidates together (pass list for listwise mode)
        results_batch = await judge._async_compute_reward(query_task, candidate_outputs)
        
        if results_batch and len(results_batch) > 0 and len(results_batch[0]) > 0:
            # Extract ranks for each candidate (score field contains rank in listwise mode)
            results = []
            for j, (output, grader_score) in enumerate(zip(candidate_outputs, results_batch[0]), 1):
                rank = grader_score.score if hasattr(grader_score, 'score') else j
                results.append((j, output.metadata['final_answer'], rank))
            
            # Sort by rank (ascending, rank 1 is best)
            results.sort(key=lambda x: x[2])
            
            print("\nRanking (best to worst):")
            for display_rank, (idx, answer, model_rank) in enumerate(results, 1):
                print(f"  {display_rank}. Candidate {idx}: '{answer}' (Model Rank: {model_rank})")
        else:
            print("No results returned from evaluation")
    
    print("\n" + "=" * 60)
    print("Example 3 completed!")
    print("=" * 60)


# ============================================
# Helper Functions
# ============================================

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
            metadata={
                "answer": answer,
                "score": score  # Pointwise label
            }
        )
        
        workflow_task = WorkflowTask(
            task_id=f"ref_sample_{i}",
            task=task,
        )
        
        samples.append(workflow_task)
    
    return samples


def create_listwise_reference_samples(num_samples: int = 5) -> List[WorkflowTask]:
    """
    Create reference samples for Listwise mode rubric generation.
    
    Each sample should contain multiple outputs with different quality levels.
    This helps the model learn to distinguish between good and bad answers.
    """
    samples = []
    
    # Math problems with multiple candidate answers and their quality rankings
    # Lower rank = better quality (rank 1 is best)
    problems = [
        ("What is 10 + 15?", [
            ("25", 1),           # Perfect answer
            ("Twenty-five", 2),  # Correct but different format
            ("24", 3),           # Close but wrong
            ("30", 4),           # Wrong
        ]),
        ("Calculate 6 * 7", [
            ("42", 1),           # Perfect
            ("6*7=42", 2),       # Correct with work shown
            ("43", 3),           # Off by one
            ("36", 4),           # Wrong (6*6)
        ]),
        ("What is 50 - 18?", [
            ("32", 1),           # Perfect
            ("50-18=32", 2),     # Correct with work
            ("33", 3),           # Close
            ("42", 4),           # Wrong
        ]),
        ("Find 12 / 4", [
            ("3", 1),            # Perfect
            ("3.0", 2),          # Correct, decimal format
            ("4", 3),            # Wrong
            ("2", 4),            # Very wrong
        ]),
        ("What is 2^5?", [
            ("32", 1),           # Perfect
            ("2*2*2*2*2=32", 2), # Correct with work
            ("16", 3),           # 2^4, common mistake
            ("10", 4),           # 2*5, wrong operation
        ]),
    ]
    
    for i in range(min(num_samples, len(problems))):
        query, candidates = problems[i]
        
        # Create task with metadata containing all candidates and their ranks
        task = Task(
            main_query=query,
            task_id=f"listwise_ref_{i}",
            metadata={
                "candidates": [{"answer": ans, "rank": rank} for ans, rank in candidates]
            }
        )
        
        workflow_task = WorkflowTask(
            task_id=f"listwise_ref_{i}",
            task=task,
        )
        
        samples.append(workflow_task)
    
    return samples


def create_listwise_test_samples(num_queries: int = 3) -> List[tuple[WorkflowTask, List[WorkflowOutput]]]:
    """
    Create test queries with multiple candidate outputs for Listwise evaluation.
    
    Returns:
        List of (query_task, list_of_candidate_outputs) tuples
    """
    test_data = []
    
    # Test queries with multiple candidate answers
    queries = [
        ("What is 45 + 37?", [
            "82",              # Correct
            "45+37=82",        # Correct with work
            "83",              # Off by one
            "72",              # Wrong
            "Eighty-two",      # Correct, word form
        ]),
        ("Calculate 9 * 8", [
            "72",              # Correct
            "73",              # Close
            "81",              # 9*9, common mistake
            "9*8=72",          # Correct with work
            "64",              # 8*8, wrong
        ]),
        ("What is 100 - 45?", [
            "55",              # Correct
            "100-45=55",       # Correct with work
            "65",              # Wrong
            "Fifty-five",      # Correct, word form
            "54",              # Off by one
        ]),
    ]
    
    for i in range(min(num_queries, len(queries))):
        query, candidates = queries[i]
        
        # Create task
        task = Task(
            main_query=query,
            task_id=f"listwise_test_{i}",
            metadata={}  # No reference needed for evaluation
        )
        
        workflow_task = WorkflowTask(
            task_id=f"listwise_test_{i}",
            task=task,
        )
        
        # Create output for each candidate
        candidate_outputs = []
        for j, candidate_answer in enumerate(candidates):
            output = WorkflowOutput(
                metadata={"final_answer": candidate_answer}
            )
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
        
        task = Task(
            main_query=query,
            task_id=f"test_sample_{i}",
            metadata={"answer": reference}
        )
        
        workflow_task = WorkflowTask(
            task_id=f"test_sample_{i}",
            task=task,
        )
        
        workflow_output = WorkflowOutput(
            metadata={"final_answer": model_output}
        )
        
        samples.append((workflow_task, workflow_output))
    
    return samples



# ============================================
# Main Entry Point
# ============================================

async def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "RM Auto Grader Judge Examples" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Run examples
    try:
        await example_pregerated_rubrics()
        # await example_batch_rubrics()
        await example_listwise_mode()
        
        print("\n\n" + "═" * 60)
        print("All examples completed successfully!")
        print("═" * 60)
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())

