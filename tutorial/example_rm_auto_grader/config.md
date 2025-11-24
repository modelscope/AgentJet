# Example configuration for RM Auto Grader Judge
# This configuration integrates RM Gallery's AutoGrader capabilities into astune

astune:
  # Task judge configuration using RM Auto Grader
  task_judge:
    # Use RMAutoGraderJudge for pre-generated rubrics
    # Or use RMAutoGraderBatchJudge for online rubric generation

    # ========================================
    # Model Configuration
    # ========================================
    # LLM model for rubric generation and evaluation
    model_name: "qwen3-32b"  # Options: qwen-plus, qwen-max, gpt-4, gpt-3.5-turbo, etc.

    # ========================================
    # Grader Mode Configuration
    # ========================================
    grader_mode: "pointwise"  # Options: "pointwise" or "listwise"
    language: "en"  # Options: "en" or "zh"

    # Score range for pointwise evaluation
    min_score: 0
    max_score: 10

    # Success threshold (0.0 - 1.0) - what normalized score counts as success
    success_threshold: 0.7

    # ========================================
    # Rubric Generation Configuration
    # ========================================
    # Sampling mode for rubric generation
    sampling_mode: "all_samples"  # Options: "all_samples" or "smart_sampling"

    # Number of rubrics to generate per sample
    generate_number: 3

    # Maximum epochs for iterative refinement
    max_epochs: 3

    # Maximum retry attempts for LLM API calls
    max_retries: 5

    # Batch processing settings (for smart_sampling mode)
    batch_size: 10
    mcr_batch_size: 10

    # Aggregation mode for final rubrics
    aggregation_mode: "keep_all"  # Options: "keep_all" or "merge_similar"

    # ========================================
    # Reference Samples Configuration
    # ========================================
    # Path to reference samples (for pre-generating rubrics)
    # reference_samples_path: "data/reference_samples.jsonl"

    # Number of reference samples to use
    num_reference_samples: 20

    # ========================================
    # Field Mapping Configuration
    # ========================================
    # Field names for extracting data from WorkflowTask and WorkflowOutput
    query_field: "main_query"  # Field in task containing the query
    answer_field: "final_answer"  # Field in output metadata containing the answer
    reference_field: "answer"  # Field in task.metadata containing reference answer

    # ========================================
    # Grader name for logging
    # ========================================
    grader_name: "RM Auto Grader"


# ============================================
# Batch Judge Specific Configuration
# ============================================
# Uncomment and use these settings when using RMAutoGraderBatchJudge

# astune:
#   task_judge:
#     class_name: RMAutoGraderBatchJudge
#
#     # ... (include all settings from above)
#
#     # Warmup phase settings
#     warmup_samples: 20  # Collect N samples before generating rubrics
#
#     # Regeneration settings
#     regenerate_interval: 0  # Regenerate rubrics every N evaluations (0 = never)


# ============================================
# Example for Math Problem Evaluation
# ============================================
# astune:
#   task_judge:
#     class_name: RMAutoGraderJudge
#     model_name: "qwen-plus"
#     grader_mode: "pointwise"
#     language: "en"
#     min_score: 0
#     max_score: 10
#     success_threshold: 0.8
#     sampling_mode: "all_samples"
#     generate_number: 5
#     max_epochs: 3
#     aggregation_mode: "merge_similar"
#     num_reference_samples: 30
#     query_field: "main_query"
#     answer_field: "final_answer"
#     reference_field: "answer"


# ============================================
# Example for Agent Task Evaluation
# ============================================
# astune:
#   task_judge:
#     class_name: RMAutoGraderJudge
#     model_name: "gpt-4"
#     grader_mode: "pointwise"
#     language: "en"
#     min_score: 0
#     max_score: 100
#     success_threshold: 0.7
#     sampling_mode: "smart_sampling"
#     generate_number: 3
#     max_epochs: 2
#     batch_size: 15
#     mcr_batch_size: 10
#     aggregation_mode: "keep_all"
#     num_reference_samples: 50
#     query_field: "main_query"
#     answer_field: "agent_output"
#     reference_field: "expected_outcome"

