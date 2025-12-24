"""
RM Gallery Iterative Rubric Judge Integration

This module integrates RM Gallery's IterativeRubricsGenerator capabilities into astuner's judge system.
It provides a data-driven approach to evaluate workflow outputs using automatically
generated rubrics from training samples.

Key Features:
- Automatic rubric generation from training/validation samples using iterative Propose-Evaluate-Revise loop
- Support for both pointwise and listwise evaluation modes
- MCR²-based smart sampling for large datasets
- Optional LLM-based categorization to organize rubrics
- Flexible scoring based on LLM-generated rubrics
- Seamless integration with astuner's workflow system
"""

import asyncio
import json
import os
from typing import List, Optional

from beast_logger import print_dict
from loguru import logger
from rm_gallery.core.generator.iterative_rubric.generator import (
    IterativeListwiseRubricsGeneratorConfig,
    IterativePointwiseRubricsGeneratorConfig,
    IterativeRubricsGenerator,
)
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.graders.schema import GraderMode
from rm_gallery.core.models.dashscope_chat_model import DashScopeChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum

from astuner.schema.task import Task, WorkflowOutput
from astuner.task_judge.base_judge import BaseJudge


class AutoGraderJudge(BaseJudge):
    """
    A data-driven judge that uses RM Gallery's IterativeRubricsGenerator to evaluate workflow outputs.

    This judge automatically generates evaluation rubrics from a set of reference samples
    and then uses those rubrics to score new workflow outputs. It uses an iterative
    Propose-Evaluate-Revise loop to ensure high-quality rubrics.

    Workflow:
    1. Initialize with configuration and reference samples
    2. Generate rubrics from reference samples using iterative refinement (one-time setup)
    3. Evaluate each workflow output against the generated rubrics

    Example Config (in YAML):
        task_judge:
          # RM Gallery Model Configuration
          model_name: "qwen-plus"  # or "gpt-4", "claude-3-sonnet", etc.
          api_key: "your-api-key"
          base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"  # optional

          # Rubric Generation Configuration
          grader_mode: "pointwise"  # or "listwise"
          language: "en"  # or "zh"

          # Advanced Configuration (optional, uses sensible defaults)
          query_specific_generate_number: 1  # number of rubrics per sample (default: 1)
          enable_categorization: false  # use LLM-based categorization (default: false)
          categories_number: 5  # number of categories when categorization enabled (default: 5)

          # Reference samples for rubric generation
          input_data_type: "dataset_file"  # or other supported types
          dataset_file:
            training:
              file_path: "tutorial/example_rm_auto_grader/rubrics_train.jsonl"

          # Custom field mapping (optional, uses defaults if not specified)
          query_field: "main_query"  # field in task containing query
          answer_field: "final_answer"  # field in metadata containing answer
          reference_field: "answer"  # field in task.metadata containing reference

          # Pointwise mode settings (only for pointwise mode)
          min_score: 0  # minimum score
          max_score: 10  # maximum score
    """

    def __init__(self, config):
        """Initialize the AutoGraderJudge.

        Args:
            config: Configuration object containing model and rubric generation settings
        """
        super().__init__(config)

        self.config = config

        # Initialize the model FIRST
        # Get API key from config or environment
        import os

        api_key = getattr(
            config.astuner.task_judge.rubrics_auto_grader, "api_key", None
        ) or os.getenv("DASHSCOPE_API_KEY")

        self.model = DashScopeChatModel(
            model=config.astuner.task_judge.rubrics_auto_grader.model_name,
            api_key=api_key,
            stream=False,
            enable_thinking=False,
        )

        # Parse config (needs self.model to be initialized)
        self.generator_config = self._parse_config()

        # Storage for generated grader
        self.llm_grader: Optional[LLMGrader] = None
        self.rubrics_generated = False

        # Field mappings for data extraction
        self.query_field = getattr(
            config.astuner.task_judge.rubrics_auto_grader, "query_field", "main_query"
        )
        self.answer_field = getattr(
            config.astuner.task_judge.rubrics_auto_grader, "answer_field", "final_answer"
        )
        self.reference_field = getattr(
            config.astuner.task_judge.rubrics_auto_grader, "reference_field", "answer"
        )

        logger.info(
            f"AutoGraderJudge initialized with mode={self.generator_config.grader_mode.value}, "
            f"language={self.generator_config.language.value}"
        )

    def _parse_config(
        self,
    ) -> IterativePointwiseRubricsGeneratorConfig | IterativeListwiseRubricsGeneratorConfig:
        """Parse astuner config into IterativeRubricsGeneratorConfig."""
        judge_config = self.config.astuner.task_judge.rubrics_auto_grader

        # Parse grader mode
        grader_mode_str = getattr(judge_config, "grader_mode", "pointwise").lower()
        grader_mode = (
            GraderMode.POINTWISE if grader_mode_str == "pointwise" else GraderMode.LISTWISE
        )

        # Parse language
        language_str = getattr(judge_config, "language", "en").upper()
        language = LanguageEnum.ZH if language_str == "ZH" else LanguageEnum.EN

        # Common configuration parameters
        common_config = {
            "model": self.model,
            "grader_name": getattr(judge_config, "grader_name", "RM Iterative Rubric Grader"),
            "language": language,
            "enable_categorization": getattr(judge_config, "enable_categorization", False),
            "query_specific_generate_number": getattr(
                judge_config, "query_specific_generate_number", 1
            ),
            "categories_number": getattr(judge_config, "categories_number", 5),
            "max_retries": getattr(judge_config, "max_retries", 5),
            "max_epochs": getattr(judge_config, "max_epochs", 3),
            "batch_size": getattr(judge_config, "batch_size", 10),
            "mcr_batch_size": getattr(judge_config, "mcr_batch_size", 10),
            "min_increment_threshold": getattr(judge_config, "min_increment_threshold", 0.002),
            "patience": getattr(judge_config, "patience", 2),
            "max_iterations": getattr(judge_config, "max_iterations", 50),
            "max_total_rubrics": getattr(judge_config, "max_total_rubrics", 200),
            "custom_evaluation_prompt": getattr(judge_config, "custom_evaluation_prompt", None),
        }

        # Create mode-specific config
        if grader_mode == GraderMode.POINTWISE:
            return IterativePointwiseRubricsGeneratorConfig(
                **common_config,
                min_score=getattr(judge_config, "min_score", 0),
                max_score=getattr(judge_config, "max_score", 10),
            )
        else:
            return IterativeListwiseRubricsGeneratorConfig(**common_config)

    async def read_reference_samples_from_dataset(self) -> List[Task]:
        # read dataset from config
        from astuner.task_reader import RouterTaskReader

        reader = RouterTaskReader(
            reader_type=self.config.astuner.task_judge.rubrics_auto_grader.input_data_type,
            reader_config=self.config.astuner.task_judge.rubrics_auto_grader,
        )
        return reader.task_reader.get_training_tasks()

    async def generate_rubrics_from_samples(self, reference_samples: List[Task] = []) -> None:
        """
        Generate evaluation rubrics from reference samples using iterative refinement.

        This method should be called once during initialization with a set of
        reference tasks that represent the types of problems to be evaluated.

        Args:
            reference_samples: List of Task objects with reference data
        """

        if len(reference_samples) == 0:
            reference_samples = await self.read_reference_samples_from_dataset()

        logger.info(f"Generating rubrics from {len(reference_samples)} reference samples...")

        # Convert Task samples to the format expected by IterativeRubricsGenerator
        training_dataset = []
        for sample in reference_samples:
            data_item = self._task_to_training_data(sample)
            if data_item:
                training_dataset.append(data_item)

        if not training_dataset:
            raise ValueError("No valid training data could be created from reference samples")

        logger.info(f"Created {len(training_dataset)} training samples for rubric generation")

        # Create IterativeRubricsGenerator
        generator = IterativeRubricsGenerator(config=self.generator_config)

        # Generate rubrics and get LLMGrader
        self.llm_grader = await generator.generate(dataset=training_dataset)

        # Save the grader
        experiment_dir = self.config.astuner.experiment_dir
        grader_save_dir = os.path.join(experiment_dir, "auto_grader.json")
        # make dirs if not exist
        os.makedirs(experiment_dir, exist_ok=True)
        print_dict({"message": "Saving generated grader config to", "path": grader_save_dir})
        json.dump(
            self.llm_grader.to_dict(),
            open(grader_save_dir, "w", encoding="utf-8"),
            indent=4,
            ensure_ascii=False,
        )

        self.rubrics_generated = True

        logger.info("Rubrics generated successfully!")
        logger.info(f"Generated rubrics:\n{self.llm_grader.rubrics}")

    async def load_rubrics_from_cache(self) -> None:
        """
        Load a pre-generated grader configuration from file.

        Args:
            grader_config_path: Path to the JSON file containing the grader config
        """

        # Load grader config and inject model
        try:
            experiment_dir = self.config.astuner.experiment_dir
            grader_save_dir = os.path.join(experiment_dir, "auto_grader.json")
            grader_config = json.load(open(grader_save_dir, "r", encoding="utf-8"))
            grader_config["model"] = self.model
            self.llm_grader = LLMGrader.from_config(grader_config)
        except Exception:
            logger.exception("Failed to load grader config from")
            await self.generate_rubrics_from_samples([])

    def _task_to_training_data(self, task: Task) -> Optional[dict]:
        """
        Convert Task to training data format for IterativeRubricsGenerator.

        Args:
            task: The workflow task containing query and reference with labels

        Returns:
            Training data dict or None if conversion fails

        Expected formats:
            Pointwise: {"query": str, "response": str, "label_score": int}
            Listwise: {"query": str, "responses": List[str], "label_rank": List[int]}
        """
        try:
            # Extract query
            query = getattr(task, self.query_field, "")
            if not query and hasattr(task, "metadata"):
                query = task.metadata.get(self.query_field, "")

            if not query:
                raise ValueError(f"Query field '{self.query_field}' not found in task")

            metadata = task.metadata if hasattr(task, "metadata") else {}

            if self.generator_config.grader_mode == GraderMode.POINTWISE:
                # Pointwise: expect metadata with "answer" and "score"
                if "answer" in metadata and "score" in metadata:
                    return {
                        "query": query,
                        "response": metadata["answer"],
                        "label_score": metadata["score"],
                    }
                else:
                    raise ValueError(
                        f"Metadata must contain 'answer' and 'score' for pointwise training data in task {task.task_id}"
                    )

            else:  # LISTWISE
                # Listwise: expect metadata with "candidates" containing list of {answer, rank}
                if "candidates" in metadata and isinstance(metadata["candidates"], list):
                    responses = []
                    label_ranks = []
                    for candidate in metadata["candidates"]:
                        responses.append(candidate["answer"])
                        label_ranks.append(candidate["rank"])

                    return {
                        "query": query,
                        "responses": responses,
                        "label_rank": label_ranks,
                    }
                else:
                    raise ValueError(
                        f"Metadata must contain 'candidates' list for listwise training data in task {task.task_id}"
                    )

        except Exception as e:
            logger.warning(f"Failed to convert task to training data: {e}")
            return None

    async def _async_compute_reward(
        self, task: Task, workflow_output: WorkflowOutput | List[WorkflowOutput]
    ):
        """
        Asynchronously compute reward using the generated rubrics.

        Args:
            task: The task being evaluated
            workflow_output: Single output for pointwise, or list of outputs for listwise

        Returns:
            For pointwise: tuple (raw_reward, is_success)
            For listwise: List of ranking results
        """
        if not self.rubrics_generated or self.llm_grader is None:
            raise RuntimeError(
                "Rubrics have not been generated yet. "
                "Call generate_rubrics_from_samples() first."
            )

        # Extract query
        query = getattr(task, self.query_field, "")
        if not query and hasattr(task, "metadata"):
            query = task.metadata.get(self.query_field, "")

        # Evaluate using LLMGrader
        try:
            if self.generator_config.grader_mode == GraderMode.POINTWISE:
                # Pointwise evaluation: single output
                if isinstance(workflow_output, list):
                    # If list provided, evaluate first output
                    answer = workflow_output[0].metadata.get(self.answer_field, "")
                else:
                    answer = workflow_output.metadata.get(self.answer_field, "")

                result = await self.llm_grader.aevaluate(query=query, answer=answer)
                return result

            else:  # LISTWISE
                # Listwise evaluation: multiple outputs
                if not isinstance(workflow_output, list):
                    logger.error("Listwise mode requires a list of workflow outputs")
                    return None

                # Format responses for listwise evaluation
                responses = []
                for output in workflow_output:
                    responses.append(output.metadata.get(self.answer_field, ""))

                # Format answer as required by listwise grader
                answer = "\n\n".join(
                    [f"Response {i+1}:\n{resp}" for i, resp in enumerate(responses)]
                )

                result = await self.llm_grader.aevaluate(
                    query=query, answer=answer, num_responses=len(responses)
                )
                return result

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return None

    def compute_reward(self, task: Task, workflow_output: WorkflowOutput) -> tuple:
        """
        Compute reward for a workflow output (synchronous wrapper).

        This is the main interface called by astuner's workflow system.

        Args:
            task: The task being evaluated
            workflow_output: The output to evaluate

        Returns:
            tuple: (raw_reward, is_success)
        """
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # If we get here, we're in an async context
            # We need to use nest_asyncio or raise an error
            try:
                import nest_asyncio

                nest_asyncio.apply()
                return loop.run_until_complete(self._async_compute_reward(task, workflow_output))
            except ImportError:
                raise RuntimeError(
                    "compute_reward() was called from an async context. "
                    "Please use 'await judge._async_compute_reward(task, output)' instead, "
                    "or install nest_asyncio: pip install nest_asyncio"
                )
        except RuntimeError:
            # No event loop running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._async_compute_reward(task, workflow_output))
            finally:
                loop.close()
