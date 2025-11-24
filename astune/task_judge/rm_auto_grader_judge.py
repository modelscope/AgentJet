"""
RM Gallery Auto Grader Judge Integration

This module integrates RM Gallery's AutoGrader capabilities into astune's judge system.
It provides a data-driven approach to evaluate workflow outputs using automatically
generated rubrics from training samples.

Key Features:
- Automatic rubric generation from training/validation samples
- Support for both pointwise and listwise evaluation modes
- Flexible scoring based on LLM-generated rubrics
- Seamless integration with astune's workflow system
"""

import asyncio
from typing import List, Optional, Dict, Any
from loguru import logger

from astune.workflow import WorkflowOutput, WorkflowTask
from astune.task_judge.judge_base import JudgeBase


from rm_gallery.core.grader.auto.auto_grader import AutoGrader, AutoGraderConfig
from rm_gallery.core.grader.auto.auto_rubrics import AutoRubricsConfig, SamplingMode, AggregationMode
from rm_gallery.core.grader.base import GraderMode, LLMGrader
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.data import EvalCase
from rm_gallery.core.schema.template import LanguageEnum

from rm_gallery.core.grader.base import aevaluate_with_cases


class RMAutoGraderJudge(JudgeBase):
    """
    A data-driven judge that uses RM Gallery's AutoGrader to evaluate workflow outputs.

    This judge automatically generates evaluation rubrics from a set of reference samples
    and then uses those rubrics to score new workflow outputs.

    Workflow:
    1. Initialize with configuration and reference samples
    2. Generate rubrics from reference samples (one-time setup)
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
          min_score: 0
          max_score: 10

          # AutoRubrics Configuration
          sampling_mode: "all_samples"  # or "smart_sampling"
          generate_number: 3  # number of rubrics per sample
          max_epochs: 3  # max iterations for refinement
          aggregation_mode: "keep_all"  # or "merge_similar"

          # Reference samples for rubric generation
          reference_samples_path: "path/to/reference_samples.jsonl"  # optional
          num_reference_samples: 20  # number of samples to use for rubric generation

          # Custom field mapping (optional)
          query_field: "main_query"  # field in task containing query
          answer_field: "final_answer"  # field in metadata containing answer
          reference_field: "answer"  # field in task.metadata containing reference
    """

    def __init__(self, config):
        """Initialize the RMAutoGraderJudge.

        Args:
            config: Configuration object containing model and rubric generation settings
        """
        super().__init__(config)

        self.config = config
        self.grader_config = self._parse_config()

        # Initialize the model
        self.model = OpenAIChatModel(model=config.astune.task_judge.rubrics_auto_grader.model_name, stream=False)

        # Storage for generated grader
        self.llm_grader: Optional[LLMGrader] = None
        self.rubrics_generated = False

        # Field mappings for data extraction
        self.query_field = getattr(config.astune.task_judge.rubrics_auto_grader, 'query_field', 'main_query')
        self.answer_field = getattr(config.astune.task_judge.rubrics_auto_grader, 'answer_field', 'final_answer')
        self.reference_field = getattr(config.astune.task_judge.rubrics_auto_grader, 'reference_field', 'answer')

        logger.info(
            f"RMAutoGraderJudge initialized with mode={self.grader_config.method_config.grader_mode.value}, "
            f"language={self.grader_config.method_config.language.value}"
        )

    def _parse_config(self) -> AutoGraderConfig:
        """Parse astune config into AutoGraderConfig."""
        judge_config = self.config.astune.task_judge.rubrics_auto_grader

        # Parse grader mode
        grader_mode_str = getattr(judge_config, 'grader_mode', 'pointwise').lower()
        grader_mode = GraderMode.POINTWISE if grader_mode_str == 'pointwise' else GraderMode.LISTWISE

        # Parse language
        language_str = getattr(judge_config, 'language', 'en').upper()
        language = LanguageEnum.ZH if language_str == 'ZH' else LanguageEnum.EN

        # Parse sampling mode
        sampling_mode_str = getattr(judge_config, 'sampling_mode', 'all_samples')
        sampling_mode = SamplingMode.ALL_SAMPLES if sampling_mode_str == 'all_samples' else SamplingMode.SMART_SAMPLING

        # Parse aggregation mode
        aggregation_mode_str = getattr(judge_config, 'aggregation_mode', 'keep_all')
        aggregation_mode = AggregationMode.KEEP_ALL if aggregation_mode_str == 'keep_all' else AggregationMode.MERGE_SIMILAR

        # Create AutoRubricsConfig
        rubrics_config = AutoRubricsConfig(
            sampling_mode=sampling_mode,
            grader_mode=grader_mode,
            language=language,
            generate_number=getattr(judge_config, 'generate_number', 3),
            max_retries=getattr(judge_config, 'max_retries', 5),
            max_epochs=getattr(judge_config, 'max_epochs', 3),
            min_score=getattr(judge_config, 'min_score', 0),
            max_score=getattr(judge_config, 'max_score', 10),
            batch_size=getattr(judge_config, 'batch_size', 10),
            mcr_batch_size=getattr(judge_config, 'mcr_batch_size', 10),
            aggregation_mode=aggregation_mode,
        )

        # Create AutoGraderConfig
        auto_grader_config = AutoGraderConfig(
            method="auto_rubrics",
            method_config=rubrics_config,
            grader_name=getattr(judge_config, 'grader_name', 'RM Auto Grader'),
        )

        return auto_grader_config

    async def generate_rubrics_from_samples(self, reference_samples: List[WorkflowTask]) -> None:
        """
        Generate evaluation rubrics from reference samples.

        This method should be called once during initialization with a set of
        reference tasks that represent the types of problems to be evaluated.

        Args:
            reference_samples: List of WorkflowTask objects with reference data
        """
        logger.info(f"Generating rubrics from {len(reference_samples)} reference samples...")

        # Convert WorkflowTask samples to EvalCase format for rubric generation
        # Use reference answers as example "good" outputs
        eval_cases = []
        for sample in reference_samples:
            eval_case = self._workflow_task_to_eval_case(sample, workflow_output=None, for_rubric_generation=True)
            if eval_case:
                eval_cases.append(eval_case)

        if not eval_cases:
            raise ValueError("No valid eval cases could be created from reference samples")

        logger.info(f"Created {len(eval_cases)} eval cases for rubric generation")

        # Create AutoGrader
        auto_grader = AutoGrader(
            model=self.model,
            parser=None,
            config=self.grader_config,
        )

        # Generate rubrics and get LLMGrader
        self.llm_grader = await auto_grader.run(eval_cases)
        self.rubrics_generated = True

        logger.info("Rubrics generated successfully!")
        logger.info(f"Generated rubrics:\n{self.llm_grader.rubrics}")


    def _workflow_task_to_eval_case(
        self,
        workflow_task: WorkflowTask,
        workflow_output: Optional[WorkflowOutput | List[WorkflowOutput]] = None,
        for_rubric_generation: bool = False
    ) -> Optional[EvalCase]:
        """
        Convert WorkflowTask (and optionally WorkflowOutput) to EvalCase format.

        Args:
            workflow_task: The workflow task containing query and reference
            workflow_output: Single output or list of outputs (for listwise evaluation)
            for_rubric_generation: If True, create training format with labeled data

        Returns:
            EvalCase object or None if conversion fails
        """
        try:
            # Extract query
            query = getattr(workflow_task.task, self.query_field, "")
            if not query and hasattr(workflow_task.task, 'metadata'):
                query = workflow_task.task.metadata.get(self.query_field, "")

            # Extract reference answer
            reference = ""
            if hasattr(workflow_task.task, 'metadata') and self.reference_field in workflow_task.task.metadata:
                reference = workflow_task.task.metadata[self.reference_field]

            # Build input dict - reference should always be in input for comparison
            input_dict = {
                "query": query,
            }

            # Build output dict
            outputs = []

            if for_rubric_generation:
                # For rubric generation: directly construct outputs from metadata
                # Metadata should contain pre-labeled data (with score/rank)

                grader_mode = self.grader_config.method_config.grader_mode
                metadata = workflow_task.task.metadata if hasattr(workflow_task.task, 'metadata') else {}

                if grader_mode == GraderMode.POINTWISE:
                    # Pointwise: expect metadata with "answer" and "score"
                    if 'answer' in metadata and 'score' in metadata:
                        outputs.append({
                            "answer": metadata['answer'],
                            "score": metadata['score']
                        })
                    else:
                        logger.warning(f"No labeled data found for pointwise rubric generation in task {workflow_task.task_id}")
                        return None

                else:  # LISTWISE
                    # Listwise: expect metadata with "candidates" containing list of {answer, rank}
                    if 'candidates' in metadata and isinstance(metadata['candidates'], list):
                        for candidate in metadata['candidates']:
                            outputs.append({
                                "answer": candidate['answer'],
                                "rank": candidate['rank']
                            })
                    else:
                        logger.warning(f"No labeled data found for listwise rubric generation in task {workflow_task.task_id}")
                        return None
            else:
                # For evaluation: use the actual model output (no labels)
                if workflow_output:
                    # Handle both single output and list of outputs
                    if isinstance(workflow_output, list):
                        for output in workflow_output:
                            answer = output.metadata.get(self.answer_field, "")
                            outputs.append({"answer": answer})
                    else:
                        answer = workflow_output.metadata.get(self.answer_field, "")
                        outputs.append({"answer": answer})
                else:
                    logger.warning(f"No workflow output provided for evaluation of task {workflow_task.task_id}")
                    return None

            if not outputs:
                logger.warning(f"No outputs found for task {workflow_task.task_id}")
                return None

            return EvalCase(input=input_dict, outputs=outputs)

        except Exception as e:
            logger.warning(f"Failed to convert workflow task to eval case: {e}")
            return None

    async def _async_compute_reward(
        self,
        workflow_task: WorkflowTask,
        workflow_output: WorkflowOutput | List[WorkflowOutput]
    ):
        """
        Asynchronously compute reward using the generated rubrics.

        Args:
            workflow_task: The task being evaluated
            workflow_output: Single output for pointwise, or list of outputs for listwise

        Returns:
            For pointwise: tuple (raw_reward, is_success)
            For listwise: List[List[GraderScore]]
        """
        if not self.rubrics_generated or self.llm_grader is None:
            raise RuntimeError(
                "Rubrics have not been generated yet. "
                "Call generate_rubrics_from_samples() first."
            )

        # Create eval_case(s) based on input
        eval_case = self._workflow_task_to_eval_case(workflow_task, workflow_output, for_rubric_generation=False)

        if not eval_case:
            logger.error("Failed to create eval case from workflow task and output")
            return None

        # Evaluate using LLMGrader - it handles both pointwise and listwise internally
        try:
            results = await aevaluate_with_cases(self.llm_grader, eval_cases=[eval_case])
            # For all other cases (listwise, or pointwise with multiple outputs), return raw results
            return results

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return None

    def compute_reward(
        self,
        workflow_task: WorkflowTask,
        workflow_output: WorkflowOutput
    ) -> tuple:
        """
        Compute reward for a workflow output (synchronous wrapper).

        This is the main interface called by astune's workflow system.

        Args:
            workflow_task: The task being evaluated
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
                return loop.run_until_complete(
                    self._async_compute_reward(workflow_task, workflow_output)
                )
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
                return loop.run_until_complete(
                    self._async_compute_reward(workflow_task, workflow_output)
                )
            finally:
                loop.close()

