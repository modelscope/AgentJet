from typing import Any, Dict, List, Optional, Union

from ajet.schema.document import Document
from ajet.schema.task import Task
from ajet.task_rollout.dashscope_llm_bridge import create_external_llm_fn


class BaseDataGenerator:
    def __init__(self, config):
        """
        Initialize the TaskGeneratorBase class.

        Args:
            config: Optional configuration object (LLM model, Maximum response length)
        """
        self.config = config
        self.sampling_params = self.config.data_generation.sampling_params or {}
        self.llm_client = create_external_llm_fn(
            alien_llm_model=self.config.data_generation.llm_model,
            alien_llm_response_length=self.config.data_generation.llm_response_length,
        )

    def generate_task(
        self,
        source_task: Optional[Task] = None,
        document: Optional[Document] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[Task, List[Task]]:
        """
        Generate a new task.

        Args:
            source_task: Source task for imitation (optional)
            document: Knowledge source (optional)
            extra_metadata: Additional metadata for the new task

        Returns:
            Generated Task instance
        """
        system_prompt = self._build_system_prompt(source_task, document)
        user_prompt = self._build_user_prompt(source_task, document)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Call the new LLM client
        # Returns: {"role": "assistant", "content": "..."}
        response = self.llm_client(messages=messages, sampling_params_override=self.sampling_params)
        # Extract content from response
        raw_response = response.get("content", "")
        new_task = self._parse_llm_output_to_task(raw_response, source_task, document, extra_metadata)
        return new_task

    def _build_system_prompt(
        self,
        source_task: Optional[Task],
        document: Optional[Document],
    ) -> str:
        raise NotImplementedError

    def _build_user_prompt(
        self,
        source_task: Optional[Task],
        document: Optional[Document],
    ) -> str:
        raise NotImplementedError

    def _parse_llm_output_to_task(
        self,
        raw_response: Any,
        source_task: Optional[Task],
        document: Optional[Document] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[Task, List[Task]]:
        raise NotImplementedError
