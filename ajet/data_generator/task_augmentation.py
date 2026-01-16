import re
from typing import Any, Dict, Optional, Union

from ajet.data_generator.base_data_generator import BaseDataGenerator
from ajet.schema.document import Document
from ajet.schema.task import Task


class TaskAugmentor(BaseDataGenerator):
    """
    Task Augmentation:
    - Generate new queries based on reference Query (and optional Document)
    """

    def _build_system_prompt(
        self,
        source_task: Optional[Task],
        document: Optional[Document] = None,
    ) -> str:
        """
        Build system prompt for task augmentation.
        The prompt adapts based on whether a document is provided.
        """
        base_prompt = "You are a professional expert in query generation.\n" "Your goal is to generate ONE new user query that:\n" "- Is semantically related to the reference query (similar topic/domain/intent),\n" "- Preserves the original query's style, language, task type, and approximate length,\n" "- Is natural, diverse, and fluent,\n" "- Is NOT a direct copy or minor edit of the original query.\n"

        # Conditional instructions based on document availability
        document_instructions = ""
        if document is not None and document.content:
            document_instructions = "\n" "Document context is provided for reference:\n" "- Infer the document's overall topic or domain (do NOT assume the query is tied to a specific paragraph),\n" "- Ensure the new query is compatible with that overall topic/domain,\n" "- The new query should feel naturally related to the document's theme.\n" "\n"

        # Output format requirements to ensure structured response
        output_requirements = "You MUST:\n" "- Avoid copying the original text verbatim,\n" "- Avoid minimal edits such as just changing a few words or reordering phrases,\n" "- Avoid adding explanations or commentary,\n" "- Output ONLY a valid JSON object with a single field 'query'.\n" "\n" "Example output format:\n" '{"query": "<new query text>"}\n'

        return base_prompt + document_instructions + output_requirements

    def _build_user_prompt(
        self,
        source_task: Optional[Task],
        document: Optional[Document] = None,
    ) -> str:
        """
        Build user prompt for task augmentation.
        Handles both document-present and document-absent scenarios.
        """
        if source_task is None or not source_task.main_query:
            raise ValueError("TaskAugmentor requires a task for reference.")

        original_query = source_task.main_query

        # Build the reference part (query + optional document)
        reference_info = "Reference information:\n" f"[Query]: {original_query}\n"

        # Add document content if provided
        doc_part = ""
        if document is not None and document.content:
            # Only add document-related content if a document is actually provided
            reference_info += "[Document]:\n"
            doc_part = "Here is the reference document content:\n" f"{document.content}\n" "\n" "Use this document as background knowledge while generating a new query.\n"

        user_prompt = f"{reference_info}" f"{doc_part}" "\n" "Now, generate ONE new user query that is suitable for the same context.\n" "\n" "Important rules:\n" "- Do NOT directly copy or minimally edit the original query.\n" "- Do NOT output explanations, comments, or any extra text.\n" "- Output ONLY a JSON object with the following structure:\n" '{"query": "<new query text>"}\n'

        return user_prompt

    def _parse_llm_output_to_task(
        self,
        raw_response: Any,
        source_task: Optional[Task],
        document: Optional[Document] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Parse LLM output and convert it to a Task object.
        """
        # Handle different response formats from various LLM clients
        if isinstance(raw_response, dict) and "content" in raw_response:
            # Compatible with certain client return structures
            response = raw_response["content"]
        else:
            response = str(raw_response)

        # Parse JSON from LLM response
        try:
            data = self._parse_json_response(response)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM JSON output: {e}. Raw response: {response}")

        # Extract the generated query from parsed JSON
        new_query = data.get("query", "").strip()
        if not new_query:
            raise ValueError(f"No 'query' field found in LLM output JSON. Raw JSON: {data}")

        # Construct metadata for the new task
        new_metadata = {}
        if extra_metadata:
            new_metadata.update(extra_metadata)
        # Store provenance information for traceability
        new_metadata["source_task_id"] = source_task.task_id if source_task else ""
        new_metadata["aug_type"] = "task_augmentation"
        if document:
            new_metadata["source_doc_id"] = document.doc_id

        new_task = Task(
            main_query=new_query,
            init_messages=[],
            task_id="",  # Will be assigned by the system later
            env_type=source_task.env_type if source_task else "no_env",
            metadata=new_metadata,
        )
        return new_task

    def _parse_json_response(self, response: str) -> Union[dict, list, str, float, int, bool, None]:
        """
        Parse LLM response string into JSON.
        """
        # Remove Markdown code block markers (```json and ```) if present
        response = re.sub(r"^```json|```$", "", response, flags=re.MULTILINE).strip()
        from agentscope._utils._common import _json_loads_with_repair

        return _json_loads_with_repair(response)
