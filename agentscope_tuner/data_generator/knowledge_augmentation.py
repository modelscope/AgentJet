import json
import re
from typing import Any, Dict, List, Optional

from agentscope_tuner.data_generator.base_data_generator import BaseDataGenerator
from agentscope_tuner.schema.document import Document
from agentscope_tuner.schema.task import Task


class KnowledgeAugmentor(BaseDataGenerator):
    """
    Knowledge Augmentation:
    - Generate new tasks from Document
    """

    def _build_system_prompt(
        self,
        source_task: Optional[Task] = None,
        document: Optional[Document] = None,
    ) -> str:
        # The template can be read from self.config, but here we hardcode an example for now
        return (
            "You are an Expert Question Generation Assistant.\n"
            "Your task is to read long, complex documents and generate a large set of high-quality, non-repetitive questions that thoroughly cover all aspects of the provided content.\n"
            "**Global Rules:**\n"
            "1. Coverage: Cover all sections, topics, major themes, nuanced details, facts, arguments, examples.\n"
            "2. Diversity: Include factual, conceptual, comparative, analytical, application, and critical thinking questions. Avoid overly trivial or repetitive questions.\n"
            "3. Quality: Questions must be clear, specific, unique, and relevant to the document. Avoid vague or generic questions.\n"
            "4. Depth: Include multi-step reasoning, chronological, cause-effect, data-driven, and abstract-contextual questions.\n"
            "5. Formatting: Output must be in a JSON list of dictionaries, each dictionary containing `query` and `related_doc` keys.\n"
            "    - `query` = the generated question (one sentence, ending with a question mark)\n"
            "    - `related_doc` = the exact excerpt or closely matching text from the document that related_docs or relates to the question\n"
            "6. Boundaries: The `related_doc` field must be taken directly from the provided document; do not fabricate or introduce information from outside sources.\n"
            "7. Few-shot: If given sample questions, match style and complexity but ensure diversity.\n"
            "8. Non-repetition: Ensure no two questions are duplicates or paraphrases of the same idea. If content overlaps, merge rather than replicate.\n"
            "Always strictly follow these rules in every output."
        )

    def _build_user_prompt(
        self,
        source_task: Optional[Task] = None,
        document: Optional[Document] = None,
    ) -> str:
        if document is None or not document.content:
            raise ValueError("KnowledgeAugmentor requires a document for reference.")

        ref_doc = document.content

        user_part = []
        N = 10  # 10 is the hyperparameter we found that produces relatively stable outputs
        user_part.append(
            f"Generate exactly {N} unique, high-quality questions from the following document according to the rules in the system prompt above."
        )
        user_part.append(
            "For each question, provide the corresponding reference excerpt from the document in the `related_doc` field."
        )
        user_part.append("[DOCUMENT START]")
        user_part.append(ref_doc)
        user_part.append("[DOCUMENT END]")
        user_part.append("Now generate queries that is suitable for the JSON format.")
        user_part.append("Return your output strictly in JSON format as follows:")
        user_part.append("[")
        user_part.append(
            '  {"query": "Question text here?", "related_doc": "Direct excerpt from the document here."},'
        )
        user_part.append(
            '  {"query": "Question text here?", "related_doc": "Direct excerpt from the document here."},'
        )
        user_part.append("]")
        return "\n".join(user_part)

    def _parse_llm_output_to_task(
        self,
        raw_response: Any,
        source_task: Optional[Task] = None,
        document: Optional[Document] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Task]:
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

        # data: List[Dict[str, str]]
        all_generated_tasks = []
        for task in data:
            # Extract the generated query from parsed JSON
            new_query = task.get("query", "").strip()
            if not new_query:
                continue
            related_doc = task.get("related_doc", "").strip()
            # Construct metadata for the new task
            new_metadata = {}
            if extra_metadata:
                new_metadata.update(extra_metadata)
            # Store provenance information for traceability
            if related_doc:
                new_metadata["related_doc"] = related_doc
                new_metadata["related_doc_source"] = document.doc_id
            new_task = Task(
                main_query=new_query,
                init_messages=[],
                task_id="",  # Will be assigned by the system later
                env_type=source_task.env_type if source_task else "no_env",
                metadata=new_metadata,
            )
            all_generated_tasks.append(new_task)
        return all_generated_tasks

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response string into JSON.
        """
        # Remove Markdown code block markers (```json and ```) if present
        response = re.sub(r"^```json|```$", "", response, flags=re.MULTILINE).strip()
        return json.loads(response)
