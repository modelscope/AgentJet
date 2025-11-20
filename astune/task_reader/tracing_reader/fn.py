import re
from typing import Any, Dict, List, Union


class Fn:
    """
    A class that defines a task with specific requirements, input parameters, and output format.
    It uses LlmClient to execute the task and parses the result using markdown-kv format.
    """

    def __init__(self,
                 name: str,
                 description: str,
                 alien_llm_chat_fn: Any,
                 input_schema: Dict[str, str],
                 output_schema: Dict[str, str],
                 sampling_params: Dict[str, Any] = {}):
        """
        Initialize the Fn class.

        Args:
            name: The name of the function/task
            description: Description of what the task does
            llm_client: The LLM client to use for execution
            input_schema: Dictionary defining the input parameters format (name -> description)
            output_schema: Dictionary defining the output format (name -> description)
            sampling_params: Parameters for LLM sampling
        """
        self.name = name
        self.description = description
        self.alien_llm_chat_fn = alien_llm_chat_fn
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.sampling_params = sampling_params or {}

    def _build_system_prompt(self) -> str:
        """
        Build the system prompt that defines the task requirements, input format, and output format.

        Returns:
            str: The system prompt for the LLM
        """
        prompt_parts = []

        # Task description
        prompt_parts.append(f"## Task: {self.name}")
        prompt_parts.append(f"Description: {self.description}")
        prompt_parts.append("")

        # Input format
        prompt_parts.append("## Input Parameters")
        if self.input_schema:
            prompt_parts.append("The input parameters should be provided in the following format:")
            for param_name, param_desc in self.input_schema.items():
                prompt_parts.append(f"- {param_name}: {param_desc}")
            prompt_parts.append("")
            prompt_parts.append("Input can be either:")
            prompt_parts.append("1. A single record (dict) with the above parameters")
            prompt_parts.append("2. Multiple records (list of dicts) with the same parameters")
        else:
            prompt_parts.append("This task takes no input parameters.")
        prompt_parts.append("")

        # Output format
        prompt_parts.append("## Output Format")
        prompt_parts.append("You must return the result in markdown-kv format as shown below:")
        prompt_parts.append("```markdown")
        if self.output_schema:
            prompt_parts.append("## record 0")
            for field_name, field_desc in self.output_schema.items():
                prompt_parts.append(f"{field_name}=value")
        else:
            prompt_parts.append("## record 0")
            prompt_parts.append("result=success")
        prompt_parts.append("```")
        prompt_parts.append("")

        # Multiple records instruction
        prompt_parts.append("If there are multiple results, use separate records with increasing indices:")
        prompt_parts.append("```markdown")
        prompt_parts.append("## record 0")
        prompt_parts.append("name=first_result")
        prompt_parts.append("value=some_value")
        prompt_parts.append("")
        prompt_parts.append("## record 1")
        prompt_parts.append("name=second_result")
        prompt_parts.append("value=another_value")
        prompt_parts.append("```")
        prompt_parts.append("")

        # Important instructions
        prompt_parts.append("## Important Instructions")
        prompt_parts.append("1. Always follow the exact markdown-kv format shown above")
        prompt_parts.append("2. Do not include any other text or formatting in your response")
        prompt_parts.append("3. Only return the markdown-kv formatted result")

        return "\n".join(prompt_parts)

    def _parse_markdown_kv(self, text: str) -> Union[Dict[str, str], List[Dict[str, str]]]:
        """
        Parse markdown-kv formatted text into dictionary or list of dictionaries.

        Args:
            text: The markdown-kv formatted text to parse

        Returns:
            Union[Dict[str, str], List[Dict[str, str]]]: Parsed result as dict or list of dicts
        """
        # Remove code block markers if present
        text = re.sub(r'```.*?\n', '', text, flags=re.DOTALL)
        text = re.sub(r'```', '', text)

        # Split by record headers
        record_pattern = r'## record \d+'
        record_sections = re.split(record_pattern, text)

        # Remove empty sections
        record_sections = [section.strip() for section in record_sections if section.strip()]

        results = []
        for section in record_sections:
            # Parse key-value pairs
            kv_dict = {}
            lines = section.strip().split('\n')
            for line in lines:
                if '=' in line:
                    key, value = line.split('=', 1)
                    kv_dict[key.strip()] = value.strip()
            if kv_dict:  # Only add non-empty dictionaries
                results.append(kv_dict)

        # Return single dict if only one record, otherwise return list
        if len(results) == 1:
            return results[0]
        return results

    def _format_input_parameters(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
        """
        Format input parameters for the user prompt.

        Args:
            input_data: Input parameters as dict or list of dicts

        Returns:
            str: Formatted string representation of input parameters
        """
        if isinstance(input_data, list):
            if not input_data:
                return "No input parameters provided."

            formatted_parts = ["Input parameters (multiple records):"]
            for i, record in enumerate(input_data):
                formatted_parts.append(f"\nRecord {i}:")
                for key, value in record.items():
                    formatted_parts.append(f"  - {key}: {value}")
            return "\n".join(formatted_parts)
        elif isinstance(input_data, dict):
            if not input_data:
                return "No input parameters provided."

            formatted_parts = ["Input parameters:"]
            for key, value in input_data.items():
                formatted_parts.append(f"  - {key}: {value}")
            return "\n".join(formatted_parts)
        else:
            return f"Input parameters: {input_data}"

    def __call__(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, str], List[Dict[str, str]]]:
        """
        Execute the task with the given parameters.

        Args:
            input_data: Input parameters for the task (dict or list of dicts)

        Returns:
            Union[Dict[str, str], List[Dict[str, str]]]: The parsed result from the LLM
        """
        # Build the user prompt with input parameters
        user_prompt = self._format_input_parameters(input_data)

        # Build messages for LLM
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_prompt}
        ]

        # Call LLM
        response = self.alien_llm_chat_fn(messages, self.sampling_params)

        # Parse and return result
        return self._parse_markdown_kv(response["content"])
