import random
import time
import os
import asyncio
from textwrap import dedent
from loguru import logger
from agentscope.model import DashScopeChatModel

class RobustDashScopeChatModel(DashScopeChatModel):
    """
    A robust version of DashScopeChatModel that includes retry logic and multiple API key handling.
    This class extends the DashScopeChatModel from agentscope and adds:
    1. Support for multiple API keys separated by '|' in environment variables
    2. Automatic retry logic with backup API keys
    3. Error handling with appropriate logging
    """

    def __init__(
        self,
        model_name="qwen3-max",
        stream=False,
        max_try=4,
        **kwargs
    ):
        # Check for environment variables
        self._check_env_variables()

        # Parse API keys from environment variables
        self.regular_key_list = os.environ.get("DASHSCOPE_API_KEY", "").split("|")
        self.backup_key_list = os.environ.get("DASHSCOPE_API_KEY_BACKUP", "").split("|") if os.environ.get("DASHSCOPE_API_KEY_BACKUP") else []

        api_key = random.choice(self.regular_key_list)

        # Store retry parameters
        self.max_try = max_try

        # Initialize the parent class
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            stream=stream,
            **kwargs
        )

    def _check_env_variables(self):
        """Check if required environment variables are set."""
        if os.environ.get("DASHSCOPE_API_KEY") is None:
            raise RuntimeError(dedent("""
                Please set the DASHSCOPE_API_KEY environment variable.
                You can get the API keys from https://www.dashscope.com/.
                Example:
                export DASHSCOPE_API_KEY='sk-xxxxxx|sk-yyyyyy'
                export DASHSCOPE_API_KEY_BACKUP='sk-zzzzzz' (optional)
            """))

    async def __call__(self, messages, tools=None, tool_choice=None, structured_model=None, **kwargs):
        """
        Override the __call__ method to add retry logic and API key rotation.

        Args:
            messages: The messages to send to the model
            tools: Optional list of tools
            tool_choice: Optional tool choice
            structured_model: Optional structured model
            **kwargs: Additional arguments to pass to the API

        Returns:
            The response from the model

        Raises:
            RuntimeError: If all retry attempts fail
        """
        for n_try in range(self.max_try):
            try:
                # Select API key based on retry attempt
                if n_try < self.max_try // 2:
                    # For first half of attempts, use regular keys
                    self.api_key = random.choice(self.regular_key_list)
                elif n_try == self.max_try // 2 and self.backup_key_list:
                    # At middle attempt, try backup key if available
                    self.api_key = random.choice(self.backup_key_list)
                else:
                    # For remaining attempts, use any available key
                    self.api_key = random.choice(self.regular_key_list + self.backup_key_list)

                # Call the parent class's __call__ method
                response = await super().__call__(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    structured_model=structured_model,
                    **kwargs
                )
                return response

            except Exception as e:
                logger.bind(exception=True).exception(f"Error calling DashScope API: {e}")
                time.sleep(5)  # Wait before retrying
                print(f"Error calling DashScope API: {e}, retrying ({n_try + 1}/{self.max_try})...")

        # If all attempts fail
        raise RuntimeError(f"Failed to get response from DashScope API after {self.max_try} attempts")
