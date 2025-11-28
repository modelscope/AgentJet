import os
import random
import time
from textwrap import dedent

from loguru import logger
from openai import OpenAI


def construct_alien_llm_chat_fn(alien_llm_model, alien_llm_response_length):
    def alien_llm_chat_fn(messages, sampling_params_override={}, request_id=""):
        max_try = 4
        alien_model_name = alien_llm_model
        alien_model_response_length = alien_llm_response_length

        if (
            os.environ.get("DASHSCOPE_API_KEY") is None
            or os.environ.get("DASHSCOPE_API_KEY_BACKUP") is None
        ):
            raise RuntimeError(
                dedent(
                    """
                Please set the DASHSCOPE_API_KEY and DASHSCOPE_API_KEY_BACKUP environment variables.
                You can get the API keys from https://www.dashscope.com/.
                Example:
                export DASHSCOPE_API_KEY='sk-xxxxxx|sk-yyyyyy'
                export DASHSCOPE_API_KEY_BACKUP='sk-zzzzzz'
            """
                )
            )

        regular_key_list = os.environ.get("DASHSCOPE_API_KEY")
        backup_key_list = os.environ.get("DASHSCOPE_API_KEY_BACKUP")
        if regular_key_list is not None:
            regular_key_list = regular_key_list.split("|")
        else:
            raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set.")
        if backup_key_list is not None:
            backup_key_list = backup_key_list.split("|")
        else:
            backup_key_list = []

        for n_try in range(max_try):
            try:
                if n_try < max_try // 2:
                    api_key = random.choice(regular_key_list)
                elif n_try == max_try // 2:
                    api_key = random.choice(backup_key_list)
                else:
                    api_key = random.choice(regular_key_list + backup_key_list)
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                sampling_params = dict(
                    n=1,
                    max_completion_tokens=alien_model_response_length,
                    temperature=0.0,
                )
                sampling_params.update(sampling_params_override)
                completion = client.chat.completions.create(
                    model=alien_model_name,
                    messages=messages,
                    extra_body=sampling_params,
                )
                message = completion.choices[0].message.model_dump(
                    exclude_unset=True, exclude_none=True
                )
                if "content" not in message:
                    message["content"] = ""
                return {"role": message["role"], "content": message["content"]}
            except Exception as e:
                logger.bind(exception=True).exception(f"Error calling alien llm: {e}")
                time.sleep(5)
                print(f"Error calling alien llm: {e}, retrying...")
        raise RuntimeError(f"Failed to get response from alien llm after {max_try} attempts")

    return alien_llm_chat_fn
