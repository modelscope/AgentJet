import random
import time
import os
from textwrap import dedent
from openai import OpenAI
from loguru import logger

def construct_alien_llm_chat_fn(config, rollout_config):
    def alien_llm_chat_fn(messages, request_id=""):
        max_try = 4
        alien_model_name = config.context_manager.context_template_alien_llm_model
        alien_model_response_length = config.context_manager.context_template_alien_model_response_length

        if os.environ.get("DASHSCOPE_API_KEY") is None or os.environ.get("DASHSCOPE_API_KEY_BACKUP") is None:
            raise RuntimeError(dedent("""
                Please set the DASHSCOPE_API_KEY and DASHSCOPE_API_KEY_BACKUP environment variables.
                You can get the API keys from https://www.dashscope.com/.
                Example:
                export DASHSCOPE_API_KEY='sk-xxxxxx|sk-yyyyyy'
                export DASHSCOPE_API_KEY_BACKUP='sk-zzzzzz'
            """))

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
                    api_key=random.choice(regular_key_list)
                elif n_try == max_try // 2:
                    api_key=random.choice(backup_key_list)
                else:
                    api_key=random.choice(regular_key_list + backup_key_list)
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                sampling_params = dict(
                    n=1,
                    max_completion_tokens=alien_model_response_length,
                )
                sampling_params["temperature"] = 0
                completion = client.chat.completions.create(
                    model=alien_model_name,
                    messages=messages,
                    extra_body=sampling_params
                )
                message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
                if "content" not in message: message["content"] = ""
                return {"role": message["role"], "content": message['content']}
            except Exception as e:
                logger.bind(exception=True).exception(f"Error calling alien llm: {e}")
                time.sleep(5)
                print(f"Error calling alien llm: {e}, retrying...")
        raise RuntimeError(f"Failed to get response from alien llm after {max_try} attempts")
    return alien_llm_chat_fn
