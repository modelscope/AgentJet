from loguru import logger
from ajet import AjetTuner, Workflow, WorkflowOutput, WorkflowTask
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat import ChatCompletionMessageToolCall
from textwrap import dedent

import json
import asyncio
import requests


# ------------------------------------------------------
# Simple version - no tool call
# ------------------------------------------------------


class ExampleMathLearn(Workflow):

    name: str = "math_agent_workflow"
    system_prompt: str = dedent("""
        You are an agent specialized in solving math problems.
        Please solve the math problem given to you.
        You can write and execute Python code to perform calculation or verify your answer.
        You should return your final answer within \\boxed{{}}.
    """)

    async def execute(self, workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:   # type: ignore
        # tuner to api key
        url_and_apikey = tuner.as_oai_baseurl_apikey()
        base_url = url_and_apikey.base_url
        api_key = url_and_apikey.api_key

        # take out query
        query = workflow_task.task.main_query

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]

        # use raw http requests (non-streaming) to get response
        response = requests.post(
             f"{base_url}/chat/completions",
             json={
                 "model": "whatever", # Of course, this `model` field will be ignored.
                 "messages": messages,
             },
             headers={
                 "Authorization": f"Bearer {api_key}"
             }
        )
        final_answer = response.json()['choices'][0]['message']['content']
        return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})





