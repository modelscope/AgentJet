from loguru import logger
from ajet import AjetTuner, Workflow, WorkflowOutput, WorkflowTask
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat import ChatCompletionMessageToolCall
from textwrap import dedent

import json
import asyncio
import requests
from langchain.agents import create_agent


# ------------------------------------------------------
# Simple version - no tool call
# ------------------------------------------------------


class ExampleMathLearn(Workflow):
    name: str = "math_agent_workflow"
    system_prompt: str = dedent(
        """
        You are an agent specialized in solving math problems.
        Please solve the math problem given to you.
        You can write and execute Python code to perform calculation or verify your answer.
        You should return your final answer within \\boxed{{}}.
    """
    )

    async def execute(self, workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:  # type: ignore
        # tuner to api key
        url_and_apikey = tuner.as_oai_baseurl_apikey()
        base_url = url_and_apikey.base_url
        api_key = url_and_apikey.api_key

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url=base_url,
            api_key=lambda: api_key,
        )
        agent = create_agent(
            model=llm,
            system_prompt=self.system_prompt,
        )

        # take out query
        query = workflow_task.task.main_query

        response = agent.invoke(
            {
                "messages": [{"role": "user", "content": query}],
            }
        )

        final_answer = response["messages"][-1].content
        return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})
