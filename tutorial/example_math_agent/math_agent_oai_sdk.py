from loguru import logger
from ajet import Workflow, WorkflowOutput, WorkflowTask
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat import ChatCompletionMessageToolCall
from ajet.tuner_v2 import TunerV2
from textwrap import dedent

import json
import asyncio


# ------------------------------------------------------
# Simple version - no tool call
# ------------------------------------------------------


class ExampleMathLearn_Simple_NoToolCall(Workflow):
    name: str = "math_agent_workflow"
    system_prompt: str = dedent("""
        You are an agent specialized in solving math problems.
        Please solve the math problem given to you.
        You can write and execute Python code to perform calculation or verify your answer.
        You should return your final answer within \\boxed{{}}.
    """)

    async def execute(self, workflow_task: WorkflowTask, model_tuner: TunerV2) -> WorkflowOutput:   # type: ignore
        query = workflow_task.task.main_query
        client = model_tuner.as_raw_openai_sdk_client()

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
        reply_message: ChatCompletion = await client.chat.completions.create(messages=messages)
        final_answer = reply_message.choices[0].message.content
        return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})










# ------------------------------------------------------
# Tool use version
# ------------------------------------------------------

class ExampleMathLearn(Workflow):

    name: str = "math_agent_workflow"
    system_prompt: str = dedent("""
        You are an agent specialized in solving math problems with tools.
        Please solve the math problem given to you.
        You can write and execute Python code to perform calculation or verify your answer.
        You should return your final answer within \\boxed{{}}.
    """)
    available_functions: list = [
        {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": "Execute the given Python code in a temp file and capture the return code, standard output, and error. Note that you should print something or you will get empty return.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to be executed."
                        },
                        "timeout": {
                            "type": "number",
                            "description": "The maximum time (in seconds) allowed for the code to run.",
                            "default": 300
                        }
                    },
                    "required": ["code"]
                }
            }
        },
    ]

    async def execute(self, workflow_task: WorkflowTask, model_tuner: TunerV2) -> WorkflowOutput:   # type: ignore


        query = workflow_task.task.main_query
        client = model_tuner.as_raw_openai_sdk_client()

        # call 1: get response with tool call
        messages = [
            { "role": "system", "content": self.system_prompt },
            { "role": "user", "content": query }
        ]
        reply_message: ChatCompletion = await client.chat.completions.create(messages=messages, tools=self.available_functions)
        if (reply_message.choices[0].message.content):
            messages.append({
                "role": "assistant",
                "content": reply_message.choices[0].message.content
            })

        # If the model called a tool
        if (reply_message.choices[0].message) and (reply_message.choices[0].message.tool_calls):
            tool_calls: list[ChatCompletionMessageToolCall] = reply_message.choices[0].message.tool_calls
            for tool_call in tool_calls:
                if tool_call.function.name == "execute_python_code":
                    arguments = json.loads(tool_call.function.arguments)

                    def sync_wrapper():
                        import subprocess
                        import sys
                        process = subprocess.run(
                            [sys.executable, "-c", arguments["code"]],
                            timeout=arguments.get("timeout", 300),
                            capture_output=True,
                            text=True
                        )
                        return process.stdout

                    result = await asyncio.to_thread(sync_wrapper)
                    tool_result_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps({
                            "return_code": str(result),
                        })
                    }
                    messages.append(tool_result_message)

            # Step 3: Make a follow-up API call with the tool result
            final_response: ChatCompletion = await client.chat.completions.create(
                messages=messages,
            )
            final_stage_response = final_response.choices[0].message.content
        else:
            final_stage_response = reply_message.choices[0].message.content


        return WorkflowOutput(reward=None, metadata={"final_answer": final_stage_response})
