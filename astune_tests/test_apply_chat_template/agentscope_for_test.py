from agentscope.message import Msg
from beast_logger import print_dict
from loguru import logger
from pydantic import BaseModel, Field

from astune import ModelTuner, Workflow, WorkflowOutput, WorkflowTask
from astune.utils.testing_utils import GoodbyeException, TestFailException


def extract_final_answer(result) -> str:
    """Extract the final answer from the agent's response."""
    try:
        if (
            hasattr(result, "metadata")
            and isinstance(result.metadata, dict)
            and "result" in result.metadata
        ):
            return result.metadata["result"]
        if hasattr(result, "content"):
            if isinstance(result.content, dict) and "result" in result.content:
                return result.content["result"]
            return str(result.content)
        return str(result)
    except Exception as e:
        logger.warning(f"Extract final answer error: {e}. Raw: {result}")
        return str(result)


class FinalResult(BaseModel):
    result: str = Field(
        description="Your solution of the given math problem. Put your final answer in boxed format, e.g., \\boxed{42}"
    )


system_prompt = """
You are an agent specialized in solving math problems with tools.
Please solve the math problem given to you.
You can write and execute Python code to perform calculation or verify your answer.
You should return your final answer within \\boxed{{}}.
"""


class ExampleMathLearn(Workflow):
    name: str = "math_agent_workflow"

    async def agentscope_execute(
        self, workflow_task: WorkflowTask, model_tuner: ModelTuner
    ) -> WorkflowOutput:
        from agentscope.agent import ReActAgent
        from agentscope.formatter import DashScopeChatFormatter
        from agentscope.memory import InMemoryMemory
        from agentscope.tool import Toolkit, execute_python_code

        query = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
        self.toolkit = Toolkit()
        self.toolkit.register_tool_function(execute_python_code)
        self.agent = ReActAgent(
            name="math_react_agent",
            sys_prompt=system_prompt,
            model=model_tuner,
            formatter=DashScopeChatFormatter(),
            toolkit=self.toolkit,
            memory=InMemoryMemory(),
        )
        self.agent.set_console_output_enabled(False)
        msg = Msg("user", query, role="user")
        result = await self.agent.reply(msg, structured_model=FinalResult)
        final_answer = extract_final_answer(result)
        return WorkflowOutput(reward=None, metadata={"final_answer": final_answer})


class TEST_LAMBDA(object):
    def __init__(self):
        pass

    def __call__(self, key, value):
        if key == "prompt_text":
            expected = '<|im_start|>system\n\nYou are an agent specialized in solving math problems with tools.\nPlease solve the math problem given to you.\nYou can write and execute Python code to perform calculation or verify your answer.\nYou should return your final answer within \\boxed{{}}.\n\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "execute_python_code", "parameters": {"properties": {"code": {"description": "The Python code to be executed.", "type": "string"}, "timeout": {"default": 300, "description": "The maximum time (in seconds) allowed for the code to run.", "type": "number"}}, "required": ["code"], "type": "object"}, "description": "Execute the given python code in a temp file and capture the return\\n\\ncode, standard output and error. Note you must `print` the output to get\\nthe result, and the tmp file will be removed right after the execution."}}\n{"type": "function", "function": {"name": "generate_response", "parameters": {"properties": {"response": {"description": "Your response to the user.", "type": "string"}, "result": {"description": "Your solution of the given math problem. Put your final answer in boxed format, e.g., \\\\boxed{42}", "type": "string"}}, "required": ["response", "result"], "type": "object"}, "description": "Generate a response. Note only the input argument `response` is\\n\\nvisible to the others, you should include all the necessary\\ninformation in the `response` argument."}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nA robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?<|im_end|>\n<|im_start|>assistant\n'
            if value != expected:
                print_dict(
                    {
                        "expected": expected,
                        "value": value,
                        "key": key,
                    }
                )
                raise TestFailException("Prompt text does not match expected value.")
        elif key == "parsed_tool_calls":
            if len(value) > 0:
                raise GoodbyeException("Test passed!")
            else:
                raise TestFailException("No tool calls parsed when some were expected.")
        else:
            raise TestFailException(f"Unrecognized test key: {key}")

    def mock(self, key):
        if key == "mock_decoded_text":
            return 'To find the total number of bolts needed for the robe, we need to add the number of blue fiber bolts to the number of white fiber bolts. According to the problem, a robe takes 2 bolts of blue fiber and half that much white fiber. Therefore, the number of white fiber bolts is half of 2, which is 1. \n\nLet\'s calculate the total number of bolts.\n<tool_call>\n{"name": "execute_python_code", "arguments": {"code": "blue_fiber_bolts = 2\\nwhite_fiber_bolts = blue_fiber_bolts / 2\\ntotal_bolts = blue_fiber_bolts + white_fiber_bolts\\ntotal_bolts"}}\n</tool_call>\nresponse = "The total number of bolts needed for the robe is {}."\nresult = "\\\\boxed{3}"\n<tool_call>\n{"name": "generate_response", "arguments": {"response": "The total number of bolts needed for the robe is 3.", "result": "\\\\boxed{3}"}}\n</tool_call><|im_end|>'
        else:
            raise TestFailException(f"Unrecognized mock key: {key}")
