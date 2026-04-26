# -*- coding: utf-8 -*-
"""
AIME Math Agent with Multi-turn Python Code Execution.

Based on:
- rstar2_agent/rollout/rstar2_agent_loop.py (agent loop logic)
- rstar2_agent/tools/code_judge_tool.py (Python tool implementation)
- code-judge/app/libs/executors/executor.py (subprocess-based execution)

Features:
- Multi-turn agent loop with history tracking
- Subprocess-based Python code execution with timeout
- OpenAI SDK for API calls
"""

import asyncio
import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from textwrap import dedent
from uuid import uuid4

from openai import OpenAI

from ajet.copilot.job import AgentJetJob
from ajet.schema.task import Task, WorkflowOutput
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey


TIMEOUT_EXIT_CODE = -101


# ==================== Python Code Execution Tool ====================
# Adapted from code-judge/app/libs/executors/executor.py

@dataclass
class ProcessExecuteResult:
    stdout: str
    stderr: str
    exit_code: int
    cost: float

    @property
    def success(self) -> bool:
        return self.exit_code == 0


def _run_as_pg(args, input=None, capture_output=False, timeout=None, check=False, **kwargs):
    from subprocess import Popen, PIPE, TimeoutExpired, CalledProcessError, CompletedProcess

    kwargs['start_new_session'] = True
    if input is not None:
        if kwargs.get('stdin') is not None:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = PIPE

    if capture_output:
        if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
            raise ValueError('stdout and stderr arguments may not both be used with capture_output.')
        kwargs['stdout'] = PIPE
        kwargs['stderr'] = PIPE

    with Popen(args, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except TimeoutExpired as exc:
            try:
                import os
                os.killpg(process.pid, 9)
            except Exception:
                pass
            process.wait()
            raise
        except Exception:
            try:
                import os
                os.killpg(process.pid, 9)
            except Exception:
                pass
            raise
        try:
            import os
            os.killpg(process.pid, 9)
        except Exception:
            pass
        retcode = process.poll()
        if check and retcode:
            raise CalledProcessError(retcode, process.args, output=stdout, stderr=stderr)
    return CompletedProcess(process.args, retcode, stdout, stderr)


class PythonExecutor:
    def __init__(self, timeout: int = 30, memory_limit_mb: int = 512):
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb

    def execute(self, code: str, stdin: str = "") -> ProcessExecuteResult:
        pre_template = dedent(f"""\
            import signal
            import resource
            import os
            import sys
            import time

            os.environ['OPENBLAS_NUM_THREADS'] = '1'

            def _exec_set_alarm_timeout(timeout):
                signal.signal(signal.SIGALRM, _exec_time_exceeded)
                signal.alarm(timeout)

            def _exec_time_exceeded(*_):
                print('Suicide from timeout.', flush=True)
                try:
                    os.killpg(0, 9)
                except Exception:
                    pass
                os._exit({TIMEOUT_EXIT_CODE})

            def _exec_set_max_runtime(seconds):
                soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
                resource.setrlimit(resource.RLIMIT_CPU, (seconds, hard))

            _exec_set_alarm_timeout({self.timeout})
            _exec_set_max_runtime({self.timeout})

            _exec_time_start = time.perf_counter()
            """)
        post_template = dedent("""\
            _exec_time_end = time.perf_counter()
            _exec_duration = _exec_time_end - _exec_time_start
            """)

        with tempfile.TemporaryDirectory() as tmp_path:
            source_path = f"{tmp_path}/source.py"
            with open(source_path, 'w') as f:
                f.write(pre_template)
                f.write("\n")
                f.write(code)
                f.write("\n")
                f.write(post_template)
                f.flush()

            time_start = time.perf_counter()
            try:
                std_input = stdin.encode() if stdin else None
                result = _run_as_pg(
                    ["python3", source_path],
                    cwd=tmp_path,
                    shell=False,
                    check=False,
                    capture_output=True,
                    timeout=self.timeout + 1,
                    input=std_input
                )
                stdout = result.stdout.decode()
                stderr = result.stderr.decode()
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                stdout = ""
                stderr = "TimeoutExpired"
                exit_code = TIMEOUT_EXIT_CODE

            time_end = time.perf_counter()

            return ProcessExecuteResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                cost=time_end - time_start
            )


class PythonTool:
    def __init__(self, timeout: int = 30, max_tool_response_length: int = 8000):
        self.executor = PythonExecutor(timeout=timeout)
        self.max_tool_response_length = max_tool_response_length
        self._instance_dict = {}

    async def create(self, instance_id: str = None, **kwargs) -> tuple[str, dict]:
        if instance_id is None:
            instance_id = str(uuid4())
        history_tool_calls = kwargs.get("history_tool_calls", [])
        self._instance_dict[instance_id] = {
            "history_tool_calls": history_tool_calls,
        }
        return instance_id, {"text": ""}

    async def execute(self, instance_id: str, parameters: dict) -> tuple[dict, float, dict]:
        code = parameters.get("code", "")
        stdin = parameters.get("input", "")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.executor.execute(code, stdin)
        )

        if result.exit_code == TIMEOUT_EXIT_CODE:
            text = "Error: Execution timed out"
        elif not result.success:
            text = f"Error: {result.stderr}"
        else:
            text = result.stdout

        if len(text) > self.max_tool_response_length:
            text = text[:self.max_tool_response_length] + "...(truncated)"

        return {"text": text}, result.cost, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


# ==================== Tool Schema ====================

PYTHON_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_code_with_standard_io",
        "description": "Execute Python code with standard input and capture standard output. This function takes a Python code string and an input string, provides the input string through standard input (stdin) to the code, and captures and returns any output produced through standard output (stdout). If the executed code raises an exception, the error message will be captured and returned instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "A string containing Python code to be executed. The code can read from standard input using the input() function."
                },
                "input": {
                    "type": "string",
                    "description": "A string that will be provided as standard input to the code when it calls input()."
                }
            },
            "required": ["code", "input"]
        }
    }
}


# ==================== Tool Parser ====================

def extract_tool_calls(response_content: str) -> tuple[str, list[dict]]:
    tool_calls = []
    pattern = r'<tool_call>\s*\n?({.*?})\n?\s*</tool_call>'
    matches = re.findall(pattern, response_content, re.DOTALL)

    for match in matches:
        try:
            tool_data = json.loads(match)
            tool_calls.append({
                "name": tool_data.get("name"),
                "arguments": tool_data.get("arguments", {})
            })
        except json.JSONDecodeError:
            continue

    if not tool_calls:
        no_tool_pattern = r'\[TOOL_CALLS\]\s*\n?({.*?})\n?\s*\[/TOOL_CALLS\]'
        no_tool_matches = re.findall(no_tool_pattern, response_content, re.DOTALL)
        for match in no_tool_matches:
            try:
                tool_data = json.loads(match)
                tool_calls.append({
                    "name": tool_data.get("name"),
                    "arguments": tool_data.get("arguments", {})
                })
            except json.JSONDecodeError:
                continue

    return response_content, tool_calls


# ==================== Agent Loop ====================

class AgentLoop:
    def __init__(
        self,
        client: OpenAI,
        tool_schemas: list[dict],
        tool_instances: dict,
        max_assistant_turns: int = 5,
        max_response_length: int = 8192,
        max_tool_response_length: int = 8000,
        tool_response_truncate_side: str = "right",
    ):
        self.client = client
        self.tool_schemas = tool_schemas
        self.tool_instances = tool_instances
        self.max_assistant_turns = max_assistant_turns
        self.max_response_length = max_response_length
        self.max_tool_response_length = max_tool_response_length
        self.tool_response_truncate_side = tool_response_truncate_side

    def _truncate_response(self, text: str) -> str:
        if len(text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                return text[:self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                return "(truncated)..." + text[-self.max_tool_response_length:]
            else:
                length = self.max_tool_response_length // 2
                return text[:length] + "...(truncated)..." + text[-length:]
        return text

    async def run(self, messages: list[dict], sampling_params: dict) -> tuple[str, list[dict], int]:
        history_tool_calls = []
        total_tokens_used = 0

        user_turns, assistant_turns = 0, 0
        all_response_content = []

        system_prompt = dedent("""\
            You are an expert mathematician specialized in solving challenging math competition problems.

            You have access to a Python code execution tool. Use it to:
            1. Perform calculations and verify your answers
            2. Run code when you need precise computation
            3. Test your hypotheses before giving final answers

            Instructions:
            1. Think through the problem step by step
            2. Use the python_code_with_standard_io tool when you need to execute code
            3. Show your reasoning clearly
            4. Put your final numerical answer inside \\boxed{} at the end

            For each function call, return a json object within <tool_call></tool_call> XML tags:
            <tool_call>
            {"name": "python_code_with_standard_io", "arguments": {"code": "your python code", "input": "stdin input if needed"}}
            </tool_call>""")

        formatted_messages = [msg for msg in messages if msg.get("role") != "system"]
        if not any(msg.get("role") == "system" for msg in messages):
            formatted_messages.insert(0, {"role": "system", "content": system_prompt})

        while True:
            assistant_turns += 1

            max_tokens = sampling_params.get("max_tokens", 4096)
            if total_tokens_used + max_tokens > self.max_response_length:
                max_tokens = self.max_response_length - total_tokens_used
                if max_tokens <= 0:
                    break

            response = self.client.chat.completions.create(
                model=sampling_params.get("model", "gpt-4o"),
                messages=formatted_messages,
                tools=self.tool_schemas if self.tool_schemas else None,
                tool_choice="auto" if self.tool_schemas else None,
                temperature=sampling_params.get("temperature", 1.0),
                max_tokens=max_tokens,
            )

            response_message = response.choices[0].message
            response_content = response_message.content or ""
            all_response_content.append(response_content)

            total_tokens_used += response.usage.total_tokens if response.usage else 0

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    history_tool_calls.append({
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments)
                    })

            formatted_messages.append({"role": "assistant", "content": response_content})

            if assistant_turns >= self.max_assistant_turns:
                break

            if not response_message.tool_calls:
                break

            if total_tokens_used >= self.max_response_length:
                break

            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if tool_name in self.tool_instances:
                    try:
                        tool_instance_id, _ = await self.tool_instances[tool_name].create(
                            history_tool_calls=history_tool_calls[:-1]
                        )
                        tool_response, _, _ = await self.tool_instances[tool_name].execute(
                            tool_instance_id, tool_args
                        )
                        await self.tool_instances[tool_name].release(tool_instance_id)
                    except Exception as e:
                        tool_response = {"text": f"Error executing tool: {e}"}

                    truncated_text = self._truncate_response(tool_response.get("text", ""))
                    formatted_messages.append({
                        "role": "tool",
                        "content": truncated_text,
                        "tool_call_id": tool_call.id,
                    })
                    total_tokens_used += len(truncated_text)

                    if total_tokens_used >= self.max_response_length:
                        break
                else:
                    formatted_messages.append({
                        "role": "tool",
                        "content": f"Error: Unknown tool {tool_name}",
                        "tool_call_id": tool_call.id,
                    })

            user_turns += 1

        final_response = "\n".join(all_response_content)
        return final_response, formatted_messages, assistant_turns


# ==================== rStar2-style Reward Functions ====================

def compute_score(model_output: str, ground_truth: str) -> float:
    try:
        from tutorial.opencode_build_aime.verl_reward_fn.prime_math import compute_score as prime_compute_score
        prime_score = prime_compute_score(model_output, ground_truth)[0]
        if prime_score:
            return 1.0
    except Exception:
        pass
    try:
        from tutorial.opencode_build_aime.verl_reward_fn.math_verify import compute_score as math_verify_compute_score
        math_verify_score = math_verify_compute_score(model_output, ground_truth)
        if math_verify_score:
            return 1.0
    except Exception:
        pass
    return 0.0


def compute_reward(solution_str: str, ground_truth: str) -> dict:
    score = compute_score(solution_str, ground_truth)
    correct = score == 1.0
    return {
        "score": score,
        "acc": correct,
        "pred": "",
    }


# ==================== Agent Execution ====================

def execute_agent(
    task: Task,
    api_baseurl_key: OpenaiBaseUrlAndApiKey,
    ajet_job: AgentJetJob,
) -> WorkflowOutput:
    base_url = api_baseurl_key.base_url
    api_key = api_baseurl_key.api_key

    query = task.main_query
    if query in ["Empty", "[not defined]", ""] or not query:
        prompt = task.metadata.get("prompt", [])
        if isinstance(prompt, list) and len(prompt) > 0:
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    query = msg.get("content", "")
                    break
            if not query and len(prompt) > 0:
                last_msg = prompt[-1]
                if isinstance(last_msg, dict):
                    query = last_msg.get("content", "")
                elif isinstance(last_msg, str):
                    query = last_msg
        elif isinstance(prompt, str):
            query = prompt

    ground_truth = task.metadata.get("ground_truth", "")
    if not ground_truth:
        ground_truth = task.metadata.get("answer", "")
    if not ground_truth:
        reward_model = task.metadata.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth", "")

    prompt = task.metadata.get("prompt", [])
    if isinstance(prompt, list) and len(prompt) > 0:
        messages = []
        for msg in prompt:
            if isinstance(msg, dict):
                messages.append(msg)
    else:
        messages = [
            {"role": "user", "content": query}
        ]

    tool_schemas = [PYTHON_TOOL_SCHEMA]
    tool_instances = {
        "python_code_with_standard_io": PythonTool(timeout=30),
    }

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=300,
    )

    agent_loop = AgentLoop(
        client=client,
        tool_schemas=tool_schemas,
        tool_instances=tool_instances,
        max_assistant_turns=5,
        max_response_length=ajet_job.max_response_length,
    )

    sampling_params = {
        "model": ajet_job.model,
    }

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    model_output, _, num_turns = loop.run_until_complete(
        agent_loop.run(messages, sampling_params)
    )

    reward_result = compute_reward(model_output, ground_truth)

    return WorkflowOutput(
        reward=reward_result["score"],
        metadata={
            "model_output": model_output,
            "ground_truth": ground_truth,
            "predicted": reward_result["pred"],
            "correct": reward_result["acc"],
            "num_turns": num_turns,
        }
    )


def run_agent_and_compute_reward(
    task: Task,
    base_url: str,
    api_key: str,
    ajet_job: AgentJetJob,
) -> WorkflowOutput:
    api_baseurl_key = OpenaiBaseUrlAndApiKey(
        base_url=base_url,
        api_key=api_key,
    )
    return execute_agent(task, api_baseurl_key, ajet_job)


if __name__ == "__main__":
    print("Testing agent_run_v3.py...")
    print("Multi-turn agent with Python code execution tool")
    print("Based on rstar2_agent/rollout/rstar2_agent_loop.py")
