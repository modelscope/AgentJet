# -*- coding: utf-8 -*-
"""
AIME Math Agent (Raw HTTP variant).

Same multi-turn Python-code-execution agent as `agent_run_v3.py`, but the LLM call
is issued with a bare `requests` POST to the OpenAI-compatible `/chat/completions`
endpoint instead of the OpenAI SDK. The Python sandbox tool, tool schema, and reward
function are reused as-is from `agent_run_v3.py` so the only difference vs the
baseline is the client framework that talks to the AgentJet swarm.

Exposes `execute_agent(task, api_baseurl_key, ajet_job) -> WorkflowOutput`, matching
agent_run_v3 so it is a drop-in for the swarm trainer.
"""

import asyncio
import json
from textwrap import dedent

import requests

from ajet.schema.task import Task, WorkflowOutput
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.utils.message_utils import is_token_overflow_message

# Reuse the validated sandbox tool, tool schema, and reward from the OpenAI baseline.
from tutorial.opencode_build_aime.agent_run_v3 import (
    PythonTool,
    PYTHON_TOOL_SCHEMA,
    compute_reward,
    AimeAgentConfigLike,
)


SYSTEM_PROMPT = dedent("""\
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


class AgentLoop:
    """Mirror of agent_run_v3.AgentLoop, issuing the chat-completion via raw HTTP."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        tool_schemas: list[dict],
        tool_instances: dict,
        max_assistant_turns: int = 5,
        max_response_length: int = 8192,
        max_tool_response_length: int = 8000,
        tool_response_truncate_side: str = "right",
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
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

    def _chat_completion(self, messages: list[dict], max_tokens: int):
        """Raw HTTP chat-completion. Returns (content, tool_calls, total_tokens)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": self.tool_schemas if self.tool_schemas else None,
            "tool_choice": "auto" if self.tool_schemas else None,
            "temperature": 1.0,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        tool_calls = []
        for tc in (message.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": args})
        usage = data.get("usage") or {}
        return content, tool_calls, usage.get("total_tokens", 0) or 0

    async def run(self, messages: list[dict], sampling_params: dict) -> tuple[str, list[dict], int]:
        history_tool_calls = []
        total_tokens_used = 0
        assistant_turns = 0
        all_response_content = []

        formatted_messages = [msg for msg in messages if msg.get("role") != "system"]
        if not any(msg.get("role") == "system" for msg in messages):
            formatted_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        while True:
            assistant_turns += 1

            max_tokens = sampling_params.get("max_tokens", 4096)
            if total_tokens_used + max_tokens > self.max_response_length:
                max_tokens = self.max_response_length - total_tokens_used
                if max_tokens <= 0:
                    break

            content, tool_calls, used = self._chat_completion(formatted_messages, max_tokens)
            all_response_content.append(content)
            total_tokens_used += used

            if is_token_overflow_message(content):
                formatted_messages.append({"role": "assistant", "content": content})
                break

            for tc in tool_calls:
                history_tool_calls.append({"name": tc["name"], "arguments": tc["arguments"]})

            formatted_messages.append({"role": "assistant", "content": content})

            if assistant_turns >= self.max_assistant_turns:
                break
            if not tool_calls:
                break
            if total_tokens_used >= self.max_response_length:
                break

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
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

                    truncated_text = self._truncate_response(tool_response.get("text", "")) or "(no output)"
                    formatted_messages.append({
                        "role": "tool",
                        "content": truncated_text,
                        "tool_call_id": tc["id"],
                    })
                    total_tokens_used += len(truncated_text)
                    if total_tokens_used >= self.max_response_length:
                        break
                else:
                    formatted_messages.append({
                        "role": "tool",
                        "content": f"Error: Unknown tool {tool_name}",
                        "tool_call_id": tc["id"],
                    })

        final_response = "\n".join(all_response_content)
        return final_response, formatted_messages, assistant_turns


def execute_agent(
    task: Task,
    api_baseurl_key: OpenaiBaseUrlAndApiKey,
    ajet_job: AimeAgentConfigLike,
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
        messages = [msg for msg in prompt if isinstance(msg, dict)]
    else:
        messages = [{"role": "user", "content": query}]

    tool_schemas = [PYTHON_TOOL_SCHEMA]
    tool_instances = {"python_code_with_standard_io": PythonTool(timeout=30)}

    agent_loop = AgentLoop(
        base_url=base_url,
        api_key=api_key,
        model=ajet_job.model,
        tool_schemas=tool_schemas,
        tool_instances=tool_instances,
        max_assistant_turns=5,
        max_response_length=ajet_job.max_response_length,
    )

    sampling_params = {"model": ajet_job.model}

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
            "framework": "rawhttp",
        },
    )


def run_agent_and_compute_reward(
    task: Task,
    base_url: str,
    api_key: str,
    ajet_job: AimeAgentConfigLike,
) -> WorkflowOutput:
    api_baseurl_key = OpenaiBaseUrlAndApiKey(base_url=base_url, api_key=api_key)
    return execute_agent(task, api_baseurl_key, ajet_job)
