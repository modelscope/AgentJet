# -*- coding: utf-8 -*-
"""
AIME Math Agent using the OpenAI Responses API (`client.responses.create`).

This is the Responses-API twin of `agent_run_v3.py`. It reuses v3's:
  - PythonExecutor / PythonTool (sandbox tool implementation)
  - PYTHON_TOOL_SCHEMA ( Responses-API variant declared below — same shape,
    different envelope: `{"type":"function","name":...,"parameters":...}`
    instead of the chat-completions `{"type":"function","function":{...}}` )
  - extract_tool_calls / compute_score / compute_reward (reward logic)

What changes vs v3:
  - LLM client uses `client.responses.create(...)` (POST /v1/responses)
  - Multi-turn history is carried as Responses-API `input` items:
      * user text      → {"role":"user","content":...}
      * assistant text → {"type":"message","role":"assistant","content":[{"type":"output_text","text":...}]}
      * tool call      → {"type":"function_call","call_id":...,"name":...,"arguments":...}
      * tool output    → {"type":"function_call_output","call_id":...,"output":...}
  - Tool calls come from `response.output` items of type `function_call`
    (not from `choices[0].message.tool_calls` as in chat completions)

This file deliberately stays a near-clone of agent_run_v3.py so that the
ablation comparison (Responses vs Chat Completions) only varies the LLM
client — same model, same prompt, same tool, same reward.
"""

import asyncio
import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from textwrap import dedent
from typing import Protocol
from uuid import uuid4

from openai import OpenAI

from ajet.copilot.job import AgentJetJob
from ajet.schema.task import Task, WorkflowOutput
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.utils.message_utils import is_token_overflow_message

# Reuse v3's tool implementation + reward logic verbatim.
from tutorial.opencode_build_aime.agent_run_v3 import (
    TIMEOUT_EXIT_CODE,
    ProcessExecuteResult,
    _run_as_pg,
    PythonExecutor,
    PythonTool,
    compute_score,
    compute_reward,
)


class AimeAgentConfigLike(Protocol):
    model: str
    max_response_length: int


# ---------------------------------------------------------------------------
# Responses-API function tool schema.
# Identical metadata to v3's PYTHON_TOOL_SCHEMA, but with the flat envelope
# the Responses API expects: {"type":"function","name":...,"parameters":...}
# (vs chat completions' {"type":"function","function":{"name":...,"parameters":...}}).
# ---------------------------------------------------------------------------

PYTHON_TOOL_SCHEMA_RESPONSES = {
    "type": "function",
    "name": "python_code_with_standard_io",
    "description": (
        "Execute Python code with standard input and capture standard output. "
        "This function takes a Python code string and an input string, provides "
        "the input string through standard input (stdin) to the code, and "
        "captures and returns any output produced through standard output (stdout). "
        "If the executed code raises an exception, the error message will be "
        "captured and returned instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "A string containing Python code to be executed. The code can read from standard input using the input() function.",
            },
            "input": {
                "type": "string",
                "description": "A string that will be provided as standard input to the code when it calls input().",
            },
        },
        "required": ["code", "input"],
    },
    "strict": False,
}


def _new_call_id() -> str:
    return f"call_{uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Agent loop — Responses-API variant.
# ---------------------------------------------------------------------------

class ResponsesAgentLoop:
    """Multi-turn agent loop driven by client.responses.create()."""

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
                return text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                return "(truncated)..." + text[-self.max_tool_response_length:]
            else:
                length = self.max_tool_response_length // 2
                return text[:length] + "...(truncated)..." + text[-length:]
        return text

    async def run(self, messages: list[dict], sampling_params: dict) -> tuple[str, list[dict], int]:
        """Run the Responses-API agent loop.

        `messages` follows chat-completions shape (role/content) on entry; we
        convert it to a Responses `input` list once at the start. After that
        we keep appending Responses input items as the conversation grows.
        Returns (final_text, chat_completions_messages_snapshot, num_turns).
        """
        history_tool_calls = []
        total_tokens_used = 0
        assistant_turns = 0
        all_response_text = []

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

        # Seed the Responses-API `input` list from chat-style messages.
        # We drop any inbound system message and rely on `instructions` instead.
        input_items: list[dict] = []
        user_messages_seen = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                continue
            if role == "user":
                input_items.append({"role": "user", "content": content if isinstance(content, str) else str(content)})
                user_messages_seen += 1
            elif role == "assistant":
                # Represent prior assistant turns as message items carrying output_text.
                text = content if isinstance(content, str) else str(content)
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                })

        if user_messages_seen == 0:
            # Defensive: ensure there's at least one user turn to respond to.
            input_items.append({"role": "user", "content": "(no input)"})

        # Snapshot of the conversation in chat-completions shape, kept in sync
        # so callers that expect v3's return shape still work.
        chat_snapshot: list[dict] = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") != "system":
                chat_snapshot.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        while True:
            assistant_turns += 1

            max_tokens = sampling_params.get("max_tokens", 4096)
            if total_tokens_used + max_tokens > self.max_response_length:
                max_tokens = self.max_response_length - total_tokens_used
                if max_tokens <= 0:
                    break

            response = self.client.responses.create(
                model=sampling_params.get("model", "gpt-4o"),
                input=input_items,
                instructions=system_prompt,
                tools=self.tool_schemas if self.tool_schemas else None,
                tool_choice="auto" if self.tool_schemas else None,
                temperature=sampling_params.get("temperature", 1.0),
                max_output_tokens=max_tokens,
            )

            # Aggregate assistant text via the SDK convenience property.
            response_text = response.output_text or ""
            all_response_text.append(response_text)
            total_tokens_used += (
                response.usage.total_tokens if response.usage else 0
            )

            # Snapshot-update chat history with this assistant turn's text.
            chat_snapshot.append({"role": "assistant", "content": response_text})

            # AgentJet signals prompt overflow via a synthetic assistant message.
            if is_token_overflow_message(response_text):
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": response_text, "annotations": []}],
                })
                break

            # Collect function_call items from this response.
            function_calls = [
                item for item in (response.output or []) if getattr(item, "type", None) == "function_call"
            ]

            # First, append the assistant turn as a Responses input item so the
            # next round sees its full output (text + tool calls). The OpenAI
            # Responses API accepts the *same* output items we just received as
            # input on the next turn — that's the canonical multi-turn pattern.
            for item in response.output:
                # Re-serialize each output item back into the input list as-is.
                try:
                    input_items.append(item.model_dump())
                except Exception:
                    # Fallback: build a minimal message item if dumping fails.
                    if getattr(item, "type", None) == "message":
                        input_items.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": response_text, "annotations": []}],
                        })

            if function_calls:
                for fc in function_calls:
                    history_tool_calls.append({
                        "name": fc.name,
                        "arguments": json.loads(fc.arguments) if fc.arguments else {},
                    })

            if assistant_turns >= self.max_assistant_turns:
                break

            if not function_calls:
                break

            if total_tokens_used >= self.max_response_length:
                break

            # Execute each tool call and append a function_call_output item.
            for fc in function_calls:
                tool_name = fc.name
                try:
                    tool_args = json.loads(fc.arguments) if fc.arguments else {}
                except json.JSONDecodeError:
                    tool_args = {}

                call_id = fc.call_id or fc.id or _new_call_id()

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
                    total_tokens_used += len(truncated_text)

                    chat_snapshot.append({
                        "role": "tool",
                        "content": truncated_text,
                        "tool_call_id": call_id,
                        "name": tool_name,
                    })

                    input_items.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": truncated_text,
                    })
                else:
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": f"Error: Unknown tool {tool_name}",
                    })

                if total_tokens_used >= self.max_response_length:
                    break

        final_response = "\n".join(all_response_text)
        return final_response, chat_snapshot, assistant_turns


# ---------------------------------------------------------------------------
# Agent execution (drop-in replacement for v3's execute_agent).
# ---------------------------------------------------------------------------

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

    tool_schemas = [PYTHON_TOOL_SCHEMA_RESPONSES]
    tool_instances = {"python_code_with_standard_io": PythonTool(timeout=30)}

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=300,
    )

    agent_loop = ResponsesAgentLoop(
        client=client,
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
            "api": "responses",
        },
    )


def run_agent_and_compute_reward(
    task: Task,
    base_url: str,
    api_key: str,
    ajet_job: AgentJetJob,
) -> WorkflowOutput:
    api_baseurl_key = OpenaiBaseUrlAndApiKey(base_url=base_url, api_key=api_key)
    return execute_agent(task, api_baseurl_key, ajet_job)


if __name__ == "__main__":
    print("Testing agent_run_responses.py ...")
    print("Multi-turn agent (Responses API) with Python code execution tool")
