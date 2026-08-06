# -*- coding: utf-8 -*-
"""
AIME Math Agent using the Anthropic Messages API (`client.messages.create`).

This is the Messages-API twin of `agent_run_responses.py` / `agent_run_v3.py`.
It reuses v3's:
  - PythonExecutor / PythonTool (sandbox tool implementation)
  - compute_score / compute_reward (reward logic)

What changes vs v3 / responses:
  - LLM client uses `client.messages.create(...)` (POST /v1/messages)
  - Tool schema uses the Anthropic envelope:
      {"name":..., "description":..., "input_schema":...}
    (vs chat-completions' {"type":"function","function":{...}} and
       responses'      {"type":"function","name":...,"parameters":...})
  - Multi-turn history is carried as Anthropic messages:
      * user text      → {"role":"user","content": str}
      * assistant turn → {"role":"assistant","content":[ blocks ]}
                         (text blocks {"type":"text","text":...} and
                          tool_use blocks {"type":"tool_use","id":...,"name":...,"input":...})
      * tool output    → {"role":"user","content":[
                             {"type":"tool_result","tool_use_id":...,"content":...}]}
  - Tool calls come from `response.content` blocks of type `tool_use`.

Auth note: the AgentJet interchange server routes requests via a token in the
`authorization` header, but the Anthropic SDK sends its api_key as `x-api-key`.
The server's /v1/messages route accepts either header, so we pass the routing
token straight through as the SDK `api_key`.

This file deliberately stays a near-clone of agent_run_responses.py so that the
ablation comparison (Messages vs Chat Completions vs Responses) only varies the
LLM client — same model, same prompt, same tool, same reward.
"""

import asyncio
import json
from textwrap import dedent
from typing import Protocol

from anthropic import Anthropic

from ajet.copilot.job import AgentJetJob
from ajet.schema.task import Task, WorkflowOutput
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from ajet.utils.message_utils import is_token_overflow_message

# Reuse v3's tool implementation + reward logic verbatim.
from tutorial.opencode_build_aime.agent_run_v3 import (
    PythonTool,
    compute_reward,
)


class AimeAgentConfigLike(Protocol):
    model: str
    max_response_length: int


# ---------------------------------------------------------------------------
# Anthropic Messages-API function tool schema.
# Same metadata as v3's PYTHON_TOOL_SCHEMA, but the Anthropic envelope:
#   {"name":..., "description":..., "input_schema":...}
# ---------------------------------------------------------------------------

PYTHON_TOOL_SCHEMA_MESSAGES = {
    "name": "python_code_with_standard_io",
    "description": (
        "Execute Python code with standard input and capture standard output. "
        "This function takes a Python code string and an input string, provides "
        "the input string through standard input (stdin) to the code, and "
        "captures and returns any output produced through standard output (stdout). "
        "If the executed code raises an exception, the error message will be "
        "captured and returned instead."
    ),
    "input_schema": {
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
}


def _new_tool_use_id() -> str:
    import uuid
    return f"toolu_{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Agent loop — Anthropic Messages-API variant.
# ---------------------------------------------------------------------------

class MessagesAgentLoop:
    """Multi-turn agent loop driven by client.messages.create()."""

    def __init__(
        self,
        client: Anthropic,
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
        """Run the Anthropic Messages-API agent loop.

        `messages` follows chat-completions shape (role/content) on entry; we
        convert it to Anthropic messages once at the start (system goes to the
        top-level `system` param, not into the message list). Returns
        (final_text, chat_completions_messages_snapshot, num_turns) — the same
        return shape as v3 / responses so the trainer is agnostic to the API.
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

        # Seed the Anthropic messages list from chat-style input. System turns
        # are dropped here and sent via the top-level `system` param instead.
        anthropic_messages: list[dict] = []
        user_messages_seen = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                continue
            if role == "user":
                anthropic_messages.append({"role": "user", "content": content if isinstance(content, str) else str(content)})
                user_messages_seen += 1
            elif role == "assistant":
                text = content if isinstance(content, str) else str(content)
                anthropic_messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

        if user_messages_seen == 0:
            anthropic_messages.append({"role": "user", "content": "(no input)"})

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

            response = self.client.messages.create(
                model=sampling_params.get("model", "claude-3-5-sonnet-20241022"),
                max_tokens=max_tokens,
                system=system_prompt,
                messages=anthropic_messages,
                tools=self.tool_schemas if self.tool_schemas else None,
                tool_choice={"type": "auto"} if self.tool_schemas else None,
                temperature=sampling_params.get("temperature", 1.0),
            )

            # response.content is a list of TextBlock / ToolUseBlock objects.
            text_parts: list[str] = []
            tool_use_blocks = []
            for block in (response.content or []):
                btype = getattr(block, "type", None)
                if btype == "text":
                    text_parts.append(getattr(block, "text", "") or "")
                elif btype == "tool_use":
                    tool_use_blocks.append(block)

            response_text = "".join(text_parts)
            all_response_text.append(response_text)
            total_tokens_used += (
                getattr(response.usage, "output_tokens", 0) if response.usage else 0
            )

            # Snapshot-update chat history with this assistant turn's text.
            chat_snapshot.append({"role": "assistant", "content": response_text})

            # AgentJet signals prompt overflow via a synthetic assistant message.
            if is_token_overflow_message(response_text):
                anthropic_messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
                break

            # Append the full assistant turn (text + tool_use blocks) back into
            # the Anthropic message list so the next round sees its output.
            assistant_blocks = []
            for block in (response.content or []):
                try:
                    assistant_blocks.append(block.model_dump())
                except Exception:
                    if getattr(block, "type", None) == "text":
                        assistant_blocks.append({"type": "text", "text": getattr(block, "text", "")})
            if assistant_blocks:
                anthropic_messages.append({"role": "assistant", "content": assistant_blocks})

            if tool_use_blocks:
                for tu in tool_use_blocks:
                    history_tool_calls.append({
                        "name": getattr(tu, "name", ""),
                        "arguments": getattr(tu, "input", {}) or {},
                    })

            if assistant_turns >= self.max_assistant_turns:
                break

            if not tool_use_blocks:
                break

            if total_tokens_used >= self.max_response_length:
                break

            # Execute each tool call, then return ALL tool results in a single
            # user message (Anthropic requires parallel tool results batched).
            tool_results = []
            for tu in tool_use_blocks:
                tool_name = getattr(tu, "name", "")
                tool_args = getattr(tu, "input", {}) or {}
                call_id = getattr(tu, "id", None) or _new_tool_use_id()

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
                else:
                    truncated_text = f"Error: Unknown tool {tool_name}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": truncated_text,
                })

            anthropic_messages.append({"role": "user", "content": tool_results})

            if total_tokens_used >= self.max_response_length:
                break

        final_response = "\n".join(all_response_text)
        return final_response, chat_snapshot, assistant_turns


# ---------------------------------------------------------------------------
# Agent execution (drop-in replacement for v3's / responses' execute_agent).
# ---------------------------------------------------------------------------

def _base_url_for_anthropic(base_url: str) -> str:
    """The Anthropic SDK appends `/v1/messages` to base_url itself.

    The swarm server hands out an OpenAI-style base_url ending in `/v1`
    (so the OpenAI SDK hits `/v1/chat/completions`). Strip the trailing `/v1`
    so the Anthropic SDK reconstructs `/v1/messages` instead of `/v1/v1/messages`.
    """
    base = (base_url or "").rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


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

    tool_schemas = [PYTHON_TOOL_SCHEMA_MESSAGES]
    tool_instances = {"python_code_with_standard_io": PythonTool(timeout=30)}

    # api_key carries the AgentJet routing token; the server's /v1/messages
    # route reads it from the SDK's `x-api-key` header.
    client = Anthropic(
        base_url=_base_url_for_anthropic(base_url),
        api_key=api_key,
        timeout=300,
    )

    agent_loop = MessagesAgentLoop(
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
            "api": "messages",
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
    print("Testing agent_run_messages.py ...")
    print("Multi-turn agent (Anthropic Messages API) with Python code execution tool")
