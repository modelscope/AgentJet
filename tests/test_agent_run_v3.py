"""Regression tests for tutorial/opencode_build_aime/agent_run_v3.py.

Covers the schema-vs-instance split in AgentLoop: the OpenAI API must
receive the schema list, while tool execution must go through real
tool instances. Previously both were conflated into a single ``tools``
dict whose values were schema dicts, so ``.create()`` raised
``'dict' object has no attribute 'create'`` the moment the model
emitted a tool call.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from tutorial.opencode_build_aime.agent_run_v3 import (
    PYTHON_TOOL_SCHEMA,
    AgentLoop,
    PythonTool,
    extract_tool_calls,
)


# ---------------------------------------------------------------------------
# AIME transcript fixture — the exact conversation that triggered the
# original `'dict' object has no attribute 'create'` bug. Kept verbatim
# (as a raw string, so the `\n` inside the JSON stay as two chars and are
# interpreted by json.loads, not Python's string literal parser).
# ---------------------------------------------------------------------------

AIME_USER_PROBLEM = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem.\n\n"
    "If we define $\\otimes(a,b,c)$ by\n"
    "\\begin{align*}\n"
    "\\otimes(a,b,c) = \\frac{\\max(a,b,c)- \\min(a,b,c)}"
    "{a+b+c-\\min(a,b,c)-\\max(a,b,c)},\n"
    "\\end{align*}\n"
    "compute $\\otimes(\\otimes(7,1,3),\\otimes(-3,-4,2),1)$.\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)

# The code the model emitted inside its <tool_call> block (decoded).
AIME_TOOL_CODE = (
    "def otimes(a, b, c):\n"
    "    max_val = max(a, b, c)\n"
    "    min_val = min(a, b, c)\n"
    "    middle_val = a + b + c - max_val - min_val\n"
    "    return (max_val - min_val) / middle_val\n"
    "\n"
    "first_inner = otimes(7, 1, 3)\n"
    "second_inner = otimes(-3, -4, 2)\n"
    "result = otimes(first_inner, second_inner, 1)\n"
    "print(result)"
)

# Verbatim assistant content from the failing transcript (truncated <think>,
# but the <tool_call> block is exactly what the model emitted).
AIME_ASSISTANT_CONTENT = r"""<think>
Reasoning elided — the model breaks the problem into three otimes calls.
</think>

<tool_call>
{"name": "python_code_with_standard_io", "arguments": {"code": "def otimes(a, b, c):\n    max_val = max(a, b, c)\n    min_val = min(a, b, c)\n    middle_val = a + b + c - max_val - min_val\n    return (max_val - min_val) / middle_val\n\nfirst_inner = otimes(7, 1, 3)\nsecond_inner = otimes(-3, -4, 2)\nresult = otimes(first_inner, second_inner, 1)\nprint(result)", "input": ""}}
</tool_call>"""


def _mock_tool(
    create_return=("iid-1", {"text": ""}),
    execute_return=({"text": "ok"}, 0.0, {}),
    create_exc=None,
    execute_exc=None,
    release_exc=None,
):
    tool = MagicMock()
    tool.create = AsyncMock(
        side_effect=create_exc, return_value=create_return if create_exc is None else None
    )
    tool.execute = AsyncMock(
        side_effect=execute_exc, return_value=execute_return if execute_exc is None else None
    )
    tool.release = AsyncMock(
        side_effect=release_exc, return_value=None
    )
    return tool


def _make_openai_response(content="", tool_calls=None, total_tokens=10):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(total_tokens=total_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class TestAgentLoopSchemaInstanceSplit(unittest.TestCase):
    def test_tool_call_executes_through_instance(self):
        """Regression: model emits a tool_call → PythonTool.execute runs."""
        tool_call = _make_tool_call(
            "call_1",
            "python_code_with_standard_io",
            {"code": "print(2 + 2)", "input": ""},
        )
        responses = [
            _make_openai_response(content="let me compute", tool_calls=[tool_call]),
            _make_openai_response(content="The answer is 4.", tool_calls=None),
        ]

        client = MagicMock()
        client.chat.completions.create.side_effect = responses

        python_tool = PythonTool(timeout=10)
        loop = AgentLoop(
            client=client,
            tool_schemas=[PYTHON_TOOL_SCHEMA],
            tool_instances={"python_code_with_standard_io": python_tool},
            max_assistant_turns=4,
        )

        final, messages, turns = asyncio.run(
            loop.run([{"role": "user", "content": "what is 2+2?"}], {"model": "fake"})
        )

        self.assertEqual(turns, 2)
        self.assertIn("The answer is 4.", final)

        tool_messages = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_1")
        self.assertIn("4", tool_messages[0]["content"])
        self.assertNotIn("Error executing tool", tool_messages[0]["content"])

    def test_api_receives_schema_list_not_instances(self):
        """The OpenAI client must be called with schemas, never with PythonTool objects."""
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            content="done", tool_calls=None
        )

        python_tool = PythonTool(timeout=10)
        loop = AgentLoop(
            client=client,
            tool_schemas=[PYTHON_TOOL_SCHEMA],
            tool_instances={"python_code_with_standard_io": python_tool},
            max_assistant_turns=1,
        )

        asyncio.run(loop.run([{"role": "user", "content": "hi"}], {"model": "fake"}))

        _, kwargs = client.chat.completions.create.call_args
        self.assertEqual(kwargs["tools"], [PYTHON_TOOL_SCHEMA])
        self.assertEqual(kwargs["tool_choice"], "auto")
        for tool in kwargs["tools"]:
            self.assertIsInstance(tool, dict)

    def test_unknown_tool_name_produces_error_message(self):
        bogus = _make_tool_call("call_x", "not_a_real_tool", {})
        responses = [
            _make_openai_response(content="", tool_calls=[bogus]),
            _make_openai_response(content="giving up", tool_calls=None),
        ]
        client = MagicMock()
        client.chat.completions.create.side_effect = responses

        loop = AgentLoop(
            client=client,
            tool_schemas=[PYTHON_TOOL_SCHEMA],
            tool_instances={"python_code_with_standard_io": PythonTool(timeout=10)},
            max_assistant_turns=3,
        )

        _, messages, _ = asyncio.run(
            loop.run([{"role": "user", "content": "x"}], {"model": "fake"})
        )

        tool_messages = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("Unknown tool", tool_messages[0]["content"])

    def test_empty_tool_registry_disables_tools_param(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_openai_response(
            content="no tools", tool_calls=None
        )

        loop = AgentLoop(
            client=client,
            tool_schemas=[],
            tool_instances={},
            max_assistant_turns=1,
        )
        asyncio.run(loop.run([{"role": "user", "content": "hi"}], {"model": "fake"}))

        _, kwargs = client.chat.completions.create.call_args
        self.assertIsNone(kwargs["tools"])
        self.assertIsNone(kwargs["tool_choice"])


class TestToolExecutionBlock(unittest.TestCase):
    """Focused tests for the tool-call execution block in AgentLoop.run."""

    def _run(self, loop, messages=None):
        messages = messages or [{"role": "user", "content": "go"}]
        return asyncio.run(loop.run(messages, {"model": "fake"}))

    def _build_loop(self, client, tool_instances, **kwargs):
        return AgentLoop(
            client=client,
            tool_schemas=[PYTHON_TOOL_SCHEMA],
            tool_instances=tool_instances,
            max_assistant_turns=kwargs.pop("max_assistant_turns", 4),
            **kwargs,
        )

    def test_multiple_tool_calls_in_one_message_all_execute_in_order(self):
        calls = [
            _make_tool_call("call_A", "t", {"x": 1}),
            _make_tool_call("call_B", "t", {"x": 2}),
        ]
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_openai_response(content="", tool_calls=calls),
            _make_openai_response(content="final", tool_calls=None),
        ]

        tool = _mock_tool(execute_return=({"text": "R"}, 0.0, {}))
        loop = self._build_loop(client, {"t": tool})

        _, messages, _ = self._run(loop)

        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["call_A", "call_B"])
        self.assertEqual(tool.create.await_count, 2)
        self.assertEqual(tool.execute.await_count, 2)
        self.assertEqual(tool.release.await_count, 2)
        exec_args = [c.args[1] for c in tool.execute.await_args_list]
        self.assertEqual(exec_args, [{"x": 1}, {"x": 2}])

    def test_create_exception_becomes_error_text(self):
        call = _make_tool_call("call_1", "t", {})
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_openai_response(content="", tool_calls=[call]),
            _make_openai_response(content="done", tool_calls=None),
        ]

        tool = _mock_tool(create_exc=RuntimeError("boom-create"))
        loop = self._build_loop(client, {"t": tool})

        _, messages, _ = self._run(loop)

        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("Error executing tool", tool_msgs[0]["content"])
        self.assertIn("boom-create", tool_msgs[0]["content"])
        tool.execute.assert_not_awaited()
        tool.release.assert_not_awaited()

    def test_execute_exception_becomes_error_text(self):
        call = _make_tool_call("call_1", "t", {})
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_openai_response(content="", tool_calls=[call]),
            _make_openai_response(content="done", tool_calls=None),
        ]

        tool = _mock_tool(execute_exc=ValueError("boom-exec"))
        loop = self._build_loop(client, {"t": tool})

        _, messages, _ = self._run(loop)

        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertIn("Error executing tool", tool_msgs[0]["content"])
        self.assertIn("boom-exec", tool_msgs[0]["content"])
        tool.create.assert_awaited_once()
        tool.release.assert_not_awaited()

    def test_release_exception_overwrites_response_with_error_text(self):
        """Observed behavior: release() raising clobbers a successful execute result.

        Documents current behavior so a future refactor is a conscious choice,
        not an accident.
        """
        call = _make_tool_call("call_1", "t", {})
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_openai_response(content="", tool_calls=[call]),
            _make_openai_response(content="done", tool_calls=None),
        ]

        tool = _mock_tool(
            execute_return=({"text": "real result"}, 0.0, {}),
            release_exc=RuntimeError("boom-release"),
        )
        loop = self._build_loop(client, {"t": tool})

        _, messages, _ = self._run(loop)

        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("Error executing tool", tool_msgs[0]["content"])
        self.assertIn("boom-release", tool_msgs[0]["content"])
        self.assertNotIn("real result", tool_msgs[0]["content"])

    def test_tool_call_id_is_preserved_on_all_paths(self):
        ok_call = _make_tool_call("id-ok", "t", {})
        bad_call = _make_tool_call("id-missing", "not_registered", {})
        err_call = _make_tool_call("id-err", "t", {})

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_openai_response(content="", tool_calls=[ok_call, bad_call, err_call]),
            _make_openai_response(content="done", tool_calls=None),
        ]

        # Succeed first call, raise on second registered call.
        tool = MagicMock()
        tool.create = AsyncMock(side_effect=[("iid-1", {"text": ""}), ("iid-2", {"text": ""})])
        tool.execute = AsyncMock(
            side_effect=[({"text": "R"}, 0.0, {}), RuntimeError("boom")]
        )
        tool.release = AsyncMock(return_value=None)

        loop = self._build_loop(client, {"t": tool})
        _, messages, _ = self._run(loop)

        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(
            [m["tool_call_id"] for m in tool_msgs], ["id-ok", "id-missing", "id-err"]
        )
        self.assertEqual(tool_msgs[0]["content"], "R")
        self.assertIn("Unknown tool not_registered", tool_msgs[1]["content"])
        self.assertIn("Error executing tool", tool_msgs[2]["content"])

    def test_long_tool_response_is_truncated(self):
        call = _make_tool_call("call_1", "t", {})
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_openai_response(content="", tool_calls=[call]),
            _make_openai_response(content="done", tool_calls=None),
        ]

        big = "x" * 500
        tool = _mock_tool(execute_return=({"text": big}, 0.0, {}))
        loop = self._build_loop(
            client,
            {"t": tool},
            max_tool_response_length=100,
            tool_response_truncate_side="left",
        )

        _, messages, _ = self._run(loop)

        tool_msgs = [m for m in messages if m["role"] == "tool"]
        content = tool_msgs[0]["content"]
        # truncate_side="left" keeps the leading chunk and appends a marker.
        self.assertTrue(content.endswith("...(truncated)"))
        self.assertLessEqual(len(content), 100 + len("...(truncated)"))

    def test_create_receives_history_without_final_call(self):
        """history_tool_calls[:-1] must be passed to create()."""
        call = _make_tool_call("call_1", "t", {"k": "v"})
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_openai_response(content="", tool_calls=[call]),
            _make_openai_response(content="done", tool_calls=None),
        ]
        tool = _mock_tool()
        loop = self._build_loop(client, {"t": tool})

        self._run(loop)

        # Single-call message: history had exactly one entry (the current call),
        # so create() should get an empty history list.
        tool.create.assert_awaited_once()
        self.assertEqual(tool.create.await_args.kwargs, {"history_tool_calls": []})

    def test_max_response_length_break_skips_remaining_tool_calls(self):
        calls = [
            _make_tool_call("call_A", "t", {}),
            _make_tool_call("call_B", "t", {}),
        ]
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_openai_response(
                content="", tool_calls=calls, total_tokens=0
            ),
            _make_openai_response(content="final", tool_calls=None),
        ]

        # Response text length (50) must push cumulative count past cap on first call.
        big_enough = "y" * 50
        tool = _mock_tool(execute_return=({"text": big_enough}, 0.0, {}))
        loop = self._build_loop(
            client,
            {"t": tool},
            max_response_length=40,
            max_tool_response_length=100,
        )

        _, messages, _ = self._run(loop)

        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "call_A")
        self.assertEqual(tool.execute.await_count, 1)


class TestAimeTranscriptRegression(unittest.TestCase):
    """Replay the exact transcript that first surfaced the bug.

    Before the fix, the agent loop raised ``'dict' object has no attribute
    'create'`` the moment the model emitted this <tool_call> block. These
    tests pin down that the same input now flows through to a real
    subprocess and returns the right answer (4.0).
    """

    def test_extract_tool_calls_parses_transcript_block(self):
        _, calls = extract_tool_calls(AIME_ASSISTANT_CONTENT)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "python_code_with_standard_io")
        self.assertEqual(calls[0]["arguments"]["input"], "")
        self.assertEqual(calls[0]["arguments"]["code"], AIME_TOOL_CODE)

    def test_agent_loop_runs_transcript_end_to_end(self):
        structured_call = _make_tool_call(
            "call_aime_1",
            "python_code_with_standard_io",
            {"code": AIME_TOOL_CODE, "input": ""},
        )
        final_assistant_text = (
            "Computing each otimes step by hand confirms the result.\n"
            "Answer: 4\n\\boxed{4}"
        )
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_openai_response(
                content=AIME_ASSISTANT_CONTENT, tool_calls=[structured_call]
            ),
            _make_openai_response(content=final_assistant_text, tool_calls=None),
        ]

        loop = AgentLoop(
            client=client,
            tool_schemas=[PYTHON_TOOL_SCHEMA],
            tool_instances={"python_code_with_standard_io": PythonTool(timeout=15)},
            max_assistant_turns=4,
        )

        final, messages, turns = asyncio.run(
            loop.run(
                [{"role": "user", "content": AIME_USER_PROBLEM}],
                {"model": "fake-aime"},
            )
        )

        # Real Python execution path was reached (the original bug short-circuited here).
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "call_aime_1")
        self.assertNotIn("'dict' object has no attribute 'create'", tool_msgs[0]["content"])
        self.assertNotIn("Error executing tool", tool_msgs[0]["content"])
        self.assertIn("4.0", tool_msgs[0]["content"])

        # Loop continued, took a second model turn, and emitted the final answer.
        self.assertEqual(turns, 2)
        self.assertIn("Answer: 4", final)

    def test_agent_loop_propagates_user_problem_and_schema(self):
        """The first API call carries the user's AIME prompt and only schema dicts."""
        structured_call = _make_tool_call(
            "call_aime_2",
            "python_code_with_standard_io",
            {"code": "print(1+1)", "input": ""},
        )
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _make_openai_response(content="thinking", tool_calls=[structured_call]),
            _make_openai_response(content="Answer: 4", tool_calls=None),
        ]

        loop = AgentLoop(
            client=client,
            tool_schemas=[PYTHON_TOOL_SCHEMA],
            tool_instances={"python_code_with_standard_io": PythonTool(timeout=10)},
            max_assistant_turns=3,
        )
        asyncio.run(
            loop.run(
                [{"role": "user", "content": AIME_USER_PROBLEM}],
                {"model": "fake-aime"},
            )
        )

        first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        user_msgs = [m for m in first_kwargs["messages"] if m["role"] == "user"]
        self.assertTrue(any("\\otimes" in m["content"] for m in user_msgs))
        self.assertEqual(first_kwargs["tools"], [PYTHON_TOOL_SCHEMA])
        for tool in first_kwargs["tools"]:
            self.assertIsInstance(tool, dict)


class TestPythonToolDirect(unittest.TestCase):
    """Sanity check that PythonTool itself still works end-to-end."""

    def test_execute_runs_code_and_returns_stdout(self):
        tool = PythonTool(timeout=10)

        async def _go():
            iid, _ = await tool.create()
            result, _, _ = await tool.execute(iid, {"code": "print(6 * 7)", "input": ""})
            await tool.release(iid)
            return result

        result = asyncio.run(_go())
        self.assertIn("42", result["text"])


if __name__ == "__main__":
    unittest.main()
