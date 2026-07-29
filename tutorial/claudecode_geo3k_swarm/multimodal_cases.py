# -*- coding: utf-8 -*-
"""
Shared definitions for the multimodal token-id test cases.

Single source of truth for the four cases the user asked for, used by both:

  * ``gen_ground_truth.py``      — POSTs each case's messages to a running
                                   ``vllm serve`` OpenAI endpoint and records the
                                   vLLM-side prompt tokenization → ``./token_id/<case>.md``
  * ``test_multimodal_cases.py`` — drives the same messages through the live
                                   swarm so the trainer captures the *training*
                                   tokenization → ``./token_id/<case>.debug.md``,
                                   then compares.

Each case is a full OpenAI ``messages`` list (with data-URL images) plus an
optional ``tools`` list. Images are deterministic solid PNGs at fixed sizes so
the image-pad counts are stable for this model's patch/merge settings:

    256x168 -> 54 image-pad tokens, grid [1,12,18]
    336x224 -> 96 image-pad tokens, grid [1,16,24]

The four cases:
  1. two_images_one_msg     — one user turn carrying TWO images + text.
  2. image_text_image_turns — image (turn 1) / assistant text / image + text
                              (turn 2): images interleaved across turns.
  3. pure_text              — text only, zero images (mirrors token_id_io.md B).
  4. img_text_img_text_tool — image / assistant tool_call / tool text response /
                              image + text. Tool turns are text-only.

"Stick to vLLM behavior": within a single message we let the request path lay
out images then text exactly as ``build_multimodal_messages`` does; the
cross-turn interleave is what exercises the multi-message merge fix.
"""

from typing import Dict

from PIL import Image

from ajet.schema.task import Task
from ajet.utils.multimodal import encode_image_as_data_url


# distinct sizes -> distinct grids so a dropped image is detectable
IMG_A_SIZE = (256, 168)   # 54 pads, grid [1,12,18]
IMG_B_SIZE = (336, 224)   # 96 pads, grid [1,16,24]

CASE_NAMES = [
    "two_images_one_msg",
    "image_text_image_turns",
    "pure_text",
    "img_text_img_text_tool",
]

# expected total <|image_pad|> (151655) count per case, for assertions
EXPECTED_IMAGE_TOKENS = {
    "two_images_one_msg": 150,       # 54 + 96
    "image_text_image_turns": 150,   # 54 + 96
    "pure_text": 0,
    "img_text_img_text_tool": 150,   # 54 + 96
}

SYSTEM_PROMPT = "You are a careful assistant. Put the final answer in \\boxed{}."

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "lookup_note",
            "description": "Look up a short text note by key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }
]


def _img(size) -> Image.Image:
    return Image.new("RGB", size, "white")


def _data_url(size) -> str:
    return encode_image_as_data_url(_img(size))


def _img_block(size) -> dict:
    return {"type": "image_url", "image_url": {"url": _data_url(size)}}


def _text_block(text) -> dict:
    return {"type": "text", "text": text}


def build_case_messages(case: str) -> Dict:
    """Return {"messages": [...], "tools": [...]} for a case name."""
    system = {"role": "system", "content": SYSTEM_PROMPT}

    if case == "two_images_one_msg":
        messages = [
            system,
            {"role": "user", "content": [
                _img_block(IMG_A_SIZE),
                _img_block(IMG_B_SIZE),
                _text_block("Compare the two figures and find y."),
            ]},
        ]
        return {"messages": messages, "tools": []}

    if case == "image_text_image_turns":
        messages = [
            system,
            {"role": "user", "content": [
                _img_block(IMG_A_SIZE),
                _text_block("Here is the first figure."),
            ]},
            {"role": "assistant", "content": "Understood, I see the first figure."},
            {"role": "user", "content": [
                _img_block(IMG_B_SIZE),
                _text_block("Now compare with the second figure and find y."),
            ]},
        ]
        return {"messages": messages, "tools": []}

    if case == "pure_text":
        messages = [
            system,
            {"role": "user", "content": "Find y given 8 = y. Think step by step."},
        ]
        return {"messages": messages, "tools": []}

    if case == "img_text_img_text_tool":
        # image / assistant tool_call / tool text response / image + text.
        messages = [
            system,
            {"role": "user", "content": [
                _img_block(IMG_A_SIZE),
                _text_block("Here is the first figure. Look up note 'hint' if needed."),
            ]},
            {"role": "assistant", "content": "", "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup_note",
                                 "arguments": "{\"key\": \"hint\"}"},
                }
            ]},
            {"role": "tool", "content": "hint: y equals the labeled side.",
             "tool_call_id": "call_1"},
            {"role": "user", "content": [
                _img_block(IMG_B_SIZE),
                _text_block("Now compare with the second figure and find y."),
            ]},
        ]
        return {"messages": messages, "tools": TOOL_SCHEMA}

    raise ValueError(f"unknown case: {case}")


def build_case_task(case: str) -> Task:
    """A swarm Task whose metadata carries the full message list + tools.

    task_id == case name so the trainer capture can name files per case.
    """
    spec = build_case_messages(case)
    return Task(
        task_id=case,
        main_query=case,
        metadata={
            "messages": spec["messages"],
            "tools": spec["tools"],
            "ground_truth": "n/a",
        },
    )


def _execute_multimodal_case(task: Task, api_baseurl_key) -> "object":
    """Drive one case's messages through the episode endpoint.

    Only the *prompt* tokenization matters for this test, so the model's answer
    is ignored and a constant reward is returned. Uses the OpenAI SDK against
    the swarm's per-episode fake endpoint (base_url/api_key).
    """
    from openai import OpenAI
    from ajet.schema.task import WorkflowOutput

    spec = build_case_messages(task.task_id)
    client = OpenAI(base_url=api_baseurl_key.base_url, api_key=api_baseurl_key.api_key)

    kwargs = dict(
        model="AgentJet-Model",
        messages=spec["messages"],
        max_tokens=64,
        temperature=1.0,
    )
    if spec["tools"]:
        kwargs["tools"] = spec["tools"]
        kwargs["tool_choice"] = "none"  # tool turns are canned; don't loop

    final_answer = ""
    try:
        resp = client.chat.completions.create(**kwargs)
        final_answer = resp.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001 — prompt capture is what matters
        final_answer = f"(request error ignored: {e})"

    return WorkflowOutput(reward=0.0, metadata={"final_answer": final_answer})
