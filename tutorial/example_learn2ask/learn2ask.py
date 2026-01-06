
import re
import time
import asyncio
import threading

from agentscope.message import Msg
from loguru import logger

from ajet import ModelTuner, Workflow, WorkflowOutput, WorkflowTask
from ajet.utils.robust_dashscope import RobustDashScopeChatModel

system_prompt = """# Task
You are a medical assistant. Your task is to understand the ongoing conversation and continue the medical inquiry in English.

## Guidelines
- Each response must contain exactly one clear and concise medical question with 2 to 3 answer choices.
- Do not repeat any previous question.
- Your response must be a single sentence.
- If enough information has been gathered to make a medication suggestion, output only: <stop />
"""

reward_prompt = """# Task
You are an evaluation assistant. The user will provide a dialogue history between a doctor and a patient. You must analyze the dialogue and evaluate the doctor's last message.

# Grading Policy
## Format Score
- 1.0: The doctor's last message contains exactly **one question**.
- 0.5: The doctor's last message contains **two questions**.
- 0.0: The doctor's last message contains **three or more questions**.

## Content Score
Reference Information contains the information that the doctor has not known.

- 1.0: The question(s) **directly ask about** item in the Reference Information.
- 0.1: The question(s) are a general type of question that could be asked for any symptoms.
- 0.0: The question(s) are **irrelevant** to all items in the Reference Information.

### You should

- ONLY if the doctor asks a question that helps to collect information and diagnose the patient, it is a good question.
- A ambiguous question should get 0.
    - For example, the doctor asks "How long have you been feeling this way?", but "this way" is not clear in the previous messages.
    - For example, the doctor asks "Do you feel bad?". This is a meaningless question that does not provide any useful information.

# Reference Information

{}

# Output Format
<think>Explain your reasoning for the format and content scores clearly and concisely.</think>
<format_score>Insert only the format score as a float (e.g., 1.0, 0.5, 0.0)</format_score>
<content_score>Insert only the content score as a float (e.g., 1.0, 0.5, 0.0)</content_score>

> ✅ Important:
> - Output **exactly** the three tags shown above.
> - Do **not** include any additional text, explanation, or formatting outside the tags.
> - Scores must be based **only** on the doctor's **last message** and the provided Reference Information.
> - Ensure clarity and precision in your evaluation reasoning within the `<think>` tag.
"""


llm = RobustDashScopeChatModel("qwen-plus", stream=False)


async def llm_reward(init_messages: list[Msg], response: str, truth_info: str):
    def format_messages(messages: list[Msg]) -> str:
        result_str = ""
        for msg in messages:
            if msg.role == "user":
                result_str += f"patient: {msg.content}\n"
            if msg.role == "assistant":
                result_str += f"doctor: {msg.content}\n"
        return result_str

    def parse_tag_string(text: str):
        pattern = r"<(\w+)>(.*?)</\1>"
        matches = re.findall(pattern, text)
        result = {}
        for tag, value in matches:
            result[tag] = value
        return result

    history = format_messages([] + init_messages + [Msg("assistant", response, role="assistant")])
    messages = [
        {"role": "system", "content": reward_prompt.format(truth_info)},
        {"role": "user", "content": history},
    ]

    try_count, max_retries = 0, 5
    while try_count <= max_retries:
        try:

            async def get_content():
                from agentscope.model import ChatResponse

                response = await llm(messages)

                if isinstance(response, ChatResponse):
                    res = "".join([x["text"] for x in response.content if "text" in x])
                else:
                    res = ""
                    async for chunk in response:
                        res += "".join([x["text"] for x in chunk.content if "text" in x])
                return res

            content = await get_content()
            score_dict = parse_tag_string(content)
            return score_dict
        except Exception as e:
            if try_count > max_retries:
                logger.warning("retried too many times, abort task.")
                return None
            else:
                logger.warning(f"error: {e}, response:{response}, retrying...")
                time.sleep(2**try_count)


async def reward_fn(init_messages: list[Msg], response: str, truth_action: str, truth_info: str):
    """
    content_score: R_a, the reward for response quality
    action_score: R_s, the reward for decision correctness
    format_score: P, the reward for response format
    """

    action_response = "stop" if "<stop />" in response else "continue"
    if truth_action == action_response:
        action_score = 1.0
        if truth_action == "continue":
            score_dict = await llm_reward(init_messages, response, truth_info)
            if score_dict is not None:
                format_score = float(score_dict.get("format_score", 0.0))
                content_score = float(score_dict.get("content_score", 0.0))
            else:
                format_score, content_score = 0.0, 0.0
        else:
            content_score = 1.0
            format_score = 1.0 if response == "<stop />" else 0.0
    else:
        action_score, format_score, content_score = 0.0, 0.0, 0.0

    # treat as self.train_mode == "Ra+Rs", the default setting
    final_reward = action_score * (1 + 2 * content_score) + format_score

    return final_reward


_reward_semaphore = threading.Semaphore(16)

async def reward_fn_with_semaphore(*args, **kwargs):

    get_sem_ok = False
    while not get_sem_ok:
        get_sem_ok = _reward_semaphore.acquire(blocking=False)
        if not get_sem_ok:
            await asyncio.sleep(1)

    try:
        fn_result = await reward_fn(*args, **kwargs)
    finally:
        _reward_semaphore.release()

    return fn_result


class ExampleLearn2Ask(Workflow):
    name: str = "math_agent_workflow"

    async def execute(self, workflow_task: WorkflowTask, model_tuner: ModelTuner) -> WorkflowOutput:
        from agentscope.agent import ReActAgent
        from agentscope.formatter import DashScopeChatFormatter
        from agentscope.memory import InMemoryMemory

        messages = workflow_task.task.init_messages
        assert isinstance(messages, list)
        truth_action = workflow_task.task.metadata["decision_truth"] or "continue"
        truth_info = workflow_task.task.metadata["info_truth"]

        self.agent = ReActAgent(
            name="math_react_agent",
            sys_prompt=system_prompt,
            model=model_tuner,
            formatter=DashScopeChatFormatter(),
            toolkit=None,
            memory=InMemoryMemory(),
            max_iters=1,
        )
        self.agent.set_console_output_enabled(False)
        msg = [
            # Msg("system", system_prompt, role="system"),
            *[Msg(name=x["role"], content=x["content"], role=x["role"]) for x in messages]
        ]
        result = await self.agent.reply(msg)
        if isinstance(result.content, str):
            response = result.content
        elif isinstance(result.content, list):
            response = result.content[0]["text"]  # type: ignore
        else:
            raise NotImplementedError(f"do not know how to handle {type(result.content)}")
        reward = await reward_fn_with_semaphore(msg, response, truth_action, truth_info)
        return WorkflowOutput(reward=reward)
