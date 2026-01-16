import re

from ajet.task_judge.base_judge import BaseJudge
from ajet.task_rollout.dashscope_llm_bridge import create_external_llm_fn
from ajet.workflow import WorkflowOutput, WorkflowTask


class MathAnswerAsJudge(BaseJudge):
    def __init__(self, config):
        self.config = config

    def compute_reward(self, workflow_task: WorkflowTask, workflow_output: WorkflowOutput) -> tuple:
        raw_reward = 0
        final_answer = workflow_output.metadata["final_answer"]  # By default there's no final_answer; register it by calling ajet_proxy.update_judge_input_dictionary(final_answer=final_answer) in the workflow
        reference_answer = workflow_task.task.metadata["answer"]
        reference_answer = reference_answer.split("####")[-1].strip()

        pattern = r"\\boxed\{([^}]*)\}"
        match = re.search(pattern, final_answer)
        if match:
            result = match.group(1)
            is_success = result == reference_answer
        else:
            is_success = False

        raw_reward = 1.0 if is_success else 0.0
        return raw_reward, is_success


class MathAnswerAndLlmAsJudge(BaseJudge):
    def __init__(self, config):
        self.config = config

    def compute_reward(self, workflow_task: WorkflowTask, workflow_output: WorkflowOutput) -> tuple:
        raw_reward = 0
        final_answer = workflow_output.metadata["final_answer"]  # By default there's no final_answer; register it by calling ajet_proxy.update_judge_input_dictionary(final_answer=final_answer) in the workflow
        reference_answer = workflow_task.task.metadata["answer"]
        reference_answer = reference_answer.split("####")[-1].strip()

        external_llm_fn = create_external_llm_fn(
            alien_llm_model=self.config.ajet.task_judge.alien_llm_model,
            alien_llm_response_length=self.config.ajet.task_judge.alien_llm_response_length,
        )
        messages = [
            {
                "role": "system",
                "content": "Is my result correct? If correct, say <Correct>, otherwise say <NotCorrect>.",
            },
            {
                "role": "user",
                "content": f"Is my result correct?\n\n\n----\nMy result: {final_answer}\n\n\n----\nReal result: {reference_answer}",
            },
        ]
        res = external_llm_fn(messages=messages)
        if "<Correct>" in res["content"]:
            is_success = True
            raw_reward = 1.0
        else:
            is_success = False
            raw_reward = 0.0
        return raw_reward, is_success
