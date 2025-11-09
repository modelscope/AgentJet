from astune.task_judge.judge_base import JudgeBase
import re

class MathAnswerAsJudge(JudgeBase):

    def __init__(self, config):
        self.config = config

    def compute_reward(self, judge_input_dictionary) -> tuple:
        raw_reward = 0

        final_answer = judge_input_dictionary['final_answer']
        task_core_arg = judge_input_dictionary['task_core_arg']
        reference_answer = task_core_arg.task.metadata['answer']
        reference_answer = reference_answer.split('####')[-1].strip()

        pattern = r'\\boxed\{([^}]*)\}'
        match = re.search(pattern, final_answer)
        if match:
            result = match.group(1)
            is_success = result == reference_answer
        else:
            is_success = False

        raw_reward = 1.0 if is_success else 0.0
        return raw_reward, is_success


class MathAnswerAndLlmAsJudge(JudgeBase):

    def __init__(self, config):
        self.config = config

    def compute_reward(self, judge_input_dictionary) -> tuple:
        raw_reward = 0

        final_answer = judge_input_dictionary['final_answer']
        task_core_arg = judge_input_dictionary['task_core_arg']
        reference_answer = task_core_arg.task.metadata['answer']

        from astune.context_manager.cmt_foreign_llm import construct_alien_llm_chat_fn
        alien_llm_chat_fn = construct_alien_llm_chat_fn(self.config)
        messages = [
            {
                'role':'system',
                'content':f'Is my result correct? If correct, say <Correct>, otherwise say <NotCorrect>.'
            },
            {
                'role':'user',
                'content':f'Is my result correct?\n\n\n----\nMy result: {final_answer}\n\n\n----\nReal result: {reference_answer}'
            }
        ]
        res = alien_llm_chat_fn(messages=messages)
        if '<Correct>' in res['content']:
            is_success = True
            raw_reward = 1.0
        else:
            is_success = False
            raw_reward = 0.0
        return raw_reward, is_success