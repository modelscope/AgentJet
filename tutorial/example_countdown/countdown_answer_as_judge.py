import re

from ajet.task_judge.base_judge import BaseJudge
from ajet.workflow import WorkflowOutput, WorkflowTask


class CountdownAnswerAsJudge(BaseJudge):
    def __init__(self, config):
        self.config = config
        self.format_score = 0.1
        self.correct_score = 1.0

    def _validate_equation(self, equation_str, available_numbers):
        try:
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

            available_numbers = sorted(available_numbers)
            numbers_in_eq = sorted(numbers_in_eq)

            return numbers_in_eq == available_numbers
        except Exception:
            return False

    def _evaluate_equation(self, equation_str):
        try:
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation_str):
                raise ValueError("Invalid characters in equation.")

            result = eval(equation_str, {"__builtins__": None}, {})
            return result
        except Exception:
            return None

    def _compute_score(self, equation, target, numbers):
        if equation is None:
            return 0

        if not self._validate_equation(equation, numbers):
            return self.format_score

        try:
            result = self._evaluate_equation(equation)
            if result is None:
                return self.format_score

            if abs(result - target) < 1e-5:
                return self.correct_score
            else:
                return self.format_score
        except Exception:
            return self.format_score

    def compute_reward(self, workflow_task: WorkflowTask, workflow_output: WorkflowOutput) -> tuple:
        raw_reward = 0
        final_answer = workflow_output.metadata["final_answer"]
        target = workflow_output.metadata["target"]
        numbers = workflow_output.metadata["nums"]

        if target is None or not numbers:
            return 0.0, False

        pattern = r"\\boxed\{([^}]*)\}"
        match = re.search(pattern, final_answer)

        if match:
            result = match.group(1)
            raw_reward = self._compute_score(result, target, numbers)
            is_success = raw_reward >= self.correct_score
        else:
            raw_reward = 0.0
            is_success = False

        return raw_reward, is_success
