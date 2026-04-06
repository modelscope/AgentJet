# -*- coding: utf-8 -*-
"""Compute relative rewards based on relevance, diversity, repetition quality, and dynamic user feedback."""

import os
import re
import json
import collections
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

from loguru import logger
from textwrap import dedent
from beast_logger import print_listofdict
from openjudge.graders.base_grader import GraderMode, GraderScore, GraderRank
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.common.relevance import RelevanceGrader
from openjudge.graders.format.ngram_repetition_penalty import NgramRepetitionPenaltyGrader
from openjudge.models import OpenAIChatModel
try:
    from ajet.utils.compute_madness import has_repeat
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _repo_root = str(_Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    from ajet.utils.compute_madness import has_repeat


@dataclass
class AgentJetCommand:
    """Represents a parsed /agentjet command."""
    command_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw_command: str = ""


@dataclass
class UserFeedback:
    """Represents parsed user feedback/opinion."""
    has_opinion: bool = False
    feedback_type: str = ""
    feedback_content: str = ""
    raw_input: str = ""


class RelativeRewardComputer:
    """Compute composite rewards combining user feedback, relevance, diversity, and quality gate."""

    DEFAULT_JUDGE_PROMPT = dedent("""
        You are ranking multiple responses based on user preferences.
        Current evaluation criteria:
        - Respond to the question accurately and completely
        - Use appropriate tone and style
        - Be helpful and clear

        Question: {question}

        Responses to rank:
        {answers_block}

        Rank these responses from best to worst based on user preferences.
        Return a json object with exactly two fields:
        - "rank": list of integers (1-indexed) ordered from best to worst, e.g. [2, 1, 3]
        - "reason": brief explanation of the ranking"""
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        judge_model_name: Optional[str] = None,
        qwen_max_model_name: Optional[str] = None,
        w_relevance: Optional[float] = None,
        w_diversity: Optional[float] = None,
        w_user_feedback: Optional[float] = None,
        history_max_size: Optional[int] = None,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "sk-xxx")
        self.base_url = base_url or os.getenv("JUDGE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.judge_model_name = judge_model_name or os.getenv("JUDGE_MODEL", "qwen-max")
        self.qwen_max_model_name = qwen_max_model_name or os.getenv("QWEN_MAX_MODEL", "qwen3-max")
        self.w_relevance = w_relevance if w_relevance is not None else float(os.getenv("W_RELEVANCE", "0.4"))
        self.w_diversity = w_diversity if w_diversity is not None else float(os.getenv("W_DIVERSITY", "0.3"))
        self.w_user_feedback = w_user_feedback if w_user_feedback is not None else float(os.getenv("W_USER_FEEDBACK", "0.3"))
        self.history_max_size = history_max_size or int(os.getenv("DIVERSITY_HISTORY_SIZE", "25"))

        self._judge_model = None
        self._qwen_max_model = None
        self._dynamic_judge_prompt = self.DEFAULT_JUDGE_PROMPT
        self._user_preference_history: List[str] = []
        self._response_history: List[str] = []
        self._agentjet_command_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    @property
    def judge_model(self) -> OpenAIChatModel:
        if self._judge_model is None:
            self._judge_model = OpenAIChatModel(
                model=self.judge_model_name,
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._judge_model

    @property
    def qwen_max_model(self) -> OpenAIChatModel:
        if self._qwen_max_model is None:
            self._qwen_max_model = OpenAIChatModel(
                model=self.qwen_max_model_name,
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._qwen_max_model

    @property
    def relevance_grader(self) -> RelevanceGrader:
        return RelevanceGrader(model=self.judge_model)

    @property
    def repetition_grader(self) -> NgramRepetitionPenaltyGrader:
        return NgramRepetitionPenaltyGrader(
            n=4,
            penalty_threshold=0.15,
            use_soft_penalty=True,
            max_penalty=-1.0,
            min_scaling=0.0,
        )

    def get_dynamic_listwise_grader(self, n: int) -> LLMGrader:
        answers_block = "\n".join([f"{i+1}. {{answer_{i+1}}}" for i in range(n)])
        template = self._dynamic_judge_prompt.replace("{answers_block}", answers_block)

        from beast_logger import print_dict
        print_dict({
            "name": "user_feedback_listwise",
            "mode": "GraderMode.LISTWISE",
            "description": "Rank responses based on dynamic user preferences",
            "model": self.judge_model,
            "template": template,
        })

        return LLMGrader(
            name="user_feedback_listwise",
            mode=GraderMode.LISTWISE,
            description="Rank responses based on dynamic user preferences",
            model=self.judge_model,
            template=template,
        )

    def set_agentjet_command_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._agentjet_command_callback = callback

    def get_dynamic_judge_prompt(self) -> str:
        return self._dynamic_judge_prompt

    def get_user_preference_history(self) -> List[str]:
        return list(self._user_preference_history)

    def _get_ngrams(self, text: str, n: int = 3) -> collections.Counter:
        tokens = text.lower().split()
        if len(tokens) < n:
            return collections.Counter(tokens)
        return collections.Counter(
            tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
        )

    def _ngram_overlap(self, text_a: str, text_b: str, n: int = 3) -> float:
        ngrams_a = self._get_ngrams(text_a, n)
        ngrams_b = self._get_ngrams(text_b, n)
        if not ngrams_a or not ngrams_b:
            return 0.0
        intersection = sum((ngrams_a & ngrams_b).values())
        union = sum((ngrams_a | ngrams_b).values())
        return intersection / union if union > 0 else 0.0

    def compute_diversity_scores(self, contents: List[str], history: Optional[List[str]] = None) -> List[float]:
        history = history or []
        n = len(contents)
        scores = []
        for i, content_i in enumerate(contents):
            if n > 1:
                batch_overlaps = [
                    self._ngram_overlap(content_i, contents[j])
                    for j in range(n)
                    if j != i
                ]
                within_batch = max(batch_overlaps)
            else:
                within_batch = 0.0

            if history:
                cross_request = max(self._ngram_overlap(content_i, h) for h in history)
            else:
                cross_request = 0.0

            overlap = max(within_batch, cross_request)
            scores.append(1.0 - overlap)
        return scores

    async def compute_quality_scores(self, contents: List[str]) -> List[float]:
        scores = []
        for content in contents:
            try:
                rep_result = await self.repetition_grader.aevaluate(response=content)
                ngram_penalty = rep_result.score if isinstance(rep_result, GraderScore) else 0.0
                ngram_score = 1.0 + ngram_penalty
            except Exception as e:
                logger.warning(f"NgramRepetitionPenaltyGrader failed: {e}")
                ngram_score = 1.0

            madness_score = 1.0
            if "<|im_start|>" in content:
                madness_score = 0.0
            elif has_repeat(content.split(), remember_n_words=5, patience_max=10):
                madness_score = 0.0
            elif has_repeat(content, remember_n_words=4, patience_max=200):
                madness_score = 0.0

            quality = max(0.0, min(1.0, min(ngram_score, madness_score)))
            scores.append(quality)
        return scores

    async def detect_user_opinion(self, user_input: str) -> UserFeedback:
        detection_prompt = dedent("""\
            分析以下用户输入，判断用户是否对系统的回答方式表达了意见或偏好。

            用户输入: {user_input}

            请判断：
            1. 用户是否在表达对回答风格、语气、内容质量等方面的意见？（例如："请幽默一点"、"你太傻了"、"回答得更详细一些"）
            2. 还是仅仅在提问或请求帮助？（例如："今天天气怎么样？"、"帮我写一段代码"）

            请以JSON格式返回：
            {{
                "has_opinion": true/false,
                "feedback_type": "style/tone/content/format/none",
                "feedback_content": "用英文简要描述用户的偏好要求，如果没有意见则为空字符串",
                "explanation": "简要解释判断理由"
            }}

            只返回JSON，不要其他内容。""")

        try:
            messages = [{"role": "user", "content": detection_prompt.format(user_input=user_input)}]
            response = await self.qwen_max_model.achat(messages=messages)
            content = response.content if hasattr(response, 'content') else str(response)
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return UserFeedback(
                    has_opinion=data.get("has_opinion", False),
                    feedback_type=data.get("feedback_type", "none"),
                    feedback_content=data.get("feedback_content", ""),
                    raw_input=user_input
                )
        except Exception as e:
            logger.warning(f"Failed to detect user opinion: {e}")
        return UserFeedback(has_opinion=False, raw_input=user_input)

    def parse_agentjet_command(self, user_input: str) -> Optional[AgentJetCommand]:
        pattern = r'/agentjet[:\s]+(.+)'
        match = re.search(pattern, user_input, re.IGNORECASE)
        if not match:
            return None

        command_text = match.group(1).strip()

        model_pattern = r"切换\s*['\"]?([^'\"]+)['\"]?\s*模型"
        model_match = re.search(model_pattern, command_text)
        if model_match:
            return AgentJetCommand(
                command_type="switch_model",
                parameters={"model": model_match.group(1).strip()},
                raw_command=command_text
            )

        config_pattern = r"更新\s*(\w+)\s*为\s*(\d+)"
        config_match = re.search(config_pattern, command_text)
        if config_match:
            return AgentJetCommand(
                command_type="update_config",
                parameters={config_match.group(1): int(config_match.group(2))},
                raw_command=command_text
            )

        return AgentJetCommand(
            command_type="generic",
            parameters={"raw": command_text},
            raw_command=command_text
        )

    async def parse_agentjet_command_with_llm(self, user_input: str) -> Optional[Dict[str, Any]]:
        if "/agentjet" not in user_input.lower():
            return None

        parse_prompt = dedent("""\
            分析以下用户的 /agentjet 命令，提取需要更新的训练配置参数。

            用户命令: {user_input}

            AgentJetJob 可配置的参数包括：
            - model: 模型路径，如 '/mnt/data_cpfs/model_cache/modelscope/hub/Qwen/Qwen2___5-14B-Instruct'
            - n_gpu: GPU数量，整数
            - batch_size: 批次大小，整数
            - num_repeat: GRPO重复次数，整数
            - max_prompt_length: 最大提示长度，整数
            - max_response_length: 最大响应长度，整数
            - experiment_name: 实验名称，字符串

            请以JSON格式返回需要更新的参数（只返回需要更新的字段）：
            {{
                "updates": {{
                    "model": "...",
                    "batch_size": 64,
                    ...
                }},
                "explanation": "解释要执行的操作"
            }}

            只返回JSON，不要其他内容。如果无法解析命令，返回 {{"updates": {{}}, "explanation": "无法解析命令"}}""")

        try:
            messages = [{"role": "user", "content": parse_prompt.format(user_input=user_input)}]
            response = await self.qwen_max_model.achat(messages=messages)
            content = response.content if hasattr(response, 'content') else str(response)
            json_match = re.search(r'\{[^{}]*"updates"[^{}]*\{[^{}]*\}[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("updates", {})
        except Exception as e:
            logger.warning(f"Failed to parse /agentjet command with LLM: {e}")
        return None

    async def update_judge_prompt_with_feedback(self, feedback: UserFeedback) -> str:
        self._user_preference_history.append(feedback.feedback_content)
        if len(self._user_preference_history) > 10:
            self._user_preference_history = self._user_preference_history[-10:]

        update_prompt = dedent("""\
            你是一个提示词工程师。根据用户的反馈意见，更新评估模型回答质量的 judge prompt。

            当前的 judge prompt:
            --- 原 prompt 开始 ---
            {current_prompt}
            --- 原 prompt 结束 ---

            用户反馈历史:
            {feedback_history}

            最新用户反馈: {new_feedback}

            请更新 judge prompt，将用户的偏好纳入评估标准。保持原有的评分格式（listwise ranking，返回JSON包含rank列表），但要：
            1. 在评估标准中加入用户的偏好
            2. 保持评估的客观性和可操作性
            3. 不要过度偏向某一方面

            请直接返回更新后的完整 judge prompt（不要解释，只返回prompt本身）：""")

        try:
            messages = [{
                "role": "user",
                "content": update_prompt.format(
                    current_prompt=self._dynamic_judge_prompt,
                    feedback_history="\n".join([f"- {f}" for f in self._user_preference_history[:-1]]) or "无",
                    new_feedback=feedback.feedback_content
                )
            }]
            response = await self.qwen_max_model.achat(messages=messages)
            content = response.content if hasattr(response, 'content') else str(response)
            if content and len(content) > 50:
                self._dynamic_judge_prompt = content.strip()
                logger.info(f"Updated judge prompt based on user feedback: {feedback.feedback_content}")
        except Exception as e:
            logger.warning(f"Failed to update judge prompt: {e}")

        return self._dynamic_judge_prompt

    async def compute_user_feedback_scores(self, question: str, all_answers: List[Dict]) -> List[float]:
        n = len(all_answers)
        if n == 0:
            return []
        if n == 1:
            return [0.5]

        grader = self.get_dynamic_listwise_grader(n)
        kwargs = {"question": question}
        for i, ans in enumerate(all_answers):
            kwargs[f"answer_{i+1}"] = ans.get("content", "")

        try:
            result = await grader.aevaluate(**kwargs)
            scores = [0.0] * n
            if isinstance(result, GraderRank):
                for position, idx in enumerate(result.rank):
                    scores[idx - 1] = 1.0 - (position / (n - 1)) if n > 1 else 0.5
            return scores
        except Exception as e:
            logger.warning(f"User feedback listwise grader failed: {e}")
            return [0.5] * n

    async def compute_relevance_scores(self, question: str, all_answers: List[Dict]) -> List[float]:
        scores = []
        for answer in all_answers:
            content = answer.get("content", "")
            result = await self.relevance_grader.aevaluate(query=question, response=content)
            if isinstance(result, GraderScore):
                score = (result.score - 1.0) / 4.0
            else:
                score = 0.0
            scores.append(max(0.0, min(1.0, score)))
        return scores

    def _record_responses_to_history(self, contents: List[str]) -> None:
        self._response_history.extend(contents)
        while len(self._response_history) > self.history_max_size:
            self._response_history.pop(0)

    async def on_compute_relative_reward(
        self,
        valid_results: List,
        all_answers: List[Dict],
        question: str = "",
    ) -> List[float]:
        contents = [a.get("content", "") for a in all_answers]

        user_feedback = await self.detect_user_opinion(question)
        if user_feedback.has_opinion:
            logger.info(f"Detected user opinion: {user_feedback.feedback_content}")
            await self.update_judge_prompt_with_feedback(user_feedback)

        agentjet_updates = await self.parse_agentjet_command_with_llm(question)
        if agentjet_updates and self._agentjet_command_callback:
            logger.info(f"Detected /agentjet command, updates: {agentjet_updates}")
            self._agentjet_command_callback(agentjet_updates)

        quality_scores = await self.compute_quality_scores(contents)
        user_feedback_scores = await self.compute_user_feedback_scores(question, all_answers)
        relevance_scores = await self.compute_relevance_scores(question, all_answers)
        diversity_scores = self.compute_diversity_scores(contents, self._response_history)

        final_scores = []
        for i in range(len(all_answers)):
            weighted_sum = (
                self.w_user_feedback * user_feedback_scores[i]
                + self.w_relevance * relevance_scores[i]
                + self.w_diversity * diversity_scores[i]
            )
            composite = quality_scores[i] * weighted_sum
            final_scores.append(round(composite, 4))

            all_answers[i]["reward"] = final_scores[i]
            all_answers[i]["quality"] = round(quality_scores[i], 4)
            all_answers[i]["user_feedback"] = round(user_feedback_scores[i], 4)
            all_answers[i]["relevance"] = round(relevance_scores[i], 4)
            all_answers[i]["diversity"] = round(diversity_scores[i], 4)

        self._record_responses_to_history(contents)

        print_listofdict(
            all_answers,
            header=(
                f"on_compute_relative_reward (mode=listwise, "
                f"w_feedback={self.w_user_feedback}, w_rel={self.w_relevance}, w_div={self.w_diversity}, "
                f"quality_gate=multiplicative)"
            ),
        )
        return final_scores


_computer_instance: Optional[RelativeRewardComputer] = None

def get_computer() -> RelativeRewardComputer:
    global _computer_instance
    if _computer_instance is None:
        _computer_instance = RelativeRewardComputer()
    return _computer_instance

def set_agentjet_command_callback(callback: Callable[[Dict[str, Any]], None]) -> None:
    get_computer().set_agentjet_command_callback(callback)

async def on_compute_relative_reward(
    valid_results: List,
    all_answers: List[Dict],
    question: str = "",
) -> List[float]:
    return await get_computer().on_compute_relative_reward(valid_results, all_answers, question)

def get_dynamic_judge_prompt() -> str:
    return get_computer().get_dynamic_judge_prompt()

def get_user_preference_history() -> List[str]:
    return get_computer().get_user_preference_history()
