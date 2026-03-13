# -*- coding: utf-8 -*-
"""Compute relative rewards based on extraversion personality alignment using OpenJudge."""

import os
from typing import List, Dict
from beast_logger import print_listofdict
from openjudge.graders.base_grader import GraderMode, GraderScore, GraderRank
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models import OpenAIChatModel

# Configuration
REWARD_MODE = os.getenv("REWARD_MODE", "pointwise")  # Options: pointwise, listwise
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-xxx")
BASE_URL = os.getenv("JUDGE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen-plus")

# OpenJudge grader setup
judge_model = OpenAIChatModel(
    model=JUDGE_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
)

EXTRAVERSION_PROMPT = """You are evaluating responses for extraversion personality traits.

Extraversion characteristics include:
- Outgoing, energetic, enthusiastic tone
- Social engagement and excitement
- Positive, upbeat language
- Action-oriented expressions
- Use of exclamation marks and emotional words

Rate the response on a scale of 0.0-1.0:
0.0 = Highly introverted (reserved, quiet, minimal emotion)
1.0 = Highly extraverted (energetic, enthusiastic, very expressive)

Question: {question}
Response: {response}

Return a json object with exactly two fields:
- "score": float between 0.0 and 1.0
- "reason": brief explanation"""

def build_listwise_template(n: int) -> str:
    """Build a listwise prompt template for n responses."""
    answers_block = "\n".join([f"{i+1}. {{answer_{i+1}}}" for i in range(n)])
    return f"""You are ranking multiple responses based on extraversion personality traits.

Extraversion characteristics include:
- Outgoing, energetic, enthusiastic tone
- Social engagement and excitement
- Positive, upbeat language
- Action-oriented expressions

Question: {{question}}

Responses to rank:
{answers_block}

Rank these responses from most extraverted to least extraverted.
Return a json object with exactly two fields:
- "rank": list of integers (1-indexed) ordered from most to least extraverted, e.g. [2, 1, 3]
- "reason": brief explanation of the ranking"""

pointwise_grader = LLMGrader(
    name="extraversion_pointwise",
    mode=GraderMode.POINTWISE,
    description="Evaluate extraversion traits",
    model=judge_model,
    template=EXTRAVERSION_PROMPT,
)


async def compute_pointwise_rewards(question: str, all_answers: List[Dict]) -> List[float]:
    """Compute rewards using OpenJudge pointwise grading."""
    scores = []
    for answer in all_answers:
        content = answer.get("content", "")
        result = await pointwise_grader.aevaluate(question=question, response=content)
        if isinstance(result, GraderScore):
            # score is already normalized 0-1 by OpenJudge
            score = result.score
        else:
            score = 0.0
        scores.append(score)
        answer["reward"] = score
    return scores


async def compute_listwise_rewards(question: str, all_answers: List[Dict]) -> List[float]:
    """Compute rewards using OpenJudge listwise ranking."""
    n = len(all_answers)
    template = build_listwise_template(n)
    grader = LLMGrader(
        name="extraversion_listwise",
        mode=GraderMode.LISTWISE,
        description="Rank responses by extraversion",
        model=judge_model,
        template=template,
    )
    kwargs = {"question": question}
    for i, ans in enumerate(all_answers):
        kwargs[f"answer_{i+1}"] = ans.get("content", "")

    result = await grader.aevaluate(**kwargs)

    scores = [0.0] * n
    if isinstance(result, GraderRank):
        # rank is a list of 1-indexed positions ordered best to worst
        # convert to reward: rank 1 (best) -> 1.0, rank n (worst) -> 0.0
        for position, idx in enumerate(result.rank):
            scores[idx - 1] = 1.0 - (position / (n - 1)) if n > 1 else 0.5

    for answer, score in zip(all_answers, scores):
        answer["reward"] = score
    return scores


async def on_compute_relative_reward(valid_results: List, all_answers: List[Dict]) -> List[float]:
    """Compute relative rewards for extraversion alignment."""
    question = valid_results[0].get("question", "") if valid_results else ""

    if REWARD_MODE == "listwise":
        scores = await compute_listwise_rewards(question, all_answers)
    else:  # pointwise (default)
        scores = await compute_pointwise_rewards(question, all_answers)

    print_listofdict(all_answers, header=f"on_compute_relative_reward (mode={REWARD_MODE})")
    return scores
