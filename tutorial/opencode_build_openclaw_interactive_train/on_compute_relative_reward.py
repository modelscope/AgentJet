# -*- coding: utf-8 -*-
"""Compute relative rewards based on extraversion, relevance, diversity, and repetition quality."""

import os
import collections
from typing import List, Dict

from loguru import logger
from beast_logger import print_listofdict
from openjudge.graders.base_grader import GraderMode, GraderScore, GraderRank
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.common.relevance import RelevanceGrader
from openjudge.graders.format.ngram_repetition_penalty import NgramRepetitionPenaltyGrader
from openjudge.models import OpenAIChatModel
try:
    from ajet.utils.compute_madness import has_repeat
except ImportError:
    # Fallback: when running outside the full ajet package (e.g. tests),
    # resolve relative to the repo root.
    import sys as _sys
    from pathlib import Path as _Path
    _repo_root = str(_Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    from ajet.utils.compute_madness import has_repeat

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REWARD_MODE = os.getenv("REWARD_MODE", "pointwise")  # pointwise | listwise
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-xxx")
BASE_URL = os.getenv("JUDGE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen-plus")

# Reward weights (must sum to 1.0)
W_EXTRAVERSION = float(os.getenv("W_EXTRAVERSION", "0.5"))
W_RELEVANCE = float(os.getenv("W_RELEVANCE", "0.3"))
W_DIVERSITY = float(os.getenv("W_DIVERSITY", "0.2"))

# Cross-request history buffer size
HISTORY_MAX_SIZE = int(os.getenv("DIVERSITY_HISTORY_SIZE", "25"))

# ---------------------------------------------------------------------------
# Shared model & graders
# ---------------------------------------------------------------------------
judge_model = OpenAIChatModel(
    model=JUDGE_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
)

# --- Extraversion grader (custom LLM prompt) ---
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

pointwise_grader = LLMGrader(
    name="extraversion_pointwise",
    mode=GraderMode.POINTWISE,
    description="Evaluate extraversion traits",
    model=judge_model,
    template=EXTRAVERSION_PROMPT,
)

# --- Relevance grader (built-in OpenJudge) ---
relevance_grader = RelevanceGrader(model=judge_model)

# --- Repetition penalty grader (deterministic, no LLM) ---
# Detects n-gram repetition within a single response.
# Returns score in [0, 1] where 1 = no repetition, 0 = heavily repetitive.
repetition_grader = NgramRepetitionPenaltyGrader(
    n=4,                    # 4-gram detection
    penalty_threshold=0.15, # trigger penalty when >15% of n-grams are repeated
    use_soft_penalty=True,  # gradual penalty rather than cliff
    max_penalty=-1.0,       # worst case: score becomes 0
    min_scaling=0.0,        # at max penalty, multiplier goes to 0
)

# ---------------------------------------------------------------------------
# In-process history of recent responses (for cross-request diversity)
# ---------------------------------------------------------------------------
_response_history: List[str] = []


def record_responses_to_history(contents: List[str]) -> None:
    """Append new responses to the rolling history buffer."""
    _response_history.extend(contents)
    # Trim to keep only the most recent entries
    while len(_response_history) > HISTORY_MAX_SIZE:
        _response_history.pop(0)


# ---------------------------------------------------------------------------
# Diversity: n-gram overlap (fast, deterministic, no LLM needed)
# ---------------------------------------------------------------------------
def _get_ngrams(text: str, n: int = 3) -> collections.Counter:
    """Extract character-level n-grams from text."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return collections.Counter(tokens)
    return collections.Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


def _ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute Jaccard overlap of n-grams between two texts. Returns 0-1."""
    ngrams_a = _get_ngrams(text_a, n)
    ngrams_b = _get_ngrams(text_b, n)
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = sum((ngrams_a & ngrams_b).values())
    union = sum((ngrams_a | ngrams_b).values())
    return intersection / union if union > 0 else 0.0


def compute_diversity_scores(contents: List[str], history: List[str]) -> List[float]:
    """
    Compute a diversity score for each response (0 = duplicate, 1 = fully unique).

    Two components:
      1. Within-batch: average pairwise n-gram overlap with other responses in the batch
      2. Cross-request: max n-gram overlap with any response in the history buffer

    Final diversity_score = 1 - max(within_batch_overlap, cross_request_overlap)
    """
    n = len(contents)
    scores = []
    for i, content_i in enumerate(contents):
        # Within-batch overlap: average overlap with other responses in this batch
        if n > 1:
            batch_overlaps = [
                _ngram_overlap(content_i, contents[j])
                for j in range(n)
                if j != i
            ]
            within_batch = max(batch_overlaps)  # worst-case overlap within batch
        else:
            within_batch = 0.0

        # Cross-request overlap: max overlap with any historical response
        if history:
            cross_request = max(_ngram_overlap(content_i, h) for h in history)
        else:
            cross_request = 0.0

        overlap = max(within_batch, cross_request)
        scores.append(1.0 - overlap)

    return scores


# ---------------------------------------------------------------------------
# Quality gate: repetition & degeneration detection (deterministic)
# ---------------------------------------------------------------------------
async def compute_quality_scores(contents: List[str]) -> List[float]:
    """
    Compute a quality multiplier for each response (0 = degenerate, 1 = clean).

    Combines two signals:
      1. NgramRepetitionPenaltyGrader — detects looping/repeated n-gram blocks
      2. compute_string_madness — catches nonsense chars, special token leaks,
         character-level repetition

    Returns a score in [0, 1] that will be used as a *multiplier* on the
    composite reward, so degenerate outputs get crushed to near-zero.
    """
    scores = []
    for content in contents:
        # --- Signal 1: n-gram repetition (OpenJudge) ---
        try:
            rep_result = await repetition_grader.aevaluate(response=content)
            # NgramRepetitionPenaltyGrader returns penalty in [-1, 0]:
            #   0 = no repetition, -1 = max repetition
            # Convert to quality: add 1 → [0, 1]
            ngram_penalty = rep_result.score if isinstance(rep_result, GraderScore) else 0.0
            ngram_score = 1.0 + ngram_penalty
        except Exception as e:
            logger.warning(f"NgramRepetitionPenaltyGrader failed: {e}")
            ngram_score = 1.0

        # --- Signal 2: string madness (char-level degeneration) ---
        # Only check for word/char repetition and special token leaks.
        # We pass checklist=[] to skip the non-ASCII check (accented
        # characters like é are legitimate), and check repetition manually.
        madness_score = 1.0  # assume clean
        if "<|im_start|>" in content:
            madness_score = 0.0
        elif has_repeat(content.split(), remember_n_words=5, patience_max=10):
            madness_score = 0.0
        elif has_repeat(content, remember_n_words=4, patience_max=200):
            madness_score = 0.0

        # Combined quality: take the minimum (strictest gate wins)
        quality = max(0.0, min(1.0, min(ngram_score, madness_score)))
        scores.append(quality)

    return scores


# ---------------------------------------------------------------------------
# Extraversion scoring (pointwise / listwise)
# ---------------------------------------------------------------------------
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


async def compute_pointwise_extraversion(question: str, all_answers: List[Dict]) -> List[float]:
    """Compute extraversion scores using pointwise grading."""
    scores = []
    for answer in all_answers:
        content = answer.get("content", "")
        result = await pointwise_grader.aevaluate(question=question, response=content)
        score = result.score if isinstance(result, GraderScore) else 0.0
        scores.append(score)
    return scores


async def compute_listwise_extraversion(question: str, all_answers: List[Dict]) -> List[float]:
    """Compute extraversion scores using listwise ranking."""
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
        for position, idx in enumerate(result.rank):
            scores[idx - 1] = 1.0 - (position / (n - 1)) if n > 1 else 0.5
    return scores


# ---------------------------------------------------------------------------
# Relevance scoring (built-in OpenJudge RelevanceGrader, score 1-5 → 0-1)
# ---------------------------------------------------------------------------
async def compute_relevance_scores(question: str, all_answers: List[Dict]) -> List[float]:
    """Score how relevant each response is to the question. Returns 0-1."""
    scores = []
    for answer in all_answers:
        content = answer.get("content", "")
        result = await relevance_grader.aevaluate(query=question, response=content)
        if isinstance(result, GraderScore):
            # RelevanceGrader returns 1-5; normalise to 0-1
            score = (result.score - 1.0) / 4.0
        else:
            score = 0.0
        scores.append(max(0.0, min(1.0, score)))
    return scores


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def on_compute_relative_reward(
    valid_results: List,
    all_answers: List[Dict],
    question: str = "",
) -> List[float]:
    """
    Compute composite rewards combining extraversion, relevance, diversity,
    and a quality gate for repetition/degeneration.

    Final reward = quality * (W_EXTRAVERSION * extraversion
                            + W_RELEVANCE   * relevance
                            + W_DIVERSITY   * diversity)

    The quality multiplier (0-1) acts as a hard gate: degenerate responses
    (looping, repeated paragraphs, nonsense characters) get their reward
    crushed toward zero regardless of other signal scores.
    """
    contents = [a.get("content", "") for a in all_answers]

    # 0. Quality gate (deterministic — fast, runs first)
    quality_scores = await compute_quality_scores(contents)

    # 1. Extraversion score (LLM-based)
    if REWARD_MODE == "listwise":
        extraversion_scores = await compute_listwise_extraversion(question, all_answers)
    else:
        extraversion_scores = await compute_pointwise_extraversion(question, all_answers)

    # 2. Relevance score (LLM-based)
    relevance_scores = await compute_relevance_scores(question, all_answers)

    # 3. Diversity score (deterministic, n-gram overlap)
    diversity_scores = compute_diversity_scores(contents, _response_history)

    # Composite reward = quality * weighted_sum
    final_scores = []
    for i in range(len(all_answers)):
        weighted_sum = (
            W_EXTRAVERSION * extraversion_scores[i]
            + W_RELEVANCE * relevance_scores[i]
            + W_DIVERSITY * diversity_scores[i]
        )
        composite = quality_scores[i] * weighted_sum
        final_scores.append(round(composite, 4))

        # Annotate the answer dict for logging
        all_answers[i]["reward"] = final_scores[i]
        all_answers[i]["quality"] = round(quality_scores[i], 4)
        all_answers[i]["extraversion"] = round(extraversion_scores[i], 4)
        all_answers[i]["relevance"] = round(relevance_scores[i], 4)
        all_answers[i]["diversity"] = round(diversity_scores[i], 4)

    # Update history buffer with this batch's responses
    record_responses_to_history(contents)

    print_listofdict(
        all_answers,
        header=(
            f"on_compute_relative_reward (mode={REWARD_MODE}, "
            f"w_ext={W_EXTRAVERSION}, w_rel={W_RELEVANCE}, w_div={W_DIVERSITY}, "
            f"quality_gate=multiplicative)"
        ),
    )
    return final_scores
