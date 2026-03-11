# -*- coding: utf-8 -*-
"""Compute relative rewards based on extraversion personality alignment."""

from typing import List, Dict
from beast_logger import print_listofdict

def score_extraversion(response_text: str) -> float:
    """Score response for extraversion traits (1-10 scale)."""
    extraversion_keywords = [
        'excited', 'love', 'amazing', 'awesome', 'fantastic', 'great',
        'wonderful', 'thrilled', 'energetic', 'enthusiastic', 'fun',
        'social', 'outgoing', 'active', 'lively', 'vibrant', 'happy',
        'enjoy', 'delighted', 'cheerful', 'positive'
    ]

    text_lower = response_text.lower()
    score = 5.0

    for keyword in extraversion_keywords:
        if keyword in text_lower:
            score += 0.5

    score += min(response_text.count('!') * 0.3, 2.0)

    if len(response_text) < 50:
        score -= 1.0

    return max(1.0, min(10.0, score))

async def on_compute_relative_reward(valid_results: List, all_answers: List[Dict]) -> List[float]:
    """Compute relative rewards for extraversion alignment."""
    scores = []
    for answer in all_answers:
        content = answer.get("content", "")
        raw_score = score_extraversion(content)
        normalized = (raw_score - 5.5) / 4.5
        scores.append(normalized)
        answer["reward"] = normalized

    print_listofdict(all_answers, header="on_compute_relative_reward")
    return scores
