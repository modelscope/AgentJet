import re
from typing import List


def preserve_chinese(text):
    # Use regular expressions to match all Chinese characters
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    # Concatenate the matched Chinese characters into a string
    return "".join(chinese_chars)


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram_chinese(text: str, ngram_size: int):
        import jieba

        text = preserve_chinese(text)
        seg_list = list(jieba.cut(text))
        # print(seg_list)
        return zip(*[seg_list[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram_chinese(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            if total == 0:
                rewards.append(0.0)
                continue

            # total is a fixed value (sentence tokenization length)
            # Assuming no repetitions at all, total = len(ngrams), scaling ~= 0, reward = 0
            # Assuming full repetition, len(ngrams) = 1, scaling ~= 1, reward = -1
            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward

