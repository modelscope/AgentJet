# flake8: noqa: W605
import re
from functools import cache

# Regex fragments for each whitelist category
WHITE_LIST_REGEX_PARTS = {
    # Common symbols
    "common_symbols": "‘’“”–—…•™©®°±µ′″℉℃·×",
    # Chinese punctuation
    "chinese_punct": "，。！？、；：“”‘’（）【】《》（）——……「」『』",
    # Emoji ranges
    "emoji": (
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\u2702-\u27B0"
        "\u24C2-\U0001F251"
    ),
    # Chinese characters
    "chinese": (
        "\u4E00-\u9FFF"
        "\u3400-\u4DBF"
        "\U00020000-\U0002A6DF"
        "\U0002A700-\U0002B73F"
        "\U0002B740-\U0002B81F"
        "\U0002B820-\U0002CEAF"
        "\uF900-\uFAFF"
        "\U0002F800-\U0002FA1F"
    ),
}


@cache
def build_pattern(white_list):
    """Build a regex based on the provided whitelist categories."""
    allowed_parts = ["\x00-\x7F"]  # All ASCII
    for name in white_list:
        if name in WHITE_LIST_REGEX_PARTS:
            allowed_parts.append(WHITE_LIST_REGEX_PARTS[name])
    # Merge allowed ranges into one character class, then use a negated class to match disallowed characters
    allowed_class = "".join(allowed_parts)
    pattern = f"[^{allowed_class}]"  # Match disallowed characters
    return re.compile(pattern)


def has_non_ascii(text, white_list=("common_symbols", "emoji", "chinese", "chinese_punct")):
    pattern = build_pattern(white_list)
    return bool(pattern.search(text))


def has_repeat(token, remember_n_words=5, patience_max=10):
    record_words = []
    patience = patience_max
    for char in token:
        if char not in record_words:
            record_words += [char]
            if len(record_words) > remember_n_words:
                record_words = record_words[1:]
            patience = patience_max
        else:
            patience -= 1
            if patience <= 0:
                return True
    return False


def compute_string_madness(completion, detail=False, checklist=["nonsense"]) -> float:
    all_reward = 0.0
    if ("nonsense" in checklist) and ("non_ascii" in checklist):
        all_reward += compute_string_madness_char(completion, detail=detail)
    elif ("nonsense" in checklist) and ("non_ascii" not in checklist):
        all_reward += compute_string_madness_char(completion, detail=detail, skip_non_ascii=True)
    if "format_type_1" in checklist:
        all_reward += compute_string_madness_format(completion, detail=detail, format_type="type_1")

    return all_reward


def compute_string_madness_format(completion, detail, format_type) -> float:
    if format_type == "type_1":
        """

        <think> ... </think>

        ```python
        code
        ```

        """
        # Check that <think> and </think> appear exactly once and in order
        if not completion.strip().startswith(r"<think>"):
            # print("not start with <think>")
            return -1.0
        if completion.count(r"<think>") != 1 or completion.count(r"</think>") != 1:
            # print("not one think")
            return -1.0
        if completion.index(r"<think>") > completion.index(r"</think>"):
            # print("think tag order wrong")
            return -1.0
        # remove think part
        think_part = completion[
            completion.index(r"<think>") : completion.index(r"</think>") + len(r"</think>")
        ]
        rest_part = completion.replace(think_part, "")
        # Check that ```python and ``` appear exactly once and in order
        if not rest_part.strip().startswith(r"```python"):
            # print("not start with ```python")
            return -1.0
        if not rest_part.strip().endswith(r"```"):
            # print("not end with ```")
            return -1.0
        if rest_part.count(r"```python") != 1 or rest_part.count(r"```") != 2:
            # print("not one ```python")
            return -1.0
        if rest_part.index(r"```python") > rest_part.rindex(r"```"):
            # print("``` tag order wrong")
            return -1.0
        return 0.0
    else:
        raise NotImplementedError(f"format_type {format_type} not implemented")


def compute_string_madness_char(completion, detail=False, skip_non_ascii=False) -> float:
    # if detail:
    #     result = {
    #         "has_non_ascii": has_non_ascii(completion),
    #         "has_repeat": has_repeat(completion.split(), remember_n_words=5, patience_max=10),
    #         "has_repeat_x": has_repeat(completion, remember_n_words=4, patience_max=200),
    #         "has_wrong_sp_token": "<|im_start|>" in completion,
    #         # 'non_ascii': {ch for ch in completion if ord(ch) > 127}
    #     }
    #     if has_non_ascii(completion):
    #         for char in completion:
    #             if has_non_ascii(char):
    #                 print(f"---")
    #                 print(f"found non-ascii char: {char} ord={ord(char)}")
    #     print(result)
    #     return result

    if "<|im_start|>" in completion:
        return -1.0

    if skip_non_ascii:
        if has_non_ascii(completion):
            return -1.0

    if has_repeat(completion.split(), remember_n_words=5, patience_max=10):
        return -1.0

    if has_repeat(completion, remember_n_words=4, patience_max=200):
        return -1.0

    return 0


def repetition_penalty_reward_scalar_debug(completion):
    for i in range(len(completion)):
        p = completion[:i]
        result = compute_string_madness(p)
        if result != 0:
            return completion
    return ""


