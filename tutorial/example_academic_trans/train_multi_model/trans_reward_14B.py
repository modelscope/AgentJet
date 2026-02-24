import re
import json
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from typing import List
from textwrap import dedent


# Examples of proper noun detection task
proper_noun_examples = [
    {
        "original": "We find that the EMBB is dominated by GW bursts from stellar mass black holes",
        "translation": "我们发现，EMBB主要由恒星级黑洞发出的GWs爆发主导",
        "good_detection": json.dumps([
            {
                "original_word": "We",
                "wrong_translation": "我们",
                "wrong_reason": "使用了第一人称代词，应改为'本研究'或'本文'",
                "correct_translation": "本研究"
            },
            {
                "original_word": "GWs",
                "wrong_translation": "GWs",
                "wrong_reason": "GWs有简洁的中文表达'引力波'，不应使用英文缩写",
                "correct_translation": "引力波"
            }
        ], ensure_ascii=False),
        "hint": "应检测出两个错误：1）第一人称代词'我们' 2）GWs应翻译为'引力波'"
    },
    {
        "original": "To improve the transferability of ViT, we introduce a novel and effective module, named Domain Transferable-guided Attention Block (DTAB).",
        "translation": "为了提高ViT的迁移能力，本文引入了一个新颖且有效的模块，称为域可迁移引导注意力块（DTAB）",
        "good_detection": json.dumps([
            {
                "original_word": "DTAB",
                "wrong_translation": "域可迁移引导注意力块（DTAB）",
                "wrong_reason": "首次出现自定义缩写时，应该提供英文全称。应为'域可迁移引导注意力块（Domain Transferable-guided Attention Block，DTAB）'",
                "correct_translation": "域可迁移引导注意力块（Domain Transferable-guided Attention Block，DTAB）"
            }
        ], ensure_ascii=False),
        "hint": "应检测出缩写问题：首次出现DTAB时未提供英文全称"
    }
]


PROPER_NOUN_DETECTION_USER_PROMPT = """
Evaluate the quality of proper noun error detection based on the specific task requirements.

Original English text:
{original}

Chinese translation (contains errors):
{translation}

Agent's detection result (JSON format):
{detection_result}

The agent's task is to detect translation errors of discipline-specific proper nouns and provide corrections in JSON format.
"""


def get_proper_noun_detection_system_prompt() -> str:
    """Get the proper noun detection quality system prompt."""
    examples_text = ""
    for i, ex in enumerate(proper_noun_examples, 1):
        examples_text += dedent(f"""
            Example {i}:
            - Original: "{ex['original']}"
            - Translation: "{ex['translation']}"
            - Good Detection: {ex['good_detection']}
            - Hint: {ex['hint']}
        """)

    return dedent("""
        You are evaluating the performance of an agent (14B model) whose task is to detect translation errors of discipline-specific proper nouns.

        The agent receives:
        1. Original English text
        2. A rough Chinese translation (may contain errors)

        The agent should output a JSON list of detected errors in this format:
        [{"original_word": "xxx", "wrong_translation": "xxx", "wrong_reason": "xxx", "correct_translation": "xxx"}, ...]

        Evaluation criteria for the 14B model (Agent 2):

        **High-quality detection should:**
        1. **Identify all critical errors** - Detect errors in:
           - First-person pronouns (我/我们)
           - Abbreviation translation issues (using abbreviations when Chinese is concise, or vice versa)
           - Missing full form for custom abbreviations (e.g., DTAB should have full English form on first mention)
           - Discipline-specific proper noun errors (technical terms, domain terminology)
           - Word choice issues (colloquial vs. academic)

        2. **Provide accurate corrections** - Each detected error should have:
           - Correct identification of the original word
           - Accurate description of what's wrong
           - Appropriate correction

        3. **Avoid false positives** - Don't flag correct translations as errors

        4. **Use proper JSON format** - Output should be valid JSON that can be parsed

        **Examples of good detection:**
        [[examples_text]]

        Rate the detection quality on a scale of 0-2:

        0 = Poor detection (missed critical errors, many false positives, or invalid JSON format)
        1 = Acceptable detection (caught some errors but missed important ones, or had some false positives)
        2 = Excellent detection (caught all major errors, accurate corrections, minimal false positives, proper JSON format)

        Evaluation approach:
        * Parse the agent's JSON output (if invalid JSON, automatic score 0)
        * Compare with the actual errors in the translation
        * Check if major error types are detected:
          - First-person pronouns (critical)
          - Abbreviation issues (important)
          - Custom abbreviation full forms (important)
          - Proper noun translation errors (critical)
        * Assess correction quality - are the suggested corrections appropriate?
        * Check for false positives - did the agent flag things that aren't actually errors?

        Return your response in this format:
        <reasoning>
        Your analysis of the detection quality
        </reasoning>
        <detected_errors>
        List of errors the agent successfully detected
        </detected_errors>
        <missed_errors>
        List of errors the agent should have detected but missed
        </missed_errors>
        <false_positives>
        List of items incorrectly flagged as errors
        </false_positives>
        <score>X</score>

        The score must be 0, 1, or 2.
    """.replace("[[examples_text]]", examples_text))


def parse_proper_noun_detection_response(text: str) -> dict:
    """Parse XML-formatted proper noun detection evaluation response."""
    score_match = re.search(r"<score>\s*(\d+)\s*</score>", text)
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    detected_match = re.search(r"<detected_errors>(.*?)</detected_errors>", text, re.DOTALL)
    missed_match = re.search(r"<missed_errors>(.*?)</missed_errors>", text, re.DOTALL)
    false_pos_match = re.search(r"<false_positives>(.*?)</false_positives>", text, re.DOTALL)

    score = int(score_match.group(1)) if score_match else 0
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text

    def parse_list(match_obj):
        if match_obj:
            items_text = match_obj.group(1)
            return [
                line.strip().lstrip("- ")
                for line in items_text.strip().split("\n")
                if line.strip() and not line.strip().lstrip("- ").lower().startswith("none")
            ]
        return []

    detected_errors = parse_list(detected_match)
    missed_errors = parse_list(missed_match)
    false_positives = parse_list(false_pos_match)

    return {
        "score": score,
        "reason": reasoning,
        "detected_errors": detected_errors,
        "missed_errors": missed_errors,
        "false_positives": false_positives
    }


def build_proper_noun_detection_messages(original_text: str, translation: str, detection_result: str) -> List[dict]:
    """Build messages for evaluating proper noun detection quality."""
    return [
        {
            "role": "system",
            "content": get_proper_noun_detection_system_prompt()
        },
        {
            "role": "user",
            "content": PROPER_NOUN_DETECTION_USER_PROMPT.format(
                original=original_text,
                translation=translation,
                detection_result=detection_result
            ),
        },
    ]


class ProperNounDetectionGrader(LLMGrader):
    """Grader for evaluating the 14B model's proper noun detection quality."""

    def __init__(self, model: BaseChatModel | dict):
        super().__init__(
            name="proper_noun_detection_quality",
            mode=GraderMode.POINTWISE,
            description="Evaluate the quality of proper noun error detection by the 14B model",
            model=model,
            template="",  # Placeholder, not used
        )

    async def aevaluate(self, original_text: str, translation: str, detection_result: str, normalize=True) -> GraderScore:
        """
        Evaluate the quality of proper noun error detection.

        Args:
            original_text: Original English text
            translation: Rough Chinese translation (may contain errors)
            detection_result: The 14B model's detection output (JSON string)
            normalize: Whether to normalize score to [0, 1] range

        Returns:
            GraderScore with score in range [0, 1] if normalized, else [0, 2]
        """
        try:
            messages = build_proper_noun_detection_messages(original_text, translation, detection_result)
            response = await self.model.achat(messages=messages)
            content = await extract_response_content(response)
            parsed = parse_proper_noun_detection_response(content)

            if normalize:
                parsed["score"] = parsed["score"] / 2.0

            return GraderScore(
                name=self.name,
                score=parsed["score"],
                reason=parsed["reason"],
                metadata={
                    "detected_errors": parsed["detected_errors"],
                    "missed_errors": parsed["missed_errors"],
                    "false_positives": parsed["false_positives"]
                },
            )
        except Exception as e:
            return GraderError(name=self.name, error=str(e))


async def extract_response_content(response) -> str:
    """Extract content from various response formats."""
    if hasattr(response, 'content'):
        return response.content
    elif isinstance(response, dict) and 'content' in response:
        return response['content']
    elif isinstance(response, str):
        return response
    else:
        raise ValueError(f"Unable to extract content from response: {type(response)}")
