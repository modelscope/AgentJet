import re
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from typing import List
from textwrap import dedent
from beast_logger import print_listofdict


examples = [
    {
        "original": "We find that the EMBB is dominated by GW bursts from stellar mass black holes",
        "bad": "我们发现，EMBB主要由恒星级黑洞发出的GWs爆发主导",
        "hint": "1) 我们->本研究/本文（删除第一人称） 2) GWs->引力波（有简洁的中文表达），但EMBB保留（没有简洁的中文表达） 3. 调换语序，这句话中的重点是“恒星级黑洞发出的引力波”，所以调换语序突出重点。",
        "good": "本研究发现恒星级黑洞发出的引力波爆发在EMBB中占主导地位"
    },
    {
        "original": "In a previous paper (Gayon &amp; Bois 2008a), we have shown the general efficiency of retrograde resonances for stabilizing compact planetary systems.",
        "bad": "在先前的一篇论文（Gayon & Bois 2008a）中，本文展示了逆向共振在稳定紧凑行星系统中的普遍效率。",
        "hint": "修复主语，删除冗余的逗号，替换“效率”为“有效性”更符合学术表达。",
        "good": "先前的一篇论文（Gayon & Bois 2008a）阐释了逆向共振在稳定紧凑行星系统中的普遍有效性。"
    },
    {
        "original": "To improve the transferability of ViT, we introduce a novel and effective module, named Domain Transferable-guided Attention Block (DTAB).",
        "bad": "为了提高ViT的迁移能力，本文引入了一个新颖且有效的模块，称为域可迁移引导注意力块（DTAB）",
        "hint": "1）语言顺序和表达不符合中文习惯 2）没有在首次出现自定义缩写时，给出英文全称",
        "good": "为提高ViT的迁移能力，本文引入了名为“域可迁移引导注意力块”（Domain Transferable-guided Attention Block，DTAB）的新颖且有效的模块。"
    },
    {
        "original": "Extensive experiments were conducted on UCF-HMDB, Kinetics-Gameplay, and Kinetics-NEC Drone datasets, with different backbones, like ResNet101, I3D, and STAM, to verify the effectiveness of TransferAttn compared with state-of-the-art approaches.",
        "bad": "在UCF-HMDB、Kinetics-Gameplay和Kinetics-NEC Drone数据集上进行了广泛的实验，使用了不同的骨干网络，如ResNet101、I3D和STAM，以验证TransferAttn与现有最先进方法相比的有效性。",
        "hint": "1）改变语言顺序后，主语缺失 2）举例时，表述不够简洁",
        "good": "本研究在UCF-HMDB、Kinetics-Gameplay和Kinetics-NEC Drone数据集上进行了广泛的实验，使用了ResNet101、I3D和STAM等骨干网络来验证TransferAttn与现有最先进方法相比的有效性。"
    }
]


examples_eval = examples + [

]



TRANSLATION_QUALITY_USER_PROMPT = """
Evaluate the quality of this Chinese translation based on the specific error types demonstrated in the examples.

Original English text:
{original}

Chinese translation to evaluate:
{translation}
"""



def get_translation_quality_system_prompt() -> str:
    """Get the translation quality system prompt."""
    examples_text = ""
    for i, ex in enumerate(examples_eval, 1):
        examples_text += dedent(f"""
            Example {i}:
            - Original: "{ex['original']}"
            - Bad Translation: "{ex['bad']}"
            - Issues: {ex['hint']}
            - Good Translation: "{ex['good']}"
        """)


    return dedent("""
            You are an objective translation quality evaluator for academic paper translations from English to Chinese. Your task is to identify ONLY the specific types of errors demonstrated in the provided examples - not general translation quality issues.

            重点关注（但不限于）以下问题类型（如示例所示）：

            1. **错误使用第一人称代词** - 禁止使用"我们"。正确的方法是使用"本研究"、"本文"、“研究者”，或者直接删除we并改写句子替换主语。不要漏掉出现的任何第一人称代词。
            2. **缩写翻译错误** - 当存在简洁的中文表达时使用缩写（例如，使用"GWs"而非"引力波"），或翻译本应保留英文的缩写（如"EMBB"）
            3. **语序问题** - 未调整句子结构以符合中文学术风格强调重点的习惯
            4. **主谓不一致、主语缺失** - 由于句子结构不当导致主语混乱（例如，"在...中，本文展示..."中主语混淆）
            5. **用词不当** - 使用口语化或不正确的术语而非恰当的学术表达
            6. **多余标点和停顿** - 不必要的逗号或其他标点符号影响中文阅读流畅性
            7. **主语不清晰** - 中文句子主语缺失或不明确。例如：“通过该实验，证明了该药物对癌细胞有抑制作用”（缺少主语）
            8. **缩写问题** - 首次出现自定义缩写、且原文中已经提供自定义缩写的英文全称时，没有在首次出现的地方提供英文全称。
                （正确的例子：`We have used the LAsMA heterodyne array installed on the Atacama Pathfinder EXperiment (APEX)`->`本研究使用了安装在阿塔卡马探路者实验望远镜（APEX, Atacama Pathfinder EXperiment）上的LAsMA外差阵列`）
            9. **专有名词翻译错误** - 领域特定的专有名词翻译错误，例如技术术语、学科术语等。如错把Agent翻译成“代理”（实际上应为“智能体”）等。
            10. **表意偏差** - 翻译结果与原文在意义上存在偏差，导致信息传达不准确。

            **Examples of these errors:**
            [[examples_text]]
            Rate the translation on a scale of 0-2:

            0 = Severely impairs readability (multiple critical errors from the categories above that make the text difficult to understand)
            1 = Contain errors, reduces Chinese reading efficiency (many instances of the error types above)
            2 = No errors from the example categories detected (translation is free of the specific error types demonstrated)

            Note:
            * For each key issue found, provide the specific error, its type, and where it appears in the translation.
            * Be precise about which error category each issue belongs to.
            * Focus on objective errors matching the example patterns, not subjective preferences.
            * 当出现  **语序问题**、**主谓不一致、主语缺失**、**主语不清晰**、**专有名词翻译错误**、**表意偏差** 等严重问题时，直接给 0 分。
            * 逐句分析，切勿遗漏。

            Think carefully before flagging any error. Ask yourself: Does this match one of the specific error types from the examples? Is this truly an objective error or just a stylistic preference?

            Return your response in this format:
            <reasoning>
            Your analysis
            </reasoning>
            <key_issues>
            - Error Type: [category]. Error: [specific issue]. Location: [where it appears in the translation]
            </key_issues>
            <score>X</score>

            The score must be 0, 1, 2. Each key issue should be on its own line starting with a dash. If no errors are found, the key_issues section should be empty or state "None detected".
        """.replace("[[examples_text]]", examples_text))



def parse_translation_quality_response(text: str) -> dict:
    """Parse XML-formatted translation quality response."""
    score_match = re.search(r"<score>\s*(\d+)\s*</score>", text)
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    issues_match = re.search(r"<key_issues>(.*?)</key_issues>", text, re.DOTALL)

    score = int(score_match.group(1)) if score_match else 0
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text

    key_issues = []
    if issues_match:
        issues_text = issues_match.group(1)
        # Filter out empty lines and "None detected" type messages
        key_issues = [
            line.strip().lstrip("- ")
            for line in issues_text.strip().split("\n")
            if line.strip() and not line.strip().lstrip("- ").lower().startswith("none")
        ]

    return {"score": score, "reason": reasoning, "key_issues": key_issues}


def build_translation_quality_messages(original_text: str, translation: str) -> List[dict]:
    return [
        {
            "role": "system",
            "content": get_translation_quality_system_prompt()
        },
        {
            "role": "user",
            "content": TRANSLATION_QUALITY_USER_PROMPT.format(
                original=original_text,
                translation=translation
            ),
        },
    ]


class TranslationQualityGrader(LLMGrader):
    def __init__(self, model: BaseChatModel | dict):
        super().__init__(
            name="translation_quality",
            mode=GraderMode.POINTWISE,
            description="Evaluate translation quality based on specific error patterns",
            model=model,
            template="",  # Placeholder, not used
        )

    async def aevaluate(self, original_text: str, translation: str, normalize=True) -> GraderScore:
        try:
            messages = build_translation_quality_messages(original_text, translation)
            response = await self.model.achat(messages=messages)
            content = await extract_response_content(response)
            parsed = parse_translation_quality_response(content)

            if normalize:
                parsed["score"] = parsed["score"] / 2.0

            return GraderScore(
                name=self.name,
                score=parsed["score"],
                reason=parsed["reason"],
                metadata={"key_issues": parsed["key_issues"]},
            )
        except Exception as e:
            return GraderError(name=self.name, error=str(e))


async def extract_response_content(response) -> str:
    if hasattr(response, 'content'):
        return response.content
    elif isinstance(response, dict) and 'content' in response:
        return response['content']
    elif isinstance(response, str):
        return response
    else:
        raise ValueError(f"Unable to extract content from response: {type(response)}")
