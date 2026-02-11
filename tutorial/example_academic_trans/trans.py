
import re
import os
import asyncio
from openai import OpenAI

from ajet import WorkflowOutput
from ajet.schema.task import Task
from ajet.utils.retry import retry_with_backoff
from ajet.tuner_lib.as_oai_baseurl_apikey import OpenaiBaseUrlAndApiKey
from beast_logger import print_listofdict

# Import reward computation from trans_reward.py
from openjudge.models import OpenAIChatModel
from .trans_reward import TranslationQualityGrader, examples



@retry_with_backoff(max_retry=3)
def execute_agent(task: Task, api_baseurl_key: OpenaiBaseUrlAndApiKey):
    # Prepare base_url, api_key
    base_url, api_key = (api_baseurl_key.base_url, api_baseurl_key.api_key)
    grader_base_url, grader_api_key = ("https://dashscope.aliyuncs.com/compatible-mode/v1", os.environ.get("DASHSCOPE_API_KEY", ""))
    # Read dataset item
    title = task.metadata['title']
    authors = task.metadata['authors']
    abstract = task.metadata['abstract']

    messages, rough_translate = rough_translate_agent(base_url, api_key, abstract)
    # print_listofdict(messages, header="rough_translate_agent", mod="c")

    # messages, fix_nouns = detect_hard_proper_nouns(messages, base_url, api_key, abstract, rough_translate)
    messages, fix_nouns = detect_hard_proper_nouns(messages, grader_base_url, grader_api_key, abstract, rough_translate)
    # print_listofdict(messages, header="detect_hard_proper_nouns", mod="c")

    messages, final_translation = produce_final_translation(messages, base_url, api_key, abstract, rough_translate, fix_nouns)
    print_listofdict(messages, header="final_translation", mod="c")

    if final_translation is None:
        raw_reward = 0.0
    else:
        grader = TranslationQualityGrader(
            model=OpenAIChatModel(base_url=grader_base_url, api_key=grader_api_key, model="qwen3-max-2026-01-23")
        )
        grader_score = asyncio.run(asyncio.wait_for(grader.aevaluate(original_text=abstract, translation=final_translation), timeout=120))
        raw_reward = grader_score.score
        # print(f"Grader Score: {grader_score.score}, Reason: {grader_score.reason}, Metadata: {grader_score.metadata}")
    return WorkflowOutput(reward=raw_reward, metadata={
        "rough_translate": rough_translate,
        "fix_nouns": fix_nouns,
        "final_translation": final_translation
    })


def produce_final_translation(messages, base_url, api_key, abstract, rough_translate, fix_nouns):
    messages = messages + [
        {
            "role": "user",
            "content": "Please produce the final, corrected Chinese translation by applying all the corrections listed above. "
                       "Output only the final translation between <final_result> ... </final_result>, so I will extract result with regex."
        },
    ]

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model="agentjet-model",
        messages=messages
    )
    final_translation = response.choices[0].message.content

    messages += [
        {
            "role": "assistant",
            "content": final_translation
        }
    ]

    # Extract final translation
    match = re.search(r"<final_result>(.*?)</final_result>", final_translation, re.DOTALL)
    if match:
        final_translation = match.group(1).strip()
    else:
        final_translation = None

    return messages, final_translation



def detect_hard_proper_nouns(messages, base_url, api_key, abstract, rough_translate):
    messages = messages + [

        {
            "role": "user",
            "content":  "You new job is to detect translation errors of discipline-specific proper nouns. "
                        "Use json to list all errors found in the translation result and provide correction. "
                        "Json format: [{\"original_word\": \"xxx\", \"wrong_translation\": \"xxx\", \"wrong_reason\": \"xxx\", \"correct_translation\": \"xxx\"}, ...]. "
                        "If no errors are found, return an empty list []."
                        "Please list all translation errors of discipline-specific proper nouns found in the translation result according to the requirements."
        },

    ]

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model="qwen3-max-2026-01-23",
        messages=messages,
        timeout=60,
        # extra_body={"enable_thinking":True}
    )
    fix_nouns = response.choices[0].message.content
    messages += [
        {
            "role": "assistant",
            "content": fix_nouns
        }
    ]
    return messages, fix_nouns


def rough_translate_agent(base_url, api_key, abstract):
    messages = [
        {
            "role": "system",
            "content":
                "You are a professional language translator. "
                "Translate the given Academic English text into Chinese accurately. "
                "During the translation process, it is necessary to meet the linguistic norms of Chinese academic papers "
                "such as conforming to the logic of the Chinese language, being simple, rigorous, and concise, "
                "and avoiding the use of first-person pronouns when passive voice is appropriate. "
                "Ensure that specialized terms are translated correctly according to academic standards. "
                "Replace 我/我们 with 本研究 or 本文 or 研究者 or simply remove it and rephrase the sentence. "
                "If an English abbreviation is short in Chinese, use Chinese. "
                "If an English abbreviation is long in Chinese, use English abbreviation. "
                "To use an English abbreviation, if the author has mentioned the full form first, mention the full form at its first appearance. "
                "e.g. `We have used the LAsMA heterodyne array installed on the Atacama Pathfinder EXperiment (APEX)` should be translated as "
                "`本研究使用了安装在阿塔卡马探路者实验望远镜（APEX, Atacama Pathfinder EXperiment）上的LAsMA外差阵列`. "
        },
        {
            "role": "user",
            "content": abstract
        }
    ]

    for ex in examples:
        messages[0]['content'] += f"\n\nExample:\n\tOriginal: {ex['original']}\n\tBad Translation: {ex['bad']}\n\tHint: {ex['hint']}\n\tGood Translation: {ex['good']}"

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model="agentjet-model",
        messages=messages
    )
    rough_translate = response.choices[0].message.content
    messages += [
        {
            "role": "assistant",
            "content": rough_translate
        }
    ]

    return messages, rough_translate
