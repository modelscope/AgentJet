import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, PROJECT_ROOT)
print(f"PROJECT_ROOT: {PROJECT_ROOT}")

from openjudge.models import OpenAIChatModel
from tutorial.example_deep_finance.judge import PresentationQualityGrader


async def main():
    # 你也可以只写：model = OpenAIChatModel(model="qwen3-32b")
    # 并用环境变量 OPENAI_API_KEY / OPENAI_BASE_URL（QuickStart里推荐这种方式）
    model = OpenAIChatModel(
        model="qwen-flash",
        extra_body={"enable_thinking": False, "temperature": 0, "top_p": 1, "seed": 0},
    )

    grader = PresentationQualityGrader(model=model)

    report = """
        # 藏格矿业分析报告

        ## 执行摘要
        - 核心结论：...

        ## 财务对比
        | 公司 | 营收 | 净利 |
        |---|---:|---:|
        | A | 20 | 5 |

        ## 风险与下一步
        - 风险：...
        - 下一步：...
        """
    res = await grader.aevaluate(report_content=report, user_query="分析藏格矿业的财务状况")
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
