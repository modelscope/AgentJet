"""
CGCV Grader - Citation-Grounded Claim Verification
引用锚定的断言验证评分器
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderScore

# import path 兼容两种写法
try:
    from openjudge.models import OpenAIChatModel
except Exception:  # pragma: no cover
    from openjudge.models.openai_chat_model import OpenAIChatModel

from .prompt import (
    CGCV_SYSTEM_PROMPT_ZH, 
    CGCV_SYSTEM_PROMPT_EN,
    CGCV_USER_PROMPT_TEMPLATE_ZH,
    CGCV_USER_PROMPT_TEMPLATE_EN,
    get_cgcv_prompts
)
from .json_utils import (
    strict_load_json, 
    validate_cgcv_schema, 
    parse_cgcv_result,
    construct_cgcv_prompt,
    compute_cgcv_score,
    CGCVResult,
    ClaimStatus
)


class CGCVGrader(BaseGrader):
    """
    Citation-Grounded Claim Verification (CGCV) Grader
    引用锚定的断言验证评分器
    
    核心理念：引用是断言与证据之间的"锚点"
    
    验证流程：
    1. 断言提取 (Claim Extraction)
    2. 引用检查 (Citation Checking)
    3. 来源追溯 (Source Tracing)
    4. 内容对齐验证 (Content Alignment)
    
    验证状态：
    - verified: 验证通过
    - citation_missing: 引用缺失
    - citation_broken: 引用断裂
    - subject_misalign: 对象错位
    - predicate_misalign: 属性错位
    - object_misalign: 值错位
    - qualifier_misalign: 限定错位
    
    评分机制：
    - score = verified_claims / total_claims
    - 范围: [0, 1]
    
    输入：traj（完整对话轨迹）
    输出：GraderScore(name, score, reason)
    """

    def __init__(
        self,
        model: OpenAIChatModel,
        name: str = "cgcv",
        language: str = "zh",
        **kwargs: Any,
    ):
        """
        初始化 CGCV Grader
        
        Args:
            model: OpenAI 兼容的聊天模型
            name: Grader 名称
            language: 语言选择，"zh" 或 "en"
            **kwargs: 其他参数传递给 BaseGrader
        """
        super().__init__(name=name, **kwargs)
        self.model = model
        self.language = language.lower()
        
        # 根据语言选择 prompt
        self.system_prompt, self.user_prompt_template = get_cgcv_prompts(self.language)

    @staticmethod
    def create_default_model(
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        deterministic: bool = True,
        enable_thinking: bool = False,
        seed: int = 0,
    ) -> OpenAIChatModel:
        """
        创建默认模型
        
        Args:
            model_name: 模型名称
            api_key: API Key，默认从环境变量读取
            base_url: API Base URL，默认从环境变量读取
            deterministic: 是否使用确定性配置
            enable_thinking: 是否启用思考模式
            seed: 随机种子
            
        Returns:
            OpenAIChatModel 实例
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")

        extra_body: Dict[str, Any] = {}
        if deterministic:
            extra_body.update({
                "temperature": 0,
                "top_p": 1,
                "seed": seed,
                "presence_penalty": 0,
                "frequency_penalty": 0,
            })
        if enable_thinking is False:
            extra_body["enable_thinking"] = False

        kwargs: Dict[str, Any] = {"model": model_name}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if extra_body:
            kwargs["extra_body"] = extra_body

        return OpenAIChatModel(**kwargs)

    async def aevaluate(
        self,
        traj: Any,
        **_: Any,
    ) -> GraderScore:
        """
        异步评估入口
        
        Args:
            traj: 对话轨迹，格式为 [{"role": ..., "content": ...}, ...] 
                  或者 {"messages": [...]} 格式
        
        Returns:
            GraderScore(name, score, reason)
        """
        # 1. 提取 messages（兼容两种格式）
        if isinstance(traj, dict):
            messages_list = traj.get("messages", [])
        elif isinstance(traj, list):
            messages_list = traj
        else:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="BadInput: traj must be list or dict with 'messages'",
            )
        
        if not messages_list:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="BadInput: empty trajectory",
            )

        # 2. 构建 prompt
        user_prompt = construct_cgcv_prompt(messages_list, self.user_prompt_template)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 3. 调用模型
        try:
            resp = await self.model.achat(messages)
            raw_text = getattr(resp, "content", None)
            if raw_text is None:
                raw_text = str(resp)
        except Exception as e:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=f"ModelCallError: {type(e).__name__}: {e}",
            )

        # 4. 解析 JSON
        obj, jerr = strict_load_json(str(raw_text))
        if obj is None:
            snippet = str(raw_text)[:200].replace("\n", " ")
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=f"ParseError: {jerr}; raw[:200]={snippet}",
            )

        # 5. 验证 schema
        obj, serr = validate_cgcv_schema(obj)
        if obj is None:
            snippet = str(raw_text)[:200].replace("\n", " ")
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=f"SchemaError: {serr}; raw[:200]={snippet}",
            )

        # 6. 解析结果并计算分数
        result = parse_cgcv_result(obj)
        score, reason = compute_cgcv_score(result)
        
        return GraderScore(name=self.name, score=score, reason=reason)

    def evaluate(
        self,
        traj: Any,
        **kwargs: Any,
    ) -> GraderScore:
        """
        同步评估入口（通过 asyncio 包装异步方法）
        
        Args:
            traj: 对话轨迹
            **kwargs: 其他参数
            
        Returns:
            GraderScore
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.aevaluate(traj, **kwargs))

    def get_detailed_result(
        self,
        traj: Any,
    ) -> Tuple[GraderScore, Optional[CGCVResult]]:
        """
        获取详细评估结果（包含每个断言的验证详情）
        
        Args:
            traj: 对话轨迹
            
        Returns:
            (GraderScore, CGCVResult) 元组
        """
        import asyncio
        
        async def _detailed_evaluate():
            # 复用主流程逻辑
            if isinstance(traj, dict):
                messages_list = traj.get("messages", [])
            elif isinstance(traj, list):
                messages_list = traj
            else:
                return GraderScore(
                    name=self.name,
                    score=0.0,
                    reason="BadInput: traj must be list or dict with 'messages'",
                ), None
            
            if not messages_list:
                return GraderScore(
                    name=self.name,
                    score=0.0,
                    reason="BadInput: empty trajectory",
                ), None

            user_prompt = construct_cgcv_prompt(messages_list, self.user_prompt_template)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            try:
                resp = await self.model.achat(messages)
                raw_text = getattr(resp, "content", None)
                if raw_text is None:
                    raw_text = str(resp)
            except Exception as e:
                return GraderScore(
                    name=self.name,
                    score=0.0,
                    reason=f"ModelCallError: {type(e).__name__}: {e}",
                ), None

            obj, jerr = strict_load_json(str(raw_text))
            if obj is None:
                return GraderScore(
                    name=self.name,
                    score=0.0,
                    reason=f"ParseError: {jerr}",
                ), None

            obj, serr = validate_cgcv_schema(obj)
            if obj is None:
                return GraderScore(
                    name=self.name,
                    score=0.0,
                    reason=f"SchemaError: {serr}",
                ), None

            result = parse_cgcv_result(obj)
            score, reason = compute_cgcv_score(result)
            
            return GraderScore(name=self.name, score=score, reason=reason), result
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(_detailed_evaluate())


# =============================================================================
# Convenience Functions
# =============================================================================

def create_cgcv_grader(
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    language: str = "zh",
    **kwargs
) -> CGCVGrader:
    """
    便捷函数：创建 CGCV Grader
    
    Args:
        model_name: 模型名称
        api_key: API Key
        base_url: API Base URL
        language: 语言 ("zh" 或 "en")
        **kwargs: 其他模型参数
        
    Returns:
        CGCVGrader 实例
        
    Example:
        >>> grader = create_cgcv_grader("gpt-4o", language="zh")
        >>> result = await grader.aevaluate(trajectory)
        >>> print(f"Score: {result.score}, Reason: {result.reason}")
    """
    model = CGCVGrader.create_default_model(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )
    return CGCVGrader(model=model, language=language)
