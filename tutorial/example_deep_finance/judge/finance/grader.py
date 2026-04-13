"""
FinanceCompositionEvaluator - 基于 OpenJudge 的 Finance 组合评估器

功能：
- 根据 domain 路由到对应的 grader 集合
- 执行 pairwise 评估（比较 training answer 和 reference answer）
- 返回 0-1 范围的分数

支持的 domain:
- stock_analysis: 股票分析
- industry_research: 行业研究  
- macro_analysis: 宏观分析
- event_interpretation: 事件解读
- stock_search: 股票搜索
"""

from typing import Dict, Any, List, Type

from openjudge.graders.base_grader import BaseGrader

# import path 兼容两种写法
try:
    from openjudge.models import OpenAIChatModel
except Exception:  # pragma: no cover
    from openjudge.models.openai_chat_model import OpenAIChatModel

# Finance Graders from OpenJudge cookbooks
from cookbooks.finance_grader.stock_analysis.valuation_analysis import ValuationAnalysisGrader
from cookbooks.finance_grader.stock_analysis.fundamental_analysis import FundamentalAnalysisGrader
from cookbooks.finance_grader.stock_analysis.overall_logic import OverallLogicGrader
from cookbooks.finance_grader.stock_analysis.stock_risk_analysis import StockRiskAnalysisGrader
from cookbooks.finance_grader.macro_analysis.macro_analysis import MacroAnalysisGrader
from cookbooks.finance_grader.macro_analysis.concept_explanation import ConceptExplanationGrader
from cookbooks.finance_grader.industry_research.characteristics_analysis import CharacteristicsAnalysisGrader
from cookbooks.finance_grader.industry_research.risk_analysis import RiskAnalysisGrader
from cookbooks.finance_grader.industry_research.underlying_comparison import UnderlyingComparisonGrader
from cookbooks.finance_grader.event_interpretation.event_analysis import EventAnalysisGrader
from cookbooks.finance_grader.event_interpretation.event_identification import EventIdentificationGrader
from cookbooks.finance_grader.stock_search.search_relevance import SearchRelevanceGrader
from cookbooks.finance_grader.stock_search.search_integrity import SearchIntegrityGrader
from cookbooks.finance_grader.stock_search.search_timeliness import SearchTimelinessGrader


class FinanceCompositionEvaluator:
    """
    基于 OpenJudge 的 Finance 组合评估器（替代 rm_gallery.FinanceComposition）
    
    功能：
    - 根据 domain 路由到对应的 grader 集合
    - 执行 pairwise 评估（比较 training answer 和 reference answer）
    - 返回 0-1 范围的分数
    
    支持的 domain:
    - stock_analysis: 股票分析
    - industry_research: 行业研究  
    - macro_analysis: 宏观分析
    - event_interpretation: 事件解读
    - stock_search: 股票搜索
    """
    
    # Domain 到 Grader 类的映射（与 RM-Gallery 保持一致）
    DOMAIN_GRADERS: Dict[str, List[Type[BaseGrader]]] = {
        "stock_analysis": [
            ValuationAnalysisGrader,
            # FundamentalAnalysisGrader,
            # OverallLogicGrader,
            # StockRiskAnalysisGrader,
        ],
        "industry_research": [
            CharacteristicsAnalysisGrader,
            # RiskAnalysisGrader,
            # UnderlyingComparisonGrader,
        ],
        "macro_analysis": [
            MacroAnalysisGrader,
            # ConceptExplanationGrader,
        ],
        "event_interpretation": [
            EventAnalysisGrader,
            # EventIdentificationGrader,
        ],
        "stock_search": [
            SearchRelevanceGrader,
            # SearchIntegrityGrader,
            # SearchTimelinessGrader,
        ],
    }

    # 暴露所有可用的 grader 类（供外部参考或扩展）
    AVAILABLE_GRADERS = {
        "stock_analysis": {
            "ValuationAnalysisGrader": ValuationAnalysisGrader,
            "FundamentalAnalysisGrader": FundamentalAnalysisGrader,
            "OverallLogicGrader": OverallLogicGrader,
            "StockRiskAnalysisGrader": StockRiskAnalysisGrader,
        },
        "industry_research": {
            "CharacteristicsAnalysisGrader": CharacteristicsAnalysisGrader,
            "RiskAnalysisGrader": RiskAnalysisGrader,
            "UnderlyingComparisonGrader": UnderlyingComparisonGrader,
        },
        "macro_analysis": {
            "MacroAnalysisGrader": MacroAnalysisGrader,
            "ConceptExplanationGrader": ConceptExplanationGrader,
        },
        "event_interpretation": {
            "EventAnalysisGrader": EventAnalysisGrader,
            "EventIdentificationGrader": EventIdentificationGrader,
        },
        "stock_search": {
            "SearchRelevanceGrader": SearchRelevanceGrader,
            "SearchIntegrityGrader": SearchIntegrityGrader,
            "SearchTimelinessGrader": SearchTimelinessGrader,
        },
    }
    
    def __init__(self, model: OpenAIChatModel, params: Dict[str, Any] = None):
        """
        初始化 FinanceCompositionEvaluator
        
        Args:
            model: OpenAIChatModel 实例
            params: 额外参数（保留兼容性）
        """
        self.model = model
        self.params = params or {}
        self._grader_cache: Dict[str, List[BaseGrader]] = {}
        
    def _get_graders_for_domain(self, domain: str) -> List[BaseGrader]:
        """
        获取指定 domain 的 grader 实例列表（带缓存）
        """
        if domain not in self._grader_cache:
            grader_classes = self.DOMAIN_GRADERS.get(domain, [])
            self._grader_cache[domain] = [
                grader_cls(model=self.model) for grader_cls in grader_classes
            ]
        return self._grader_cache[domain]

    @classmethod
    def get_supported_domains(cls) -> List[str]:
        """获取所有支持的 domain 列表"""
        return list(cls.DOMAIN_GRADERS.keys())

    @classmethod
    def get_graders_for_domain_class(cls, domain: str) -> List[Type[BaseGrader]]:
        """获取指定 domain 的 grader 类列表（静态方法）"""
        return cls.DOMAIN_GRADERS.get(domain, [])
    
    async def aevaluate(self, query: str, current: str, reference: str, domain: str) -> float:
        """
        执行 pairwise 评估（异步版本，避免重复创建 event loop）
        
        Args:
            query: 用户查询
            current: 当前模型生成的回答 (training)
            reference: 参考答案
            domain: 任务领域（用于路由到对应 graders）
            
        Returns:
            float: 0-1 范围的分数
                - 1.0: current 优于 reference
                - 0.0: reference 优于 current
                - 0.5: 无法评估或出错
        """
        if not domain or domain not in self.DOMAIN_GRADERS:
            print(f"⚠️ FinanceCompositionEvaluator: Unknown domain '{domain}', returning 0.5")
            return 0.5
            
        graders = self._get_graders_for_domain(domain)
        if not graders:
            print(f"⚠️ FinanceCompositionEvaluator: No graders for domain '{domain}', returning 0.5")
            return 0.5
        
        # 运行所有 graders
        scores = []
        for grader in graders:
            try:
                result = await grader.aevaluate(
                    query=query,
                    answer_1=current,    # training model output
                    answer_2=reference,  # reference answer
                )
                
                # 解析 GraderRank 结果
                if hasattr(result, 'rank') and isinstance(result.rank, list):
                    # rank = [1, 2] 表示 answer_1 (current) 更好 -> score = 1.0
                    # rank = [2, 1] 表示 answer_2 (reference) 更好 -> score = 0.0
                    if result.rank[0] == 1:
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
                else:
                    scores.append(0.5)  # 无法解析，返回中间值
                    
            except Exception as e:
                grader_name = getattr(grader, 'name', grader.__class__.__name__)
                print(f"⚠️ FinanceCompositionEvaluator: Grader {grader_name} failed: {e}")
                scores.append(0.5)
        
        # 计算平均分数
        if scores:
            return sum(scores) / len(scores)
        return 0.5
