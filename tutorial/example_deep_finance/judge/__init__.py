# 使得可以通过 from judge import PresentationQualityGrader 直接引用
# from tutorial.example_deep_finance.judge.grounding.grader import GroundingGrader
from tutorial.example_deep_finance.judge.presentation_quality.grader import PresentationQualityGrader
# from tutorial.example_deep_finance.judge.research_depth.grader import ResearchDepthGrader
# from tutorial.example_deep_finance.judge.research_breadth.grader import ResearchBreadthGrader

# 以后添加了其他 grader 也可以加在这里
# from .grounding.grader import GroundingGrader
# from .research_breadth.grader import ResearchBreadthGrader
# __all__ = ["PresentationQualityGrader", "GroundingGrader", "ResearchDepthGrader", "ResearchBreadthGrader"]
__all__ = ["PresentationQualityGrader"]
