# 使得可以通过 from judge import PresentationQualityGrader 直接引用
from .grounding.grader import GroundingGrader
from .presentation_quality.grader import PresentationQualityGrader
<<<<<<< HEAD
from .cgcv.grader import CGCVGrader
from .audit.grader import AuditGrader
from .traceability.grader import TraceabilityRewardGrader
from .ebtu.grader import EBTUTraceabilityGrader
=======
>>>>>>> origin/main
# from .research_depth.grader import ResearchDepthGrader
# from .research_breadth.grader import ResearchBreadthGrader

# 以后添加了其他 grader 也可以加在这里
# from .grounding.grader import GroundingGrader
# from .research_breadth.grader import ResearchBreadthGrader
# __all__ = ["PresentationQualityGrader", "GroundingGrader", "ResearchDepthGrader", "ResearchBreadthGrader"]
<<<<<<< HEAD
__all__ = ["PresentationQualityGrader", "GroundingGrader", "CGCVGrader", "AuditGrader", "TraceabilityRewardGrader", "EBTUTraceabilityGrader"]
=======
__all__ = ["PresentationQualityGrader", "GroundingGrader"]
>>>>>>> origin/main
