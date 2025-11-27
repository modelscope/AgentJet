from loguru import logger

try:
    from astuner.backbone.trinity_compat_workflow import (
        ASTunerTaskReader,
        ASTunerWorkflowWrap,
        TrinityCompatWorkflow,
    )

    __all__ = [
        "TrinityCompatWorkflow",
        "ASTunerWorkflowWrap",
        "ASTunerTaskReader",
    ]
except ImportError:
    logger.warning("trinity is not available.")
