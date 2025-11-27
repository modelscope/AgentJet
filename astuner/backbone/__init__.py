from loguru import logger

try:
    from astune.backbone.trinity_compat_workflow import (
        ASTunerTaskReader,
        ASTunetWorkflowWrap,
        TrinityCompatWorkflow,
    )

    __all__ = [
        "TrinityCompatWorkflow",
        "ASTunetWorkflowWrap",
        "ASTunerTaskReader",
    ]
except ImportError:
    logger.warning("trinity is not available.")
