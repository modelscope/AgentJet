from loguru import logger

try:
    from ajet.backbone.trainer_trinity import (
        AjetTaskReader,
        AjetWorkflowWrap,
        TrinityRolloutManager,
    )

    __all__ = [
        "TrinityRolloutManager",
        "AjetWorkflowWrap",
        "AjetTaskReader",
    ]
except ImportError:
    logger.warning("trinity is not available.")
