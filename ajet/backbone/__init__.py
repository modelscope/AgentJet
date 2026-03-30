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
    pass
    # logger.info("trinity is not available.")
