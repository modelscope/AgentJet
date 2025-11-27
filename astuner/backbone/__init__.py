from loguru import logger

try:
    from astuner.backbone.trinity_compat_workflow import *
except ImportError:
    logger.warning("trinity is not available.")
