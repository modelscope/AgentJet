from loguru import logger
try:
    from astune.backbone.trinity_compat_workflow import *
except ImportError:
    logger.warning("trinity is not available.")
