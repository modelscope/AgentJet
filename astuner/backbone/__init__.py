from loguru import logger

try:
    pass
except ImportError:
    logger.warning("trinity is not available.")
