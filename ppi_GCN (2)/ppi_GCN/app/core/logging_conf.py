from loguru import logger
import sys
from app.core.config import settings

def setup_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
        serialize=True
    )
    logger.add(
        "logs/app.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        serialize=True
    )