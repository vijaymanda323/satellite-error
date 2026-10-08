# pyrefly: ignore [missing-import]
from loguru import logger   
logger.add(
    "logs/download.log",
    rotation="10 MB",
    retention=10
)