# /app/core/logger.py

import logging

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler("logs/server.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)