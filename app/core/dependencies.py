from fastapi import Header, HTTPException, status
import os
from app.core.logger import logger

def verify_api_key(nlp_api_key: str = Header(...)):
    if nlp_api_key != os.getenv("API_KEY"):
        logger.warning("⚠️ Invalid API key provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )