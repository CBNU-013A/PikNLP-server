# /app/core/server.py

from fastapi import FastAPI, Depends
from fastapi.exceptions import RequestValidationError

from .core.config import load_env
from .core.logger import logger
from .core.exceptions import global_exception_handler, validation_exception_handler
from .core.dependencies import verify_api_key

from .routes.inference import router as inference_router
from .routes.common import router as common_router
from .routes.generate import router as generate_router
from .routes.train import router as train_router

logger.info("Starting PikNLP Server...")

load_env()

app = FastAPI(
    title="PikNLP Server",
    description="PikNLP Server",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Router registration
app.include_router(common_router, tags=["common"])
app.include_router(inference_router, prefix="/api/v1", tags=["inference"], dependencies=[Depends(verify_api_key)])
app.include_router(generate_router, prefix="/api/v1", tags=["generate"], dependencies=[Depends(verify_api_key)])
app.include_router(train_router, prefix="/api/v1", tags=["train"], dependencies=[Depends(verify_api_key)])

# Middleware and exception handlers
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)