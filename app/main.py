from fastapi import FastAPI, Depends, Header, HTTPException, status
from .inference import router as inference_router
from .loadModel import model_loader
import torch
import logging
import os

if "ENV" not in os.environ:
    from dotenv import load_dotenv
    load_dotenv()

from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler("logs/server.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PikNLP Server",
    description="PikNLP Server",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

def verify_api_key(nlp_api_key: str = Header(...)):
    if nlp_api_key != os.getenv("API_KEY"):
        logger.warning("⚠️ Invalid API key provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error: %s", exc.errors())
    return await request_validation_exception_handler(request, exc)

@app.get("/")
def top():
    return "It's Pik!"

@app.get("/health")
def health():
    logger.info("GET /health - Health check requested.")
    return {
        "status": "ok",
        "model_loaded": model_loader.model is not None,
        "device": str(model_loader.device),
        "cuda_available": torch.cuda.is_available(),
        "API-MODE": os.getenv("ENV")
        }

if os.getenv("ENV") == "test":
    logger.info("Test mode enabled. add /error endpoint.")
    @app.get("/error")
    def raise_error():
        logger.info("GET /error - Raising an error for testing.")
        raise Exception("intentional error for testing")

app.include_router(inference_router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18000, reload=True)