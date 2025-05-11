from fastapi import FastAPI
from .inference import router as inference_router
from .loadModel import model_loader
import torch

app = FastAPI(
    title="PikNLP Server",
    description="PikNLP Server",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

@app.get("/")
def top():
    return "It's Pik!"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loader.model is not None,
        "device": str(model_loader.device),
        "cuda_available": torch.cuda.is_available()
        }

app.include_router(inference_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18000, reload=True)