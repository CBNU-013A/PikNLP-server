from fastapi import APIRouter
from .schema import PredictRequest, PredictResponse
from .loadModel import model_loader

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    result = await model_loader.predict(request.text)
    return PredictResponse(sentiments=result)
