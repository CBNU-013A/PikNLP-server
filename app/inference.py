from fastapi import APIRouter, HTTPException
from .schema import PredictRequest, PredictResponse
from .loadModel import model_loader
import logging


logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    logger.info("POST /predict - text received: %s", request)
    try:
        result = await model_loader.predict(request.text)
        return PredictResponse(sentiments=result)
    except Exception as e:
        logger.exception("POST /predict - error during prediction: %s", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")