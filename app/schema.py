from pydantic import BaseModel
from typing import Dict

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiments: Dict[str, str]  # 카테고리별 감성 분석 결과 (pos, neg, none)

class CategoriesResponse(BaseModel):
    categories: list[str]  # 카테고리 목록 
    