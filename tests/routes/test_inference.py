# /tests/routes/test_inference.py

import os
import pytest
from unittest.mock import patch, AsyncMock

API_KEY = os.getenv("API_KEY", "test")

# TC01: POST /predict

# TC01-1: normal case - 200
@pytest.mark.parametrize("text", [
    "뷰도 좋고 직원들도 친절했어요!",     # normal
    "너무 시끄럽고 지저분했어요",         # negative
    "좋아요 " * 1000,            # long text
    "😡 서비스가 최악이었어요",          # emoji included
])
@pytest.mark.asyncio
async def test_predict_success_cases(async_client, text):
    res = await async_client.post(
        "/api/v1/predict",
        json={"text": text},
        headers={"NLP-API-KEY": API_KEY},
    )
    assert res.status_code == 200
    data = res.json()
    assert "sentiments" in data
    assert isinstance(data["sentiments"], dict)

# TC01-2: 빈 text → 422
@pytest.mark.asyncio
async def test_predict_empty_text(async_client):
    res = await async_client.post(
        "/api/v1/predict",
        json={"text": ""},
        headers={"NLP-API-KEY": API_KEY},
    )
    assert res.status_code == 422

# TC01-3: internal error → 500
@pytest.mark.asyncio
async def test_predict_internal_error(async_client):
    with patch("app.services.inference_runner.model_loader.predict", new_callable=AsyncMock) as mock_predict:
        mock_predict.side_effect = RuntimeError("mocked failure")
        res = await async_client.post(
            "/api/v1/predict",
            json={"text": "서비스 최악"},
            headers={"NLP-API-KEY": API_KEY},
        )
    assert res.status_code == 500
    assert res.json()["detail"] == "Internal Server Error"

# TC01-4: 잘못된 API 키 → 401
@pytest.mark.asyncio
async def test_predict_invalid_api_key(async_client):
    res = await async_client.post(
        "/api/v1/predict",
        json={"text": "서비스가 별로였어요"},
        headers={"NLP-API-KEY": "wrong-key"},
    )
    assert res.status_code == 401

# TC02: GET /categories

# TC02-1: 정상 요청 → 200
@pytest.mark.asyncio
async def test_get_categories_success(async_client):
    res = await async_client.get("/api/v1/categories", headers={"NLP-API-KEY": API_KEY})
    assert res.status_code == 200
    data = res.json()
    assert "categories" in data
    assert isinstance(data["categories"], list)
    assert all(isinstance(cat, str) for cat in data["categories"])

# TC02-2: internal error → 500
@pytest.mark.asyncio
async def test_get_categories_internal_error(async_client):
    with patch("app.services.inference_runner.model_loader.get_categories", new_callable=AsyncMock) as mock_get_categories:
        mock_get_categories.side_effect = RuntimeError("mocked failure")
        res = await async_client.get("/api/v1/categories", headers={"NLP-API-KEY": API_KEY})
    assert res.status_code == 500
    assert res.json()["detail"] == "Internal Server Error"

# TC02-3: 잘못된 API 키 → 401
@pytest.mark.asyncio
async def test_get_categories_invalid_api_key(async_client):
    res = await async_client.get("/api/v1/categories", headers={"NLP-API-KEY": "invalid"})
    assert res.status_code == 401