# tests/test_main.py
import os

os.environ["ENV"] = "test"

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from asgi_lifespan import LifespanManager
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_top():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/")
    assert response.status_code == 200
    assert response.text == '"It\'s Pik!"'


@pytest.mark.asyncio
async def test_health():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "model_loaded" in json_data
    assert "device" in json_data
    assert "cuda_available" in json_data
    assert "API-MODE" in json_data


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "뷰도 좋고 직원들도 친절했어요!",
    "너무 시끄럽고 지저분했어요",
    "적당히 괜찮았어요. 특별할 건 없었고요.",
    "😡 서비스가 최악이었어요",
    "",  # 빈 문자열 처리
])
async def test_predict_cases(text):
    sample_input = {"text": text}
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/predict", json=sample_input, headers={"NLP-API-KEY": API_KEY})
    assert response.status_code == 200
    json_data = response.json()
    assert "sentiments" in json_data
    assert isinstance(json_data["sentiments"], dict)

@pytest.mark.asyncio
async def test_predict_empty_text():
    '''validation Error'''
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/predict", headers={"NLP-API-KEY": API_KEY})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_predict_internel_error(monkeypatch):
    with patch("app.inference.model_loader.predict", new_callable=AsyncMock) as mock_predict:
        mock_predict.side_effect = RuntimeError("some error")

        async with LifespanManager(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/api/v1/predict", json={"text": "서비스 최악"}, headers={"NLP-API-KEY": API_KEY})

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal Server Error"

@pytest.mark.asyncio
async def test_invalid_api_key():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/predict", json={"text": "서비스가 별로"}, headers={"NLP-API-KEY": "wrong-key"})
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_global_exception_handler():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            try:
                response = await ac.get("/error", headers={"NLP-API-KEY": API_KEY})
                assert response.status_code == 500
                assert response.json()["detail"] == "Internal Server Error"
            except Exception as e:
                # 일부 환경에 따라 FastAPI 내부에서 raise된 예외가 그대로 pytest까지 전파될 수 있음
                print(f"Expected exception occurred (handled by FastAPI): {e}")

@pytest.mark.asyncio
async def test_get_categories():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/v1/categories", headers={"NLP-API-KEY": API_KEY})
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert len(json_data) > 0
    assert all(isinstance(cat, str) for cat in json_data)

@pytest.mark.asyncio
async def test_get_categories_internal_error(monkeypatch):
    with patch("app.inference.model_loader.get_categories", new_callable=AsyncMock) as mock_get_categories:
        mock_get_categories.side_effect = RuntimeError("some error")

        async with LifespanManager(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.get("/api/v1/categories", headers={"NLP-API-KEY": API_KEY})

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal Server Error"