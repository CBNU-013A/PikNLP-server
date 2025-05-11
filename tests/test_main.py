import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from asgi_lifespan import LifespanManager


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
            response = await ac.post("/api/v1/predict", json=sample_input)
    assert response.status_code == 200
    json_data = response.json()
    assert "sentiments" in json_data
    assert isinstance(json_data["sentiments"], dict)