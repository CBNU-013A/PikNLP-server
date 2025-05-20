# tests/services/test_inference_runner.py

import pytest
import torch

from app.services.inference_runner import ModelLoader

@pytest.fixture(scope="module")
def model_loader():
    return ModelLoader(config_path="app/services/config.yaml")

# TC1: convert_to_feature Method Tests

# TC1-1: Normal Case
def test_convert_to_feature_normal(model_loader):
    text = "이 장소 정말 좋았어요"
    category = "분위기"
    features = model_loader.convert_to_feature(text, category)

    assert set(features.keys()) >= {"input_ids", "attention_mask", "token_type_ids"}
    for v in features.values():
        assert isinstance(v, torch.Tensor)
        assert v.shape == (1, model_loader.config["model"]["max_length"])

# TC1-2: long text Case
def test_convert_to_feature_long_text(model_loader):
    long_test = "정말정말 " * 1000
    features = model_loader.convert_to_feature(long_test, "서비스")
    assert all(t.shape[1] == model_loader.config["model"]["max_length"] for t in features.values())

# TC1-3: empty text Case
def test_convert_to_feature_empty_text(model_loader):
    features = model_loader.convert_to_feature("", "분위기")
    assert features["input_ids"].shape[1] == model_loader.config["model"]["max_length"]

# TC1-4: special characters Case
def test_convert_to_feature_special_characters(model_loader):
    text = "안녕하세요! 😊"
    category = "서비스"
    features = model_loader.convert_to_feature(text, category)

    assert set(features.keys()) >= {"input_ids", "attention_mask", "token_type_ids"}
    for v in features.values():
        assert isinstance(v, torch.Tensor)
        assert v.shape == (1, model_loader.config["model"]["max_length"])

# TC1-5: None Case, Invalid Type (expecting ValueError)
@pytest.mark.parametrize("text, category", [
    (None, "서비스"),
    ("안녕하세요", None),
    (1234, "가격"),
    ("안녕하세요", ["not", "a", "string"]),
    (None, None)
])
def test_convert_to_feature_type_errors(model_loader, text, category):
    with pytest.raises(ValueError):
        model_loader.convert_to_feature(text, category)

# TC1-6: Combination Case
@pytest.mark.parametrize("text, category", [
    ("", ""), 
    ("좋아요", ""),
    ("", "서비스"),
])
def test_convert_to_feature_combination_cases(model_loader, text, category):
    result = model_loader.convert_to_feature(text, category)
    assert isinstance(result, dict)

# TC2: predict Method Tests

# TC2-(1-4): Valid cases
@pytest.mark.parametrize("text", [
    "뷰도 좋고 직원들도 친절했어요!",   # TC2-1: Normal Case
    "",   # TC2-2: Empty Test Case
    "안녕하세요! 😊",  # TC2-3: Special Characters Case
    "좋아요 " * 1000, # TC2-4: Long Text Case
])
@pytest.mark.asyncio
async def test_predict_valid_cases(model_loader, text):
    result = await model_loader.predict(text)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(model_loader.id2label.values())
    assert all(v in {"pos", "neg", "none"} for v in result.values())

# TC2-5: Invalid Type Case
@pytest.mark.parametrize("invalid_input", [None, 1234, [], {}])
@pytest.mark.asyncio
async def test_predict_invalid_input_type(model_loader, invalid_input):
    with pytest.raises(ValueError, match="text and category must be strings|text must be a string"):
        await model_loader.predict(invalid_input)

# TC2-6: Sentiment values are valid
@pytest.mark.asyncio
async def test_predict_sentiment_value_range(model_loader):
    text = "정말 좋아요!"
    result = await model_loader.predict(text)

    for category, sentiment in result.items():
        assert sentiment in {"pos", "neg", "none"}, f"Invalid sentiment '{sentiment}' for category '{category}'"

# TC3: get_categories Method Tests

@pytest.mark.asyncio
async def test_get_categories(model_loader):
    categories = await model_loader.get_categories()

    # TC3-1: is Type List
    assert isinstance(categories, list)

    # TC3-2: is Type String
    assert all(isinstance(cat, str) for cat in categories)

    # TC3-3: is Not Empty
    assert len(categories) > 0

    # TC3-4: is Unique
    assert len(set(categories)) == len(categories)

    # TC3-5: is Equal to id2label
    assert set(categories) == set(model_loader.id2label.values())
