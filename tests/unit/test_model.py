from app.loadModel import ModelLoader
import torch
import pytest
import asyncio

def test_convert_to_feature_structure():
    model_loader = ModelLoader()
    text = "이 장소 정말 좋았어요"
    category = "atmosphere"
    features = model_loader.convert_to_feature(text, category)

    assert "input_ids" in features
    assert "attention_mask" in features
    assert "token_type_ids" in features

    max_length = model_loader.config['model']['max_length']
    assert features["input_ids"].shape == (1, max_length)
    assert features["attention_mask"].shape == (1, max_length)
    assert features["token_type_ids"].shape == (1, max_length)
    assert isinstance(features["input_ids"], torch.Tensor)


def test_convert_to_feature_empty_text():
    model_loader = ModelLoader()
    text = ""
    category = "atmosphere"
    features = model_loader.convert_to_feature(text, category)

    assert features["input_ids"].shape[1] == model_loader.config["model"]["max_length"]
    assert features["attention_mask"].shape == (1, model_loader.config["model"]["max_length"])


@pytest.mark.asyncio
async def test_predict_output_structure():
    model_loader = ModelLoader()
    text = "날씨도 좋고 조용해서 힐링됐어요"
    result = await model_loader.predict(text)

    assert isinstance(result, dict)
    assert all(label in {"pos", "neg", "none"} for label in result.values())

