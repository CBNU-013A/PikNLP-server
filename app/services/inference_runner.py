# /app/services/inference_runner.py

from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch
import yaml
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..core.logger import logger

class ModelLoader:
    def __init__(self, config_path: str = "app/services/config.yaml"):
        # config 파일 로드
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logger.info("Loaded configuration file from %s", config_path)
            
        # device 설정
        device_config = self.config['model']['device']
        self.device = torch.device(device_config if device_config == 'cuda' and torch.cuda.is_available() else 'cpu')
        logger.info("Using device: %s", self.device)

        # 모델 및 토크나이저 로드
        self.model_name = self.config['model']['name']
        self.tokenizer_name = self.config['model']['tokenizer_name']
        self.tokenizer = ElectraTokenizer.from_pretrained(self.tokenizer_name)
        self.model = ElectraForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        logger.info("Loaded model: %s", self.model_name)
        logger.info("Loaded tokenizer: %s", self.tokenizer_name)

        # 모델의 카테고리 매핑 로드
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(self.model_name)
        self.id2label = model_config.id2label
        self.label2id = model_config.label2id
            
        # 감성 레이블 매핑
        self.sentiment_map = {0: "pos", 1: "neg", 2: "none"}
        
        # 스레드 풀 생성 (CPU 작업용)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config['inference']['num_workers'])
        logger.info("✅ ModelLoader initialized successfully.")

    def convert_to_feature(self, text: str, category: str):
        # 입력값 검증
        if not isinstance(text, str) or not isinstance(category, str):
            raise ValueError("text and category must be strings")

        max_length = self.config['model']['max_length']
        encoded = self.tokenizer(
            text=text,
            text_pair=category,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": encoded.get("token_type_ids", torch.zeros((1, max_length), dtype=torch.long))
        }
    
    async def _predict_category(self, text: str, category: str) -> tuple[str, str]:
        logger.debug("Starting prediction for category: %s", category)
        # 토크나이징은 CPU에서 수행
        inputs = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self.convert_to_feature,
            text,
            category
        )
        
        # GPU 연산은 별도의 스레드에서 실행
        def run_inference():
            inputs_gpu = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs_gpu)
                predictions = torch.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
            return self.sentiment_map[predicted_class]
            
        sentiment = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            run_inference
        )
        logger.debug("Predicted sentiment for category %s: %s", category, sentiment)
        
        return category, sentiment
        
    async def predict(self, text: str) -> dict:
        # 입력값 검증
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        logger.info("Starting prediction for text: %s", text)
        # 모든 카테고리에 대해 병렬로 추론 수행
        tasks = [
            self._predict_category(text, category)
            for category in self.id2label.values()
        ]
        
        # 결과 수집
        results = {}
        for category, sentiment in await asyncio.gather(*tasks):
            results[category] = sentiment
        logger.info("✅ Completed prediction")
            
        return results
    
    async def get_categories(self) -> list[str]:
        logger.info("Fetching categories")
        # 카테고리 목록 반환
        categories = list(self.id2label.values())
        logger.info("✅ Fetched categories: %s", categories)
        return categories

# 싱글톤 인스턴스 생성
model_loader = ModelLoader()
