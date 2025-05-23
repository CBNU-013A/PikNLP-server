# PikNLP-server

[PikNLP](https://github.com/CBNU-013A/PikNLP)(ELECTRA 기반 자연어 처리 모델)을 서빙하는 FastAPI 서버

## 실행방법

> [!warning]
> Docker를 사용하지 않는 경우, [Poetry](https://python-poetry.org/docs/#installing-with-pipx)가 필요합니다.

### poetry 환경 구성
```bash
# 1. 의존성 설치
poetry install
# 2. 서버 실행(port=18000)
poetry run python run.py
# or
poetry run uvicorn app.main:app <port>
```

### Docker

GPU(pytorch:2.7.0-cuda12.6-cudnn9-runtime) 사용

```bash
docker build -t piknlp-server .
docker run --gpus all -p <port> --env-file .env piknlp-server:latest
```


## 📡 API 명세

### 1. `GET /health`

- 서버 및 모델 상태 확인

#### ✅ 응답 예시
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "cuda_available": true,
  "API-MODE": "production"
}
```

### 2. `POST /api/v1/predict`
- 리뷰 텍스트에 대한 감성 예측 수행

#### 📥 요청 형식 (JSON)
```json
{
  "text": "날씨도 좋고 조용해서 힐링됐어요"
}
```

#### 📤 응답 형식

> [!Note] 
> 분석 카테고리는 지속적으로 개선 예정입니다.

```json
{
  "sentiments": {
    "가격": "none",
    "가족": "none",
    "역사": "none",
    "접근성": "none",
    "사진": "none",
    "경관": "pos",
    "계절": "none",
    "공원": "pos",
    "문화": "none",
    "체험": "none",
    "음식": "none",
    "시설": "none",
    "자연": "pos"
  }
}

```

### 3. `GET /api/v1/categories`

- 모델에서 분석할 카테고리 목록 확인

#### ✅ 응답 예시

```json
{
  "categories": [
    "가격",
    "가족",
    "역사",
    "접근성",
    "사진",
    "경관",
    "계절",
    "공원",
    "문화",
    "체험",
    "음식",
    "시설",
    "자연"
  ]
}
```