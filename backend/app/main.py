"""FastAPI 메인 애플리케이션"""
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드 (backend 디렉토리 기준)
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI
from app.api import api_router

# API 프리픽스
API_PREFIX = "/api/v1"

app = FastAPI(
    title="DACOS API",
    description="강의안 PDF 기반 문제 생성 API",
    version="1.0.0"
)

app.include_router(api_router, prefix=API_PREFIX)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {"message": "DACOS API"}


@app.get("/health")
async def health():
    """헬스 체크"""
    return {"status": "ok"}


