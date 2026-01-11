"""API 라우터 모음"""
from fastapi import APIRouter
from app.api import pdfs, jobs, grading

api_router = APIRouter()
api_router.include_router(pdfs.router)
api_router.include_router(jobs.router)
api_router.include_router(grading.router)


