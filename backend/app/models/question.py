"""Question 모델"""
from pydantic import BaseModel
from typing import Literal, Optional, List


class Question(BaseModel):
    """문제"""
    question_id: str
    type: Literal["MCQ", "SAQ"]
    question: str
    options: Optional[List[str]] = None
    source: Optional[str] = None  # optional 필드


class QuestionListResponse(BaseModel):
    """문제 목록 응답"""
    items: List[Question]


