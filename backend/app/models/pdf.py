"""PDF 모델"""
from pydantic import BaseModel
from typing import Literal


class PDFStatus(BaseModel):
    """PDF 상태"""
    pdf_id: str
    status: Literal["UPLOADED", "PROCESSING", "COMPLETED", "FAILED"]
    page_count: int


