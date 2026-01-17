"""Job 모델"""
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Literal, Optional


class Progress(BaseModel):
    """진행 상황"""
    stage: str
    done: int
    total: int


class JobError(BaseModel):
    """Job 에러 정보"""
    code: str
    message: str


class JobStatus(BaseModel):
    """Job 상태"""
    model_config = ConfigDict(populate_by_name=True)
    
    job_id: str
    status: Literal["QUEUED", "RUNNING", "DONE", "FAILED"]
    progress: Optional[Progress] = None
    error: Optional[JobError] = None
    
    def model_dump(self, **kwargs):
        """None 값을 제외하고 직렬화 (API 명세에 맞춤)"""
        if "exclude_none" not in kwargs:
            kwargs["exclude_none"] = True
        return super().model_dump(**kwargs)
    
    def model_dump_json(self, **kwargs):
        """JSON 직렬화 시 None 값 제외 (API 명세에 맞춤)"""
        if "exclude_none" not in kwargs:
            kwargs["exclude_none"] = True
        return super().model_dump_json(**kwargs)


class TypesRatio(BaseModel):
    """문제 유형 비율"""
    MCQ: float = Field(..., ge=0.0, le=1.0, description="객관식 비율 (0~1)")
    SAQ: float = Field(..., ge=0.0, le=1.0, description="단답형 비율 (0~1)")

    @model_validator(mode='after')
    def validate_sum(self):
        """MCQ와 SAQ의 합이 1.0인지 검증"""
        total = self.MCQ + self.SAQ
        if abs(total - 1.0) > 0.001:  # 부동소수점 오차 허용
            raise ValueError(f"types_ratio.MCQ + types_ratio.SAQ는 1.0이어야 합니다. 현재: {total}")
        return self


class Chunking(BaseModel):
    """PDF 처리 방식"""
    mode: Literal["whole", "chunked"] = Field(..., description="PDF 처리 방식 (whole/chunked)")
    pages_per_chunk: Optional[int] = Field(None, ge=1, description="chunked 모드일 때 페이지 수")

    @model_validator(mode='after')
    def validate_pages_per_chunk(self):
        """chunked 모드일 때 pages_per_chunk 필수"""
        if self.mode == "chunked" and self.pages_per_chunk is None:
            raise ValueError("chunked 모드일 때 pages_per_chunk는 필수입니다.")
        return self


class JobCreateBody(BaseModel):
    """문제 생성 Job 요청 Body"""
    num_questions: int = Field(..., ge=1, description="생성할 문제 개수")
    difficulty: Literal["easy", "medium", "hard", "mixed"] = Field(..., description="난이도")
    types_ratio: TypesRatio = Field(..., description="문제 유형 비율")
    chunking: Chunking = Field(..., description="PDF 처리 방식")


class JobCreateResponse(BaseModel):
    """문제 생성 Job 응답"""
    job_id: str
    pdf_id: str
    status: Literal["QUEUED", "RUNNING", "DONE", "FAILED"]


