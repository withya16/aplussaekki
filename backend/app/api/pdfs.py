"""PDF API"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.pdf_service import PDFService
from app.storage.question_store import QuestionStore
from app.models.pdf import PDFStatus
from app.models.question import QuestionListResponse, Question
from app.core.errors import PDFNotFoundError

router = APIRouter(prefix="/pdfs", tags=["pdfs"])


@router.post("", response_model=PDFStatus, status_code=201)
async def upload_pdf(file: UploadFile = File(...)):
    """
    강의안 PDF 업로드
    
    - **file**: 업로드할 PDF 파일
    """
    # 1) 파일 존재/타입 검증
    if file is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "MISSING_FILE", "message": "file 필드는 필수입니다."}
        )

    # Content-Type이 정확히 application/pdf가 아닐 수도 있어서 filename도 같이 체크
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()

    if not (content_type == "application/pdf" or filename.endswith(".pdf")):
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_FILE_TYPE", "message": "PDF 파일만 업로드할 수 있습니다."}
        )

    try:
        pdf_status = await PDFService.upload_pdf(file)
        return pdf_status
    except ValueError as e:
        # PDF 파싱 불가 등
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_PDF", "message": str(e)}
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail={"error": "UPLOAD_FAILED", "message": "PDF 업로드 처리 중 서버 오류가 발생했습니다."}
        )

@router.post("/{pdf_id}/jobs/question-generation", status_code=202)
async def create_question_generation_job(pdf_id: str):
    """
    문제 생성 Job 시작
    
    - **pdf_id**: PDF ID
    """
    from app.storage.pdf_store import PDFStore
    import subprocess
    import time
    
    # PDF 존재 확인
    if not PDFStore.pdf_exists(pdf_id):
        raise PDFNotFoundError(pdf_id)
    
    # Job ID 생성
    job_id = f"job_{pdf_id}_{int(time.time())}"
    
    # 엔진 실행 (백그라운드)
    subprocess.Popen([
        "python", "-m", "backend.engine.cli.run_job",
        "--pdf_id", pdf_id
    ])
    
    return {
        "job_id": job_id,
        "pdf_id": pdf_id,
        "status": "QUEUED"
    }

@router.get("/{pdf_id}/questions", response_model=QuestionListResponse)
async def get_questions(pdf_id: str):
    """
    문제 조회
    
    - **pdf_id**: PDF ID
    """
    # PDF 존재 확인
    from app.storage.pdf_store import PDFStore
    if not PDFStore.pdf_exists(pdf_id):
        raise PDFNotFoundError(pdf_id)
    
    # 문제 목록 로드
    questions_data = QuestionStore.load_questions(pdf_id)
    if questions_data is None:
        return QuestionListResponse(items=[])
    
    # Question 모델로 변환 (안전하게)
    questions = []
    for q_data in questions_data:
        if isinstance(q_data, Question):
            questions.append(q_data)
        else:
            questions.append(Question(**q_data))
    
    return QuestionListResponse(items=questions)

