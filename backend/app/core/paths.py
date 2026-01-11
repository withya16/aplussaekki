"""경로 유틸리티 및 데이터 디렉토리 관리"""
from pathlib import Path

# backend/ 디렉토리 경로
BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
DATA_DIR = BASE_DIR / "data"

# 데이터 하위 디렉토리
PDF_DIR = DATA_DIR / "pdfs"
JOB_DIR = DATA_DIR / "jobs"
QUESTION_DIR = DATA_DIR / "questions"
QUESTIONS_DIR = QUESTION_DIR  # 별칭 (question_store에서 사용)
RESULTS_DIR = DATA_DIR / "results"  # 엔진이 저장하는 경로
HISTORY_DIR = DATA_DIR / "history"  # history_store에서 사용


def ensure_dirs() -> None:
    """필요한 디렉토리 생성"""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    QUESTION_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def get_pdf_path(pdf_id: str) -> Path:
    """PDF 파일 경로 반환"""
    return PDF_DIR / f"{pdf_id}.pdf"


def get_job_path(job_id: str) -> Path:
    """Job JSON 파일 경로 반환"""
    return JOB_DIR / f"{job_id}.json"


def get_questions_path(pdf_id: str) -> Path:
    """문제 JSON 파일 경로 반환"""
    return QUESTION_DIR / f"{pdf_id}.json"


# 모듈 로드 시 디렉토리 생성
ensure_dirs()
