# Aplus

강의안 PDF 기반 문제 생성 서비스

## 개요

Aplus는 강의안 PDF를 업로드하면 AI를 활용하여 자동으로 학습 문제를 생성하고, 채점 및 오답노트 기능을 제공하는 웹 서비스입니다.

## 주요 기능

- **PDF 업로드**: 강의안 PDF 파일 업로드
- **문제 생성**: AI 기반 자동 문제 생성 (객관식, 주관식 등)
- **문제 풀이**: 생성된 문제 풀이 인터페이스
- **채점**: 사용자 답안 자동 채점
- **오답노트**: 틀린 문제 모아보기

## 기술 스택

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.x
- **Dependencies**:
  - `fastapi` - 웹 프레임워크
  - `uvicorn` - ASGI 서버
  - `pypdf`, `pymupdf` - PDF 처리
  - `openai` - AI 문제 생성
  - `pydantic` - 데이터 검증

### Frontend
- **Framework**: React 19
- **Language**: TypeScript
- **Build Tool**: Vite
- **Routing**: React Router DOM

## 프로젝트 구조

```
repo/
├── backend/
│   ├── app/                    # FastAPI 애플리케이션
│   │   ├── api/                # API 라우터
│   │   │   ├── pdfs.py         # PDF 업로드/문제 조회 API
│   │   │   ├── grading.py      # 채점 API
│   │   │   └── jobs.py         # Job 관리 API
│   │   ├── models/             # Pydantic 모델
│   │   ├── services/           # 비즈니스 로직
│   │   ├── storage/            # 데이터 저장소
│   │   └── main.py             # FastAPI 엔트리포인트
│   ├── core/                   # 핵심 처리 모듈
│   │   ├── pdf_text.py         # PDF 텍스트 추출
│   │   ├── section_indexer.py  # 섹션 분석
│   │   ├── question_generator.py # 문제 생성
│   │   ├── question_verifier.py  # 문제 검증
│   │   └── ...
│   ├── engine/                 # CLI 도구
│   ├── artifacts/              # 처리 결과물 저장
│   ├── data/                   # 데이터 저장
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── api/                # API 클라이언트
    │   ├── components/         # React 컴포넌트
    │   │   ├── PDFUploader.tsx
    │   │   ├── OptionsForm.tsx
    │   │   ├── QuestionCard.tsx
    │   │   ├── QuizContainer.tsx
    │   │   └── ...
    │   ├── pages/              # 페이지 컴포넌트
    │   │   ├── UploadPage.tsx
    │   │   ├── OptionsPage.tsx
    │   │   ├── LoadingPage.tsx
    │   │   ├── QuizPage.tsx
    │   │   └── WrongNotesPage.tsx
    │   ├── hooks/              # Custom Hooks
    │   ├── types/              # TypeScript 타입
    │   └── App.tsx             # 메인 앱
    └── package.json
```

## 설치 및 실행

### Backend

```bash
cd backend

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
# .env 파일에 OPENAI_API_KEY 설정 필요

# 서버 실행
uvicorn app.main:app --reload
```

Backend 서버: `http://localhost:8000`

### Frontend

```bash
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

Frontend 서버: `http://localhost:5173`

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/pdfs` | PDF 업로드 |
| POST | `/api/v1/pdfs/{pdf_id}/jobs/question-generation` | 문제 생성 Job 시작 |
| GET | `/api/v1/pdfs/{pdf_id}/questions` | 생성된 문제 조회 |
| POST | `/api/v1/questions/{question_id}/grade` | 답안 채점 |
| GET | `/api/v1/users/me/wrong-questions` | 오답노트 조회 |
| GET | `/health` | 헬스 체크 |

## 환경 변수

`.env` 파일을 `backend/` 디렉토리에 생성:

```env
OPENAI_API_KEY=your_openai_api_key
```
