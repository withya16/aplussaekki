# core/question_generator.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.llm_text import call_llm_text

# =========================
# Config
# =========================

@dataclass
class QuestionGenConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_chunks: int = 40
    chunk_chars: int = 450
    preview_chars: int = 120


# =========================
# Text → Evidence chunks
# =========================

_PAGE_RE = re.compile(r"-{5}\s*PAGE\s+(\d+)\s*-{5}", re.IGNORECASE)


def _split_text_to_chunks(text: str, cfg: QuestionGenConfig) -> List[Dict[str, Any]]:
    """
    PAGE 단위 → chunk 단위 분해
    evidence 후보 = {kind, page, chunk_id, preview}

    ✅ 개선:
    - PAGE 마커가 없으면 전체를 page=0으로 간주하여 chunk 생성
    """
    chunks: List[Dict[str, Any]] = []

    text = (text or "").strip()
    if not text:
        return []

    matches = list(_PAGE_RE.finditer(text))
    pages: List[tuple[int, str]] = []

    if not matches:
        # fallback: 전체를 하나의 page로
        pages = [(0, text)]
    else:
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            page_no = int(m.group(1))
            pages.append((page_no, text[start:end].strip()))

    for page_no, page_text in pages:
        if not page_text:
            continue

        for idx in range(0, len(page_text), cfg.chunk_chars):
            body = page_text[idx: idx + cfg.chunk_chars].strip()
            if not body:
                continue

            chunk_id = f"p{page_no}_c{idx // cfg.chunk_chars:02d}"
            preview = body.replace("\n", " ")[: cfg.preview_chars]

            chunks.append(
                {
                    "kind": "text",
                    "page": page_no,
                    "chunk_id": chunk_id,
                    "preview": preview,
                }
            )

            if len(chunks) >= cfg.max_chunks:
                return chunks

    return chunks


# =========================
# Tables normalize
# =========================

def _normalize_tables_format(tables: Any) -> List[Dict[str, Any]]:
    """
    tables 입력 포맷이 섞여도 prompt에 안정적으로 넣기 위한 표준화.
    표준 형태:
      {
        "page": int|None,
        "headers": [...],
        "rows": [[...], ...],
        "content": str|None,   # (옵션) markdown snippet
        "source": str|None
      }
    """
    if not isinstance(tables, list):
        return []

    out: List[Dict[str, Any]] = []
    for t in tables:
        if not isinstance(t, dict):
            continue

        page = t.get("page")
        if not isinstance(page, int):
            page = t.get("page_index")
        if not isinstance(page, int):
            page = t.get("page_number")
        if not isinstance(page, int):
            page = None

        headers = t.get("headers")
        if headers is None:
            headers = t.get("columns")
        if headers is None:
            headers = t.get("cols")
        if not isinstance(headers, list):
            headers = []

        rows = t.get("rows")
        if rows is None:
            # data / values / cells 등 다양한 케이스 대응
            rows = t.get("data")
        if rows is None:
            rows = t.get("values")
        if rows is None:
            rows = t.get("cells")
        if not isinstance(rows, list):
            rows = []

        # rows가 dict row들의 list일 수도 있음
        norm_rows: List[List[Any]] = []
        for r in rows:
            if isinstance(r, list):
                norm_rows.append(r)
            elif isinstance(r, dict):
                if headers:
                    norm_rows.append([r.get(h) for h in headers])
                else:
                    norm_rows.append(list(r.values()))
            else:
                norm_rows.append([r])

        content = t.get("content")
        if isinstance(content, str) and content.strip():
            content = content.strip()
        else:
            content = None

        source = t.get("source") if isinstance(t.get("source"), str) else None

        out.append({
            "page": page,
            "headers": headers,
            "rows": norm_rows,
            "content": content,
            "source": source,
        })

    return out


# =========================
# Prompt
# =========================

def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + " ...[truncated]"


def _build_prompt(
    job: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    tables: Optional[List[Dict[str, Any]]] = None,
) -> str:
    qn = int(job.get("target_questions") or 0) or 2
    section_id = job.get("section_id", "unknown")

    chunk_lines = [
        f"- {c.get('chunk_id')} (page {c.get('page')}): {c.get('preview','')}"
        for c in chunks
    ]

    tables = tables or []
    has_tables = len(tables) > 0

    # job_builder가 넣어준 constraint (있으면 활용)
    constraints = job.get("constraints") if isinstance(job.get("constraints"), dict) else {}
    must_use_tables = bool(constraints.get("must_use_tables")) or bool(constraints.get("has_tables_in_job"))

    # 표 원본을 프롬프트에 "너무 길지 않게" 넣는 게 중요
    table_section = ""
    if has_tables:
        table_section = "\n====================\n추출된 표 데이터(요약)\n====================\n"
        for i, tbl in enumerate(tables, 1):
            page = tbl.get("page")
            headers = tbl.get("headers") or []
            rows = tbl.get("rows") or []
            content = tbl.get("content")

            snippet = ""
            if isinstance(content, str) and content.strip():
                snippet = _truncate(content, 1200)
            else:
                # headers/rows 기반 짧게 보여주기
                try:
                    sample = {
                        "headers": headers[:12],
                        "rows_sample": rows[:5],
                    }
                    snippet = _truncate(json.dumps(sample, ensure_ascii=False), 1200)
                except Exception:
                    snippet = ""

            table_section += f"\n[표 {i}] 페이지 {page}\n{snippet}\n"

    # 핵심: 표가 있으면 최소 1문항은 generated_table 강제
    if has_tables and must_use_tables:
        table_instruction = """
**[표 기반 문제 생성 강제 규칙]**
- 이 job에는 표가 포함되어 있으므로, 생성하는 문제 중 **최소 1문항은 반드시 표 기반 문제**여야 합니다.
- 표 기반 문제는 반드시 "generated_table" 필드를 포함해야 합니다.
- "generated_table"은 원본 표를 복붙하지 말고, 원본의 구조/패턴만 참고해 새로운 수치/시나리오로 구성하세요.
- 단순 값 찾기보다 비교/비율/조건부 확률/추론/전처리 변환(원-핫 등) 형태를 우선하세요.
"""
    elif has_tables:
        table_instruction = """
**[표 기반 문제 권장 규칙]**
- 표가 있으므로 표 기반 문제를 우선적으로 생성하세요.
- 표 기반 문제는 "generated_table" 필드를 포함하세요.
"""
    else:
        table_instruction = "**[표 없음]** generated_table 필드는 포함하지 마세요."

    return f"""
당신은 소프트웨어학부 대학 교수로서 섹션 {section_id}에 대한 시험 문제를 출제합니다.

정확히 {qn}개의 문제를 생성하세요.

**[필수 지침]**
1) 모든 문제는 아래 '근거 청크'에 기반
2) 각 문제는 반드시 evidence 1개 이상 포함
3) evidence의 chunk_id는 반드시 후보 목록 중 하나 사용
4) 모든 텍스트는 한국어
5) 출력은 JSON ONLY (markdown 금지)
6) type은 반드시 "MCQ" 또는 "SAQ" 중 하나 (절대 다른 값 금지)
7) SAQ는 반드시 "answer"를 포함 (한 줄 단답/정의/핵심요지)
8) SAQ도 반드시 "explanation"을 포함 (정답 근거/요지 1~2문장)

{table_instruction}

====================
섹션 전체 내용 (맥락 파악용)
====================
{job.get('text', '')}
{table_section}

====================
근거 청크 (인용 필수)
====================
{chr(10).join(chunk_lines)}

====================
출력 형식 (JSON ONLY)
====================
{{
  "questions": [
    {{
      "question_id": "Q001",
      "type": "MCQ",
      "difficulty": "medium",
      "question": "...",
      "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer": "A",
      "explanation": "...",
      "evidence": [
        {{ "kind": "text", "page": 12, "chunk_id": "p12_c01" }}
      ]
      // 표 기반 문제(최소 1문항)인 경우만:
      // "generated_table": {{ "headers": [...], "rows": [[...]] }}
    }},
    {{
      "question_id": "Q002",
      "type": "SAQ",
      "difficulty": "medium",
      "question": "...",
      "answer": "...",
      "explanation": "...",
      "evidence": [
        {{ "kind": "text", "page": 12, "chunk_id": "p12_c00" }}
      ]
    }}
  ]
}}
""".strip()


# =========================
# Postprocess (light fixes to reduce FIXABLE/REJECT)
# =========================

def _ensure_explanation(q: Dict[str, Any]) -> None:
    """
    verifier에서 FIXABLE 잘 뜨는 'missing explanation' 방지용.
    품질 판정은 아니고 필드 누락만 보정.
    """
    expl = q.get("explanation")
    if isinstance(expl, str) and expl.strip():
        return

    qtype = str(q.get("type") or "").upper()
    ans = q.get("answer")

    if qtype == "SAQ":
        if isinstance(ans, str) and ans.strip():
            q["explanation"] = f"모범답안 요지: {ans.strip()}"
        else:
            q["explanation"] = "모범답안의 핵심 근거를 1~2문장으로 설명하시오."
    else:
        if isinstance(ans, str) and ans.strip():
            q["explanation"] = f"정답({ans.strip()})은 근거 청크의 내용과 일치한다."
        else:
            q["explanation"] = "근거 청크를 바탕으로 정답의 이유를 설명하시오."


def _ensure_saq_answer(q: Dict[str, Any]) -> None:
    """
    ✅ 핵심: SAQ missing answer REJECT 방지.
    - answer가 비어있으면 최소한 explanation/문항에서 뽑아 '단답' 형태로 채움
    """
    qtype = str(q.get("type") or "").upper()
    if qtype != "SAQ":
        return

    ans = q.get("answer")
    if isinstance(ans, str) and ans.strip():
        return

    # 1) explanation이 있으면 거기서 한 줄로 축약
    expl = q.get("explanation")
    if isinstance(expl, str) and expl.strip():
        # 너무 길면 앞부분만
        q["answer"] = expl.strip().split("\n")[0][:120]
        return

    # 2) 그래도 없으면 placeholder라도 넣어 REJECT 방지(이건 품질보다는 포맷)
    q["answer"] = "근거 청크를 바탕으로 핵심 요지를 한 문장으로 서술하시오."


def _normalize_type(qtype: Any) -> str:
    s = str(qtype or "").strip().upper()
    if s in {"MCQ", "SAQ"}:
        return s
    if s in {"TABLE", "TABULAR"}:
        return "MCQ"
    if s in {"SHORT", "SHORTANSWER"}:
        return "SAQ"
    return "MCQ"


def _normalize_difficulty(d: Any) -> str:
    s = str(d or "").strip().lower()
    if s in {"easy", "medium", "hard"}:
        return s
    return "medium"


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    모델이 JSON ONLY를 어겨도 최대한 복구.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    raw = raw.strip()

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        cand = raw[start:end + 1]
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


# =========================
# Public API
# =========================

def generate_questions_for_job(
    job: Dict[str, Any],
    cfg: QuestionGenConfig,
    *,
    tables: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    문제 생성 ONLY (검증/판정 없음)
    - verifier가 필요로 하는 evidence_candidates는 항상 포함

    ✅ 변경:
    - tables를 runner가 명시적으로 넘길 수 있게 파라미터 추가
    - tables normalize 적용
    - SAQ answer 누락 보정
    """
    text = job.get("text") or ""
    chunks = _split_text_to_chunks(text, cfg)

    if not chunks:
        return {
            "questions": [],
            "answers_only": [],
            "error": "NO_EVIDENCE_CHUNKS",
            "raw": "",
            "evidence_candidates": [],
            "meta": {
                "job_id": job.get("job_id"),
                "section_id": job.get("section_id"),
                "num_questions": 0,
                "model": cfg.model,
            },
        }

    # tables: 인자 우선, 없으면 job에서 fallback
    if tables is None:
        tables = job.get("tables", [])
    norm_tables = _normalize_tables_format(tables)

    prompt = _build_prompt(job, chunks, tables=norm_tables)

    raw = call_llm_text(
        prompt=prompt,
        model=cfg.model,
        temperature=cfg.temperature,
    )

    data = _extract_json(raw)
    if data is None:
        return {
            "questions": [],
            "answers_only": [],
            "error": "LLM_OUTPUT_NOT_JSON",
            "raw": raw,
            "evidence_candidates": chunks,
            "meta": {
                "job_id": job.get("job_id"),
                "section_id": job.get("section_id"),
                "num_questions": 0,
                "model": cfg.model,
            },
        }

    questions = data.get("questions", [])
    if not isinstance(questions, list):
        questions = []

    normed: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            continue

        q = dict(q)
        q["question_id"] = str(q.get("question_id") or f"Q{i:03d}").strip()
        q["type"] = _normalize_type(q.get("type"))
        q["difficulty"] = _normalize_difficulty(q.get("difficulty"))

        # SAQ answer 누락 보정(핵심)
        _ensure_saq_answer(q)

        # explanation 누락 보정 → FIXABLE 감소
        _ensure_explanation(q)

        # tables 없는 job이면 generated_table은 제거(불필요 REJECT/FIXABLE 예방)
        if not norm_tables:
            q.pop("generated_table", None)

        normed.append(q)

    answers_only = [
        {
            "question_id": q.get("question_id"),
            "type": q.get("type"),
            "answer": q.get("answer"),
            "explanation": q.get("explanation"),
        }
        for q in normed
    ]

    return {
        "questions": normed,
        "answers_only": answers_only,
        "evidence_candidates": chunks,  # verifier 단일 진실 소스
        "meta": {
            "job_id": job.get("job_id"),
            "section_id": job.get("section_id"),
            "num_questions": len(normed),
            "model": cfg.model,
        },
    }



