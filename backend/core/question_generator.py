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
    """PAGE 단위 → chunk 단위 분해"""
    chunks: List[Dict[str, Any]] = []
    text = (text or "").strip()
    if not text:
        return []

    matches = list(_PAGE_RE.finditer(text))
    pages: List[tuple[int, str]] = []

    if not matches:
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

            chunks.append({
                "kind": "text",
                "page": page_no,
                "chunk_id": chunk_id,
                "preview": preview,
            })

            if len(chunks) >= cfg.max_chunks:
                return chunks

    return chunks


# =========================
# Tables normalize
# =========================

def _normalize_tables_format(tables: Any) -> List[Dict[str, Any]]:
    """표 포맷 표준화"""
    if not isinstance(tables, list):
        return []

    out: List[Dict[str, Any]] = []
    for t in tables:
        if not isinstance(t, dict):
            continue

        page = t.get("page")
        if not isinstance(page, int):
            page = t.get("page_index") or t.get("page_number")
        if not isinstance(page, int):
            page = None

        headers = t.get("headers") or t.get("columns") or t.get("cols")
        if not isinstance(headers, list):
            headers = []

        rows = t.get("rows") or t.get("data") or t.get("values") or t.get("cells")
        if not isinstance(rows, list):
            rows = []

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


def _compute_type_distribution(qn: int, types_ratio: Optional[Dict[str, float]]) -> Dict[str, int]:
    """
    types_ratio에 따라 MCQ/SAQ 문제 수 계산

    Args:
        qn: 총 문제 수
        types_ratio: {"MCQ": 0.7, "SAQ": 0.3} 형태

    Returns:
        {"MCQ": n, "SAQ": m}
    """
    if not types_ratio or not isinstance(types_ratio, dict):
        # 기본값: 전부 MCQ
        return {"MCQ": qn, "SAQ": 0}

    mcq_ratio = float(types_ratio.get("MCQ", 1.0))
    saq_ratio = float(types_ratio.get("SAQ", 0.0))

    # 비율 정규화
    total_ratio = mcq_ratio + saq_ratio
    if total_ratio <= 0:
        return {"MCQ": qn, "SAQ": 0}

    mcq_ratio /= total_ratio
    saq_ratio /= total_ratio

    n_mcq = round(qn * mcq_ratio)
    n_saq = qn - n_mcq

    return {"MCQ": max(0, n_mcq), "SAQ": max(0, n_saq)}


def _compute_difficulty_distribution(
    qn: int,
    difficulty: str,
) -> Dict[str, int]:
    """
    difficulty 설정에 따라 난이도별 문제 수 계산

    Args:
        qn: 총 문제 수
        difficulty: "easy" | "medium" | "hard" | "mixed"

    Returns:
        {"easy": n, "medium": m, "hard": k}
    """
    difficulty = (difficulty or "mixed").strip().lower()

    if difficulty == "easy":
        return {"easy": qn, "medium": 0, "hard": 0}
    elif difficulty == "medium":
        return {"easy": 0, "medium": qn, "hard": 0}
    elif difficulty == "hard":
        return {"easy": 0, "medium": 0, "hard": qn}
    else:  # mixed
        # 균등 분배: easy:2, medium:2, hard:1 (5문제 기준)
        n_easy = max(1, qn // 4)
        n_hard = max(1, qn // 4)
        n_medium = qn - n_easy - n_hard
        return {"easy": n_easy, "medium": n_medium, "hard": n_hard}


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
    num_tables = len(tables)
    has_tables = num_tables > 0

    constraints = job.get("constraints") if isinstance(job.get("constraints"), dict) else {}
    must_use_tables = bool(constraints.get("must_use_tables")) or bool(constraints.get("has_tables_in_job"))

    # types_ratio에 따른 MCQ/SAQ 분포 계산
    types_ratio = job.get("types_ratio")
    type_dist = _compute_type_distribution(qn, types_ratio)
    n_mcq = type_dist["MCQ"]
    n_saq = type_dist["SAQ"]

    # difficulty에 따른 난이도 분포 계산
    difficulty_setting = job.get("difficulty", "mixed")
    diff_dist = _compute_difficulty_distribution(qn, difficulty_setting)
    n_easy = diff_dist["easy"]
    n_medium = diff_dist["medium"]
    n_hard = diff_dist["hard"]

    # 표 기반 문제 개수(표가 있으면 최소 절반)
    if has_tables:
        n_table_questions = max(1, (qn + 1) // 2)
    else:
        n_table_questions = 0

    # 표 데이터를 프롬프트에 포함
    table_section = ""
    if has_tables:
        table_section = "\n" + "=" * 70 + "\n📊 추출된 표 데이터(원문에서 파싱됨)\n" + "=" * 70 + "\n"
        for i, tbl in enumerate(tables, 1):
            page = tbl.get("page")
            headers = tbl.get("headers") or []
            rows = tbl.get("rows") or []
            content = tbl.get("content")

            snippet = ""
            if isinstance(content, str) and content.strip():
                snippet = _truncate(content.strip(), 1500)
            else:
                try:
                    sample = {
                        "headers": headers[:12],
                        "rows_sample": rows[:8],
                        "total_rows": len(rows),
                    }
                    snippet = _truncate(json.dumps(sample, ensure_ascii=False, indent=2), 1500)
                except Exception:
                    snippet = ""

            table_section += f"\n[표 {i}] (페이지 {page})\n{snippet}\n"

    # 표 지시문(교수 스타일로 강화)
    if has_tables:
        table_instruction = f"""
{"=" * 70}
📊 표 기반 문항 출제 규칙 (표가 있는 섹션이면 필수)
{"=" * 70}

- 총 {qn}문항 중 **최소 {n_table_questions}문항은 반드시 표 기반**으로 출제.
- 표 기반 문항은 JSON에 "generated_table" 필드를 **반드시 포함**.
- "generated_table"은 원본 표를 그대로 복사하지 말고, **구조는 참고하되 수치/시나리오를 새로 구성**.
- 표는 반드시 "풀이 가능"해야 하며, 외부 지식 없이 표만으로 풀려야 함.

[generated_table 품질 조건]
- headers는 의미/단위가 드러나야 함 (예: "지연(ms)", "정확도(%)", "비용(원)")
- 행 3~8개, 열 2~5개 권장
- 수치 열 최소 2개 포함 (비교/계산 가능하도록)
- 계산/비율/효율 문제는 explanation에 **중간 계산 1줄** 포함

[표 기반 문항 유형(다양하게 섞기)]
1) 비교/우선순위: 최대/최소/차이/개선폭
2) 계산/비율/증감률/효율: 평균, 비율, 증가율, 성능/시간 등
3) 조건부 추출: 임계값, 조건 충족 행 개수, 필터링 결과
4) 패턴/추세: 시간/버전/실험조건 변화에 따른 경향
5) 해석/결론: 데이터로부터 타당한 결론 선택

[금지]
- 단순 값 찾기("A의 값은?") 수준의 문항
- 표 없이도 풀 수 있는 문항
- 원본 표 그대로 복붙
"""
    else:
        if must_use_tables:
            # 표가 있어야 하는데 없는 경우: 모델이 헛소리 표를 만들지 않도록 경고를 넣음
            table_instruction = f"""
{"=" * 70}
⚠️ 표 사용 요구됨(중요) / 하지만 현재 입력 tables가 비어있음
{"=" * 70}
- 이 섹션은 표 기반 문항이 요구되지만, 제공된 표 데이터가 없습니다.
- **임의로 표를 꾸며내지 마세요.**
- 표가 없으므로 이번 출력에서는 "generated_table"을 포함하지 마세요.
"""
        else:
            table_instruction = """
**[표 없음]** 이 섹션에는 표가 없으므로 "generated_table" 필드는 포함하지 마세요.
"""

    # 핵심: 교수급 품질 체크리스트(LLM이 스스로 준수하도록 강제)
    quality_block = f"""
{"=" * 70}
🧑‍🏫 교수 출제 원칙 (타당도/변별도/채점가능성)
{"=" * 70}

[1] 근거 정합성 (최우선)
- 각 문항은 evidence를 **1~2개만** 사용.
- 정답/해설은 evidence 청크의 내용(정의/관계/절차/결론)에서만 도출.
- 근거 없이 외부 지식으로만 풀리는 문항 금지.

[2] 단일 정답성 (MCQ)
- 정답은 **오직 1개**여야 함. 복수 정답 가능성이 조금이라도 있으면 문항을 다시 작성.
- 질문 문장에 조건/범위를 명확히 포함 (예: 특정 상황, 전제, 기준).

[3] 오답 설계 (변별도)
- 오답은 "흔한 오개념/유사개념 혼동/경계조건 착각/부분적으로만 맞는 진술" 기반으로 설계.
- 정답과 오답의 길이/형태 유사하게.
- 무관한 오답/너무 자명한 오답 금지.

[4] 난이도 정의 (조작적으로 준수)
- easy: 정의/용어/핵심 문장 확인 (추론 0~1 step)
- medium: 작은 상황 적용/비교/간단 계산 (추론 1~2 step)
- hard: 경계조건/반례/복합 추론/트레이드오프/다단계 계산 (추론 3 step 이상)

[5] 해설 규칙
- explanation은 **2~4문장**.
- 반드시 "왜 정답인지" + "대표 오답 1개가 왜 틀렸는지" 포함.
- 계산형은 중간 계산 1줄 포함.

[6] 문항 다양성
- 동일한 질문 패턴 2회 이상 반복 금지.
- 가능하면 비교/분석형 ≥1, 적용형 ≥1 포함.
"""

    # 출력 스키마 강화: API 명세 필드명 사용
    output_schema = """
{
  "questions": [
    {
      "question_id": "Q001",
      "type": "MCQ",
      "difficulty": "easy",
      "question_text": "문제 내용",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "B",
      "explanation": "2~4문장 해설(정답 근거 + 대표 오답 반박 포함).",
      "source_pages": [5, 6],
      "evidence": [{"kind": "text", "page": 5, "chunk_id": "p5_c00"}],

      "learning_objective": "이 문항이 평가하는 학습 목표(한 줄)",
      "common_misconception": "학생이 자주 하는 오개념(한 줄)",

      "generated_table": {
        "headers": ["열1", "열2"],
        "rows": [["값1","값2"], ["값3","값4"]]
      },
      "table_refs": ["table_5_1"]
    }
  ]
}
""".strip()

    # 유형 구성 텍스트
    type_composition = f"MCQ(객관식) {n_mcq}개"
    if n_saq > 0:
        type_composition += f" / SAQ(단답형) {n_saq}개"

    # 난이도 구성 텍스트
    diff_parts = []
    if n_easy > 0:
        diff_parts.append(f"easy {n_easy}개")
    if n_medium > 0:
        diff_parts.append(f"medium {n_medium}개")
    if n_hard > 0:
        diff_parts.append(f"hard {n_hard}개")
    diff_composition = " / ".join(diff_parts) if diff_parts else "mixed"

    return f"""
당신은 소프트웨어학부 대학 교수로서, 섹션 "{section_id}"에 대한 **고품질 시험 문제**를 출제합니다.

**정확히 {qn}개의 문제**를 생성하세요.

{"=" * 70}
🎯 출제 구성(필수)
{"=" * 70}
- 문제 유형: {type_composition}
- 난이도 분포: {diff_composition}
- 모든 문제는 아래 "근거 청크 목록"에 기반해야 함
- evidence는 각 문항당 1~2개만 사용
- chunk_id는 제공된 목록에 있는 것만 사용
- 모든 텍스트는 한국어
- JSON 형식 출력 (마크다운 블록 금지, 설명 문장 금지)

{"=" * 70}
🚫 형식/품질 위반 시 처리
{"=" * 70}
- JSON이 아니면 실패
- 정답이 애매하거나 복수정답 가능성이 있으면 실패
- evidence와 무관한 문제면 실패
- SAQ는 1~5단어 "용어/구"로만 답 (괄호로 장황한 설명 금지)

{quality_block}

{table_instruction}

{"=" * 70}
섹션 전체 내용(원문)
{"=" * 70}
{job.get('text', '')}
{table_section}

{"=" * 70}
근거 청크 목록(여기서만 evidence 선택 가능)
{"=" * 70}
{chr(10).join(chunk_lines)}

{"=" * 70}
출력 형식(JSON ONLY)
{"=" * 70}
{output_schema}

지금 바로 **{qn}개**의 문제를 JSON으로만 출력하세요.
""".strip()


# =========================
# Postprocess
# =========================

def _ensure_explanation(q: Dict[str, Any]) -> None:
    """explanation 누락 방지"""
    expl = q.get("explanation")
    if isinstance(expl, str) and expl.strip():
        return

    qtype = str(q.get("type") or "").upper()
    ans = q.get("correct_answer") or q.get("answer")

    if qtype == "SAQ":
        if isinstance(ans, str) and ans.strip():
            q["explanation"] = f"정답: '{ans.strip()}'. 근거 청크의 핵심 내용입니다."
        else:
            q["explanation"] = "근거 청크를 바탕으로 단답을 작성하세요."
    else:
        if isinstance(ans, str) and ans.strip():
            q["explanation"] = f"정답은 {ans.strip()}입니다. 근거 청크의 내용과 일치합니다."
        else:
            q["explanation"] = "근거 청크를 바탕으로 정답을 선택하세요."


def _ensure_saq_answer(q: Dict[str, Any]) -> None:
    """SAQ answer 누락 방지 (단답형 강조)"""
    qtype = str(q.get("type") or "").upper()
    if qtype != "SAQ":
        return

    ans = q.get("correct_answer") or q.get("answer")
    if isinstance(ans, str) and ans.strip():
        q["correct_answer"] = ans.strip().split("\n")[0][:100]
        return

    expl = q.get("explanation")
    if isinstance(expl, str) and expl.strip():
        q["correct_answer"] = expl.strip().split("\n")[0][:100]
        return

    q["correct_answer"] = "근거 청크를 바탕으로 핵심 용어를 단답으로 작성하세요."


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


def _normalize_question_fields(q: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM 출력 필드명을 API 명세 필드명으로 정규화

    명세 필드명:
    - question_text (not question)
    - options (not choices)
    - correct_answer (not answer)
    - source_pages (evidence에서 추출)
    - table_refs (generated_table이 있으면 추가)
    """
    out = dict(q)

    # question_text 정규화
    if "question" in out and "question_text" not in out:
        out["question_text"] = out.pop("question")

    # options 정규화
    if "choices" in out and "options" not in out:
        out["options"] = out.pop("choices")

    # correct_answer 정규화
    if "answer" in out and "correct_answer" not in out:
        out["correct_answer"] = out.pop("answer")

    # source_pages 추출 (evidence에서)
    if "source_pages" not in out:
        evidence = out.get("evidence", [])
        if isinstance(evidence, list):
            pages = []
            for ev in evidence:
                if isinstance(ev, dict) and isinstance(ev.get("page"), int):
                    pages.append(ev["page"])
            if pages:
                out["source_pages"] = sorted(set(pages))

    # table_refs 생성 (generated_table이 있는 경우)
    if "table_refs" not in out and out.get("generated_table"):
        source_pages = out.get("source_pages", [])
        if source_pages:
            out["table_refs"] = [f"table_{source_pages[0]}_1"]
        else:
            out["table_refs"] = ["table_ref"]

    return out


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """JSON 추출 (markdown 블록 제거)"""
    if not isinstance(raw, str) or not raw.strip():
        return None
    raw = raw.strip()

    # ```json ... ``` 제거
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

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
    문제 생성 (검증 없음)
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

    # 정확한 개수 맞추기
    target = int(job.get("target_questions") or 0) or 2
    if len(questions) > target:
        questions = questions[:target]

    normed: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            continue

        q = dict(q)
        q["question_id"] = str(q.get("question_id") or f"Q{i:03d}").strip()
        q["type"] = _normalize_type(q.get("type"))
        q["difficulty"] = _normalize_difficulty(q.get("difficulty"))

        # 필드명 정규화 (API 명세 준수)
        q = _normalize_question_fields(q)

        _ensure_saq_answer(q)
        _ensure_explanation(q)

        # 표 없으면 generated_table, table_refs 제거
        if not norm_tables:
            q.pop("generated_table", None)
            q.pop("table_refs", None)

        normed.append(q)

    answers_only = [
        {
            "question_id": q.get("question_id"),
            "type": q.get("type"),
            "correct_answer": q.get("correct_answer"),
            "explanation": q.get("explanation"),
        }
        for q in normed
    ]

    return {
        "questions": normed,
        "answers_only": answers_only,
        "evidence_candidates": chunks,
        "meta": {
            "job_id": job.get("job_id"),
            "section_id": job.get("section_id"),
            "num_questions": len(normed),
            "model": cfg.model,
            "target_questions": target,
            "actual_questions": len(normed),
        },
    }
