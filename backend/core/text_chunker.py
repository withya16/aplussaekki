# core/question_generator.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from core.llm_text import call_llm_text
from core.text_chunker import TextChunkConfig, split_text_to_chunks

# =========================
# ENUM
# =========================

QUESTION_TYPES = ["MCQ", "SAQ"]
DIFFICULTIES = ["easy", "medium", "hard"]

# =========================
# Config
# =========================

@dataclass
class QuestionGenConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.3

    # chunking 설정
    chunk_chars: int = 450
    preview_chars: int = 120
    max_chunks: int = 60
    include_full_text: bool = False


# =========================
# JSON Parsing
# =========================

def _extract_json(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return raw

    # ```json ... ``` 코드블록 제거
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 가장 바깥 JSON 객체 추출
    s = raw.find("{")
    e = raw.rfind("}")
    if s >= 0 and e > s:
        return raw[s : e + 1]

    return raw


def _safe_json_loads(raw: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(_extract_json(raw))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# =========================
# Normalization (검증/판정 ❌)
# =========================

def _normalize_type(t: Any) -> str:
    s = str(t or "").strip().upper()
    if s in QUESTION_TYPES:
        return s
    if s in ["MULTIPLE CHOICE", "MC", "MULTI"]:
        return "MCQ"
    if s in ["SHORT ANSWER", "SA", "SHORT"]:
        return "SAQ"
    return "MCQ"


def _normalize_difficulty(d: Any) -> str:
    s = str(d or "").strip().lower()
    if s in DIFFICULTIES:
        return s
    if s in ["e", "simple", "basic"]:
        return "easy"
    if s in ["m", "moderate", "normal"]:
        return "medium"
    if s in ["h", "difficult", "complex"]:
        return "hard"
    return "medium"


def _normalize_questions_list(
    data: Dict[str, Any],
    expected_n: int,
) -> List[Dict[str, Any]]:
    qs = data.get("questions", [])
    if not isinstance(qs, list):
        return []

    out: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for i, q in enumerate(qs, start=1):
        if not isinstance(q, dict):
            continue

        q2 = dict(q)

        # question_id
        qid = q2.get("question_id")
        if not isinstance(qid, str) or not qid.strip():
            qid = f"Q{i:03d}"
        qid = qid.strip()
        if qid in seen_ids:
            qid = f"Q{i:03d}"
        seen_ids.add(qid)
        q2["question_id"] = qid

        # type / difficulty
        q2["type"] = _normalize_type(q2.get("type"))
        q2["difficulty"] = _normalize_difficulty(q2.get("difficulty"))

        # 🔴 verdict는 generator 단계에서 항상 OK
        q2["verdict"] = "OK"

        # 🟡 evidence.page 정수 캐스팅
        evs = q2.get("evidence")
        if isinstance(evs, list):
            for ev in evs:
                if isinstance(ev, dict) and "page" in ev:
                    try:
                        ev["page"] = int(ev["page"])
                    except Exception:
                        pass

        out.append(q2)

    if expected_n > 0 and len(out) > expected_n:
        out = out[:expected_n]

    return out


# =========================
# Prompt
# =========================

def _build_prompt(job: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
    qn = int(job.get("target_questions") or 0) or 2

    chunk_lines = [
        f"- {c.get('chunk_id')} (page {c.get('page')}): {c.get('preview','')}"
        for c in chunks
    ]

    return f"""
You are an exam question generator.

Generate exactly {qn} questions based ONLY on the provided content.
Do NOT invent facts.

====================
CONTENT
====================
{job.get("text","")}

====================
EVIDENCE CANDIDATES
====================
Each evidence MUST reference one of the following chunk_ids.

{chr(10).join(chunk_lines)}

====================
QUESTION REQUIREMENTS
====================
- question_id: unique (Q001, Q002, ...)
- type: "MCQ" or "SAQ"
- difficulty: "easy" | "medium" | "hard"
- MCQ:
  - exactly 4 choices
  - answer must be A/B/C/D
- SAQ:
  - no choices
  - short text answer
- explanation: concise and clear
- evidence:
  - at least one item
  - MUST use chunk_id from candidates
  - format: {{ "kind":"text", "page":number, "chunk_id":"..." }}

====================
OUTPUT FORMAT (JSON ONLY)
====================
{{
  "questions": [
    {{
      "question_id": "Q001",
      "type": "MCQ",
      "difficulty": "medium",
      "question": "...",
      "choices": ["A ...", "B ...", "C ...", "D ..."],
      "answer": "B",
      "explanation": "...",
      "evidence": [
        {{ "kind":"text", "page":12, "chunk_id":"p12_c00" }}
      ]
    }}
  ]
}}
""".strip()


# =========================
# Helpers
# =========================

def _build_answers_only(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "question_id": q.get("question_id"),
            "type": q.get("type"),
            "answer": q.get("answer"),
            "explanation": q.get("explanation"),
        }
        for q in questions
    ]


# =========================
# Public API
# =========================

def generate_questions_for_job(
    job: Dict[str, Any],
    cfg: QuestionGenConfig,
) -> Dict[str, Any]:
    job_text = job.get("text") or ""
    qn = int(job.get("target_questions") or 0)

    # 1) evidence 후보 생성
    chunk_cfg = TextChunkConfig(
        chunk_chars=cfg.chunk_chars,
        preview_chars=cfg.preview_chars,
        max_chunks=cfg.max_chunks,
        include_full_text=cfg.include_full_text,
    )
    evidence_candidates = split_text_to_chunks(job_text, chunk_cfg)

    if not evidence_candidates:
        return {
            "questions": [],
            "answers_only": [],
            "meta": {
                "job_id": job.get("job_id"),
                "section_id": job.get("section_id"),
                "error": "NO_EVIDENCE_CHUNKS",
                "num_questions": 0,
                "model": cfg.model,
            },
            "evidence_candidates": [],
        }

    # 2) LLM 호출
    prompt = _build_prompt(job, evidence_candidates)
    raw = call_llm_text(
        prompt=prompt,
        model=cfg.model,
        temperature=cfg.temperature,
    )

    # 3) JSON 파싱
    data = _safe_json_loads(raw)
    if data is None:
        return {
            "questions": [],
            "answers_only": [],
            "meta": {
                "job_id": job.get("job_id"),
                "section_id": job.get("section_id"),
                "error": "LLM_OUTPUT_NOT_JSON",
                "num_questions": 0,
                "model": cfg.model,
            },
            "raw_preview": (raw or "")[:500],
            "evidence_candidates": evidence_candidates,
        }

    # 4) 최소 정규화
    questions = _normalize_questions_list(data, expected_n=qn)
    answers_only = _build_answers_only(questions)

    return {
        "questions": questions,
        "answers_only": answers_only,
        "meta": {
            "job_id": job.get("job_id"),
            "section_id": job.get("section_id"),
            "num_questions": len(questions),
            "model": cfg.model,
        },
        # verifier 단일 진실 소스
        "evidence_candidates": evidence_candidates,
    }
