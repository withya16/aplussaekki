# backend/engine/core/question_verifier.py
"""
Phase 1: 로컬 구조 검증 (즉시, Job별)
- 필수 필드 존재
- ENUM 유효성
- MCQ/SAQ 구조 검증
- 속도: 즉시 (~0.01초)
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# ENUM 정의 (API 명세)
# =============================================================================

class QuestionType(str, Enum):
    MCQ = "MCQ"  # 객관식
    SAQ = "SAQ"  # 단답형


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Verdict(str, Enum):
    OK = "OK"           # 정상 → 그대로 사용
    FIXABLE = "FIXABLE" # 수정 가능 → 재생성 트리거
    REJECT = "REJECT"   # 반려 → 폐기


# Set 버전 (빠른 lookup용)
QUESTION_TYPES = {t.value for t in QuestionType}
DIFFICULTIES = {d.value for d in Difficulty}
VERDICTS = {v.value for v in Verdict}


# =============================================================================
# Helpers
# =============================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _upgrade_verdict(current: str, new: str) -> str:
    """verdict 우선순위: REJECT > FIXABLE > OK"""
    priority = {"OK": 0, "FIXABLE": 1, "REJECT": 2}
    c = current if current in priority else "OK"
    n = new if new in priority else "OK"
    return n if priority[n] > priority[c] else c


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _normalize_text(text: str) -> str:
    """텍스트 정규화 (중복 비교용)"""
    if not text:
        return ""
    return " ".join(text.lower().split())


# =============================================================================
# Phase 1: 로컬 구조 검증
# =============================================================================

class StructureVerifyConfig:
    """구조 검증 설정"""
    MIN_QUESTION_LEN: int = 10
    MAX_QUESTION_LEN: int = 700
    MIN_EXPLANATION_LEN: int = 10
    MIN_ANSWER_LEN: int = 1
    MCQ_CHOICE_COUNT: int = 4
    MCQ_VALID_ANSWERS: Set[str] = {"A", "B", "C", "D"}


def verify_question_structure(
    question: Dict[str, Any],
    config: Optional[StructureVerifyConfig] = None,
) -> Dict[str, Any]:
    """
    단일 문제 구조 검증

    Returns:
        {
            "question_id": str,
            "verdict": "OK" | "FIXABLE" | "REJECT",
            "issues": List[str],
            "verified_at": str
        }
    """
    if config is None:
        config = StructureVerifyConfig()

    issues: List[str] = []
    verdict = "OK"

    # 1) 필수 필드 존재 검증
    qid = question.get("question_id", "")
    if not _is_nonempty_str(qid):
        issues.append("missing question_id")
        verdict = _upgrade_verdict(verdict, "REJECT")

    qtype = question.get("type")
    if qtype not in QUESTION_TYPES:
        issues.append(f"invalid type: {qtype} (expected MCQ or SAQ)")
        verdict = _upgrade_verdict(verdict, "REJECT")

    # 2) question_text 검증
    q_text = question.get("question_text") or question.get("question", "")
    if not _is_nonempty_str(q_text):
        issues.append("missing question_text")
        verdict = _upgrade_verdict(verdict, "REJECT")
    else:
        q_len = len(q_text.strip())
        if q_len < config.MIN_QUESTION_LEN:
            issues.append(f"question_text too short ({q_len} < {config.MIN_QUESTION_LEN})")
            verdict = _upgrade_verdict(verdict, "FIXABLE")
        elif q_len > config.MAX_QUESTION_LEN:
            issues.append(f"question_text too long ({q_len} > {config.MAX_QUESTION_LEN})")
            verdict = _upgrade_verdict(verdict, "FIXABLE")

    # 3) difficulty 검증
    diff = question.get("difficulty")
    if diff not in DIFFICULTIES:
        issues.append(f"invalid difficulty: {diff}")
        verdict = _upgrade_verdict(verdict, "FIXABLE")

    # 4) MCQ 구조 검증
    if qtype == "MCQ":
        # 4-1) choices/options 검증
        choices = question.get("choices") or question.get("options", [])
        if not isinstance(choices, list):
            issues.append("MCQ missing choices/options")
            verdict = _upgrade_verdict(verdict, "REJECT")
        elif len(choices) != config.MCQ_CHOICE_COUNT:
            issues.append(f"MCQ must have {config.MCQ_CHOICE_COUNT} choices (got {len(choices)})")
            verdict = _upgrade_verdict(verdict, "REJECT")
        else:
            # 빈 선지 검사
            empty_choices = [i for i, c in enumerate(choices) if not _is_nonempty_str(str(c))]
            if empty_choices:
                issues.append(f"MCQ has empty choices at index: {empty_choices}")
                verdict = _upgrade_verdict(verdict, "REJECT")

            # 중복 선지 검사
            norm_choices = [_normalize_text(str(c)) for c in choices]
            if len(set(norm_choices)) != len(norm_choices):
                issues.append("MCQ has duplicate choices")
                verdict = _upgrade_verdict(verdict, "FIXABLE")

        # 4-2) answer 검증 (A/B/C/D)
        answer = question.get("answer") or question.get("correct_answer", "")
        if answer not in config.MCQ_VALID_ANSWERS:
            issues.append(f"MCQ answer must be one of {config.MCQ_VALID_ANSWERS} (got: {answer})")
            verdict = _upgrade_verdict(verdict, "REJECT")

    # 5) SAQ 구조 검증
    if qtype == "SAQ":
        answer = question.get("answer") or question.get("correct_answer", "")
        if not _is_nonempty_str(answer):
            issues.append("SAQ missing answer")
            verdict = _upgrade_verdict(verdict, "REJECT")

        # SAQ는 choices가 없어야 함
        choices = question.get("choices") or question.get("options")
        if choices and isinstance(choices, list) and len(choices) > 0:
            issues.append("SAQ should not have choices")
            verdict = _upgrade_verdict(verdict, "FIXABLE")

    # 6) explanation 검증
    explanation = question.get("explanation", "")
    if not _is_nonempty_str(explanation):
        issues.append("missing explanation")
        verdict = _upgrade_verdict(verdict, "FIXABLE")
    elif len(explanation.strip()) < config.MIN_EXPLANATION_LEN:
        issues.append(f"explanation too short ({len(explanation.strip())} < {config.MIN_EXPLANATION_LEN})")
        verdict = _upgrade_verdict(verdict, "FIXABLE")

    # 7) generated_table 검증 (있는 경우만)
    gen_table = question.get("generated_table")
    if gen_table is not None:
        if not isinstance(gen_table, dict):
            issues.append("generated_table must be object")
            verdict = _upgrade_verdict(verdict, "FIXABLE")
        else:
            headers = gen_table.get("headers")
            rows = gen_table.get("rows")

            if not isinstance(headers, list) or not headers:
                issues.append("generated_table missing headers")
                verdict = _upgrade_verdict(verdict, "FIXABLE")

            if not isinstance(rows, list) or not rows:
                issues.append("generated_table missing rows")
                verdict = _upgrade_verdict(verdict, "FIXABLE")
            elif isinstance(headers, list) and headers:
                for i, row in enumerate(rows):
                    if not isinstance(row, list):
                        issues.append(f"generated_table row {i} must be array")
                        verdict = _upgrade_verdict(verdict, "FIXABLE")
                        break
                    if len(row) != len(headers):
                        issues.append(f"generated_table row {i} length mismatch")
                        verdict = _upgrade_verdict(verdict, "FIXABLE")

    # confidence 계산: 구조 검증은 확정적이므로 기본 1.0, 이슈마다 감소
    if verdict == "OK":
        confidence = 1.0
    elif verdict == "FIXABLE":
        confidence = max(0.5, 1.0 - len(issues) * 0.1)
    else:  # REJECT
        confidence = max(0.3, 0.8 - len(issues) * 0.1)

    return {
        "question_id": qid if _is_nonempty_str(qid) else "unknown",
        "verdict": verdict,
        "issues": issues,
        "confidence": round(confidence, 2),
        "verified_at": _now_iso(),
    }


def verify_questions_batch(
    questions: List[Dict[str, Any]],
    config: Optional[StructureVerifyConfig] = None,
) -> Dict[str, Any]:
    """
    문제 배치 구조 검증

    Returns:
        {
            "verified_at": str,
            "summary": {"OK": int, "FIXABLE": int, "REJECT": int},
            "results": List[VerifyResult],
            "questions": List[Question with verdict]
        }
    """
    if config is None:
        config = StructureVerifyConfig()

    results: List[Dict[str, Any]] = []
    questions_with_verdict: List[Dict[str, Any]] = []
    summary = {"OK": 0, "FIXABLE": 0, "REJECT": 0}

    seen_texts: Set[str] = set()

    for q in questions:
        if not isinstance(q, dict):
            continue

        result = verify_question_structure(q, config)

        # Job 내 중복 검사
        q_text = q.get("question_text") or q.get("question", "")
        norm_text = _normalize_text(q_text)
        if norm_text and norm_text in seen_texts:
            result["issues"].append("duplicate question in batch")
            result["verdict"] = _upgrade_verdict(result["verdict"], "FIXABLE")
        if norm_text:
            seen_texts.add(norm_text)

        results.append(result)
        summary[result["verdict"]] += 1

        # 원본 문제에 verdict/issues/confidence 추가
        q_out = dict(q)
        q_out["verdict"] = result["verdict"]
        q_out["issues"] = result["issues"]
        q_out["confidence"] = result.get("confidence", 1.0)
        q_out["verified_at"] = result["verified_at"]
        questions_with_verdict.append(q_out)

    return {
        "verified_at": _now_iso(),
        "summary": summary,
        "results": results,
        "questions": questions_with_verdict,
        "stats": {
            "total": len(results),
            "unique_questions": len(seen_texts),
        }
    }


# =============================================================================
# Job 단위 검증 (기존 인터페이스 호환)
# =============================================================================

def verify_questions_for_job(
    *,
    job: Dict[str, Any],
    generator_result: Dict[str, Any],
    evidence_chunks: Optional[List[Dict[str, Any]]] = None,  # 더 이상 사용 안함 (호환성 유지)
) -> Dict[str, Any]:
    """
    Job 단위 구조 검증 (Phase 1)

    기존 인터페이스 호환을 위해 유지.
    내부적으로 verify_questions_batch 호출.
    """
    questions = generator_result.get("questions", [])
    if not isinstance(questions, list):
        questions = []

    batch_result = verify_questions_batch(questions)

    return {
        "job_id": job.get("job_id"),
        "section_id": job.get("section_id"),
        "verified_at": batch_result["verified_at"],
        "summary": batch_result["summary"],
        "questions": batch_result["questions"],
        "stats": batch_result["stats"],
    }
