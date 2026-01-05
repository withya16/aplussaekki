# core/question_verifier.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

# =========================
# ENUM (명세 고정)
# =========================
QUESTION_TYPES = {"MCQ", "SAQ"}
DIFFICULTIES = {"easy", "medium", "hard"}
VERDICTS = {"OK", "FIXABLE", "REJECT"}


# =========================
# Helpers
# =========================
def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _collect_valid_chunk_ids(chunks: List[Dict[str, Any]]) -> Set[str]:
    return {
        c.get("chunk_id")
        for c in chunks
        if isinstance(c, dict)
        and isinstance(c.get("chunk_id"), str)
        and c.get("chunk_id").strip()
    }


def _upgrade_verdict(current: str, new: str) -> str:
    """
    verdict 우선순위: REJECT > FIXABLE > OK
    더 심각한 쪽으로만 업그레이드
    """
    priority = {"OK": 0, "FIXABLE": 1, "REJECT": 2}
    c = current if current in priority else "OK"
    n = new if new in priority else "OK"
    return n if priority[n] > priority[c] else c


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


# =========================
# Core Verifier
# =========================
def verify_questions_for_job(
    *,
    job: Dict[str, Any],
    generator_result: Dict[str, Any],
    evidence_chunks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Generator 결과에 대해 구조/명세/참조 무결성 검증만 수행 (LLM 호출 없음)

    원칙:
    - Generator verdict는 존중하되(기본값), 규칙 위반 시 더 심각한 쪽으로만 업그레이드
    - 정책적/주관적 품질평가 최소화 (길이 체크는 과도한 REJECT 유발 방지 위해 대부분 FIXABLE)
    - chunker 단일 진실 소스: generator_result["evidence_candidates"]를 기본으로 사용
    """

    questions = generator_result.get("questions", [])
    if not isinstance(questions, list):
        questions = []

    # ✅ chunker 단일 진실 소스: generator_result가 들고온 evidence_candidates를 기본 사용
    if evidence_chunks is None:
        evidence_chunks = generator_result.get("evidence_candidates", [])
    if not isinstance(evidence_chunks, list):
        evidence_chunks = []

    valid_chunk_ids = _collect_valid_chunk_ids(evidence_chunks)

    verified: List[Dict[str, Any]] = []
    summary = {"OK": 0, "FIXABLE": 0, "REJECT": 0}

    seen_question_ids: Set[str] = set()
    seen_question_texts: Set[str] = set()

    # 길이 기준 (너무 공격적이면 FIXABLE 폭증/오판 가능 → 완화)
    MIN_Q_LEN = 8
    MAX_Q_LEN = 700
    MIN_EXPL_LEN = 8

    for q in questions:
        if not isinstance(q, dict):
            continue

        # Generator verdict/issue 존중 (단, invalid면 OK로 시작)
        gen_verdict = q.get("verdict", "OK")
        current_verdict = gen_verdict if gen_verdict in VERDICTS else "OK"

        gen_issues = q.get("issues", [])
        if not isinstance(gen_issues, list):
            gen_issues = []

        additional_issues: List[str] = []

        # =================
        # 1) question_id 중복/누락
        # =================
        qid = q.get("question_id")
        if _is_nonempty_str(qid):
            if qid in seen_question_ids:
                additional_issues.append(f"duplicate question_id: {qid}")
                current_verdict = _upgrade_verdict(current_verdict, "REJECT")
            seen_question_ids.add(qid)
        else:
            additional_issues.append("missing question_id")
            current_verdict = _upgrade_verdict(current_verdict, "REJECT")

        # =================
        # 2) question text 중복 & 길이
        # =================
        qtext_raw = q.get("question")
        qtext_norm = (qtext_raw or "").strip().lower()

        if qtext_norm:
            if qtext_norm in seen_question_texts:
                additional_issues.append("duplicate question text")
                current_verdict = _upgrade_verdict(current_verdict, "FIXABLE")
            seen_question_texts.add(qtext_norm)

            # 길이 체크는 FIXABLE (주관적)
            qlen = len(qtext_norm)
            if qlen < MIN_Q_LEN:
                additional_issues.append(f"question too short (<{MIN_Q_LEN} chars)")
                current_verdict = _upgrade_verdict(current_verdict, "FIXABLE")
            elif qlen > MAX_Q_LEN:
                additional_issues.append(f"question too long (>{MAX_Q_LEN} chars)")
                current_verdict = _upgrade_verdict(current_verdict, "FIXABLE")
        else:
            additional_issues.append("missing question text")
            current_verdict = _upgrade_verdict(current_verdict, "REJECT")

        # =================
        # 3) ENUM 검증
        # =================
        qtype = q.get("type")
        if qtype not in QUESTION_TYPES:
            additional_issues.append(f"invalid question type: {qtype}")
            current_verdict = _upgrade_verdict(current_verdict, "REJECT")

        diff = q.get("difficulty")
        if diff not in DIFFICULTIES:
            additional_issues.append(f"invalid difficulty: {diff}")
            current_verdict = _upgrade_verdict(current_verdict, "FIXABLE")

        # =================
        # 4) evidence 존재 & chunk_id/page 무결성
        # =================
        evs = q.get("evidence")
        if not isinstance(evs, list) or not evs:
            additional_issues.append("missing evidence")
            current_verdict = _upgrade_verdict(current_verdict, "REJECT")
        else:
            for ev in evs:
                if not isinstance(ev, dict):
                    additional_issues.append("evidence item must be object")
                    current_verdict = _upgrade_verdict(current_verdict, "REJECT")
                    continue

                cid = ev.get("chunk_id")
                if not _is_nonempty_str(cid):
                    additional_issues.append("evidence missing chunk_id")
                    current_verdict = _upgrade_verdict(current_verdict, "REJECT")
                    continue

                if cid not in valid_chunk_ids:
                    additional_issues.append(f"evidence references unknown chunk_id: {cid}")
                    current_verdict = _upgrade_verdict(current_verdict, "REJECT")

                # page는 명세상 number이지만, 실전에서는 누락/오류가 잦음 → FIXABLE로만 처리
                pg = ev.get("page")
                if pg is None or not isinstance(pg, int):
                    additional_issues.append("evidence missing/invalid page")
                    current_verdict = _upgrade_verdict(current_verdict, "FIXABLE")

                # kind도 있으면 체크(강제는 아님)
                kind = ev.get("kind")
                if kind is not None and kind not in {"text", "table"}:
                    additional_issues.append(f"evidence invalid kind: {kind}")
                    current_verdict = _upgrade_verdict(current_verdict, "FIXABLE")

        # =================
        # 5) MCQ 구조 검증
        # =================
        if qtype == "MCQ":
            ans = q.get("answer")
            if ans not in {"A", "B", "C", "D"}:
                additional_issues.append("MCQ answer must be A/B/C/D")
                current_verdict = _upgrade_verdict(current_verdict, "REJECT")

            choices = q.get("choices")
            if not isinstance(choices, list):
                additional_issues.append("MCQ missing choices")
                current_verdict = _upgrade_verdict(current_verdict, "REJECT")
            elif len(choices) != 4:
                additional_issues.append("MCQ must have exactly 4 choices")
                current_verdict = _upgrade_verdict(current_verdict, "REJECT")
            else:
                # choices는 non-empty string이어야 함
                if any((not isinstance(x, str)) or (not x.strip()) for x in choices):
                    additional_issues.append("MCQ choices must be non-empty strings")
                    current_verdict = _upgrade_verdict(current_verdict, "REJECT")
                else:
                    # choices 중복 체크 (완화: 공백/대소문자 무시)
                    norm = [x.strip().lower() for x in choices]
                    if len(set(norm)) != 4:
                        additional_issues.append("duplicate MCQ choices")
                        current_verdict = _upgrade_verdict(current_verdict, "FIXABLE")

        # =================
        # 6) SAQ 구조 검증
        # =================
        if qtype == "SAQ":
            ans = q.get("answer")
            if not _is_nonempty_str(ans):
                additional_issues.append("SAQ missing answer")
                current_verdict = _upgrade_verdict(current_verdict, "REJECT")

            # SAQ는 choices 없어야 하나, LLM이 choices: []/null을 넣는 경우가 흔함 → 완화
            ch = q.get("choices", None)
            if ch not in (None, [], ""):
                additional_issues.append("SAQ should not include choices")
                current_verdict = _upgrade_verdict(current_verdict, "FIXABLE")

        # =================
        # 7) explanation 검증 (REJECT는 피하고 대부분 FIXABLE)
        # =================
        explanation = q.get("explanation", "")
        expl = str(explanation).strip() if explanation is not None else ""
        if not expl:
            additional_issues.append("missing explanation")
            current_verdict = _upgrade_verdict(current_verdict, "FIXABLE")
        elif len(expl) < MIN_EXPL_LEN:
            additional_issues.append(f"explanation too short (<{MIN_EXPL_LEN} chars)")
            current_verdict = _upgrade_verdict(current_verdict, "FIXABLE")

        # =================
        # 최종 결과 저장
        # =================
        q_out = dict(q)
        q_out["verdict"] = current_verdict
        q_out["issues"] = list(gen_issues) + additional_issues
        q_out["verified_at"] = _now_iso()

        verified.append(q_out)
        summary[current_verdict] += 1

    return {
        "job_id": job.get("job_id"),
        "section_id": job.get("section_id"),
        "verified_at": _now_iso(),
        "summary": summary,
        "questions": verified,
        "stats": {
            "total": len(verified),
            "unique_question_ids": len(seen_question_ids),
            "unique_question_texts": len(seen_question_texts),
            "valid_chunk_ids": len(valid_chunk_ids),
        },
    }
