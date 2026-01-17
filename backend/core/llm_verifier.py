# backend/engine/core/llm_verifier.py
"""
Phase 2: LLM Batch 품질 검증
- 문제-정답 일관성
- 해설 품질
- 오답 선지 합리성
- 속도: ~10개/1 API call (2-3초)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.llm_text import call_llm_text


# =============================================================================
# Config
# =============================================================================

@dataclass
class LLMVerifyConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    batch_size: int = 10  # 한 번에 검증할 문제 수
    max_retries: int = 2


# =============================================================================
# Helpers
# =============================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """JSON 추출 (markdown 블록 제거)"""
    if not isinstance(raw, str) or not raw.strip():
        return None
    raw = raw.strip()

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
        try:
            obj = json.loads(raw[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


# =============================================================================
# LLM Batch Verification
# =============================================================================

def _build_verify_prompt(questions: List[Dict[str, Any]]) -> str:
    """검증용 프롬프트 생성"""
    q_items = []
    for i, q in enumerate(questions, 1):
        qtype = q.get("type", "?")
        q_text = q.get("question_text") or q.get("question", "")
        answer = q.get("answer") or q.get("correct_answer", "")
        explanation = q.get("explanation", "")
        choices = q.get("choices") or q.get("options", [])

        q_preview = q_text[:300] + ("..." if len(q_text) > 300 else "")
        expl_preview = explanation[:200] + ("..." if len(explanation) > 200 else "")

        item = f"""
[문제 {i}]
- ID: {q.get("question_id", "?")}
- 유형: {qtype}
- 난이도: {q.get("difficulty", "?")}
- 문제: {q_preview}
"""
        if qtype == "MCQ" and choices:
            item += f"- 선지: {choices}\n"
        item += f"- 정답: {answer}\n"
        item += f"- 해설: {expl_preview}\n"

        q_items.append(item)

    questions_text = "\n".join(q_items)

    return f"""당신은 교육 전문가입니다. 다음 시험 문제들의 품질을 검증하세요.

{"="*60}
검증 대상 문제들
{"="*60}
{questions_text}

{"="*60}
검증 기준
{"="*60}
1. **정답 정확성**: 제시된 정답이 실제로 맞는가?
2. **해설 일관성**: 해설이 정답을 올바르게 설명하는가?
3. **문제 명확성**: 문제가 모호하지 않고 명확한가?
4. **MCQ 선지 품질**: 오답 선지가 합리적인가? (너무 쉽게 배제 가능하지 않은가?)
5. **복수 정답 가능성**: 복수 정답이 가능한 문제인가?

{"="*60}
출력 형식 (JSON ONLY)
{"="*60}
{{
  "results": [
    {{
      "question_id": "문제ID",
      "verdict": "OK" | "FIXABLE" | "REJECT",
      "issues": ["이슈1", "이슈2"],
      "confidence": 0.95
    }}
  ],
  "summary": {{
    "total": 10,
    "ok": 8,
    "fixable": 1,
    "reject": 1
  }}
}}

검증 결과를 JSON으로만 출력하세요.
"""


def verify_questions_llm(
    questions: List[Dict[str, Any]],
    config: Optional[LLMVerifyConfig] = None,
) -> Dict[str, Any]:
    """
    LLM을 사용한 문제 품질 검증

    Args:
        questions: 검증할 문제 리스트
        config: LLM 검증 설정

    Returns:
        {
            "verified_at": str,
            "model": str,
            "summary": {"OK": int, "FIXABLE": int, "REJECT": int},
            "results": List[{question_id, verdict, issues, confidence}],
            "questions": List[Question with llm_verdict]
        }
    """
    if config is None:
        config = LLMVerifyConfig()

    if not questions:
        return {
            "verified_at": _now_iso(),
            "model": config.model,
            "summary": {"OK": 0, "FIXABLE": 0, "REJECT": 0},
            "results": [],
            "questions": [],
        }

    # 배치 단위로 처리
    all_results: List[Dict[str, Any]] = []
    questions_with_verdict: List[Dict[str, Any]] = []

    for batch_start in range(0, len(questions), config.batch_size):
        batch = questions[batch_start:batch_start + config.batch_size]
        batch_results = _verify_batch(batch, config)
        all_results.extend(batch_results)

    # 결과를 question_id로 매핑
    result_map = {r["question_id"]: r for r in all_results}

    summary = {"OK": 0, "FIXABLE": 0, "REJECT": 0}

    for q in questions:
        qid = q.get("question_id", "unknown")
        q_out = dict(q)

        if qid in result_map:
            llm_result = result_map[qid]
            q_out["llm_verdict"] = llm_result["verdict"]
            q_out["llm_issues"] = llm_result.get("issues", [])
            q_out["llm_confidence"] = llm_result.get("confidence", 0.0)
            summary[llm_result["verdict"]] += 1
        else:
            # LLM 결과 없으면 기본값
            q_out["llm_verdict"] = "OK"
            q_out["llm_issues"] = []
            q_out["llm_confidence"] = 0.0
            summary["OK"] += 1

        questions_with_verdict.append(q_out)

    return {
        "verified_at": _now_iso(),
        "model": config.model,
        "summary": summary,
        "results": all_results,
        "questions": questions_with_verdict,
    }


def _verify_batch(
    batch: List[Dict[str, Any]],
    config: LLMVerifyConfig,
) -> List[Dict[str, Any]]:
    """단일 배치 검증"""
    prompt = _build_verify_prompt(batch)

    for attempt in range(config.max_retries):
        try:
            raw = call_llm_text(
                prompt=prompt,
                model=config.model,
                temperature=config.temperature,
            )

            data = _extract_json(raw)
            if data and "results" in data:
                results = data["results"]
                if isinstance(results, list):
                    return _normalize_results(results, batch)

        except Exception as e:
            print(f"⚠️ LLM 검증 실패 (attempt {attempt + 1}): {e}")
            continue

    # 실패 시 기본값 반환
    return [
        {
            "question_id": q.get("question_id", "unknown"),
            "verdict": "OK",  # LLM 실패 시 구조 검증 결과 신뢰
            "issues": ["LLM verification failed"],
            "confidence": 0.0,
        }
        for q in batch
    ]


def _normalize_results(
    results: List[Dict[str, Any]],
    batch: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """LLM 결과 정규화"""
    normalized: List[Dict[str, Any]] = []
    result_map = {}

    for r in results:
        if isinstance(r, dict) and "question_id" in r:
            result_map[r["question_id"]] = r

    for q in batch:
        qid = q.get("question_id", "unknown")

        if qid in result_map:
            r = result_map[qid]
            verdict = r.get("verdict", "OK")
            if verdict not in {"OK", "FIXABLE", "REJECT"}:
                verdict = "OK"

            normalized.append({
                "question_id": qid,
                "verdict": verdict,
                "issues": r.get("issues", []) if isinstance(r.get("issues"), list) else [],
                "confidence": float(r.get("confidence", 0.9)) if r.get("confidence") else 0.9,
            })
        else:
            normalized.append({
                "question_id": qid,
                "verdict": "OK",
                "issues": [],
                "confidence": 0.5,
            })

    return normalized


# =============================================================================
# 최종 verdict 결합 (Phase 1 + Phase 2)
# =============================================================================

def combine_verdicts(
    structure_verdict: str,
    llm_verdict: str,
) -> str:
    """
    구조 검증 + LLM 검증 결과 결합

    우선순위: REJECT > FIXABLE > OK
    """
    priority = {"OK": 0, "FIXABLE": 1, "REJECT": 2}
    s = structure_verdict if structure_verdict in priority else "OK"
    l = llm_verdict if llm_verdict in priority else "OK"

    return s if priority[s] >= priority[l] else l


def merge_verification_results(
    questions_with_structure: List[Dict[str, Any]],
    questions_with_llm: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    구조 검증 결과와 LLM 검증 결과 병합

    Returns:
        최종 verdict가 포함된 문제 리스트
    """
    # question_id로 LLM 결과 매핑
    llm_map = {
        q.get("question_id"): q
        for q in questions_with_llm
        if isinstance(q, dict)
    }

    merged: List[Dict[str, Any]] = []

    for q in questions_with_structure:
        qid = q.get("question_id")
        q_out = dict(q)

        llm_q = llm_map.get(qid, {})

        structure_verdict = q.get("verdict", "OK")
        llm_verdict = llm_q.get("llm_verdict", "OK")

        # 최종 verdict 결정
        final_verdict = combine_verdicts(structure_verdict, llm_verdict)

        # 모든 이슈 병합
        all_issues = list(q.get("issues", []))
        all_issues.extend(llm_q.get("llm_issues", []))

        q_out["verdict"] = final_verdict
        q_out["structure_verdict"] = structure_verdict
        q_out["llm_verdict"] = llm_verdict
        q_out["llm_confidence"] = llm_q.get("llm_confidence", 0.0)
        q_out["issues"] = all_issues
        q_out["verified_at"] = _now_iso()

        merged.append(q_out)

    return merged
