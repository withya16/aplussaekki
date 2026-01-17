# backend/engine/core/aggregate_verifier.py
"""
Phase 3: Aggregate 검증 (전체 완료 후)
- 총 OK 문제 수 vs 요청 수
- 전체 문제 간 중복 검사
- 부족/중복 시 재생성 대상 식별
- 속도: 즉시 (로컬)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AggregateVerifyResult:
    """Aggregate 검증 결과"""
    target_total: int = 0
    actual_total: int = 0
    actual_ok: int = 0
    deficit: int = 0  # 부족한 문제 수

    duplicate_count: int = 0
    duplicate_groups: List[Dict[str, Any]] = field(default_factory=list)

    # Job별 부족 정보
    job_deficits: Dict[str, int] = field(default_factory=dict)  # {job_id: 부족 개수}

    is_satisfied: bool = False
    issues: List[str] = field(default_factory=list)


@dataclass
class RegenerationTarget:
    """재생성 대상"""
    job_id: str
    section_id: str
    additional_count: int  # 추가로 생성해야 할 문제 수
    reason: str  # "deficit" | "duplicate"
    exclude_texts: List[str] = field(default_factory=list)  # 제외할 문제 텍스트 (중복 방지)


# =============================================================================
# Helpers
# =============================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _normalize_text(text: str) -> str:
    """텍스트 정규화 (중복 비교용)"""
    if not text:
        return ""
    return " ".join(text.lower().split())


# =============================================================================
# Phase 3: Aggregate 검증
# =============================================================================

def verify_aggregate(
    *,
    target_total: int,
    all_questions: List[Dict[str, Any]],
    job_targets: Optional[Dict[str, int]] = None,
) -> AggregateVerifyResult:
    """
    Aggregate 레벨 검증

    Args:
        target_total: 요청된 총 문제 개수
        all_questions: 전체 문제 리스트 (모든 Job의 questions 병합)
        job_targets: {job_id: target_questions} 맵 (Job별 목표)

    Returns:
        AggregateVerifyResult
    """
    result = AggregateVerifyResult(target_total=target_total)

    # 1. Job별 OK 문제 수 집계
    job_ok_counts: Dict[str, int] = {}
    seen_texts: Dict[str, List[Dict[str, Any]]] = {}  # normalized_text -> [question_info]

    for q in all_questions:
        if not isinstance(q, dict):
            continue

        job_id = q.get("job_id", "unknown")
        verdict = q.get("verdict", "OK")

        result.actual_total += 1

        if verdict == "OK":
            result.actual_ok += 1
            job_ok_counts[job_id] = job_ok_counts.get(job_id, 0) + 1

        # 중복 검사를 위해 텍스트 수집 (OK 문제만)
        if verdict == "OK":
            q_text = q.get("question_text") or q.get("question", "")
            norm_text = _normalize_text(q_text)

            if norm_text:
                q_info = {
                    "job_id": job_id,
                    "question_id": q.get("question_id"),
                    "question_text": q_text[:100],
                    "section_id": q.get("section_id"),
                }

                if norm_text not in seen_texts:
                    seen_texts[norm_text] = []
                seen_texts[norm_text].append(q_info)

    # 2. 중복 문제 검출
    for norm_text, q_list in seen_texts.items():
        if len(q_list) > 1:
            result.duplicate_count += len(q_list) - 1  # 첫 번째 제외
            result.duplicate_groups.append({
                "normalized_text": norm_text[:50] + "...",
                "count": len(q_list),
                "questions": q_list,
            })

    # 3. Job별 부족 계산
    if job_targets:
        for job_id, target in job_targets.items():
            actual = job_ok_counts.get(job_id, 0)
            if actual < target:
                result.job_deficits[job_id] = target - actual

    # 4. 총 부족 계산
    result.deficit = max(0, target_total - result.actual_ok)

    # 5. 이슈 목록 생성
    if result.deficit > 0:
        result.issues.append(
            f"문제 부족: {result.actual_ok}/{target_total} (부족: {result.deficit}개)"
        )

    if result.duplicate_count > 0:
        result.issues.append(f"중복 문제: {result.duplicate_count}개")

    if result.job_deficits:
        deficit_items = [f"{jid}(-{cnt})" for jid, cnt in result.job_deficits.items()]
        result.issues.append(f"Job별 부족: {', '.join(deficit_items)}")

    # 6. 만족 여부
    result.is_satisfied = (result.deficit == 0 and result.duplicate_count == 0)

    return result


# =============================================================================
# 재생성 대상 식별
# =============================================================================

def identify_regeneration_targets(
    verify_result: AggregateVerifyResult,
    jobs: List[Dict[str, Any]],
    all_questions: List[Dict[str, Any]],
) -> List[RegenerationTarget]:
    """
    재생성이 필요한 Job과 필요 개수 식별

    Args:
        verify_result: Aggregate 검증 결과
        jobs: 원본 Job 리스트
        all_questions: 전체 문제 리스트

    Returns:
        List[RegenerationTarget]
    """
    targets: List[RegenerationTarget] = []

    # Job ID -> Job 맵 생성
    job_map = {
        j.get("job_id"): j
        for j in jobs
        if isinstance(j, dict) and j.get("job_id")
    }

    # 기존 문제 텍스트 수집 (중복 방지용)
    existing_texts = get_existing_question_texts(all_questions)

    # 1. Job별 부족분 처리
    processed_jobs: Set[str] = set()

    for job_id, deficit in verify_result.job_deficits.items():
        if job_id in job_map and deficit > 0:
            job = job_map[job_id]
            targets.append(RegenerationTarget(
                job_id=job_id,
                section_id=job.get("section_id", ""),
                additional_count=deficit,
                reason="deficit",
                exclude_texts=list(existing_texts),
            ))
            processed_jobs.add(job_id)

    # 2. 중복 문제가 있는 Job도 재생성 대상
    for dup_group in verify_result.duplicate_groups:
        dup_questions = dup_group.get("questions", [])

        # 첫 번째를 제외한 나머지의 job_id 수집
        for q_info in dup_questions[1:]:  # 첫 번째 제외
            job_id = q_info.get("job_id")

            if job_id and job_id in job_map:
                # 이미 있는 타겟에 추가
                existing_target = next(
                    (t for t in targets if t.job_id == job_id),
                    None
                )

                if existing_target:
                    existing_target.additional_count += 1
                    if existing_target.reason == "deficit":
                        existing_target.reason = "deficit+duplicate"
                else:
                    job = job_map[job_id]
                    targets.append(RegenerationTarget(
                        job_id=job_id,
                        section_id=job.get("section_id", ""),
                        additional_count=1,
                        reason="duplicate",
                        exclude_texts=list(existing_texts),
                    ))

    return targets


def get_existing_question_texts(questions: List[Dict[str, Any]]) -> Set[str]:
    """
    기존 생성된 문제 텍스트들 수집 (재생성 시 제외용)
    """
    texts: Set[str] = set()

    for q in questions:
        if not isinstance(q, dict):
            continue

        # OK인 문제만 수집
        if q.get("verdict") != "OK":
            continue

        q_text = q.get("question_text") or q.get("question", "")
        norm_text = _normalize_text(q_text)
        if norm_text:
            texts.add(norm_text)

    return texts


# =============================================================================
# 결과 저장/로드
# =============================================================================

def save_aggregate_result(
    result: AggregateVerifyResult,
    out_path: Path,
) -> None:
    """Aggregate 검증 결과 저장"""
    payload = {
        "verified_at": _now_iso(),
        "target_total": result.target_total,
        "actual_total": result.actual_total,
        "actual_ok": result.actual_ok,
        "deficit": result.deficit,
        "duplicate_count": result.duplicate_count,
        "duplicate_groups": result.duplicate_groups,
        "job_deficits": result.job_deficits,
        "is_satisfied": result.is_satisfied,
        "issues": result.issues,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_aggregate_result(path: Path) -> Optional[AggregateVerifyResult]:
    """Aggregate 검증 결과 로드"""
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))

        result = AggregateVerifyResult(
            target_total=data.get("target_total", 0),
            actual_total=data.get("actual_total", 0),
            actual_ok=data.get("actual_ok", 0),
            deficit=data.get("deficit", 0),
            duplicate_count=data.get("duplicate_count", 0),
            duplicate_groups=data.get("duplicate_groups", []),
            job_deficits=data.get("job_deficits", {}),
            is_satisfied=data.get("is_satisfied", False),
            issues=data.get("issues", []),
        )
        return result

    except Exception:
        return None


# =============================================================================
# 중복 문제 제거
# =============================================================================

def remove_duplicates(
    questions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    중복 문제 제거

    Returns:
        (유지할 문제 리스트, 제거된 문제 리스트)
    """
    seen_texts: Set[str] = set()
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for q in questions:
        if not isinstance(q, dict):
            continue

        q_text = q.get("question_text") or q.get("question", "")
        norm_text = _normalize_text(q_text)

        if norm_text in seen_texts:
            q_out = dict(q)
            q_out["removed_reason"] = "duplicate"
            removed.append(q_out)
        else:
            kept.append(q)
            if norm_text:
                seen_texts.add(norm_text)

    return kept, removed
