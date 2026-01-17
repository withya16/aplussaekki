# backend/engine/core/verification_pipeline.py
"""
통합 검증 파이프라인
- Phase 1: 로컬 구조 검증
- Phase 2: LLM Batch 품질 검증
- Phase 3: Aggregate 검증
- 자동 재생성 루프
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from backend.engine.core.question_verifier import (
    verify_questions_batch,
    StructureVerifyConfig,
    Verdict,
)
from backend.engine.core.llm_verifier import (
    verify_questions_llm,
    merge_verification_results,
    LLMVerifyConfig,
)
from backend.engine.core.aggregate_verifier import (
    verify_aggregate,
    identify_regeneration_targets,
    remove_duplicates,
    save_aggregate_result,
    AggregateVerifyResult,
    RegenerationTarget,
)


# =============================================================================
# Config
# =============================================================================

@dataclass
class VerificationPipelineConfig:
    """통합 검증 파이프라인 설정"""
    # Phase 1: 구조 검증
    structure_config: Optional[StructureVerifyConfig] = None

    # Phase 2: LLM 검증
    enable_llm_verify: bool = True
    llm_config: Optional[LLMVerifyConfig] = None

    # Phase 3: Aggregate 검증
    target_total: int = 10

    # 재생성 설정
    max_regeneration_rounds: int = 3
    max_retries_per_job: int = 2

    # 출력 경로
    out_dir: Optional[Path] = None


# =============================================================================
# Helpers
# =============================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


# =============================================================================
# 단일 Job 검증 (Phase 1 + Phase 2)
# =============================================================================

def verify_job_questions(
    questions: List[Dict[str, Any]],
    config: VerificationPipelineConfig,
) -> Dict[str, Any]:
    """
    단일 Job의 문제들을 검증 (Phase 1 + Phase 2)

    Returns:
        {
            "verified_at": str,
            "phase1_summary": {...},
            "phase2_summary": {...},
            "final_summary": {"OK": int, "FIXABLE": int, "REJECT": int},
            "questions": List[Question with verdict]
        }
    """
    # Phase 1: 구조 검증
    phase1_result = verify_questions_batch(
        questions,
        config.structure_config
    )

    questions_after_phase1 = phase1_result["questions"]

    # Phase 2: LLM 검증 (선택적)
    if config.enable_llm_verify:
        # OK인 문제만 LLM 검증 (비용 절감)
        ok_questions = [
            q for q in questions_after_phase1
            if q.get("verdict") == "OK"
        ]

        if ok_questions:
            phase2_result = verify_questions_llm(
                ok_questions,
                config.llm_config
            )

            # Phase 1 + Phase 2 결과 병합
            final_questions = merge_verification_results(
                questions_after_phase1,
                phase2_result["questions"]
            )
            phase2_summary = phase2_result["summary"]
        else:
            final_questions = questions_after_phase1
            phase2_summary = {"OK": 0, "FIXABLE": 0, "REJECT": 0}
    else:
        final_questions = questions_after_phase1
        phase2_summary = {"OK": 0, "FIXABLE": 0, "REJECT": 0, "skipped": True}

    # 최종 summary 계산
    final_summary = {"OK": 0, "FIXABLE": 0, "REJECT": 0}
    for q in final_questions:
        v = q.get("verdict", "OK")
        if v in final_summary:
            final_summary[v] += 1

    return {
        "verified_at": _now_iso(),
        "phase1_summary": phase1_result["summary"],
        "phase2_summary": phase2_summary,
        "final_summary": final_summary,
        "questions": final_questions,
    }


# =============================================================================
# 전체 파이프라인 (Phase 1 + 2 + 3 + 재생성)
# =============================================================================

def run_verification_pipeline(
    *,
    all_questions: List[Dict[str, Any]],
    jobs: List[Dict[str, Any]],
    job_targets: Dict[str, int],
    config: VerificationPipelineConfig,
    regenerate_fn: Optional[Callable[[RegenerationTarget], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    전체 검증 파이프라인 실행

    Args:
        all_questions: 전체 문제 리스트
        jobs: 원본 Job 리스트
        job_targets: {job_id: target_questions}
        config: 파이프라인 설정
        regenerate_fn: 재생성 함수 (target -> new_questions)

    Returns:
        {
            "verified_at": str,
            "target_total": int,
            "actual_ok": int,
            "is_satisfied": bool,
            "regeneration_rounds": int,
            "questions": List[Question],
            "aggregate_result": AggregateVerifyResult,
            "removed_duplicates": List[Question]
        }
    """
    current_questions = list(all_questions)
    total_regeneration_rounds = 0
    all_removed: List[Dict[str, Any]] = []

    for round_num in range(config.max_regeneration_rounds + 1):
        print(f"\n{'='*60}")
        print(f"검증 라운드 {round_num + 1}")
        print(f"{'='*60}")

        # Phase 1 + 2: 각 문제 검증
        print("▶ Phase 1+2: 구조 및 품질 검증...")
        verified_result = verify_job_questions(current_questions, config)
        current_questions = verified_result["questions"]

        print(f"  구조 검증: {verified_result['phase1_summary']}")
        if config.enable_llm_verify:
            print(f"  LLM 검증: {verified_result['phase2_summary']}")
        print(f"  최종: {verified_result['final_summary']}")

        # Phase 3: Aggregate 검증
        print("▶ Phase 3: Aggregate 검증...")
        aggregate_result = verify_aggregate(
            target_total=config.target_total,
            all_questions=current_questions,
            job_targets=job_targets,
        )

        print(f"  목표: {aggregate_result.target_total}개")
        print(f"  실제 OK: {aggregate_result.actual_ok}개")
        print(f"  부족: {aggregate_result.deficit}개")
        print(f"  중복: {aggregate_result.duplicate_count}개")

        # 만족하면 종료
        if aggregate_result.is_satisfied:
            print("✅ 검증 통과!")
            break

        # 마지막 라운드면 종료
        if round_num >= config.max_regeneration_rounds:
            print(f"⚠️ 최대 재생성 라운드 도달 ({config.max_regeneration_rounds})")
            break

        # 재생성 함수가 없으면 종료
        if regenerate_fn is None:
            print("⚠️ 재생성 함수 없음, 검증 종료")
            break

        # 중복 제거
        if aggregate_result.duplicate_count > 0:
            print(f"▶ 중복 제거 중...")
            current_questions, removed = remove_duplicates(current_questions)
            all_removed.extend(removed)
            print(f"  제거됨: {len(removed)}개")

        # 재생성 대상 식별
        targets = identify_regeneration_targets(
            aggregate_result,
            jobs,
            current_questions,
        )

        if not targets:
            print("⚠️ 재생성 대상 없음")
            break

        # 재생성 실행
        print(f"▶ 재생성 중... ({len(targets)}개 Job)")
        for target in targets:
            print(f"  - {target.job_id}: +{target.additional_count}개 ({target.reason})")

            try:
                new_questions = regenerate_fn(target)
                if new_questions:
                    # 새 문제에 job_id 추가
                    for q in new_questions:
                        q["job_id"] = target.job_id
                        q["section_id"] = target.section_id
                        q["regenerated"] = True
                        q["regeneration_round"] = round_num + 1

                    current_questions.extend(new_questions)
                    print(f"    → {len(new_questions)}개 생성됨")
            except Exception as e:
                print(f"    → 재생성 실패: {e}")

        total_regeneration_rounds += 1

    # 최종 결과 저장
    if config.out_dir:
        save_aggregate_result(
            aggregate_result,
            config.out_dir / "aggregate_verify_result.json"
        )

    return {
        "verified_at": _now_iso(),
        "target_total": config.target_total,
        "actual_ok": aggregate_result.actual_ok,
        "is_satisfied": aggregate_result.is_satisfied,
        "regeneration_rounds": total_regeneration_rounds,
        "questions": current_questions,
        "aggregate_result": aggregate_result,
        "removed_duplicates": all_removed,
        "final_summary": {
            "total": len(current_questions),
            "ok": sum(1 for q in current_questions if q.get("verdict") == "OK"),
            "fixable": sum(1 for q in current_questions if q.get("verdict") == "FIXABLE"),
            "reject": sum(1 for q in current_questions if q.get("verdict") == "REJECT"),
        }
    }


# =============================================================================
# 간편 API
# =============================================================================

def verify_and_filter_ok(
    questions: List[Dict[str, Any]],
    enable_llm: bool = True,
) -> List[Dict[str, Any]]:
    """
    문제 검증 후 OK인 것만 반환

    Args:
        questions: 검증할 문제 리스트
        enable_llm: LLM 검증 사용 여부

    Returns:
        OK verdict인 문제만 포함된 리스트
    """
    config = VerificationPipelineConfig(
        enable_llm_verify=enable_llm,
    )

    result = verify_job_questions(questions, config)

    return [
        q for q in result["questions"]
        if q.get("verdict") == "OK"
    ]


def get_rejection_reasons(
    questions: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    문제별 거절 사유 반환

    Returns:
        {question_id: [issues]}
    """
    return {
        q.get("question_id", "unknown"): q.get("issues", [])
        for q in questions
        if q.get("verdict") in ("FIXABLE", "REJECT")
    }
