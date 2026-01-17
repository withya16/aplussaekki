# core/run_question_pipeline.py
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import json
import logging
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Callable, Tuple

from core.question_generator import QuestionGenConfig, generate_questions_for_job
from core.question_verifier import verify_questions_for_job, verify_questions_batch
from core.llm_verifier import verify_questions_llm, merge_verification_results, LLMVerifyConfig
from core.aggregate_verifier import (
    verify_aggregate,
    identify_regeneration_targets,
    remove_duplicates,
    save_aggregate_result,
    AggregateVerifyResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_log_lock = Lock()


# =============================================================================
# Spec ENUM (API 명세 맞춤)
# =============================================================================

JOB_STATUS_SPEC = {"QUEUED", "RUNNING", "DONE", "FAILED"}
PROGRESS_STAGE_SPEC = {"PARSING", "GENERATING", "VERIFYING", "SAVING"}
DETAIL_STAGE_SPEC = {"TEXT_EXTRACT", "PAGE_CLASSIFY", "TABLE_EXTRACT_MM"}  # optional


# =============================================================================
# Runner-local Status (기존 로직 유지용)
# =============================================================================

class JobStatusLocal:
    QUEUED = "QUEUED"
    GENERATING = "GENERATING"
    VERIFYING = "VERIFYING"
    DONE = "DONE"
    FAILED = "FAILED"


# =============================================================================
# IO Helpers
# =============================================================================

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Windows BOM 대응: utf-8-sig로 열면 BOM 자동 제거"""
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, ln in enumerate(f, start=1):
            ln = (ln or "").strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    items.append(obj)
                else:
                    logger.warning(f"Invalid JSON object (not dict) at line {line_no}")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON at line {line_no}: {e}")
                continue
    return items


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _resolve_jobs_path(out_dir: Path, jobs_jsonl: str) -> Path:
    """
    --jobs가
    - "question_jobs.jsonl" 처럼 파일명만 오면 out_dir 아래에서 찾고,
    - "artifacts/lecture/question_jobs.jsonl" 처럼 경로가 포함돼 있으면 cwd 기준으로 사용,
    - 절대경로면 그대로 사용.
    """
    jp = Path(jobs_jsonl)
    if jp.is_absolute():
        return jp
    if len(jp.parts) > 1:
        return jp
    return out_dir / jp


# =============================================================================
# Retry
# =============================================================================

def _retry(fn: Callable[[int], Any], max_retries: int = 5, base_delay: float = 1.0):
    last: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(attempt)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
                time.sleep(min(delay, 20))
    raise last if last else RuntimeError("retry failed")


# =============================================================================
# Preview
# =============================================================================

def _preview_block(job_id: str, section_id: str, questions: List[Dict[str, Any]], k: int) -> str:
    if k <= 0:
        return ""
    lines: List[str] = []
    lines.append("=" * 90)
    lines.append(f"[PREVIEW] job={job_id} section={section_id} ({min(k,len(questions))}/{len(questions)})")
    lines.append("-" * 90)
    for i, q in enumerate(questions[:k], start=1):
        qt = (q.get("question") or "").replace("\n", " ")
        if len(qt) > 200:
            qt = qt[:200] + "..."
        qtype = q.get("type", "?")
        diff = q.get("difficulty", "?")
        verdict = q.get("verdict", "?")
        lines.append(f"{i}) [{qtype}/{diff}/{verdict}] {qt}")
        if "answer" in q:
            lines.append(f"   ans: {q['answer']}")
    lines.append("=" * 90)
    return "\n".join(lines)


def _error_preview_block(job_id: str, section_id: str, error: str) -> str:
    return "\n".join([
        "=" * 90,
        f"[PREVIEW] job={job_id} section={section_id}  ERROR",
        "-" * 90,
        error,
        "=" * 90,
    ])


# =============================================================================
# Model Selection
# =============================================================================

def _has_tables(job: Dict[str, Any]) -> bool:
    constraints = job.get("constraints", {})
    if isinstance(constraints, dict) and constraints.get("has_tables_in_job"):
        return True
    tins = job.get("tables_in_job")
    if isinstance(tins, list) and len(tins) > 0:
        return True
    tbls = job.get("tables")
    if isinstance(tbls, list) and len(tbls) > 0:
        return True
    return False


def _model_plan_for_job(
    job: Dict[str, Any],
    *,
    default_model: str,
    table_model: str,
    fallback_model: str,
) -> List[str]:
    """
    model 시도 순서 반환.
    - 표 job: [table_model, fallback_model] (중복 제거)
    - 텍스트 only: [default_model]
    - job에 model이 명시돼 있으면 맨 앞에 끼워 넣되 중복 제거
    """
    plan: List[str] = []
    job_model = job.get("model")
    if isinstance(job_model, str) and job_model.strip():
        plan.append(job_model.strip())

    if _has_tables(job):
        plan.append(table_model)
        plan.append(fallback_model)
    else:
        plan.append(default_model)

    seen: Set[str] = set()
    out: List[str] = []
    for m in plan:
        m = (m or "").strip()
        if not m or m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def _normalize_job_tables(job: Dict[str, Any]) -> None:
    """
    runner 단계에서 키가 섞이는 걸 통일:
    - tables가 없고 tables_in_job만 있으면 tables로 복사
    - 반대로 tables만 있고 tables_in_job 없으면 tables_in_job으로도 복사
    """
    tins = job.get("tables_in_job")
    tbls = job.get("tables")

    if (not isinstance(tbls, list) or len(tbls) == 0) and isinstance(tins, list) and len(tins) > 0:
        job["tables"] = tins

    if (not isinstance(tins, list) or len(tins) == 0) and isinstance(tbls, list) and len(tbls) > 0:
        job["tables_in_job"] = tbls


# =============================================================================
# tables format normalizer (runner-side)
# =============================================================================

def _normalize_tables_format(tables: Any) -> List[Dict[str, Any]]:
    """
    다양한 형태로 들어오는 tables를 generator가 쓰기 좋은 표준 형태로 정리.
    표준 형태:
      {
        "headers": [...],
        "rows": [[...], ...],
        "caption": str | None,
        "page": int | None,
        "source": str | None
      }
    """
    if not isinstance(tables, list):
        return []

    out: List[Dict[str, Any]] = []
    for t in tables:
        if not isinstance(t, dict):
            continue

        headers = t.get("headers")
        if headers is None:
            headers = t.get("header")
        if headers is None:
            headers = t.get("columns")
        if headers is None:
            headers = t.get("cols")

        rows = t.get("rows")
        if rows is None:
            rows = t.get("data")
        if rows is None:
            rows = t.get("values")
        if rows is None and isinstance(t.get("cells"), list):
            rows = t.get("cells")

        if not isinstance(headers, list):
            headers = []
        headers = [str(x) for x in headers]

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

        caption = t.get("caption") if isinstance(t.get("caption"), str) else None

        page = t.get("page")
        if not isinstance(page, int):
            page = t.get("page_index")
        if not isinstance(page, int):
            page = t.get("page_number")
        if not isinstance(page, int):
            page = None

        source = t.get("source") if isinstance(t.get("source"), str) else None

        out.append({
            "headers": headers,
            "rows": norm_rows,
            "caption": caption,
            "page": page,
            "source": source,
        })

    return out


# =============================================================================
# Error type tagging (for retry policy)
# =============================================================================

class ErrorType:
    GENERATOR = "GENERATOR_ERROR"
    VERIFIER = "VERIFIER_ERROR"
    IO = "IO_ERROR"
    UNKNOWN = "UNKNOWN_ERROR"


# =============================================================================
# NEW: API-spec job state writer (data/jobs/{job_id}.json)
# =============================================================================

def _spec_status_and_progress(local_status: str, progress_value: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
    """
    runner 내부 상태를 API 명세 상태/진행률로 매핑.
    - 명세 status: QUEUED/RUNNING/DONE/FAILED
    - 명세 progress.stage: PARSING/GENERATING/VERIFYING/SAVING
    - 명세 progress.progress: 0.0 ~ 1.0 (정량적 진행률)
    """
    if local_status == JobStatusLocal.QUEUED:
        return "QUEUED", {"stage": "PARSING", "progress": progress_value or 0.0}
    if local_status == JobStatusLocal.GENERATING:
        return "RUNNING", {"stage": "GENERATING", "progress": progress_value or 0.3}
    if local_status == JobStatusLocal.VERIFYING:
        return "RUNNING", {"stage": "VERIFYING", "progress": progress_value or 0.7}
    if local_status == JobStatusLocal.DONE:
        return "DONE", {"stage": "SAVING", "progress": 1.0}
    if local_status == JobStatusLocal.FAILED:
        return "FAILED", {"stage": "SAVING", "progress": progress_value or 0.0}
    # fallback
    return "RUNNING", {"stage": "GENERATING", "progress": progress_value or 0.5}


def _write_job_state(
    *,
    data_dir: Path,
    pdf_id: str,
    job_id: str,
    section_id: Optional[str],
    local_status: str,
    model: Optional[str] = None,
    attempts: Optional[int] = None,
    model_plan: Optional[List[str]] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    detail_stage: Optional[str] = None,
    progress_value: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    명세 기반 Job 상태 파일 저장:
      data/jobs/{job_id}.json
    프론트의 GET /jobs/{job_id} 구현 시 이 파일을 그대로 읽어 반환 가능하도록 설계.
    """
    status, progress = _spec_status_and_progress(local_status, progress_value)

    if detail_stage:
        # optional
        progress["detail_stage"] = detail_stage

    now = _now_iso()
    path = data_dir / "jobs" / f"{job_id}.json"
    prev = _load_json_safe(path) or {}

    created_at = prev.get("created_at") if isinstance(prev.get("created_at"), str) else now

    payload: Dict[str, Any] = {
        "job_id": job_id,
        "pdf_id": pdf_id,
        "section_id": section_id,
        "status": status,
        "progress": progress,
        "created_at": created_at,
        "updated_at": now,
    }

    if model is not None:
        payload["model"] = model
    if attempts is not None:
        payload["attempts"] = int(attempts)
    if model_plan is not None:
        payload["model_plan"] = model_plan

    if error_code:
        payload["error"] = {
            "error": error_code,
            "message": error_message or error_code,
        }

    if extra and isinstance(extra, dict):
        payload.update(extra)

    _atomic_write_json(path, payload)


# =============================================================================
# NEW: results writer (data/results/{pdf_id}.questions.json)
# =============================================================================

def _flatten_and_write_results(
    *,
    data_dir: Path,
    pdf_id: str,
    job_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    questions_verified/job_*.json(또는 그 로드 결과)을 기반으로
    data/results/{pdf_id}.questions.json 을 생성.

    NOTE:
    - job별 question_id는 Q001, Q002처럼 중복될 가능성이 높아서,
      API의 /questions/{question_id}/grade 같은 라우팅을 고려하면 전역 unique가 안전함.
      따라서 여기서 question_id를 "{job_id}_{original}" 로 재작성하고
      original_question_id를 남긴다.
    """
    all_questions: List[Dict[str, Any]] = []
    counts = {"OK": 0, "FIXABLE": 0, "REJECT": 0}

    for it in job_items:
        if not isinstance(it, dict):
            continue
        if it.get("status") != JobStatusLocal.DONE:
            continue

        job_id = it.get("job_id")
        section_id = it.get("section_id")
        qs = it.get("questions", [])
        if not isinstance(job_id, str) or not isinstance(qs, list):
            continue

        for q in qs:
            if not isinstance(q, dict):
                continue
            q2 = dict(q)

            orig_qid = q2.get("question_id")
            if isinstance(orig_qid, str) and orig_qid.strip():
                q2["original_question_id"] = orig_qid
                q2["question_id"] = f"{job_id}_{orig_qid}"
            else:
                # 혹시 누락되면 그래도 유니크하게
                q2["original_question_id"] = orig_qid
                q2["question_id"] = f"{job_id}_QXXX"

            q2["job_id"] = job_id
            q2["section_id"] = section_id
            all_questions.append(q2)

            v = q2.get("verdict")
            if v in counts:
                counts[v] += 1

    payload = {
        "pdf_id": pdf_id,
        "updated_at": _now_iso(),
        "summary": {
            "total": len(all_questions),
            "verdict_ok": counts["OK"],
            "verdict_fixable": counts["FIXABLE"],
            "verdict_reject": counts["REJECT"],
        },
        "questions": all_questions,
    }

    out_path = data_dir / "results" / f"{pdf_id}.questions.json"
    _atomic_write_json(out_path, payload)
    return payload


# =============================================================================
# Pipeline Runner
# =============================================================================

def run(
    *,
    out_dir: Path,
    jobs_jsonl: str,
    pdf_id: str,
    max_workers: int,
    max_retries: int,
    overwrite: bool,
    retry_errors: bool,
    default_model: str,
    table_model: str,
    fallback_model: str,
    temperature: float,
    preview: int,
    ordered_preview: bool,
    preview_log: Optional[Path],
    save_answers_only: bool,
    save_generated: bool = False,
    # NEW
    data_dir: Path = Path("data"),
    # 검증 파이프라인 옵션
    enable_llm_verify: bool = True,
    target_total: int = 0,  # 0이면 job_targets 합계 사용
    max_regeneration_rounds: int = 2,
) -> Dict[str, Any]:

    jobs_path = _resolve_jobs_path(out_dir, jobs_jsonl)
    if not jobs_path.exists():
        raise FileNotFoundError(f"jobs file not found: {jobs_path}")

    jobs = _read_jsonl(jobs_path)
    if not jobs:
        raise ValueError(f"No jobs found: {jobs_path}")

    verified_dir = out_dir / "questions_verified"
    verified_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = out_dir / "questions_generated"
    if save_generated:
        generated_dir.mkdir(parents=True, exist_ok=True)

    answers_dir = out_dir / "answers_only"
    if save_answers_only:
        answers_dir.mkdir(parents=True, exist_ok=True)

    def verified_out(jid: str) -> Path:
        return verified_dir / f"job_{jid}.json"

    def generated_out(jid: str) -> Path:
        return generated_dir / f"job_{jid}.json"

    def answers_out(jid: str) -> Path:
        return answers_dir / f"job_{jid}.json"

    # 기존 결과 스캔(verified 기준)
    existing: Dict[str, Dict[str, Any]] = {}
    for f in verified_dir.glob("job_*.json"):
        obj = _load_json_safe(f)
        if obj and isinstance(obj.get("job_id"), str):
            existing[obj["job_id"]] = obj

    def should_run(job: Dict[str, Any]) -> bool:
        jid = job.get("job_id")
        if not isinstance(jid, str) or not jid:
            return False

        if overwrite:
            return True
        if jid not in existing:
            return True
        if retry_errors and existing[jid].get("status") == JobStatusLocal.FAILED:
            et = existing[jid].get("error_type")
            if et in (ErrorType.GENERATOR, ErrorType.UNKNOWN, None):
                return True
        return False

    todo = [j for j in jobs if should_run(j)]
    all_ids = [j.get("job_id") for j in jobs if isinstance(j.get("job_id"), str)]
    todo_ids = {j["job_id"] for j in todo if isinstance(j.get("job_id"), str)}

    logger.info(
        f"jobs_total={len(jobs)} todo={len(todo)} workers={max_workers} "
        f"ordered_preview={ordered_preview} save_generated={save_generated} "
        f"default_model={default_model} table_model={table_model} fallback_model={fallback_model}"
    )

    # Gen config 캐싱(model별)
    gen_cfg_cache: Dict[str, QuestionGenConfig] = {}

    def get_gen_cfg(model: str) -> QuestionGenConfig:
        if model not in gen_cfg_cache:
            gen_cfg_cache[model] = QuestionGenConfig(model=model, temperature=temperature)
        return gen_cfg_cache[model]

    # Ordered preview
    ready_preview: Dict[str, str] = {}
    completed: Set[str] = set()
    next_idx = 0

    def flush_ordered():
        nonlocal next_idx
        if preview <= 0:
            return
        while next_idx < len(all_ids):
            jid = all_ids[next_idx]
            if jid not in todo_ids:
                next_idx += 1
                continue
            if jid not in completed:
                break
            block = ready_preview.pop(jid, None)
            if block:
                print("\n" + block + "\n")
                if preview_log:
                    _append_log(preview_log, block)
                    _append_log(preview_log, "")
            next_idx += 1

    def output_preview(jid: str, block: str):
        if ordered_preview:
            ready_preview[jid] = block
            flush_ordered()
        else:
            print("\n" + block + "\n")
            if preview_log:
                _append_log(preview_log, block)
                _append_log(preview_log, "")

    # =============================================================================
    # Pipeline: Generate → Verify
    # =============================================================================

    def _save_generated_debug(
        *,
        jid: str,
        model_used: str,
        gen_result: Dict[str, Any],
        job: Dict[str, Any],
        status: str,
        extra_error: Optional[str] = None,
        normalized_tables: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not save_generated:
            return
        now = _now_iso()
        _atomic_write_json(generated_out(jid), {
            "pdf_id": pdf_id,
            "job_id": jid,
            "section_id": job.get("section_id"),
            "status": status,
            "model": model_used,
            "attempts": int(gen_result.get("meta", {}).get("attempts", 1) or 1),
            "job_pages": job.get("job_pages"),
            "constraints": job.get("constraints"),
            "tables_in_job": job.get("tables_in_job", []),
            "tables": normalized_tables if normalized_tables is not None else job.get("tables", []),
            "questions": gen_result.get("questions", []),
            "answers_only": gen_result.get("answers_only", []),
            "meta": gen_result.get("meta", {}),
            "evidence_candidates": gen_result.get("evidence_candidates", []),
            "error": gen_result.get("error") or extra_error,
            "updated_at": now,
        })

    def run_job_pipeline(job: Dict[str, Any]) -> Dict[str, Any]:
        jid = job["job_id"]
        section_id = job.get("section_id")

        # (명세 job state) 처음에 QUEUED 기록
        _write_job_state(
            data_dir=data_dir,
            pdf_id=pdf_id,
            job_id=jid,
            section_id=section_id if isinstance(section_id, str) else None,
            local_status=JobStatusLocal.QUEUED,
            detail_stage=None,
        )

        # tables 키 통일
        _normalize_job_tables(job)

        # tables format normalize
        raw_tables = job.get("tables", [])
        norm_tables = _normalize_tables_format(raw_tables)
        job["tables"] = norm_tables
        job["tables_in_job"] = norm_tables  # 디버그/일관성

        model_plan = _model_plan_for_job(
            job,
            default_model=default_model,
            table_model=table_model,
            fallback_model=fallback_model,
        )

        # ---- 1) GENERATOR ----
        last_gen: Dict[str, Any] = {}
        last_model_used: str = model_plan[0] if model_plan else default_model

        for mi, model in enumerate(model_plan, start=1):
            gen_cfg = get_gen_cfg(model)

            def gen_attempt(n: int):
                now = _now_iso()

                # runner-local verified_out 기록(기존 유지)
                _atomic_write_json(verified_out(jid), {
                    "pdf_id": pdf_id,
                    "job_id": jid,
                    "section_id": section_id,
                    "status": JobStatusLocal.GENERATING,
                    "model": model,
                    "attempts": n,
                    "model_try": mi,
                    "model_plan": model_plan,
                    "updated_at": now,
                })

                # (명세 job state) RUNNING/GENERATING 기록
                _write_job_state(
                    data_dir=data_dir,
                    pdf_id=pdf_id,
                    job_id=jid,
                    section_id=section_id if isinstance(section_id, str) else None,
                    local_status=JobStatusLocal.GENERATING,
                    model=model,
                    attempts=n,
                    model_plan=model_plan,
                    detail_stage=None,
                )

                res = generate_questions_for_job(job, gen_cfg, tables=norm_tables)
                meta = res.get("meta", {})
                if isinstance(meta, dict):
                    meta["attempts"] = n
                    res["meta"] = meta
                return res

            gen_result = _retry(gen_attempt, max_retries=max_retries)
            last_gen = gen_result
            last_model_used = model

            # 폴백 조건: 에러 코드 / 빈 questions
            if gen_result.get("error") in ("LLM_OUTPUT_NOT_JSON", "NO_EVIDENCE_CHUNKS"):
                _save_generated_debug(
                    jid=jid,
                    model_used=model,
                    gen_result=gen_result,
                    job=job,
                    status="FAILED",
                    extra_error=str(gen_result.get("error")),
                    normalized_tables=norm_tables,
                )
                continue

            qs = gen_result.get("questions", [])
            if isinstance(qs, list) and len(qs) > 0:
                _save_generated_debug(
                    jid=jid,
                    model_used=model,
                    gen_result=gen_result,
                    job=job,
                    status="DONE",
                    normalized_tables=norm_tables,
                )
                break

            _save_generated_debug(
                jid=jid,
                model_used=model,
                gen_result=gen_result,
                job=job,
                status="FAILED",
                extra_error="EMPTY_QUESTIONS",
                normalized_tables=norm_tables,
            )

        # ---- generator 최종 실패 ----
        if not isinstance(last_gen.get("questions"), list) or len(last_gen.get("questions", [])) == 0:
            now = _now_iso()

            fail_payload = {
                "pdf_id": pdf_id,
                "job_id": jid,
                "section_id": section_id,
                "status": JobStatusLocal.FAILED,
                "error_type": ErrorType.GENERATOR,
                "error": last_gen.get("error") or "EMPTY_QUESTIONS",
                "model": last_model_used,
                "model_plan": model_plan,
                "attempts": int(last_gen.get("meta", {}).get("attempts", 1) or 1),
                "questions": [],
                "summary": {"OK": 0, "FIXABLE": 0, "REJECT": 0},
                "stats": {
                    "total": 0,
                    "valid_chunk_ids": len(last_gen.get("evidence_candidates", []) or []),
                },
                "verified_at": now,
                "updated_at": now,
            }
            _atomic_write_json(verified_out(jid), fail_payload)

            # (명세 job state) FAILED 기록
            _write_job_state(
                data_dir=data_dir,
                pdf_id=pdf_id,
                job_id=jid,
                section_id=section_id if isinstance(section_id, str) else None,
                local_status=JobStatusLocal.FAILED,
                model=last_model_used,
                attempts=int(last_gen.get("meta", {}).get("attempts", 1) or 1),
                model_plan=model_plan,
                error_code=str(fail_payload.get("error") or "GENERATOR_FAILED"),
                error_message="문제 생성에 실패했습니다.",
            )

            return fail_payload

        # ---- 2) VERIFY ----
        now = _now_iso()
        _atomic_write_json(verified_out(jid), {
            "pdf_id": pdf_id,
            "job_id": jid,
            "section_id": section_id,
            "status": JobStatusLocal.VERIFYING,
            "model": last_model_used,
            "updated_at": now,
        })

        # (명세 job state) RUNNING/VERIFYING
        _write_job_state(
            data_dir=data_dir,
            pdf_id=pdf_id,
            job_id=jid,
            section_id=section_id if isinstance(section_id, str) else None,
            local_status=JobStatusLocal.VERIFYING,
            model=last_model_used,
            attempts=int(last_gen.get("meta", {}).get("attempts", 1) or 1),
            model_plan=model_plan,
        )

        # Phase 1: 구조 검증
        verified = verify_questions_for_job(
            job=job,
            generator_result=last_gen,
            evidence_chunks=None,
        )

        # Phase 2: LLM 품질 검증 (OK인 문제만)
        if enable_llm_verify:
            ok_questions = [
                q for q in verified.get("questions", [])
                if q.get("verdict") == "OK"
            ]
            if ok_questions:
                try:
                    llm_result = verify_questions_llm(
                        ok_questions,
                        LLMVerifyConfig(model=last_model_used, temperature=0.1)
                    )
                    # 결과 병합
                    verified["questions"] = merge_verification_results(
                        verified.get("questions", []),
                        llm_result.get("questions", [])
                    )
                    # summary 재계산
                    new_summary = {"OK": 0, "FIXABLE": 0, "REJECT": 0}
                    for q in verified["questions"]:
                        v = q.get("verdict", "OK")
                        if v in new_summary:
                            new_summary[v] += 1
                    verified["summary"] = new_summary
                    verified["llm_verify_done"] = True
                except Exception as e:
                    logger.warning(f"LLM 검증 실패: {e}")
                    verified["llm_verify_done"] = False

        # ---- 2.5) FIXABLE 문제 재생성 (Job 단위, 최대 3회) ----
        MAX_FIXABLE_RETRIES = 3
        fixable_retry_count = 0
        target_q = int(job.get("target_questions") or 0) or 2

        while fixable_retry_count < MAX_FIXABLE_RETRIES:
            current_summary = verified.get("summary", {})
            fixable_count = current_summary.get("FIXABLE", 0)
            reject_count = current_summary.get("REJECT", 0)
            ok_count = current_summary.get("OK", 0)

            # OK 문제가 목표 개수 이상이면 종료
            if ok_count >= target_q:
                break

            # FIXABLE/REJECT 문제가 없으면 종료
            need_regen = fixable_count + reject_count
            if need_regen == 0:
                break

            fixable_retry_count += 1
            logger.info(f"  [{jid}] FIXABLE 재생성 시도 {fixable_retry_count}/{MAX_FIXABLE_RETRIES}: "
                        f"OK={ok_count}, FIXABLE={fixable_count}, REJECT={reject_count}, target={target_q}")

            # (명세 job state) 재생성 중
            _write_job_state(
                data_dir=data_dir,
                pdf_id=pdf_id,
                job_id=jid,
                section_id=section_id if isinstance(section_id, str) else None,
                local_status=JobStatusLocal.GENERATING,
                model=last_model_used,
                attempts=fixable_retry_count,
                model_plan=model_plan,
                progress_value=0.5,
            )

            # OK 문제 유지, FIXABLE/REJECT 문제는 제외
            ok_questions = [q for q in verified.get("questions", []) if q.get("verdict") == "OK"]
            excluded_texts = [
                (q.get("question_text") or q.get("question", "")).lower().strip()
                for q in ok_questions
            ]

            # 부족한 개수만큼 재생성
            additional_needed = target_q - ok_count
            if additional_needed <= 0:
                break

            # 재생성용 job 복사
            regen_job = dict(job)
            regen_job["target_questions"] = additional_needed
            regen_job["exclude_texts"] = excluded_texts

            try:
                gen_cfg = get_gen_cfg(last_model_used)
                regen_result = generate_questions_for_job(regen_job, gen_cfg, tables=norm_tables)
                new_questions = regen_result.get("questions", [])

                if new_questions:
                    # 새 문제 구조 검증
                    new_verified = verify_questions_batch(new_questions)
                    new_qs_with_verdict = new_verified.get("questions", [])

                    # 기존 OK 문제 + 새 문제 병합
                    merged_questions = ok_questions + new_qs_with_verdict

                    # summary 재계산
                    new_summary = {"OK": 0, "FIXABLE": 0, "REJECT": 0}
                    for q in merged_questions:
                        v = q.get("verdict", "OK")
                        if v in new_summary:
                            new_summary[v] += 1

                    verified["questions"] = merged_questions
                    verified["summary"] = new_summary
                    verified["fixable_retries"] = fixable_retry_count

                    logger.info(f"    재생성 결과: +{len(new_questions)}개, 새 OK={new_summary['OK']}")
                else:
                    logger.warning(f"    재생성 실패: 빈 결과")
                    break
            except Exception as e:
                logger.warning(f"    재생성 중 오류: {e}")
                break

        # ---- 3) DONE (runner-local) ----
        now = _now_iso()
        payload = {
            "pdf_id": pdf_id,
            "job_id": jid,
            "section_id": section_id,
            "status": JobStatusLocal.DONE,
            "model": last_model_used,
            "model_plan": model_plan,
            "attempts": int(last_gen.get("meta", {}).get("attempts", 1) or 1),
            "questions": verified.get("questions", []),
            "summary": verified.get("summary", {}),
            "stats": verified.get("stats", {}),
            "verified_at": verified.get("verified_at"),
            "updated_at": now,
        }
        _atomic_write_json(verified_out(jid), payload)

        # (명세 job state) DONE (SAVING)
        _write_job_state(
            data_dir=data_dir,
            pdf_id=pdf_id,
            job_id=jid,
            section_id=section_id if isinstance(section_id, str) else None,
            local_status=JobStatusLocal.DONE,
            model=last_model_used,
            attempts=int(last_gen.get("meta", {}).get("attempts", 1) or 1),
            model_plan=model_plan,
        )

        if save_answers_only and last_gen.get("answers_only"):
            _atomic_write_json(answers_out(jid), {
                "pdf_id": pdf_id,
                "job_id": jid,
                "section_id": section_id,
                "status": JobStatusLocal.DONE,
                "answers": last_gen["answers_only"],
                "updated_at": now,
            })

        return payload

    # =============================================================================
    # 병렬 실행
    # =============================================================================

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_job_pipeline, j): j for j in todo}

        for i, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            jid = job["job_id"]

            try:
                payload = fut.result()
                completed.add(jid)

                status = payload.get("status", "?")
                qs = payload.get("questions", [])
                model_used = payload.get("model", "?")
                summary = payload.get("summary", {})

                ok = summary.get("OK", 0)
                fixable = summary.get("FIXABLE", 0)
                reject = summary.get("REJECT", 0)

                if status == JobStatusLocal.DONE:
                    logger.info(
                        f"[{i}/{len(futures)}] {jid}: DONE "
                        f"questions={len(qs)} (OK={ok}, FIXABLE={fixable}, REJECT={reject}, model={model_used})"
                    )
                    if preview > 0:
                        block = _preview_block(jid, str(payload.get("section_id", "")), qs, preview)
                        output_preview(jid, block)
                else:
                    err = payload.get("error") or payload.get("error_type") or "FAILED"
                    qn = len(qs) if isinstance(qs, list) else 0
                    logger.info(
                        f"[{i}/{len(futures)}] {jid}: FAILED "
                        f"questions={qn} (model={model_used}) reason={err}"
                    )
                    if preview > 0:
                        err_block = _error_preview_block(jid, str(job.get("section_id", "")), str(err))
                        output_preview(jid, err_block)

            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt received. Stopping...")
                raise

            except Exception as e:
                completed.add(jid)

                now_fail = _now_iso()
                _atomic_write_json(verified_out(jid), {
                    "pdf_id": pdf_id,
                    "job_id": jid,
                    "section_id": job.get("section_id"),
                    "status": JobStatusLocal.FAILED,
                    "error_type": ErrorType.UNKNOWN,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "updated_at": now_fail,
                })

                # (명세 job state) FAILED
                _write_job_state(
                    data_dir=data_dir,
                    pdf_id=pdf_id,
                    job_id=jid,
                    section_id=job.get("section_id") if isinstance(job.get("section_id"), str) else None,
                    local_status=JobStatusLocal.FAILED,
                    error_code="UNKNOWN_ERROR",
                    error_message=str(e),
                )

                logger.error(f"[{i}/{len(futures)}] {jid}: FAILED {repr(e)}")

                if preview > 0:
                    err_block = _error_preview_block(jid, str(job.get("section_id", "")), repr(e))
                    output_preview(jid, err_block)

    # Tail flush
    if ordered_preview and preview > 0:
        flush_ordered()
        for jid in all_ids[next_idx:]:
            if jid in ready_preview:
                block = ready_preview[jid]
                print("\n" + block + "\n")
                if preview_log:
                    _append_log(preview_log, block)
                    _append_log(preview_log, "")

    # =============================================================================
    # Aggregate (verified 기준)
    # =============================================================================

    items: List[Dict[str, Any]] = []
    for f in sorted(verified_dir.glob("job_*.json")):
        obj = _load_json_safe(f)
        if obj:
            items.append(obj)

    def _safe_questions_len(it: Dict[str, Any]) -> int:
        qs2 = it.get("questions")
        return len(qs2) if isinstance(qs2, list) else 0

    # =============================================================================
    # Phase 3: Aggregate 검증 및 재생성 루프
    # =============================================================================

    # 전체 문제 수집
    all_questions: List[Dict[str, Any]] = []
    for it in items:
        if it.get("status") != JobStatusLocal.DONE:
            continue
        job_id = it.get("job_id")
        section_id = it.get("section_id")
        for q in it.get("questions", []):
            if isinstance(q, dict):
                q_copy = dict(q)
                q_copy["job_id"] = job_id
                q_copy["section_id"] = section_id
                all_questions.append(q_copy)

    # Job별 target_questions 맵 생성
    job_targets: Dict[str, int] = {}
    for job in jobs:
        jid = job.get("job_id")
        tq = job.get("target_questions", 0)
        if jid:
            job_targets[jid] = tq

    # target_total 결정
    effective_target = target_total if target_total > 0 else sum(job_targets.values())

    # Aggregate 검증
    aggregate_result = verify_aggregate(
        target_total=effective_target,
        all_questions=all_questions,
        job_targets=job_targets,
    )

    logger.info(f"Aggregate 검증: 목표={effective_target}, OK={aggregate_result.actual_ok}, "
                f"부족={aggregate_result.deficit}, 중복={aggregate_result.duplicate_count}")

    # 재생성 루프 (부족/중복 있을 경우)
    regeneration_rounds = 0
    if not aggregate_result.is_satisfied and max_regeneration_rounds > 0:
        logger.info(f"재생성 필요: {aggregate_result.issues}")

        for regen_round in range(max_regeneration_rounds):
            if aggregate_result.is_satisfied:
                break

            regeneration_rounds += 1
            logger.info(f"재생성 라운드 {regeneration_rounds}/{max_regeneration_rounds}")

            # 중복 제거
            if aggregate_result.duplicate_count > 0:
                all_questions, removed = remove_duplicates(all_questions)
                logger.info(f"  중복 제거: {len(removed)}개")

            # 재생성 대상 식별
            targets = identify_regeneration_targets(
                aggregate_result,
                jobs,
                all_questions,
            )

            if not targets:
                logger.info("  재생성 대상 없음")
                break

            # 재생성 실행
            for target in targets:
                logger.info(f"  재생성: {target.job_id} +{target.additional_count}개 ({target.reason})")

                # 해당 Job 찾기
                target_job = next((j for j in jobs if j.get("job_id") == target.job_id), None)
                if not target_job:
                    continue

                try:
                    # 추가 문제 생성
                    target_job_copy = dict(target_job)
                    target_job_copy["target_questions"] = target.additional_count
                    target_job_copy["exclude_texts"] = target.exclude_texts

                    gen_cfg = QuestionGenConfig(model=default_model, temperature=temperature)
                    gen_result = generate_questions_for_job(target_job_copy, gen_cfg)

                    new_questions = gen_result.get("questions", [])
                    if new_questions:
                        # 구조 검증
                        verified_new = verify_questions_batch(new_questions)

                        verified_questions = verified_new.get("questions", [])
                        for q in verified_questions:
                            q["job_id"] = target.job_id
                            q["section_id"] = target.section_id
                            q["regenerated"] = True
                            q["regeneration_round"] = regen_round + 1
                            all_questions.append(q)

                        # ✅ items에도 재생성된 문제 추가 (aggregate JSON 저장용)
                        for it in items:
                            if it.get("job_id") == target.job_id:
                                if "questions" not in it or not isinstance(it["questions"], list):
                                    it["questions"] = []
                                it["questions"].extend(verified_questions)
                                # summary 업데이트
                                ok_count = sum(1 for q in it["questions"] if q.get("verdict") == "OK")
                                fixable_count = sum(1 for q in it["questions"] if q.get("verdict") == "FIXABLE")
                                reject_count = sum(1 for q in it["questions"] if q.get("verdict") == "REJECT")
                                it["summary"] = {"OK": ok_count, "FIXABLE": fixable_count, "REJECT": reject_count}
                                it["updated_at"] = _now_iso()
                                # FAILED 상태였으면 DONE으로 변경
                                if it.get("status") == JobStatusLocal.FAILED and ok_count > 0:
                                    it["status"] = JobStatusLocal.DONE
                                break

                        logger.info(f"    생성됨: {len(new_questions)}개")
                except Exception as e:
                    logger.warning(f"    재생성 실패: {e}")

            # 재검증
            aggregate_result = verify_aggregate(
                target_total=effective_target,
                all_questions=all_questions,
                job_targets=job_targets,
            )

            logger.info(f"  재검증: OK={aggregate_result.actual_ok}, 부족={aggregate_result.deficit}")

    # Aggregate 결과 저장
    save_aggregate_result(aggregate_result, out_dir / "aggregate_verify_result.json")

    agg = {
        "pdf_id": pdf_id,
        "updated_at": _now_iso(),
        "summary": {
            "jobs_total": len(jobs),
            "jobs_done": sum(1 for it in items if it.get("status") == JobStatusLocal.DONE),
            "jobs_failed": sum(1 for it in items if it.get("status") == JobStatusLocal.FAILED),
            "questions_total": len(all_questions),
            "verdict_ok": aggregate_result.actual_ok,
            "verdict_fixable": sum(1 for q in all_questions if q.get("verdict") == "FIXABLE"),
            "verdict_reject": sum(1 for q in all_questions if q.get("verdict") == "REJECT"),
            "target_total": effective_target,
            "is_satisfied": aggregate_result.is_satisfied,
            "regeneration_rounds": regeneration_rounds,
        },
        "paths": {
            "verified_dir": str(verified_dir),
            "generated_dir": str(generated_dir) if save_generated else None,
            "answers_dir": str(answers_dir) if save_answers_only else None,
        },
        "aggregate_issues": aggregate_result.issues,
        "items": items,
    }
    _atomic_write_json(out_dir / "questions_verified_aggregate.json", agg)

    # =============================================================================
    # NEW: Spec result file write (data/results/{pdf_id}.questions.json)
    # =============================================================================
    _write_job_state(
        data_dir=data_dir,
        pdf_id=pdf_id,
        job_id=f"{pdf_id}__RESULTS",
        section_id=None,
        local_status=JobStatusLocal.DONE,
        extra={"note": "pseudo job for results write"},
    )
    results_payload = _flatten_and_write_results(data_dir=data_dir, pdf_id=pdf_id, job_items=items)

    agg["results_written"] = {
        "path": str((data_dir / "results" / f"{pdf_id}.questions.json")),
        "summary": results_payload.get("summary", {}),
    }

    return agg


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="통합 파이프라인: 생성 → 검증 → 결과 저장(명세 경로)")

    ap.add_argument("--out_dir", default="artifacts/lecture")
    ap.add_argument("--jobs", default="question_jobs.jsonl")
    ap.add_argument("--pdf_id", default="lecture")

    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_retry_errors", action="store_true")

    ap.add_argument("--model", default="gpt-4o-mini", help="기본 모델(텍스트 only job)")
    ap.add_argument("--table_model", default="gpt-4o-mini", help="표 포함 job 1차 모델")
    ap.add_argument("--fallback_model", default="gpt-4o-mini", help="표 포함 job 2차 폴백 모델")

    ap.add_argument("--temperature", type=float, default=0.3)

    ap.add_argument("--preview", type=int, default=0, help="미리보기 문제 수")
    ap.add_argument("--ordered_preview", action="store_true", help="순서대로 출력")
    ap.add_argument("--preview_log", default="", help="미리보기 로그 파일")

    ap.add_argument("--no_answers_only", action="store_true")
    ap.add_argument("--save_generated", action="store_true", help="generator 중간산출물 저장(디버그용)")

    # NEW: 명세 기반 저장 루트
    ap.add_argument("--data_dir", default="data", help="명세 저장 루트 (data/jobs, data/results 등)")

    # 검증 파이프라인 옵션
    ap.add_argument("--no_llm_verify", action="store_true", help="LLM 품질 검증 비활성화")
    ap.add_argument("--target_total", type=int, default=0, help="목표 문제 총 개수 (0이면 job별 합계)")
    ap.add_argument("--max_regen_rounds", type=int, default=2, help="최대 재생성 라운드 (0이면 재생성 안함)")

    args = ap.parse_args()

    run(
        out_dir=Path(args.out_dir),
        jobs_jsonl=args.jobs,
        pdf_id=args.pdf_id,
        max_workers=args.workers,
        max_retries=args.max_retries,
        overwrite=args.overwrite,
        retry_errors=not args.no_retry_errors,
        default_model=args.model,
        table_model=args.table_model,
        fallback_model=args.fallback_model,
        temperature=args.temperature,
        preview=args.preview,
        ordered_preview=args.ordered_preview,
        preview_log=Path(args.preview_log) if args.preview_log else None,
        save_answers_only=not args.no_answers_only,
        save_generated=args.save_generated,
        data_dir=Path(args.data_dir),
        # 검증 파이프라인 옵션
        enable_llm_verify=not args.no_llm_verify,
        target_total=args.target_total,
        max_regeneration_rounds=args.max_regen_rounds,
    )


if __name__ == "__main__":
    main()
