# core/run_table_extract_mm.py
from __future__ import annotations

import argparse
import json
import logging
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.table_mm import extract_tables_mm

PROMPT_VERSION = "extract_v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def _retry(fn, max_retries: int = 5, base_delay: float = 1.0):
    last = None
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
    raise last


def _load_existing_results(per_page_dir: Path) -> Dict[int, Dict[str, Any]]:
    """페이지별 결과 로드 (page_*.json)"""
    existing: Dict[int, Dict[str, Any]] = {}
    for f in sorted(per_page_dir.glob("page_*.json")):
        obj = _load_json_safe(f)
        if not obj:
            continue
        try:
            existing[int(obj["page_index"])] = obj
        except Exception:
            continue
    return existing


def _normalize_tables(tables: Any, page_index: int) -> List[Dict[str, Any]]:
    """
    후속 단계(패키징/검증/문제생성)가 편하도록 최소 정규화.
    - page_index 강제
    - table_id 없으면 t01.. 부여
    - format/content 기본값 보강
    """
    if not isinstance(tables, list):
        return []
    out: List[Dict[str, Any]] = []
    seq = 1
    for t in tables:
        if not isinstance(t, dict):
            continue
        t2 = dict(t)
        t2["page_index"] = page_index
        if not t2.get("table_id"):
            t2["table_id"] = f"t{seq:02d}"
        t2.setdefault("format", "markdown")
        t2.setdefault("content", "")
        t2.setdefault("title", None)
        out.append(t2)
        seq += 1
    return out


def _write_aggregate(out_dir: Path, pdf_id: str, existing_by_page: Dict[int, Dict[str, Any]]) -> None:
    """집계 파일 생성 (메모리 dict 기반)"""
    # page_index -> tables 맵
    by_page: Dict[str, List[Dict[str, Any]]] = {}
    items = sorted(existing_by_page.values(), key=lambda x: int(x.get("page_index", 1e9)))

    for it in items:
        pi = it.get("page_index")
        if pi is None:
            continue
        by_page[str(pi)] = it.get("tables", []) or []

    agg = {
        "pdf_id": pdf_id,
        "prompt_version": PROMPT_VERSION,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "items": items,
        "by_page": by_page,
        "summary": {
            "num_pages_with_tables": len(by_page),
            "num_tables_total": sum(len(v) for v in by_page.values()),
            "num_errors": sum(1 for it in items if it.get("status") == "error"),
        },
    }
    _atomic_write_json(out_dir / "tables_by_page.json", agg)


def run(
    out_dir: Path = Path("artifacts/lecture"),
    pdf_id: str = "lecture",
    max_workers: int = 2,
    max_retries: int = 5,
    overwrite: bool = False,
    retry_errors: bool = True,
    flush_every: int = 5,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    pages_dir = out_dir / "pages_png"
    status_path = out_dir / "page_status.json"

    if not status_path.exists():
        raise FileNotFoundError("page_status.json not found. Run run_table_presence first.")

    status = _load_json_safe(status_path) or {}
    pages = status.get("pages", [])

    table_pages = sorted(
        int(p["page_index"])
        for p in pages
        if p.get("status", "ok") == "ok" and p.get("has_table") is True
    )

    # 페이지별 결과 저장 폴더
    per_page_dir = out_dir / "tables_by_page"
    per_page_dir.mkdir(parents=True, exist_ok=True)

    # 이미 처리된 것 스캔(메모리 dict로 유지)
    existing_by_page: Dict[int, Dict[str, Any]] = _load_existing_results(per_page_dir)

    def _should_do(pi: int) -> bool:
        if overwrite:
            return True
        if pi not in existing_by_page:
            return True
        if retry_errors and existing_by_page[pi].get("status") == "error":
            return True
        return False

    todo: List[int] = [pi for pi in table_pages if _should_do(pi)]
    todo.sort()

    logger.info(f"has_table_pages={len(table_pages)}, todo={len(todo)}, workers={max_workers}")

    # ✅ 이미 다 했으면 집계만 최신화하고 종료
    if not todo:
        logger.info("Nothing to do. Writing aggregate and exiting.")
        _write_aggregate(out_dir, pdf_id, existing_by_page)
        return _load_json_safe(out_dir / "tables_by_page.json") or {}

    def _extract(pi: int) -> Tuple[int, List[Dict[str, Any]], int]:
        img = pages_dir / f"page_{pi:03d}.png"
        if not img.exists():
            raise FileNotFoundError(f"Missing image: {img}")

        def _do(attempt: int):
            r = extract_tables_mm(img, pi)
            return r.tables, attempt

        tables, attempts = _retry(_do, max_retries=max_retries)
        tables = _normalize_tables(tables, page_index=pi)
        return pi, tables, attempts

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_extract, pi): pi for pi in todo}

        for i, fut in enumerate(as_completed(fut_map), 1):
            completed += 1
            pi = fut_map[fut]
            out_path = per_page_dir / f"page_{pi:03d}.json"

            try:
                pi2, tables, attempts = fut.result()
                payload = {
                    "page_index": pi2,
                    "page_png": str((pages_dir / f"page_{pi2:03d}.png").relative_to(out_dir)),
                    "status": "ok",
                    "attempts": attempts,
                    "tables": tables,
                    "prompt_version": PROMPT_VERSION,
                    "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
                }
                _atomic_write_json(out_path, payload)
                existing_by_page[pi2] = payload  # ✅ 메모리 갱신
                logger.info(f"[{i}/{len(fut_map)}] page {pi2}: {len(tables)} tables (attempts={attempts})")

            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt received. Stopping...")
                raise

            except Exception as e:
                payload = {
                    "page_index": pi,
                    "page_png": str((pages_dir / f"page_{pi:03d}.png").relative_to(out_dir)),
                    "status": "error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "attempts": max_retries,
                    "tables": [],
                    "prompt_version": PROMPT_VERSION,
                    "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
                }
                _atomic_write_json(out_path, payload)
                existing_by_page[pi] = payload  # ✅ 메모리 갱신
                logger.error(f"[{i}/{len(fut_map)}] page {pi} ERROR: {repr(e)}")

            # 주기적으로 집계
            if flush_every > 0 and (completed % flush_every == 0 or completed == len(fut_map)):
                _write_aggregate(out_dir, pdf_id, existing_by_page)

    # ✅ 최종 집계 1회 보장
    _write_aggregate(out_dir, pdf_id, existing_by_page)
    return _load_json_safe(out_dir / "tables_by_page.json") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="artifacts/lecture")
    ap.add_argument("--pdf_id", type=str, default="lecture")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_retry_errors", action="store_true")
    ap.add_argument("--flush_every", type=int, default=5)
    args = ap.parse_args()

    run(
        out_dir=Path(args.out_dir),
        pdf_id=args.pdf_id,
        max_workers=args.workers,
        max_retries=args.max_retries,
        overwrite=args.overwrite,
        retry_errors=not args.no_retry_errors,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
