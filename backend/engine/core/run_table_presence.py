# core/run_table_presence.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import random
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.mm_table_presence import detect_table_presence_mm


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


def _page_index(png: Path) -> int:
    return int(png.stem.split("_")[-1])


def _retry(fn, max_retries: int = 5, base_delay: float = 1.0):
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(attempt)
        except Exception as e:
            last = e
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
            time.sleep(min(delay, 15))
    raise last


def run(
    out_dir: Path = Path("artifacts/lecture"),
    pdf_id: str = "lecture",
    max_pages: Optional[int] = None,
    max_workers: int = 3,
    max_retries: int = 5,
    retry_errors: bool = True,
    flush_every: int = 1,
) -> Dict[str, Any]:

    out_dir = Path(out_dir)
    pages_dir = out_dir / "pages_png"
    status_path = out_dir / "page_status.json"

    all_pngs = sorted(pages_dir.glob("page_*.png"))
    if not all_pngs:
        raise FileNotFoundError("pages_png not found. Run prepare first.")

    pngs = all_pngs[:max_pages] if max_pages else all_pngs
    page_count_total = len(all_pngs)

    existing = _load_json_safe(status_path)
    pages_status: Dict[int, Dict[str, Any]] = {}

    if existing:
        for row in existing.get("pages", []):
            try:
                pages_status[int(row["page_index"])] = row
            except Exception:
                continue

    def _should_do(pi: int) -> bool:
        if pi not in pages_status:
            return True
        if retry_errors and pages_status[pi].get("status") == "error":
            return True
        return False

    todo: List[Tuple[int, Path]] = [(_page_index(p), p) for p in pngs if _should_do(_page_index(p))]
    todo.sort(key=lambda x: x[0])

    def _detect(pi: int, png: Path) -> Tuple[int, Path, bool, Optional[str], int]:
        def _do(attempt: int):
            r = detect_table_presence_mm(png, pi)
            return r.has_table, attempt
        has_table, attempts = _retry(_do, max_retries=max_retries)
        return pi, png, bool(has_table), None, attempts

    print(f"[presence] total_pages={len(all_pngs)} selected={len(pngs)} todo={len(todo)} workers={max_workers}")

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_detect, pi, png) for pi, png in todo]

        for i, fut in enumerate(as_completed(futures), 1):
            completed += 1
            try:
                pi, png, has_table, err, attempts = fut.result()
                pages_status[pi] = {
                    "page_index": pi,
                    "page_png": str(png.relative_to(out_dir)),
                    "has_table": bool(has_table),
                    "status": "ok",
                    "attempts": attempts,
                }
                print(f"[{i}/{len(futures)}] page {pi} has_table={has_table} (attempts={attempts})")

            except Exception as e:
                print(f"[{i}/{len(futures)}] ERROR: {repr(e)}")

            if flush_every > 0 and (completed % flush_every == 0 or completed == len(futures)):
                _atomic_write_json(status_path, {
                    "pdf_id": pdf_id,
                    "page_count": page_count_total,
                    "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
                    "prompt_version": "presence_v1",
                    "pages": sorted(pages_status.values(), key=lambda x: x["page_index"]),
                    "summary": {
                        "num_pages": len(pages_status),
                        "num_ok": sum(1 for v in pages_status.values() if v.get("status") == "ok"),
                        "num_errors": sum(1 for v in pages_status.values() if v.get("status") == "error"),
                        "num_has_table": sum(
                            1 for v in pages_status.values()
                            if v.get("status") == "ok" and v.get("has_table") is True
                        ),
                    }
                })

    return _load_json_safe(status_path) or {}


# =========================
# CLI wrapper (NEW)
# =========================

def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="표 존재 탐지(MM): 페이지별 has_table 판단")
    ap.add_argument("--out_dir", default="artifacts/lecture")
    ap.add_argument("--pdf_id", default="lecture")
    ap.add_argument("--max_pages", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--no_retry_errors", action="store_true")
    ap.add_argument("--flush_every", type=int, default=1)
    ap.add_argument("--print_json", action="store_true", help="결과 JSON을 stdout으로 출력")

    args = ap.parse_args(argv)

    result = run(
        out_dir=Path(args.out_dir),
        pdf_id=args.pdf_id,
        max_pages=args.max_pages,
        max_workers=args.workers,
        max_retries=args.max_retries,
        retry_errors=not args.no_retry_errors,
        flush_every=args.flush_every,
    )

    if args.print_json:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()



