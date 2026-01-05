# core/job_builder.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class JobBuilderConfig:
    out_dir: Path = Path("artifacts/lecture")
    pdf_id: str = "lecture"

    # Inputs
    sections_filename: str = "sections.json"
    pages_text_filename: str = "pages_text.json"
    tables_by_page_filename: str = "tables_by_page.json"

    # Outputs
    jobs_jsonl: str = "question_jobs.jsonl"
    index_json: str = "question_jobs_index.json"
    overwrite: bool = True

    # Section size control
    MIN_CHARS: int = 2000
    TARGET_CHARS: int = 9000
    MAX_CHARS: int = 13000

    # Small section buffering (by pages)
    SMALL_BUFFER_PREV_PAGES: int = 1
    SMALL_BUFFER_NEXT_PAGES: int = 1

    # If still too small after buffering, allow deterministic merge with next section (max 1 merge)
    ALLOW_MERGE_TINY_WITH_NEXT: bool = True

    # Text formatting
    page_separator: str = "\n\n----- PAGE {page_index} -----\n\n"

    # Question allocation
    TOTAL_Q: int = 0  # ✅ 0이면 자동 산정
    MIN_Q_PER_SECTION: int = 2
    MAX_Q_PER_SECTION: int = 10
    TABLE_BONUS: float = 2.5

    # Auto TOTAL_Q policy
    AUTO_Q_PER_SECTION: int = 3         # 기본: 섹션당 3문제
    AUTO_CHARS_PER_Q: int = 3000        # 길이 기반 보정
    AUTO_BLEND_SECTION: float = 0.7     # 섹션기반 비중
    AUTO_BLEND_LENGTH: float = 0.3      # 길이기반 비중
    AUTO_MIN_TOTAL: int = 30
    AUTO_MAX_TOTAL: int = 120

    # Table enforcement
    REQUIRE_TABLE_Q_IF_TABLES: bool = True


# =========================
# IO helpers
# =========================

def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# Parsing helpers
# =========================

def _ensure_pages_list(pages_text: Any) -> List[Dict[str, Any]]:
    if isinstance(pages_text, dict) and isinstance(pages_text.get("pages"), list):
        return pages_text["pages"]
    if isinstance(pages_text, list):
        return pages_text
    raise ValueError("pages_text.json must be a list or an object with a 'pages' list")


def _ensure_sections_list(sections_obj: Any) -> List[Dict[str, Any]]:
    """
    sections.json variations:
    - [{...}, {...}]
    - {"sections":[...]} / {"items":[...]} / {"data":[...]}
    - {"index":{"items":[...]}} / {"result":{"sections":[...]}} (rare)
    """
    if isinstance(sections_obj, list):
        return [s for s in sections_obj if isinstance(s, dict)]

    if isinstance(sections_obj, dict):
        # 1-depth wrappers
        for key in ("sections", "items", "data"):
            v = sections_obj.get(key)
            if isinstance(v, list):
                return [s for s in v if isinstance(s, dict)]

        # 2-depth wrappers
        for key in ("index", "result", "payload", "output"):
            v = sections_obj.get(key)
            if isinstance(v, dict):
                for key2 in ("sections", "items", "data"):
                    v2 = v.get(key2)
                    if isinstance(v2, list):
                        return [s for s in v2 if isinstance(s, dict)]

    raise ValueError("sections.json must be a list of section objects (or an object wrapping such a list)")


def _page_text(page_obj: Dict[str, Any]) -> str:
    for k in ("text", "page_text", "content", "raw_text"):
        v = page_obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    spans = page_obj.get("spans") or page_obj.get("layout", {}).get("spans") or []
    if isinstance(spans, list) and spans:
        parts = []
        for s in spans:
            if not isinstance(s, dict):
                continue
            t = s.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
        return "\n".join(parts).strip()

    return ""


def _normalize_tables_by_page(tables_obj: Any) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    if tables_obj is None:
        return out

    if isinstance(tables_obj, dict) and isinstance(tables_obj.get("by_page"), dict):
        for k, v in tables_obj["by_page"].items():
            try:
                pi = int(k)
            except Exception:
                continue
            out[pi] = [t for t in v if isinstance(t, dict)] if isinstance(v, list) else []
        return out

    if isinstance(tables_obj, dict) and isinstance(tables_obj.get("items"), list):
        for it in tables_obj["items"]:
            if not isinstance(it, dict):
                continue
            try:
                pi = int(it.get("page_index"))
            except Exception:
                continue
            tables = it.get("tables", [])
            out[pi] = [t for t in tables if isinstance(t, dict)] if isinstance(tables, list) else []
        return out

    if isinstance(tables_obj, dict):
        for k, v in tables_obj.items():
            try:
                pi = int(k)
            except Exception:
                continue
            out[pi] = [t for t in v if isinstance(t, dict)] if isinstance(v, list) else []
        return out

    if isinstance(tables_obj, list):
        for it in tables_obj:
            if not isinstance(it, dict) or "page_index" not in it:
                continue
            try:
                pi = int(it["page_index"])
            except Exception:
                continue
            tables = it.get("tables", [])
            out[pi] = [t for t in tables if isinstance(t, dict)] if isinstance(tables, list) else []
        return out

    return out


def _section_pages(sec: Dict[str, Any], num_pages: int) -> List[int]:
    pages = sec.get("pages")
    if isinstance(pages, list) and pages:
        pages = [p for p in pages if isinstance(p, int)]
    else:
        ps = sec.get("page_start")
        pe = sec.get("page_end")
        if isinstance(ps, int) and isinstance(pe, int) and pe >= ps:
            pages = list(range(ps, pe + 1))
        else:
            pages = []
    pages = [p for p in pages if 0 <= p < num_pages]
    pages = sorted(set(pages))
    return pages


def _build_text_for_pages(
    pages: List[int],
    pages_list: List[Dict[str, Any]],
    sep: str,
) -> Tuple[str, Dict[int, str]]:
    parts: List[str] = []
    page_texts: Dict[int, str] = {}
    for p in pages:
        txt = _page_text(pages_list[p])
        page_texts[p] = txt
        parts.append(sep.format(page_index=p) + txt)
    merged = "".join(parts).strip()
    return merged, page_texts


def _tables_for_pages(pages: List[int], tables_by_page: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in pages:
        for t in tables_by_page.get(p, []) or []:
            if not isinstance(t, dict):
                continue
            t2 = dict(t)
            t2.setdefault("page_index", p)
            out.append(t2)
    return out


# =========================
# TOTAL_Q auto policy
# =========================

def _auto_total_q(num_sections: int, total_chars: int, cfg: JobBuilderConfig) -> int:
    # 섹션 기반
    q_sec = num_sections * cfg.AUTO_Q_PER_SECTION
    # 길이 기반
    q_len = int(math.ceil(max(total_chars, 1) / max(cfg.AUTO_CHARS_PER_Q, 1)))
    # 블렌딩
    q = int(round(cfg.AUTO_BLEND_SECTION * q_sec + cfg.AUTO_BLEND_LENGTH * q_len))
    # clamp
    q = max(cfg.AUTO_MIN_TOTAL, min(cfg.AUTO_MAX_TOTAL, q))
    return q


# =========================
# Allocation policy
# =========================

def _compute_section_weights(section_infos: List[Dict[str, Any]], table_bonus: float) -> List[float]:
    weights: List[float] = []
    for s in section_infos:
        c = max(0, int(s["char_count"]))
        w = math.sqrt(max(c, 1))
        if s.get("num_tables", 0) > 0:
            w += table_bonus
        weights.append(max(w, 1.0))
    return weights


def _allocate_questions(
    section_infos: List[Dict[str, Any]],
    total_q: int,
    min_q: int,
    max_q: int,
    table_bonus: float,
) -> List[int]:
    if not section_infos:
        return []

    weights = _compute_section_weights(section_infos, table_bonus=table_bonus)
    wsum = sum(weights)

    raw = [total_q * (w / wsum) for w in weights]
    alloc = [int(math.floor(x)) for x in raw]

    # min/max
    alloc = [max(min_q, min(max_q, a)) for a in alloc]
    cur = sum(alloc)

    # reduce if too many
    if cur > total_q:
        idxs = sorted(range(len(alloc)), key=lambda i: alloc[i], reverse=True)
        guard = 0
        while cur > total_q and guard < 100000:
            changed = False
            for j in idxs:
                if cur <= total_q:
                    break
                if alloc[j] > min_q:
                    alloc[j] -= 1
                    cur -= 1
                    changed = True
            if not changed:
                break
            guard += 1

    # add if too few
    if cur < total_q:
        idxs = sorted(range(len(alloc)), key=lambda i: weights[i], reverse=True)
        guard = 0
        while cur < total_q and guard < 100000:
            changed = False
            for j in idxs:
                if cur >= total_q:
                    break
                if alloc[j] < max_q:
                    alloc[j] += 1
                    cur += 1
                    changed = True
            if not changed:
                break
            guard += 1

    return alloc


# =========================
# Job building
# =========================

def _split_pages_into_jobs(
    pages: List[int],
    page_texts: Dict[int, str],
    target_chars: int,
    max_chars: int,
) -> List[List[int]]:
    jobs: List[List[int]] = []
    cur: List[int] = []
    cur_chars = 0

    def page_len(p: int) -> int:
        return len(page_texts.get(p, ""))

    for p in pages:
        pl = page_len(p)

        if cur and (cur_chars + pl) > max_chars:
            jobs.append(cur)
            cur = []
            cur_chars = 0

        cur.append(p)
        cur_chars += pl

        if cur_chars >= target_chars:
            jobs.append(cur)
            cur = []
            cur_chars = 0

    if cur:
        jobs.append(cur)

    return [list(dict.fromkeys(g)) for g in jobs if g]


def _expand_with_buffer(pages: List[int], num_pages_total: int, prev_n: int, next_n: int) -> List[int]:
    if not pages:
        return pages
    start = pages[0]
    end = pages[-1]
    bstart = max(0, start - prev_n)
    bend = min(num_pages_total - 1, end + next_n)
    return list(range(bstart, bend + 1))


def _build_jobs(cfg: JobBuilderConfig) -> Dict[str, Any]:
    base = Path(cfg.out_dir)
    sections_path = base / cfg.sections_filename
    pages_text_path = base / cfg.pages_text_filename
    tables_path = base / cfg.tables_by_page_filename

    if not sections_path.exists():
        raise FileNotFoundError(f"Missing: {sections_path}")
    if not pages_text_path.exists():
        raise FileNotFoundError(f"Missing: {pages_text_path}")

    # ✅ sections.json schema variations absorb
    sections_obj = _read_json(sections_path)
    sections = _ensure_sections_list(sections_obj)

    pages_text_obj = _read_json(pages_text_path)
    pages_list = _ensure_pages_list(pages_text_obj)
    num_pages_total = len(pages_list)

    tables_obj = _read_json(tables_path) if tables_path.exists() else None
    tables_by_page = _normalize_tables_by_page(tables_obj)

    # 1) Collect section stats first (for auto TOTAL_Q + allocation)
    section_infos: List[Dict[str, Any]] = []
    total_chars_all_sections = 0

    for i, sec in enumerate(sections):
        if not isinstance(sec, dict):
            continue
        section_id = sec.get("section_id") or f"S{i:03d}"
        title = sec.get("title") or ""
        pages = _section_pages(sec, num_pages_total)
        text, _page_texts = _build_text_for_pages(pages, pages_list, cfg.page_separator)
        tables = _tables_for_pages(pages, tables_by_page)
        cc = len(text)
        total_chars_all_sections += cc
        section_infos.append({
            "section_id": section_id,
            "title": title,
            "pages": pages,
            "char_count": cc,
            "num_tables": len(tables),
            "has_tables": len(tables) > 0,
        })

    # 2) Decide TOTAL_Q (auto if 0)
    total_q_final = cfg.TOTAL_Q
    if total_q_final <= 0:
        total_q_final = _auto_total_q(len(section_infos), total_chars_all_sections, cfg)

    # 3) Allocate questions per section deterministically
    alloc = _allocate_questions(
        section_infos,
        total_q=total_q_final,
        min_q=cfg.MIN_Q_PER_SECTION,
        max_q=cfg.MAX_Q_PER_SECTION,
        table_bonus=cfg.TABLE_BONUS,
    )
    for s, q in zip(section_infos, alloc):
        s["target_questions"] = int(q)

    # 4) Build jobs with size control (buffer/merge/split)
    jobs: List[Dict[str, Any]] = []
    sec_summary: List[Dict[str, Any]] = []

    i = 0
    while i < len(sections):
        sec = sections[i]
        if not isinstance(sec, dict):
            i += 1
            continue

        section_id = sec.get("section_id") or f"S{i:03d}"
        title = sec.get("title") or ""
        pages = _section_pages(sec, num_pages_total)

        primary_pages = pages[:]
        job_pages = pages[:]
        buffered = False
        merged_with_next = False
        merged_section_ids = [section_id]

        # build initial context
        text, page_texts = _build_text_for_pages(job_pages, pages_list, cfg.page_separator)
        tables = _tables_for_pages(job_pages, tables_by_page)
        char_count = len(text)
        has_tables = len(tables) > 0

        # small: add buffer pages
        if char_count < cfg.MIN_CHARS and pages:
            job_pages = _expand_with_buffer(
                pages,
                num_pages_total=num_pages_total,
                prev_n=cfg.SMALL_BUFFER_PREV_PAGES,
                next_n=cfg.SMALL_BUFFER_NEXT_PAGES,
            )
            buffered = True
            text, page_texts = _build_text_for_pages(job_pages, pages_list, cfg.page_separator)
            tables = _tables_for_pages(job_pages, tables_by_page)
            char_count = len(text)
            has_tables = len(tables) > 0

        # still tiny: merge with next section (max 1) deterministically
        if cfg.ALLOW_MERGE_TINY_WITH_NEXT and char_count < cfg.MIN_CHARS and (i + 1) < len(sections):
            sec_next = sections[i + 1]
            if isinstance(sec_next, dict):
                next_id = sec_next.get("section_id") or f"S{i+1:03d}"
                next_pages = _section_pages(sec_next, num_pages_total)
                merged_pages = sorted(set(job_pages + next_pages))
                text, page_texts = _build_text_for_pages(merged_pages, pages_list, cfg.page_separator)
                tables = _tables_for_pages(merged_pages, tables_by_page)
                char_count = len(text)
                has_tables = len(tables) > 0
                merged_with_next = True
                merged_section_ids = [section_id, next_id]
                job_pages = merged_pages

        # split to jobs
        page_jobs = _split_pages_into_jobs(
            pages=job_pages,
            page_texts=page_texts,
            target_chars=cfg.TARGET_CHARS,
            max_chars=cfg.MAX_CHARS,
        )

        target_q = next((x["target_questions"] for x in section_infos if x["section_id"] == section_id), cfg.MIN_Q_PER_SECTION)

        n_jobs = max(1, len(page_jobs))
        base_q = target_q // n_jobs
        rem = target_q % n_jobs
        per_job_q = [base_q + (1 if j < rem else 0) for j in range(n_jobs)]
        per_job_q = [max(1, int(x)) for x in per_job_q]

        # table job marking (earliest job containing any table page)
        table_job_index: Optional[int] = None
        if cfg.REQUIRE_TABLE_Q_IF_TABLES and has_tables:
            for j, g in enumerate(page_jobs):
                if any((tables_by_page.get(p) for p in g)):
                    table_job_index = j
                    break
            if table_job_index is None:
                table_job_index = 0

        for j, page_group in enumerate(page_jobs):
            grp_text, _grp_page_texts = _build_text_for_pages(page_group, pages_list, cfg.page_separator)
            grp_tables = _tables_for_pages(page_group, tables_by_page)
            job_id = f"{section_id}_J{j+1:02d}"

            jobs.append({
                "job_id": job_id,
                "pdf_id": cfg.pdf_id,

                "section_id": section_id,
                "section_title": title,

                "primary_pages": primary_pages,
                "job_pages": page_group,
                "merged_section_ids": merged_section_ids,
                "buffered": buffered,
                "merged_with_next": merged_with_next,

                "text": grp_text,
                "tables": grp_tables,

                "target_questions": int(per_job_q[j]),
                "constraints": {
                    "must_use_tables": bool(cfg.REQUIRE_TABLE_Q_IF_TABLES and (table_job_index == j)),
                    "has_tables_in_job": len(grp_tables) > 0,
                },
                "stats": {
                    "char_count": len(grp_text),
                    "num_pages": len(page_group),
                    "num_tables": len(grp_tables),
                },
                "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            })

        sec_summary.append({
            "section_id": section_id,
            "title": title,
            "pages": pages,
            "primary_pages": primary_pages,
            "job_pages_union": sorted(set(p for g in page_jobs for p in g)),
            "buffered": buffered,
            "merged_with_next": merged_with_next,
            "merged_section_ids": merged_section_ids,
            "target_questions": int(target_q),
            "num_jobs": len(page_jobs),
            "has_tables": has_tables,
            "char_count_final": char_count,
        })

        i += 2 if merged_with_next else 1

    jobs_path = Path(cfg.out_dir) / cfg.jobs_jsonl
    index_path = Path(cfg.out_dir) / cfg.index_json
    if jobs_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"{jobs_path} exists. Use --overwrite")

    lines = [json.dumps(j, ensure_ascii=False) for j in jobs]
    _write_text(jobs_path, "\n".join(lines) + ("\n" if lines else ""))

    index = {
        "pdf_id": cfg.pdf_id,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "inputs": {
            "sections": str(sections_path),
            "pages_text": str(pages_text_path),
            "tables_by_page": str(tables_path) if tables_path.exists() else None,
        },
        "policy": {
            "MIN_CHARS": cfg.MIN_CHARS,
            "TARGET_CHARS": cfg.TARGET_CHARS,
            "MAX_CHARS": cfg.MAX_CHARS,
            "SMALL_BUFFER_PREV_PAGES": cfg.SMALL_BUFFER_PREV_PAGES,
            "SMALL_BUFFER_NEXT_PAGES": cfg.SMALL_BUFFER_NEXT_PAGES,
            "ALLOW_MERGE_TINY_WITH_NEXT": cfg.ALLOW_MERGE_TINY_WITH_NEXT,
            "MIN_Q_PER_SECTION": cfg.MIN_Q_PER_SECTION,
            "MAX_Q_PER_SECTION": cfg.MAX_Q_PER_SECTION,
            "TABLE_BONUS": cfg.TABLE_BONUS,
            "REQUIRE_TABLE_Q_IF_TABLES": cfg.REQUIRE_TABLE_Q_IF_TABLES,
            "TOTAL_Q_requested": cfg.TOTAL_Q,
            "TOTAL_Q_effective": total_q_final,
            "AUTO": {
                "AUTO_Q_PER_SECTION": cfg.AUTO_Q_PER_SECTION,
                "AUTO_CHARS_PER_Q": cfg.AUTO_CHARS_PER_Q,
                "AUTO_BLEND_SECTION": cfg.AUTO_BLEND_SECTION,
                "AUTO_BLEND_LENGTH": cfg.AUTO_BLEND_LENGTH,
                "AUTO_MIN_TOTAL": cfg.AUTO_MIN_TOTAL,
                "AUTO_MAX_TOTAL": cfg.AUTO_MAX_TOTAL,
                "total_chars_all_sections": total_chars_all_sections,
                "num_sections_used": len(section_infos),
            }
        },
        "summary": {
            "num_sections": len(sec_summary),
            "num_jobs": len(jobs),
            "total_target_questions": sum(j["target_questions"] for j in jobs),
            "num_table_jobs": sum(1 for j in jobs if j["constraints"]["must_use_tables"]),
        },
        "sections": sec_summary,
    }
    _write_json(index_path, index)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="artifacts/lecture")
    ap.add_argument("--pdf_id", type=str, default="lecture")

    ap.add_argument("--total_q", type=int, default=0, help="0이면 자동 산정")
    ap.add_argument("--min_q", type=int, default=2)
    ap.add_argument("--max_q", type=int, default=10)

    ap.add_argument("--min_chars", type=int, default=2000)
    ap.add_argument("--target_chars", type=int, default=9000)
    ap.add_argument("--max_chars", type=int, default=13000)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_merge_tiny", action="store_true")
    ap.add_argument("--buffer_prev", type=int, default=1)
    ap.add_argument("--buffer_next", type=int, default=1)
    ap.add_argument("--no_require_table_q", action="store_true")

    # auto policy knobs (optional)
    ap.add_argument("--auto_q_per_section", type=int, default=3)
    ap.add_argument("--auto_chars_per_q", type=int, default=3000)
    ap.add_argument("--auto_min_total", type=int, default=30)
    ap.add_argument("--auto_max_total", type=int, default=120)

    args = ap.parse_args()

    cfg = JobBuilderConfig(
        out_dir=Path(args.out_dir),
        pdf_id=args.pdf_id,
        overwrite=args.overwrite,

        TOTAL_Q=args.total_q,
        MIN_Q_PER_SECTION=args.min_q,
        MAX_Q_PER_SECTION=args.max_q,

        MIN_CHARS=args.min_chars,
        TARGET_CHARS=args.target_chars,
        MAX_CHARS=args.max_chars,

        SMALL_BUFFER_PREV_PAGES=args.buffer_prev,
        SMALL_BUFFER_NEXT_PAGES=args.buffer_next,
        ALLOW_MERGE_TINY_WITH_NEXT=not args.no_merge_tiny,

        REQUIRE_TABLE_Q_IF_TABLES=not args.no_require_table_q,

        AUTO_Q_PER_SECTION=args.auto_q_per_section,
        AUTO_CHARS_PER_Q=args.auto_chars_per_q,
        AUTO_MIN_TOTAL=args.auto_min_total,
        AUTO_MAX_TOTAL=args.auto_max_total,
    )

    index = _build_jobs(cfg)
    print(f"[OK] jobs={index['summary']['num_jobs']} total_q={index['policy']['TOTAL_Q_effective']}")
    print(f" -> {Path(args.out_dir) / cfg.jobs_jsonl}")
    print(f" -> {Path(args.out_dir) / cfg.index_json}")


if __name__ == "__main__":
    main()

