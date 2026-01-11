# core/section_indexer.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import re
from collections import Counter


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\u00a0", " ").strip())


def _is_noise(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return True
    if len(t) <= 2:
        return True
    if len(t) > 140:
        return True
    if re.fullmatch(r"\d+(\s*/\s*\d+)?", t):
        return True
    low = t.lower()
    if low.startswith(("figure", "fig.", "table", "page ")):
        return True
    return False


def _group_spans_to_lines(spans: List[Dict[str, Any]], y_tol: float = 8.0) -> List[Dict[str, Any]]:
    """
    Group spans into rough text lines by y0 proximity.
    Returns list of:
      {"y0":..., "y1":..., "text":..., "size_max":..., "size_avg":...}
    """
    if not spans:
        return []

    spans = sorted(spans, key=lambda sp: (sp["bbox"][1], sp["bbox"][0]))
    groups: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_y: Optional[float] = None

    for sp in spans:
        y0 = float(sp["bbox"][1])
        if cur_y is None:
            cur_y = y0
            cur = [sp]
            continue
        if abs(y0 - cur_y) <= y_tol:
            cur.append(sp)
        else:
            groups.append(cur)
            cur = [sp]
            cur_y = y0

    if cur:
        groups.append(cur)

    out = []
    for g in groups:
        txt = _norm(" ".join(_norm(x.get("text", "")) for x in g))
        if _is_noise(txt):
            continue
        size_vals = [float(x.get("size", 0.0)) for x in g if float(x.get("size", 0.0)) > 0]
        if not size_vals:
            continue
        out.append({
            "y0": sum(float(x["bbox"][1]) for x in g) / len(g),
            "y1": max(float(x["bbox"][3]) for x in g),
            "text": txt,
            "size_max": max(size_vals),
            "size_avg": sum(size_vals) / len(size_vals),
        })
    return out


def _title_score(cand: Dict[str, Any], page_h: float) -> float:
    text = cand["text"]
    size = float(cand["size_max"])
    y0 = float(cand["y0"])

    if _is_noise(text):
        return -1e9

    score = 0.0
    score += size * 1.2

    if page_h > 0:
        rel = y0 / page_h
        if rel <= 0.10:
            score += 14
        elif rel <= 0.20:
            score += 10
        elif rel <= 0.35:
            score += 6
        else:
            score -= 3

    L = len(text)
    if 5 <= L <= 70:
        score += 8
    elif L <= 120:
        score += 3
    else:
        score -= 5

    low = text.lower()
    if re.search(r"\b(chapter|section)\b", low):
        score += 10
    if re.match(r"^\d+(\.\d+)*\b", text):
        score += 10
    if re.match(r"^\d+-\d+\b", text):
        score += 8

    return score


def _normalize_title_for_merge(title: str) -> str:
    t = (title or "").strip()
    if not t:
        return ""
    t = re.sub(r"\(\s*\d+\s*/\s*\d+\s*\)\s*$", "", t).strip()
    t = re.sub(r"(\s*[-:]\s*)?\bpart\s+\d+\b\s*$", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\(\s*cont\.?\s*\)\s*$", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\bcontinued\b\s*$", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _merge_adjacent_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for s in sections:
        if not merged:
            merged.append(s)
            continue

        prev = merged[-1]
        prev_title = _normalize_title_for_merge(prev.get("title") or "")
        cur_title = _normalize_title_for_merge(s.get("title") or "")

        if prev_title and cur_title and prev_title.lower() == cur_title.lower():
            prev["pages"].extend(s.get("pages", []))
            prev["title"] = prev_title
        else:
            merged.append(s)

    return merged


def run_section_indexer(
    *,
    out_dir: Path,
    pdf_id: str = "lecture",
    in_path: Optional[Path] = None,
    out_sections: Optional[Path] = None,
    out_page_titles: Optional[Path] = None,
    # ---- tuning knobs ----
    top_region_ratio: float = 0.35,
    repeat_threshold_ratio: float = 0.55,
    min_repeat_pages: int = 3,
    new_section_score: float = 30.0,
) -> Dict[str, Any]:
    """
    pages_text.json(prepare 결과)에서 layout/spans를 이용해 페이지 제목 후보를 뽑고
    섹션 구간(페이지 범위)을 생성한다.

    기본 I/O:
      in_path         = {out_dir}/pages_text.json
      out_sections    = {out_dir}/sections.json
      out_page_titles = {out_dir}/page_titles.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path is None:
        in_path = out_dir / "pages_text.json"
    if out_sections is None:
        out_sections = out_dir / "sections.json"
    if out_page_titles is None:
        out_page_titles = out_dir / "page_titles.json"

    data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    pages = data["pages"]
    n_pages = len(pages)

    # 1) gather top-region candidate lines per page and count repetitions (header removal)
    header_counter = Counter()
    per_page_lines: List[List[Dict[str, Any]]] = []

    for p in pages:
        layout = p.get("layout") or {}
        page_h = float(layout.get("page_h", 0.0))
        spans = layout.get("spans") or []

        top_spans = [sp for sp in spans if page_h and float(sp["bbox"][1]) <= page_h * top_region_ratio]
        lines = _group_spans_to_lines(top_spans)

        lines_sorted = sorted(lines, key=lambda x: x["size_max"], reverse=True)[:6]
        per_page_lines.append(lines_sorted)

        for ln in lines_sorted:
            t = ln["text"]
            if not _is_noise(t):
                header_counter[t] += 1

    # 2) detect repeated headers
    repeated_headers = set()
    threshold = max(min_repeat_pages, int(n_pages * repeat_threshold_ratio)) if n_pages else min_repeat_pages
    for txt, c in header_counter.items():
        if c >= threshold:
            repeated_headers.add(txt)

    # 3) score candidates excluding repeated headers
    page_debug = []
    page_best: List[Dict[str, Any]] = []

    for p, lines in zip(pages, per_page_lines):
        layout = p.get("layout") or {}
        page_h = float(layout.get("page_h", 0.0))
        pi = int(p["page_index"])

        scored = []
        for ln in lines:
            if ln["text"] in repeated_headers:
                continue
            sc = _title_score(ln, page_h)
            scored.append({
                "text": ln["text"],
                "score": float(sc),
                "size_max": float(ln["size_max"]),
                "y0": float(ln["y0"]),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0] if scored else None

        page_best.append({
            "page_index": pi,
            "title_candidate": best["text"] if best and best["score"] > 0 else None,
            "title_score": best["score"] if best else None,
        })

        page_debug.append({
            "page_index": pi,
            "top_candidates": scored[:8],
        })

    # 4) build sections from page_best
    sections: List[Dict[str, Any]] = []
    cur = None

    for row in page_best:
        pi = row["page_index"]
        title = row["title_candidate"]
        score = row["title_score"] if row["title_score"] is not None else -1e9

        is_new = bool(title) and score >= new_section_score

        if cur is None:
            cur = {"section_id": "S001", "title": title, "pages": [pi]}
            continue

        if is_new:
            sections.append(cur)
            sid = f"S{len(sections)+1:03d}"
            cur = {"section_id": sid, "title": title, "pages": [pi]}
        else:
            cur["pages"].append(pi)

    if cur is not None:
        sections.append(cur)

    # 5) post-merge adjacent same-topic sections
    sections = _merge_adjacent_sections(sections)

    out_sections_obj = {
        "pdf_id": pdf_id,
        "page_count": data.get("page_count"),
        "config": {
            "top_region_ratio": top_region_ratio,
            "repeat_threshold_ratio": repeat_threshold_ratio,
            "min_repeat_pages": min_repeat_pages,
            "new_section_score": new_section_score,
        },
        "repeated_headers_removed": sorted(list(repeated_headers)),
        "sections": sections,
    }

    _atomic_write_json(Path(out_sections), out_sections_obj)
    _atomic_write_json(Path(out_page_titles), {"pdf_id": pdf_id, "pages": page_debug})

    print(f"Saved → {out_sections} (sections={len(sections)})")
    print(f"Saved → {out_page_titles}")

    return {
        "in_path": str(Path(in_path).resolve()),
        "out_sections": str(Path(out_sections).resolve()),
        "out_page_titles": str(Path(out_page_titles).resolve()),
        "sections_count": len(sections),
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="섹션 분할: pages_text.json → sections.json/page_titles.json")

    ap.add_argument("--out_dir", default="artifacts/lecture")
    ap.add_argument("--pdf_id", default="lecture")

    ap.add_argument("--in_path", default="", help="기본: {out_dir}/pages_text.json")
    ap.add_argument("--out_sections", default="", help="기본: {out_dir}/sections.json")
    ap.add_argument("--out_page_titles", default="", help="기본: {out_dir}/page_titles.json")

    ap.add_argument("--top_region_ratio", type=float, default=0.35)
    ap.add_argument("--repeat_threshold_ratio", type=float, default=0.55)
    ap.add_argument("--min_repeat_pages", type=int, default=3)
    ap.add_argument("--new_section_score", type=float, default=30.0)

    ap.add_argument("--print_json", action="store_true", help="결과 JSON을 stdout으로 출력")

    args = ap.parse_args(argv)

    res = run_section_indexer(
        out_dir=Path(args.out_dir),
        pdf_id=args.pdf_id,
        in_path=Path(args.in_path) if args.in_path else None,
        out_sections=Path(args.out_sections) if args.out_sections else None,
        out_page_titles=Path(args.out_page_titles) if args.out_page_titles else None,
        top_region_ratio=args.top_region_ratio,
        repeat_threshold_ratio=args.repeat_threshold_ratio,
        min_repeat_pages=args.min_repeat_pages,
        new_section_score=args.new_section_score,
    )

    if args.print_json:
        print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
