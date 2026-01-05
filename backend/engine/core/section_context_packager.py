# core/section_context_packager.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Config
# =========================

@dataclass(frozen=True)
class PackagerConfig:
    out_dir: Path = Path("artifacts/lecture")
    pdf_id: str = "lecture"

    sections_filename: str = "sections.json"
    pages_text_filename: str = "pages_text.json"
    tables_by_page_filename: str = "tables_by_page.json"

    out_subdir: str = "section_contexts"
    overwrite: bool = False

    # Text building
    include_page_texts: bool = True
    prefer_spans: bool = False  # if True, build page text from spans deterministically
    page_separator: str = "\n\n----- PAGE {page_index} -----\n\n"

    # Optional: keep a lightweight block list to reduce noise
    drop_empty_lines: bool = True
    max_consecutive_blank_lines: int = 2

    # Optional: embed anchors for later verification / generation
    build_text_blocks: bool = True
    block_max_chars: int = 800  # split page text into blocks of ~N chars
    block_min_chars: int = 200  # try not to make tiny blocks


# =========================
# IO helpers
# =========================

def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# =========================
# Parsing helpers
# =========================

def _ensure_pages_list(pages_text: Any) -> List[Dict[str, Any]]:
    """
    pages_text.json format variations:
    - {"pages":[{...}, ...]}
    - [{...}, ...]
    """
    if isinstance(pages_text, dict) and isinstance(pages_text.get("pages"), list):
        return pages_text["pages"]
    if isinstance(pages_text, list):
        return pages_text
    raise ValueError("pages_text.json must be a list or an object with a 'pages' list")


def _ensure_sections_list(sections_obj: Any) -> List[Dict[str, Any]]:
    """
    sections.json format variations (프로젝트 진행 중 스키마 흔들림 대응):
    - [{"section_id":..., "title":..., "pages":[...]} , ...]                       ✅ 권장
    - {"sections":[...]} or {"items":[...]} or {"data":[...]}                       ✅ 흔한 래핑
    - {"index":{"items":[...]}} / {"result":{"sections":[...]}}                     ✅ 중첩 래핑(가끔)
    """
    if isinstance(sections_obj, list):
        return [s for s in sections_obj if isinstance(s, dict)]

    if isinstance(sections_obj, dict):
        # 1) 1-depth wrapper
        for key in ("sections", "items", "data"):
            v = sections_obj.get(key)
            if isinstance(v, list):
                return [s for s in v if isinstance(s, dict)]

        # 2) 2-depth wrapper (index/result 등)
        for key in ("index", "result", "payload", "output"):
            v = sections_obj.get(key)
            if isinstance(v, dict):
                for key2 in ("sections", "items", "data"):
                    v2 = v.get(key2)
                    if isinstance(v2, list):
                        return [s for s in v2 if isinstance(s, dict)]

    raise ValueError("sections.json must be a list of section objects (or an object wrapping such a list)")


def _normalize_tables_by_page(tables_obj: Any) -> Dict[int, List[Dict[str, Any]]]:
    """
    tables_by_page.json variations:
    - {"by_page":{"3":[...], "4":[...]}, ...}
    - {"items":[{"page_index":3,"tables":[...]}...], ...}
    - {"3":[...], "4":[...]} (legacy)
    - [{"page_index":3,"tables":[...]}...]
    """
    out: Dict[int, List[Dict[str, Any]]] = {}

    if tables_obj is None:
        return out

    # most expected in your runner: {"by_page": {...}}
    if isinstance(tables_obj, dict) and isinstance(tables_obj.get("by_page"), dict):
        for k, v in tables_obj["by_page"].items():
            try:
                pi = int(k)
            except Exception:
                continue
            if isinstance(v, list):
                out[pi] = [t for t in v if isinstance(t, dict)]
            else:
                out[pi] = []
        return out

    # {"items":[...]}
    if isinstance(tables_obj, dict) and isinstance(tables_obj.get("items"), list):
        for it in tables_obj["items"]:
            if not isinstance(it, dict):
                continue
            try:
                pi = int(it.get("page_index"))
            except Exception:
                continue
            tables = it.get("tables", [])
            if isinstance(tables, list):
                out[pi] = [t for t in tables if isinstance(t, dict)]
            else:
                out[pi] = []
        return out

    # legacy dict: {"3":[...]}
    if isinstance(tables_obj, dict):
        # if it's not the above schema, try interpret as page->tables
        for k, v in tables_obj.items():
            try:
                pi = int(k)
            except Exception:
                continue
            if isinstance(v, list):
                out[pi] = [t for t in v if isinstance(t, dict)]
            else:
                out[pi] = []
        return out

    # list of items
    if isinstance(tables_obj, list):
        for it in tables_obj:
            if not isinstance(it, dict):
                continue
            if "page_index" not in it:
                continue
            try:
                pi = int(it["page_index"])
            except Exception:
                continue
            tables = it.get("tables", [])
            if isinstance(tables, list):
                out[pi] = [t for t in tables if isinstance(t, dict)]
            else:
                out[pi] = []
        return out

    return out


def _safe_get_page_text(page_obj: Dict[str, Any], prefer_spans: bool) -> str:
    """
    Try multiple fields; if prefer_spans=True, rebuild from spans deterministically.
    """
    if prefer_spans:
        spans = page_obj.get("spans") or page_obj.get("layout", {}).get("spans") or []
        if isinstance(spans, list) and spans:
            def key_fn(s: Dict[str, Any]) -> Tuple[float, float]:
                bbox = s.get("bbox") or s.get("bbox_xyxy") or s.get("rect") or [0, 0, 0, 0]
                try:
                    return (float(bbox[1]), float(bbox[0]))  # y, x
                except Exception:
                    return (0.0, 0.0)

            spans_sorted = sorted([s for s in spans if isinstance(s, dict)], key=key_fn)
            parts: List[str] = []
            for s in spans_sorted:
                t = s.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            return "\n".join(parts).strip()

    # prebuilt text fields
    for k in ("text", "page_text", "content", "raw_text"):
        v = page_obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # fallback spans
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


def _squeeze_blank_lines(text: str, max_consecutive: int) -> str:
    if max_consecutive < 1:
        return "\n".join([ln for ln in text.splitlines() if ln.strip()])

    out_lines: List[str] = []
    blanks = 0
    for ln in text.splitlines():
        if ln.strip() == "":
            blanks += 1
            if blanks <= max_consecutive:
                out_lines.append("")
        else:
            blanks = 0
            out_lines.append(ln)
    return "\n".join(out_lines).strip()


def _split_into_blocks(text: str, page_index: int, max_chars: int, min_chars: int) -> List[Dict[str, Any]]:
    """
    Deterministic chunking by paragraphs/lines -> blocks ~max_chars.
    Produces blocks with ids like p003_b001.
    """
    if not text.strip():
        return []

    # paragraph-ish split (keep deterministic)
    paras = [p.strip() for p in text.split("\n")]

    blocks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            blocks.append("\n".join(buf).strip())
            buf = []
            buf_len = 0

    for p in paras:
        if not p:
            # treat blank line as separator (but keep minimal)
            if buf_len >= min_chars:
                flush()
            continue

        add_len = len(p) + (1 if buf else 0)
        if buf_len + add_len > max_chars and buf_len >= min_chars:
            flush()

        buf.append(p)
        buf_len += add_len

    flush()

    out: List[Dict[str, Any]] = []
    for idx, b in enumerate(blocks, start=1):
        out.append({
            "id": f"p{page_index:03d}_b{idx:03d}",
            "page_index": page_index,
            "text": b,
            "char_count": len(b),
        })
    return out


# =========================
# Main builder
# =========================

def build_section_contexts(cfg: PackagerConfig) -> Dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    base = out_dir  # artifacts live directly here, not nested by pdf_id in current layout

    sections_path = base / cfg.sections_filename
    pages_text_path = base / cfg.pages_text_filename
    tables_path = base / cfg.tables_by_page_filename

    if not sections_path.exists():
        raise FileNotFoundError(f"Missing: {sections_path}")
    if not pages_text_path.exists():
        raise FileNotFoundError(f"Missing: {pages_text_path}")

    sections_obj = _read_json(sections_path)
    sections = _ensure_sections_list(sections_obj)

    pages_text_obj = _read_json(pages_text_path)
    pages_list = _ensure_pages_list(pages_text_obj)

    tables_obj = _read_json(tables_path) if tables_path.exists() else None
    tables_by_page = _normalize_tables_by_page(tables_obj)

    out_subdir = base / cfg.out_subdir
    out_subdir.mkdir(parents=True, exist_ok=True)

    index_items: List[Dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).astimezone().isoformat()

    for si, sec in enumerate(sections):
        if not isinstance(sec, dict):
            continue

        section_id = sec.get("section_id") or f"S{si:03d}"
        title = sec.get("title") or ""
        pages = sec.get("pages")

        # fallback reconstruction
        if not isinstance(pages, list) or not pages:
            ps = sec.get("page_start")
            pe = sec.get("page_end")
            if isinstance(ps, int) and isinstance(pe, int) and pe >= ps:
                pages = list(range(ps, pe + 1))
            else:
                pages = []

        pages = [p for p in pages if isinstance(p, int)]
        pages = [p for p in pages if 0 <= p < len(pages_list)]

        out_path = out_subdir / f"section_{section_id}.json"
        if out_path.exists() and not cfg.overwrite:
            index_items.append({
                "section_id": section_id,
                "title": title,
                "path": str(out_path.relative_to(base)),
                "pages": pages,
            })
            continue

        page_texts: List[Dict[str, Any]] = []
        merged_parts: List[str] = []
        all_text_blocks: List[Dict[str, Any]] = []
        all_tables: List[Dict[str, Any]] = []

        for p in pages:
            page_obj = pages_list[p]
            txt = _safe_get_page_text(page_obj, prefer_spans=cfg.prefer_spans)

            if cfg.drop_empty_lines:
                txt = _squeeze_blank_lines(txt, cfg.max_consecutive_blank_lines)

            if cfg.include_page_texts:
                page_texts.append({"page_index": p, "text": txt})

            merged_parts.append(cfg.page_separator.format(page_index=p) + txt)

            if cfg.build_text_blocks:
                all_text_blocks.extend(_split_into_blocks(
                    txt, page_index=p, max_chars=cfg.block_max_chars, min_chars=cfg.block_min_chars
                ))

            tlist = tables_by_page.get(p, [])
            if isinstance(tlist, list) and tlist:
                for t in tlist:
                    if not isinstance(t, dict):
                        continue
                    t2 = dict(t)
                    t2.setdefault("page_index", p)
                    t2.setdefault("table_id", t2.get("table_id") or t2.get("id") or None)
                    t2.setdefault("format", t2.get("format") or "markdown")
                    t2.setdefault("content", t2.get("content") or "")
                    all_tables.append(t2)

        merged_text = "".join(merged_parts).strip()

        payload: Dict[str, Any] = {
            "pdf_id": cfg.pdf_id,
            "section_id": section_id,
            "title": title,
            "pages": pages,
            "page_start": pages[0] if pages else None,
            "page_end": pages[-1] if pages else None,
            "text": merged_text,
            "page_texts": page_texts if cfg.include_page_texts else None,
            "tables": all_tables,
            "anchors": {
                "text_blocks": all_text_blocks if cfg.build_text_blocks else None,
                "tables": [
                    {
                        "id": f"p{t.get('page_index', -1):03d}_{t.get('table_id') or 't??'}",
                        "page_index": t.get("page_index"),
                        "table_id": t.get("table_id"),
                        "title": t.get("title", None),
                    }
                    for t in all_tables
                ],
            },
            "stats": {
                "num_pages": len(pages),
                "num_tables": len(all_tables),
                "char_count": len(merged_text),
                "num_text_blocks": len(all_text_blocks),
            },
            "sources": {
                "sections_json": str(sections_path.relative_to(base)),
                "pages_text_json": str(pages_text_path.relative_to(base)),
                "tables_by_page_json": str(tables_path.relative_to(base)) if tables_path.exists() else None,
            },
            "generated_at": now_iso,
            "config": {
                "prefer_spans": cfg.prefer_spans,
                "include_page_texts": cfg.include_page_texts,
                "build_text_blocks": cfg.build_text_blocks,
                "block_max_chars": cfg.block_max_chars,
                "block_min_chars": cfg.block_min_chars,
            },
        }

        _atomic_write_json(out_path, payload)

        index_items.append({
            "section_id": section_id,
            "title": title,
            "path": str(out_path.relative_to(base)),
            "pages": pages,
            "stats": payload["stats"],
        })

    index_obj = {
        "pdf_id": cfg.pdf_id,
        "out_dir": str(out_subdir.relative_to(base)),
        "generated_at": now_iso,
        "num_sections": len(index_items),
        "items": index_items,
    }
    _atomic_write_json(out_subdir / "index.json", index_obj)
    return index_obj


# =========================
# CLI
# =========================

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="artifacts/lecture")
    ap.add_argument("--pdf_id", type=str, default="lecture")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--prefer_spans", action="store_true")
    ap.add_argument("--no_page_texts", action="store_true")
    ap.add_argument("--no_text_blocks", action="store_true")
    ap.add_argument("--block_max_chars", type=int, default=800)
    ap.add_argument("--block_min_chars", type=int, default=200)
    args = ap.parse_args()

    cfg = PackagerConfig(
        out_dir=Path(args.out_dir),
        pdf_id=args.pdf_id,
        overwrite=args.overwrite,
        prefer_spans=args.prefer_spans,
        include_page_texts=not args.no_page_texts,
        build_text_blocks=not args.no_text_blocks,
        block_max_chars=args.block_max_chars,
        block_min_chars=args.block_min_chars,
    )

    index = build_section_contexts(cfg)
    print(f"[OK] section contexts: {index['num_sections']}")
    print(f" -> {Path(args.out_dir) / cfg.out_subdir / 'index.json'}")


if __name__ == "__main__":
    main()

