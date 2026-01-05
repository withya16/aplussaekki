# core/pdf_text.py

from __future__ import annotations

import fitz  # PyMuPDF
from pathlib import Path
import json
import re


def extract_pdf_text(
    pdf_path: Path,
    pdf_id: str,
    out_dir: Path
) -> Path:
    """
    Extract page-wise text from PDF using PyMuPDF.

    Output schema (stable) + additive 'layout':
      {
        "pdf_id": str,
        "page_count": int,
        "pages": [
          {
            "page_index": int,      # 0-based
            "page_number": int,     # 1-based (human-friendly)
            "raw_text": str,
            "raw_len": int,
            "lines": [str, ...],
            "maybe_section_title": str | None,

            # NEW (additive)
            "layout": {
              "page_w": float,
              "page_h": float,
              "spans": [
                {
                  "text": str,
                  "size": float,
                  "flags": int,
                  "bbox": [x0, y0, x1, y1],
                  "block_no": int,
                  "line_no": int,
                  "span_no": int
                }
              ]
            }
          }
        ]
      }

    Returns:
        Path to pages_text.json
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))

    pages = []
    for page_index, page in enumerate(doc):
        raw_text = page.get_text("text") or ""
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        maybe_title = _guess_section_title(lines)

        # --- NEW: layout spans (폰트 크기/좌표) ---
        page_w = float(page.rect.width)
        page_h = float(page.rect.height)

        spans = []
        d = page.get_text("dict")
        for b_i, block in enumerate(d.get("blocks", [])):
            if "lines" not in block:
                continue
            for l_i, line in enumerate(block.get("lines", [])):
                for s_i, sp in enumerate(line.get("spans", [])):
                    txt = _clean_text(sp.get("text", ""))
                    if not txt:
                        continue
                    bbox = sp.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    spans.append({
                        "text": txt,
                        "size": float(sp.get("size", 0.0)),
                        "flags": int(sp.get("flags", 0)),
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "block_no": int(b_i),
                        "line_no": int(l_i),
                        "span_no": int(s_i),
                    })

        pages.append({
            "page_index": page_index,
            "page_number": page_index + 1,
            "raw_text": raw_text,
            "raw_len": len(raw_text),
            "lines": lines,
            "maybe_section_title": maybe_title,

            # NEW (additive)
            "layout": {
                "page_w": page_w,
                "page_h": page_h,
                "spans": spans,
            },
        })

    result = {
        "pdf_id": pdf_id,
        "page_count": len(doc),
        "pages": pages,
    }

    # ✅ 0 bytes 방지: tmp에 먼저 쓰고 replace (원자적 저장)
    out_path = out_dir / "pages_text.json"
    tmp_path = out_dir / "pages_text.json.tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.flush()

    tmp_path.replace(out_path)
    return out_path


def _clean_text(s: str) -> str:
    # NBSP 제거 + 공백 정리
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _guess_section_title(lines: list[str]) -> str | None:
    """
    Very light heuristic:
    - Top 5 lines
    - Looks like a section header
    """
    for line in lines[:5]:
        if _looks_like_section(line):
            return line
    return None


def _looks_like_section(line: str) -> bool:
    line = (line or "").strip()
    if len(line) > 80:
        return False

    patterns = [
        r"^\d+[\.\)]\s+.+",      # 1. Intro / 1) Intro
        r"^Chapter\s+\d+.*",     # Chapter 2 ...
        r"^\d+-\d+\s+.+",        # 2-1 Overview
        r"^\d+\.\d+\s+.+",       # 2.1 Overview
    ]

    return any(re.match(p, line, re.IGNORECASE) for p in patterns)
