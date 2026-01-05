# core/table_mm.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.llm_mm import call_mm_json


# =========================
# Result schema
# =========================

@dataclass(frozen=True)
class TableExtractResult:
    page_index: int
    tables: List[Dict[str, Any]]


# =========================
# Prompt
# =========================

PROMPT_EXTRACT_TABLES = """Return ONLY valid JSON. No explanation, no markdown outside JSON.

Task:
Extract ALL tables from the given page image.

Definition of "table":
- A structure with rows AND columns
- Includes simple 2-column tables
- Includes tables with headers or without headers

Ignore:
- Charts, plots, diagrams
- Equations or formulas
- Bullet lists
- Plain paragraphs
- Pseudo-tables made only with spacing

Output schema:
{
  "tables": [
    {
      "table_id": "t01",
      "title": null | "short optional title",
      "format": "markdown",
      "content": "| A | B |\\n|---|---|\\n| 1 | 2 |"
    }
  ]
}

Rules:
- Always return "tables" as a list (empty list if no tables).
- Use GitHub-flavored markdown table format.
- One object per detected table.
- table_id must be unique per page (t01, t02, ...).
- Do NOT add any text outside JSON.
"""


# =========================
# Core extraction function
# =========================

def extract_tables_mm(image_path: Path, page_index: int) -> TableExtractResult:
    """
    Extract tables from a single page image using a multimodal model.

    - Input: page image path
    - Output: TableExtractResult
    - Policy:
        * Invalid or non-JSON output -> raise
        * tables is always a list
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out = call_mm_json(
        prompt=PROMPT_EXTRACT_TABLES,
        image_path=image_path,
        model="gpt-4o",          # 표 추출은 고성능 모델 고정
        temperature=0.0,
    )

    # -------------------------
    # Basic validation
    # -------------------------
    if not isinstance(out, dict):
        raise ValueError(f"MM output is not a dict on page {page_index}: {out}")

    tables = out.get("tables")
    if not isinstance(tables, list):
        raise ValueError(f"Invalid 'tables' field on page {page_index}: {out}")

    # -------------------------
    # Normalize tables
    # -------------------------
    normalized: List[Dict[str, Any]] = []
    seq = 1

    for t in tables:
        if not isinstance(t, dict):
            continue

        table_id = t.get("table_id") or f"t{seq:02d}"
        title = t.get("title", None)
        fmt = t.get("format", "markdown")
        content = t.get("content", "")

        if not isinstance(content, str):
            content = ""

        normalized.append({
            "page_index": page_index,
            "table_id": table_id,
            "title": title,
            "format": fmt,
            "content": content.strip(),
        })

        seq += 1

    return TableExtractResult(
        page_index=page_index,
        tables=normalized,
    )
