# core/mm_table_presence.py
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from core.llm_mm import call_mm_json


@dataclass(frozen=True)
class TablePresenceResult:
    page_index: int
    has_table: bool


PROMPT_TABLE_EXISTS = """Return ONLY valid JSON, no extra text.

Schema:
{"t": true|false}

Set "t" to true ONLY if there is a table with clear rows AND columns.
Do NOT count charts, diagrams, code blocks, formulas, or plain text.
"""


def detect_table_presence_mm(image_path: Path, page_index: int) -> TablePresenceResult:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out = call_mm_json(
        prompt=PROMPT_TABLE_EXISTS,
        image_path=image_path,
        model="gpt-4o-mini",
        temperature=0.0,
    )

    if not isinstance(out, dict) or "t" not in out:
        raise ValueError(f"Invalid MM output on page {page_index}: {out}")

    return TablePresenceResult(page_index=page_index, has_table=bool(out["t"]))




