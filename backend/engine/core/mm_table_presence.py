# core/mm_table_presence.py
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from openai import OpenAI

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


PROMPT_TABLE_EXISTS_BATCH = """You are given {n} page images. For EACH page, determine if it contains a table.

Schema:
{{"results": [{{"page": 0, "t": true|false}}, {{"page": 1, "t": true|false}}, ...]}}

Rules:
- "page" is the 0-based index of the image in order shown
- "t" = true ONLY if page has a table with clear rows AND columns
- Do NOT count charts, diagrams, code blocks, formulas, or plain text
- Return results for ALL {n} pages
"""


def _image_to_data_url(image_path: Path) -> str:
    """Convert local image file to a base64 data URL."""
    image_path = Path(image_path)
    suffix = image_path.suffix.lower().lstrip(".")
    if suffix not in {"png", "jpg", "jpeg", "webp"}:
        raise ValueError(f"Unsupported image type: .{suffix}")

    mime = "image/png" if suffix == "png" else ("image/webp" if suffix == "webp" else "image/jpeg")
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _extract_json(text: str) -> Dict[str, Any]:
    """Try to parse JSON from model output."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start:end + 1]
        return json.loads(chunk)

    raise ValueError(f"Could not parse JSON from output: {text[:200]}")


def detect_table_presence_mm(image_path: Path, page_index: int) -> TablePresenceResult:
    """단일 페이지 표 탐지 (기존 인터페이스 유지)"""
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


def detect_table_presence_batch(
    pages: List[Tuple[int, Path]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> List[TablePresenceResult]:
    """
    배치 표 탐지: 여러 페이지를 한 번의 API 호출로 처리

    Args:
        pages: [(page_index, image_path), ...] 리스트
        model: 사용할 모델
        temperature: 모델 온도

    Returns:
        List[TablePresenceResult] 각 페이지의 결과
    """
    if not pages:
        return []

    # 단일 페이지면 기존 함수 사용
    if len(pages) == 1:
        pi, path = pages[0]
        return [detect_table_presence_mm(path, pi)]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key)

    # 이미지들을 content 배열로 구성
    content: List[Dict[str, Any]] = []

    # 프롬프트 추가
    prompt = PROMPT_TABLE_EXISTS_BATCH.format(n=len(pages))
    content.append({
        "type": "input_text",
        "text": f"Return ONLY valid JSON. Do not include code fences.\n{prompt}"
    })

    # 각 페이지 이미지 추가
    page_index_map: Dict[int, int] = {}  # batch_idx -> real_page_index
    for batch_idx, (page_index, image_path) in enumerate(pages):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        data_url = _image_to_data_url(image_path)
        content.append({
            "type": "input_image",
            "image_url": data_url,
        })
        page_index_map[batch_idx] = page_index

    # API 호출
    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": content,
        }],
        temperature=temperature,
    )

    text_out = getattr(resp, "output_text", None)
    if not text_out:
        text_out = str(resp)

    # JSON 파싱
    data = _extract_json(text_out)

    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f"Invalid batch output format: {data}")

    results_raw = data["results"]
    if not isinstance(results_raw, list):
        raise ValueError(f"results must be a list: {results_raw}")

    # 결과 매핑
    results: List[TablePresenceResult] = []
    result_map: Dict[int, bool] = {}

    for r in results_raw:
        if isinstance(r, dict) and "page" in r and "t" in r:
            batch_idx = int(r["page"])
            result_map[batch_idx] = bool(r["t"])

    # 원래 순서대로 결과 반환
    for batch_idx, (page_index, _) in enumerate(pages):
        has_table = result_map.get(batch_idx, False)
        results.append(TablePresenceResult(
            page_index=page_index,
            has_table=has_table,
        ))

    return results




