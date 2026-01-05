# core/llm_mm.py
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI


def _image_to_data_url(image_path: Path) -> str:
    """
    Convert local image file to a base64 data URL.
    OpenAI Responses API supports base64-encoded data URLs as image inputs. :contentReference[oaicite:0]{index=0}
    """
    image_path = Path(image_path)
    suffix = image_path.suffix.lower().lstrip(".")
    if suffix not in {"png", "jpg", "jpeg", "webp"}:
        raise ValueError(f"Unsupported image type: .{suffix}")

    mime = "image/png" if suffix == "png" else ("image/webp" if suffix == "webp" else "image/jpeg")
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Try hard to parse JSON from model output.
    - Prefer pure JSON
    - Fallback: extract first {...} block
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")

    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) try to extract a JSON object block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start : end + 1]
        return json.loads(chunk)

    raise ValueError(f"Could not parse JSON from output: {text[:200]}")


def call_mm_json(
    *,
    prompt: str,
    image_path: Path,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Multimodal call (image + prompt) -> JSON dict.

    Uses OpenAI Responses API (recommended for new projects). :contentReference[oaicite:1]{index=1}
    API key is read from OPENAI_API_KEY environment variable. :contentReference[oaicite:2]{index=2}
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=api_key)

    data_url = _image_to_data_url(image_path)

    # Ask for strict JSON in the response text.
    # (Structured Outputs exists, but keeping this minimal & robust for MVP.)
    full_prompt = (
        "Return ONLY valid JSON. Do not include code fences.\n"
        f"{prompt.strip()}"
    )

    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": full_prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
        temperature=temperature,
        # timeout parameter support can differ by transport; keep optional.
    )

    # openai-python exposes output_text convenience in docs/examples. :contentReference[oaicite:3]{index=3}
    text_out = getattr(resp, "output_text", None)
    if not text_out:
        # Fallback: try to reconstruct from response structure if needed
        # (Usually output_text is present.)
        text_out = str(resp)

    return _extract_json(text_out)
