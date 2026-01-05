# core/llm_text.py
from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


def call_llm_text(
    *,
    prompt: str,
    model: str,
    temperature: float = 0.3,
    max_output_tokens: int = 2000,
    timeout: Optional[float] = None,
) -> str:
    """
    Text-only LLM call wrapper.
    Uses OpenAI Responses API.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    # Responses API returns output_text aggregated
    return resp.output_text or ""
