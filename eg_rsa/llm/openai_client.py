from __future__ import annotations

import os
from typing import Optional


class OpenAIClient:
    """Optional OpenAI client wrapper for EG-RSA EditAgent.

    This module imports the OpenAI package lazily so the project can still run
    without OpenAI installed when using fallback or Ollama.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = float(temperature)
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
