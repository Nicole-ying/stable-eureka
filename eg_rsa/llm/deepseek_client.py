from __future__ import annotations

import os


class DeepSeekClient:
    """OpenAI-compatible DeepSeek chat client for EG-RSA."""

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        credential_env: str = "DEEPSEEK_API_KEY",
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.2,
        timeout: float = 300.0,
    ):
        from openai import OpenAI

        credential = os.environ.get(credential_env)
        if not credential:
            raise RuntimeError(f"Missing required environment variable: {credential_env}")

        self.model = model
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.client = OpenAI(
            api_key=credential,
            base_url=base_url,
            timeout=self.timeout,
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Return only valid JSON for EG-RSA reward editing."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            stream=False,
        )
        return response.choices[0].message.content or ""
