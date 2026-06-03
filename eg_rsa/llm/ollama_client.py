from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict


class OllamaClient:
    """Minimal Ollama HTTP client used by EG-RSA EditAgent.

    It uses the local Ollama generate API and returns the raw response text.
    The EditAgent is responsible for parsing JSON from that text.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
        timeout: int = 120,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = float(temperature)
        self.timeout = int(timeout)

    def generate(self, prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return str(result.get("response", ""))
