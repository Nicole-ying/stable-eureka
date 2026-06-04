from __future__ import annotations

from typing import Any, Dict, Optional

from eg_rsa.llm.deepseek_client import DeepSeekClient
from eg_rsa.llm.ollama_client import OllamaClient
from eg_rsa.llm.openai_client import OpenAIClient


def build_llm_client(config: Dict[str, Any]) -> Optional[Any]:
    """Build an optional LLM client from eg_rsa.edit_agent config.

    Supported backends:
      - fallback: no LLM client, deterministic edit policy
      - ollama: local Ollama HTTP API
      - openai: OpenAI chat completions API
      - deepseek: DeepSeek chat API
    """

    edit_cfg = config.get("eg_rsa", {}).get("edit_agent", {}) or {}
    backend = edit_cfg.get("backend", "fallback")
    if backend in (None, "", "fallback"):
        return None
    if backend == "ollama":
        return OllamaClient(
            model=edit_cfg.get("model", "qwen2.5:14b"),
            host=edit_cfg.get("host", "http://localhost:11434"),
            temperature=float(edit_cfg.get("temperature", 0.2)),
            timeout=int(edit_cfg.get("timeout", 120)),
        )
    if backend == "openai":
        return OpenAIClient(
            model=edit_cfg.get("model", "gpt-4o-mini"),
            api_key=edit_cfg.get("api_key"),
            temperature=float(edit_cfg.get("temperature", 0.2)),
        )
    raise ValueError(f"Unsupported EG-RSA edit agent backend: {backend}")
