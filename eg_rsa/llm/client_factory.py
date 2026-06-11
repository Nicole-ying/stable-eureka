from __future__ import annotations

from typing import Any, Dict, Optional

from eg_rsa.llm.deepseek_client import DeepSeekClient
from eg_rsa.llm.ollama_client import OllamaClient
from eg_rsa.llm.openai_client import OpenAIClient


def _get_timeout(config: Dict[str, Any], default: float = 300.0) -> float:
    eg_cfg = config.get("eg_rsa", {}) or {}
    edit_cfg = eg_cfg.get("edit_agent", {}) or {}
    source_cfg = eg_cfg.get("schema_source", {}) or {}
    bootstrap_cfg = eg_cfg.get("bootstrap", {}) or {}

    for cfg in [edit_cfg, source_cfg, bootstrap_cfg]:
        if isinstance(cfg, dict) and cfg.get("timeout") is not None:
            return float(cfg.get("timeout"))
    return float(default)


def build_llm_client(config: Dict[str, Any]) -> Optional[Any]:
    """Build an optional LLM client from eg_rsa.edit_agent config.

    Supported backends:
      - fallback: no LLM client, deterministic edit policy
      - ollama: local Ollama HTTP API
      - openai: OpenAI chat completions API
      - deepseek: DeepSeek chat API
    """

    eg_cfg = config.get("eg_rsa", {}) or {}
    edit_cfg = eg_cfg.get("edit_agent", {}) or {}
    source_cfg = eg_cfg.get("schema_source", {}) or {}

    backend = edit_cfg.get("backend") or source_cfg.get("backend") or "fallback"

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

    if backend == "deepseek":
        return DeepSeekClient(
            model=edit_cfg.get("model") or source_cfg.get("model", "deepseek-v4-pro"),
            credential_env=edit_cfg.get("credential_env") or source_cfg.get("credential_env", "DEEPSEEK_API_KEY"),
            temperature=float(edit_cfg.get("temperature", 0.2)),
            timeout=_get_timeout(config, default=300.0),
        )

    raise ValueError(f"Unsupported EG-RSA edit agent backend: {backend}")
