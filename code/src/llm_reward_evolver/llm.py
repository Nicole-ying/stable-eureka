from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Protocol

import requests


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


@dataclass
class DeepSeekClient:
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = "https://api.deepseek.com/v1/chat/completions"
    temperature: float = 0.4
    timeout: int = 90

    def complete(self, prompt: str) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")

        response = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


@dataclass
class OllamaClient:
    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://localhost:11434/api/generate"
    temperature: float = 0.3
    timeout: int = 600

    def complete(self, prompt: str) -> str:
        model = os.getenv("OLLAMA_MODEL", self.model)
        base_url = os.getenv("OLLAMA_HOST", self.base_url)
        temperature = float(os.getenv("OLLAMA_TEMPERATURE", str(self.temperature)))
        timeout = int(os.getenv("OLLAMA_TIMEOUT", str(self.timeout)))
        response = requests.post(
            base_url,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["response"]


class MockLLMClient:
    """Deterministic local client for dry-runs and tests."""

    def complete(self, prompt: str) -> str:
        lower = prompt.lower()
        if "fixed weights" in lower or "without dynamic weighting" in lower:
            return _static_reward_code()
        if "based only on this scalar" in lower or "scalar score" in lower:
            return _scalar_only_reward_code()
        if "feedback" in lower or "诊断" in prompt:
            return _refined_reward_code()
        return _initial_reward_code()


def build_llm_client(provider: str, model: str) -> LLMClient:
    provider = provider.lower()
    if provider == "deepseek":
        return DeepSeekClient(model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    if provider == "ollama":
        return OllamaClient(model=model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"))
    if provider == "mock":
        return MockLLMClient()
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _initial_reward_code() -> str:
    return """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Initial one-shot reward: intentionally weak shaping for stress tests.
    x = float(next_obs[0]) if len(next_obs) > 0 else 0.0
    angle = float(next_obs[2]) if len(next_obs) > 2 else 0.0
    velocity = float(next_obs[1]) if len(next_obs) > 1 else 0.0
    action_cost = float(sum(abs(a) for a in action)) if hasattr(action, "__iter__") else abs(float(action))

    if training_progress < 0.33:
        w_balance, w_task, w_eff = 0.30, 0.05, 0.65
    elif training_progress < 0.66:
        w_balance, w_task, w_eff = 0.25, 0.10, 0.65
    else:
        w_balance, w_task, w_eff = 0.20, 0.15, 0.65

    r_balance = -abs(angle) - 0.1 * abs(x)
    r_task = 0.02 * float(original_reward) - 0.05 * abs(velocity)
    r_efficiency = -0.05 * action_cost
    return w_balance * r_balance + w_task * r_task + w_eff * r_efficiency
"""


def _refined_reward_code() -> str:
    return """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Refined HRDC reward: keep balance important while preserving env score.
    pos = float(next_obs[0]) if len(next_obs) > 0 else 0.0
    angle = float(next_obs[2]) if len(next_obs) > 2 else 0.0
    velocity = float(next_obs[1]) if len(next_obs) > 1 else 0.0
    action_cost = float(sum(abs(a) for a in action)) if hasattr(action, "__iter__") else abs(float(action))

    if training_progress < 0.33:
        w_balance, w_task, w_eff = 0.65, 0.30, 0.05
    elif training_progress < 0.66:
        w_balance, w_task, w_eff = 0.40, 0.50, 0.10
    else:
        w_balance, w_task, w_eff = 0.30, 0.60, 0.10

    if abs(angle) > 0.20:
        w_balance *= 1.25

    total = w_balance + w_task + w_eff
    w_balance, w_task, w_eff = w_balance / total, w_task / total, w_eff / total
    r_balance = 1.0 - min(abs(angle) / 0.25, 2.0)
    r_center = -0.02 * abs(pos)
    r_smooth = -0.005 * action_cost
    r_task = float(original_reward) + 0.01 * velocity + r_center
    return w_balance * r_balance + w_task * r_task + w_eff * r_smooth
"""


def _scalar_only_reward_code() -> str:
    return """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Scalar-only ablation: partially corrected but lacks diagnostic guidance.
    pos = float(next_obs[0]) if len(next_obs) > 0 else 0.0
    angle = float(next_obs[2]) if len(next_obs) > 2 else 0.0
    action_cost = float(sum(abs(a) for a in action)) if hasattr(action, "__iter__") else abs(float(action))

    r_balance = -abs(angle) - 0.10 * abs(pos)
    r_task = 0.12 * float(original_reward)
    r_smooth = -0.04 * action_cost
    return 0.58 * r_balance + 0.27 * r_task + 0.15 * r_smooth
"""


def _static_reward_code() -> str:
    return """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Ablation reward: decomposed components but fixed weights.
    pos = float(next_obs[0]) if len(next_obs) > 0 else 0.0
    angle = float(next_obs[2]) if len(next_obs) > 2 else 0.0
    action_cost = float(sum(abs(a) for a in action)) if hasattr(action, "__iter__") else abs(float(action))

    r_balance = 1.0 - min(abs(angle) / 0.25, 2.0)
    r_task = 0.18 * float(original_reward) - 0.05 * abs(pos)
    r_efficiency = -0.04 * action_cost
    return 0.55 * r_balance + 0.30 * r_task + 0.15 * r_efficiency
"""
