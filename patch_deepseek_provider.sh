#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] check repo layout..."
test -d mvp || { echo "ERROR: please run this script at repo root"; exit 1; }

backup_dir="backup_before_deepseek_provider_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/config.py \
  mvp/models.py \
  mvp/configs/cartpole_clean_deepseek_small.yaml \
  scripts/run_clean_cartpole_deepseek_small.sh
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/5] patch mvp/config.py..."
cat > mvp/config.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    provider: str = "openai"  # openai | deepseek | ollama | mock

    llm_model: str = "gpt-4.1"
    vlm_model: str = "gpt-4.1-mini"

    # OpenAI-compatible settings.
    openai_base_url: str | None = None
    openai_api_key_env: str = "OPENAI_API_KEY"

    # DeepSeek API settings.
    # DeepSeek is OpenAI-compatible, but should use its own base_url and key env.
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY"

    # Ollama settings.
    ollama_host: str = "http://localhost:11434"

    temperature: float = 0.7
    max_tokens: int = 1200


@dataclass
class RLConfig:
    env_id: str = "LunarLander-v3"
    total_timesteps: int = 30_000
    eval_episodes: int = 3
    learning_rate: float = 3e-4
    gamma: float = 0.99


@dataclass
class EvolutionConfig:
    generations: int = 4
    population_size: int = 3
    elite_size: int = 1
    reflection_top_k: int = 2
    target_score: float | None = None
    max_stagnation_generations: int | None = None


@dataclass
class MVPConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    workspace: Path = Path("runs/mvp")
    seed: int = 42

    @property
    def memory_path(self) -> Path:
        return self.workspace / "memory.jsonl"

    @property
    def videos_dir(self) -> Path:
        return self.workspace / "videos"

    @property
    def checkpoints_dir(self) -> Path:
        return self.workspace / "checkpoints"


def _deep_update(target: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            target[k] = _deep_update(target[k], v)
        else:
            target[k] = v
    return target


def load_config(path: str | Path | None = None) -> MVPConfig:
    cfg = MVPConfig()
    if path is None:
        return cfg

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    base = {
        "model": cfg.model.__dict__.copy(),
        "rl": cfg.rl.__dict__.copy(),
        "evolution": cfg.evolution.__dict__.copy(),
        "workspace": str(cfg.workspace),
        "seed": cfg.seed,
    }
    merged = _deep_update(base, raw)

    return MVPConfig(
        model=ModelConfig(**merged["model"]),
        rl=RLConfig(**merged["rl"]),
        evolution=EvolutionConfig(**merged["evolution"]),
        workspace=Path(merged["workspace"]),
        seed=int(merged["seed"]),
    )
PY

echo "[3/5] patch mvp/models.py..."
cat > mvp/models.py <<'PY'
import base64
import json
import os
from pathlib import Path

import ollama
from openai import OpenAI

from .config import ModelConfig


class ModelGateway:
    """Thin model abstraction without external agent frameworks."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.provider = config.provider.lower()

        self.openai_client = None
        self.ollama_client = None

        if self.provider == "openai":
            kwargs = {
                "api_key": os.environ.get(config.openai_api_key_env),
            }
            if config.openai_base_url:
                kwargs["base_url"] = config.openai_base_url
            self.openai_client = OpenAI(**kwargs)

        elif self.provider == "deepseek":
            api_key = os.environ.get(config.deepseek_api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing DeepSeek API key. Please export {config.deepseek_api_key_env}=<your_key>"
                )
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url=config.deepseek_base_url,
            )

        elif self.provider == "ollama":
            self.ollama_client = ollama.Client(host=config.ollama_host)

        elif self.provider == "mock":
            pass

        else:
            raise ValueError(
                f"Unsupported model provider: {config.provider}. "
                "Expected one of: openai, deepseek, ollama, mock."
            )

    def chat(self, system: str, user: str) -> str:
        if self.provider in ("openai", "deepseek"):
            response = self.openai_client.chat.completions.create(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content or ""

        if self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": self.config.temperature},
            )
            return response["message"]["content"]

        # mock provider
        return (
            "```python\n"
            "def compute_reward(obs, action, next_obs, done, info):\n"
            "    obs_arr = np.asarray(obs, dtype=float).reshape(-1)\n"
            "    next_arr = np.asarray(next_obs, dtype=float).reshape(-1)\n"
            "    delta = next_arr - obs_arr\n"
            "    progress = float(np.clip(np.linalg.norm(obs_arr) - np.linalg.norm(next_arr), -5.0, 5.0))\n"
            "    stability = float(-0.05 * np.tanh(np.linalg.norm(delta)))\n"
            "    act_arr = np.asarray(action, dtype=float).reshape(-1)\n"
            "    effort = float(-0.01 * np.tanh(np.linalg.norm(act_arr)))\n"
            "    terminal = float(-1.0 if done else 0.0)\n"
            "    total = progress + stability + effort + terminal\n"
            "    components = {\n"
            "        'progress': progress,\n"
            "        'stability': stability,\n"
            "        'effort': effort,\n"
            "        'terminal': terminal,\n"
            "    }\n"
            "    return float(total), components\n"
            "```\n"
            "RATIONALE: clean bounded transition reward using only public transition inputs."
        )

    def judge_video(self, system_prompt: str, rubric: str, video_path: Path) -> dict:
        if self.provider in ("openai", "deepseek"):
            # DeepSeek 当前主要用于文本 reward generation。
            # 如果没有真实多模态能力，这里返回 0 分，不影响 selection_score；
            # selection_score 仍由 private evaluator return 决定。
            if self.provider == "deepseek":
                return {
                    "score": 0.0,
                    "reason": "deepseek_text_only_judge_skipped",
                    "strengths": [],
                    "weaknesses": [],
                }

            b64 = base64.b64encode(video_path.read_bytes()).decode("utf-8")
            data_url = f"data:image/gif;base64,{b64}"
            response = self.openai_client.chat.completions.create(
                model=self.config.vlm_model,
                temperature=0.2,
                max_tokens=400,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": rubric},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
            )
            content = response.choices[0].message.content or "{}"
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"score": 0.0, "reason": f"judge_parse_error: {content[:200]}"}

        if self.provider == "ollama":
            response = self.ollama_client.chat(
                model=self.config.vlm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"{rubric}\nVideo file path: {video_path}. "
                            "If no visual access, return score 0 and explain limitation."
                        ),
                    },
                ],
                format="json",
                options={"temperature": 0.2},
            )
            try:
                return json.loads(response["message"]["content"])
            except json.JSONDecodeError:
                return {"score": 0.0, "reason": "ollama_json_parse_error"}

        return {"score": 0.0, "reason": "mock_judge_no_vision"}
PY

echo "[4/5] write DeepSeek clean CartPole config..."
cat > mvp/configs/cartpole_clean_deepseek_small.yaml <<'YAML'
model:
  provider: deepseek
  llm_model: deepseek-v4-flash
  vlm_model: deepseek-v4-flash
  deepseek_base_url: https://api.deepseek.com
  deepseek_api_key_env: DEEPSEEK_API_KEY
  temperature: 0.7
  max_tokens: 1200

rl:
  env_id: CartPole-v1
  total_timesteps: 8000
  eval_episodes: 3
  learning_rate: 0.0003
  gamma: 0.99

evolution:
  generations: 2
  population_size: 2
  elite_size: 1
  reflection_top_k: 2

workspace: runs/clean_cartpole_deepseek_g2p2_t8k
seed: 42
YAML

echo "[5/5] write DeepSeek run script..."
cat > scripts/run_clean_cartpole_deepseek_small.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="runs/clean_cartpole_deepseek_g2p2_t8k"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "ERROR: DEEPSEEK_API_KEY is not set."
  echo "Please run:"
  echo "  export DEEPSEEK_API_KEY='your_key_here'"
  exit 1
fi

rm -rf "$WORKSPACE"

python run_mvp.py \
  --config mvp/configs/cartpole_clean_deepseek_small.yaml

python scripts/audit_clean_run.py "$WORKSPACE"
SH

chmod +x scripts/run_clean_cartpole_deepseek_small.sh

python -m py_compile \
  mvp/config.py \
  mvp/models.py

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Next:"
echo "  export DEEPSEEK_API_KEY='your_key_here'"
echo "  bash scripts/run_clean_cartpole_deepseek_small.sh"
