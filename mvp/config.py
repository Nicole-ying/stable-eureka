from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    provider: str = "openai"  # openai | ollama | mock
    llm_model: str = "gpt-4.1"
    vlm_model: str = "gpt-4.1-mini"
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
