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

    openai_base_url: str | None = None
    openai_api_key_env: str = "OPENAI_API_KEY"

    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY"
    deepseek_thinking: str = "disabled"
    deepseek_reasoning_effort: str | None = None

    ollama_host: str = "http://localhost:11434"
    ollama_num_ctx: int | None = 16384

    temperature: float = 0.7
    max_tokens: int = 2500


@dataclass
class RLConfig:
    env_id: str = "LunarLander-v3"

    # Training / evaluation budget
    total_timesteps: int = 30_000
    eval_episodes: int = 3

    # PPO hyperparameters.
    # For LunarLander-v3 long runs we use RL-Baselines3-Zoo style values:
    #   n_steps=1024, batch_size=64, gae_lambda=0.98,
    #   gamma=0.999, n_epochs=4, ent_coef=0.01.
    n_envs: int = 1
    vec_env_type: str = "dummy"  # dummy | subproc
    device: str = "auto"  # auto | cpu | cuda
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    gamma: float = 0.99
    ent_coef: float = 0.0
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Artifact controls.
    # Rendering videos is useful for debugging but should be disabled
    # for long single-chain runs unless visual inspection is needed.
    render_video: bool = True


@dataclass
class EvolutionConfig:
    generations: int = 4
    population_size: int = 3
    elite_size: int = 1
    reflection_top_k: int = 2
    target_score: float | None = None
    max_stagnation_generations: int | None = None


@dataclass
class MemoryConfig:
    candidate_lesson_top_k: int = 5
    env_lesson_top_k: int = 8
    ltm_lesson_top_k: int = 5
    parent_code_top_k: int = 2
    parent_code_max_chars: int = 12000
    feedback_max_chars: int = 12000
    memory_context_max_chars: int = 12000


@dataclass
class MVPConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    workspace: Path = Path("runs/mvp")
    seed: int = 42

    @property
    def memory_path(self) -> Path:
        return self.workspace / "memory.jsonl"

    @property
    def candidate_lessons_path(self) -> Path:
        return self.workspace / "candidate_lessons.jsonl"

    @property
    def env_lessons_path(self) -> Path:
        return self.workspace / "env_lessons.jsonl"

    @property
    def env_memory_path(self) -> Path:
        return self.workspace / "env_memory.md"

    @property
    def ltm_lessons_path(self) -> Path:
        # Cross-run / cross-environment memory under runs/.
        return self.workspace.parent / "ltm_lessons.jsonl"

    @property
    def llm_dir(self) -> Path:
        return self.workspace / "llm"

    @property
    def artifacts_dir(self) -> Path:
        return self.workspace / "artifacts"

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
        "memory": cfg.memory.__dict__.copy(),
        "workspace": str(cfg.workspace),
        "seed": cfg.seed,
    }
    merged = _deep_update(base, raw)

    return MVPConfig(
        model=ModelConfig(**merged["model"]),
        rl=RLConfig(**merged["rl"]),
        evolution=EvolutionConfig(**merged["evolution"]),
        memory=MemoryConfig(**merged.get("memory", {})),
        workspace=Path(merged["workspace"]),
        seed=int(merged["seed"]),
    )
