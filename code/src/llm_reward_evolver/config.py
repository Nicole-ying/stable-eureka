from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ExperimentConfig:
    env_name: str = "CartPole-v1"
    task_description: str = "Keep the agent alive and solve the task."
    method: str = "fdre_hrdc"
    llm_provider: str = "mock"
    llm_model: str = "mock-reward-coder"
    max_iterations: int = 3
    total_timesteps: int = 50_000
    eval_episodes: int = 5
    target_score: float = 475.0
    reward_clip: float = 10.0
    output_dir: str = "outputs/run"
    seed: int = 42
    patience: int = 3
    min_improvement: float = 1.0
    force_iterations_before_patience: int = 2
    reward_error_fallback: str = "original"
    reward_repair_attempts: int = 1
    num_seeds: int = 1
    feedback_mode: str = "diagnostic"
    reward_structure: str = "hrdc"
    training_algorithm: str = "ppo"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {key: value for key, value in data.items() if key in valid}
        return cls(**filtered)

    def with_overrides(self, **kwargs: Any) -> "ExperimentConfig":
        clean = {key: value for key, value in kwargs.items() if value is not None}
        return replace(self, **clean)


def load_config(path: Optional[str]) -> ExperimentConfig:
    if not path:
        return ExperimentConfig()

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".json", ""}:
            data = json.load(handle)
        elif config_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError("YAML config requires PyYAML. Use JSON or install pyyaml.") from exc
            data = yaml.safe_load(handle)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    return ExperimentConfig.from_dict(data or {})
