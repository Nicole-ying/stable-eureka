#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


SEEDS = [0, 1, 2]


def write_config(
    path: Path,
    env_id: str,
    workspace: str,
    seed: int,
    total_timesteps: int,
    generations: int = 2,
    population_size: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""model:
  provider: deepseek
  llm_model: deepseek-v4-flash
  vlm_model: deepseek-v4-flash
  deepseek_base_url: https://api.deepseek.com
  deepseek_api_key_env: DEEPSEEK_API_KEY
  deepseek_thinking: disabled
  temperature: 0.7
  max_tokens: 2000

rl:
  env_id: {env_id}
  total_timesteps: {total_timesteps}
  eval_episodes: 3
  learning_rate: 0.0003
  gamma: 0.99

evolution:
  generations: {generations}
  population_size: {population_size}
  elite_size: 1
  reflection_top_k: 2

workspace: {workspace}
seed: {seed}
""",
        encoding="utf-8",
    )


def main() -> None:
    for seed in SEEDS:
        write_config(
            path=Path(f"mvp/configs/cartpole_clean_deepseek_seed{seed}.yaml"),
            env_id="CartPole-v1",
            workspace=f"runs/clean_cartpole_deepseek_seed{seed}_g2p2_t8k",
            seed=seed,
            total_timesteps=8000,
        )

        write_config(
            path=Path(f"mvp/configs/lunar_lander_clean_deepseek_seed{seed}.yaml"),
            env_id="LunarLander-v3",
            workspace=f"runs/clean_lunar_lander_deepseek_seed{seed}_g2p2_t30k",
            seed=seed,
            total_timesteps=30000,
        )

    print("Generated clean multiseed configs for seeds:", SEEDS)


if __name__ == "__main__":
    main()
