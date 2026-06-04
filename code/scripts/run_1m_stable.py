"""Run the 1M-timestep stability test with the final FDRE-HRDC reward function.

This replays the canonical reward on LunarLander-v3 for 1M timesteps across
3 seeds, matching the delivery data in data/fdre_hrdc_stable_1m_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from llm_reward_evolver.canonical_rewards import LUNARLANDER_FDRE_HRDC
from llm_reward_evolver.reward import RewardProgram
from llm_reward_evolver.trainer import train_agent


def main() -> None:
    env_name = "LunarLander-v3"
    total_timesteps = 1_000_000
    eval_episodes = 20
    target_score = 200.0
    seeds = [42, 43, 44]

    reward = RewardProgram(LUNARLANDER_FDRE_HRDC, reward_clip=10.0, error_fallback="original")
    records = []

    for seed in seeds:
        print(f"Training seed={seed} for {total_timesteps} timesteps...")
        result = train_agent(
            env_name=env_name,
            reward_program=reward,
            total_timesteps=total_timesteps,
            eval_episodes=eval_episodes,
            target_score=target_score,
            seed=seed,
            training_algorithm="ppo",
        )
        records.append({
            "method": "fdre_hrdc_stable_1m",
            "seed": seed,
            "score": result.stats.mean_eval_score,
            "success_rate": result.stats.success_rate,
            "mean_episode_length": result.stats.mean_episode_length,
            "interrupted": result.stats.interrupted,
            "reward_error_count": result.stats.reward_error_count,
            "failure_mode": result.stats.failure_mode,
        })
        print(f"  score={result.stats.mean_eval_score:.3f}, "
              f"success_rate={result.stats.success_rate:.3f}, "
              f"errors={result.stats.reward_error_count}")

    scores = [r["score"] for r in records]
    success_rates = [r["success_rate"] for r in records]
    summary = {
        "method": "fdre_hrdc_stable_1m",
        "env_name": env_name,
        "total_timesteps": total_timesteps,
        "eval_episodes": eval_episodes,
        "target_score": target_score,
        "mean_score": mean(scores),
        "score_std": pstdev(scores) if len(scores) > 1 else 0.0,
        "success_rate": mean(success_rates),
        "success_std": pstdev(success_rates) if len(success_rates) > 1 else 0.0,
        "mean_episode_length": mean(r["mean_episode_length"] for r in records),
        "all_seeds_over_200": all(s >= 200 for s in scores),
        "solved_over_200": mean(scores) >= 200,
        "interrupted": any(r["interrupted"] for r in records),
        "reward_error_count": sum(r["reward_error_count"] for r in records),
        "seed_scores": scores,
        "seed_success_rates": success_rates,
        "records": records,
    }

    output_path = Path("outputs/1m_stable_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. mean={summary['mean_score']:.2f} ± {summary['score_std']:.2f}, "
          f"all_over_200={summary['all_seeds_over_200']}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
