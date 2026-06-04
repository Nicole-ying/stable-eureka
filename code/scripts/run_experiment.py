from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_reward_evolver.config import load_config
from llm_reward_evolver.evolver import RewardEvolver
from llm_reward_evolver.feedback import TrainingStats, build_feedback
from llm_reward_evolver.llm import MockLLMClient, build_llm_client, extract_code
from llm_reward_evolver.prompts import build_initial_prompt
from llm_reward_evolver.reports import write_suite_outputs
from llm_reward_evolver.reward import RewardProgram
from llm_reward_evolver.suite import ExperimentSuite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FDRE + HRDC reward evolution.")
    parser.add_argument("--config", default="config.example.json", help="JSON/YAML config path.")
    parser.add_argument("--llm-provider", default=None, choices=["mock", "deepseek", "ollama"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate reward generation without RL training.")
    parser.add_argument("--suite", action="store_true", help="Run original baseline, LLM-once, and FDRE.")
    return parser.parse_args()


def dry_run(config) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_initial_prompt(
        config.env_name,
        config.task_description,
        "Box/Discrete space will be inspected during real training.",
        "Box/Discrete action space will be inspected during real training.",
    )
    code = extract_code(MockLLMClient().complete(prompt))
    program = RewardProgram(code, reward_clip=config.reward_clip)
    sample_reward = program([0.0, 0.0, 0.02, 0.0], 0, [0.0, 0.0, 0.01, 0.0], 1.0, {}, 0.2)
    (output_dir / "reward_iter_0.py").write_text(code, encoding="utf-8")

    stats = TrainingStats(
        mean_eval_score=sample_reward,
        success_rate=0.0,
        mean_episode_length=0.0,
        trend="dry_run",
        converged=False,
        failure_mode="dry-run only; no RL training was executed",
    )
    (output_dir / "feedback_iter_0.txt").write_text(build_feedback(stats), encoding="utf-8")
    (output_dir / "history.json").write_text(
        json.dumps(
            [
                {
                    "iteration": 0,
                    "score": sample_reward,
                    "success_rate": 0.0,
                    "mean_episode_length": 0.0,
                    "converged": False,
                    "failure_mode": stats.failure_mode,
                    "reward_code_path": str(output_dir / "reward_iter_0.py"),
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"dry-run ok, sample reward={sample_reward:.4f}, output={output_dir}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config).with_overrides(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    if args.dry_run:
        dry_run(config)
        return

    llm = build_llm_client(config.llm_provider, config.llm_model)
    if args.suite:
        results = ExperimentSuite(config, llm).run()
        write_suite_outputs(results, config.output_dir)
        print(json.dumps([item.__dict__ for item in results], ensure_ascii=False, indent=2))
        return

    records = RewardEvolver(config, llm).run()
    print(json.dumps([record.__dict__ for record in records], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
