#!/usr/bin/env python3
import argparse
from pathlib import Path

from .config import load_config
from .exporters import export_memory_csv
from .orchestrator import RewardEvolutionOrchestrator, format_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autonomous reward-evolution MVP")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--provider", type=str, default=None, help="Override model provider: openai|ollama|mock")
    parser.add_argument("--env-id", type=str, default=None, help="Override gym env id")
    parser.add_argument("--timesteps", type=int, default=None, help="Override RL total timesteps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.provider:
        cfg.model.provider = args.provider
    if args.env_id:
        cfg.rl.env_id = args.env_id
    if args.timesteps:
        cfg.rl.total_timesteps = args.timesteps

    orchestrator = RewardEvolutionOrchestrator(cfg)
    best = orchestrator.run()
    report_path = Path(cfg.workspace) / "report.md"
    format_report(best, report_path)
    csv_path = Path(cfg.workspace) / "memory.csv"
    export_memory_csv(cfg.memory_path, csv_path)
    print(f"Done. Report at: {report_path}")
    print(f"Memory CSV at: {csv_path}")


if __name__ == "__main__":
    main()
