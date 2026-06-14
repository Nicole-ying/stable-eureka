"""
Run a batch of reward function candidates.
Usage: python run_batch.py <batch_name> [--timesteps 30000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from train import train_and_eval
from memory import RewardSearchMemory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_name", help="e.g. v1")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    batch = args.batch_name
    rewards_dir = Path("rewards") / batch
    work_dir = Path("results") / batch
    work_dir.mkdir(parents=True, exist_ok=True)

    memory = RewardSearchMemory(Path("memory"))

    reward_files = sorted(rewards_dir.glob("*.py"))
    if not reward_files:
        print(f"No reward files found in {rewards_dir}")
        sys.exit(1)

    print(f"=== Batch: {batch} ===")
    print(f"Reward files: {[f.name for f in reward_files]}")
    print(f"Timesteps per candidate: {args.timesteps}")
    print(f"Total candidates: {len(reward_files)}")
    print()

    results = []
    for rf in reward_files:
        name = rf.stem  # e.g. v1_sparse_terminal
        print(f"{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        reward_code = rf.read_text()
        t0 = time.time()

        try:
            result = train_and_eval(
                reward_code=reward_code,
                run_name=name,
                work_dir=work_dir,
                total_timesteps=args.timesteps,
                device=args.device,
            )
            elapsed = time.time() - t0
            result["batch"] = batch
            result["reward_file"] = str(rf)
            result["wall_time_seconds"] = round(elapsed, 1)
            result["reward_code"] = reward_code

            # Print key metrics
            print(f"  hidden_return:      {result['hidden_return_mean']:>10.1f}")
            print(f"  generated_return:   {result['generated_return_mean']:>10.1f}")
            print(f"  gap:                {result['generated_hidden_gap']:>10.1f}")
            print(f"  landing_rate:       {result['landing_rate']:>10.0%}")
            print(f"  final_distance:     {result['final_distance_mean']:>10.3f}")
            print(f"  episode_length:     {result['episode_length_mean']:>10.1f}")
            print(f"  component_returns:  {result['component_returns']}")
            print(f"  train_time:         {result['train_time_seconds']}s")
            print(f"  wall_time:          {elapsed:.0f}s")

            # Log to memory
            memory.log_experiment({
                "run_name": name,
                "batch": batch,
                "hidden_return_mean": result["hidden_return_mean"],
                "generated_return_mean": result["generated_return_mean"],
                "generated_hidden_gap": result["generated_hidden_gap"],
                "landing_rate": result["landing_rate"],
                "crash_rate": result["crash_rate"],
                "final_distance_mean": result["final_distance_mean"],
                "episode_length_mean": result["episode_length_mean"],
                "action_mean": result["action_mean"],
                "component_returns": result["component_returns"],
                "total_timesteps": args.timesteps,
            })

            results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"run_name": name, "error": str(e)})

        print()

    # --- Summary ---
    print(f"{'='*60}")
    print(f"BATCH SUMMARY: {batch}")
    print(f"{'='*60}")
    print(f"{'Name':<35} {'hidden':>10} {'generated':>10} {'gap':>10} {'land%':>8} {'dist':>8} {'ep_len':>8}")
    print("-" * 95)
    for r in results:
        if "error" in r:
            print(f"{r['run_name']:<35} {'ERROR: ' + r['error'][:40]}")
        else:
            print(f"{r['run_name']:<35} {r['hidden_return_mean']:>10.1f} {r['generated_return_mean']:>10.1f} {r['generated_hidden_gap']:>10.1f} {r['landing_rate']:>8.0%} {r['final_distance_mean']:>8.3f} {r['episode_length_mean']:>8.1f}")

    # Best
    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["hidden_return_mean"])
        print(f"\nBest: {best['run_name']} (hidden={best['hidden_return_mean']:.1f})")

    # Save summary
    summary_path = work_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
