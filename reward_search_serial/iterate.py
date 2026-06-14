"""
Serial iteration runner. Trains ONE reward function at a time.
Usage: python iterate.py --name iter_001 [--timesteps 2000000] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

from train import train_and_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="e.g. iter_001")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward", default="current_reward.py")
    args = parser.parse_args()

    reward_path = Path(args.reward)
    if not reward_path.exists():
        print(f"ERROR: {reward_path} not found. Write your reward function there first.")
        sys.exit(1)

    reward_code = reward_path.read_text()
    work_dir = Path("results")

    print(f"{'='*60}")
    print(f"Iteration: {args.name}")
    print(f"Reward: {reward_path}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Device: {args.device}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print()

    t0 = time.time()
    result = train_and_eval(
        reward_code=reward_code,
        run_name=args.name,
        work_dir=work_dir,
        total_timesteps=args.timesteps,
        device=args.device,
    )
    elapsed = time.time() - t0

    # Print detailed results
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.name}")
    print(f"{'='*60}")
    print(f"  Wall time:            {elapsed/60:.1f} min")
    print(f"  hidden_return:        {result['hidden_return_mean']:>10.1f} ± {result['hidden_return_std']:.1f}")
    print(f"  generated_return:     {result['generated_return_mean']:>10.1f} ± {result['generated_return_std']:.1f}")
    print(f"  gap (gen - hidden):   {result['generated_hidden_gap']:>10.1f}")
    print()
    print(f"  landing_rate:         {result['landing_rate']:>10.0%}")
    print(f"  crash_rate:           {result['crash_rate']:>10.0%}")
    print(f"  timeout_rate:         {result['timeout_rate']:>10.0%}")
    print(f"  episode_length:       {result['episode_length_mean']:>10.1f} ± {result['episode_length_std']:.1f}")
    print(f"  final_distance:       {result['final_distance_mean']:>10.3f}")
    print()
    print(f"  step_distance_avg:    {result['step_distance_mean']:>10.3f}")
    print(f"  step_velocity_avg:    {result['step_velocity_mean']:>10.3f}")
    print(f"  step_angle_avg:       {result['step_angle_mean']:>10.3f}")
    print()
    print(f"  action_distribution:")
    for a, p in result['action_distribution'].items():
        bar = '█' * int(float(p) * 40)
        print(f"    action {a}: {float(p):>6.1%}  {bar}")
    print()
    print(f"  component_returns (per episode):")
    for k, v in result['component_returns_mean'].items():
        std = result['component_returns_std'].get(k, 0)
        print(f"    {k:<25} {v:>10.1f} ± {std:.1f}")
    print()
    print(f"  final_states (first 5):")
    for i, s in enumerate(result['final_states'][:5]):
        print(f"    ep{i}: x={s['x']:.3f} y={s['y']:.3f} vx={s['vx']:.3f} vy={s['vy']:.3f} angle={s['angle']:.3f} legs=({s['left_leg']:.1f},{s['right_leg']:.1f})")
    print()

    # Save iteration record
    memory_dir = Path("memory/iterations")
    memory_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "name": args.name,
        "timestamp": datetime.now().isoformat(),
        "reward_code": reward_code,
        "reward_file": str(reward_path),
        "timesteps": args.timesteps,
        "result": {k: v for k, v in result.items() if k != "final_states"},
        "final_states": result["final_states"],
    }
    with open(memory_dir / f"{args.name}.json", "w") as f:
        json.dump(record, f, indent=2, default=str)

    print(f"Iteration record saved to memory/iterations/{args.name}.json")


if __name__ == "__main__":
    main()
