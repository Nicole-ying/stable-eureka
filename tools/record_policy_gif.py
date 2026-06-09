from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model.zip")
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--out", required=True, help="Output GIF path")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="rgb_array")
    model = PPO.load(args.model)

    reset_out = env.reset(seed=args.seed)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    frames = []
    ep_return = 0.0
    ep_len = 0
    done = False

    while not done and ep_len < args.max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=not args.stochastic)
        step_out = env.step(action)

        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, _ = step_out
            done = bool(done)

        ep_return += float(np.asarray(reward).reshape(-1)[0])
        ep_len += 1

    final_frame = env.render()
    if final_frame is not None:
        frames.append(final_frame)

    env.close()

    if not frames:
        raise RuntimeError("No rendered frames were produced.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps)

    print(f"[OK] saved: {out}")
    print(f"return={ep_return:.2f}, length={ep_len}, frames={len(frames)}")


if __name__ == "__main__":
    main()
