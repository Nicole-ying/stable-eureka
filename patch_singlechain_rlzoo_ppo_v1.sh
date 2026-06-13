#!/usr/bin/env bash
set -euo pipefail

echo "[1/7] check repo root..."
test -d mvp || { echo "ERROR: run this script at repo root"; exit 1; }

backup_dir="backup_before_singlechain_rlzoo_ppo_v1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for f in \
  mvp/config.py \
  mvp/rl_worker.py \
  mvp/orchestrator.py \
  scripts/check_run_quality.py \
  mvp/configs/eg_rsa_lunar_deepseek_seed0_singlechain_g3_t300k.yaml \
  mvp/configs/eg_rsa_lunar_deepseek_seed0_singlechain_g10_t2m.yaml
do
  if [ -f "$f" ]; then
    mkdir -p "$backup_dir/$(dirname "$f")"
    cp "$f" "$backup_dir/$f"
  fi
done

echo "[2/7] patch mvp/config.py..."
python - <<'PY'
from pathlib import Path

p = Path("mvp/config.py")
s = p.read_text(encoding="utf-8")

old = '''@dataclass
class RLConfig:
    env_id: str = "LunarLander-v3"


    total_timesteps: int = 30_000
    eval_episodes: int = 3
    learning_rate: float = 3e-4
    gamma: float = 0.99
'''

new = '''@dataclass
class RLConfig:
    env_id: str = "LunarLander-v3"

    # Training budget
    total_timesteps: int = 30_000
    eval_episodes: int = 3

    # RL Zoo PPO-style parameters for LunarLander-v3.
    # RL Zoo reference:
    #   n_envs=16, n_steps=1024, batch_size=64,
    #   gae_lambda=0.98, gamma=0.999, n_epochs=4, ent_coef=0.01.
    n_envs: int = 1
    vec_env_type: str = "dummy"  # dummy | subproc
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

    # Artifact controls
    render_video: bool = True
'''

if old not in s:
    raise SystemExit("Could not find RLConfig block. Please inspect mvp/config.py manually.")

s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
PY

echo "[3/7] rewrite mvp/rl_worker.py with VecEnv support..."
cat > mvp/rl_worker.py <<'PY'
from __future__ import annotations

import ast
import math
import types
from pathlib import Path
from typing import Callable

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from .config import RLConfig


RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, bool, dict], tuple[float, dict]]


FORBIDDEN_NAMES = {
    "env_reward",
    "fitness_score",
    "benchmark_reward",
    "official_reward",
    "original_reward",
    "hidden_reward",
    "_hidden_env_reward",
    "compute_fitness_score",
}


def compile_reward_function(reward_code: str) -> RewardFn:
    raw_lower = reward_code.lower()
    for name in FORBIDDEN_NAMES:
        if name.lower() in raw_lower:
            raise ValueError(f"forbidden token appears in reward code: {name}")

    tree = ast.parse(reward_code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.ClassDef, ast.Lambda)):
            raise ValueError(f"unsupported syntax in reward code: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"forbidden name in reward code: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            raise ValueError(f"forbidden attribute in reward code: {node.attr}")

    module = types.ModuleType("reward_module")
    module.__dict__["np"] = np
    module.__dict__["math"] = math
    exec(compile(tree, filename="<reward_code>", mode="exec"), module.__dict__)

    if "compute_reward" not in module.__dict__:
        raise ValueError("reward code must define compute_reward(obs, action, next_obs, done, info)")
    return module.__dict__["compute_reward"]


class RewardFunctionWrapper(gym.Wrapper):
    """
    EG-RSA wrapper.

    Policy sees the original Gym observation.
    Reward function sees public transition inputs:
      obs, action, next_obs, done, info

    It never receives env_reward. The original env reward is stored only as
    private evaluation signal in safe_info["_hidden_env_reward"].
    """

    def __init__(self, env: gym.Env, reward_fn: RewardFn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs
        return obs, info

    def step(self, action):
        next_obs, hidden_env_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        out = self.reward_fn(
            self._prev_obs,
            action,
            next_obs,
            done,
            info,
        )
        if not isinstance(out, tuple) or len(out) != 2:
            raise ValueError("compute_reward must return (reward, components)")

        reward, components = out
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("generated reward is not finite")
        if not isinstance(components, dict):
            raise ValueError("reward components must be a dict")

        safe_info = dict(info)
        safe_info["_hidden_env_reward"] = float(hidden_env_reward)
        safe_info["_reward_components"] = {str(k): float(v) for k, v in components.items()}

        self._prev_obs = next_obs
        return next_obs, reward, terminated, truncated, safe_info


def _make_train_env(env_id: str, reward_code: str, rank: int):
    """
    Build env thunk for DummyVecEnv/SubprocVecEnv.

    The reward code is compiled inside each worker process. This avoids
    pickling dynamically exec-created function objects across processes.
    """
    def _init():
        reward_fn = compile_reward_function(reward_code)
        env = gym.make(env_id)
        env = RewardFunctionWrapper(env, reward_fn)
        # Avoid forcing exact seeding here; orchestrator already controls Python/NumPy seeds.
        # For long experiments, stochasticity across vector envs is desirable.
        return env

    return _init


def _make_vec_env(cfg: RLConfig, reward_code: str) -> VecEnv:
    n_envs = max(1, int(cfg.n_envs))
    env_fns = [_make_train_env(cfg.env_id, reward_code, rank=i) for i in range(n_envs)]

    vec_type = str(cfg.vec_env_type).lower().strip()
    if vec_type == "subproc" and n_envs > 1:
        return SubprocVecEnv(env_fns, start_method="fork")
    return DummyVecEnv(env_fns)


class RLWorker:
    def __init__(self, cfg: RLConfig):
        self.cfg = cfg

    def train_and_eval(self, reward_code: str, ckpt_path: Path) -> dict[str, object]:
        reward_fn = compile_reward_function(reward_code)

        train_env = _make_vec_env(self.cfg, reward_code)
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_rate=self.cfg.learning_rate,
            gamma=self.cfg.gamma,
            n_steps=self.cfg.n_steps,
            batch_size=self.cfg.batch_size,
            n_epochs=self.cfg.n_epochs,
            gae_lambda=self.cfg.gae_lambda,
            ent_coef=self.cfg.ent_coef,
            clip_range=self.cfg.clip_range,
            vf_coef=self.cfg.vf_coef,
            max_grad_norm=self.cfg.max_grad_norm,
        )
        model.learn(total_timesteps=self.cfg.total_timesteps)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(ckpt_path))
        train_env.close()

        eval_env = RewardFunctionWrapper(gym.make(self.cfg.env_id), reward_fn)

        generated_returns = []
        hidden_returns = []
        episode_lengths = []
        component_returns: dict[str, list[float]] = {}
        action_values = []

        for _ in range(self.cfg.eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            gen_ret = 0.0
            hid_ret = 0.0
            ep_len = 0
            comp_sum: dict[str, float] = {}

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_values.append(np.asarray(action, dtype=float).reshape(-1))

                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = bool(terminated or truncated)

                gen_ret += float(reward)
                hid_ret += float(info.get("_hidden_env_reward", 0.0))
                ep_len += 1

                for k, v in info.get("_reward_components", {}).items():
                    comp_sum[k] = comp_sum.get(k, 0.0) + float(v)

            generated_returns.append(gen_ret)
            hidden_returns.append(hid_ret)
            episode_lengths.append(ep_len)

            for k, v in comp_sum.items():
                component_returns.setdefault(k, []).append(v)

        eval_env.close()

        action_arr = np.concatenate(action_values) if action_values else np.zeros((1,), dtype=float)
        component_mean = {k: float(np.mean(v)) for k, v in component_returns.items()}

        return {
            "eval_generated_return": float(np.mean(generated_returns)),
            "eval_hidden_return": float(np.mean(hidden_returns)),
            "eval_episode_length": float(np.mean(episode_lengths)),
            "component_returns": component_mean,
            "diagnostics": {
                "generated_private_gap": float(np.mean(generated_returns) - np.mean(hidden_returns)),
                "action_mean": float(np.mean(action_arr)),
                "action_std": float(np.std(action_arr)),
                "episode_length_mean": float(np.mean(episode_lengths)),
                "component_returns": component_mean,
                "n_envs": int(self.cfg.n_envs),
                "vec_env_type": str(self.cfg.vec_env_type),
                "n_steps": int(self.cfg.n_steps),
                "batch_size": int(self.cfg.batch_size),
                "n_epochs": int(self.cfg.n_epochs),
                "gae_lambda": float(self.cfg.gae_lambda),
                "gamma": float(self.cfg.gamma),
                "ent_coef": float(self.cfg.ent_coef),
            },
        }

    def render_rollout_video(self, ckpt_path: Path, video_path: Path) -> Path:
        model = PPO.load(str(ckpt_path))
        env = gym.make(self.cfg.env_id, render_mode="rgb_array")
        obs, _ = env.reset()
        done = False
        frames = []

        while not done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

        video_path.parent.mkdir(parents=True, exist_ok=True)
        if frames:
            imageio.mimsave(video_path, frames, fps=30)
        env.close()
        return video_path
PY

echo "[4/7] patch orchestrator.py render_video gate..."
python - <<'PY'
from pathlib import Path

p = Path("mvp/orchestrator.py")
s = p.read_text(encoding="utf-8")

old = '''                        try:
                            self.worker.render_rollout_video(ckpt, video)
                            judge_score, judge_reason, judge_details = self.judge.judge(
                                clean_interface,
                                train_result,
                                video,
                            )
                        except Exception as e:
                            judge_score = 0.0
                            judge_reason = f"visual_judge_error: {type(e).__name__}: {e}"
                            judge_details = {"error": str(e)}
'''

new = '''                        if self.cfg.rl.render_video:
                            try:
                                self.worker.render_rollout_video(ckpt, video)
                                judge_score, judge_reason, judge_details = self.judge.judge(
                                    clean_interface,
                                    train_result,
                                    video,
                                )
                            except Exception as e:
                                judge_score = 0.0
                                judge_reason = f"visual_judge_error: {type(e).__name__}: {e}"
                                judge_details = {"error": str(e)}
                        else:
                            judge_score = 0.0
                            judge_reason = "video_render_skipped"
                            judge_details = {"render_video": False}
'''

if old not in s:
    raise SystemExit("Could not find render video block. Please inspect mvp/orchestrator.py manually.")

s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
PY

echo "[5/7] add single-chain configs..."
mkdir -p mvp/configs

cat > mvp/configs/eg_rsa_lunar_deepseek_seed0_singlechain_g3_t300k.yaml <<'YAML'
model:
  provider: deepseek
  llm_model: deepseek-v4-flash
  vlm_model: deepseek-v4-flash
  deepseek_base_url: https://api.deepseek.com
  deepseek_api_key_env: DEEPSEEK_API_KEY
  deepseek_thinking: disabled
  temperature: 0.7
  max_tokens: 2500

rl:
  env_id: LunarLander-v3

  # Sanity run budget
  total_timesteps: 300000
  eval_episodes: 10

  # RL Zoo PPO LunarLander-v3 style hyperparameters
  n_envs: 16
  vec_env_type: dummy
  n_steps: 1024
  batch_size: 64
  gae_lambda: 0.98
  gamma: 0.999
  n_epochs: 4
  ent_coef: 0.01
  learning_rate: 0.0003
  clip_range: 0.2
  vf_coef: 0.5
  max_grad_norm: 0.5

  # Keep sanity run lighter. Turn on later if needed.
  render_video: false

evolution:
  generations: 3
  population_size: 1
  elite_size: 1
  reflection_top_k: 1
  target_score:
  max_stagnation_generations:

memory:
  candidate_lesson_top_k: 8
  env_lesson_top_k: 10
  ltm_lesson_top_k: 0
  parent_code_top_k: 1
  parent_code_max_chars: 12000
  feedback_max_chars: 12000
  memory_context_max_chars: 14000

workspace: runs/eg_rsa_lunar_deepseek_seed0_singlechain_g3_t300k
seed: 0
YAML

cat > mvp/configs/eg_rsa_lunar_deepseek_seed0_singlechain_g10_t2m.yaml <<'YAML'
model:
  provider: deepseek
  llm_model: deepseek-v4-flash
  vlm_model: deepseek-v4-flash
  deepseek_base_url: https://api.deepseek.com
  deepseek_api_key_env: DEEPSEEK_API_KEY
  deepseek_thinking: disabled
  temperature: 0.7
  max_tokens: 2500

rl:
  env_id: LunarLander-v3

  # Long run budget
  total_timesteps: 2000000
  eval_episodes: 10

  # RL Zoo PPO LunarLander-v3 style hyperparameters
  n_envs: 16
  vec_env_type: dummy
  n_steps: 1024
  batch_size: 64
  gae_lambda: 0.98
  gamma: 0.999
  n_epochs: 4
  ent_coef: 0.01
  learning_rate: 0.0003
  clip_range: 0.2
  vf_coef: 0.5
  max_grad_norm: 0.5

  render_video: false

evolution:
  generations: 10
  population_size: 1
  elite_size: 1
  reflection_top_k: 1
  target_score:
  max_stagnation_generations:

memory:
  candidate_lesson_top_k: 8
  env_lesson_top_k: 10
  ltm_lesson_top_k: 0
  parent_code_top_k: 1
  parent_code_max_chars: 12000
  feedback_max_chars: 12000
  memory_context_max_chars: 14000

workspace: runs/eg_rsa_lunar_deepseek_seed0_singlechain_g10_t2m
seed: 0
YAML

echo "[6/7] patch check_run_quality.py with single-chain summary..."
python - <<'PY'
from pathlib import Path

p = Path("scripts/check_run_quality.py")
s = p.read_text(encoding="utf-8")

insert = r'''
    # Single-chain / evolution trend summary.
    if memory_rows:
        by_gen = {}
        for r in memory_rows:
            try:
                g = int(r.get("generation", -1))
                score = float(r.get("hidden_eval_return", r.get("selection_score", -1e18)))
            except Exception:
                continue
            by_gen.setdefault(g, []).append(score)

        if by_gen:
            print("\nGeneration best private_eval_return:")
            prev_best = None
            best_so_far = None
            for g in sorted(by_gen):
                gen_best = max(by_gen[g])
                best_so_far = gen_best if best_so_far is None else max(best_so_far, gen_best)
                if prev_best is None:
                    delta = 0.0
                else:
                    delta = gen_best - prev_best
                print(f"- g{g}: gen_best={gen_best:.3f}, delta_vs_prev={delta:.3f}, best_so_far={best_so_far:.3f}")
                prev_best = gen_best
'''

anchor = '''    if warnings:
        print("\\nWARNINGS:")
'''

if "Generation best private_eval_return:" not in s:
    s = s.replace(anchor, insert + "\n" + anchor)

p.write_text(s, encoding="utf-8")
PY

echo "[7/7] syntax checks..."
python -m py_compile mvp/*.py scripts/check_run_quality.py scripts/check_eureka_step_input.py

echo ""
echo "PATCH DONE."
echo "Backup saved at: $backup_dir"
echo ""
echo "Next sanity run:"
echo "  rm -rf runs/eg_rsa_lunar_deepseek_seed0_singlechain_g3_t300k"
echo "  python run_mvp.py --config mvp/configs/eg_rsa_lunar_deepseek_seed0_singlechain_g3_t300k.yaml"
echo "  python scripts/check_run_quality.py runs/eg_rsa_lunar_deepseek_seed0_singlechain_g3_t300k"
echo ""
echo "If dummy is too slow and subprocess works on your machine, change vec_env_type: subproc."
