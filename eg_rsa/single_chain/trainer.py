from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .env_builder import make_single_env
from .json_tools import write_json
from .target_behavior_evaluator import build_checkpoint_stability_report, build_target_behavior_report


INTERNAL_INFO_KEYS = {
    "TimeLimit.truncated",
    "terminal_observation",
    "episode",
    "_eg_episode_step",
    "_eg_action",
    "_eg_done",
    "_eg_final_state",
}


class SingleChainEvalCallback(BaseCallback):
    def __init__(self, trainer: "SingleChainPPOTrainer", eval_freq: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.trainer = trainer
        self.eval_freq = max(1, int(eval_freq))
        self.history: Dict[str, List[Any]] = defaultdict(list)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        metrics, trajectories = self.trainer.evaluate(self.model)
        metrics["timesteps"] = int(self.num_timesteps)
        for key, value in metrics.items():
            if isinstance(value, dict):
                continue
            self.history[key].append(value)
        self.history["component_means"].append(metrics.get("component_means", {}))
        self.history["action_distribution"].append(metrics.get("action_distribution", {}))
        self.history["action_space_report"].append(metrics.get("action_space_report", {}))
        write_json(self.trainer.output_dir / "evals.json", dict(self.history))
        write_json(self.trainer.output_dir / "trajectory_summary_last_eval.json", {"episodes": trajectories})
        write_json(self.trainer.output_dir / "checkpoint_stability_report.json", build_checkpoint_stability_report(dict(self.history)))
        return True


class SingleChainPPOTrainer:
    """Train one generated reward function and record EG-RSA bootstrap traces."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str | Path,
        env_cls: Type[gym.Env],
        environment_understanding: Dict[str, Any],
        target_alignment_contract: Dict[str, Any],
        reward_schema: Dict[str, Any],
        candidate_id: str = "single_chain_iter0",
        generation: int = 0,
        creation_type: str = "initial_llm_generation",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_cls = env_cls
        self.environment_understanding = environment_understanding
        self.target_alignment_contract = target_alignment_contract
        self.reward_schema = reward_schema
        self.candidate_id = candidate_id
        self.generation = int(generation)
        self.creation_type = creation_type

        self.env_cfg = config.get("environment", {})
        self.rl_cfg = config.get("rl", {})
        self.train_cfg = self.rl_cfg.get("training", {})
        self.eval_cfg = self.rl_cfg.get("eval", {})
        controller_cfg = config.get("controller", {}) or {}
        search_cfg = config.get("search", {}) or {}
        self.primary_metric = controller_cfg.get("primary_metric") or search_cfg.get("primary_metric") or "fitness_score"

    def _make_vec_env(self, n_envs: int, seed: int | None) -> DummyVecEnv:
        env_kwargs = self.env_cfg.get("kwargs") or {}
        max_episode_steps = self.env_cfg.get("max_episode_steps")

        def make_fn(rank: int):
            def _init():
                rank_seed = None if seed is None else int(seed) + rank
                return make_single_env(self.env_cls, env_kwargs, max_episode_steps, rank_seed)
            return _init

        return DummyVecEnv([make_fn(i) for i in range(int(n_envs))])

    @staticmethod
    def _linear_schedule(initial_value: float) -> Any:
        def schedule(progress_remaining: float) -> float:
            return float(progress_remaining) * float(initial_value)
        return schedule

    def _ppo_kwargs(self, train_env: DummyVecEnv) -> Dict[str, Any]:
        algo_params = self.rl_cfg.get("algo_params", {})
        architecture = self.rl_cfg.get("architecture") or {}
        use_schedule = bool(self.train_cfg.get("use_linear_schedule", False))
        policy_kwargs = None
        if architecture:
            policy_kwargs = {
                "activation_fn": getattr(torch.nn, architecture.get("activation_fn", "ReLU")),
                "net_arch": architecture.get("net_arch", None),
                "share_features_extractor": architecture.get("share_features_extractor", False),
            }

        lr = algo_params.get("learning_rate", 3e-4)
        clip = algo_params.get("clip_range", 0.2)
        ent = algo_params.get("ent_coef", 0.0)
        if use_schedule:
            lr = self._linear_schedule(lr)
            clip = self._linear_schedule(clip)
            # ent_coef must stay a float — SB3 multiplies it directly with tensors

        return {
            "policy": algo_params.get("policy", "MlpPolicy"),
            "env": train_env,
            "policy_kwargs": policy_kwargs,
            "learning_rate": lr,
            "n_steps": algo_params.get("n_steps", 2048),
            "batch_size": algo_params.get("batch_size", 64),
            "n_epochs": algo_params.get("n_epochs", 10),
            "gamma": algo_params.get("gamma", 0.99),
            "gae_lambda": algo_params.get("gae_lambda", 0.95),
            "clip_range": clip,
            "ent_coef": ent,
            "vf_coef": algo_params.get("vf_coef", 0.5),
            "max_grad_norm": algo_params.get("max_grad_norm", 0.5),
            "seed": self.train_cfg.get("seed", None),
            "device": self.train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            "tensorboard_log": str(self.output_dir / "tensorboard"),
            "verbose": int(self.train_cfg.get("verbose", 1)),
        }

    def train(self) -> Dict[str, Any]:
        if (self.rl_cfg.get("algo") or "ppo").lower() != "ppo":
            raise ValueError("single_chain trainer currently supports PPO only")

        n_envs = int(self.train_cfg.get("num_envs", 4))
        seed = self.train_cfg.get("seed", 0)
        total_timesteps = int(self.train_cfg.get("total_timesteps", 200000))
        num_evals = int(self.eval_cfg.get("num_evals", 5))
        eval_freq = max(1, total_timesteps // max(1, n_envs) // max(1, num_evals))

        train_env = self._make_vec_env(n_envs=n_envs, seed=seed)
        model = PPO(**self._ppo_kwargs(train_env))

        callback = SingleChainEvalCallback(self, eval_freq=eval_freq)
        model.learn(total_timesteps=total_timesteps, tb_log_name="single_chain", callback=callback)
        model.save(self.output_dir / "final_model")
        model.save(self.output_dir / "model")

        final_metrics, final_trajectories = self.evaluate(model)
        write_json(self.output_dir / "final_eval.json", final_metrics)
        write_json(self.output_dir / "trajectory_summary.json", {"episodes": final_trajectories})

        eval_history = dict(callback.history)
        checkpoint_stability = build_checkpoint_stability_report(eval_history)
        target_behavior_report = build_target_behavior_report(
            metrics=final_metrics,
            trajectories=final_trajectories,
            max_episode_steps=self.env_cfg.get("max_episode_steps"),
            fitness_score_auxiliary=final_metrics.get("fitness_score"),
            target_alignment_contract=self.target_alignment_contract or self.environment_understanding,
        )
        model_selection = self._final_model_selection(final_metrics, checkpoint_stability)
        write_json(self.output_dir / "checkpoint_stability_report.json", checkpoint_stability)
        write_json(self.output_dir / "target_behavior_report.json", target_behavior_report)
        write_json(self.output_dir / "model_selection.json", model_selection)

        final_metrics_for_selected = dict(final_metrics)
        final_metrics_for_selected["selected_model_source"] = "final_checkpoint"
        final_metrics_for_selected["model_selection_metric"] = self.primary_metric
        final_metrics_for_selected["model_selection_score"] = float(final_metrics.get(self.primary_metric, final_metrics.get("fitness_score", 0.0)))
        write_json(self.output_dir / "selected_eval.json", final_metrics_for_selected)
        write_json(self.output_dir / "selected_trajectory_summary.json", {"episodes": final_trajectories})

        component_metrics = {
            "component_means": final_metrics.get("component_means", {}),
            "component_keys": sorted(final_metrics.get("component_means", {}).keys()),
            "note": "Component means are final-checkpoint episode-level sums from info returned by compute_reward. Peak checkpoints are diagnostics only.",
        }
        write_json(self.output_dir / "component_metrics.json", component_metrics)

        reward_trace = self.build_reward_trace(final_metrics, final_trajectories)
        reward_trace["model_selection"] = model_selection
        reward_trace["target_behavior_report"] = target_behavior_report
        reward_trace["checkpoint_stability_report"] = checkpoint_stability
        write_json(self.output_dir / "reward_trace.json", reward_trace)
        train_env.close()
        return reward_trace

    def _final_model_selection(self, final_metrics: Dict[str, Any], checkpoint_stability: Dict[str, Any]) -> Dict[str, Any]:
        final_score = float(final_metrics.get(self.primary_metric, final_metrics.get("fitness_score", 0.0)))
        return {
            "file_type": "model_selection",
            "primary_metric": self.primary_metric,
            "selected_model_source": "final_checkpoint",
            "selected_score": final_score,
            "final_score": final_score,
            "used_best_checkpoint": False,
            "peak_checkpoint_is_diagnostic_only": True,
            "checkpoint_stability_report": checkpoint_stability,
        }

    def evaluate(self, model: PPO) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        n_eval_episodes = int(self.eval_cfg.get("n_eval_episodes", 5))
        deterministic = bool(self.eval_cfg.get("deterministic", True))
        seed = self.eval_cfg.get("seed", 1000)
        eval_env = self._make_vec_env(n_envs=1, seed=seed)

        rewards: List[float] = []
        fitness_scores: List[float] = []
        lengths: List[int] = []
        info_sums_per_episode: List[Dict[str, float]] = []
        trajectories: List[Dict[str, Any]] = []
        action_counter: Counter[str] = Counter()

        for ep in range(n_eval_episodes):
            obs = eval_env.reset()
            done = [False]
            episode_reward = 0.0
            episode_length = 0
            info_sums: Dict[str, float] = defaultdict(float)
            final_state = None
            final_info: Dict[str, Any] = {}
            ep_action_counter: Counter[str] = Counter()

            while not done[0]:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, infos = eval_env.step(action)
                info = dict(infos[0])
                r = float(np.asarray(reward).reshape(-1)[0])
                episode_reward += r
                episode_length += 1

                action_value = info.get("_eg_action", _action_to_key(action))
                action_key = json.dumps(action_value, sort_keys=True)
                action_counter[action_key] += 1
                ep_action_counter[action_key] += 1

                for key, value in info.items():
                    if key in INTERNAL_INFO_KEYS:
                        continue
                    if _is_number(value):
                        info_sums[key] += float(value)

                if bool(done[0]):
                    final_info = info
                    if "_eg_final_state" in info:
                        final_state = info["_eg_final_state"]
                    elif "terminal_observation" in info:
                        final_state = np.asarray(info["terminal_observation"], dtype=float).tolist()
                    else:
                        final_state = np.asarray(obs[0], dtype=float).tolist()

            fitness_score = float(info_sums.get("fitness_score", 0.0))
            rewards.append(float(episode_reward))
            fitness_scores.append(fitness_score)
            lengths.append(int(episode_length))
            info_sums_per_episode.append(dict(info_sums))
            terminal_classification = _classify_terminal(info_sums, final_info)
            trajectories.append(
                {
                    "episode_id": ep,
                    "length": int(episode_length),
                    "return_generated": float(episode_reward),
                    "return_fitness": fitness_score,
                    "final_state": final_state,
                    "terminal_classification": terminal_classification,
                    "component_returns": dict(sorted(info_sums.items())),
                    "action_histogram": dict(ep_action_counter),
                }
            )

        component_means: Dict[str, float] = {}
        all_keys = sorted({key for ep_info in info_sums_per_episode for key in ep_info.keys()})
        for key in all_keys:
            component_means[key] = float(np.mean([ep_info.get(key, 0.0) for ep_info in info_sums_per_episode]))

        total_actions = sum(action_counter.values()) or 1
        action_distribution = {key: float(value / total_actions) for key, value in action_counter.items()}
        action_space_report = _action_space_report(eval_env.action_space, action_distribution)

        success_rate = float(np.mean([t["terminal_classification"] == "success_like_terminal" for t in trajectories]))
        unsafe_rate = float(np.mean([t["terminal_classification"] == "unsafe_terminal" for t in trajectories]))
        out_of_bounds_rate = float(np.mean([t["terminal_classification"] == "out_of_bounds" for t in trajectories]))

        metrics = {
            "reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "fitness_score": float(np.mean(fitness_scores)) if fitness_scores else 0.0,
            "fitness_score_std": float(np.std(fitness_scores)) if fitness_scores else 0.0,
            "episode_length": float(np.mean(lengths)),
            "episode_length_std": float(np.std(lengths)),
            "component_means": component_means,
            "action_distribution": action_distribution,
            "action_space_report": action_space_report,
            "success_like_terminal_rate": success_rate,
            "unsafe_terminal_rate": unsafe_rate,
            "out_of_bounds_rate": out_of_bounds_rate,
        }
        eval_env.close()
        return metrics, trajectories

    def build_reward_trace(self, metrics: Dict[str, Any], trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        component_means = metrics.get("component_means", {})
        generated = float(metrics.get("reward", 0.0))
        fitness = float(metrics.get("fitness_score", 0.0))
        gap = generated - fitness
        episode_length = float(metrics.get("episode_length", 0.0))
        max_steps = float(self.env_cfg.get("max_episode_steps") or 0)

        hovering_risk = bool(max_steps and episode_length >= 0.9 * max_steps and metrics.get("success_like_terminal_rate", 0.0) < 0.2)
        proxy_hacking_risk = bool(generated > 0 and metrics.get("success_like_terminal_rate", 0.0) <= 0.0 and hovering_risk)
        terminal_hacking_risk = bool(
            component_means.get("terminal_reward", 0.0) > 0
            and metrics.get("success_like_terminal_rate", 0.0) < 0.2
        )

        return {
            "file_type": "reward_trace",
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "creation_type": self.creation_type,
            "primary_metrics": {
                "fitness_score": fitness,
                "selection_metric": "fitness_score_auxiliary",
            },
            "proxy_metrics": {
                "generated_reward": generated,
                "generated_minus_fitness_gap": gap,
            },
            "behavior_metrics": {
                "episode_length": episode_length,
                "success_like_terminal_rate": metrics.get("success_like_terminal_rate", 0.0),
                "unsafe_terminal_rate": metrics.get("unsafe_terminal_rate", 0.0),
                "out_of_bounds_rate": metrics.get("out_of_bounds_rate", 0.0),
                "action_distribution": metrics.get("action_distribution", {}),
                "action_space_report": metrics.get("action_space_report", {}),
            },
            "component_returns": component_means,
            "alignment_flags": {
                "hovering_risk": hovering_risk,
                "proxy_reward_behavior_misalignment_risk": proxy_hacking_risk,
                "terminal_hacking_risk": terminal_hacking_risk,
            },
            "environment_understanding_summary": self.environment_understanding.get("task_goal", {}),
            "target_alignment_summary": self.target_alignment_contract.get("primary_objective", {}),
            "reward_schema_summary": self.reward_schema,
            "trajectory_examples": trajectories,
        }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating, bool))


def _action_to_key(action: Any) -> Any:
    arr = np.asarray(action)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.astype(float).tolist()


def _action_space_report(action_space: gym.Space, action_distribution: Dict[str, float]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "space_type": type(action_space).__name__,
        "used_action_keys": sorted(action_distribution.keys()),
    }
    if isinstance(action_space, gym.spaces.Discrete):
        start = int(getattr(action_space, "start", 0))
        expected = [json.dumps(start + i, sort_keys=True) for i in range(int(action_space.n))]
        report.update({
            "n": int(action_space.n),
            "start": start,
            "expected_action_keys": expected,
            "unused_action_keys": [key for key in expected if key not in action_distribution],
            "coverage_ratio": float(len([key for key in expected if key in action_distribution]) / max(1, len(expected))),
        })
    return report


def _classify_terminal(info_sums: Dict[str, float], final_info: Dict[str, Any]) -> str:
    if info_sums.get("success_like_terminal", 0.0) > 0 or info_sums.get("safe_landing_terminal", 0.0) > 0:
        return "success_like_terminal"
    if info_sums.get("out_of_bounds", 0.0) > 0 or info_sums.get("out_of_bounds_terminal", 0.0) > 0:
        return "out_of_bounds"
    if info_sums.get("unsafe_terminal", 0.0) > 0 or info_sums.get("crash_terminal", 0.0) > 0:
        return "unsafe_terminal"
    if final_info.get("TimeLimit.truncated", False) or info_sums.get("timeout_terminal", 0.0) > 0:
        return "timeout_or_truncated"
    return "unknown_terminal"
