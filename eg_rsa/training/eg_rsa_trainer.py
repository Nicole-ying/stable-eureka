from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import yaml
from stable_baselines3 import PPO

from eg_rsa.diagnostics.event_evaluator import EventEvaluator
from eg_rsa.diagnostics.task_metrics import TaskMetricEvaluator
from eg_rsa.diagnostics.trajectory_recorder import TrajectoryRecorder
from eg_rsa.env_adapters.box_obs_adapter import BoxObsAdapter
from eg_rsa.evaluation.posthoc_evaluator import PosthocEvaluator
from eg_rsa.reward.schema import RewardSchema
from eg_rsa.training.schema_reward_wrapper import SchemaRewardWrapper


class EGRSATrainer:
    """Train a policy with a schema reward and record real trajectories."""

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        diag_path = Path(config["eg_rsa"]["diagnostic_spec_path"])
        self.diagnostic_spec = yaml.safe_load(diag_path.read_text(encoding="utf-8"))

    def train_and_record(self, reward_schema: RewardSchema) -> List[Dict[str, Any]]:
        env = self._make_env(reward_schema)
        model = self._make_model(env)
        total_timesteps = int(self.config["rl"]["training"].get("total_timesteps", 100000))
        model.learn(total_timesteps=total_timesteps)
        model_path = self.output_dir / "model"
        model.save(model_path)

        eval_env = self._make_env(reward_schema)
        recorder = self._make_recorder()
        n_episodes = int(self.config["rl"].get("evaluation", {}).get("num_episodes", 5))
        seed = self.config["rl"].get("evaluation", {}).get("seed", None)
        trajectories = recorder.record_policy(model, eval_env, n_episodes=n_episodes, seed=seed)
        TrajectoryRecorder.save_jsonl(self.output_dir / "trajectories.jsonl", trajectories)
        self._write_json(self.output_dir / "trajectories.json", trajectories)

        if bool(self.config.get("posthoc_eval", {}).get("enabled", True)):
            posthoc = PosthocEvaluator(
                env_id=self.config["environment"]["gym_id"],
                env_kwargs=self.config["environment"].get("kwargs") or {},
            )
            posthoc_episodes = int(self.config.get("posthoc_eval", {}).get("num_episodes", n_episodes))
            posthoc_seed = self.config.get("posthoc_eval", {}).get("seed", seed)
            posthoc_result = posthoc.evaluate_model(model, gym.make(self.config["environment"]["gym_id"], **(self.config["environment"].get("kwargs") or {})), n_episodes=posthoc_episodes, seed=posthoc_seed)
            PosthocEvaluator.save(self.output_dir / "posthoc_eval.json", posthoc_result)

        return trajectories

    def _make_env(self, reward_schema: RewardSchema):
        env_id = self.config["environment"].get("gym_id")
        if not env_id:
            raise ValueError("EG-RSA real training requires environment.gym_id, e.g. LunarLander-v3")
        env_kwargs = self.config["environment"].get("kwargs") or {}
        env = gym.make(env_id, **env_kwargs)
        adapter = self._make_adapter()
        event_evaluator = EventEvaluator(self.diagnostic_spec.get("events", {}))
        task_metric_evaluator = TaskMetricEvaluator(self.diagnostic_spec.get("task_metrics", {}))
        return SchemaRewardWrapper(env, reward_schema, adapter, task_metric_evaluator, event_evaluator)

    def _make_adapter(self) -> BoxObsAdapter:
        mapping = self.diagnostic_spec.get("observation_mapping", {})
        if not mapping:
            raise ValueError("diagnostic spec must contain observation_mapping")
        return BoxObsAdapter(mapping)

    def _make_recorder(self) -> TrajectoryRecorder:
        adapter = self._make_adapter()
        event_evaluator = EventEvaluator(self.diagnostic_spec.get("events", {}))
        task_metric_evaluator = TaskMetricEvaluator(self.diagnostic_spec.get("task_metrics", {}))
        return TrajectoryRecorder(adapter, task_metric_evaluator, event_evaluator)

    def _make_model(self, env):
        rl_cfg = self.config.get("rl", {})
        algo_params = dict(rl_cfg.get("algo_params", {}))
        training_cfg = rl_cfg.get("training", {})
        policy = algo_params.pop("policy", "MlpPolicy")
        return PPO(
            policy=policy,
            env=env,
            seed=training_cfg.get("seed", None),
            device=training_cfg.get("device", "auto"),
            verbose=int(training_cfg.get("verbose", 0)),
            **algo_params,
        )

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
