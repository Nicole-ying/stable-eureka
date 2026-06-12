from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from eg_rsa.diagnostics.event_evaluator import EventEvaluator
from eg_rsa.diagnostics.task_metrics import TaskMetricEvaluator
from eg_rsa.diagnostics.trajectory_recorder import TrajectoryRecorder
from eg_rsa.env_adapters.box_obs_adapter import BoxObsAdapter
from eg_rsa.env_adapters.action_primitive_mapper import ActionPrimitiveMapper
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

    def train_and_record(
        self,
        reward_schema: RewardSchema,
        init_model_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        n_envs = int(self.config["rl"]["training"].get("n_envs", 1))
        train_env = self._make_training_vec_env(reward_schema, n_envs)
        model = self._make_model(train_env, init_model_path=init_model_path)
        total_timesteps = int(self.config["rl"]["training"].get("total_timesteps", 100000))
        reset_num_timesteps = init_model_path is None
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=reset_num_timesteps)
        model_path = self.output_dir / "model.zip"
        model.save(model_path)
        train_env.close()

        eval_env = self._make_env(reward_schema)
        recorder = self._make_recorder()
        n_episodes = int(self.config["rl"].get("evaluation", {}).get("num_episodes", 5))
        seed = self.config["rl"].get("evaluation", {}).get("seed", None)
        trajectories = recorder.record_policy(model, eval_env, n_episodes=n_episodes, seed=seed)
        eval_env.close()
        TrajectoryRecorder.save_jsonl(self.output_dir / "trajectories.jsonl", trajectories)
        self._write_json(self.output_dir / "trajectories.json", trajectories)

        if bool(self.config.get("posthoc_eval", {}).get("enabled", True)):
            posthoc = PosthocEvaluator(
                env_id=self.config["environment"]["gym_id"],
                env_kwargs=self.config["environment"].get("kwargs") or {},
            )
            posthoc_episodes = int(self.config.get("posthoc_eval", {}).get("num_episodes", n_episodes))
            posthoc_seed = self.config.get("posthoc_eval", {}).get("seed", seed)
            posthoc_result = posthoc.evaluate_model(
                model,
                gym.make(self.config["environment"]["gym_id"], **(self.config["environment"].get("kwargs") or {})),
                n_episodes=posthoc_episodes,
                seed=posthoc_seed,
            )
            PosthocEvaluator.save(self.output_dir / "posthoc_eval.json", posthoc_result)

        self._write_json(
            self.output_dir / "training_metadata.json",
            {
                "init_model_path": str(init_model_path) if init_model_path else None,
                "continued_from_checkpoint": init_model_path is not None,
                "total_timesteps_added": total_timesteps,
                "saved_model_path": str(model_path),
            },
        )
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
        action_mapper = ActionPrimitiveMapper.from_runtime_spec(self.diagnostic_spec)
        return SchemaRewardWrapper(env, reward_schema, adapter, task_metric_evaluator, event_evaluator, action_mapper=action_mapper)

    def _make_training_vec_env(self, reward_schema: RewardSchema, n_envs: int = 1):
        if n_envs <= 1:
            return self._make_env(reward_schema)

        env_id = self.config["environment"].get("gym_id")
        if not env_id:
            raise ValueError("EG-RSA real training requires environment.gym_id")
        env_kwargs = self.config["environment"].get("kwargs") or {}

        def _make_wrapped_env():
            env = gym.make(env_id, **env_kwargs)
            adapter = self._make_adapter()
            event_evaluator = EventEvaluator(self.diagnostic_spec.get("events", {}))
            task_metric_evaluator = TaskMetricEvaluator(self.diagnostic_spec.get("task_metrics", {}))
            action_mapper = ActionPrimitiveMapper.from_runtime_spec(self.diagnostic_spec)
            return SchemaRewardWrapper(env, reward_schema, adapter, task_metric_evaluator, event_evaluator, action_mapper=action_mapper)

        return SubprocVecEnv([_make_wrapped_env for _ in range(n_envs)], start_method="fork")

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

    def _make_model(self, env, init_model_path: Optional[Path] = None):
        rl_cfg = self.config.get("rl", {})
        training_cfg = rl_cfg.get("training", {})
        if init_model_path is not None:
            init_model_path = Path(init_model_path)
            if not init_model_path.exists():
                raise FileNotFoundError(f"Continuation checkpoint not found: {init_model_path}")
            return PPO.load(
                str(init_model_path),
                env=env,
                device=training_cfg.get("device", "auto"),
            )

        algo_params = dict(rl_cfg.get("algo_params", {}))
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
