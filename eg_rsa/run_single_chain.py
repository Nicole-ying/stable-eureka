from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from eg_rsa.llm.client_factory import build_llm_client
from eg_rsa.single_chain.agents import JsonAgent
from eg_rsa.single_chain.expert_priors import get_expert_priors
from eg_rsa.single_chain.json_tools import read_text, write_json, write_text
from eg_rsa.single_chain.reward_guard import prepare_guarded_reward_env
from eg_rsa.single_chain.trainer import SingleChainPPOTrainer


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "eg_rsa" / "single_chain" / "prompts"


def make_run_dir(config: Dict[str, Any]) -> Path:
    exp_cfg = config.get("experiment", {}) or {}
    parent = ROOT / exp_cfg.get("parent", "experiments/eg_rsa_single_chain")
    name = exp_cfg.get("name", "lunar_lander_single_chain")
    if exp_cfg.get("use_datetime", True):
        name = f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = parent / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_inputs(config: Dict[str, Any], run_dir: Path) -> Dict[str, str]:
    env_cfg = config.get("environment", {}) or {}
    task_path = ROOT / env_cfg["task_description_path"]
    step_path = ROOT / env_cfg["step_path"]
    task_description = read_text(task_path)
    step_code = read_text(step_path)

    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    write_text(inputs_dir / "task_description.txt", task_description)
    write_text(inputs_dir / "step.py", step_code)
    return {"task_description": task_description, "step_code": step_code}


def as_json_text(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def run(config_path: str) -> Path:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = make_run_dir(config)
    shutil.copy(config_path, run_dir / "config.yaml")

    sandbox_inputs = copy_inputs(config, run_dir)
    llm_client = build_llm_client(config)
    if llm_client is None:
        raise RuntimeError("single_chain requires an LLM backend")

    prompts_out = run_dir / "agent_prompts"
    prompts_out.mkdir(parents=True, exist_ok=True)
    task_model_prompt = read_text(PROMPT_DIR / "task_model_prompt.txt")
    reward_designer_prompt = read_text(PROMPT_DIR / "expert_reward_designer_prompt.txt")
    repair_prompt = read_text(PROMPT_DIR / "reward_repair_prompt.txt")
    write_text(prompts_out / "task_model_prompt.txt", task_model_prompt)
    write_text(prompts_out / "expert_reward_designer_prompt.txt", reward_designer_prompt)
    write_text(prompts_out / "reward_repair_prompt.txt", repair_prompt)

    expert_priors = get_expert_priors()
    write_json(run_dir / "expert_reward_design_priors.json", expert_priors)

    agents_dir = run_dir / "agents"
    raw_dir = agents_dir / "raw_llm_outputs"

    task_model = JsonAgent("TaskModelAgent", llm_client, task_model_prompt).run(
        sandbox_inputs,
        agents_dir / "task_model.json",
        raw_dir / "task_model_raw.txt",
    )

    initial_reward = JsonAgent("ExpertRewardDesignerAgent", llm_client, reward_designer_prompt).run(
        {
            **sandbox_inputs,
            "task_model_json": as_json_text(task_model),
            "expert_reward_design_priors_json": as_json_text(expert_priors),
        },
        agents_dir / "initial_reward_schema_and_code.json",
        raw_dir / "initial_reward_schema_and_code_raw.txt",
    )

    expert_blueprint = initial_reward.get("expert_blueprint") or {}
    write_json(run_dir / "task_model.json", task_model)
    write_json(run_dir / "expert_reward_design_blueprint.json", expert_blueprint)

    # Compatibility outputs for older analysis utilities and iterative resume code.
    write_json(agents_dir / "environment_understanding.json", _environment_compat_from_task_model(task_model))
    write_json(agents_dir / "target_alignment_contract.json", _target_compat_from_task_model(task_model))

    reward_schema = initial_reward.get("reward_schema") or {}
    reward_code = initial_reward.get("reward_code") or ""
    env_cls, reward_schema, reward_code, guard_summary = prepare_guarded_reward_env(
        config=config,
        output_dir=run_dir,
        reward_schema=reward_schema,
        reward_code=reward_code,
        environment_understanding=task_model,
        target_alignment_contract=task_model,
        llm_client=llm_client,
        repair_dir=agents_dir / "reward_repairs",
    )

    trainer = SingleChainPPOTrainer(
        config=config,
        output_dir=run_dir / "training",
        env_cls=env_cls,
        environment_understanding=task_model,
        target_alignment_contract=task_model,
        reward_schema=reward_schema,
    )
    reward_trace = trainer.train()
    write_json(run_dir / "reward_trace.json", reward_trace)
    write_text(run_dir / "DONE.txt", "single-chain EG-RSA bootstrap run finished\n")
    return run_dir


def _environment_compat_from_task_model(task_model: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_type": "environment_understanding",
        "source": "task_model_compat",
        "environment_name": task_model.get("environment_name"),
        "task_goal": task_model.get("primary_objective"),
        "reward_function_interface": task_model.get("reward_function_interface"),
        "state_space": task_model.get("state_space_summary"),
        "action_space": task_model.get("action_space_summary"),
        "termination_modes": task_model.get("failure_or_stop_signals_available_to_reward"),
        "success_like_ending": task_model.get("success_signals_available_to_reward"),
        "failure_like_endings": task_model.get("failure_or_stop_signals_available_to_reward"),
        "behavior_trajectory_prior": task_model.get("intended_behavior_phases"),
        "reward_design_risks": task_model.get("dangerous_local_optima"),
    }


def _target_compat_from_task_model(task_model: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_type": "target_alignment_contract",
        "source": "task_model_compat",
        "environment_name": task_model.get("environment_name"),
        "primary_objective": task_model.get("primary_objective"),
        "objective_decomposition": task_model.get("intended_behavior_phases"),
        "behavior_trajectory_alignment": task_model.get("intended_behavior_phases"),
        "metric_priority": task_model.get("primary_selection_metric"),
        "primary_selection_metric": task_model.get("primary_selection_metric", "fitness_score"),
        "proxy_metrics": task_model.get("proxy_metrics"),
        "diagnostic_metrics": task_model.get("diagnostic_metrics"),
        "success_criteria": task_model.get("success_signals_available_to_reward"),
        "failure_modes": task_model.get("dangerous_local_optima"),
        "bad_local_optima": task_model.get("dangerous_local_optima"),
        "reward_generator_constraints": task_model.get("reward_design_constraints"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EG-RSA single-chain bootstrap pipeline")
    parser.add_argument("--config", required=True, help="Path to EG-RSA single-chain YAML config")
    args = parser.parse_args()
    run_dir = run(args.config)
    print(f"EG-RSA single-chain run finished: {run_dir}")


if __name__ == "__main__":
    main()
