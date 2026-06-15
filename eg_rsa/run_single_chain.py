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
        raise RuntimeError("single_chain requires an LLM backend: eg_rsa.edit_agent.backend must be ollama or deepseek")

    prompts_out = run_dir / "agent_prompts"
    prompts_out.mkdir(parents=True, exist_ok=True)
    env_prompt = read_text(PROMPT_DIR / "environment_understanding_prompt.txt")
    target_prompt = read_text(PROMPT_DIR / "target_alignment_prompt.txt")
    reward_prompt = read_text(PROMPT_DIR / "initial_reward_schema_code_prompt.txt")
    write_text(prompts_out / "environment_understanding_prompt.txt", env_prompt)
    write_text(prompts_out / "target_alignment_prompt.txt", target_prompt)
    write_text(prompts_out / "initial_reward_schema_code_prompt.txt", reward_prompt)
    write_text(prompts_out / "reward_repair_prompt.txt", read_text(PROMPT_DIR / "reward_repair_prompt.txt"))

    agents_dir = run_dir / "agents"
    raw_dir = agents_dir / "raw_llm_outputs"

    environment_understanding = JsonAgent("EnvironmentUnderstandingAgent", llm_client, env_prompt).run(
        sandbox_inputs,
        agents_dir / "environment_understanding.json",
        raw_dir / "environment_understanding_raw.txt",
    )

    target_alignment_contract = JsonAgent("TargetAlignmentAgent", llm_client, target_prompt).run(
        {
            **sandbox_inputs,
            "environment_understanding_json": as_json_text(environment_understanding),
        },
        agents_dir / "target_alignment_contract.json",
        raw_dir / "target_alignment_contract_raw.txt",
    )

    initial_reward = JsonAgent("InitialRewardSchemaAndCodeAgent", llm_client, reward_prompt).run(
        {
            **sandbox_inputs,
            "environment_understanding_json": as_json_text(environment_understanding),
            "target_alignment_contract_json": as_json_text(target_alignment_contract),
        },
        agents_dir / "initial_reward_schema_and_code.json",
        raw_dir / "initial_reward_schema_and_code_raw.txt",
    )

    reward_schema = initial_reward.get("reward_schema") or {}
    reward_code = initial_reward.get("reward_code") or ""
    env_cls, reward_schema, reward_code, guard_summary = prepare_guarded_reward_env(
        config=config,
        output_dir=run_dir,
        reward_schema=reward_schema,
        reward_code=reward_code,
        environment_understanding=environment_understanding,
        target_alignment_contract=target_alignment_contract,
        llm_client=llm_client,
        repair_dir=agents_dir / "reward_repairs",
    )

    trainer = SingleChainPPOTrainer(
        config=config,
        output_dir=run_dir / "training",
        env_cls=env_cls,
        environment_understanding=environment_understanding,
        target_alignment_contract=target_alignment_contract,
        reward_schema=reward_schema,
    )
    reward_trace = trainer.train()
    write_json(run_dir / "reward_trace.json", reward_trace)
    write_text(run_dir / "DONE.txt", "single-chain EG-RSA bootstrap run finished\n")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EG-RSA single-chain bootstrap pipeline")
    parser.add_argument("--config", required=True, help="Path to EG-RSA single-chain YAML config")
    args = parser.parse_args()
    run_dir = run(args.config)
    print(f"EG-RSA single-chain run finished: {run_dir}")


if __name__ == "__main__":
    main()
