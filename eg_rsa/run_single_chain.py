from __future__ import annotations

import argparse
import json
import re
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
        raise RuntimeError("single_chain requires an LLM backend")

    prompts_out = run_dir / "agent_prompts"
    prompts_out.mkdir(parents=True, exist_ok=True)
    task_model_prompt = read_text(PROMPT_DIR / "task_model_prompt.txt")
    reward_designer_prompt = read_text(PROMPT_DIR / "expert_reward_designer_prompt.txt")
    repair_prompt = read_text(PROMPT_DIR / "reward_repair_prompt.txt")
    write_text(prompts_out / "task_model_prompt.txt", task_model_prompt)
    write_text(prompts_out / "expert_reward_designer_prompt.txt", reward_designer_prompt)
    write_text(prompts_out / "reward_repair_prompt.txt", repair_prompt)

    agents_dir = run_dir / "agents"
    raw_dir = agents_dir / "raw_llm_outputs"

    # ---- LLM #1: TaskModelAgent — Markdown reference document ----
    task_model_prompt_text = JsonAgent("TaskModelAgent", llm_client, task_model_prompt).build_prompt(sandbox_inputs)
    task_model_md = llm_client.generate(task_model_prompt_text)
    write_text(raw_dir / "task_model_raw.txt", task_model_md)

    task_model_path_md = agents_dir / "task_model.md"
    write_text(task_model_path_md, task_model_md)
    write_text(run_dir / "task_model.md", task_model_md)

    task_model_compat = {"file_type": "task_model", "markdown_text": task_model_md}
    write_json(agents_dir / "task_model.json", task_model_compat)
    write_json(run_dir / "task_model.json", task_model_compat)

    # ---- LLM #2: ExpertRewardDesignerAgent — Markdown with code blocks ----
    designer_prompt_text = reward_designer_prompt.replace("{{task_model_md}}", task_model_md)
    designer_md = llm_client.generate(designer_prompt_text)
    write_text(raw_dir / "initial_reward_design_raw.txt", designer_md)

    # Save full design document.
    write_text(agents_dir / "expert_reward_design.md", designer_md)
    write_text(run_dir / "expert_reward_design.md", designer_md)

    # Extract code blocks from Markdown.
    reward_code = _extract_code_block(designer_md, "python") or ""
    reward_schema_raw = _extract_code_block(designer_md, "json") or "[]"
    try:
        reward_schema = json.loads(reward_schema_raw)
    except (json.JSONDecodeError, TypeError):
        reward_schema = []

    # Compatibility outputs for downstream consumers.
    write_json(agents_dir / "environment_understanding.json", _environment_compat_from_task_model(task_model_compat))
    write_json(agents_dir / "target_alignment_contract.json", _target_compat_from_task_model(task_model_compat))
    write_json(agents_dir / "initial_reward_schema_and_code.json", {
        "reward_schema": reward_schema,
        "reward_code": reward_code,
        "expert_blueprint": {"markdown_text": designer_md},
    })
    write_json(run_dir / "expert_reward_design_blueprint.json", {"markdown_text": designer_md})
    env_cls, reward_schema, reward_code, guard_summary = prepare_guarded_reward_env(
        config=config,
        output_dir=run_dir,
        reward_schema=reward_schema,
        reward_code=reward_code,
        environment_understanding=task_model_compat,
        target_alignment_contract=task_model_compat,
        llm_client=llm_client,
        repair_dir=agents_dir / "reward_repairs",
    )

    trainer = SingleChainPPOTrainer(
        config=config,
        output_dir=run_dir / "training",
        env_cls=env_cls,
        environment_understanding=task_model_compat,
        target_alignment_contract=task_model_compat,
        reward_schema=reward_schema,
    )
    reward_trace = trainer.train()
    write_json(run_dir / "reward_trace.json", reward_trace)
    write_text(run_dir / "DONE.txt", "single-chain EG-RSA bootstrap run finished\n")
    return run_dir


def _extract_code_block(markdown: str, language: str) -> str | None:
    """Extract the first fenced code block of the given language from Markdown."""
    pattern = rf"```{language}\s*\n(.*?)```"
    match = re.search(pattern, markdown, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try any ```<lang> block when language is "python"
    if language == "python":
        match = re.search(r"```python\s*\n(.*?)```", markdown, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def _environment_compat_from_task_model(task_model: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_type": "environment_understanding",
        "source": "task_model_compat",
        "markdown_text": task_model.get("markdown_text", ""),
    }


def _target_compat_from_task_model(task_model: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_type": "target_alignment_contract",
        "source": "task_model_compat",
        "markdown_text": task_model.get("markdown_text", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EG-RSA single-chain bootstrap pipeline")
    parser.add_argument("--config", required=True, help="Path to EG-RSA single-chain YAML config")
    args = parser.parse_args()
    run_dir = run(args.config)
    print(f"EG-RSA single-chain run finished: {run_dir}")


if __name__ == "__main__":
    main()
