from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from eg_rsa.llm.client_factory import build_llm_client
from eg_rsa.run_single_chain import run as run_bootstrap
from eg_rsa.single_chain.agents import JsonAgent
from eg_rsa.single_chain.env_builder import load_env_class, prepare_env_code
from eg_rsa.single_chain.evidence import EvidenceBuilder
from eg_rsa.single_chain.json_tools import read_json, read_text, write_json, write_text
from eg_rsa.single_chain.memory_manager import MemoryManager
from eg_rsa.single_chain.reward_validation import write_reward_code_files
from eg_rsa.single_chain.trainer import SingleChainPPOTrainer


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "eg_rsa" / "single_chain" / "prompts"


def as_json_text(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def run_iterative(config_path: str, bootstrap_run_dir: Optional[str] = None) -> Path:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if bootstrap_run_dir:
        run_dir = Path(bootstrap_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"bootstrap run dir not found: {run_dir}")
    else:
        run_dir = run_bootstrap(str(config_path))

    shutil.copy(config_path, run_dir / "iterative_config.yaml")

    llm_client = build_llm_client(config)
    if llm_client is None:
        raise RuntimeError("iterative single-chain search requires an LLM backend")

    search_cfg = config.get("search", {}) or {}
    total_iterations = int(search_cfg.get("iterations", 2))
    if total_iterations < 1:
        total_iterations = 1

    memory_cfg = config.get("memory", {}) or {}
    memory_path = run_dir / memory_cfg.get("path", "memory/reward_memory.jsonl")
    memory = MemoryManager(memory_path, top_k=int(memory_cfg.get("retrieve_top_k", 5)))

    env_understanding = read_json(run_dir / "agents" / "environment_understanding.json")
    target_contract = read_json(run_dir / "agents" / "target_alignment_contract.json")
    env_summary = _environment_summary(env_understanding)
    target_summary = _target_summary(target_contract)
    write_json(run_dir / "environment_understanding_summary.json", env_summary)
    write_json(run_dir / "target_alignment_summary.json", target_summary)

    current_dir = run_dir
    current_evidence = EvidenceBuilder.build(current_dir, output_path=current_dir / "iteration_evidence.json")
    best_evidence: Dict[str, Any] = current_evidence
    write_json(run_dir / "best_iteration_evidence.json", best_evidence)

    for next_iteration in range(1, total_iterations):
        search_dir = current_dir / "search"
        search_dir.mkdir(parents=True, exist_ok=True)

        current_evidence = EvidenceBuilder.build(current_dir, output_path=current_dir / "iteration_evidence.json")
        retrieved = memory.retrieve(current_evidence)
        write_json(search_dir / "retrieved_memory.json", {"items": retrieved})

        current_reward_schema = read_json(current_dir / "reward" / "reward_schema.json")
        current_reward_code = read_text(current_dir / "reward" / "reward_code.py")

        reflection_prompt = read_text(PROMPT_DIR / "reflection_prompt.txt")
        reflection = JsonAgent("ReflectionAgent", llm_client, reflection_prompt).run(
            {
                "environment_understanding_summary_json": as_json_text(env_summary),
                "target_alignment_contract_summary_json": as_json_text(target_summary),
                "current_reward_schema_json": as_json_text(current_reward_schema),
                "current_reward_code": current_reward_code,
                "iteration_evidence_json": as_json_text(current_evidence),
                "best_iteration_evidence_json": as_json_text(best_evidence),
                "retrieved_memory_json": as_json_text({"items": retrieved}),
            },
            search_dir / "reflection_decision.json",
            search_dir / "reflection_raw.txt",
        )

        if reflection.get("search_decision", {}).get("recommended_next_action") == "stop_success":
            write_text(run_dir / "ITERATIVE_DONE.txt", f"Stopped at iteration {next_iteration - 1}: stop_success\n")
            break

        retrieved_after_reflection = memory.retrieve(current_evidence, reflection)
        write_json(search_dir / "retrieved_memory_after_reflection.json", {"items": retrieved_after_reflection})

        revision_prompt = read_text(PROMPT_DIR / "reward_revision_prompt.txt")
        revision = JsonAgent("RewardRevisionAgent", llm_client, revision_prompt).run(
            {
                "environment_understanding_summary_json": as_json_text(env_summary),
                "target_alignment_contract_summary_json": as_json_text(target_summary),
                "current_reward_schema_json": as_json_text(current_reward_schema),
                "current_reward_code": current_reward_code,
                "iteration_evidence_json": as_json_text(current_evidence),
                "reflection_decision_json": as_json_text(reflection),
                "retrieved_memory_json": as_json_text({"items": retrieved_after_reflection}),
            },
            search_dir / "revised_reward_schema_and_code.json",
            search_dir / "reward_revision_raw.txt",
        )

        next_dir = run_dir / "iterations" / f"iter_{next_iteration:03d}"
        next_dir.mkdir(parents=True, exist_ok=True)
        _materialize_revision(config, next_dir, revision, env_summary, target_summary, next_iteration)

        next_evidence = EvidenceBuilder.build(next_dir, output_path=next_dir / "iteration_evidence.json")
        memory_record = memory.append_transition(
            iteration_from=next_iteration - 1,
            iteration_to=next_iteration,
            before_evidence=current_evidence,
            after_evidence=next_evidence,
            reflection=reflection,
            revision=revision,
            output_copy_path=next_dir / "memory_transition.json",
        )

        if _fitness(next_evidence) > _fitness(best_evidence):
            best_evidence = next_evidence
            write_json(run_dir / "best_iteration_evidence.json", best_evidence)
            write_json(run_dir / "best_memory_transition.json", memory_record)

        current_dir = next_dir

    write_text(run_dir / "ITERATIVE_DONE.txt", "single-chain iterative search finished\n")
    return run_dir


def _materialize_revision(
    config: Dict[str, Any],
    next_dir: Path,
    revision: Dict[str, Any],
    env_summary: Dict[str, Any],
    target_summary: Dict[str, Any],
    iteration: int,
) -> None:
    reward_schema = revision.get("reward_schema") or {}
    reward_code = revision.get("reward_code") or ""
    reward_dir = next_dir / "reward"
    write_json(reward_dir / "reward_schema.json", reward_schema)
    validation = write_reward_code_files(reward_code, reward_dir)
    if not validation.get("valid", False):
        raise RuntimeError(f"Revised reward failed validation. See {reward_dir / 'validation_report.json'}")

    write_json(next_dir / "environment_understanding_summary.json", env_summary)
    write_json(next_dir / "target_alignment_summary.json", target_summary)
    write_json(next_dir / "revision_metadata.json", revision.get("edit_summary", {}))

    env_cfg = config.get("environment", {}) or {}
    env_py = prepare_env_code(ROOT / env_cfg["env_code_dir"], next_dir / "env_code", reward_code)
    env_cls = load_env_class(env_py, env_cfg.get("class_name", "LunarLander"))

    trainer = SingleChainPPOTrainer(
        config=config,
        output_dir=next_dir / "training",
        env_cls=env_cls,
        environment_understanding=env_summary,
        target_alignment_contract=target_summary,
        reward_schema=reward_schema,
        candidate_id=f"iter_{iteration:03d}",
        generation=iteration,
        creation_type="llm_reward_revision",
    )
    reward_trace = trainer.train()
    write_json(next_dir / "reward_trace.json", reward_trace)


def _fitness(evidence: Dict[str, Any]) -> float:
    try:
        return float(evidence.get("primary_metrics", {}).get("fitness_score", -1e18))
    except Exception:
        return -1e18


def _environment_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_type": "environment_understanding_summary",
        "environment_name": data.get("environment_name"),
        "task_goal": data.get("task_goal"),
        "reward_function_interface": data.get("reward_function_interface"),
        "state_space": data.get("state_space"),
        "action_space": data.get("action_space"),
        "termination_modes": data.get("termination_modes"),
        "success_like_ending": data.get("success_like_ending"),
        "failure_like_endings": data.get("failure_like_endings"),
        "behavior_trajectory_prior": data.get("behavior_trajectory_prior"),
        "reward_design_risks": data.get("reward_design_risks"),
        "reward_hacking_risks": data.get("reward_hacking_risks"),
    }


def _target_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_type": "target_alignment_contract_summary",
        "environment_name": data.get("environment_name"),
        "primary_objective": data.get("primary_objective"),
        "objective_decomposition": data.get("objective_decomposition"),
        "behavior_trajectory_alignment": data.get("behavior_trajectory_alignment"),
        "metric_priority": data.get("metric_priority"),
        "primary_selection_metric": data.get("primary_selection_metric"),
        "proxy_metrics": data.get("proxy_metrics"),
        "diagnostic_metrics": data.get("diagnostic_metrics"),
        "success_criteria": data.get("success_criteria"),
        "failure_modes": data.get("failure_modes"),
        "bad_local_optima": data.get("bad_local_optima"),
        "reward_hacking_definitions": data.get("reward_hacking_definitions"),
        "reward_generator_constraints": data.get("reward_generator_constraints"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run iterative EG-RSA single-chain reward search")
    parser.add_argument("--config", required=True, help="Path to EG-RSA single-chain YAML config")
    parser.add_argument("--bootstrap-run-dir", default=None, help="Optional existing single-chain run dir to continue from")
    args = parser.parse_args()
    run_dir = run_iterative(args.config, bootstrap_run_dir=args.bootstrap_run_dir)
    print(f"EG-RSA iterative single-chain search finished: {run_dir}")


if __name__ == "__main__":
    main()
