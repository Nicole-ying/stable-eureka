from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from eg_rsa.llm.client_factory import build_llm_client
from eg_rsa.run_single_chain import run as run_bootstrap
from eg_rsa.single_chain.agents import JsonAgent
from eg_rsa.single_chain.controller import SearchController, write_controller_decision
from eg_rsa.single_chain.evidence import EvidenceBuilder
from eg_rsa.single_chain.expert_memory import build_expert_memory_context
from eg_rsa.single_chain.expert_priors import get_expert_priors
from eg_rsa.single_chain.json_tools import read_json, read_text, write_json, write_text
from eg_rsa.single_chain.memory_manager import MemoryManager
from eg_rsa.single_chain.reward_guard import prepare_guarded_reward_env
from eg_rsa.single_chain.trainer import SingleChainPPOTrainer


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "eg_rsa" / "single_chain" / "prompts"


def as_json_text(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def run_iterative(
    config_path: str,
    bootstrap_run_dir: Optional[str] = None,
    start_parent_dir: Optional[str] = None,
    start_best_dir: Optional[str] = None,
) -> Path:
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
    total_iterations = max(1, int(search_cfg.get("iterations", 2)))

    memory_cfg = config.get("memory", {}) or {}
    memory_path = run_dir / memory_cfg.get("path", "memory/reward_memory.jsonl")
    memory = MemoryManager(memory_path, top_k=int(memory_cfg.get("retrieve_top_k", 5)))
    controller = SearchController(config)

    expert_priors = get_expert_priors()
    write_json(run_dir / "expert_reward_design_priors.json", expert_priors)
    expert_blueprint = _load_expert_blueprint(run_dir)
    write_json(run_dir / "expert_reward_design_blueprint.json", expert_blueprint)

    env_understanding = _read_json_with_fallback(run_dir / "agents" / "environment_understanding.json", run_dir / "agents" / "task_model.json")
    target_contract = _read_json_with_fallback(run_dir / "agents" / "target_alignment_contract.json", run_dir / "agents" / "task_model.json")
    env_summary = _environment_summary(env_understanding)
    target_summary = _target_summary(target_contract)
    write_json(run_dir / "environment_understanding_summary.json", env_summary)
    write_json(run_dir / "target_alignment_summary.json", target_summary)

    next_iteration_start = _next_iteration_index(run_dir)

    if start_best_dir:
        best_dir = Path(start_best_dir)
        best_evidence = EvidenceBuilder.build(best_dir, output_path=best_dir / "iteration_evidence.json")
    elif (run_dir / "best_iteration_evidence.json").exists():
        best_evidence = read_json(run_dir / "best_iteration_evidence.json")
        best_dir = _candidate_dir_for_evidence(run_dir, best_evidence) or run_dir
        best_evidence = EvidenceBuilder.build(best_dir, output_path=best_dir / "iteration_evidence.json")
    else:
        best_dir = run_dir
        best_evidence = EvidenceBuilder.build(best_dir, output_path=best_dir / "iteration_evidence.json")

    if start_parent_dir:
        parent_dir = Path(start_parent_dir)
        parent_evidence = EvidenceBuilder.build(parent_dir, output_path=parent_dir / "iteration_evidence.json")
    elif next_iteration_start > 1:
        parent_dir = best_dir
        parent_evidence = best_evidence
    else:
        parent_dir = run_dir
        parent_evidence = EvidenceBuilder.build(parent_dir, output_path=parent_dir / "iteration_evidence.json")

    write_json(run_dir / "best_iteration_evidence.json", best_evidence)
    write_text(run_dir / "BEST_PARENT.txt", str(best_dir) + "\n")
    write_text(run_dir / "CURRENT_PARENT.txt", str(parent_dir) + "\n")
    write_json(run_dir / "resume_state.json", {
        "next_iteration_start": next_iteration_start,
        "parent_dir": str(parent_dir),
        "best_dir": str(best_dir),
        "total_iterations": total_iterations,
    })

    consecutive_rejections = 0

    for next_iteration in range(next_iteration_start, total_iterations):
        search_dir = parent_dir / "search"
        search_dir.mkdir(parents=True, exist_ok=True)

        parent_evidence = EvidenceBuilder.build(parent_dir, output_path=parent_dir / "iteration_evidence.json")
        retrieved = memory.retrieve(parent_evidence)
        write_json(search_dir / "retrieved_memory.json", {"items": retrieved})
        expert_memory_context = build_expert_memory_context(
            current_evidence=parent_evidence,
            best_evidence=best_evidence,
            retrieved_memory=retrieved,
            output_path=search_dir / "expert_memory_context.json",
        )

        current_reward_schema = read_json(parent_dir / "reward" / "reward_schema.json")
        current_reward_code = read_text(parent_dir / "reward" / "reward_code.py")

        expert_revision_prompt = read_text(PROMPT_DIR / "expert_reward_revision_prompt.txt")
        revision_bundle = JsonAgent("ExpertRewardRevisionAgent", llm_client, expert_revision_prompt).run(
            {
                "expert_reward_design_priors_json": as_json_text(expert_priors),
                "expert_reward_design_blueprint_json": as_json_text(expert_blueprint),
                "expert_memory_context_json": as_json_text(expert_memory_context),
                "environment_understanding_summary_json": as_json_text(env_summary),
                "target_alignment_contract_summary_json": as_json_text(target_summary),
                "current_reward_schema_json": as_json_text(current_reward_schema),
                "current_reward_code": current_reward_code,
                "iteration_evidence_json": as_json_text(parent_evidence),
                "best_iteration_evidence_json": as_json_text(best_evidence),
                "retrieved_memory_json": as_json_text({"items": retrieved}),
            },
            search_dir / "expert_reward_revision_bundle.json",
            search_dir / "expert_reward_revision_raw.txt",
        )
        reflection = revision_bundle.get("reflection_decision") or {
            "file_type": "reflection_decision",
            "agent_name": "ExpertRewardRevisionAgent",
            "source": "expert_reward_revision_bundle",
            "search_decision": revision_bundle.get("search_decision", {}),
            "causal_diagnosis": revision_bundle.get("causal_diagnosis", {}),
        }
        revision = revision_bundle.get("revised_reward_schema_and_code") or revision_bundle
        write_json(search_dir / "reflection_decision.json", reflection)
        write_json(search_dir / "revised_reward_schema_and_code.json", revision)

        if reflection.get("search_decision", {}).get("recommended_next_action") == "stop_success":
            write_text(run_dir / "ITERATIVE_DONE.txt", f"Stopped at iteration {next_iteration - 1}: stop_success\n")
            break

        candidate_dir = run_dir / "iterations" / f"iter_{next_iteration:03d}"
        if candidate_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing candidate directory: {candidate_dir}")
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _materialize_revision(config, candidate_dir, revision, env_summary, target_summary, llm_client, next_iteration)

        candidate_evidence = EvidenceBuilder.build(candidate_dir, output_path=candidate_dir / "iteration_evidence.json")
        decision = controller.decide(
            parent_evidence=parent_evidence,
            candidate_evidence=candidate_evidence,
            best_evidence=best_evidence,
            parent_dir=parent_dir,
            candidate_dir=candidate_dir,
            best_dir=best_dir,
            consecutive_rejections=consecutive_rejections,
        )
        write_controller_decision(candidate_dir / "controller_decision.json", decision)
        write_controller_decision(run_dir / "last_controller_decision.json", decision)

        memory_record = memory.append_transition(
            iteration_from=next_iteration - 1,
            iteration_to=next_iteration,
            before_evidence=parent_evidence,
            after_evidence=candidate_evidence,
            reflection=reflection,
            revision=revision,
            output_copy_path=candidate_dir / "memory_transition.json",
        )

        if decision.get("accepted_as_elite"):
            best_dir = candidate_dir
            best_evidence = candidate_evidence
            write_json(run_dir / "best_iteration_evidence.json", best_evidence)
            write_json(run_dir / "best_memory_transition.json", memory_record)
            write_text(run_dir / "BEST_PARENT.txt", str(best_dir) + "\n")

        if decision.get("rejected"):
            consecutive_rejections += 1
        else:
            consecutive_rejections = 0

        if decision.get("should_stop"):
            write_text(run_dir / "ITERATIVE_DONE.txt", f"Stopped after {consecutive_rejections} consecutive rejected candidates.\n")
            break

        next_parent_dir = Path(decision.get("next_search_parent", {}).get("dir") or decision.get("rollback", {}).get("next_parent_dir") or candidate_dir)
        if not next_parent_dir.exists():
            next_parent_dir = best_dir
        parent_dir = next_parent_dir
        write_text(run_dir / "CURRENT_PARENT.txt", str(parent_dir) + "\n")

    write_text(run_dir / "ITERATIVE_DONE.txt", "single-chain iterative search finished\n")
    return run_dir


def _read_json_with_fallback(primary: Path, fallback: Path) -> Dict[str, Any]:
    if primary.exists():
        return read_json(primary)
    return read_json(fallback)


def _load_expert_blueprint(run_dir: Path) -> Dict[str, Any]:
    for path in [
        run_dir / "expert_reward_design_blueprint.json",
        run_dir / "agents" / "expert_reward_design_blueprint.json",
        run_dir / "agents" / "initial_reward_schema_and_code.json",
    ]:
        if path.exists():
            try:
                data = read_json(path)
                if "expert_blueprint" in data:
                    return data.get("expert_blueprint") or {}
                return data
            except Exception:
                pass
    return {
        "file_type": "expert_reward_design_blueprint",
        "fallback": True,
        "initial_reward_hard_constraints": [
            "Separate objective, progress, regularizers, and diagnostics.",
            "Use expert_memory_context to avoid repeating rejected edit patterns.",
        ],
    }


def _materialize_revision(
    config: Dict[str, Any],
    next_dir: Path,
    revision: Dict[str, Any],
    env_summary: Dict[str, Any],
    target_summary: Dict[str, Any],
    llm_client: Any,
    iteration: int,
) -> None:
    reward_schema = revision.get("reward_schema") or {}
    reward_code = revision.get("reward_code") or ""
    env_cls, reward_schema, reward_code, guard_summary = prepare_guarded_reward_env(
        config=config,
        output_dir=next_dir,
        reward_schema=reward_schema,
        reward_code=reward_code,
        environment_understanding=env_summary,
        target_alignment_contract=target_summary,
        llm_client=llm_client,
        repair_dir=next_dir / "search" / "reward_repairs",
    )

    write_json(next_dir / "environment_understanding_summary.json", env_summary)
    write_json(next_dir / "target_alignment_summary.json", target_summary)
    write_json(next_dir / "revision_metadata.json", revision.get("edit_summary", {}))

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


def _next_iteration_index(run_dir: Path) -> int:
    iterations_dir = run_dir / "iterations"
    if not iterations_dir.exists():
        return 1
    max_idx = 0
    for child in iterations_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"iter_(\d+)", child.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def _candidate_dir_for_evidence(run_dir: Path, evidence: Dict[str, Any]) -> Optional[Path]:
    candidate_id = str(evidence.get("candidate_id") or "")
    if candidate_id in {"single_chain_iter0", "iter_000", run_dir.name}:
        return run_dir
    match = re.fullmatch(r"iter_(\d+)", candidate_id)
    if match:
        path = run_dir / "iterations" / f"iter_{int(match.group(1)):03d}"
        if path.exists():
            return path
    return None


def _environment_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_type": "environment_understanding_summary",
        "environment_name": data.get("environment_name"),
        "task_goal": data.get("task_goal") or data.get("primary_objective"),
        "reward_function_interface": data.get("reward_function_interface"),
        "state_space": data.get("state_space") or data.get("state_space_summary"),
        "action_space": data.get("action_space") or data.get("action_space_summary"),
        "termination_modes": data.get("termination_modes") or data.get("failure_or_stop_signals_available_to_reward"),
        "success_like_ending": data.get("success_like_ending") or data.get("success_signals_available_to_reward"),
        "failure_like_endings": data.get("failure_like_endings") or data.get("failure_or_stop_signals_available_to_reward"),
        "behavior_trajectory_prior": data.get("behavior_trajectory_prior") or data.get("intended_behavior_phases"),
        "reward_design_risks": data.get("reward_design_risks") or data.get("dangerous_local_optima"),
    }


def _target_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_type": "target_alignment_contract_summary",
        "environment_name": data.get("environment_name"),
        "primary_objective": data.get("primary_objective"),
        "objective_decomposition": data.get("objective_decomposition") or data.get("intended_behavior_phases"),
        "behavior_trajectory_alignment": data.get("behavior_trajectory_alignment") or data.get("intended_behavior_phases"),
        "metric_priority": data.get("metric_priority") or data.get("primary_selection_metric"),
        "primary_selection_metric": data.get("primary_selection_metric", "fitness_score"),
        "proxy_metrics": data.get("proxy_metrics"),
        "diagnostic_metrics": data.get("diagnostic_metrics"),
        "success_criteria": data.get("success_criteria") or data.get("success_signals_available_to_reward"),
        "failure_modes": data.get("failure_modes") or data.get("dangerous_local_optima"),
        "bad_local_optima": data.get("bad_local_optima") or data.get("dangerous_local_optima"),
        "reward_hacking_definitions": data.get("reward_hacking_definitions"),
        "reward_generator_constraints": data.get("reward_generator_constraints") or data.get("reward_design_constraints"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run iterative EG-RSA single-chain reward search")
    parser.add_argument("--config", required=True, help="Path to EG-RSA single-chain YAML config")
    parser.add_argument("--bootstrap-run-dir", default=None, help="Optional existing single-chain run dir to continue from")
    parser.add_argument("--start-parent-dir", default=None, help="Optional candidate directory to edit next")
    parser.add_argument("--start-best-dir", default=None, help="Optional elite candidate directory")
    args = parser.parse_args()
    run_dir = run_iterative(
        args.config,
        bootstrap_run_dir=args.bootstrap_run_dir,
        start_parent_dir=args.start_parent_dir,
        start_best_dir=args.start_best_dir,
    )
    print(f"EG-RSA iterative single-chain search finished: {run_dir}")


if __name__ == "__main__":
    main()
