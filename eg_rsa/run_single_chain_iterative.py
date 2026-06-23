from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from eg_rsa.llm.client_factory import build_llm_client
from eg_rsa.run_single_chain import run as run_bootstrap, _extract_code_block
from eg_rsa.single_chain.agents import JsonAgent
from eg_rsa.single_chain.controller import write_controller_decision
from eg_rsa.single_chain.controller_v2 import SearchController
from eg_rsa.single_chain.evidence import EvidenceBuilder
from eg_rsa.single_chain.expert_memory import build_memory_context_md
from eg_rsa.single_chain.expert_search_strategy import run_expert_search_strategy
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

    # Load full environment reference for LLM #4 context (generated at bootstrap).
    task_model_md = read_text(run_dir / "agents" / "task_model.md")

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
        pre_strategy_parent_evidence = EvidenceBuilder.build(parent_dir, output_path=parent_dir / "iteration_evidence.json")
        pre_strategy_memory = memory.retrieve(pre_strategy_parent_evidence)
        pre_strategy_md = build_memory_context_md(
            current_evidence=pre_strategy_parent_evidence,
            retrieved_memory=pre_strategy_memory,
        )
        selected_parent_dir, selected_anchor_dir, strategy_decision = run_expert_search_strategy(
            llm_client=llm_client,
            prompt_path=PROMPT_DIR / "expert_search_strategy_prompt.txt",
            run_dir=run_dir,
            parent_dir=parent_dir,
            best_dir=best_dir,
            next_iteration=next_iteration,
            environment_summary={"markdown_text": task_model_md},
            target_summary={"markdown_text": pre_strategy_md},
        )
        parent_dir = selected_parent_dir
        best_dir = selected_anchor_dir
        best_evidence = EvidenceBuilder.build(best_dir, output_path=best_dir / "iteration_evidence.json")
        write_json(run_dir / "best_iteration_evidence.json", best_evidence)
        write_text(run_dir / "BEST_PARENT.txt", str(best_dir) + "\n")
        write_text(run_dir / "CURRENT_PARENT.txt", str(parent_dir) + "\n")

        search_dir = parent_dir / "search"
        search_dir.mkdir(parents=True, exist_ok=True)
        write_json(search_dir / f"expert_search_strategy_decision_iter_{next_iteration:03d}.json", strategy_decision)

        parent_evidence = EvidenceBuilder.build(parent_dir, output_path=parent_dir / "iteration_evidence.json")
        retrieved = memory.retrieve(parent_evidence)
        current_reward_code = read_text(parent_dir / "reward" / "reward_code.py")

        # Build the 4 Markdown inputs for LLM #4.
        diagnostics_md = EvidenceBuilder.build_diagnostics_md(
            parent_dir, output_path=search_dir / "iteration_diagnostics.md",
        )
        memory_context_md = build_memory_context_md(
            current_evidence=parent_evidence,
            retrieved_memory=retrieved,
            strategy_decision=strategy_decision,
            output_path=search_dir / "expert_memory_context.md",
        )

        # LLM #4: ExpertRewardRevisionAgent — Markdown in, Markdown out.
        revision_prompt_tmpl = read_text(PROMPT_DIR / "expert_reward_revision_prompt.txt")
        revision_prompt = (
            revision_prompt_tmpl
            .replace("{{task_model_md}}", task_model_md)
            .replace("{{current_reward_code}}", current_reward_code)
            .replace("{{iteration_diagnostics}}", diagnostics_md)
            .replace("{{expert_memory_context}}", memory_context_md)
        )
        revision_md = llm_client.generate(revision_prompt)
        write_text(search_dir / "expert_reward_revision_raw.md", revision_md)
        write_text(search_dir / "expert_reward_revision.md", revision_md)

        # Extract code blocks from revision output.
        new_reward_code = _extract_code_block(revision_md, "python") or current_reward_code
        new_reward_schema_raw = _extract_code_block(revision_md, "json") or "[]"
        try:
            new_reward_schema = json.loads(new_reward_schema_raw)
        except (json.JSONDecodeError, TypeError):
            new_reward_schema = []

        # Parse structured diagnosis and edit summary from the Markdown revision output.
        parsed_revision = _parse_revision_markdown(revision_md)
        revision = {
            "reward_schema": new_reward_schema,
            "reward_code": new_reward_code,
            "edit_summary": parsed_revision.get("edit_summary", {}),
        }
        reflection = {
            "file_type": "reflection_decision",
            "source": "markdown_revision",
            "diagnosis": parsed_revision.get("diagnosis", {}),
            "causal_chain": parsed_revision.get("causal_chain", ""),
            "expected_outcome": parsed_revision.get("expected_outcome", ""),
        }

        if reflection.get("search_decision", {}).get("recommended_next_action") == "stop_success":
            write_text(run_dir / "ITERATIVE_DONE.txt", f"Stopped at iteration {next_iteration - 1}: stop_success\n")
            break

        candidate_dir = run_dir / "iterations" / f"iter_{next_iteration:03d}"
        if candidate_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing candidate directory: {candidate_dir}")
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _materialize_revision(
            config,
            candidate_dir,
            revision,
            {"markdown_text": task_model_md},
            {"markdown_text": task_model_md},
            llm_client,
            next_iteration,
            parent_reward_code=current_reward_code,
            memory_context={
                "markdown_text": memory_context_md,
                "negative_edges": strategy_decision.get("negative_edges", []),
                "hard_constraints_for_next_revision": strategy_decision.get(
                    "revision_brief", {}
                ).get("forbidden_edit_types", []),
            },
        )

        # Check if reward validation failed and training was skipped
        reward_trace_path = candidate_dir / "reward_trace.json"
        training_skipped = False
        if reward_trace_path.exists():
            rt = read_json(reward_trace_path)
            if rt.get("behavior_metrics", {}).get("skip_reason"):
                training_skipped = True

        if training_skipped:
            # Log the skip and continue to the next iteration with the same parent
            write_text(run_dir / "SKIPPED_ITERATIONS.txt",
                       f"Iteration {next_iteration}: reward validation failed, skipped training\n",
                       mode="a")
            consecutive_rejections += 1
            write_text(run_dir / "CURRENT_PARENT.txt", str(parent_dir) + "\n")
            continue

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
        decision["expert_search_strategy_decision_before_revision"] = strategy_decision
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
            controller_decision=decision,
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


def _parse_revision_markdown(md: str) -> Dict[str, Any]:
    """Extract structured diagnosis and edit summary from the LLM's Markdown revision output.

    The ExpertRewardRevisionAgent outputs Markdown with sections:
    ## Diagnosis (Root Cause, Evidence, Causal Chain)
    ## Edit (What Changed, Why, Expected Outcome)

    This parser extracts those into structured dicts for memory storage.
    """
    import re

    result: Dict[str, Any] = {
        "edit_summary": {},
        "diagnosis": {},
        "causal_chain": "",
        "expected_outcome": "",
    }

    if not md or not isinstance(md, str):
        return result

    # --- Extract Diagnosis section ---
    diag_section = _md_section(md, "Diagnosis")
    if diag_section:
        root_cause = _md_field(diag_section, "Root Cause")
        evidence = _md_field(diag_section, "Evidence")
        causal = _md_field(diag_section, "Causal Chain")
        result["diagnosis"] = {
            "main_problem": root_cause or "",
            "evidence": evidence or "",
        }
        result["causal_chain"] = causal or ""

    # --- Extract Edit section ---
    edit_section = _md_section(md, "Edit")
    if edit_section:
        what_changed = _md_field(edit_section, "What Changed")
        why_changed = _md_field(edit_section, "Why")
        expected = _md_field(edit_section, "Expected Outcome")
        result["expected_outcome"] = expected or ""

        # Parse "What Changed" into structured form: "component: from X to Y"
        changed_components = []
        if what_changed:
            # Try to extract component-level changes
            comp_pattern = r'(?:^|\n)\s*(?:\*\s*|\-\s*|\d+\.\s*)?(?:`)?(\w[\w_]*)(?:`)?\s*:?\s*(?:from|changed|modified|updated|adjusted|increased|decreased|removed|added|replaced).*?(?:$|(?=\n\s*(?:\*|\-|\d+\.|\w+:)))'
            for m in re.finditer(r'([\w_]+)\s*:?\s*(.*?)(?:$|\n)', what_changed, re.MULTILINE):
                line = m.group(0).strip()
                if len(line) > 5 and any(kw in line.lower() for kw in ('change', 'from', 'to', 'remov', 'add', 'increas', 'decreas', 'modif', 'adjust', 'replac', 'convert')):
                    changed_components.append(line[:200])

        edit_scope = "targeted"
        if any(kw in what_changed.lower() for kw in ('remov', 'delet')) if what_changed else False:
            edit_scope = "component_removal"
        if any(kw in what_changed.lower() for kw in ('add', 'new component', 'introduc')) if what_changed else False:
            edit_scope = "component_addition"

        result["edit_summary"] = {
            "edit_scope": edit_scope,
            "what_changed": what_changed or "",
            "why": why_changed or "",
            "changed_design": changed_components if changed_components else [what_changed[:300]] if what_changed else [],
            "changed_active_terms": changed_components,
        }

    return result


def _md_section(md: str, section_name: str) -> str:
    """Extract a named Markdown section (## Section Name) from the document."""
    import re
    # Match from "## Section Name" until next "## " heading of same level.
    # Uses ##(?!#) to match exactly "##" not followed by a third # (avoids matching ###).
    pattern = rf'(?:^|\n)##(?!#)\s+{re.escape(section_name)}\s*\n(.*?)(?=\n##(?!#)\s+\w|\Z)'
    match = re.search(pattern, md, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _md_field(section_text: str, field_name: str) -> str:
    """Extract a named field from a Markdown section.

    Looks for patterns like:
    ### Root Cause
    **Root Cause**: text
    - Root Cause: text
    Root Cause: text
    """
    import re
    patterns = [
        # Pattern 1: ### Field Name heading — capture until next ### or ## heading (but not ####)
        rf'(?:^|\n)\s*###(?!#)\s+{re.escape(field_name)}\s*\n\s*(.*?)(?=\n\s*###(?!#)\s+\w|\n\s*##(?!#)\s+\w|\n\s*$|\Z)',
        # Pattern 2: **Field Name**: bold label — capture until next ** or heading
        rf'(?:^|\n)\s*\*\*{re.escape(field_name)}\*\*\s*:?\s*(.*?)(?=\n\s*\*\*|\n\s*#|\n\s*$|\Z)',
        # Pattern 3: - Field Name: list item — capture until next list item or heading
        rf'(?:^|\n)\s*[\-\*]\s+{re.escape(field_name)}\s*:?\s*(.*?)(?=\n\s*[\-\*]\s+\w|\n\s*#|\Z)',
        # Pattern 4: Field Name: plain text label — capture until next label or heading
        rf'(?:^|\n)\s*{re.escape(field_name)}\s*:?\s*(.*?)(?=\n\s*\w+\s*:|\n\s*#|\Z)',
    ]
    for pat in patterns:
        match = re.search(pat, section_text, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            # Trim trailing partial sentences (stop at double newline or next field)
            cutoff = re.search(r'\n\s*\n|\n\s*[\-\*]\s+\w|\n\s*\w+\s*:', value)
            if cutoff:
                value = value[:cutoff.start()].strip()
            return value[:500]  # Cap at 500 chars
    return ""


def _materialize_revision(
    config: Dict[str, Any],
    next_dir: Path,
    revision: Dict[str, Any],
    env_summary: Dict[str, Any],
    target_summary: Dict[str, Any],
    llm_client: Any,
    iteration: int,
    parent_reward_code: str | None = None,
    memory_context: Dict[str, Any] | None = None,
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
        reference_reward_code=parent_reward_code,
        memory_context=memory_context,
    )

    write_json(next_dir / "environment_understanding_summary.json", env_summary)
    write_json(next_dir / "target_alignment_summary.json", target_summary)
    write_json(next_dir / "revision_metadata.json", revision.get("edit_summary", {}))

    # Graceful fallback: reward validation failed after all repair attempts.
    # Skip training for this candidate instead of crashing the entire experiment.
    if env_cls is None:
        write_json(next_dir / "reward_guard_summary.json", guard_summary)
        write_json(next_dir / "reward_trace.json", {
            "file_type": "reward_trace",
            "candidate_id": f"iter_{iteration:03d}",
            "generation": iteration,
            "creation_type": "llm_reward_revision",
            "primary_metrics": {"fitness_score": float("-inf"), "selection_metric": "fitness_score_auxiliary"},
            "behavior_metrics": {"success_like_terminal_rate": 0.0, "skip_reason": "reward_validation_failed"},
        })
        return

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
    md = data.get("markdown_text", "")
    return {
        "file_type": "environment_understanding_summary",
        "markdown_text": md,
        # Legacy fields for backward compat — extracted from JSON task_model if available.
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
    md = data.get("markdown_text", "")
    return {
        "file_type": "target_alignment_contract_summary",
        "markdown_text": md,
        # Legacy fields for backward compat.
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
    parser = argparse.ArgumentParser(description="Run EG-RSA single-chain iterative search")
    parser.add_argument("--config", required=True, help="Path to iterative EG-RSA single-chain YAML config")
    parser.add_argument("--bootstrap-run-dir", default=None, help="Existing bootstrap run directory to continue from")
    parser.add_argument("--start-parent-dir", default=None, help="Optional parent candidate directory to resume from")
    parser.add_argument("--start-best-dir", default=None, help="Optional best/elite candidate directory to resume from")
    args = parser.parse_args()
    run_dir = run_iterative(
        args.config,
        bootstrap_run_dir=args.bootstrap_run_dir,
        start_parent_dir=args.start_parent_dir,
        start_best_dir=args.start_best_dir,
    )
    print(f"EG-RSA iterative single-chain run finished: {run_dir}")


if __name__ == "__main__":
    main()
