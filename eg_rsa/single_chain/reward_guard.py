from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from eg_rsa.single_chain.agents import JsonAgent
from eg_rsa.single_chain.env_builder import load_env_class, prepare_env_code
from eg_rsa.single_chain.json_tools import read_text, write_json, write_text
from eg_rsa.single_chain.reward_runtime import smoke_test_reward_env, validate_reward_runtime_safety
from eg_rsa.single_chain.reward_validation import write_reward_code_files


ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "eg_rsa" / "single_chain" / "prompts"


def as_json_text(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def prepare_guarded_reward_env(
    config: Dict[str, Any],
    output_dir: str | Path,
    reward_schema: Dict[str, Any],
    reward_code: str,
    environment_understanding: Dict[str, Any],
    target_alignment_contract: Dict[str, Any],
    llm_client: Any | None,
    repair_dir: str | Path,
    max_repair_attempts: int | None = None,
) -> Tuple[Any, Dict[str, Any], str, Dict[str, Any]]:
    """Write, validate, smoke-test, and optionally repair reward code.

    Returns (env_cls, reward_schema, reward_code, guard_summary).
    """
    output_dir = Path(output_dir)
    repair_dir = Path(repair_dir)
    repair_dir.mkdir(parents=True, exist_ok=True)

    repair_cfg = config.get("reward_repair", {}) or {}
    if max_repair_attempts is None:
        max_repair_attempts = int(repair_cfg.get("max_attempts", 2))
    smoke_steps = int(repair_cfg.get("smoke_steps", 8))

    attempts = []
    last_reports: Dict[str, Any] = {}
    current_schema = reward_schema or {}
    current_code = reward_code or ""

    for attempt in range(max_repair_attempts + 1):
        reward_dir = output_dir / "reward"
        write_json(reward_dir / "reward_schema.json", current_schema)
        validation = write_reward_code_files(current_code, reward_dir)
        runtime_safety = validate_reward_runtime_safety(current_code)
        write_json(reward_dir / "runtime_safety_report.json", runtime_safety)

        env_cls = None
        smoke_report: Dict[str, Any] = {"valid": False, "stage": "skipped"}
        if validation.get("valid") and runtime_safety.get("valid"):
            env_cfg = config.get("environment", {}) or {}
            env_py = prepare_env_code(ROOT / env_cfg["env_code_dir"], output_dir / "env_code", current_code)
            env_cls = load_env_class(env_py, env_cfg.get("class_name", "LunarLander"))
            smoke_report = smoke_test_reward_env(
                env_cls=env_cls,
                env_kwargs=env_cfg.get("kwargs") or {},
                max_episode_steps=env_cfg.get("max_episode_steps"),
                seed=(config.get("rl", {}).get("training", {}) or {}).get("seed", 0),
                output_path=reward_dir / "smoke_test_report.json",
                n_steps=smoke_steps,
            )
        else:
            write_json(reward_dir / "smoke_test_report.json", smoke_report)

        attempt_report = {
            "attempt": attempt,
            "validation_valid": bool(validation.get("valid")),
            "runtime_safety_valid": bool(runtime_safety.get("valid")),
            "smoke_test_valid": bool(smoke_report.get("valid")),
            "validation": validation,
            "runtime_safety": runtime_safety,
            "smoke_test": smoke_report,
        }
        attempts.append(attempt_report)
        last_reports = attempt_report

        if validation.get("valid") and runtime_safety.get("valid") and smoke_report.get("valid"):
            summary = {
                "file_type": "reward_guard_summary",
                "valid": True,
                "repair_attempts_used": attempt,
                "attempts": attempts,
            }
            write_json(output_dir / "reward_guard_summary.json", summary)
            return env_cls, current_schema, current_code, summary

        if attempt >= max_repair_attempts or llm_client is None:
            break

        repair_prompt = read_text(PROMPT_DIR / "reward_repair_prompt.txt")
        repair = JsonAgent("RewardRepairAgent", llm_client, repair_prompt).run(
            {
                "environment_understanding_summary_json": as_json_text(environment_understanding),
                "target_alignment_contract_summary_json": as_json_text(target_alignment_contract),
                "current_reward_schema_json": as_json_text(current_schema),
                "current_reward_code": current_code,
                "validation_report_json": as_json_text(validation),
                "runtime_safety_report_json": as_json_text(runtime_safety),
                "smoke_test_report_json": as_json_text(smoke_report),
            },
            repair_dir / f"reward_repair_{attempt + 1:02d}.json",
            repair_dir / f"reward_repair_{attempt + 1:02d}_raw.txt",
        )
        current_schema = repair.get("reward_schema") or current_schema
        current_code = repair.get("reward_code") or current_code

    summary = {
        "file_type": "reward_guard_summary",
        "valid": False,
        "repair_attempts_used": len(attempts) - 1,
        "attempts": attempts,
        "last_reports": last_reports,
    }
    write_json(output_dir / "reward_guard_summary.json", summary)
    raise RuntimeError(f"Reward failed guarded validation/smoke test. See {output_dir / 'reward_guard_summary.json'}")
