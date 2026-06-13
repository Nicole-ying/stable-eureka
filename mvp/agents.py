from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm_logging import write_llm_call
from .models import ModelGateway
from .reward_schema import normalize_schema
from .reward_spec import parse_validate_compile_reward_spec


PROMPT_DIR = Path(__file__).parent / "prompts"

PRIVATE_TERMS = (
    "env_reward",
    "hidden_env_reward",
    "_hidden_env_reward",
    "fitness_score",
    "compute_fitness_score",
    "benchmark_reward",
    "official_reward",
    "original_reward",
    "hidden_reward",
)


@dataclass
class RewardDraft:
    candidate_id: str
    reward_code: str
    reward_spec: dict[str, Any]
    rationale: str
    llm_response: str
    prompt_budget: dict[str, Any]


@dataclass
class RepairDraft:
    reward_code: str
    rationale: str
    llm_response: str
    prompt_budget: dict[str, Any]


def _read_prompt(name: str, fallback: str) -> str:
    p = PROMPT_DIR / name
    if p.exists():
        return p.read_text(encoding="utf-8")
    return fallback


def _contains_private_term(text: str) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in PRIVATE_TERMS)


def _strip_import_lines(code: str) -> str:
    """
    Programmatic fallback: remove import lines from generated reward code.

    This is now mostly used by the legacy RepairAgent. RewardSpec generation
    should not emit Python code at all.
    """
    cleaned = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip() + "\n"


def _extract_code_and_rationale(text: str, stage: str) -> tuple[str, str]:
    if not text or not text.strip():
        raise ValueError(f"empty LLM response at stage={stage}")

    patterns = [
        r"```(?:python|py)\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]

    code = ""
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            break

    if not code:
        m = re.search(r"(def\s+compute_reward\s*\(.*)", text, re.DOTALL)
        if m:
            code = m.group(1).strip()
        else:
            code = text.strip()

    if "def compute_reward" not in code:
        raise ValueError(
            f"could not extract compute_reward from LLM output at stage={stage}. "
            f"output_head={text[:500]!r}"
        )

    rationale_match = re.search(r"RATIONALE:(.*)", text, re.DOTALL | re.IGNORECASE)
    rationale = rationale_match.group(1).strip() if rationale_match else f"{stage} generated code"
    return _strip_import_lines(code.strip()), rationale


def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}

    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except Exception:
        return {}


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    if not text:
        return []

    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass
    return []


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED]"


class EnvUnderstandingAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "env_understanding_system.txt",
            "Read the Eureka task_description.txt and step.py. Produce an environment understanding report. "
            "You may reason about observation/action semantics. Do not reveal or use official reward formulas, "
            "env_reward, fitness_score, or hidden evaluator details.",
        )

    def analyze(self, clean_interface: dict[str, Any], log_dir: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
        user = (
            "Eureka task_description.txt:\n"
            f"{clean_interface.get('eureka_task_description', '')}\n\n"
            "Eureka step.py:\n"
            f"{clean_interface.get('eureka_step_code', '')}\n\n"
            "Return a concise markdown report followed by a JSON object with keys: "
            "task_goal, observations, actions, termination, public_reward_design_variables, risks."
        )
        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(log_dir, self.system_prompt, user, response, {"agent": "EnvUnderstandingAgent"})
        parsed = _extract_json_object(response)
        return response, parsed, budget


class SchemaPlannerAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "schema_planner_system.txt",
            "Create an environment-aware reward schema and search plan from the Eureka task context and environment understanding. "
            "The schema must define reward components for compute_reward(obs, action, next_obs, done, info). "
            "Do not use env_reward, fitness_score, official reward formulas, or hidden evaluator details.",
        )

    def plan(
        self,
        clean_interface: dict[str, Any],
        env_understanding: str,
        log_dir: Path,
    ) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
        user = (
            "Eureka task_description.txt:\n"
            f"{clean_interface.get('eureka_task_description', '')}\n\n"
            "Eureka step.py:\n"
            f"{clean_interface.get('eureka_step_code', '')}\n\n"
            "Environment understanding:\n"
            f"{env_understanding}\n\n"
            "Return JSON with keys:\n"
            "{\n"
            '  "reward_schema": {\n'
            '    "components": [{"id": "...", "description": "...", "direction": "maximize", "required": true}],\n'
            '    "reward_abs_bound": 1000.0\n'
            "  },\n"
            '  "search_plan": "markdown string"\n'
            "}\n"
        )
        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(log_dir, self.system_prompt, user, response, {"agent": "SchemaPlannerAgent"})
        parsed = _extract_json_object(response)

        raw_schema = parsed.get("reward_schema", {}) if isinstance(parsed, dict) else {}
        schema = normalize_schema(raw_schema, clean_interface)

        search_plan = ""
        if isinstance(parsed, dict):
            search_plan = str(parsed.get("search_plan", "")).strip()
        if not search_plan:
            search_plan = response

        return schema, search_plan, response, budget


class RewardCoderAgent:
    """
    RewardSpec JSON IR generator.

    The LLM no longer writes executable Python reward code. It proposes a
    schema-constrained JSON RewardSpec, and the framework deterministically
    compiles that spec into compute_reward().
    """

    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "reward_coder_system.txt",
            "You design rewards by outputting JSON RewardSpec only. Do not write Python code. "
            "The framework will compile RewardSpec into compute_reward(obs, action, next_obs, done, info).",
        )

    def draft(
        self,
        candidate_id: str,
        clean_interface: dict[str, Any],
        env_understanding: str,
        reward_schema: dict[str, Any],
        search_plan: str,
        feedback_context: str,
        memory_context: str,
        parent_specs: list[dict[str, Any]],
        log_dir: Path,
        parent_code_max_chars: int,
    ) -> RewardDraft:
        safe_parent_specs = []
        for spec in parent_specs:
            text = json.dumps(spec, ensure_ascii=False)
            if _contains_private_term(text):
                continue
            safe_parent_specs.append(json.loads(_truncate(text, parent_code_max_chars).replace("\n...[TRUNCATED]", "")) if len(text) <= parent_code_max_chars else spec)

        parent_block = json.dumps(safe_parent_specs, ensure_ascii=False, indent=2) if safe_parent_specs else "[]"

        user = (
            f"Candidate ID: {candidate_id}\n\n"
            "Eureka task_description.txt:\n"
            f"{clean_interface.get('eureka_task_description', '')}\n\n"
            "Eureka step.py:\n"
            f"{clean_interface.get('eureka_step_code', '')}\n\n"
            "Environment understanding:\n"
            f"{env_understanding}\n\n"
            "Reward schema:\n"
            f"{json.dumps(reward_schema, ensure_ascii=False, indent=2)}\n\n"
            "Search plan:\n"
            f"{search_plan}\n\n"
            "Feedback context:\n"
            f"{feedback_context}\n\n"
            "Memory context:\n"
            f"{memory_context}\n\n"
            "Parent RewardSpecs:\n"
            f"{parent_block}\n\n"
            "Generate exactly one RewardSpec JSON object. Do not output Python code. "
            "The RewardSpec components must exactly match the reward schema component IDs. "
            "Each component must contain id, expression, and clip. Expressions may only use obs, next_obs, action, done, info, np, math, abs, min, max, float, int, bool. "
            "Use private_eval_return only as black-box feedback; do not infer hidden evaluator internals."
        )

        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(log_dir, self.system_prompt, user, response, {"agent": "RewardSpecAgent", "candidate_id": candidate_id})
        parsed = _extract_json_object(response)
        raw_spec = parsed.get("reward_spec", parsed) if isinstance(parsed, dict) else {}
        reward_spec, reward_code = parse_validate_compile_reward_spec(raw_spec, reward_schema)

        rationale = str(parsed.get("rationale", "") if isinstance(parsed, dict) else "").strip()
        if not rationale:
            rationale = str(reward_spec.get("rationale", "")).strip() or "RewardSpec JSON IR generated."

        return RewardDraft(candidate_id, reward_code, reward_spec, rationale, response, budget)


class RepairAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "repair_system.txt",
            "Repair generated reward code so it satisfies the schema and validation contract. "
            "Do not use env_reward, fitness_score, compute_fitness_score, official reward formulas, or hidden evaluator details.",
        )

    def can_repair(self, reward_code: str, validation_errors: list[str]) -> bool:
        return not _contains_private_term(reward_code + "\n" + "\n".join(validation_errors))

    def repair(
        self,
        reward_code: str,
        validation_errors: list[str],
        reward_schema: dict[str, Any],
        attempt_index: int,
        log_dir: Path,
    ) -> RepairDraft:
        user = (
            f"Repair attempt: {attempt_index}\n\n"
            f"Validation errors:\n{validation_errors}\n\n"
            "Reward schema:\n"
            f"{json.dumps(reward_schema, ensure_ascii=False, indent=2)}\n\n"
            "Candidate reward code:\n"
            f"```python\n{reward_code}\n```\n\n"
            "Minimally repair schema, syntax, numerical stability, and return-contract issues only."
        )

        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(log_dir, self.system_prompt, user, response, {"agent": "RepairAgent", "attempt": attempt_index})
        repaired_code, rationale = _extract_code_and_rationale(response, stage="repair")
        return RepairDraft(repaired_code, rationale, response, budget)


class ReflectionAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "reflection_system.txt",
            "Analyze structured reward-search evidence. Produce a concise reflection report for the next generation. "
            "Use private_eval_return only as a black-box selection score. Do not infer hidden evaluator formula.",
        )

    def reflect(
        self,
        evidence: dict[str, Any],
        previous_env_memory: str,
        memory_context: str,
        log_dir: Path,
    ) -> tuple[str, dict[str, Any]]:
        user = (
            "Structured evidence for this generation:\n"
            f"{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
            "Previous environment memory:\n"
            f"{previous_env_memory}\n\n"
            "Retrieved memory context:\n"
            f"{memory_context}\n\n"
            "Write:\n"
            "1. What worked.\n"
            "2. What failed.\n"
            "3. What to try next.\n"
            "4. Which lessons seem supported or contradicted.\n"
        )
        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(log_dir, self.system_prompt, user, response, {"agent": "ReflectionAgent"})
        return response, budget


class LessonExtractorAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "lesson_extractor_system.txt",
            "Extract reusable lessons from structured evidence and reflection. Return JSON array only. "
            "Each lesson should include lesson_type, condition, observation, explanation, recommendation, confidence, reuse_policy.",
        )

    def extract(
        self,
        evidence: dict[str, Any],
        reflection_report: str,
        scope: str,
        env_alias: str,
        generation: int,
        log_dir: Path,
        candidate_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        user = (
            f"Scope: {scope}\n"
            f"Env alias: {env_alias}\n"
            f"Generation: {generation}\n"
            f"Candidate ID: {candidate_id or 'N/A'}\n\n"
            "Structured evidence:\n"
            f"{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
            "Reflection report:\n"
            f"{reflection_report}\n\n"
            "Return JSON array of lessons. Do not include code blocks."
        )
        response = self.model.chat(self.system_prompt, user)
        budget = write_llm_call(
            log_dir,
            self.system_prompt,
            user,
            response,
            {"agent": "LessonExtractorAgent", "scope": scope, "candidate_id": candidate_id},
        )
        lessons = _extract_json_array(response)
        return lessons, budget


class VisionJudgeAgent:
    def __init__(self, model: ModelGateway):
        self.model = model
        self.system_prompt = _read_prompt(
            "vision_judge_system.txt",
            "Judge visible behavior quality only if visual evidence is available. Return JSON only.",
        )

    def judge(self, clean_interface: dict[str, Any], train_result: dict[str, Any], video_path) -> tuple[float, str, dict]:
        rubric = (
            f"Environment alias: {clean_interface.get('env_alias')}.\n"
            "Judge visible behavior quality only if visual evidence is available.\n"
            "Do not use private evaluator details or generated reward magnitude as proof of success.\n"
            "Return JSON only."
        )
        out = self.model.judge_video(self.system_prompt, rubric, video_path)
        score = float(out.get("score", 0.0))
        reason = str(out.get("reason", ""))
        if score <= 0:
            reason = reason or "no_visual_score_available"
        return max(0.0, min(100.0, score)), reason, out
