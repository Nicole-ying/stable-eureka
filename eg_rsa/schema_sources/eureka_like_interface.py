from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class EurekaLikeInterfaceBuilder:
    """Convert an environment-level Eureka-like task file into a primitive interface.

    This builder is intentionally conservative. It does not claim to parse arbitrary
    Python env.py files yet. Instead, it treats the Eureka-like task file as the
    experiment entry point and derives the primitive interface used by the AST
    bootstrap agent from that environment-level specification.
    """

    DEFAULT_FUNCTIONS = ["abs", "min", "max", "clip", "sqrt", "exp", "tanh"]
    DEFAULT_COMPONENT_TYPES = ["formula_component", "conditional_formula_component", "event_predicate"]
    DEFAULT_SEMANTIC_ROLES = [
        "dense_guidance",
        "stability_quality",
        "terminal_success",
        "safety_constraint",
        "control_cost",
    ]

    @classmethod
    def build_from_file(
        cls,
        task_file_path: str | Path,
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        task_path = Path(task_file_path)
        if not task_path.exists():
            raise FileNotFoundError(f"Eureka-like task file not found: {task_path}")

        task_spec = cls._load_task_file(task_path)
        interface = cls.build(task_spec=task_spec, source_path=task_path)

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            cls._write_json(out / "eureka_like_task_input.json", task_spec)
            cls._write_json(out / "generated_primitive_interface.json", interface)
            cls._write_json(
                out / "interface_generation_report.json",
                {
                    "source": "eureka_like_task_file",
                    "source_path": str(task_path),
                    "output_path": str(out / "generated_primitive_interface.json"),
                    "raw_env_code_input": False,
                    "env_code_parser": "planned_not_current",
                    "notes": [
                        "The Eureka-like task file is the experiment entry point.",
                        "The primitive interface is generated as an intermediate artifact.",
                        "Arbitrary raw env.py parsing is not implemented in this step.",
                    ],
                },
            )

        return interface

    @classmethod
    def build(cls, task_spec: Dict[str, Any], source_path: Optional[Path] = None) -> Dict[str, Any]:
        if not isinstance(task_spec, dict):
            raise ValueError("Eureka-like task specification must be a JSON/YAML object")

        env_source = task_spec.get("environment_source", {}) or {}
        interface_policy = task_spec.get("interface_generation_policy", {}) or {}

        observation_variables = cls._extract_variables(
            task_spec,
            env_source,
            keys=("observation_variables", "observations"),
            nested_keys=("observation_interface", "observation_space"),
        )
        action_variables = cls._extract_variables(
            task_spec,
            env_source,
            keys=("action_variables", "actions"),
            nested_keys=("action_interface", "action_space"),
        )

        observation_mapping = task_spec.get("observation_mapping")
        if not isinstance(observation_mapping, dict):
            observation_mapping = {}
            for idx, item in enumerate(observation_variables):
                if isinstance(item, dict) and item.get("name"):
                    observation_mapping[str(item["name"])] = idx

        allowed_formula_variables = cls._allowed_formula_variables(
            task_spec,
            observation_variables,
            action_variables,
        )

        env_name = task_spec.get("env") or env_source.get("env_id") or env_source.get("name") or "unknown_env"
        task_description = task_spec.get("task_description") or env_source.get("task_description") or ""
        allowed_functions = task_spec.get("allowed_formula_functions") or cls.DEFAULT_FUNCTIONS
        semantic_roles = task_spec.get("semantic_roles") or cls.DEFAULT_SEMANTIC_ROLES

        interface = {
            "version": 1,
            "env": env_name,
            "purpose": (
                "Generated primitive task interface for EG-RSA-V2. This file is derived "
                "from a Eureka-like environment-level task file and exposes primitive "
                "observation/action variables, safe formula variables/functions, semantic "
                "roles, and task text. It does not expose any previous reward schema or "
                "previous diagnostic/event definitions."
            ),
            "input_boundary": "eureka_like_task_file_to_primitive_interface",
            "source_eureka_like_task_file": str(source_path) if source_path is not None else None,
            "raw_env_code_input": False,
            "env_code_parser": "planned_not_current",
            "task_description": task_description,
            "environment_source": env_source,
            "observation_variables": observation_variables,
            "action_variables": action_variables,
            "allowed_formula_variables": allowed_formula_variables,
            "allowed_formula_functions": allowed_functions,
            "allowed_component_types_v2": task_spec.get("allowed_component_types_v2") or cls.DEFAULT_COMPONENT_TYPES,
            "bootstrap_requirements": cls._bootstrap_requirements(task_spec),
            "semantic_roles": semantic_roles,
            "observation_mapping": observation_mapping,
            "bootstrap_interface_policy": {
                "principle": (
                    "V2 avoids previous-schema leakage by converting an environment-level "
                    "task file into an isolated primitive interface. The bootstrap agent "
                    "only receives primitive observation/action variables, safe formula "
                    "functions, task text, and output format constraints."
                ),
                "formula_boundary": "Generated formulas must use only allowed_formula_variables and allowed_formula_functions.",
                "not_exposed_to_llm": [
                    "previous initial reward schema content",
                    "previous diagnostic metric definitions",
                    "previous event predicate definitions",
                    "previous task-specific reward-design files",
                ],
                "interface_generation_policy": interface_policy,
            },
            "action_cost_policy_v2": task_spec.get("action_cost_policy_v2")
            or {
                "custom_action_cost": "Use formula_component with semantic_role='control_cost'.",
                "legacy_action_penalty": "Do not use for LLM-generated custom formulas; runtime action_penalty is built-in -sum(action^2).",
            },
            "action_mapping": task_spec.get("action_mapping") or env_source.get("action_mapping", {}),
        }
        return interface

    @staticmethod
    def _load_task_file(path: Path) -> Dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yml", ".yaml"}:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Eureka-like task file must contain an object: {path}")
        return data

    @classmethod
    def _extract_variables(
        cls,
        task_spec: Dict[str, Any],
        env_source: Dict[str, Any],
        keys: tuple[str, ...],
        nested_keys: tuple[str, ...],
    ) -> List[Dict[str, Any]]:
        candidates: List[Any] = []
        for key in keys:
            candidates.append(task_spec.get(key))
            candidates.append(env_source.get(key))
        for key in nested_keys:
            nested = task_spec.get(key)
            if isinstance(nested, dict):
                candidates.extend([nested.get("variables"), nested.get("items")])
            nested = env_source.get(key)
            if isinstance(nested, dict):
                candidates.extend([nested.get("variables"), nested.get("items")])

        for candidate in candidates:
            variables = cls._normalize_variable_list(candidate)
            if variables:
                return variables
        return []

    @staticmethod
    def _normalize_variable_list(value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        variables: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                variables.append({"name": item, "description": "", "type": "float"})
            elif isinstance(item, dict) and item.get("name"):
                entry = dict(item)
                entry.setdefault("description", "")
                entry.setdefault("type", "float")
                variables.append(entry)
        return variables

    @staticmethod
    def _allowed_formula_variables(
        task_spec: Dict[str, Any],
        observation_variables: List[Dict[str, Any]],
        action_variables: List[Dict[str, Any]],
    ) -> List[str]:
        explicit = task_spec.get("allowed_formula_variables")
        if isinstance(explicit, list) and explicit:
            return [str(x) for x in explicit]
        names: List[str] = []
        for item in observation_variables + action_variables:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
        return names

    @staticmethod
    def _bootstrap_requirements(task_spec: Dict[str, Any]) -> Dict[str, Any]:
        requirements = {
            "must_generate_initial_schema": True,
            "must_generate_diagnostic_predicates": True,
            "must_assign_semantic_roles": True,
            "must_include_hacking_risk_notes": True,
            "official_environment_reward_forbidden_as_feedback": True,
            "posthoc_oracle_evaluation_allowed": True,
            "no_v1_schema_exposure": True,
            "no_v1_diagnostic_metric_exposure": True,
            "no_v1_task_description_file_exposure": True,
            "formula_must_use_only_primitive_variables": True,
        }
        user_req = task_spec.get("bootstrap_requirements")
        if isinstance(user_req, dict):
            requirements.update(user_req)
        return requirements

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
