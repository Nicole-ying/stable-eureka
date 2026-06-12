from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from eg_rsa.llm.bootstrap_agent import BootstrapAgent
from eg_rsa.reward.bootstrap_schema_validator import BootstrapSchemaValidator
from eg_rsa.reward.schema_canonicalizer import SchemaCanonicalizer
from eg_rsa.reward.schema import RewardSchema
from eg_rsa.schema_sources.base import SchemaSource
from eg_rsa.schema_sources.eureka_like_interface import EurekaLikeInterfaceBuilder


class LLMBootstrapSchemaSource(SchemaSource):
    """Create initial schema and runtime diagnostic spec.

    Supported input boundaries:
    1. source-aware bootstrap, preferred for V2.1:
       anonymous task/source summary -> BootstrapAgent -> primitive_interface + AST schema.
    2. task-file-to-interface path, kept for compatibility:
       eureka_like_task_file -> generated primitive_interface -> BootstrapAgent.
    3. direct primitive_interface path, kept for old controlled experiments.

    Source-aware mode is intentionally robust: the LLM may infer a useful
    primitive_interface but still produce a malformed executable schema. In that
    case we preserve the failed schema for audit and fall back to a deterministic
    safe scaffold generated from the inferred interface.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        llm_client: Optional[Any],
        task_description_loader: Optional[Callable[[], str]] = None,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.llm_client = llm_client
        self.task_description_loader = task_description_loader or (lambda: "")
        self.bootstrap_agent = BootstrapAgent(llm_client=llm_client)

    def _source_config(self) -> Dict[str, Any]:
        eg_cfg = self.config.get("eg_rsa", {}) or {}
        source_cfg = eg_cfg.get("schema_source")
        if isinstance(source_cfg, dict) and source_cfg.get("type") == "llm_bootstrap":
            return dict(source_cfg)
        bootstrap_cfg = eg_cfg.get("bootstrap", {}) or {}
        return dict(bootstrap_cfg)

    def load_or_create(self) -> RewardSchema:
        cfg = self._source_config()
        output_subdir = cfg.get("output_subdir", "bootstrap")
        bootstrap_dir = self.output_dir / output_subdir
        bootstrap_dir.mkdir(parents=True, exist_ok=True)

        schema_path = bootstrap_dir / "generated_initial_schema.json"
        runtime_spec_path = bootstrap_dir / "generated_diagnostics.yml"
        blueprint_path = bootstrap_dir / "reward_blueprint.json"
        reuse_if_exists = bool(cfg.get("reuse_if_exists", True))

        source_aware = self._is_source_aware(cfg)
        bootstrap_result: Optional[Dict[str, Any]] = None

        if source_aware:
            try:
                bootstrap_result = self._run_source_aware_bootstrap(cfg, bootstrap_dir)
            except Exception as exc:
                self._persist_bootstrap_failure(bootstrap_dir, exc)
                raise RuntimeError(
                    "Source-aware LLM bootstrap failed before producing a valid result. "
                    f"Raw prompt/response have been saved under: {bootstrap_dir}."
                ) from exc
            primitive_interface = bootstrap_result.get("primitive_interface")
            if not isinstance(primitive_interface, dict):
                raise ValueError("Source-aware bootstrap result missing primitive_interface")
        else:
            primitive_interface = self._load_or_generate_primitive_interface(cfg)

        runtime_spec = self._build_runtime_spec_from_primitive_interface(primitive_interface)
        runtime_spec_path.write_text(
            yaml.safe_dump(runtime_spec, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        eg_cfg = self.config.setdefault("eg_rsa", {})
        eg_cfg["diagnostic_spec_path"] = str(runtime_spec_path)
        eg_cfg["task_description_inline"] = primitive_interface.get("task_description", "")

        if reuse_if_exists and schema_path.exists() and not source_aware:
            schema_dict = json.loads(schema_path.read_text(encoding="utf-8"))
            blueprint = {}
            if blueprint_path.exists():
                blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))

            schema_dict, canonical_report = SchemaCanonicalizer.canonicalize_schema(
                schema=schema_dict,
                primitive_interface=primitive_interface,
                reward_blueprint=blueprint,
            )
            validation = BootstrapSchemaValidator.validate_schema(
                schema_dict,
                primitive_interface,
                reward_blueprint=blueprint,
            )
            self._write_json(bootstrap_dir / "schema_canonicalization_report.json", canonical_report)
            self._write_json(bootstrap_dir / "canonical_initial_schema.json", schema_dict)
            self._write_json(bootstrap_dir / "bootstrap_validation.json", validation.to_dict())
            self._write_json(schema_path, schema_dict)
            if not validation.ok:
                raise ValueError(f"Reused generated_initial_schema.json failed validation: {validation.errors}")
            return RewardSchema.from_dict(schema_dict)

        if bootstrap_result is None:
            task_description = primitive_interface.get("task_description", "") or self.task_description_loader()
            try:
                bootstrap_result = self.bootstrap_agent.generate_bootstrap(
                    primitive_interface=primitive_interface,
                    task_description=task_description,
                )
            except Exception as exc:
                self._persist_bootstrap_failure(bootstrap_dir, exc)
                raise RuntimeError(
                    "LLM bootstrap failed before producing a valid JSON object. "
                    f"Raw prompt/response have been saved under: {bootstrap_dir}. "
                    "Inspect bootstrap_response_malformed.txt and bootstrap_parse_error.json."
                ) from exc

        result, canonical_report = SchemaCanonicalizer.canonicalize_bootstrap_result(
            bootstrap_result,
            primitive_interface,
        )
        schema_dict = result.get("initial_schema")
        if not isinstance(schema_dict, dict):
            raise ValueError("Bootstrap result must contain dict field initial_schema")

        validation = BootstrapSchemaValidator.validate_bootstrap_result(result, primitive_interface)
        if not validation.ok and source_aware and bool(cfg.get("fallback_on_invalid_schema", True)):
            self._write_json(bootstrap_dir / "llm_source_aware_invalid_result.json", result)
            self._write_json(bootstrap_dir / "llm_source_aware_invalid_validation.json", validation.to_dict())
            self._write_json(bootstrap_dir / "llm_source_aware_invalid_canonicalization_report.json", canonical_report)
            result, canonical_report, validation = self._fallback_to_safe_scaffold(
                primitive_interface=primitive_interface,
                bootstrap_dir=bootstrap_dir,
                validation_errors=validation.errors,
            )
            schema_dict = result.get("initial_schema")

        self._write_text(bootstrap_dir / "bootstrap_prompt.txt", self.bootstrap_agent.last_prompt)
        self._write_text(bootstrap_dir / "bootstrap_response.txt", self.bootstrap_agent.last_response_text)
        self._write_json(bootstrap_dir / "bootstrap_response.json", result)
        self._write_json(bootstrap_dir / "schema_canonicalization_report.json", canonical_report)
        self._write_json(bootstrap_dir / "canonical_initial_schema.json", schema_dict)
        self._write_json(bootstrap_dir / "bootstrap_validation.json", validation.to_dict())
        self._write_json(schema_path, schema_dict)
        self._write_json(blueprint_path, result.get("reward_blueprint", {}) or {})
        self._write_json(bootstrap_dir / "bootstrap_agent_diagnostics.json", result.get("diagnostics", {}) or {})
        self._write_json(bootstrap_dir / "bootstrap_report.json", result.get("bootstrap_report", {}) or {})

        if not validation.ok:
            raise ValueError(f"Bootstrap schema failed validation: {validation.errors}")
        return RewardSchema.from_dict(schema_dict)

    @staticmethod
    def _is_source_aware(cfg: Dict[str, Any]) -> bool:
        mode = str(cfg.get("input_mode", "")).lower()
        return bool(cfg.get("source_aware", False)) or mode in {"source", "source_aware", "anonymous_source"}

    def _fallback_to_safe_scaffold(
        self,
        primitive_interface: Dict[str, Any],
        bootstrap_dir: Path,
        validation_errors: Any,
    ):
        """Replace invalid source-aware LLM schema with deterministic safe scaffold.

        This keeps the source-aware input boundary intact: the primitive interface is
        still inferred from the anonymous source prompt. Only the executable schema
        is replaced when the LLM-produced AST is structurally invalid.
        """
        fallback = BootstrapAgent._fallback_bootstrap(
            primitive_interface,
            primitive_interface.get("task_description", ""),
        )
        fallback["primitive_interface"] = primitive_interface
        fallback.setdefault("bootstrap_report", {})
        fallback["bootstrap_report"].update(
            {
                "source_aware_bootstrap": True,
                "primitive_interface_generated": True,
                "schema_source": "deterministic_safe_scaffold_after_invalid_llm_schema",
                "llm_schema_replaced": True,
                "llm_validation_errors": list(validation_errors or []),
            }
        )
        fallback.setdefault("diagnostics", {})
        fallback["diagnostics"].setdefault("risk_notes", [])
        fallback["diagnostics"]["risk_notes"].append(
            "The LLM-produced source-aware schema failed validation and was replaced by a deterministic safe scaffold."
        )
        canonical_fallback, fallback_report = SchemaCanonicalizer.canonicalize_bootstrap_result(
            fallback,
            primitive_interface,
        )
        fallback_schema = canonical_fallback.get("initial_schema")
        fallback_validation = BootstrapSchemaValidator.validate_bootstrap_result(
            canonical_fallback,
            primitive_interface,
        )
        self._write_json(bootstrap_dir / "safe_scaffold_bootstrap_response.json", canonical_fallback)
        self._write_json(bootstrap_dir / "safe_scaffold_validation.json", fallback_validation.to_dict())
        return canonical_fallback, fallback_report, fallback_validation

    def _run_source_aware_bootstrap(self, cfg: Dict[str, Any], bootstrap_dir: Path) -> Dict[str, Any]:
        task_path = cfg.get("eureka_like_task_path") or cfg.get("task_file_path") or cfg.get("source_task_path")
        if not task_path:
            raise ValueError("source-aware bootstrap requires eureka_like_task_path, task_file_path, or source_task_path")
        task_spec = self._load_task_file(Path(str(task_path)))
        result = self.bootstrap_agent.generate_bootstrap_from_source(task_spec)
        primitive_interface = result.get("primitive_interface")
        if not isinstance(primitive_interface, dict):
            raise ValueError("source-aware bootstrap returned no primitive_interface")
        interface_dir = self.output_dir / cfg.get("interface_output_subdir", "interface")
        self._write_json(interface_dir / "anonymous_source_input.json", task_spec)
        self._write_json(interface_dir / "generated_primitive_interface.json", primitive_interface)
        self._write_json(
            interface_dir / "interface_generation_report.json",
            {
                "source": "source_aware_bootstrap_agent",
                "source_path": str(task_path),
                "output_path": str(interface_dir / "generated_primitive_interface.json"),
                "identity_hidden_from_llm": True,
                "raw_env_code_input": bool(primitive_interface.get("raw_env_code_input", False)),
                "notes": [
                    "BootstrapAgent inferred the primitive interface and initial schema in one LLM call.",
                    "The runtime environment name is not included in the bootstrap prompt.",
                    "The primitive interface is an internal audit artifact, not the user-facing entry point.",
                ],
            },
        )
        self.config.setdefault("eg_rsa", {}).setdefault("schema_source", {})[
            "generated_primitive_interface_path"
        ] = str(interface_dir / "generated_primitive_interface.json")
        return result

    @staticmethod
    def _load_task_file(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"task file not found: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yml", ".yaml"}:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"task file must contain a JSON/YAML object: {path}")
        return data

    def _persist_bootstrap_failure(self, bootstrap_dir: Path, exc: Exception) -> None:
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        self._write_text(bootstrap_dir / "bootstrap_prompt.txt", self.bootstrap_agent.last_prompt)
        self._write_text(bootstrap_dir / "bootstrap_response_malformed.txt", self.bootstrap_agent.last_response_text)
        self._write_text(bootstrap_dir / "bootstrap_response.txt", self.bootstrap_agent.last_response_text)
        self._write_json(
            bootstrap_dir / "bootstrap_parse_error.json",
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "response_length": len(self.bootstrap_agent.last_response_text or ""),
                "prompt_length": len(self.bootstrap_agent.last_prompt or ""),
                "traceback": traceback.format_exc(),
                "notes": [
                    "LLM returned text before a valid bootstrap result could be parsed.",
                    "The raw response is saved for debugging and possible manual repair.",
                    "No generated_initial_schema.json was written from this malformed response.",
                ],
            },
        )

    def _load_or_generate_primitive_interface(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        interface_subdir = cfg.get("interface_output_subdir", "interface")
        interface_dir = self.output_dir / interface_subdir
        eureka_like_task_path = cfg.get("eureka_like_task_path") or cfg.get("task_file_path")
        primitive_path_value = cfg.get("primitive_interface_path", "")
        primitive_path_is_auto = str(primitive_path_value).lower() in {"", "auto", "generated"}

        if eureka_like_task_path and primitive_path_is_auto:
            primitive_interface = EurekaLikeInterfaceBuilder.build_from_file(
                task_file_path=eureka_like_task_path,
                output_dir=interface_dir,
            )
            self.config.setdefault("eg_rsa", {}).setdefault("schema_source", {})[
                "generated_primitive_interface_path"
            ] = str(interface_dir / "generated_primitive_interface.json")
            return primitive_interface

        if eureka_like_task_path and not primitive_path_is_auto:
            EurekaLikeInterfaceBuilder.build_from_file(
                task_file_path=eureka_like_task_path,
                output_dir=interface_dir,
            )

        primitive_path = Path(str(primitive_path_value))
        if not primitive_path.exists():
            if eureka_like_task_path:
                raise FileNotFoundError(
                    "primitive_interface_path was not found, and primitive_interface_path is not set to auto. "
                    f"Set primitive_interface_path: auto to use generated interface from {eureka_like_task_path}. "
                    f"Missing path: {primitive_path}"
                )
            raise FileNotFoundError(f"primitive_interface_path not found: {primitive_path}")
        primitive_interface = json.loads(primitive_path.read_text(encoding="utf-8"))
        self._write_json(interface_dir / "loaded_primitive_interface.json", primitive_interface)
        self._write_json(
            interface_dir / "interface_generation_report.json",
            {
                "source": "existing_primitive_interface_path",
                "primitive_interface_path": str(primitive_path),
                "output_path": str(interface_dir / "loaded_primitive_interface.json"),
                "raw_env_code_input": False,
                "env_code_parser": "planned_not_current",
                "notes": [
                    "Backward-compatible path: primitive_interface_path was provided directly.",
                    "For the source-aware path, set source_aware: true and provide an anonymous task/source file.",
                ],
            },
        )
        return primitive_interface

    @staticmethod
    def _build_runtime_spec_from_primitive_interface(primitive_interface: Dict[str, Any]) -> Dict[str, Any]:
        observation_mapping = primitive_interface.get("observation_mapping")
        if not isinstance(observation_mapping, dict) or not observation_mapping:
            observation_mapping = {}
            for idx, item in enumerate(primitive_interface.get("observation_variables", []) or []):
                if isinstance(item, dict) and item.get("name"):
                    observation_mapping[item["name"]] = idx
        observation_variables = primitive_interface.get("observation_variables", []) or []
        action_variables = primitive_interface.get("action_variables", []) or []
        bool_vars = []
        numeric_vars = []
        for item in observation_variables:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = str(item["name"])
            typ = str(item.get("type", "")).lower()
            if typ in {"bool", "boolean"}:
                bool_vars.append(name)
            else:
                numeric_vars.append(name)
        action_names = [str(item["name"]) for item in action_variables if isinstance(item, dict) and item.get("name")]
        events: Dict[str, Dict[str, Any]] = {}
        for name in bool_vars:
            events[name] = {"type": "threshold_gt", "var": name, "threshold": 0.5}
        contact_like = [name for name in bool_vars if "contact" in name.lower() or "touch" in name.lower() or "ground" in name.lower()]
        if contact_like:
            events["any_contact_evidence"] = {"type": "any", "events": contact_like}
        if len(contact_like) >= 2:
            events["all_contact_evidence"] = {"type": "all", "events": contact_like[:4]}
        if action_names:
            events["action_nonzero"] = {"type": "action_nonzero"}
        task_metrics: Dict[str, Dict[str, Any]] = {}

        def add_raw_abs_inverse_metric(metric_name: str, inputs: list[str]) -> None:
            valid = [x for x in inputs if x in observation_mapping]
            if valid:
                task_metrics[metric_name] = {"type": "raw_abs_inverse", "inputs": valid}

        add_raw_abs_inverse_metric("position_centering", [x for x in ["x", "position", "pos", "horizontal_position"] if x in observation_mapping])
        add_raw_abs_inverse_metric("velocity_smoothness", [x for x in ["vx", "vy", "velocity", "horizontal_speed", "vertical_speed"] if x in observation_mapping])
        add_raw_abs_inverse_metric("attitude_smoothness", [x for x in ["angle", "angular_velocity", "hull_angle", "hull_angular_velocity", "body_angle", "body_angular_velocity"] if x in observation_mapping])
        if not task_metrics and numeric_vars:
            add_raw_abs_inverse_metric("state_smoothness", numeric_vars[:4])
        if action_names:
            task_metrics["energy_cost"] = {"type": "action_cost"}
        if "all_contact_evidence" in events:
            task_metrics["contact_evidence"] = {"type": "event_score", "event": "all_contact_evidence"}
        elif "any_contact_evidence" in events:
            task_metrics["contact_evidence"] = {"type": "event_score", "event": "any_contact_evidence"}
        progress_metrics = [name for name in ["position_centering", "velocity_smoothness", "attitude_smoothness", "state_smoothness", "contact_evidence"] if name in task_metrics]
        if progress_metrics:
            task_metrics["progress"] = {"type": "metric_mean", "metrics": progress_metrics}
        return {
            "source": "primitive_interface_generated_runtime_spec",
            "input_boundary": primitive_interface.get("input_boundary", "primitive_interface_conditioned"),
            "identity_hidden_from_llm": bool(primitive_interface.get("identity_hidden_from_llm", False)),
            "raw_env_code_input": bool(primitive_interface.get("raw_env_code_input", False)),
            "eureka_like_input_status": primitive_interface.get("env_code_parser", "planned_not_current"),
            "observation_mapping": observation_mapping,
            "action_variables": action_variables,
            "action_mapping": primitive_interface.get("action_mapping", {}),
            "events": events,
            "task_metrics": task_metrics,
        }

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or "", encoding="utf-8")
