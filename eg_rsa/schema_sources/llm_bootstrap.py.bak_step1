from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from eg_rsa.llm.bootstrap_agent import BootstrapAgent
from eg_rsa.reward.bootstrap_schema_validator import BootstrapSchemaValidator
from eg_rsa.reward.schema_canonicalizer import SchemaCanonicalizer
from eg_rsa.reward.schema import RewardSchema
from eg_rsa.schema_sources.base import SchemaSource


class LLMBootstrapSchemaSource(SchemaSource):
    """Create initial schema and runtime diagnostic spec from primitive interface."""

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

        primitive_path = Path(cfg.get("primitive_interface_path", ""))
        if not primitive_path.exists():
            raise FileNotFoundError(f"primitive_interface_path not found: {primitive_path}")

        primitive_interface = json.loads(primitive_path.read_text(encoding="utf-8"))

        runtime_spec = self._build_runtime_spec_from_primitive_interface(primitive_interface)
        runtime_spec_path.write_text(
            yaml.safe_dump(runtime_spec, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        eg_cfg = self.config.setdefault("eg_rsa", {})
        eg_cfg["diagnostic_spec_path"] = str(runtime_spec_path)
        eg_cfg["task_description_inline"] = primitive_interface.get("task_description", "")

        if reuse_if_exists and schema_path.exists():
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

        task_description = primitive_interface.get("task_description", "") or self.task_description_loader()
        result = self.bootstrap_agent.generate_bootstrap(
            primitive_interface=primitive_interface,
            task_description=task_description,
        )

        result, canonical_report = SchemaCanonicalizer.canonicalize_bootstrap_result(
            result,
            primitive_interface,
        )

        schema_dict = result.get("initial_schema")
        if not isinstance(schema_dict, dict):
            raise ValueError("Bootstrap result must contain dict field initial_schema")

        validation = BootstrapSchemaValidator.validate_bootstrap_result(result, primitive_interface)

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
    def _build_runtime_spec_from_primitive_interface(primitive_interface: Dict[str, Any]) -> Dict[str, Any]:
        observation_mapping = primitive_interface.get("observation_mapping")
        if not isinstance(observation_mapping, dict) or not observation_mapping:
            observation_mapping = {}
            for idx, item in enumerate(primitive_interface.get("observation_variables", [])):
                if isinstance(item, dict) and item.get("name"):
                    observation_mapping[item["name"]] = idx

        events: Dict[str, Dict[str, Any]] = {}

        if "left_contact" in observation_mapping:
            events["left_contact"] = {
                "type": "threshold_gt",
                "var": "left_contact",
                "threshold": 0.5,
            }

        if "right_contact" in observation_mapping:
            events["right_contact"] = {
                "type": "threshold_gt",
                "var": "right_contact",
                "threshold": 0.5,
            }

        if "left_contact" in events or "right_contact" in events:
            events["contact"] = {
                "type": "any",
                "events": [x for x in ["left_contact", "right_contact"] if x in events],
            }

        if "left_contact" in events and "right_contact" in events:
            events["both_contact"] = {
                "type": "all",
                "events": ["left_contact", "right_contact"],
            }

        events["engine_on"] = {
            "type": "action_nonzero",
        }

        task_metrics = {
            "horizontal_centering": {
                "type": "raw_abs_inverse",
                "inputs": ["x"],
            },
            "velocity_smoothness": {
                "type": "raw_abs_inverse",
                "inputs": ["vx", "vy"],
            },
            "attitude_smoothness": {
                "type": "raw_abs_inverse",
                "inputs": ["angle", "angular_velocity"],
            },
            "energy_cost": {
                "type": "action_cost",
            },
            "progress": {
                "type": "metric_mean",
                "metrics": [
                    "horizontal_centering",
                    "velocity_smoothness",
                    "attitude_smoothness",
                ],
            },
        }

        if "both_contact" in events:
            task_metrics["two_leg_contact_evidence"] = {
                "type": "event_score",
                "event": "both_contact",
            }

        return {
            "source": "primitive_interface_generated_runtime_spec",
            "observation_mapping": observation_mapping,
            "action_variables": primitive_interface.get("action_variables", []),
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
