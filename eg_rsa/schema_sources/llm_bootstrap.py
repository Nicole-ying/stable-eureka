from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from eg_rsa.llm.bootstrap_agent import BootstrapAgent
from eg_rsa.reward.bootstrap_schema_validator import BootstrapSchemaValidator
from eg_rsa.reward.schema import RewardSchema
from eg_rsa.schema_sources.base import SchemaSource


class LLMBootstrapSchemaSource(SchemaSource):
    """Create the initial schema using an LLM bootstrap stage."""

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
        diagnostics_path = bootstrap_dir / "generated_diagnostics.yml"
        reuse_if_exists = bool(cfg.get("reuse_if_exists", True))

        primitive_path = Path(cfg.get("primitive_interface_path", ""))
        if not primitive_path.exists():
            raise FileNotFoundError(f"primitive_interface_path not found: {primitive_path}")

        primitive_interface = json.loads(primitive_path.read_text(encoding="utf-8"))

        if reuse_if_exists and schema_path.exists():
            schema_dict = json.loads(schema_path.read_text(encoding="utf-8"))
            validation = BootstrapSchemaValidator.validate_schema(schema_dict, primitive_interface)
            self._write_json(bootstrap_dir / "bootstrap_validation.json", validation.to_dict())
            if not validation.ok:
                raise ValueError(f"Reused generated_initial_schema.json failed validation: {validation.errors}")
            return RewardSchema.from_dict(schema_dict)

        task_description = self.task_description_loader()
        result = self.bootstrap_agent.generate_bootstrap(
            primitive_interface=primitive_interface,
            task_description=task_description,
        )

        schema_dict = result.get("initial_schema")
        if not isinstance(schema_dict, dict):
            raise ValueError("Bootstrap result must contain dict field initial_schema")

        validation = BootstrapSchemaValidator.validate_schema(schema_dict, primitive_interface)
        self._write_text(bootstrap_dir / "bootstrap_prompt.txt", self.bootstrap_agent.last_prompt)
        self._write_text(bootstrap_dir / "bootstrap_response.txt", self.bootstrap_agent.last_response_text)
        self._write_json(bootstrap_dir / "bootstrap_response.json", result)
        self._write_json(bootstrap_dir / "bootstrap_validation.json", validation.to_dict())
        self._write_json(schema_path, schema_dict)

        diagnostics = result.get("diagnostics", {}) or {}
        diagnostics_path.write_text(yaml.safe_dump(diagnostics, allow_unicode=True, sort_keys=False), encoding="utf-8")

        self._write_json(
            bootstrap_dir / "bootstrap_report.json",
            result.get("bootstrap_report", {}) or {},
        )

        if not validation.ok:
            raise ValueError(f"Bootstrap schema failed validation: {validation.errors}")

        return RewardSchema.from_dict(schema_dict)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or "", encoding="utf-8")
