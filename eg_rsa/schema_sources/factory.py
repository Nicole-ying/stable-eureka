from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from eg_rsa.schema_sources.base import SchemaSource
from eg_rsa.schema_sources.manual import ManualSchemaSource
from eg_rsa.schema_sources.llm_bootstrap import LLMBootstrapSchemaSource


def _infer_schema_source_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Infer schema-source config while preserving old configs.

    Preferred new config:

        eg_rsa:
          schema_source:
            type: manual | llm_bootstrap

    Backward compatibility:

        old V1 config without schema_source -> manual
        temporary V2 config with bootstrap.enabled=true -> llm_bootstrap
    """

    eg_cfg = config.get("eg_rsa", {}) or {}

    source_cfg = eg_cfg.get("schema_source")
    if isinstance(source_cfg, dict) and source_cfg.get("type"):
        return dict(source_cfg)

    bootstrap_cfg = eg_cfg.get("bootstrap", {}) or {}
    if bool(bootstrap_cfg.get("enabled", False)):
        inferred = dict(bootstrap_cfg)
        inferred["type"] = "llm_bootstrap"
        return inferred

    return {
        "type": "manual",
        "initial_schema_path": eg_cfg.get("initial_schema_path"),
    }


def build_schema_source(
    config: Dict[str, Any],
    output_dir: Path,
    llm_client: Optional[Any] = None,
    task_description_loader: Optional[Callable[[], str]] = None,
) -> SchemaSource:
    """Build a schema source from config.

    The runner calls this factory once and then only uses
    schema_source.load_or_create(). This avoids scattering V1/V2 branches
    through the training loop.
    """

    source_cfg = _infer_schema_source_config(config)
    source_type = str(source_cfg.get("type", "manual")).lower()

    if source_type == "manual":
        return ManualSchemaSource(config=config)

    if source_type == "llm_bootstrap":
        return LLMBootstrapSchemaSource(
            config=config,
            output_dir=output_dir,
            llm_client=llm_client,
            task_description_loader=task_description_loader,
        )

    raise ValueError(
        f"Unknown schema_source type: {source_type}. "
        "Expected one of: manual, llm_bootstrap."
    )
