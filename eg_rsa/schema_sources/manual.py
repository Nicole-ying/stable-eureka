from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from eg_rsa.reward.schema import RewardSchema
from eg_rsa.schema_sources.base import SchemaSource


class ManualSchemaSource(SchemaSource):
    """Load the initial reward schema from a JSON file.

    This preserves the original EG-RSA behavior used by the V1 experiments.
    It is also useful as a stable fallback and for ablations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _schema_path(self) -> Path:
        eg_cfg = self.config.get("eg_rsa", {}) or {}
        source_cfg = eg_cfg.get("schema_source", {}) or {}

        path = source_cfg.get("initial_schema_path") or eg_cfg.get("initial_schema_path")
        if not path:
            raise ValueError(
                "ManualSchemaSource requires either "
                "eg_rsa.schema_source.initial_schema_path or eg_rsa.initial_schema_path."
            )

        return Path(path)

    def load_or_create(self) -> RewardSchema:
        path = self._schema_path()
        if not path.exists():
            raise FileNotFoundError(f"Initial schema file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            return RewardSchema.from_dict(json.load(f))
