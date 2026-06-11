from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class ActionPrimitiveMapper:
    """Map raw environment actions into primitive formula variables.

    The mapper is driven by primitive_interface.action_mapping instead of
    hard-coded environment semantics.

    Supported mapping styles:

      discrete_lookup:
        {
          "type": "discrete_lookup",
          "variables": {
            "main_engine": {"2": 1.0, "default": 0.0},
            "side_engine": {"1": -1.0, "3": 1.0, "default": 0.0}
          }
        }

      continuous_indices:
        {
          "type": "continuous_indices",
          "variables": {
            "main_engine": 0,
            "side_engine": 1
          }
        }

    If no mapping is provided, it falls back to generic action_i variables.
    For backward compatibility, if the declared action variables include
    main_engine and side_engine and the action is discrete, it reproduces the
    LunarLander mapping.
    """

    def __init__(
        self,
        mapping_spec: Optional[Dict[str, Any]] = None,
        action_variables: Optional[List[Dict[str, Any]]] = None,
    ):
        self.mapping_spec = mapping_spec or {}
        self.action_variables = action_variables or []

    @classmethod
    def from_runtime_spec(cls, runtime_spec: Dict[str, Any]) -> "ActionPrimitiveMapper":
        return cls(
            mapping_spec=runtime_spec.get("action_mapping", {}) or {},
            action_variables=runtime_spec.get("action_variables", []) or [],
        )

    @classmethod
    def from_primitive_interface(cls, primitive_interface: Dict[str, Any]) -> "ActionPrimitiveMapper":
        return cls(
            mapping_spec=primitive_interface.get("action_mapping", {}) or {},
            action_variables=primitive_interface.get("action_variables", []) or [],
        )

    def map(self, action: Optional[Any]) -> Dict[str, float]:
        if action is None:
            return self._zero_declared_variables()

        mapping_type = self.mapping_spec.get("type")

        if mapping_type == "discrete_lookup":
            return self._map_discrete_lookup(action)

        if mapping_type == "continuous_indices":
            return self._map_continuous_indices(action)

        return self._fallback_map(action)

    def _zero_declared_variables(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for item in self.action_variables:
            if isinstance(item, dict) and item.get("name"):
                out[str(item["name"])] = 0.0
        return out

    def _map_discrete_lookup(self, action: Any) -> Dict[str, float]:
        arr = np.asarray(action, dtype=float).reshape(-1)
        a = int(round(float(arr[0]))) if arr.size else 0

        variables = self.mapping_spec.get("variables", {}) or {}
        out: Dict[str, float] = {}

        for name, table in variables.items():
            if not isinstance(table, dict):
                continue
            default = float(table.get("default", 0.0) or 0.0)
            value = table.get(str(a), table.get(a, default))
            out[str(name)] = float(value)

        return out

    def _map_continuous_indices(self, action: Any) -> Dict[str, float]:
        arr = np.asarray(action, dtype=float).reshape(-1)
        variables = self.mapping_spec.get("variables", {}) or {}
        out: Dict[str, float] = {}

        for name, index in variables.items():
            try:
                idx = int(index)
            except Exception:
                out[str(name)] = 0.0
                continue
            out[str(name)] = float(arr[idx]) if idx < arr.size else 0.0

        return out

    def _fallback_map(self, action: Any) -> Dict[str, float]:
        arr = np.asarray(action, dtype=float).reshape(-1)
        names = [
            str(item.get("name"))
            for item in self.action_variables
            if isinstance(item, dict) and item.get("name")
        ]

        # Backward-compatible LunarLander fallback.
        if set(names) >= {"main_engine", "side_engine"} and arr.size == 1:
            a = int(round(float(arr[0])))
            return {
                "main_engine": 1.0 if a == 2 else 0.0,
                "side_engine": -1.0 if a == 1 else (1.0 if a == 3 else 0.0),
            }

        out: Dict[str, float] = {}
        for i, value in enumerate(arr):
            out[f"action_{i}"] = float(value)

        for i, name in enumerate(names):
            out[name] = float(arr[i]) if i < arr.size else 0.0

        return out
