from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class EventEvaluator:
    """Evaluate boolean event flags from obs_map/action.

    Event specs are generic and environment-agnostic. Supported types:
      - threshold_abs: abs(var) <= threshold
      - threshold_gt: var > threshold
      - threshold_lt: var < threshold
      - all: all child events true
      - any: any child events true
      - not: child event false
      - action_nonzero: action contains any non-zero value
    """

    def __init__(self, event_specs: Optional[Dict[str, Dict[str, Any]]] = None):
        self.event_specs = event_specs or {}

    def evaluate(self, obs_map: Dict[str, float], action: Optional[Any] = None) -> Dict[str, bool]:
        events: Dict[str, bool] = {}
        for name in self.event_specs:
            events[name] = self._eval_one(name, obs_map, action, events)
        return events

    def _eval_one(
        self,
        name: str,
        obs_map: Dict[str, float],
        action: Optional[Any],
        cache: Dict[str, bool],
    ) -> bool:
        if name in cache:
            return cache[name]
        spec = self.event_specs.get(name, {})
        typ = spec.get("type")
        if typ == "threshold_abs":
            value = abs(float(obs_map.get(spec.get("var", ""), 0.0)))
            result = value <= float(spec.get("threshold", 0.0))
        elif typ == "threshold_gt":
            value = float(obs_map.get(spec.get("var", ""), 0.0))
            result = value > float(spec.get("threshold", 0.0))
        elif typ == "threshold_lt":
            value = float(obs_map.get(spec.get("var", ""), 0.0))
            result = value < float(spec.get("threshold", 0.0))
        elif typ == "all":
            result = all(self._eval_one(child, obs_map, action, cache) for child in spec.get("events", []))
        elif typ == "any":
            result = any(self._eval_one(child, obs_map, action, cache) for child in spec.get("events", []))
        elif typ == "not":
            child = spec.get("event")
            result = not self._eval_one(child, obs_map, action, cache) if child else False
        elif typ == "action_nonzero":
            result = False if action is None else bool(np.any(np.asarray(action) != 0))
        else:
            result = False
        cache[name] = bool(result)
        return bool(result)
