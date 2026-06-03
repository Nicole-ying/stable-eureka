from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class TaskMetricEvaluator:
    """Evaluate task-level diagnostic metrics from obs_map, action, and events.

    Metrics are not training rewards. They are used for diagnostics only.
    A metric spec is a dictionary such as:

    {
        "progress": {"type": "distance_to_target", "inputs": ["x", "y"], "target": [0, 0]},
        "stability": {"type": "bounded_stability", "inputs": ["vx", "vy", "angle"]},
        "success": {"type": "event_success", "event": "stable_landing_condition"}
    }
    """

    def __init__(self, metric_specs: Optional[Dict[str, Dict[str, Any]]] = None):
        self.metric_specs = metric_specs or {}

    def evaluate(
        self,
        obs_map: Dict[str, float],
        action: Optional[Any] = None,
        events: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, float]:
        events = events or {}
        metrics: Dict[str, float] = {}
        for name, spec in self.metric_specs.items():
            metric_type = spec.get("type")
            if metric_type == "distance_to_target":
                metrics[name] = self._distance_to_target(obs_map, spec)
            elif metric_type == "bounded_stability":
                metrics[name] = self._bounded_stability(obs_map, spec)
            elif metric_type == "action_cost":
                metrics[name] = self._action_cost(action)
            elif metric_type == "event_success":
                metrics[name] = 1.0 if bool(events.get(spec.get("event", name), False)) else 0.0
            elif metric_type == "raw_abs_inverse":
                metrics[name] = self._raw_abs_inverse(obs_map, spec)
            else:
                metrics[name] = 0.0
        return metrics

    @staticmethod
    def _get_values(obs_map: Dict[str, float], inputs: List[str]) -> np.ndarray:
        return np.asarray([float(obs_map.get(key, 0.0)) for key in inputs], dtype=float)

    def _distance_to_target(self, obs_map: Dict[str, float], spec: Dict[str, Any]) -> float:
        inputs = list(spec.get("inputs", []))
        target = np.asarray(spec.get("target", [0.0 for _ in inputs]), dtype=float)
        values = self._get_values(obs_map, inputs)
        if values.size == 0:
            return 0.0
        distance = float(np.linalg.norm(values - target))
        return float(1.0 / (1.0 + distance))

    def _bounded_stability(self, obs_map: Dict[str, float], spec: Dict[str, Any]) -> float:
        values = np.abs(self._get_values(obs_map, list(spec.get("inputs", []))))
        if values.size == 0:
            return 0.0
        return float(1.0 / (1.0 + float(np.sum(values))))

    @staticmethod
    def _action_cost(action: Optional[Any]) -> float:
        if action is None:
            return 0.0
        arr = np.asarray(action, dtype=float)
        return float(np.sum(np.square(arr)))

    def _raw_abs_inverse(self, obs_map: Dict[str, float], spec: Dict[str, Any]) -> float:
        values = np.abs(self._get_values(obs_map, list(spec.get("inputs", []))))
        if values.size == 0:
            return 0.0
        return float(1.0 / (1.0 + float(np.sum(values))))
