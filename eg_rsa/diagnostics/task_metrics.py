from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class TaskMetricEvaluator:
    """Evaluate task-level diagnostic metrics from obs_map, action, and events.

    Metrics are not official rewards. They are task-semantics proxies used for
    diagnostics and for constrained metric-based reward components.
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
            elif metric_type == "target_region":
                metrics[name] = self._target_region(obs_map, spec)
            elif metric_type == "bounded_stability":
                metrics[name] = self._bounded_stability(obs_map, spec)
            elif metric_type == "action_cost":
                metrics[name] = self._action_cost(action)
            elif metric_type == "event_success":
                metrics[name] = 1.0 if bool(events.get(spec.get("event", name), False)) else 0.0
            elif metric_type == "event_score":
                metrics[name] = 1.0 if bool(events.get(spec.get("event", name), False)) else 0.0
            elif metric_type == "metric_product":
                metrics[name] = self._metric_product(metrics, spec)
            elif metric_type == "metric_mean":
                metrics[name] = self._metric_mean(metrics, spec)
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

    def _target_region(self, obs_map: Dict[str, float], spec: Dict[str, Any]) -> float:
        """Continuous score for being inside or near a target region.

        This avoids treating a safe landing area as a single point. Each axis has
        a tolerance. Inside the region scores 1. Outside the region decays
        smoothly according to the normalized distance outside the boundary.
        """

        inputs = list(spec.get("inputs", []))
        center = np.asarray(spec.get("center", [0.0 for _ in inputs]), dtype=float)
        tolerance = np.asarray(spec.get("tolerance", [1.0 for _ in inputs]), dtype=float)
        tolerance = np.maximum(tolerance, 1e-6)
        values = self._get_values(obs_map, inputs)
        if values.size == 0:
            return 0.0
        outside = np.maximum(np.abs(values - center) - tolerance, 0.0) / tolerance
        outside_norm = float(np.linalg.norm(outside))
        return float(1.0 / (1.0 + outside_norm))

    def _bounded_stability(self, obs_map: Dict[str, float], spec: Dict[str, Any]) -> float:
        values = np.abs(self._get_values(obs_map, list(spec.get("inputs", []))))
        scales = np.asarray(spec.get("scales", [1.0 for _ in values]), dtype=float)
        if values.size == 0:
            return 0.0
        if scales.size != values.size:
            scales = np.ones_like(values)
        scales = np.maximum(scales, 1e-6)
        normalized = values / scales
        return float(1.0 / (1.0 + float(np.sum(normalized))))

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

    @staticmethod
    def _metric_product(metrics: Dict[str, float], spec: Dict[str, Any]) -> float:
        names = list(spec.get("metrics", []))
        if not names:
            return 0.0
        value = 1.0
        for name in names:
            value *= float(metrics.get(name, 0.0))
        return float(value)

    @staticmethod
    def _metric_mean(metrics: Dict[str, float], spec: Dict[str, Any]) -> float:
        names = list(spec.get("metrics", []))
        if not names:
            return 0.0
        return float(sum(float(metrics.get(name, 0.0)) for name in names) / max(1, len(names)))
