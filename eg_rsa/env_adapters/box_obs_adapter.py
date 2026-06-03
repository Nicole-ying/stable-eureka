from __future__ import annotations

from typing import Any, Dict

import numpy as np


class BoxObsAdapter:
    """Generic adapter for vector observations.

    mapping example:
        {"x": 0, "y": 1, "vx": 2}
    """

    def __init__(self, mapping: Dict[str, int]):
        self.mapping = dict(mapping)

    def obs_to_map(self, obs: Any) -> Dict[str, float]:
        arr = np.asarray(obs, dtype=float).reshape(-1)
        out = {}
        for name, idx in self.mapping.items():
            if idx >= arr.size:
                raise ValueError(f"Index {idx} for {name} is outside observation size {arr.size}")
            out[name] = float(arr[idx])
        return out
