from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ExperimentMode:
    """Ablation switches for EG-RSA.

    Full EG-RSA keeps memory, attribution, hack diagnostics, LLM editing, and
    operator constraints enabled. Other presets disable one or more modules for
    baseline and ablation experiments.
    """

    use_memory: bool = True
    use_attribution: bool = True
    use_hack_detector: bool = True
    use_llm_edit: bool = True
    use_operator_constraints: bool = True
    free_rewrite: bool = False
    one_shot: bool = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ExperimentMode":
        raw = config.get("eg_rsa", {}).get("experiment_mode", {}) or {}
        preset = raw.get("preset", "full")
        mode = cls()
        if preset == "full":
            pass
        elif preset == "one_shot":
            mode.one_shot = True
            mode.use_memory = False
            mode.use_attribution = False
            mode.use_hack_detector = False
            mode.use_llm_edit = False
        elif preset == "free_rewrite":
            mode.free_rewrite = True
            mode.use_memory = False
            mode.use_attribution = False
            mode.use_hack_detector = False
            mode.use_operator_constraints = False
        elif preset == "wo_memory":
            mode.use_memory = False
        elif preset == "wo_attribution":
            mode.use_attribution = False
        elif preset == "wo_hack_detector":
            mode.use_hack_detector = False
        elif preset == "fallback_only":
            mode.use_llm_edit = False
        else:
            raise ValueError(f"Unsupported EG-RSA experiment preset: {preset}")

        for key in mode.to_dict().keys():
            if key in raw:
                setattr(mode, key, bool(raw[key]))
        return mode

    def to_dict(self) -> Dict[str, bool]:
        return {
            "use_memory": self.use_memory,
            "use_attribution": self.use_attribution,
            "use_hack_detector": self.use_hack_detector,
            "use_llm_edit": self.use_llm_edit,
            "use_operator_constraints": self.use_operator_constraints,
            "free_rewrite": self.free_rewrite,
            "one_shot": self.one_shot,
        }
