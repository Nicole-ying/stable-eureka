from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from eg_rsa.reward.schema import RewardSchema


@dataclass
class SchemaDiffResult:
    added_components: List[str] = field(default_factory=list)
    removed_components: List[str] = field(default_factory=list)
    changed_components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    added_event_rules: List[str] = field(default