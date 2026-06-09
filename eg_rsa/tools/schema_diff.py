from __future__ import annotations

from typing import Any, Dict, List

from eg_rsa.reward.schema import RewardSchema


class SchemaDiffTool:
    """Compare two reward schemas and report actionable changes."""

    @staticmethod
    def diff(before: RewardSchema | Dict[str, Any], after: RewardSchema | Dict[str, Any]) -> Dict[str, Any]:
        b = before if isinstance(before, Reward