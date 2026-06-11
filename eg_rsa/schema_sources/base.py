from __future__ import annotations

from abc import ABC, abstractmethod

from eg_rsa.reward.schema import RewardSchema


class SchemaSource(ABC):
    """Abstract source of the initial reward schema.

    A SchemaSource is intentionally small. It does not train policies, edit
    schemas, run audits, or update memory. It only returns the initial
    RewardSchema used by EGRSARunner.
    """

    @abstractmethod
    def load_or_create(self) -> RewardSchema:
        """Load or create the initial reward schema for a run."""
        raise NotImplementedError
