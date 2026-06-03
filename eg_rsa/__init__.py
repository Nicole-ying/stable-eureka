"""EG-RSA: Experience-Guided Reward Search Agent.

This package contains the incremental reward-editing framework built on top of
stable-eureka.  The first implementation keeps the original stable_eureka package
untouched and adds a separate runner, reward schema, diagnostics, memory, and
operator-constrained editing modules.
"""

from eg_rsa.runner import EGRSARunner

__all__ = ["EGRSARunner"]
