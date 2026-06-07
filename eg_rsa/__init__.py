"""EG-RSA: Experience-Guided Reward Search Agent.

This package contains the incremental reward-editing framework built on top of
stable-eureka.  The first implementation keeps the original stable_eureka package
untouched and adds a separate runner, reward schema, diagnostics, memory, and
operator-constrained editing modules.
"""

__all__ = ["EGRSARunner"]

def __getattr__(name):
	if name == "EGRSARunner":
		from .runner import EGRSARunner
		return EGRSARunner
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
