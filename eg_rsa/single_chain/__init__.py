"""Single-chain EG-RSA bootstrap pipeline.

This subpackage implements the first half of EG-RSA:
1. LLM-generated environment understanding.
2. LLM-generated target-alignment contract.
3. LLM-generated initial reward schema and reward code.
4. Single reward training and structured trace recording.
"""

from .agents import JsonAgent
from .trainer import SingleChainPPOTrainer

__all__ = ["JsonAgent", "SingleChainPPOTrainer"]
