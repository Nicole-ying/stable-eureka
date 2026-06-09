# EG-RSA v1 Release Design

## Positioning

`v1release` is the first formal design version of EG-RSA. Earlier code and experiments are treated as the v0 baseline. The v0 system already shows that an LLM can tune a human-designed reward schema, but it is not yet a full agentic reward-search system.

The v1 goal is to upgrade EG-RSA into a memory-driven, tool-augmented, self-evolving reward-search agent.

## Problems exposed by v0

1. The LLM mostly fills an edit-plan JSON.