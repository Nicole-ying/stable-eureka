# v1 Tool Registry Design

## Goal

The current EG-RSA pipeline calls diagnostics in a fixed order. v1 turns diagnostics into explicit tools that an AgentActionController can request before deciding the next reward-search action.

The key change is: the LLM no longer only emits an edit_plan. It may request tools, inspect evidence, and then decide whether to edit, continue training, rollback and replan, evaluate more seeds, or enter free rewrite.

## Tool Registry Interface

Each tool should implement a small common interface:

```python
class RewardSearchTool:
    name: str
    input_schema: dict
    output_schema: dict

    def run(self, request