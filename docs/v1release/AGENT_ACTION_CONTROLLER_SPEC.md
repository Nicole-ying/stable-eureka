# AgentActionController Spec

## Purpose

Convert EG-RSA from a fixed reward-edit loop into an agentic control loop. The LLM should choose the next system action, not only emit a reward edit.

## Inputs

- current reward schema
- latest diagnostic report
- semantic outcome
- attribution report
- retrieved episodic memories
- retrieved lesson memories
- retrieved policy memories
- previous action result

## Output schema

```json
{
  "action": "apply_local_edit | structural_search | free_rewrite | continue_training | evaluate_more | rollback_replan | call_tool | early_stop",
  "confidence": 0.0,
  "reason": "brief evidence-based explanation",
  "tool_requests": [],
  "edit_plan": [],
  "memory_write_requests