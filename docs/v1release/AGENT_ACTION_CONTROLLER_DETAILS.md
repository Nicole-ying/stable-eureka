# Agent Action Controller Details

## Core Decision Object

Each iteration should produce an `AgentActionDecision` rather than only an `edit_plan`.

Required fields:

- `action`: one of `inspect`, `apply_local_edit`, `structural_search`, `free_rewrite`, `continue_training`, `evaluate_more`, `rollback_replan`, `early_stop`.
- `confidence`: value in `[0, 1]`.
- `reason`: concise evidence-based explanation.
- `tools_requested`: ordered tool calls requested before final execution.
- `edit_plan`: optional local operator edits.
- `schema_rewrite`: optional full schema rewrite proposal.
- `memory_writes`: episodic, lesson, or policy memories to write.
- `safety_checks`: required checks before execution.

## Action Meanings

`inspect` means the agent needs more evidence before editing. It may call trajectory, schema diff, attribution, or