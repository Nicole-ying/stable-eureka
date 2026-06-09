# Tool Registry Overview

v1 turns diagnostics into tools that the agent can request before editing rewards.

The LLM should not only output an edit_plan. It should decide whether it needs evidence first, choose a tool, read the returned evidence, then decide the next action.

Tool categories:

1. Read-only evidence tools: inspect trajectories, schemas, attribution, rollouts, memories.
2. Proposal tools: generate candidate edits or schema rewrites without applying them.
3. Safety tools: audit scale, detect reward dominance, check replayed bad-edit patterns.
4. Experiment tools: run short candidate sweeps or evaluate more seeds.

Common output fields:

- tool_name
- evidence_summary
- key_findings
- risk_flags
- recommended_next_actions
- machine_readable_payload
