# ScaleAuditTool Spec

## Purpose
Prevent a new reward component from dominating the objective by accident.

## When to call
- before add_component
- before free_rewrite_schema
- after any structural_search proposal
- after any rollback caused by a new dominant component

## Inputs
- current schema
- proposed edit or proposed full schema
- component attribution from recent rollout
- semantic outcome

## Checks
- estimated per-episode magnitude of new dense reward
- ratio between dense penalty and terminal bonus
- sign consistency: penalty should not erase all progress signal
- trigger frequency risk
- whether one component can exceed 50 percent of reward mass

## Output