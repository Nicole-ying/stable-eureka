# Agent Action Controller

## Goal
Turn EG-RSA from a fixed reward-edit pipeline into an agentic search loop. The LLM should not only emit `edit_plan`; it should choose the next action, request tools, read evidence, and decide whether to edit, continue training, evaluate more, rollback, or stop.

## Current v0 Limitation
The v0 runner mostly supports a linear flow:

1. train current schema
2. diagnose trajectories
3. ask LLM for edit plan
4. apply edit or rollback

This means the LLM can explain that it wants to continue training, inspect a failure, or compare schema versions, but the runner cannot faithfully execute those intentions.

## v1 Responsibility
`AgentActionController` owns high-level decisions after each iteration.

Input:
- current schema and schema hash
- diagnostic report
- semantic outcome
- attribution report
- retrieved memories