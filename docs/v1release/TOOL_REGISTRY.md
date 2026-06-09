# v1 Tool Registry Design

## Purpose

v1 changes diagnostics from a fixed pipeline into explicit tools that an agent can request. The LLM should not only output an `edit_plan`; it should be able to inspect trajectories, compare schemas, audit component scales, retrieve memories, request extra evaluation, and then choose the next action.

## Common tool contract

Each tool should expose:

- `name`: stable tool name used by AgentActionController.
- `input`: JSON-compatible request object.
- `output`: JSON-compatible evidence object.
- `summary`: short text that can be inserted into the LLM context.
- `safety_level`: `read_only`, `proposal_only`, or `