# Trajectory and Schema Tool Spec

## Purpose

Give the LLM evidence tools before it edits. The agent should inspect trajectories and schema diffs instead of guessing from summary metrics only.

## TrajectoryInspector

### Inputs
- experiment directory
- iteration id
- episode id or top-k failed episodes

### Outputs
- termination reason
- final position and velocity summary
- whether contact, safe_contact