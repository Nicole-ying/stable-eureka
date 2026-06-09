# Phased Reward Search Spec

## Purpose

Reward search should not use the same edit freedom at every stage. Early search needs safety. Later refinement needs more freedom and better tools.

## Phase 1: Constrained Search

### Condition
- success rate is low
- terminal reward is rarely paid
- hack score is high or shaping-goal mismatch exists

### Allowed actions
- apply_local_edit
- coupled_rebalancing
- add_duration_condition
- moderate weight changes
- continue_training if