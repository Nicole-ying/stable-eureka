# EG-RSA V1 to V2 Plan

## Core Motivation

V1 validates the reward self-evolution loop, but it starts from a human-written initial schema and diagnostic interface. This creates two limitations:

1. Initial schema prior is too strong.
2. Search space is constrained by manually defined semantic predicates.

V2 moves from:

    human-seeded constrained schema editing

to:

    LLM-bootstrapped reward self-evolution

## V2 Version Name

EG-RSA-V2: LLM-Bootstrapped Reward Self-Evolution

## V2 Two-Stage Framework

Stage 1: LLM Bootstrap

Input:

    task description
    primitive observation variables
    primitive action variables
    allowed formula variables
    allowed formula functions
    safety constraints
    forbidden signals

Output:

    generated_initial_schema.json
    generated_diagnostics.yml
    bootstrap_report.json

Stage 2: Reward Self-Evolution

Loop:

    generated schema
    -> compile reward
    -> RL training
    -> trajectory feedback
    -> semantic outcome
    -> role attribution
    -> memory retrieval
    -> LLM schema evolution
    -> formula/event validation
    -> risk audit
    -> accept / repair / rollback
    -> next schema

## What V2 Must Avoid

V2 should not start with manually defined high-level predicates such as:

    landing_region
    safe_contact
    stable_landing_condition
    approach_region_score
    landing_quality
    stability

These are allowed only if the LLM itself proposes them from primitive variables and task description.

## V2 Expanded Edit Freedom

Level 1: existing safe edits

    increase_weight
    decrease_weight
    disable_component

Level 2: schema synthesis

    add_formula_component
    add_conditional_formula_component
    add_event_predicate
    modify_component_formula
    modify_event_condition

Level 3: sandboxed proposal

    LLM proposes reward idea
    -> convert to schema expression
    -> static validation
    -> signal variation test
    -> short-run sandbox
    -> full iteration

## V2 Validation Requirements

1. AST whitelist check
2. variable whitelist check
3. function whitelist check
4. finite numeric output check
5. non-constant signal check
6. active-rate check
7. range / scale check
8. reward hacking risk check
9. short-run sandbox before long training

## V2 Experimental Plan

Smoke test:

    EG-RSA-V2 bootstrap smoke, 3 iterations x 100k steps

Main experiment:

    EG-RSA-V2 full, 10 iterations x 1M steps

Required baselines:

    Original PPO, 10M
    V1 initial schema frozen, 10M
    V1 full, 10 x 1M
    V2 bootstrap-only frozen, 10M
    V2 full, 10 x 1M

Priority ablations:

    V2 w/o memory
    V2 w/o attribution
    V2 w/o risk audit
