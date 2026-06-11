# EG-RSA V1 Freeze: Human-Seeded Constrained Reward Self-Evolution

## Version Name

EG-RSA-V1: Human-Seeded Constrained Reward Self-Evolution

## Purpose

This document freezes the current V1 system and its experimental interpretation.

V1 should not be described as fully autonomous reward discovery from scratch. It should be described as a controlled mechanism-verification setting.

V1 uses:

- task-informed initial schema
- semantic role attribution
- outcome memory
- LLM edit planning
- operator-constrained schema edits
- behavior risk audit
- rollback / continuation decisions

## Main Experiment

Directory:

    experiments/eg_rsa_lunar_lander_v1_role_attrib_10x1m_audit_relaxed

Configuration:

    configs/eg_rsa_deepseek_v1_role_attrib_10x1m.yml

## What V1 Demonstrates

V1 demonstrates that a reward-search agent can operate over a task-informed reward schema and use diagnostic feedback to revise rewards across iterations.

Observed mechanism chain:

    iteration 0-1:
        reward hacking / shaping-goal mismatch
        dense guidance dominates
        no success

    iteration 1 -> 2:
        LLM disables continuous approach-region reward
        LLM increases terminal stable-landing bonus
        LLM adds one-time landing-region entry event
        success appears

    iteration 2-5:
        no-edit / continue-training decisions preserve a good reward
        success becomes stable

    iteration 6 -> 7:
        an overly aggressive edit reduces dense landing/stability guidance
        policy collapses

    iteration 7 -> 9:
        memory identifies the failed edit pattern
        later edit restores intermediate landing guidance
        success recovers

## What V1 Does Not Demonstrate

V1 does not demonstrate reward discovery from raw observations or from a completely unstructured task interface.

Reasons:

1. The initial reward schema is human-seeded and task-informed.
2. The schema already contains landing-related semantic concepts.
3. The LLM is constrained to edit schema components rather than freely synthesize reward semantics.
4. Official environment reward is not used for iterative feedback, but task-specific semantic metrics are still provided.

Therefore, V1 should be positioned as:

    controlled mechanism verification under a human-seeded semantic interface

not as:

    fully autonomous reward discovery from scratch

## Why V1 Is Still Valuable

V1 isolates the internal mechanisms needed by V2:

1. diagnosis-driven editing
2. semantic role attribution
3. memory over effective and regressive edits
4. no-edit decisions when reward is aligned
5. bad-edit detection and recovery
6. safety-exploration trade-off in reward self-evolution
