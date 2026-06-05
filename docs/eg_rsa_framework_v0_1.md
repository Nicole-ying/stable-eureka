# EG-RSA v0.1 Framework Definition

This document freezes the current research and engineering interpretation of EG-RSA v0.1. Its purpose is to prevent uncontrolled patch-style changes and to keep future experiments aligned with the core thesis: memory-driven reward-function self-evolution.

## 1. Research problem

EG-RSA studies whether an LLM-guided reward search system can improve reward design through iterative training feedback, diagnosis, memory, and structured reward edits.

The goal is not to hand-code a good reward for one environment. The goal is to build a general reward-search agent that can:

1. observe failures from rollouts;
2. diagnose reward hacking and task failures;
3. retrieve previous edit experience;
4. propose constrained reward edits;
5. evaluate whether candidate edits are learnable;
6. apply only legal and attributable edits;
7. write measured outcomes back into memory;
8. use the resulting lessons to guide later iterations.

## 2. Core thesis

The core thesis is:

> Reward search should not be treated as one-shot LLM reward generation. It should be treated as an experience-guided self-evolution process where failed and successful reward edits are stored, evaluated, and reused through memory.

EG-RSA v0.1 therefore emphasizes the following:

- structured reward schemas instead of free-form Python reward code;
- edit operators instead of unconstrained reward rewrites;
- trajectory-based diagnostics instead of direct official reward feedback;
- outcome-gated memory instead of unfiltered logs;
- candidate feasibility evaluation before training newly proposed structural rewards;
- edit gates to preserve causal attribution.

## 3. Non-goals for v0.1

EG-RSA v0.1 must not drift into the following:

1. Environment-specific reward engineering.
2. Hard-coding LunarLander-specific reward formulas.
3. Using official environment reward inside prompts, memory, or edit decisions.
4. Adding a new prompt rule for every failed experiment.
5. Treating a single short-run improvement as proof of method performance.
6. Rewriting the whole reward function freely through LLM code generation.
7. Treating stored memory as useful without outcome-based quality evaluation.

Official environment reward is allowed only for posthoc reporting and external evaluation.

## 4. Frozen module boundary

EG-RSA v0.1 consists of the following modules.

### 4.1 RewardSchema

Reward functions are represented as structured components and event rules. LLMs do not write executable reward code. They output JSON edit plans.

Formal role:

- defines editable reward components;
- defines event rules;
- gives each reward item a name, type, weight, parameters, clip range, and enabled flag;
- supports safe compilation into a runtime wrapper.

### 4.2 Diagnostic layer

The diagnostic layer computes:

- reward component attribution;
- component contribution ratios;
- trigger rates;
- repeated event patterns;
- task proxy metrics;
- failure modes such as reward hacking or shaping-goal mismatch.

It does not use official reward as a decision signal.

### 4.3 EditAgent

The main LLM agent performs high-level diagnosis and proposes local reward edits when evidence is sufficient.

Expected local edits include:

- increase_weight;
- decrease_weight;
- clip_component;
- disable_component;
- convert_to_one_time_event;
- add_duration_condition.

The EditAgent may identify the need for structural search, but newly added reward structures must be checked by validator, candidate evaluator, and edit gate before execution.

### 4.4 StructuralSearchAgent

The structural search agent proposes new reward structure when local edits are insufficient. It must operate only over configured events, metrics, and allowed operators.

Expected structural edits include:

- add_event_rule;
- add_component with metric_value;
- add_component with metric_delta;
- add_component with metric_threshold_bonus;
- add_component with metric_stagnation_penalty.

Structural search must prefer process signals when terminal events are too sparse.

### 4.5 EditPlanValidator

The validator checks and normalizes JSON edit plans before execution.

Responsibilities:

- verify supported operators;
- verify target existence;
- verify event conditions only reference configured events;
- verify metric components only reference configured task metrics;
- normalize compact event-rule or metric-component forms;
- move duration_steps into event_rule.condition.duration_steps when needed.

Validator checks legality. It does not decide whether an edit is strategically good.

### 4.6 RewardCandidateEvaluator

The candidate evaluator estimates whether a newly proposed structural reward has learnable signal under the current policy distribution.

Responsibilities:

- estimate event-rule trigger rate from current rollouts;
- estimate metric variation and active rate for metric components;
- classify signal density as zero, sparse, medium, or dense;
- reject or warn against zero-signal structural rewards;
- push the search toward process signals when terminal event rewards are unreachable.

This is a framework-level module, not an environment-specific patch.

### 4.7 EditDecisionGate

The edit gate preserves attribution and prevents uncontrolled edits.

Responsibilities:

- limit the number of edits per iteration;
- reject low-evidence edits on existing targets;
- keep the highest-evidence edit when several edits are proposed;
- record rejected edits and warnings.

### 4.8 MemoryStore

Raw memory stores factual trial records:

- failure modes;
- attribution;
- edit plan;
- candidate evaluation;
- validation and gate results;
- before/after task proxy and hack metrics.

Raw memory is evidence, not recommendation.

### 4.9 LessonStore

Lesson memory distills raw memory into reusable or cautionary lessons.

Lesson quality must be outcome-gated:

- strong_positive;
- moderate_positive;
- weak_positive;
- failed;
- harmful;
- uncertain;
- decision_record.

A lesson's reuse confidence must come from measured outcomes, not from LLM confidence alone.

## 5. Current decision flow

The v0.1 execution flow is:

```text
Train policy with current RewardSchema
    ↓
Record trajectories, events, metrics, reward components
    ↓
Run attribution and failure diagnostics
    ↓
Retrieve raw memory and lesson memory
    ↓
EditAgent proposes edit_plan or structural_search
    ↓
EditPlanValidator normalizes and validates edit_plan
    ↓
RewardCandidateEvaluator evaluates new structural rewards
    ↓
EditDecisionGate controls edit scale and attribution
    ↓
Apply accepted edit_plan to RewardSchema
    ↓
Train next iteration
    ↓
Measure outcome and update MemoryStore / LessonStore
```

## 6. How to interpret current experiments

Current 100K-per-iteration experiments are mechanism-performance transition tests. They are not final long-training performance claims.

Acceptable conclusions:

- whether the pipeline runs end-to-end;
- whether memory is written and retrieved;
- whether reward-hack fixes are found;
- whether candidate rewards trigger under current rollouts;
- whether lessons are correctly classified as strong, weak, failed, or uncertain;
- whether internal proxy metrics show stable direction.

Unacceptable conclusions from a single 3-iteration run:

- final method superiority;
- solved task performance;
- generalization across environments;
- proof that a single edit is universally effective.

## 7. Current known status

Stable observations so far:

1. Converting repeatable contact reward to one-time contact reward often improves internal task proxy.
2. This edit may not always reduce hack_score immediately.
3. Terminal landing rewards are often semantically correct but can be too sparse.
4. Sustained-contact rewards can reduce hack risk but may harm task progress if not paired with process guidance.
5. LessonStore now avoids treating weak or failed structural edits as strong reusable lessons.
6. CandidateEvaluator can distinguish zero-trigger candidates from nonzero sparse candidates, but sparse-candidate policy still needs validation.

## 8. Framework risks

The major risks are:

1. Proxy mismatch: internal task_score may not correlate with official posthoc reward.
2. Sparse reward bias: LLMs tend to propose semantically correct but low-trigger event rewards.
3. Memory overconfidence: strong lessons may be too optimistic if task_score improves but hack_score does not.
4. Search conservatism: edit gates may make exploration too slow.
5. Evaluation noise: 100K runs can show trends but are not final evidence.

These risks should be addressed by experiment protocol and ablation, not ad-hoc environment-specific patches.

## 9. Code-change policy after v0.1 freeze

After this freeze, code changes should be accepted only if they fit one of these categories:

1. Correctness fix: fixes a clear mismatch between intended schema and runtime behavior.
2. Instrumentation: improves logging, summaries, or reproducibility without changing algorithm decisions.
3. Framework module refinement: improves an already defined module such as CandidateEvaluator or LessonStore using general principles.
4. Config-only experiment setup: changes budgets, seeds, output paths, or ablation toggles.
5. Documentation: clarifies assumptions, risks, and experimental interpretation.

Avoid changes that merely respond to a single bad LunarLander result unless the change is first abstracted into a general mechanism.

## 10. Experiment progression policy

Default mechanism-performance transition budget:

```text
3 iterations × 100K timesteps
```

Move to 5 × 100K only when:

- at least one structural or metric-based edit is accepted;
- candidate evaluation reports nonzero signal;
- lesson memory records the outcome correctly;
- posthoc reward does not catastrophically degrade;
- task_score or hack_score shows a plausible direction.

Move to 10 × 300K only when:

- 5 × 100K shows a consistent trend;
- full EG-RSA is not clearly worse than simple baselines;
- at least one ablation comparison is available.

Move to 10 × 500K only after 300K runs justify the compute.

## 11. Required ablations

The minimum ablation plan is:

1. Full EG-RSA.
2. No memory.
3. No candidate evaluator.
4. No edit gate.
5. No structural search.
6. Fallback / non-LLM edit baseline.

These ablations are necessary to show that memory and candidate evaluation contribute beyond prompt-only reward editing.

## 12. Current v0.1 framing sentence

EG-RSA v0.1 should be described as:

> An experience-guided reward search agent that evolves structured reward functions through rollout diagnostics, outcome-gated memory, candidate feasibility evaluation, and constrained edit execution.

This is the stable framing for the next stage of experiments.
