# Section Blueprints

## EG-RSA: Experience-Guided Reward Schema Search

### Overall Architecture

The paper follows a **mechanism-verification arc** (not a benchmark-domination arc). Every section answers one part of the same question: "What does it mean to search (rather than generate) reward functions, and what mechanisms make that search history-aware and auditable?"

---

## Section 1 — Introduction

**Paragraph count:** 6
**Total words target:** ~800

| Para | Move | Evidence/Citation Anchor | Exemplar Pattern |
|---|---|---|---|
| 1 | **Problem opening:** Reward design is a central bottleneck in RL. The reward function defines the optimization objective; poor rewards cause hacking or misaligned behavior. | C001 (Sutton & Barto), C013 (Amodei et al.) | EUREKA: opens with "designing reward functions is labor-intensive trial-and-error" |
| 2 | **LLM opportunity:** LLMs can generate reward code (EUREKA, Text2Reward, CARD). This establishes a new paradigm but has structural limits. | C006 (EUREKA), C009 (CARD), C015 (survey) | EUREKA: quickly surveys what LLMs can do for reward design |
| 3 | **Paradigm limitation:** Existing methods treat reward design as code generation + iterative refinement. Missing: structured representation, cross-iteration memory, diagnosis-driven editing, risk audit. | C006-C009 (all LLM reward work), C010-C012 (memory agents) | Reflexion/Voyager: identifies what the current paradigm cannot do |
| 4 | **Specific gap:** Reward search requires understanding *why* a reward failed, not just generating a new one. It requires storing past outcomes, constraining edits, and auditing risk. | SOTA gap map Gap 1 | EUREKA: gap statement before contribution |
| 5 | **EG-RSA proposal:** EG-RSA reformulates reward design as experience-guided reward schema search. It represents rewards as schemas, attributes policy behavior to semantic roles, stores outcome lessons, constrains LLM output to auditable operators, and integrates risk audit with rollback. | Design doc S1-2 | Voyager: system overview in one paragraph |
| 6 | **Contributions + paper structure:** (1) reward search framing with structured schema + memory, (2) semantic role attribution for diagnosis-driven editing, (3) operator-constrained editing with risk audit and rollback. | Confirmed motivation | All exemplars: 3-4 contributions + "the remainder of this paper is organized as follows" |

---

## Section 2 — Related Work

**Subsections:** 5
**Total words target:** ~1200

| Subsection | Para Count | Core Claim | Citations | Contrast With EG-RSA |
|---|---|---|---|---|
| 2.1 Reward Design in RL | 2 | Reward design is a recognized bottleneck; shaping and IRL are classical approaches but require human knowledge or demonstrations. | C001, C003, C004, C005, C039 | EG-RSA does not assume demonstrations or hand-designed potentials; it learns from training feedback. |
| 2.2 LLM-Assisted Reward Generation | 3 | LLMs can generate rewards (EUREKA, Text2Reward, CARD, Auto MC-Reward). This is effective but follows a code-generation paradigm. | C006, C007, C008, C009, C015 | EG-RSA constrains LLM output to auditable operators over a structured schema, and adds memory + attribution. |
| 2.3 Memory and Reflection in LLM Agents | 2 | Memory-based agents (Reflexion, Voyager, Generative Agents) improve through stored experience. | C010, C011, C012, C028 | EG-RSA stores structured reward-edit outcome lessons (schemas, diffs, deltas), not verbal reflections or skill code. |
| 2.4 Reward Hacking and Risk-Aware Design | 3 | Reward hacking is a major concern (Amodei, Pan); recent work on detection and mitigation (InfoRM, ensembles, tampering studies) shows the problem is unsolved. | C013, C014, C019, C020, C021 | EG-RSA integrates audit directly into the search loop, with explicit risk triage, repair, and rollback. |
| 2.5 Positioning | 1 | Summary of what EG-RSA combines and how it differs from each line of work. | All above | EG-RSA is a memory-augmented, audit-integrated reward schema search framework — not generation, not reflection, not post-hoc safety. |

---

## Section 3 — Method

**Subsections:** 6
**Total words target:** ~2000

| Subsection | Para Count | Core Content | Evidence Anchor | Key Constraint |
|---|---|---|---|---|
| 3.1 Overview | 3 | The EG-RSA search loop: schema→compile→train→diagnose→attribute→retrieve→edit→audit→lesson. Diagram reference. Contrast with Stable-Eureka loop. | Design doc S1-2; FIG-01 (workflow diagram) | Do not claim EG-RSA "solves" reward design. Frame as "reformulation." |
| 3.2 Reward Schema Representation | 2 | Componentized schema with JSON example. Semantic roles (dense_guidance, stability_quality, terminal_success, safety_constraint, control_cost). Schema vs. free code. | Design doc S3.2; schema.py | Do not claim roles are learned. They are template-instantiated. |
| 3.3 Semantic Role Attribution | 3 | How per-component reward is logged, dominant-role ratio computed, how attribution connects to diagnosis. | Design doc S3.1, S3.5; attribution.py; trajectory_recorder.py; E3 evidence | Attribution identifies which role dominates — it does not automatically prescribe edits. |
| 3.4 Experience Memory and Outcome Lessons | 2 | Memory card structure, effective_edit_lesson vs. regression_lesson, retrieval mechanism, schema_diff and metric_delta storage. | Design doc S3.5-S3.6; memory_card.py; memory_store.py; lesson_store.py; E4 evidence | Memory stores structured cards, not text. This is a design choice, not a claim of superiority. |
| 3.5 Operator-Constrained Reward Editing | 2 | Five edit operators, edit_plan JSON format, safe compiler, edit validation, decision gate. LLM is constrained; trusted logic executes. | Design doc S3.3, S3.7; operators.py; safe_compiler.py; edit_plan_validator.py; edit_decision_gate.py | The constraint is the contribution, not the specific operators. |
| 3.6 Risk-Aware Audit and Rollback | 3 | Behavior risk audit (high/medium/low), scale audit, hack detectors, repair loop, rollback decision, outcome acceptor. | Design doc S3.8; hack_detectors.py; behavior_risk_audit.py; E5 evidence | Audit is rule-based, not learned. Deadlock is an expected design tension. |
| Integration | 1 | How all components interact in one iteration. "The LLM does not generate code; it proposes structured edits over a versioned, attributed, audited schema." | Full loop trace from EXP-01 Iter 0→1 | Integration paragraph is mandatory per exemplar pattern. |

---

## Section 4 — Experiments

**Subsections:** 5
**Total words target:** ~1500

| Subsection | Para Count | Core Content | Evidence Anchor | Framing Constraint |
|---|---|---|---|---|
| 4.1 Environment and Setup | 2 | LunarLander-v3, PPO, 1M steps/iteration, 10 iterations, experiment_mode.json settings. | E1; C002 (PPO); C051 (Gymnasium) | "Fixed sufficient training budget" — not "complete training." |
| 4.2 Main EG-RSA Experiment | 3 | Run history table (E1), overall trajectory, key observation: scores improve then regress then plateau. | E1 data | This is the overview — do not present it as a benchmark result. |
| 4.3 Effective Edit Case Study (Iter 0→1) | 3 | Detailed trace: attribution before edit (r_approach_region dominant, failure modes present), edit plan, attribution after edit (r_landing_quality dominant, failures eliminated). | E1 Iter 0-1 data; E3 attribution evidence | "Case study" framing. Show the mechanism working. |
| 4.4 Audit-Induced Deadlock Case Study (Iter 2→8) | 3 | Regression (1→2), then 6 iterations of low scores. Audit reports showing medium-risk blocks. The deadlock mechanism: strict audit blocks edits needed to escape low-success regime. | E1 Iter 1-8 data; E5 audit evidence | "This is not a system failure — it reveals a genuine design tension." |
| 4.5 Relaxed Audit Policy | 2 | EXP-02 results: relaxing medium-risk blocking enables breakthrough (0.511→2.490→2.953). Comparison table: strict vs. relaxed audit outcomes. | E2 data | "Motivates risk-budget mechanism" — future work, not solved. |

---

## Section 5 — Discussion

**Paragraph count:** 4
**Total words target:** ~500

| Para | Move | Anchor |
|---|---|---|
| 1 | **What we learned:** The three case studies together show that (a) structured search with memory can produce effective edits, (b) regression happens and rollback matters, (c) audit creates a safety-exploration tradeoff. | E1, E2 |
| 2 | **Limitations:** (1) Single-environment verification, (2) template-instantiated task metrics (not auto-discovered), (3) hand-designed audit rules, (4) 10-iteration scale, (5) no comparison with EUREKA on benchmark. | Confirmed motivation scope limits |
| 3 | **Broader implications:** The audit deadlock finding suggests that any LLM reward design system with safety constraints will face this tension. Risk budgets, graduated audit, or learned audit may help. | E2; C019, C020, C021 |
| 4 | **Future work:** Multi-environment evaluation, learned audit rules, risk-budget mechanism, integration with process reward models, scaling to more iterations. | C029 (PRM survey) |

---

## Section 6 — Conclusion

**Paragraph count:** 1
**Total words target:** ~150

Summarize: EG-RSA reformulates LLM reward design as experience-guided schema search. Three mechanisms (attribution, memory, audit) verified through case studies. Audit deadlock reveals safety-exploration tension. Reward search — not generation — is the right framing for LLM-assisted reward design.
