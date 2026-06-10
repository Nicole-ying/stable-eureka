# From Reward Generation to Reward Search: Experience-Guided Schema Adaptation with Semantic Attribution and Risk-Aware Editing

## Abstract

Reinforcement learning (RL) depends on reward functions that define optimization objectives, yet designing effective rewards remains a central bottleneck: rewards must provide informative learning gradients while avoiding unintended shortcuts and remaining aligned with the true task. Recent work has shown that large language models (LLMs) can generate reward code and improve it through iterative refinement. These methods, however, treat reward design primarily as a code generation problem — the LLM emits free-form reward code, the code is evaluated, and the next iteration receives scalar feedback. What is missing is a framework that treats reward design as a search process over structured, attributed, auditable reward schemas, informed by past editing outcomes. We present EG-RSA, which reformulates LLM-assisted reward design as experience-guided reward schema search. EG-RSA represents rewards as componentized schemas with semantic role labels, attributes policy behavior to specific reward components, stores both effective and regressive edits as structured outcome lessons, constrains LLM output to auditable edit operators, and integrates behavior risk audit with automatic rollback. We evaluate EG-RSA on LunarLander-v3 through mechanism-verification case studies. An effective edit (iteration 0 to 1) demonstrates that attribution-guided memory-augmented editing can shift the dominant reward role from dense guidance to terminal success, eliminating detected failure modes. A subsequent regression and six-iteration low-success plateau under strict audit reveal a fundamental design tension: the same audit mechanism that prevents harmful edits can also block the exploration needed to escape a low-success regime. Relaxing the audit policy enables a breakthrough (task score 0.511 to 2.953), confirming that the deadlock was audit-induced rather than search-fundamental. These findings suggest that risk-aware reward search requires deliberate audit policy design, not simply strict blocking.

## 1. Introduction

In reinforcement learning, the reward function defines the optimization objective and directly shapes learned policy behavior [@sutton2018reinforcement]. Designing effective reward functions, however, remains a central bottleneck: rewards must provide informative learning gradients, avoid creating unintended shortcuts or local optima, and remain aligned with the true task objective rather than merely correlated with it [@amodei2016concrete]. Poorly designed rewards can produce agents that optimize the proxy signal while failing the intended task — a failure mode known as reward hacking or reward misspecification [@pan2022effects].

Recent work has demonstrated that large language models (LLMs) can serve as effective reward designers. EUREKA shows that LLMs can generate executable reward code and improve it through iterative in-context optimization, achieving human-level and super-human reward design across a range of RL environments [@ma2023eureka]. Text2Reward generates dense shaped reward functions from natural language task descriptions [@xie2023text2reward]. Auto MC-Reward combines a Reward Designer, Reward Critic, and Trajectory Analyzer to refine dense reward functions for sparse-reward tasks [@li2023automcreward]. CARD proposes a dynamic-feedback framework using a Coder-Evaluator with Trajectory Preference Evaluation to reduce unnecessary RL training during reward refinement [@sun2024card]. These methods establish that LLMs can be powerful tools for reward generation and iterative reward improvement [@cao2024survey].

These methods, however, primarily treat reward design as a code generation and iterative refinement problem: the LLM emits free-form reward code, the code is trained and evaluated, and the next iteration receives scalar or text feedback. While effective, this paradigm has structural limitations. First, there is no persistent structured representation of the reward function across iterations — each iteration generates new code from scratch. Second, there is no systematic diagnosis of *why* a reward failed: scalar feedback indicates that performance changed, but not which component of the reward drove the change or which failure mode emerged. Third, there is no mechanism for storing and reusing past editing outcomes — each iteration starts fresh, without access to the system's own history of effective and regressive edits. Fourth, there is no integrated safety audit: the LLM's generated code is executed without systematic pre-execution risk assessment or rollback.

What is missing is a framework that treats reward design as an experience-guided search process over structured reward schemas. Rather than generating complete reward functions from scratch, such a framework would maintain a versioned, componentized schema; diagnose which reward components drive policy behavior; retrieve relevant past editing outcomes; constrain the LLM to auditable edit operators; and gate every edit through risk audit with automatic rollback on failure.

We present EG-RSA (Experience-Guided Reward Search Agent), which reformulates LLM-assisted reward design along these lines. Starting from an initial reward schema, EG-RSA iteratively: compiles the schema into an executable reward function; trains a policy for a fixed sufficient budget; collects step-level trajectory data with per-component reward logging; performs semantic role attribution to identify which reward roles dominate policy behavior; retrieves relevant prior outcome lessons from structured memory; prompts the LLM to propose a constrained edit plan over the schema; validates, audits, and either accepts or rolls back the edit; and stores the outcome as a new structured lesson for future retrieval.

In summary, this paper makes the following contributions:

1. **Experience-guided reward schema search.** We reformulate LLM-assisted reward design as a search process over structured, versioned reward schemas, supported by cross-iteration outcome memory that records both effective and regressive edits.
2. **Semantic role attribution for diagnosis-driven editing.** We introduce per-component reward attribution with semantic role labels, enabling the system to identify which reward roles dominate policy behavior and to ground edit decisions in diagnostic evidence rather than scalar feedback alone.
3. **Operator-constrained editing with integrated risk audit and rollback.** We constrain LLM modifications to auditable edit operators over the schema, gate every proposed edit through behavior risk audit with explicit risk triage, and automatically roll back edits that produce regressive outcomes.

The remainder of this paper is organized as follows. Section 2 reviews related work in reward design, LLM-assisted reward generation, memory-based language agents, and reward hacking. Section 3 describes the EG-RSA method, including the search loop, schema representation, attribution, memory, constrained editing, and risk audit. Section 4 presents mechanism-verification experiments on LunarLander-v3, structured as three case studies. Section 5 discusses findings, limitations, and broader implications. Section 6 concludes.

## 2. Related Work

### 2.1 Reward Design and Reward Shaping in Reinforcement Learning

The reward function is the central interface between task intent and policy learning in reinforcement learning [@sutton2018reinforcement]. In practice, designing an effective reward function is often difficult: a reward signal must not only encourage task completion, but also provide useful learning gradients, avoid misleading shortcuts, and remain aligned with the true task objective.

Classical reward shaping studies how additional reward signals can accelerate learning. Potential-based reward shaping provides theoretical conditions under which transformed rewards preserve the optimal policy [@ng1999policy]. This line of work shows that shaping rewards can be useful, but also highlights a key challenge: poorly designed shaping signals may change what the agent learns or create unintended local optima. Inverse reinforcement learning (IRL) aims to infer reward functions from expert demonstrations, reducing the need for hand-written rewards [@ng2000irl; @abbeel2004apprenticeship]. These methods, however, assume access to demonstrations or expert behavior.

EG-RSA differs from these traditional approaches in both mechanism and assumptions. Unlike IRL, EG-RSA does not assume expert demonstrations. Unlike potential-based shaping, EG-RSA does not rely on hand-designed shaping potentials. Instead, EG-RSA starts from an initial reward schema and iteratively adapts it using policy training feedback, semantic failure diagnosis, reward role attribution, and accumulated editing experience.

### 2.2 LLM-Assisted Reward Generation

Recent work has established LLMs as effective reward designers. EUREKA demonstrates that LLMs can generate executable reward code and improve it through iterative in-context optimization over reward programs, outperforming human-engineered rewards in a range of environments [@ma2023eureka]. Text2Reward generates dense reward functions from natural language task descriptions and supports refinement with human feedback [@xie2023text2reward]. Auto MC-Reward further applies LLM-based dense reward design to sparse-reward tasks, combining a Reward Designer, Reward Critic, and Trajectory Analyzer to refine reward functions from collected trajectories [@li2023automcreward]. CARD proposes a dynamic-feedback framework for LLM-driven reward design, using a Coder and Evaluator together with Trajectory Preference Evaluation to reduce unnecessary RL training during reward refinement [@sun2024card]. A recent survey comprehensively organizes the LLM-enhanced RL landscape, confirming that LLM-as-reward-designer is an active and growing research direction [@cao2024survey].

These methods share a common paradigm: they treat reward design as code generation with iterative refinement. The LLM emits free-form reward code, the code is evaluated through training, and the next iteration receives feedback. EG-RSA departs from this paradigm in four ways. First, EG-RSA maintains a persistent, versioned reward schema rather than generating new code each iteration. Second, EG-RSA attributes policy behavior to specific reward components with semantic role labels, enabling diagnosis-driven rather than feedback-driven editing. Third, EG-RSA stores both effective and regressive edits as structured outcome lessons, allowing future iterations to retrieve and reuse search experience. Fourth, EG-RSA constrains the LLM to auditable edit operators and gates every edit through behavior risk audit — replacing unconstrained code generation with auditable schema manipulation.

### 2.3 Memory and Reflection in Language Agents

EG-RSA is also related to memory-based and reflection-based language agents. Reflexion proposes that language agents can improve through verbal feedback and episodic memory without updating model parameters [@shinn2023reflexion]. Voyager introduces an embodied lifelong learning agent that stores executable skills in a growing skill library and reuses them to solve new tasks [@wang2023voyager]. Generative Agents use memory streams, reflection, and planning to produce coherent long-horizon agent behavior [@park2023generative]. Surveys of LLM-based agents document the growing role of memory as a core agent capability [@zhang2024memorysurvey].

These works establish that stored experience can improve the behavior of LLM-based agents across trials. EG-RSA adopts this high-level intuition but applies it to a different object: rather than storing verbal reflections (Reflexion), skill code (Voyager), or memory streams (Generative Agents), EG-RSA stores structured reward-edit outcome lessons. Each lesson contains the previous and edited schema, the edit plan, metric deltas, detected failure modes, hack risk changes, and rollback decisions. This makes the memory directly actionable for future reward editing: when the system encounters a similar failure mode, it retrieves lessons that record how prior edits affected that mode.

### 2.4 Reward Hacking and Risk-Aware Reward Optimization

Reward hacking and reward misspecification are major concerns in reinforcement learning. When the proxy reward differs from the intended objective, an agent may exploit the reward function while failing the true task [@amodei2016concrete]. Prior work on reward misspecification shows that more capable agents can sometimes obtain higher proxy reward while achieving lower true reward, exposing a gap between optimized and intended behavior [@pan2022effects]. Recent formal work defines and taxonomizes reward hacking behaviors, providing a vocabulary for the failure modes that LLM-generated rewards may introduce [@skalse2022reward]. Theoretical analysis further shows that standard KL regularization fails under heavy-tailed reward error, motivating alternative approaches to reward safety [@langosco2024goodhart].

Several recent methods attempt to mitigate reward hacking. Reward model ensembles reduce overoptimization by aggregating diverse reward models, but share similar error patterns and do not eliminate hacking [@eisenstein2023ensembles]. Information-theoretic reward modeling uses a variational information bottleneck to filter irrelevant features and provides an online detection metric for overoptimization [@lai2024infom]. A landmark study demonstrates that LLMs can generalize from simple sycophancy to directly rewriting their own reward function, underscoring the need for non-LLM safety mechanisms [@dennison2024sycophancy].

EG-RSA addresses reward hacking risk through integrated audit rather than post-hoc mitigation. The behavior risk audit evaluates each proposed edit before it is applied, classifying edits as high, medium, or low risk based on the detected failure modes, the reward components being modified, and the current success evidence. High-risk edits are blocked; medium-risk edits are evaluated against a configurable audit policy; low-risk edits proceed. When an edit produces a regressive outcome, the system automatically rolls back to the previous schema. As our experiments show, however, this integration is not straightforward: a strict audit policy can create a deadlock by blocking the medium-risk edits needed to establish success evidence in the first place.

### 2.5 Positioning of EG-RSA

In summary, prior work has made significant progress in reward shaping, inverse reinforcement learning, LLM-based reward generation, memory-based language agents, and reward hacking mitigation. EG-RSA combines these ideas but targets a distinct problem: how to make LLM-assisted reward design history-aware, interpretable, and auditable by structuring it as a search process over reward schemas rather than a code generation process. The key difference is that EG-RSA maintains a versioned, attributed reward schema; stores structured outcome lessons from both successes and failures; constrains the LLM to auditable operators; and integrates risk audit directly into the search loop with explicit triage and rollback. This shifts the role of the LLM from a one-shot reward code generator to an experience-guided reward search operator.

## 3. Method

### 3.1 Overview

Figure 1 illustrates one full iteration of the EG-RSA search loop. Starting from a versioned reward schema, the system: (1) compiles the schema into an executable reward function via a trusted compiler; (2) trains a policy using the generated reward for a fixed sufficient budget; (3) collects step-level trajectory data with per-component reward logging; (4) performs semantic role attribution to identify which reward roles dominate policy behavior and which failure modes are present; (5) retrieves relevant prior outcome lessons from structured memory; (6) prompts the LLM to propose a constrained edit plan in JSON format; (7) validates the edit plan for schema consistency and operator legality, runs behavior risk audit and scale audit, and either accepts, repairs, or blocks the edit; and (8) stores the outcome as a new structured lesson — recording whether the edit was effective or regressive, the schema diff, metric deltas, and failure mode changes.

The policy is trained exclusively with the generated reward function. The environment's oracle reward is never used for reward selection, editing, or acceptance decisions. We report the oracle reward only as a post-hoc evaluation metric. This separation ensures that the search process does not leak ground-truth information into edit decisions.

EG-RSA's loop differs structurally from the population-based reward generation loop used in Stable-Eureka and related methods. In the population approach, the LLM generates multiple complete reward-code samples; each is trained in parallel; an oracle fitness score selects the best; and the best reward is reflected upon for the next iteration. EG-RSA replaces this with a sequential single-trajectory loop: one schema is maintained and edited across iterations, the LLM proposes structured edits rather than complete rewrites, training feedback is diagnosed through attribution rather than reduced to a scalar score, and audit gates every edit before it reaches training.

### 3.2 Reward Schema Representation

EG-RSA represents reward functions as componentized schemas rather than unconstrained Python code. A schema consists of named components, each with a type, weight, input variables, parameters, and an enabled flag. Each component is assigned a semantic role from the following taxonomy:

- **dense_guidance**: Progress and approach rewards that provide learning gradients toward the goal.
- **stability_quality**: Rewards for smooth, stable behavior (velocity limits, angle constraints, contact quality).
- **terminal_success**: Rewards paid upon achieving the terminal task objective.
- **safety_constraint**: Penalties for unsafe states or constraint violations.
- **control_cost**: Penalties on action magnitude to encourage efficient control.

A representative LunarLander schema includes components such as `r_approach_region` (dense_guidance), `r_landing_quality` (terminal_success), `r_stable_contact` (stability_quality), and `r_action_smoothness` (control_cost). Event rules — one-time rewards triggered by specific conditions — are also part of the schema.

This representation serves three purposes. First, it enables versioning: each schema is a structured artifact that can be diffed, stored, and rolled back. Second, it enables per-component attribution: because each component is separately logged, the system can measure which role dominates policy behavior. Third, it enables operator-constrained editing: the LLM proposes edits to named components with typed operators, rather than generating free-form code that is difficult to validate.

### 3.3 Semantic Role Attribution

Scalar task scores alone cannot explain why a reward failed. A policy may achieve high scores on dense progress rewards while never reaching terminal success — the shaping-goal mismatch failure mode. It may exploit a contact-based reward component through rapid toggling rather than stable landing — the repeated event exploitation failure mode. To diagnose such failures, EG-RSA computes per-component reward attribution from step-level trajectory data.

During training, the reward contributed by each component is logged at every step. After training, these per-step component rewards are aggregated per episode. The dominant component is identified as the one with the highest fraction of total episode reward, yielding a dominant-component ratio. Additional diagnostic signals are computed from the trajectory: success episode rate (fraction of episodes reaching terminal success), terminal reward payment patterns (one-time vs. repeated), contact toggle frequency, progress score trajectory, and reward repetition risk. These signals are combined into a diagnostic report that identifies active failure modes (e.g., shaping_goal_mismatch, repeated_event_exploitation) and flags risk conditions.

The attribution report — dominant role, failure modes, risk flags — is provided to the LLM edit agent alongside the current schema and retrieved outcome lessons. The LLM uses this diagnosis as evidence when proposing edits, but attribution does not mechanically determine the edit: the LLM weighs diagnostic evidence against retrieved lessons and proposes a structured edit plan. This preserves the LLM's reasoning capability while grounding it in diagnostic data.

### 3.4 Experience Memory and Outcome Lessons

EG-RSA maintains a structured memory of past reward-edit outcomes. Unlike verbal reflection [@shinn2023reflexion] or skill libraries [@wang2023voyager], EG-RSA stores outcome lessons — structured records containing the previous and edited schema (schema_diff), the metric deltas (changes in task score, semantic score, hack score), the detected failure modes before and after the edit, the hack risk change, and the rollback decision.

Lessons are classified as effective_edit_lesson (the task score improved) or regression_lesson (the task score degraded). Both types are stored and retrieved. When the system prepares to edit a reward, it queries memory for lessons with similar failure modes or schema structures. Effective lessons provide evidence for which edit strategies have worked in similar contexts; regression lessons warn against edits that previously caused degradation.

This memory mechanism is what makes the search "experience-guided." Without memory, each iteration would start fresh, with no access to the system's own history. With memory, the LLM receives not only the current diagnostic picture but also a set of relevant prior outcomes — turning feed-forward refinement into history-aware search.

### 3.5 Operator-Constrained Reward Editing

EG-RSA constrains the LLM to propose edits through a fixed set of operators rather than emitting free-form reward code. The LLM outputs a JSON edit_plan specifying one or more of the following operations:

- **increase_weight**: Increase a component's weight by a specified factor within scale bounds.
- **decrease_weight**: Decrease a component's weight.
- **add_component**: Add a new reward component with a specified type, role, inputs, and parameters.
- **add_event_rule**: Add a one-time reward rule triggered by a condition (e.g., stable landing detected).
- **disable_component**: Disable an existing component without removing it from the schema.

The edit plan is validated for schema consistency (component names exist, types are valid), operator legality (weights within scale bounds, no duplicate additions), and structural integrity before the safe compiler produces executable reward code. This constraint serves two purposes. First, it makes edits auditable: because every change goes through a known operator, the audit system can assess risk at the operator level. Second, it makes edits reversible: because the schema is versioned and edits are structured, rolling back to a previous schema is a single operation.

### 3.6 Risk-Aware Audit and Rollback

Before any edit reaches training, it passes through a behavior risk audit. The audit evaluates the proposed edit across multiple dimensions derived from the trajectory diagnostic report:

- **Reward hacking risk**: Does the edit encourage previously detected failure modes? For example, increasing a contact-based reward component when repeated event exploitation is detected raises the hacking risk.
- **Scale risk**: Does the edit create reward magnitudes that could destabilize training? Weight increases beyond calibrated bounds raise the scale risk.
- **Structural risk**: Does the edit remove safety-related components or disable constraints?

Edits are classified as high risk, medium risk, or low risk. High-risk edits are blocked. Low-risk edits proceed. The treatment of medium-risk edits depends on the audit policy. Under the strict policy, medium-risk edits are blocked when success evidence is weak (no terminal success observed in recent iterations). Under the relaxed policy, medium-risk edits are accepted with a recorded warning.

After the edited reward is trained, an outcome acceptor evaluates whether the policy regressed on key metrics. If regression is detected — the task score drops substantially, or previously resolved failure modes return — the system rolls back to the previous schema and records a regression_lesson. If the outcome is accepted, the new schema becomes the current version and an effective_edit_lesson is stored.

The audit rules used in our experiments are hand-designed for the LunarLander environment and rely on domain-specific signals (contact detection, velocity thresholds, landing region geometry). Extending the audit framework to new environments would require environment-specific rule specification or learned audit models. We discuss this limitation further in Section 5.

## 4. Experiments

### 4.1 Environment and Setup

We evaluate EG-RSA on LunarLander-v3, a continuous control task in the Gymnasium suite [@towers2024gymnasium]. The agent must land a lunar module smoothly on a designated landing pad, achieving low velocity, upright orientation, and stable ground contact. The environment provides an 8-dimensional observation space and a continuous 2-dimensional action space.

All experiments use Proximal Policy Optimization (PPO) [@schulman2017ppo] as the underlying policy optimizer, with a fixed training budget of 1 million environment steps per iteration. The reward search runs for 10 iterations. The LLM edit agent is a DeepSeek-based model. The experiment mode enables all EG-RSA components: memory retrieval, semantic role attribution, hack detection, LLM-based editing, and operator constraints. Free-form code generation is disabled. The default audit policy is strict: medium-risk edits are blocked when success evidence is weak. The environment's oracle reward is used only for post-hoc evaluation scores; it is never provided to the LLM or used in edit decisions.

### 4.2 Main Experiment Overview

Table 1 presents the 10-iteration EG-RSA run under the strict audit policy. The table reports task score (oracle evaluation), semantic score (internal diagnostic score), selection score (combined metric used by the outcome acceptor), hack score (higher values indicate more detected risks), detected failure modes, and the dominant reward component for each iteration.

| Iter | Task Score | Semantic | Selection | Hack | Failure Modes | Dominant Component |
|------|-----------|----------|-----------|------|---------------|-------------------|
| 0 | 0.478 | 0.478 | 0.556 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_approach_region |
| 1 | 0.867 | 0.950 | 1.817 | 0.00 | (none) | r_landing_quality |
| 2 | 0.412 | 0.412 | 0.423 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_approach_region |
| 3 | 0.416 | 0.416 | 0.432 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_landing_quality |
| 4 | 0.367 | 0.367 | 0.335 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_landing_quality |
| 5 | 0.631 | 0.631 | 0.862 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_approach_region |
| 6 | 0.477 | 0.477 | 0.554 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_landing_quality |
| 7 | 0.422 | 0.422 | 0.644 | 0.20 | shaping_goal_mismatch | r_landing_quality |
| 8 | 0.665 | 0.665 | 0.930 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_landing_quality |

**Table 1:** EG-RSA 10-iteration run on LunarLander-v3 under strict audit policy.

The experiment exhibits three distinct phases. First, an effective edit from iteration 0 to iteration 1: the task score improves from 0.478 to 0.867, failure modes are eliminated, and the dominant component shifts from r_approach_region (dense guidance) to r_landing_quality (terminal success). Second, a regression from iteration 1 to iteration 2: the score drops sharply to 0.412 and both failure modes return. Third, a prolonged low-success plateau from iterations 2 through 8: scores remain between 0.367 and 0.665, failure modes persist, and the system appears unable to escape. This experiment is a mechanism verification exercise, not a performance benchmark. We examine each phase as a case study to understand which mechanisms function as designed and where design tensions emerge.

### 4.3 Effective Edit Case Study (Iteration 0 to 1)

**Before the edit (iteration 0).** The initial reward schema produces a policy achieving a task score of 0.478. The attribution report identifies r_approach_region — a dense progress reward that encourages the agent to move toward the landing pad — as the dominant component, with an attribution ratio of 0.42. Two failure modes are detected. *Shaping-goal mismatch*: the agent optimizes approach progress without achieving terminal success. *Repeated event exploitation*: the agent rapidly toggles leg contact (average 370 contact events per episode) without achieving stable landing. The terminal success rate is zero; no episode reaches the stable landing condition.

**The edit.** The LLM receives the current schema, the attribution report, and the memory query result (no relevant prior lessons, as this is the first iteration). Based on the diagnosis — dense guidance dominating, terminal success absent — the LLM proposes: (1) increasing the weight of r_landing_quality, a terminal success reward; (2) adding an event rule that pays a one-time reward upon detecting stable landing; (3) slightly decreasing the weight of r_approach_region. The edit is classified as medium risk (modifying terminal reward structure under weak success evidence). Under the strict audit policy, a first-iteration exception permits the edit since no prior regression evidence exists.

**After the edit (iteration 1).** The policy trained with the edited schema achieves a task score of 0.867. The attribution report now identifies r_landing_quality as the dominant component. Both previously detected failure modes are absent. The shift in dominant component — from dense guidance to terminal success — is consistent with the intended effect of the edit. The system records an effective_edit_lesson containing the schema_diff, the metric delta (+0.389 task score), and the resolved failure modes.

### 4.4 Audit-Induced Deadlock Case Study (Iterations 1 to 8)

**Regression (iteration 1 to 2).** The edit at iteration 1, while improving the task score, introduced a schema configuration that did not generalize stably. At iteration 2, the task score drops from 0.867 to 0.412. Both failure modes return, and r_approach_region again dominates. The system records this as a regression_lesson, storing the schema_diff, the metric delta (-0.455), and the re-emerged failure modes. This lesson is flagged for retrieval in future iterations to warn against similar edit patterns.

**The deadlock (iterations 2 to 8).** Following the regression, the system enters a prolonged low-success regime. The LLM, informed by the regression lesson and the current attribution (r_approach_region or r_landing_quality dominant, failure modes persistent, zero terminal success), repeatedly proposes edits that strengthen terminal success rewards and weaken dense guidance. However, these edits are classified as medium risk: they modify terminal reward structure, and the current success evidence is weak — no terminal success has been observed since iteration 1. Under the strict audit policy, medium-risk edits under weak success evidence are blocked. The repair loop does not resolve this: repairs adjust weights or add constraints but cannot eliminate the medium-risk classification without fundamentally altering the proposed edit. The result is a deadlock: the system needs to modify terminal reward structure to discover successful landing behavior, but those modifications are precisely what the strict audit blocks.

**Deadlock as a design tension.** This deadlock is not an implementation flaw — it reveals a genuine design tension in safe reward search. The same audit mechanism that prevents harmful edits can also prevent the exploration needed to escape a low-success regime. In this deadlock, the system cannot establish the success evidence required to pass audit because audit blocks the edits that would produce that evidence. This tension is fundamental: any reward search system that integrates safety constraints must decide where to draw the line between blocking risk and enabling exploration.

### 4.5 Relaxed Audit Policy

To test whether the deadlock is caused by the audit policy rather than by a fundamental limitation of the search mechanism, we run a second experiment with a relaxed audit policy. Under the relaxed policy, medium-risk edits under weak success evidence are accepted with a recorded warning rather than blocked. All other settings are identical.

| Iter | Task Score | Semantic | Selection | Hack | Failure Modes | Dominant Component |
|------|-----------|----------|-----------|------|---------------|-------------------|
| 0 | 0.478 | 0.478 | 0.556 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_approach_region |
| 1 | 0.511 | 0.511 | 0.621 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_approach_region |
| 2 | 2.490 | 3.907 | 6.397 | 0.00 | (none) | r_landing_quality |
| 3 | 2.953 | 4.453 | 7.406 | 0.00 | (none) | r_landing_quality |

**Table 2:** EG-RSA 4-iteration run on LunarLander-v3 under relaxed audit policy.

Starting from the same initial schema (iteration 0: 0.478), the relaxed-audit run follows a different trajectory. After a modest first edit (iteration 1: 0.511), iteration 2 produces a schema that achieves a task score of 2.490 with no detected failure modes. The dominant component shifts to r_landing_quality. Iteration 3 continues to improve, reaching 2.953.

These results confirm that the deadlock observed under strict audit was audit-induced rather than search-fundamental. The relaxed audit policy permits the medium-risk edits needed to discover successful landing behavior, and the search process — attribution, memory, constrained editing — produces sustained improvement once the audit barrier is lowered.

These results do not imply that audit should be removed. Rather, they motivate a more nuanced approach to audit policy design. A risk-budget mechanism — where medium-risk edits are permitted but their outcomes are closely monitored, and rollback serves as a safety net — may provide a better balance than strict blocking. High-risk edits (e.g., removing all safety constraints) would remain blocked under any policy. The design of such a risk-budget mechanism is a direction for future work.

## 5. Discussion

The experiments reported in this paper serve as mechanism verification, not as a performance benchmark. Three findings emerge. First, attribution-guided memory-augmented editing can produce effective reward modifications: the iteration 0 to 1 case study demonstrates a structured edit that shifts the dominant reward role from dense guidance to terminal success and eliminates detected failure modes. Second, regression is a real phenomenon in reward search, and the system's ability to record regression lessons provides a mechanism for avoiding repeated failures — though the effectiveness of this warning depends on whether the audit policy permits corrective edits. Third, and most significantly, the audit deadlock reveals that integrating safety constraints into reward search creates a fundamental tension between risk prevention and exploration. The strict audit policy that protects against unsafe edits can also suppress the edits needed to establish success evidence, creating a self-reinforcing deadlock.

Several limitations bound the scope of these findings. We evaluate EG-RSA on a single environment (LunarLander-v3). The task metrics used for diagnosis are instantiated from human-specified templates rather than automatically discovered. The audit rules are hand-designed for the specific environment and rely on domain-specific signals. The experiments run for 10 iterations; longer-horizon search dynamics remain unexplored. We do not compare EG-RSA against EUREKA or other LLM reward design methods on standard benchmarks — such comparison would require multi-environment evaluation that is beyond the scope of this mechanism-verification study.

The audit deadlock finding has implications beyond EG-RSA. Any LLM-based reward design system that introduces safety constraints — whether rule-based, learned, or ensemble-based — must decide how to balance safety against the exploration needed to discover better rewards. Our results suggest that strict blocking of medium-risk changes under weak evidence can suppress the very edits needed to establish success evidence in the first place. This is not a problem unique to rule-based audit; it applies to any safety mechanism that makes accept/reject decisions based on current evidence.

Future work includes: (1) evaluating EG-RSA across multiple environments to assess the generality of the search loop; (2) developing learned audit rules that adapt to environment-specific failure modes rather than requiring hand-specification; (3) implementing and testing a risk-budget mechanism as suggested by our relaxed-audit results; (4) extending the iteration count to study longer-horizon search dynamics and potential convergence behavior; and (5) integrating process-level reward models [@zheng2025prmsurvey] to refine attribution granularity beyond per-component to per-step reward analysis.

## 6. Conclusion

We have presented EG-RSA, a framework that reformulates LLM-assisted reward design as experience-guided reward schema search. Rather than treating reward design as code generation with iterative refinement, EG-RSA maintains a versioned, attributed reward schema; stores structured outcome lessons from both effective and regressive edits; constrains LLM modifications to auditable operators; and integrates behavior risk audit with automatic rollback. Mechanism-verification experiments on LunarLander-v3 demonstrate that attribution-guided editing can produce effective reward modifications, that regression is tracked through structured outcome lessons, and that audit policy design involves a fundamental tradeoff between safety and exploration. The audit deadlock observed under strict policy and resolved under relaxed policy suggests that risk-aware reward search requires deliberate audit policy design — not simply strict blocking — and motivates future work on risk-budget mechanisms that balance safety with the exploration needed for effective reward search.

---

**Figure 1:** Overview of one EG-RSA search iteration. Starting from a versioned reward schema, the system compiles the schema, trains a policy, collects trajectory data, performs semantic role attribution, retrieves relevant outcome lessons, prompts the LLM for a constrained edit plan, audits the proposed edit, and stores the outcome as a structured lesson.

*Appendix and extended results omitted for brevity; see supplementary materials for additional iteration traces, ablation configurations, and full diagnostic reports.*
