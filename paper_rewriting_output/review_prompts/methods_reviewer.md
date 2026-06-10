# Methods & Reproducibility Reviewer

## Role

Assess methodological clarity, reproducibility, assumption justification, experimental design, and limitations.

**IMPORTANT:** You are an independent reviewer. Do NOT read or reference the other reviewers' work. Your assessment must stand entirely on its own. Do not mention what other reviewers might say.

## Rubric (score 1-5 for each)

- Method description completeness (1=insufficient, 5=fully replicable)
- Assumption justification (1=unstated, 5=explicit with rationale)
- Experimental design (1=flawed, 5=rigorous)
- Limitations acknowledgment (1=none, 5=thorough)

## Manuscript Sections

### Introduction
In reinforcement learning (RL), the reward function defines the optimization objective and directly shapes learned behavior [REF] . Designing effective rewards, however, remains a central bottleneck: rewards must provide useful gradients, avoid unintended shortcuts, and stay aligned with the true task objective rather than merely correlated with it [REF] .
Recent work has shown that large language models (LLMs) can serve as effective reward designers. EUREKA demonstrates that LLMs can generate executable reward code and improve it through iterative in-context optimization [REF] . Text2Reward generates dense shaped rewards from language descriptions [REF] . Auto MC-Reward combines a Reward Designer, Critic, and Trajectory Analyzer to refine rewards from collected trajectories [REF] . CARD proposes dynamic-feedback reward design with Trajectory Preference Evaluation [REF] . These methods establish LLMs as powerful reward design tools [REF] .
These methods, however, share a common paradigm: they treat reward design as code generation with iterative refinement. The LLM emits free-form reward code; the code is trained and evaluated; the next iteration receives scalar or text feedback. This paradigm has structural limits. There is no persistent structured representation of the reward across iterations---each iteration generates code from scratch. There is no systematic diagnosis of why a reward failed: scalar feedback signals that performance changed, but not which reward component drove the change. There is no memory of past editing outcomes---each iteration starts fresh, unable to learn from the system's own history of effective and regressive edits. There is no integrated safety audit: generated code executes without pre-execution risk assessment or rollback.
We argue that LLM-assisted reward design should be reformulated as experience-guided reward schema search . Rather than generating complete reward functions from scratch, the system should maintain a versioned, componentized schema with semantic role labels; diagnose which components drive policy behavior through per-component attribution; retrieve relevant past editing outcomes from structured memory; constrain the LLM to auditable edit operators; and gate every edit through risk audit with automatic rollback.
We present EG-RSA (Experience-Guided Reward Search Agent), which implements this reformulation. Starting from an initial reward schema, each iteration compiles the schema, trains a policy for a fixed budget, collects step-level trajectory data with per-component logging, performs semantic role attribution, retrieves relevant outcome lessons, prompts the LLM for a constrained edit plan, audits the plan, and stores the outcome as a structured lesson.
Our contributions are: enumerate [leftmargin=*,nosep] Experience-guided reward schema search. We reformulate LLM-assisted reward design as search over structured, versioned schemas with cross-iteration outcome memory. Diagnosis-driven editing 

### Related Work
Reward design in RL. The reward function is the central interface between task intent and policy learning [REF] . Classical reward shaping studies how additional signals affect policy learning, with potential-based shaping providing conditions for policy invariance [REF] . Inverse reinforcement learning infers rewards from expert demonstrations [REF] . EG-RSA differs in both mechanism and assumptions: it assumes neither expert demonstrations nor hand-designed shaping potentials. Instead, it adapts reward schemas from training feedback, attribution, and accumulated editing experience.
LLM-assisted reward generation. EUREKA established LLMs as reward designers through iterative in-context optimization of reward code~ [REF] . Text2Reward~ [REF] , Auto MC-Reward~ [REF] , and CARD~ [REF] further demonstrate that LLMs can generate and refine dense reward functions from language descriptions, trajectory analysis, and dynamic feedback. A recent survey organizes the LLM-enhanced RL landscape~ [REF] . These methods share a code-generation paradigm. EG-RSA departs in four ways: (1)~a persistent versioned schema replaces per-iteration code generation, (2)~semantic role attribution enables diagnosis-driven editing, (3)~structured outcome lessons enable history-aware search, and (4)~auditable operators with integrated risk audit replace free-form code generation.
Memory and reflection in LLM agents. Reflexion stores verbal reflections in episodic memory to improve agent decisions without weight updates~ [REF] . Voyager maintains a growing skill library for lifelong embodied learning~ [REF] . Generative Agents use memory streams for long-horizon behavior~ [REF] . A recent survey documents memory as a core agent capability~ [REF] . EG-RSA adopts the memory intuition but applies it to a new object: structured reward-edit outcome lessons containing schema diffs, metric deltas, failure modes, and rollback decisions---directly actionable for future editing, unlike verbal reflections or skill code.
Reward hacking and safety. Reward hacking is a fundamental concern: agents may exploit proxy rewards while failing the true task [REF] . Recent mitigation includes reward model ensembles~ [REF] , information-theoretic reward modeling~ [REF] , and theoretical analysis of Goodhart's law under heavy-tailed error~ [REF] . LLMs can even generalize from sycophancy to reward tampering~ [REF] . These works diagnose and mitigate hacking from outside the reward design loop. EG-RSA integrates safety directly: behavior risk audit gates every edit before training, with explicit risk triage and rollback. As our experiments show, however, strict audit can create a deadlock that blocks necessary exploration---a tension absent from post-hoc safety approaches.
Positioning. EG-RSA occupies a distinct position: it combines structured schema representation, semantic attribution, outcome memory, and integrated audit into a single search loop. This shifts the LLM's role from one-shot reward code

### Method


### Search Loop Overview
Figure~ [REF] illustrates one EG-RSA iteration. Starting from a versioned reward schema, the system: (1)~compiles the schema into an executable reward function via a trusted compiler; (2)~trains a policy using the generated reward for a fixed sufficient budget (1M steps for LunarLander); (3)~collects step-level trajectories with per-component reward logging; (4)~performs semantic role attribution to identify the dominant reward role and detect failure modes; (5)~retrieves relevant prior outcome lessons from structured memory; (6)~prompts the LLM to propose a constrained edit plan in JSON; (7)~validates, audits, and either accepts, repairs, or blocks the edit; and (8)~stores the outcome as a structured lesson.
The policy is trained exclusively with the generated reward. The environment's oracle reward is used only for post-hoc evaluation---never for reward selection or editing. This separation ensures the search process does not leak ground-truth information.
EG-RSA's loop differs structurally from the population-based reward generation loop of Stable-Eureka and similar methods. The population approach generates multiple complete reward-code samples per iteration, trains each in parallel, selects the best via an oracle fitness score, and reflects on the winner. EG-RSA replaces this with a sequential schema-editing loop: one schema is maintained and edited across iterations, the LLM proposes structured edits rather than complete rewrites, training feedback is diagnosed through attribution rather than reduced to a scalar, and audit gates every edit before training.

### Reward Schema and Semantic Role Attribution
EG-RSA represents rewards as componentized schemas rather than unconstrained code. Each component has a type, weight, input variables, parameters, and an enabled flag. Every component is assigned a semantic role: dense\_guidance (progress rewards), stability\_quality (smoothness and contact quality), terminal\_success (task completion), safety\_constraint (violation penalties), and control\_cost (action penalties). Event rules for one-time condition-triggered rewards are also schema elements.
This representation enables three capabilities. First, versioning: schemas can be diffed, stored, and rolled back. Second, per-component attribution: because each component is separately logged at every step, the system can measure which role dominates policy behavior. Third, operator-constrained editing: the LLM proposes edits to named components with typed operators, not free-form code.
Scalar scores alone cannot explain why a reward failed. A policy may score well on dense progress rewards while never achieving terminal success ( shaping-goal mismatch ), or exploit contact rewards through rapid toggling without stable landing ( repeated event exploitation ). EG-RSA computes per-component reward attribution from step-level trajectory data: per-step component rewards are aggregated per episode, the dominant component is identified by its fraction of total reward, and diagnostic signals (success rate, contact frequency, progress trajectory, reward repetition) are combined into a failure-mode report.
The attribution report---dominant role, failure modes, risk flags---is provided to the LLM alongside the current schema and retrieved lessons. The LLM uses this diagnosis as evidence when proposing edits, but attribution does not mechanically determine the edit.

### Outcome Lesson Memory
EG-RSA stores structured outcome lessons rather than verbal reflections~ [REF] or skill code~ [REF] . Each lesson contains the schema diff (previous and edited schema), metric deltas (task, semantic, and hack scores), failure modes before and after the edit, hack risk change, and the rollback decision. Lessons are classified as effective\_edit\_lesson (score improved) or regression\_lesson (score degraded). Both types are stored and retrieved by similarity to the current failure-mode or schema context. Effective lessons provide evidence for which edit strategies have worked; regression lessons warn against previously harmful edits. This turns feed-forward refinement into history-aware search.

### Operator-Constrained Editing
The LLM outputs a JSON edit plan restricted to five operators: increase\_weight , decrease\_weight , add\_component , add\_event\_rule , and disable\_component . The plan is validated for schema consistency, operator legality, and scale bounds before a safe compiler produces executable reward code. This constraint makes edits auditable (every change goes through a known operator) and reversible (rollback is a single schema restoration).

### Risk-Aware Audit and Rollback
Every proposed edit passes through a behavior risk audit before training. The audit evaluates three dimensions: hacking risk (does the edit encourage previously detected failure modes?), scale risk (do weight changes risk destabilizing training?), and structural risk (does the edit remove safety components?). Edits are classified as high, medium, or low risk. High-risk edits are blocked. Low-risk edits proceed. Medium-risk edits are treated according to the audit policy: under the strict policy, they are blocked when success evidence is weak; under the relaxed policy, they are accepted with a warning.
After training, an outcome acceptor checks for regression. If the task score drops substantially or resolved failure modes return, the system rolls back to the previous schema and records a regression lesson. Otherwise, the new schema becomes current and an effective-edit lesson is stored. The audit rules in our experiments are hand-designed for LunarLander; extending to new environments requires environment-specific rules or learned audit models.

### Experiments


### Setup
We evaluate EG-RSA on LunarLander-v3, a continuous control task in the Gymnasium suite~ [REF] . The agent must land a lunar module smoothly on a landing pad with low velocity, upright orientation, and stable ground contact (8-dim observation, 2-dim continuous action). All experiments use PPO~ [REF] with a fixed 1M-step training budget per iteration, running for 10 iterations with a DeepSeek-based LLM edit agent. All EG-RSA components are enabled: memory, attribution, hack detection, operator constraints. The oracle reward is used only for post-hoc evaluation. The default audit policy is strict.

### Main Experiment and Case Studies
Table~ [REF] presents the 10-iteration run under strict audit. The experiment exhibits three phases: an effective edit (Iter 0$ $1), a regression (Iter 1$ $2), and a prolonged low-success plateau (Iter 2--8). We examine each as a case study.
Effective edit (Iter 0$ $1). At iteration 0, the initial schema produces a policy achieving task score 0.478. Attribution identifies r\_approach\_region (dense guidance) as dominant (ratio 0.42). Two failure modes are detected: shaping-goal mismatch (no terminal success despite progress) and repeated event exploitation (rapid leg-contact toggling, 370 events/episode average). The LLM, informed by this diagnosis and having no prior lessons (first iteration), proposes: increase r\_landing\_quality weight, add a stable-landing event rule, decrease r\_approach\_region . The edit is classified medium-risk but permitted as a first-iteration exception. After training, task score reaches 0.867, r\_landing\_quality becomes dominant, and both failure modes are absent. The system records an effective-edit lesson.
Regression and audit deadlock (Iter 1$ $8). At iteration 2, the task score drops to 0.412, failure modes return, and r\_approach\_region again dominates. The system records a regression lesson (metric delta $-$0.455). Following this regression, the LLM repeatedly proposes edits that strengthen terminal success rewards and weaken dense guidance---the same strategy that succeeded at iteration 1. However, these edits are now classified as medium risk (modifying terminal reward structure), and success evidence is weak (no terminal success since iteration 1). Under the strict audit policy, medium-risk edits with weak success evidence are blocked. The repair loop cannot resolve this: repairs adjust weights but cannot eliminate the risk classification. The result is a deadlock across iterations 2--8: scores remain between 0.367--0.665, failure modes persist, and the system cannot escape.
This deadlock is not an implementation flaw---it reveals a fundamental design tension. The same audit mechanism that prevents harmful edits can also block the exploration needed to establish success evidence. The system needs to modify terminal rewards to discover successful landing, but those modifications are precisely what strict audit blocks when success evidence is absent.

### Relaxed Audit Policy
To test whether the deadlock is audit-induced, we run a second experiment with a relaxed policy: medium-risk edits under weak evidence are accepted with a warning. All other settings are identical.
Starting from the same initial schema (0.478), the relaxed-audit run follows a different trajectory. After a modest first edit (0.511), iteration 2 achieves a breakthrough: task score 2.490, no failure modes, r\_landing\_quality dominant. Iteration 3 continues to improve (2.953). These results confirm the deadlock was audit-induced: the same search mechanism---attribution, memory, constrained editing---produces sustained improvement once the audit barrier is lowered.
These results do not imply audit should be removed. Rather, they motivate a risk-budget mechanism: high-risk edits remain blocked, medium-risk edits are permitted with monitoring, and rollback serves as a safety net. Designing such a mechanism is future work.

### Discussion
These experiments serve as mechanism verification, not a performance benchmark. Three findings emerge. First, attribution-guided editing with structured memory can produce effective reward modifications, as demonstrated by the iteration 0$ $1 case study. Second, regression is real in reward search and the system tracks it through structured outcome lessons. Third, the audit deadlock reveals a fundamental tension: integrating safety constraints into reward search creates a tradeoff between risk prevention and exploration, where strict blocking can suppress the edits needed to establish success evidence.
Our study has several limitations. We evaluate on a single environment (LunarLander-v3). Task metrics are template-instantiated rather than auto-discovered. Audit rules are hand-designed for this environment. Experiments run for 10 iterations; longer-horizon dynamics remain unexplored. We do not benchmark against EUREKA or other methods---such comparison requires multi-environment evaluation beyond this mechanism-verification scope.
The audit deadlock finding has implications beyond EG-RSA. Any LLM-based reward design system that introduces safety constraints---rule-based, learned, or ensemble-based---faces the same tradeoff. Strict blocking of medium-risk changes under weak evidence can suppress the edits needed to establish that evidence in the first place.
Future work includes multi-environment evaluation, learned audit rules that adapt to environment-specific failure modes, implementation of a risk-budget mechanism, and integration of process-level reward models~ [REF] to refine attribution granularity.

### Conclusion
We reformulated LLM-assisted reward design as experience-guided schema search, introducing semantic role attribution, structured outcome memory, and integrated risk audit with rollback. Mechanism-verification experiments on LunarLander-v3 demonstrate effective editing, regression tracking, and an audit-induced deadlock that reveals a safety--exploration tradeoff. Resolving this deadlock through relaxed audit motivates future work on risk-budget mechanisms for safe reward search.

### Appendix: Extended Experiment Details


### Environment and Hyperparameters
LunarLander-v3 provides an 8-dimensional observation space (position, velocity, angle, angular velocity, leg contact) and a 2-dimensional continuous action space (main engine, side engines). PPO hyperparameters: learning rate $3 10^ -4 $, 64 parallel environments, 256-step rollout, 10 epochs per update, GAE $ =0.95$, clipping $ =0.2$. Training budget: $10^6$ environment steps per iteration.

### Experiment Mode
All EG-RSA components are enabled: use\_memory=true , use\_attribution=true , use\_hack\_detector=true , use\_llm\_edit=true , use\_operator\_constraints=true , free\_rewrite=false . The LLM edit agent is DeepSeek-based.

### Extended Iteration Traces
Full iteration data including per-iteration edit plans, audit reports, repair logs, and outcome lesson cards are available in the experiment directories under experiments/eg\_rsa\_lunar\_lander\_v1\_role\_attrib\_10x1m/ and experiments/eg\_rsa\_lunar\_lander\_v1\_role\_attrib\_10x1m\_audit\_relaxed/ .

### Planned Ablations
Planned experiments include: disabling memory retrieval (feed-forward baseline), disabling attribution (scalar-feedback-only baseline), and removing operator constraints (free-code baseline). These ablations will isolate the contribution of each mechanism. Configurations exist in the configs/ directory; experimental results are pending.

## Instructions

1. Score each rubric dimension (1-5) with a brief justification.
2. List at least 3 specific findings. Reference section names.
3. Recommend: Accept / Minor Revision / Major Revision / Reject.
4. Write your review in clear, structured Markdown.

Write only your review. Do NOT produce other files.
