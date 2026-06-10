# Contribution Review: "From Reward Generation to Reward Search"

## Assessment Overview

This paper proposes reformulating LLM-assisted reward design from code generation with iterative refinement to experience-guided schema search (EG-RSA). The system maintains a versioned, componentized reward schema, performs semantic role attribution, retrieves structured outcome lessons, constrains the LLM to auditable edit operators, and gates every edit through risk audit with rollback. Experiments on LunarLander-v3 under strict vs. relaxed audit policies reveal an audit-induced deadlock.

---

## Dimension Scores

### 1. Novelty — Is the reformulation (generation to search) genuinely novel?

**Score: 3 / 5**

The reformulation is conceptually clean and well-motivated, but its component parts draw substantially from established techniques that the paper itself acknowledges. Versioned schemas have clear antecedents in structured programming and configuration management. Structured memory for agent decision-making is a core technique in Reflexion (Shinn et al. 2023), Voyager (Wang et al. 2023), and Generative Agents (Park et al. 2023). Per-component reward attribution echoes diagnostic instrumentation practices common in RL debugging. Operator-constrained editing is a safety technique with a long lineage in program synthesis.

Where the paper earns its novelty score is in two places. First, the *integration* of these components into a single closed-loop search process is not present in prior work — no existing LLM reward-design system combines attribution, structured memory, operator constraints, and in-loop audit with rollback. Second, the *audit deadlock finding* is genuinely novel. The observation that strict safety gating can suppress the exploration needed to establish the very success evidence that would satisfy the gate is a non-obvious emergent property of the integrated system. This finding does not fall out of any individual component and was not predicted by the design.

The reformulation claim itself — "reward design should instead be structured as experience-guided schema search" (Introduction, para. 4) — is more of a reframing than a discovery. The paper rearranges known building blocks into a new architecture, which is valid engineering but does not by itself constitute a breakthrough in understanding. The deadlock finding is the paper's strongest novel contribution.

### 2. Significance — Would this change how researchers approach LLM reward design?

**Score: 3 / 5**

The audit deadlock finding has potential to influence research practice. It reveals a structural tension that any safety-augmented reward-design loop will encounter: the edits needed to escape a low-performance regime are precisely those that a conservative safety gate is most likely to block. This is not specific to EG-RSA — the paper correctly notes that any system introducing safety constraints into the design loop faces the same tradeoff (Discussion, para. 3). If the finding generalizes, it would motivate a class of mechanisms (risk budgets, adaptive thresholds, exploration windows) that current reward-design systems do not include.

However, the demonstrated significance is substantially limited by scope. The evaluation is confined to a single environment (LunarLander-v3), a single algorithm (PPO), and 10 iterations per run. There are no quantitative baselines against EUREKA, Text2Reward, or even an ablated EG-RSA variant. The paper explicitly states it is "mechanism verification, not a performance benchmark" (Discussion, para. 1), which is honest but limits the significance claim. Researchers cannot assess whether the schema-search reformulation yields practical gains — better sample efficiency, higher final scores, more reliable convergence — over the generation paradigm it critiques. The planned ablations (memory-off, attribution-off, operator-constraints-off) are listed as "pending" in the Appendix, leaving the contribution of each component unquantified.

If the deadlock finding replicates across environments and the component ablations show that attribution and memory each contribute independently, the significance would rise substantially. In its current form, the paper is a promising proof of concept rather than a demonstrated advance.

### 3. Gap Positioning — Is the gap clearly and concretely identified vs. SOTA?

**Score: 4 / 5**

The gap is positioned with unusual clarity. The Introduction (para. 3) enumerates four specific structural limits of the code-generation paradigm:

1. No persistent structured representation — each iteration generates code from scratch.
2. No systematic diagnosis of failure — scalar feedback signals performance change but not which component caused it.
3. No memory of past editing outcomes — each iteration starts fresh.
4. No integrated safety audit — generated code executes without pre-execution risk assessment or rollback.

These are falsifiable claims. If any existing system maintained a versioned schema across iterations, performed per-component attribution, stored structured outcome lessons, or gated edits through pre-execution audit, the gap would be narrower than claimed. The Related Work section (para. 2) explicitly lists four departures from prior LLM reward-generation methods, giving the reader a concrete checklist. The paper also explicitly contrasts EG-RSA's sequential schema-editing loop against the population-based generation loop of Stable-Eureka (Method, Search Loop Overview, para. 3), which is a closer architectural comparator than EUREKA alone.

The one weakness in gap positioning is that the gap is argued at the architectural level without quantitative evidence of its severity. The four structural limits are asserted as problems, and the reader is asked to accept that they matter. A brief quantitative illustration — e.g., showing that the generation paradigm's iteration-to-iteration variance or failure-mode recurrence rate is high — would strengthen the gap's concreteness. Without it, the gap relies on the reader sharing the authors' architectural intuition.

### 4. Claim Calibration — Are claims honest, specific, and not overbroad?

**Score: 4 / 5**

The paper is notably disciplined about scope. The Discussion section opens with an explicit boundary: "These experiments serve as mechanism verification, not a performance benchmark" (para. 1). Five specific limitations are listed (para. 2): single environment, template-instantiated metrics, hand-designed audit rules, 10-iteration horizon, and no baselines. This level of self-critique is above average for the field and builds reviewer trust.

The three advertised contributions (Introduction, para. 5) are deliverable: the paper does present a schema-search architecture, does perform semantic-role attribution, and does implement operator-constrained editing with risk audit. The contribution statements are specific ("reformulate LLM-assisted reward design as search over structured, versioned schemas with cross-iteration outcome memory") rather than vague ("we improve reward design").

One mild overclaim warrants attention. The paper states "We reformulate LLM-assisted reward design as experience-guided schema search" in both the Abstract and Introduction. The verb "reformulate" applied to the entire subfield, supported by a single-environment mechanism-verification study, strains the evidence base. The reformulation would be more accurately described as "We propose" or "We introduce a framework for" — reserving "we reformulate" for a paper that demonstrates the reformulation's generality across environments and baselines. Elsewhere, the paper correctly hedges with "we argue that" (Introduction, para. 4), which is appropriately calibrated. The title's "From Reward Generation to Reward Search" is also ambitious as a field-level statement, though titles are conventionally permitted more license.

The claim about the audit deadlock having "implications beyond EG-RSA" (Discussion, para. 3) is appropriately hedged with "any LLM-based reward design system that introduces safety constraints... faces the same tradeoff." This claim is falsifiable (it predicts that adding safety gating to EUREKA or Text2Reward would produce analogous deadlocks) and is stated at a level the paper's evidence can support.

---

## Specific Findings

### Finding 1: Gap enumeration is precise and falsifiable (positive)

**Location:** Introduction, paragraph 3; Related Work, paragraph 2.

The paper lists four structural limits of the code-generation paradigm: no persistent structured representation, no systematic diagnosis, no memory of past editing outcomes, and no integrated safety audit. These are not vague complaints — each one asserts the absence of a specific capability in existing systems. The Related Work section then lists four corresponding departures. This gives the reader a concrete mental model of what EG-RSA adds and what would falsify the gap claim (finding any of these capabilities in a prior system). The architectural contrast with Stable-Eureka's population-based loop (Method, Search Loop Overview) further sharpens the positioning.

### Finding 2: Audit deadlock is the strongest empirical contribution (positive)

**Location:** Experiments, "Regression and audit deadlock" and "Relaxed Audit Policy."

The deadlock phenomenon is cleanly demonstrated. Under strict audit, scores oscillate between 0.367 and 0.665 across 7 iterations (Iter 2-8), with failure modes persisting and medium-risk edits repeatedly blocked. Under relaxed audit (identical setup), the system achieves 2.490 at Iter 2 and 2.953 at Iter 3. The contrast is stark and the causal attribution is well-supported: the only changed variable is the audit policy's treatment of medium-risk edits under weak evidence. The paper correctly interprets this not as a bug but as a fundamental design tension — the safety mechanism that prevents harmful edits also blocks the exploration needed to establish success evidence. This finding is the paper's most likely route to impact.

### Finding 3: Evidence base is undersized for the reformulation claim (concern)

**Location:** Discussion, paragraph 2; Appendix, "Planned Ablations."

A paper arguing that LLM reward design should be reformulated from generation to search invites the question: does the reformulation actually work better? The paper does not answer this question quantitatively. There are no comparisons to any generation-paradigm baseline. The component ablations — which would distinguish whether the reformulation's value comes from the schema structure, the attribution, the memory, or the operator constraints — are "planned" and explicitly listed as "pending." The evaluation is a single environment (LunarLander-v3), a standard benchmark but one where reward design is relatively well-understood. The reformulation claim, while honestly scoped as mechanism verification, would carry more weight with either (a) ablation results isolating each component's contribution, or (b) a second environment demonstrating that the deadlock phenomenon is not LunarLander-specific.

---

## Recommendation

**Minor Revision**

**Rationale:** The conceptual architecture is sound, the audit deadlock is a genuinely interesting and non-obvious finding, and the paper is unusually honest about its scope and limitations. The gap is precisely articulated and the claims are mostly well-calibrated.

The required revisions are bounded:

1. **Add at least one ablation or one additional environment.** Running the memory-off or attribution-off ablation would isolate which components drive the reformulation's value. Alternatively, replicating the strict-vs-relaxed deadlock on a second environment (e.g., BipedalWalker or a MuJoCo task) would demonstrate that the deadlock is a general phenomenon rather than a LunarLander artifact. Either would substantially strengthen the evidence-to-claim link without requiring a full benchmarking suite.

2. **Soften the reformulation language.** Change "We reformulate LLM-assisted reward design as..." to "We propose a framework for experience-guided reward schema search that reformulates LLM-assisted reward design by..." in the Abstract and Introduction. The title "From Reward Generation to Reward Search" is acceptable as a title-level claim but the body text should reflect the proposal/mechanism-verification scope.

3. **Briefly justify why the four structural limits matter.** Add one sentence to the Introduction quantifying a consequence of the generation paradigm that the reformulation addresses (e.g., iteration-to-iteration variance in reward quality, or failure-mode recurrence rate), even if drawn from prior work rather than original measurement.

These revisions are achievable within a review cycle and would materially strengthen the paper without changing its architecture or findings.
