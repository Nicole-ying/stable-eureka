# Technical Methods Review: EG-RSA

**Paper:** "From Reward Generation to Reward Search: Experience-Guided Schema Adaptation with Semantic Attribution and Risk-Aware Editing"
**Reviewer focus:** Technical methods (soundness, justification, evidence, reproducibility)

---

## 1. Technical Soundness — Score: 3/5

### Summary

The paper proposes a coherent architecture — versioned schemas, per-component attribution, structured outcome memory, operator-constrained editing, and risk-aware audit — connected by a sequential loop (compile, train, attribute, retrieve, edit, audit, store). The high-level design is logically structured and the components have plausible causal relationships: attribution produces a diagnosis, diagnosis informs an edit proposal, audit gates the proposal, and the outcome feeds back into memory.

### Weaknesses

**(a) Audit classification rules are a black box.** The audit evaluates hacking risk, scale risk, and structural risk and classifies edits as high/medium/low risk. Yet the specific rules that produce these classifications are never enumerated. The paper states these rules are "hand-designed for LunarLander" (Section 3.5), but without specifying them, the audit mechanism — which is the paper's central experimental finding — cannot be evaluated or reproduced. For example, what specifically causes an edit that "strengthens terminal success rewards" to be classified as medium risk rather than low risk?

**(b) Attribution computation lacks precision.** The paper states that per-component rewards are aggregated per episode, a dominant component is identified "by its fraction of total reward," and diagnostic signals are "combined into a failure-mode report" (Section 3.2). The exact algorithm is not given: what threshold defines dominance? How are failure modes detected from the diagnostic signals? Are these rule-based detectors, learned classifiers, or LLM prompts? Without this specification, the causal claim that attribution "grounds edit decisions in diagnostic evidence" is asserted rather than demonstrated.

**(c) Memory retrieval similarity metric is unspecified.** The paper states that lessons are "retrieved by similarity to the current failure-mode or schema context" (Section 3.3), but the similarity function — embedding-based, keyword-based, exact match on failure-mode labels — is never defined. This makes the claim that memory enables "history-aware search" difficult to evaluate.

**(d) LLM prompt dependency is unaddressed.** The LLM receives the current schema, attribution report, and retrieved lessons, then produces a JSON edit plan. The prompt engineering is a critical intervening variable: a poorly designed prompt could override both attribution and memory signals. The paper provides no analysis of prompt sensitivity or prompt design, yet attributes edit quality to the EG-RSA architecture rather than to the underlying LLM's reasoning.

---

## 2. Design Justification — Score: 3/5

### Summary

Each component is accompanied by a stated rationale. The schema representation is motivated by the need for versioning, attribution, and auditable edits. Attribution is motivated by the insufficiency of scalar feedback. Memory is motivated by the absence of history in prior approaches. Operator constraints are motivated by auditability and reversibility. Audit is motivated by safety concerns from the reward hacking literature.

### Weaknesses

**(a) Justifications are conceptual, not empirical.** The paper claims EG-RSA departs from prior work in four specific ways (Section 2, "Reward generation" paragraph). However, no ablation or comparison demonstrates that any single departure — schema vs. code generation, attribution vs. scalar feedback, memory vs. no memory, audit vs. no audit — actually improves outcomes. The appendix lists planned ablations as "pending." Without them, the reader cannot distinguish genuine architectural contributions from unnecessary complexity.

**(b) Design choices lack competitive rationale.** Why five edit operators rather than three or seven? Why these five semantic roles rather than a different taxonomy? Why 1M training steps per iteration rather than 500K or 2M? The paper does not explain how these choices were made, nor does it discuss alternatives considered and rejected. This makes the design appear arbitrary rather than principled.

**(c) The shift from population-based to sequential search is asserted, not argued.** Section 3.1 contrasts EG-RSA with "population-based reward generation loop of Stable-Eureka," claiming EG-RSA replaces parallel population sampling with sequential single-schema editing. Whether this is an improvement or merely a design difference is never established. Sequential editing is more sample-efficient in training steps but more vulnerable to getting trapped (as the deadlock finding itself demonstrates), yet this tradeoff is not discussed.

---

## 3. Evidence-Claim Alignment — Score: 2/5

### Summary

The paper's strongest empirical contribution is the audit-induced deadlock finding and its resolution under relaxed audit (Tables 1 and 2). This finding is clearly presented and directly supports the paper's central message about a safety-exploration tradeoff. However, several other claims lack commensurate evidence.

### Weaknesses

**(a) Single-run evidence for core mechanism claims.** The main experiment (Table 1) is a single 10-iteration run under strict audit. The relaxed-audit experiment (Table 2) is a single 4-iteration run. No multiple seeds, no error bars, no statistical testing. The claim that the deadlock is "audit-induced" relies on comparing exactly two trajectories. With only one run per condition, the deadlock could be attributable to PPO training noise, LLM stochasticity, or initialization artifacts rather than audit policy.

**(b) The "effective edit" case study is a single instance.** The paper presents Iter 0 to 1 as evidence that "attribution-guided editing with structured memory can produce effective reward modifications" (Section 5). One positive edit, from a single run, does not establish that attribution caused the improvement. The LLM could have proposed the same edit given only the scalar score (0.478) and no attribution at all. Without a no-attribution control, the attribution mechanism's contribution is unverified.

**(c) Memory contribution is invisible.** The paper claims structured outcome lessons "turn feed-forward refinement into history-aware search" (Section 3.3), but no experiment isolates the effect of memory. At Iter 0 to 1, no prior lessons exist, so the effective edit cannot be attributed to memory. During the deadlock phase (Iter 2-8), lessons exist but edits are blocked by audit, so memory's influence is again unobservable. The paper provides no evidence that retrieved lessons actually influence LLM edit proposals.

**(d) Claims exceed experimental scope.** The Introduction claims contributions including "experience-guided reward schema search," "diagnosis-driven editing via semantic role attribution," and "operator-constrained editing with integrated risk audit" (Section 1). The experiments verify none of these contributions individually — they demonstrate the integrated system on one environment. The contribution claims are architectural proposals, not verified results.

---

## 4. Reproducibility — Score: 2/5

### Summary

The paper specifies the environment (LunarLander-v3), RL algorithm (PPO), and core PPO hyperparameters. Experiment directories are referenced for full iteration traces. However, critical reproducibility elements are absent.

### Weaknesses

**(a) LLM specification is insufficient.** The paper states the edit agent is "DeepSeek-based" (Section 4.1, Appendix B) without specifying the model version (DeepSeek-V2? V3? R1?), the API parameters (temperature, top-p, max tokens), or the prompt template. LLM behavior is highly sensitive to these choices. Two reviewers using different DeepSeek versions or different prompts would obtain different results.

**(b) Audit rules and thresholds are not provided.** The audit classification function is the paper's central mechanism, yet its implementation is described only at the conceptual level. What weight change magnitudes trigger scale risk? What structural changes trigger structural risk? What specific conditions define "weak success evidence"? Without these rules, the deadlock cannot be reproduced or studied.

**(c) Initial schema is not shown.** The starting reward schema — from which all edits proceed — is never presented. The paper reports that at Iter 0 the dominant component is `r_approach_region` with task score 0.478, but the full schema (all components, their weights, parameters, event rules) is absent. Since all edits modify this schema, its initial design is a critical experimental parameter.

**(d) No code or data release.** No repository link, supplementary code archive, or data availability statement is provided. While experiment directories are referenced (Appendix C), these are local paths on the authors' filesystem, not publicly accessible artifacts.

**(e) Single seed, no variance reporting.** As noted under Evidence-Claim Alignment, the absence of multiple random seeds prevents any assessment of result stability. The number of parallel environments (64) is specified, but the seed strategy for environment initialization and PPO training is not.

---

## Key Findings with Examples

**Finding 1: Audit mechanism is underspecified despite being the central finding.**

The paper's most interesting result — the audit-induced deadlock — cannot be evaluated or reproduced because the audit classification rules are never defined. Section 3.5 states that edits are classified as high/medium/low risk along three dimensions, and medium-risk edits under weak evidence are blocked under strict audit. However, the text never specifies what about the LLM's proposed edits (e.g., "increase terminal success reward weight, decrease dense guidance weight") triggers a medium-risk classification. This omission is structural: the audit is the independent variable in the paper's key experiment, yet its implementation is hidden.

**Finding 2: Attribution and memory contributions are claimed but unverified.**

The paper asserts that semantic role attribution "grounds edit decisions in diagnostic evidence" (Section 1) and that outcome memory enables "history-aware search" (Section 3.3). The experiment tables report the dominant role per iteration and list failure modes, showing that attribution was computed. However, no evidence connects these attribution outputs to specific LLM edit decisions, and no ablation isolates memory's effect. The appendix lists planned ablations as "pending," which means the paper's architectural claims about individual component contributions are currently unsupported. The single effective edit (Iter 0 to 1) occurs when memory is empty, so memory contributes nothing to the paper's only demonstrated success under strict audit.

**Finding 3: Evidence base is too narrow for the claimed contributions.**

The paper lists three numbered contributions (experience-guided schema search, diagnosis-driven editing, operator-constrained editing with audit) and makes comparative claims against prior work (EUREKA, Text2Reward, Auto MC-Reward, CARD). The evidence consists of two single-seed runs on one environment (LunarLander-v3) with no baselines. The paper acknowledges this limitation ("We do not benchmark against EUREKA or other methods"), but the contribution claims in the Introduction are not scoped to match the narrow evidence. The mismatch between claimed scope and demonstrated scope undermines the paper's core assertions.

---

## Overall Recommendation: **Major Revision**

**Rationale:** The paper presents a conceptually interesting architecture and an intriguing empirical finding (audit-induced deadlock). However, the evidence base is too narrow (single environment, single seed per condition, no ablations, no baselines) to support the claimed contributions, and critical methodological details (audit rules, attribution algorithm, memory retrieval metric, LLM specification, initial schema) are absent, making the work unreproducible as presented. A major revision should:

1. Provide the complete audit rule set, attribution algorithm, and initial reward schema.
2. Specify the LLM model version, prompt template, and generation parameters.
3. Report results over multiple random seeds with variance estimates.
4. Include at minimum one ablation (e.g., no-attribution, no-memory) to isolate mechanism contributions, or rescope claims to match the available evidence.
5. Release code or provide a complete specification sufficient for independent reimplementation.
