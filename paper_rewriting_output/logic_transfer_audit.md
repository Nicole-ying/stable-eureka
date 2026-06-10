# Logic Transfer Audit

Verifies that every confirmed-motivation claim in the original materials is transferred to the revised manuscript (or explicitly rejected). No claim should disappear without a decision.

| Original Claim | Source | Transferred To | Status | Audit Note |
|---|---|---|---|---|
| "From reward generation to reward search" | U02, U03 | Title (F2), Introduction P3-P5 (F6-F8), Abstract (F3) | Transferred | Core claim preserved and strengthened with mechanism-level specificity. |
| "LLM-assisted reward design as experience-guided reward search problem" | U02 | Introduction P5 (F8), Method 3.1 (F20) | Transferred | Reformulation claim anchors Introduction and Method overview. |
| "Semantic role attribution to identify which reward roles dominate policy behavior" | U04 | Introduction P5 (F8), Method 3.3 (F24-F26), Experiments 4.3 (F35-F37) | Transferred + Evidence | Added concrete attribution traces from EXP-01 showing r_approach_region → r_landing_quality shift. |
| "Outcome lesson memory: effective and regressive edits stored as structured lessons" | U05 | Introduction P5 (F8), Method 3.4 (F27), Experiments 4.4 (F38) | Transferred + Evidence | Added regression lesson evidence from Iter 1→2. |
| "Operator-constrained editing: LLM outputs edit plan, not free code" | U06 | Introduction P5 (F8), Method 3.5 (F28) | Transferred | Preserved as a core architectural contribution with explicit contrast to SOTA free-code methods. |
| "Risk-aware audit and rollback" | U06 | Introduction P5 (F8), Method 3.6 (F29-F31), Experiments 4.4-4.5 (F39-F42) | Transferred + Honest Finding | Added audit deadlock case study as honest finding; added audit limitations paragraph. |
| "EG-RSA does not rely on unconstrained LLM reward-code generation" | U14 | Method 3.5 (F28) | Transferred | Core to operator constraint contribution. |
| "Task metrics are instantiated from templates, not fully auto-discovered" | U16 | Method 3.3 (F24), Discussion P2 (F44) | Transferred + Limitation | Anti-overclaiming statement preserved in Method. Added as limitation in Discussion. |
| "Fixed sufficient training budget" (not "complete training") | U19 | Method 3.1 (F20), Experiments 4.1 (F32) | Transferred | Language constraint preserved. |
| "Oracle reward not used for reward selection or editing" | Design doc S3.1 | Method 3.1 (F20), Experiments 4.1 (F32) | Transferred | Explicit statement preserved to prevent "oracle leakage" claim. |
| "Contribution count: max 4" | Writing plan | Introduction P5 (F8) | Transferred + Reduced | Reduced to 3 contributions: (1) search framing + memory, (2) attribution, (3) audit. Per confirmed motivation. |
| "Current experiment is mechanism verification, not final performance result" | U24 | Experiments 4.2 (F33-F34), Discussion P1 (F43) | Transferred | Framing preserved and made explicit throughout. |
| "Iteration 0→1 proves attribution + memory can produce effective edit" | Writing plan S4 | Experiments 4.3 (F35-F37) | Transferred + Evidence | Expanded with before/during/after structure and concrete numbers. |
| "Iteration 1→2 proves regression lesson + rollback meaningful" | Writing plan S4 | Experiments 4.4 P1 (F38) | Transferred + Evidence | Added regression_lesson storage as the mechanism contribution. |
| "Strict audit causes deadlock — proves over-conservative audit suppresses exploration" | Writing plan S4 | Experiments 4.4 P2-P3 (F39-F40) | Transferred + Reframed | Reframed as "design tension," not "system failure." |
| "Relaxed audit fixes the deadlock — motivates risk-budget mechanism" | Writing plan S4 | Experiments 4.5 (F41-F42) | Transferred + Scope-constrained | Presented as "motivates risk-budget" future work, not as "the fix." |
| "Stable-Eureka: population-style loop with oracle fitness_score" | U15 | Method 3.1 P2 (F21) | Transferred — compressed | 5-item difference list compressed to side-by-side text diagrams + contrast paragraph. |
| "Reward hacking cannot be diagnosed from scalar returns alone" | Design doc S3.5 | Method 3.3 P1 (F24) | Transferred | Key motivation for step-level trajectory recording and attribution. |
| "Every component must be separately logged for attribution" | Design doc S3.2 | Method 3.2 (F23), Method 3.3 P2 (F25) | Transferred | Technical requirement preserved. |
| "The LLM must not directly emit Python reward code" | Design doc S3.2 | Method 3.5 (F28) | Transferred | Core constraint preserved. |

## Forbidden Overclaims — Verification

| Overclaim | Status | Where Blocked |
|---|---|---|
| "EG-RSA automatically discovers optimal rewards" | Blocked | Replaced by "EG-RSA searches reward schemas using structured experience and risk-aware editing" in Introduction P5, Discussion P1. |
| "EG-RSA outperforms EUREKA" | Blocked | No performance comparison claimed. Experiments framed as mechanism verification. |
| "Fully automated reward design" | Blocked | Task metric disclaimer in Method 3.3, Discussion P2 limitation. |
| "Solves reward hacking" | Blocked | Replaced by "reduces reward hacking risk through behavior risk audit and rollback" in Method 3.6. |
| "Generalizes to all RL environments" | Blocked | Discussion P2 limitation: single-environment. Conclusion: "demonstrated on LunarLander-v3." |

## Audit Verdict

**PASS** — All confirmed-motivation claims are transferred. All forbidden overclaims are blocked. The honest finding (audit deadlock) is preserved and reframed. No claim disappears without a documented decision.
