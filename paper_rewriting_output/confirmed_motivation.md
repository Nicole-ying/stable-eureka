# Confirmed Motivation

## Exact Confirmed Motivation

> LLM-assisted reward design should be structured as **experience-guided reward schema search** — not one-shot code generation — so that past editing outcomes (both successes and failures) inform future edits.

EG-RSA reformulates automated reward design as a sequential, history-aware, risk-audited reward search process. The key distinction from prior work is structural: existing methods treat reward design as code generation with iterative refinement; EG-RSA treats it as schema-based search with semantic role attribution, outcome lesson memory, operator-constrained editing, and integrated risk audit with rollback.

## User Confirmation

- **Status:** Confirmed (accepted recommended Option A)
- **Date:** 2026-06-10
- **Confirmation mode:** User reviewed all 4 options and directed to proceed with Option A

## Rejected Options

| Option | Reason Rejected |
|---|---|
| B (Risk-auditable editing) | Too narrow; audit is a mechanism, not the controlling contribution. Best as supporting mechanism within A. |
| C (Diagnosis-driven search) | Attribution is strong but works better as an enabler of search, not the headline. |
| D (Attribution only) | Too narrow for a journal paper; risk of reviewer pushback on contribution size. |

## Scope Limits

1. **Contribution count: 3, not 6+.** (1) Reward search framing with structured schema + memory; (2) semantic role attribution for diagnosis-driven editing; (3) operator-constrained editing with risk audit and rollback.
2. **Experiment framing: mechanism verification, not benchmark.** The 10-iteration LunarLander experiment verifies that each mechanism functions as designed — not that EG-RSA outperforms EUREKA on a benchmark.
3. **No oracle leakage.** EG-RSA uses post-hoc oracle evaluation only; the oracle reward is never used during search, editing, or selection.
4. **Task metrics are template-instantiated.** Do not claim fully automatic metric discovery; the LLM assists variable mapping within templates.
5. **Audit deadlock is an insight, not a failure.** The strict-audit → deadlock → relaxed-audit path reveals a real design tension (safety vs. exploration) in reward search.

## Forbidden Overclaims

- "EG-RSA automatically discovers optimal rewards" → Use: "EG-RSA searches reward schemas using structured experience and risk-aware editing."
- "EG-RSA outperforms EUREKA" → Use: "We verify EG-RSA's mechanisms through iterative case studies on LunarLander."
- "Fully automated reward design" → Use: "LLM-assisted reward search with human-specified task metric templates."
- "Solves reward hacking" → Use: "Reduces reward hacking risk through behavior risk audit and rollback."
- "Generalizes to all RL environments" → Use: "Demonstrated on LunarLander-v3 continuous control."
