# Evidence Bank

## EG-RSA Mechanism Verification Evidence

### E1: Main Experiment — 10-Iteration Strict-Audit Run (EXP-01)

**Source:** `experiments/eg_rsa_lunar_lander_v1_role_attrib_10x1m/`
**Environment:** LunarLander-v3, PPO, 1M steps/iteration
**Config:** `configs/eg_rsa_deepseek_v1_role_attrib_10x1m.yml`

| Iteration | Task Score | Semantic Score | Hack Score | Failure Modes | Dominant Component |
|---|---|---|---|---|---|
| 0 | 0.478 | 0.478 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_approach_region |
| 1 | 0.867 | 0.950 | 0.00 | (none) | r_landing_quality |
| 2 | 0.412 | 0.412 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_approach_region |
| 3 | 0.416 | 0.416 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_landing_quality |
| 4 | 0.367 | 0.367 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_landing_quality |
| 5 | 0.631 | 0.631 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_approach_region |
| 6 | 0.477 | 0.477 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_landing_quality |
| 7 | 0.422 | 0.422 | 0.20 | shaping_goal_mismatch | r_landing_quality |
| 8 | 0.665 | 0.665 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_landing_quality |

**Key evidence claims supported:**

1. **Effective edit (Iter 0→1):** Task score doubled (0.478→0.867), failure modes eliminated, dominant component shifted from r_approach_region (dense guidance) to r_landing_quality (terminal success). Demonstrates that semantic role attribution + outcome memory can produce a structured edit that improves task performance.

2. **Regression (Iter 1→2):** Score dropped from 0.867→0.412, failure modes returned. Demonstrates that regression is possible and that the system must handle it (motivates rollback mechanism).

3. **Audit deadlock (Iter 2→8):** Scores remained low (0.367-0.665) for 6 iterations. Failure modes persisted. Under strict audit, medium-risk edits were repeatedly blocked, preventing escape from the low-success regime. Demonstrates the safety-exploration tradeoff.

### E2: Relaxed Audit — 4-Iteration Run (EXP-02)

**Source:** `experiments/eg_rsa_lunar_lander_v1_role_attrib_10x1m_audit_relaxed/`
**Environment:** LunarLander-v3, PPO, 1M steps/iteration

| Iteration | Task Score | Semantic Score | Hack Score | Failure Modes | Dominant Component |
|---|---|---|---|---|---|
| 0 | 0.478 | 0.478 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_approach_region |
| 1 | 0.511 | 0.511 | 0.40 | repeated_event_exploitation, shaping_goal_mismatch | r_approach_region |
| 2 | 2.490 | 3.907 | 0.00 | (none) | r_landing_quality |
| 3 | 2.953 | 4.453 | 0.00 | (none) | r_landing_quality |

**Key evidence claims supported:**

4. **Deadlock escape:** Relaxing medium-risk blocking enabled a breakthrough edit (Iter 1→2: 0.511→2.490). Demonstrates that the audit deadlock observed in EXP-01 is caused by the audit policy, not by a fundamental inability of the search process.

5. **Sustained improvement:** Iter 2→3 continued to improve (2.490→2.953). Suggests that once the deadlock is broken, the search process can continue productively.

### E3: Attribution Evidence

**Source:** Iteration diagnostic reports in EXP-01 and EXP-02

- **Dominant component tracking:** Each iteration records which reward component (role) dominates policy behavior, measured via per-step reward component ratios across all trajectories.
- **Role shift evidence (Iter 0→1, EXP-01):** Dominant component changed from r_approach_region (dense guidance, ratio ~0.42) to r_landing_quality (terminal success, ratio increased). This is direct evidence that the edit plan changed which reward component drives policy behavior.
- **Role stability in deadlock:** Iterations 2-8 show dominant component oscillating between r_approach_region and r_landing_quality without settling, consistent with repeated blocked edits.

### E4: Memory Evidence

**Source:** Iteration memory directories in EXP-01

- **Outcome lesson storage:** Each iteration stores structured lessons (schema_diff, metric_delta, failure_modes, rollback_decision) in `memory/` as JSONL.
- **Lesson retrieval:** `retrieved_lessons.json`, `retrieved_memory.json`, `retrieved_outcome_lessons.json` in each iteration directory demonstrate that prior lessons are retrieved and used during edit planning.
- **Regression lesson (Iter 1→2):** The drop from 0.867→0.412 triggered a regression lesson recording the schema change, metric delta, and failure modes.

### E5: Audit Evidence

**Source:** Behavior risk audit reports in EXP-01

- **High/medium/low risk classification:** Each edit plan receives a risk tier.
- **Medium-risk blocking:** Under strict audit, medium-risk edits under weak success evidence are blocked. This is the mechanism that caused the deadlock.
- **Repair attempts:** `repair_behavior_risk_audit.json` and `repair_response.json` show that edits were repaired and resubmitted after audit rejection, but continued to be blocked.

### E6: Implementation Evidence

**Source:** `eg_rsa/` codebase

- **Schema compiler** (`eg_rsa/reward/safe_compiler.py`): Schema → executable Python reward.
- **Hack detectors** (`eg_rsa/diagnostics/hack_detectors.py`, 28KB): Concrete detection rules for misalignment, contact exploitation, stability regression.
- **Attribution module** (`eg_rsa/diagnostics/attribution.py`): Per-component reward computation and dominant-role identification.
- **Operator constraints** (`eg_rsa/reward/operators.py`): Five constrained edit operators (increase_weight, decrease_weight, add_component, add_event_rule, disable_component).

### Evidence Not Available (Do Not Claim)

- Multi-environment benchmark results
- Comparison against EUREKA/Text2Reward/CARD on standard benchmarks
- Statistical significance tests across multiple seeds
- Human evaluation of reward quality
- Ablation study results (configs exist, no results yet)
