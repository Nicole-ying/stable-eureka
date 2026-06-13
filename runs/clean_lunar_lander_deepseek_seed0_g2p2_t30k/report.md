# Clean Reward Search Run Report
best_candidate: g1_c0
schema_version: clean_reward_schema_v1_633ab10d3a
env_alias: Env-90b964d9
status: ok
selection_score_private_eval: -109.32232639758654
private_eval_return: -109.32232639758654
generated_reward_return: 149.85691157755556
repair_attempts: 0
repair_success: False
identity_warning_count: 0
semantic_term_warning_count: 0
semantic_warning_count: 0
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: ['g0_c0', 'g0_c1']

## Reflection
Deterministic clean feedback from prior candidates.
Scores are only relative to observed candidates in this run; do not infer absolute optimality.
Use only anonymous normalized feature indices and schema component IDs.

Observed ranking by private evaluation return:
- rank=1, id=g0_c0, status=ok, private_eval_return=-133.652, generated_return=146.466, generated_minus_private=280.118, repair_attempts=0, repair_success=False, identity_warning_count=0, semantic_term_warning_count=0
- rank=2, id=g0_c1, status=ok, private_eval_return=-137.354, generated_return=-7.43243, generated_minus_private=129.921, repair_attempts=0, repair_success=False, identity_warning_count=0, semantic_term_warning_count=0

Schema-preserving mutation guidance:
1. Prefer candidates with valid schema, low warning counts, and better relative private evaluation.
2. If generated_return and private_eval_return diverge, adjust reward scaling, clipping, or component weights.
3. Use anonymous feature norms and feature-change norms instead of named dimensions.
4. Keep action-cost and terminal components bounded.
5. Explore small component-weight changes rather than assuming any candidate is optimal.

Do not use physical or benchmark-specific names for observation dimensions.

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # progress: encourage larger absolute feature values, bounded
    progress = np.clip(np.mean(np.abs(next_obs)), 0.0, 1.0)

    # stability: penalize abrupt changes, bounded in [0, 1]
    diff = np.linalg.norm(next_obs - obs)
    stability = 1.0 - np.tanh(diff / 2.0)

    # effort: penalize larger actions (0,1,2,3), bounded in [0, 1]
    effort = 1.0 - action / 3.0

    # terminal: small penalty for termination
    terminal = -0.1 if done else 0.0

    total_reward = progress + stability + effort + terminal
    components = {
        'progress': progress,
        'stability': stability,
        'effort': effort,
        'terminal': terminal,
    }
    return float(total_reward), components
```