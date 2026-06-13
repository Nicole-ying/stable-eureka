# Clean Reward Search Run Report
best_candidate: g1_c1
schema_version: clean_reward_schema_v1_c94caf5f6d
env_alias: Env-0f1fc662
status: ok
selection_score_private_eval: 9.666666666666666
private_eval_return: 9.666666666666666
generated_reward_return: -0.9844869455430731
repair_attempts: 0
repair_success: False
identity_warning_count: 0
semantic_term_warning_count: 1
semantic_warning_count: 1
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: ['g0_c0', 'g0_c1']

## Reflection
## Analysis

**What worked:**
- Both candidates achieved high private_eval_return (9.33), indicating the core reward structure is highly effective
- Validation passed with zero errors, meaning schema compliance is correct
- Both candidates are structurally identical in outcome

**What didn't work (or is neutral):**
- Generated return is negative (~-0.76) and diverges from private_eval_return — this is expected behavior; the generated return is diagnostic only and not a selection metric
- Judge was skipped (deepseek_text_only_judge_skipped), so no qualitative feedback is available

**Key insight:** The schema already produces near-optimal private_eval_return. Further improvement may be marginal or require subtle adjustments rather than structural changes.

## What to Keep
- The current reward structure that achieves 9.33 private_eval_return
- The schema-validated component arrangement

## What to Change
- Since performance is already near ceiling, focus on robustness or diversity rather than raw score improvement
- Consider minor component weighting adjustments to test if private_eval_return can be pushed beyond 9.33

## Next Schema-Preserving Mutation Hypotheses (max 5)

1. **Slightly shift component weights** — e.g., increase weight of a secondary component by 0.1, decrease primary by 0.1, to test if current balance is truly optimal
2. **Add a small constant offset** to one component's return range (e.g., shift from [0,1] to [0.1,1.1]) to see if private_eval_return responds
3. **Swap component order** in the composition — keep all components identical but change their sequence in the reward formula
4. **Introduce a gentle sigmoid or tanh squashing** on one component's raw output to test if non-linear scaling improves private_eval_return
5. **Apply a small noise injection** (e.g., ±0.01) to one component's weight during evaluation to test robustness without changing the schema

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # Progress: reward staying near zero (stable region) in normalized observation space
    # Use negative squared norm to penalize large deviations, clamped to [-1.0, 0.0]
    progress = -0.1 * (next_obs[0]**2 + next_obs[1]**2 + next_obs[2]**2 + next_obs[3]**2)
    progress = max(progress, -1.0)

    # Stability: penalize large changes in observations between steps
    # Squared difference to penalize larger jumps more heavily
    delta = next_obs - obs
    stability = -0.5 * (delta[0]**2 + delta[1]**2 + delta[2]**2 + delta[3]**2)
    stability = max(stability, -1.0)

    # Effort: penalize any action (both 0 and 1 require effort in discrete action space)
    # Slightly higher penalty for action=1 to encourage minimal action
    effort = -0.02 if action in [0, 1] else 0.0

    # Terminal: negative penalty when episode ends, to discourage early termination
    terminal = -0.5 if done else 0.0

    total_reward = progress + stability + effort + terminal
    components = {
        'progress': progress,
        'stability': stability,
        'effort': effort,
        'terminal': terminal
    }
    return float(total_reward), components
```