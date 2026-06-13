# Clean Reward Search Run Report
best_candidate: g1_c0
schema_version: clean_reward_schema_v1_e18c317ff9
env_alias: Env-0f1fc662
status: ok
selection_score_private_eval: 428.0
private_eval_return: 428.0
generated_reward_return: 310.94878056417434
repair_attempts: 0
repair_success: False
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: ['g0_c0', 'g0_c1']

## Reflection
## Summary of Past Clean Candidates

### What to Keep
- Both candidates passed schema validation with no errors, indicating the reward structure is well-formed and compatible with the environment's expectations.
- The selection score (private_eval_return) is consistently higher than the generated_return, suggesting the reward signal is effectively amplifying desirable behavior beyond what the raw generation captures.
- The large gap between private evaluation and generated return (e.g., ~106 points for g0_c0) implies the reward logic is correctly identifying and rewarding latent positive behaviors not fully expressed in the generation.

### What to Change
- The private evaluation scores (361.67 and 312.33) show significant variance (~49 points). This suggests the reward function's sensitivity to certain behavioral dimensions may be unstable or overly dependent on specific generation characteristics.
- Both candidates had 0 repair attempts and no successful repairs, indicating the reward design may not have a mechanism to handle edge cases or correct for reward hacking patterns when they occur.
- The "deepseek_text_only_judge_skipped" reason suggests the judge component is not being utilized, potentially missing a valuable signal for distinguishing between candidates.

### Next Schema-Preserving Mutation Hypotheses

1. **Introduce a diversity bonus term** that rewards behavioral variation across generations, which could stabilize scores by preventing overfitting to narrow generation patterns.

2. **Add a consistency penalty** that compares reward output across similar generation conditions, reducing variance by penalizing unpredictable reward spikes.

3. **Implement a minimal repair trigger** based on generated_return dropping below a threshold relative to recent private_eval_return, enabling automatic correction of reward drift.

4. **Modulate reward sensitivity** by scaling the amplification factor between generated_return and private_eval_return, aiming for a more consistent gap ratio (e.g., target 1.3x-1.5x instead of the current ~1.4x and ~1.7x).

5. **Inject a judge feedback loop** that conditions reward scaling on a lightweight text coherence or instruction-following metric, replacing the skipped deepseek judge with a cheaper local proxy.

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # Progress: reward for being near upright and centered
    pos = next_obs[0]
    angle = next_obs[2]
    progress = np.exp(-0.5 * (pos**2 / 0.5**2 + angle**2 / 0.1**2))

    # Stability: penalize large pole angular velocity and cart velocity
    vel_penalty = abs(next_obs[1]) / 4.0
    ang_vel_penalty = abs(next_obs[3]) / 4.0
    stability = np.exp(-0.5 * (vel_penalty + ang_vel_penalty))

    # Effort: penalize action switching (approximate using action value)
    # Single action penalty: small cost for nonzero action
    effort = 1.0 - 0.1 * float(action)

    # Terminal: negative penalty when done (failure), zero otherwise
    terminal = -1.0 if done else 0.0

    total_reward = 0.4 * progress + 0.2 * stability + 0.2 * effort + 0.2 * terminal
    components = {
        'progress': float(progress),
        'stability': float(stability),
        'effort': float(effort),
        'terminal': float(terminal)
    }
    return float(total_reward), components
```