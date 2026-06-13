# Clean Reward Search Run Report
best_candidate: g1_c0
schema_version: clean_reward_schema_v1_b3cacbe4c2
env_alias: Env-90b964d9
status: ok
selection_score_private_eval: -75.15144160396606
private_eval_return: -75.15144160396606
generated_reward_return: -2.604872540269187
repair_attempts: 0
repair_success: False
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: ['g0_c0', 'g0_c1']

## Reflection
Based solely on the provided data:

**1) What to keep**
- The schema and validation structure are intact (no validation errors in either candidate).
- The ability to generate positive raw returns (g0_c1 generated_return=0.788) suggests the candidate can sometimes produce favorable outcomes.

**2) What to change**
- Both candidates have very negative private_eval_return values (-95.16 and -131.41), indicating the generated reward signal does not align with the true objective.
- The large gap between generated_return and private_eval_return in g0_c1 (0.79 vs -131.41) suggests the reward calculation is optimizing the wrong quantity or is extremely sensitive to some component.
- The generated_return in g0_c0 is negative (-0.514), which is also poor but less extreme than the private eval — suggesting a consistent misalignment.

**3) Next schema-preserving mutation hypotheses (max 5)**

1. **Scale normalization hypothesis**: The generated return may be on a very different scale than the private eval. Mutate the reward aggregation to explicitly normalize or clip raw returns before combining components.

2. **Component weighting hypothesis**: The current component weights may overemphasize a behavior that is harmful in private evaluation. Try shifting weight away from the component that correlates with high generated_return but low private_eval_return.

3. **Terminal condition sensitivity hypothesis**: The private_eval_return suggests catastrophic outcomes. Mutate the reward function to heavily penalize states or transitions that precede very negative private evaluations, perhaps by adding a safety margin or early termination penalty.

4. **Sign-flip correction hypothesis**: The negative correlation between generated_return and private_eval_return in g0_c1 suggests the reward might be inverted for one component. Mutate by flipping the sign of the component most correlated with the discrepancy.

5. **Reward clipping hypothesis**: The extreme negative private_eval_return may come from unbounded negative components. Mutate to add a lower bound (clipping) to each component's contribution to prevent runaway negative values.

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # Progress: encourage moving towards origin (negative distance change)
    dist = np.sqrt(obs[0]**2 + obs[1]**2)
    next_dist = np.sqrt(next_obs[0]**2 + next_obs[1]**2)
    progress = (dist - next_dist) / 2.5  # normalize by max position range
    progress = np.clip(progress, -1.0, 1.0)
    
    # Stability: penalize large linear and angular velocities
    speed = np.sqrt(obs[2]**2 + obs[3]**2)
    ang_vel_mag = abs(obs[5])
    stability = -0.05 * speed - 0.02 * ang_vel_mag
    stability = np.clip(stability, -1.0, 0.0)
    
    # Effort: penalize action magnitude (0-3 normalized)
    effort = -0.1 * (action / 3.0)
    effort = np.clip(effort, -0.5, 0.0)
    
    # Terminal: success/failure based on done and success flag (obs[6])
    terminal = 0.0
    if done:
        success_flag = next_obs[6] if len(next_obs) > 6 else 0.0
        if success_flag > 0.5:
            terminal = 2.0
        else:
            terminal = -2.0
    
    total_reward = 1.0 * progress + 1.0 * stability + 1.0 * effort + 1.0 * terminal
    total_reward = np.clip(total_reward, -5.0, 5.0)
    
    components = {
        'progress': float(progress),
        'stability': float(stability),
        'effort': float(effort),
        'terminal': float(terminal)
    }
    return float(total_reward), components
```