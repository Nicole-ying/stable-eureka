# Clean Reward Search Run Report
best_candidate: g1_c1
schema_version: clean_reward_schema_v1_b3cacbe4c2
env_alias: Env-90b964d9
status: ok
selection_score_private_eval: -129.33081069673662
private_eval_return: -129.33081069673662
generated_reward_return: -1.2175238451125552
repair_attempts: 0
repair_success: False
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: ['g0_c0', 'g0_c1']

## Reflection
## Summary of Candidate Performance

**What to Keep:**
- Both candidates passed schema validation (no validation errors), confirming correct structural alignment with the reward schema.

**What to Change:**
- **g0_c0** achieved a private_eval_return of **-153.85** with a generated_return of **-1.65** — the negative private score and very negative generated return indicate the reward signal is poorly aligned with desired behavior.
- **g0_c1** achieved a worse private_eval_return of **-618.57** despite a positive generated_return of **+37.91** — this large discrepancy suggests the reward function produces high raw output that does not correspond to useful behavioral shaping, likely due to a misspecified or overly permissive component logic.
- Both candidates show that the current component behavior fails to produce a reward signal that correlates with good private evaluation outcomes. The generated return is not a reliable proxy for selection score.

## Next Schema-Preserving Mutation Hypotheses (max 5)

1. **Adjust reward scaling/normalization**: Introduce a scaling factor or clipping mechanism within the component to constrain the generated return range (e.g., clamp to [-10, 10]) to prevent large positive outputs that mislead selection.

2. **Modify component logic to penalize extreme deviations**: Add a penalty term that activates when the generated return exceeds a threshold (e.g., > 5.0), encouraging the reward to stay within a bounded, meaningful range.

3. **Invert reward sign**: If the private evaluation shows negative correlation with generated return, flip the sign of the reward component to align positive generated return with positive private evaluation.

4. **Add a baseline subtraction**: Subtract a moving average or fixed baseline from the generated return to center the reward around zero, reducing the impact of constant offsets.

5. **Introduce a sparsity penalty**: Apply a small negative constant reward when the generated return is near zero or when component activity is low, encouraging the agent to produce non-trivial behavior.

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # Extract relevant observations
    x, y, vx, vy, theta, omega, left_contact, right_contact = obs
    next_x, next_y, next_vx, next_vy, next_theta, next_omega, next_left, next_right = next_obs
    
    # Progress: dense reward for forward movement in x direction, bounded [-1, 1]
    dx = next_x - x
    progress = np.clip(dx * 2.0, -1.0, 1.0)
    
    # Stability: penalize large angular deviation and angular velocity, bounded [-1, 0]
    angle_penalty = -abs(next_theta) / np.pi
    omega_penalty = -abs(next_omega) / 10.0
    stability = np.clip(angle_penalty + omega_penalty, -1.0, 0.0)
    
    # Effort: penalize non-zero actions and action switching, bounded [-1, 0]
    action_penalty = -0.1 if action != 0 else 0.0
    switch_penalty = -0.05  # approximate penalty for switching
    effort = np.clip(action_penalty + switch_penalty, -1.0, 0.0)
    
    # Terminal: reward for successful termination, penalty for failure
    terminal = 0.0
    if done:
        if next_x > 1.0:  # proxy for success
            terminal = 1.0
        else:
            terminal = -0.5
    
    # Weighted combination with scaling to keep total bounded
    total_reward = 0.4 * progress + 0.25 * stability + 0.15 * effort + 0.2 * terminal
    total_reward = np.clip(total_reward, -1.0, 1.0)
    
    components = {
        'progress': float(progress),
        'stability': float(stability),
        'effort': float(effort),
        'terminal': float(terminal)
    }
    
    return float(total_reward), components
```