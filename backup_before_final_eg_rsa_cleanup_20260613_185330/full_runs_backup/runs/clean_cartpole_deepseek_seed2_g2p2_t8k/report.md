# Clean Reward Search Run Report
best_candidate: g1_c1
schema_version: clean_reward_schema_v1_e18c317ff9
env_alias: Env-0f1fc662
status: ok
selection_score_private_eval: 448.6666666666667
private_eval_return: 448.6666666666667
generated_reward_return: 1351.8000661570388
repair_attempts: 0
repair_success: False
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: ['g0_c0', 'g0_c1']

## Reflection
## Analysis of Past Clean Candidates

**g0_c0** (selection_score=284.67, private_eval_return=284.67)
- **Why it worked**: High private return suggests component behavior that consistently produced schema-aligned reward signals across many steps. The generated return (791.11) being much higher than private return indicates the reward function produced large positive values that were then normalized/clipped by the private evaluation, but the underlying behavior was robust enough to retain high private scores.
- **Validation**: No errors, schema-compliant.

**g0_c1** (selection_score=9.67, private_eval_return=9.67)
- **Why it failed**: Very low private return. Generated return (-0.038) is near zero or slightly negative, indicating the reward function rarely or never triggered positive reward signals. The component likely failed to detect or reward the desired behavior under the schema.
- **Validation**: No errors, but functionally ineffective.

## What to Keep
- The general structure and validation pattern from g0_c0.
- Reward scaling/normalization approach that produced high private returns.

## What to Change
- Avoid reward functions that produce near-zero or negative generated returns (like g0_c1).
- Ensure reward components have sufficient sensitivity to trigger positive signals frequently.

## Next Schema-Preserving Mutation Hypotheses

1. **Adjust reward threshold**: Lower the activation threshold so the reward fires more frequently while maintaining high private return.
2. **Modify reward scaling**: Increase the multiplicative factor on positive signals to amplify small correct behaviors.
3. **Add temporal smoothing**: Apply exponential moving average to reward to reduce sparsity and stabilize signals.
4. **Change reward shape**: Use a sigmoid or clipped linear function instead of binary/sparse reward to provide graded feedback.
5. **Invert sign logic**: If negative reward is accidentally being emitted, flip the condition to reward the correct state instead.

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # Extract state components from next_obs
    x = next_obs[0]
    theta = next_obs[2]
    # Extract previous angle for stability
    theta_prev = obs[2]
    x_prev = obs[0]
    
    # Progress: reward staying near center and upright
    # Normalize angle by max allowed 0.418879 rad, position by max allowed 4.8
    angle_frac = abs(theta) / 0.418879
    pos_frac = abs(x) / 4.8
    progress = 1.0 - min(1.0, 0.5 * angle_frac + 0.3 * pos_frac)
    
    # Stability: penalize large changes in angle and position
    delta_theta = abs(theta - theta_prev) / 0.418879
    delta_x = abs(x - x_prev) / 4.8
    stability = 1.0 - min(1.0, delta_theta + 0.5 * delta_x)
    
    # Effort: penalize action taken (action is 0 or 1)
    effort = 1.0 - 0.1 * float(action)
    
    # Terminal: penalty if done, small survival bonus otherwise
    if done:
        terminal = -0.5
    else:
        terminal = 0.1
    
    total_reward = progress + stability + effort + terminal
    components = {
        'progress': progress,
        'stability': stability,
        'effort': effort,
        'terminal': terminal
    }
    return float(total_reward), components
```