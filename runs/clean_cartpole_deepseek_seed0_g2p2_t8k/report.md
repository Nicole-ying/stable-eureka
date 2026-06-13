# Clean Reward Search Run Report
best_candidate: g0_c0
schema_version: clean_reward_schema_v1_e18c317ff9
env_alias: Env-0f1fc662
status: ok
selection_score_private_eval: 500.0
private_eval_return: 500.0
generated_reward_return: 368.0974654844575
repair_attempts: 0
repair_success: False
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: []

## Reflection
No prior clean candidates. Start with bounded progress, stability, effort, and terminal components. Avoid environment-specific assumptions.

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # obs = [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    # next_obs = [next_cart_pos, next_cart_vel, next_pole_angle, next_pole_ang_vel]
    
    # Progress: reward for keeping pole angle near zero (upright)
    pole_angle = obs[2]
    angle_abs = abs(pole_angle)
    progress = 1.0 - min(angle_abs / 0.418879, 1.0)  # normalize to [0,1]
    
    # Stability: penalty for large angular velocity changes
    pole_ang_vel = obs[3]
    next_pole_ang_vel = next_obs[3]
    ang_vel_change = abs(next_pole_ang_vel - pole_ang_vel)
    stability = 1.0 - min(ang_vel_change / 2.0, 1.0)  # bounded [0,1]
    
    # Effort: penalty for switching actions (action 0 or 1)
    # We don't have previous action, so penalize based on action magnitude (always 0 or 1)
    effort = 1.0 - 0.5 * action  # action 0 -> 1.0, action 1 -> 0.5
    
    # Terminal: small survival bonus, penalty on failure
    if done:
        terminal = -1.0  # penalty for termination
    else:
        terminal = 0.1   # small survival bonus
    
    total_reward = 0.4 * progress + 0.2 * stability + 0.2 * effort + 0.2 * terminal
    components = {
        'progress': progress,
        'stability': stability,
        'effort': effort,
        'terminal': terminal
    }
    return float(total_reward), components
```