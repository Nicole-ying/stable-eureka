# Clean Reward Search Run Report
best_candidate: g0_c0
schema_version: clean_reward_schema_v1_b3cacbe4c2
env_alias: Env-90b964d9
status: ok
selection_score_private_eval: -128.02190511668834
private_eval_return: -128.02190511668834
generated_reward_return: 125.49983986603911
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
    # progress: encourage forward movement using x-position and x-velocity
    x_pos = obs[0]
    next_x_pos = next_obs[0]
    x_vel = obs[2]
    next_x_vel = next_obs[2]
    pos_change = next_x_pos - x_pos
    vel_change = next_x_vel - x_vel
    progress = pos_change + 0.1 * vel_change
    progress = np.clip(progress, -1.0, 1.0)  # bounded

    # stability: penalize large angle and angular velocity
    angle = obs[4]  # angle (radians)
    ang_vel = obs[5]  # angular velocity
    stability = 1.0 - 0.5 * (abs(angle) + 0.1 * abs(ang_vel))
    stability = np.clip(stability, -1.0, 1.0)

    # effort: penalize non-zero actions (action 0 is "do nothing")
    effort = 1.0 if action == 0 else -0.5
    effort = np.clip(effort, -1.0, 1.0)

    # terminal: reward success, penalize failure
    if done:
        terminal = 1.0 if info.get('success', False) else -1.0
    else:
        terminal = 0.0
    terminal = np.clip(terminal, -1.0, 1.0)

    total_reward = progress + stability + effort + terminal
    components = {
        'progress': progress,
        'stability': stability,
        'effort': effort,
        'terminal': terminal
    }
    return float(total_reward), components
```