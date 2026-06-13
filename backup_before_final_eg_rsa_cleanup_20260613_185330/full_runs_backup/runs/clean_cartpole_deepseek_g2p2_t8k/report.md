# Clean Reward Search Run Report
best_candidate: g1_c0
schema_version: clean_reward_schema_v1_e18c317ff9
env_alias: Env-0f1fc662
status: ok
selection_score_private_eval: 9.666666666666666
private_eval_return: 9.666666666666666
generated_reward_return: -3.05724831594356
repair_attempts: 1
repair_success: True
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: []

## Reflection
No prior clean candidates. Start with bounded progress, stability, effort, and terminal components. Avoid environment-specific assumptions.

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # Progress: encourage upright position (angle close to zero)
    angle = obs[2]  # in [-0.418879, 0.418879]
    progress = -abs(angle) / 0.418879  # normalized to [-1, 0]

    # Stability: penalise abrupt angular changes
    angular_velocity = next_obs[2] - obs[2]
    stability = -min(1.0, abs(angular_velocity) / 0.418879)  # bounded in [-1, 0]

    # Effort: penalise any non-zero action (discrete 0 or 1)
    effort = -float(action)  # bounded in [-1, 0]

    # Terminal: penalise termination (assuming failure)
    terminal = -1.0 if done else 0.0

    total_reward = progress + stability + effort + terminal
    components = {
        "progress": progress,
        "stability": stability,
        "effort": effort,
        "terminal": terminal
    }
    return total_reward, components
```