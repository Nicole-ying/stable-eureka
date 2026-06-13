# Clean Reward Search Run Report
best_candidate: g0_c0
schema_version: clean_reward_schema_v1_e18c317ff9
env_alias: Env-0f1fc662
status: ok
selection_score_private_eval: 500.0
private_eval_return: 500.0
generated_reward_return: -11.417387286999382
repair_attempts: 0
repair_success: False
judge_score: 0.0
judge_reason: mock_judge_no_vision
parents: []

## Reflection
No prior clean candidates. Start with bounded progress, stability, effort, and terminal components. Avoid environment-specific assumptions.

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    obs_arr = np.asarray(obs, dtype=float).reshape(-1)
    next_arr = np.asarray(next_obs, dtype=float).reshape(-1)
    delta = next_arr - obs_arr
    progress = float(np.clip(np.linalg.norm(obs_arr) - np.linalg.norm(next_arr), -5.0, 5.0))
    stability = float(-0.05 * np.tanh(np.linalg.norm(delta)))
    act_arr = np.asarray(action, dtype=float).reshape(-1)
    effort = float(-0.01 * np.tanh(np.linalg.norm(act_arr)))
    terminal = float(-1.0 if done else 0.0)
    total = progress + stability + effort + terminal
    components = {
        'progress': progress,
        'stability': stability,
        'effort': effort,
        'terminal': terminal,
    }
    return float(total), components
```