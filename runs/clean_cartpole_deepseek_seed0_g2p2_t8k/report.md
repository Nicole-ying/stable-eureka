# Clean Reward Search Run Report
best_candidate: g0_c0
schema_version: clean_reward_schema_v1_c94caf5f6d
env_alias: Env-0f1fc662
status: ok
selection_score_private_eval: 9.666666666666666
private_eval_return: 9.666666666666666
generated_reward_return: -2.3351234694549117
repair_attempts: 0
repair_success: False
identity_warning_count: 0
semantic_term_warning_count: 0
semantic_warning_count: 0
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: []

## Reflection
No prior clean candidates. Use only anonymous normalized feature indices. Prefer bounded functions of feature values, feature-change norms, action cost, and terminal signal. Do not assign meanings to feature dimensions.

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # progress: encourage staying in a region where obs features are near zero
    # using negative squared norm as a dense progress signal, bounded via tanh
    progress = -np.tanh(np.sum(obs**2) / 4.0)  # range roughly [-1, 0]

    # stability: penalize large changes between consecutive observations
    diff = np.linalg.norm(next_obs - obs)
    stability = -np.tanh(diff)  # range roughly [-1, 0]

    # effort: penalize action magnitude; actions are {0,1}, so penalize action=1
    effort = -0.5 * action  # range [-0.5, 0]

    # terminal: bounded penalty when episode ends (done)
    terminal = -1.0 if done else 0.0  # range [-1, 0]

    total_reward = progress + stability + effort + terminal
    components = {
        'progress': progress,
        'stability': stability,
        'effort': effort,
        'terminal': terminal
    }
    return float(total_reward), components
```