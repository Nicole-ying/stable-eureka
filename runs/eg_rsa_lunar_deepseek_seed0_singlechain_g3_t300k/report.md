# EG-RSA Reward Search Run Report
best_candidate: g0_c0
schema_version: eg_rsa_reward_schema_v1_05ce8b7470
env_alias: Env-90b964d9
status: ok
selection_score_private_eval: -107.31378535713051
private_eval_return: -107.31378535713051
generated_reward_return: -100.9865377924642
repair_attempts: 0
repair_success: False
judge_score: 0.0
judge_reason: video_render_skipped
parents: []

## Reflection / Feedback Context
No prior generation feedback.

## Diagnostics
```json
{
  "generated_private_gap": 6.327247564666308,
  "action_mean": 2.0144300144300145,
  "action_std": 1.0063691110475135,
  "episode_length_mean": 69.3,
  "component_returns": {
    "distance_penalty": -66.8241432580518,
    "velocity_penalty": -72.71724562313257,
    "angle_penalty": -1.44514891127983,
    "engine_usage_penalty": 0.0,
    "terminal": 40.0
  },
  "n_envs": 16,
  "vec_env_type": "dummy",
  "n_steps": 1024,
  "batch_size": 64,
  "n_epochs": 4,
  "gae_lambda": 0.98,
  "gamma": 0.999,
  "ent_coef": 0.01
}
```

## Prompt paths
```json
{
  "reward_coder": {
    "system": "runs/eg_rsa_lunar_deepseek_seed0_singlechain_g3_t300k/llm/generation_0/g0_c0/reward_coder/system.txt",
    "user": "runs/eg_rsa_lunar_deepseek_seed0_singlechain_g3_t300k/llm/generation_0/g0_c0/reward_coder/user.txt",
    "response": "runs/eg_rsa_lunar_deepseek_seed0_singlechain_g3_t300k/llm/generation_0/g0_c0/reward_coder/response.txt"
  }
}
```

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # Extract relevant quantities from next_obs
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # Distance penalty: encourage moving toward the pad at (0,0)
    distance = np.sqrt(x**2 + y**2)
    distance_penalty = -distance
    
    # Velocity penalty: encourage gentle landing speed
    velocity = np.sqrt(vx**2 + vy**2)
    velocity_penalty = -velocity
    
    # Angle penalty: encourage upright orientation
    angle_penalty = -abs(angle)
    
    # Engine usage penalty: discourage fuel consumption
    # Main engine (action==2) costs fuel
    engine_usage_penalty = -0.3 if action == 2 else 0.0
    
    # Terminal reward: successful landing
    # Conditions: both legs on ground, close to pad, upright, low vertical speed
    landing_condition = (
        left_contact > 0.5 and 
        right_contact > 0.5 and 
        distance < 0.1 and 
        abs(angle) < 0.1 and 
        abs(vy) < 0.1
    )
    terminal = 100.0 if landing_condition else 0.0
    
    # Sum up all components
    total_reward = distance_penalty + velocity_penalty + angle_penalty + engine_usage_penalty + terminal
    
    # Build components dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "engine_usage_penalty": engine_usage_penalty,
        "terminal": terminal,
    }
    
    return float(total_reward), components

```