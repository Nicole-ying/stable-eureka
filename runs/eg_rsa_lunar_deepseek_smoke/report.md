# EG-RSA Reward Search Run Report
best_candidate: g0_c0
schema_version: eg_rsa_reward_schema_v1_3e115df94a
env_alias: Env-90b964d9
status: ok
selection_score_private_eval: -504.7499547087998
private_eval_return: -504.7499547087998
generated_reward_return: -447.7428453907206
repair_attempts: 1
repair_success: True
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: []

## Reflection / Feedback Context
No prior generation feedback.

## Diagnostics
```json
{
  "generated_private_gap": 57.00710931807919,
  "action_mean": 2.0,
  "action_std": 0.0,
  "episode_length_mean": 76.0,
  "component_returns": {
    "terminal": 0.0,
    "velocity_penalty": -318.5556957264909,
    "angle_penalty": -14.773061126077057,
    "fuel_efficiency": -38.0,
    "distance_progress": -76.41408853815267
  }
}
```

## Prompt paths
```json
{
  "reward_coder": {
    "system": "runs/eg_rsa_lunar_deepseek_smoke/llm/generation_0/g0_c0/reward_coder/system.txt",
    "user": "runs/eg_rsa_lunar_deepseek_smoke/llm/generation_0/g0_c0/reward_coder/user.txt",
    "response": "runs/eg_rsa_lunar_deepseek_smoke/llm/generation_0/g0_c0/reward_coder/response.txt"
  },
  "repair_1": {
    "system": "runs/eg_rsa_lunar_deepseek_smoke/llm/generation_0/g0_c0/repair_1/system.txt",
    "user": "runs/eg_rsa_lunar_deepseek_smoke/llm/generation_0/g0_c0/repair_1/user.txt",
    "response": "runs/eg_rsa_lunar_deepseek_smoke/llm/generation_0/g0_c0/repair_1/response.txt"
  }
}
```

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # Unpack observations - only first 8 elements
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_obs[:8]
    
    # Extract engine usage from action
    main_engine = 1.0 if action == 2 else 0.0
    side_engine = 1.0 if action in [1, 3] else 0.0
    
    # Terminal component: safe landing at pad
    at_pad_x = abs(x) < 0.1
    at_pad_y = abs(y) < 0.1
    both_legs_grounded = (left_leg > 0.5) and (right_leg > 0.5)
    low_speed = (abs(vx) < 0.1) and (abs(vy) < 0.1)
    stable_angle = abs(angle) < 0.1
    landing_conditions = at_pad_x and at_pad_y and both_legs_grounded and low_speed and stable_angle
    
    terminal_reward = 100.0 if landing_conditions else 0.0
    
    # Velocity penalty: penalize high speeds, especially vertical
    vel_penalty = -1.5 * (vx**2 + vy**2)
    
    # Angle penalty: penalize tilt and spin
    angle_penalty = -2.0 * (angle**2 + angular_vel**2)
    
    # Fuel efficiency: penalize engine usage
    fuel_penalty = -0.5 * main_engine - 0.2 * side_engine
    
    # Distance progress: guide toward pad (x=0, y=0)
    dist_to_pad = (x**2 + y**2)**0.5
    distance_penalty = -0.5 * dist_to_pad
    
    # Additional shaping: encourage being close to pad with legs down
    near_pad = dist_to_pad < 0.3
    legs_bonus = min(0.5 * (left_leg + right_leg), 1.0) if near_pad else 0.0
    
    # Combine all components
    total_reward = terminal_reward + vel_penalty + angle_penalty + fuel_penalty + distance_penalty + legs_bonus
    
    # Clamp total_reward to bounded range
    total_reward = max(min(total_reward, 1000.0), -1000.0)
    
    components = {
        'terminal': terminal_reward,
        'velocity_penalty': vel_penalty,
        'angle_penalty': angle_penalty,
        'fuel_efficiency': fuel_penalty,
        'distance_progress': distance_penalty
    }
    
    return float(total_reward), components

```