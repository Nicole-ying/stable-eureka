# EG-RSA Reward Search Run Report
best_candidate: g0_c0
schema_version: eg_rsa_reward_schema_v1_f401f4f018
env_alias: Env-90b964d9
status: ok
selection_score_private_eval: -480.5731370508223
private_eval_return: -480.5731370508223
generated_reward_return: -87.21102223114299
repair_attempts: 0
repair_success: False
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: []

## Reflection / Feedback Context
No prior generation feedback.

## Diagnostics
```json
{
  "generated_private_gap": 393.36211481967933,
  "action_mean": 3.0,
  "action_std": 0.0,
  "episode_length_mean": 56.0,
  "component_returns": {
    "distance_shaping": -24.92944849968054,
    "velocity_penalty": -20.249119434981573,
    "angle_penalty": -32.032454296480864,
    "fuel_efficiency": 0.0,
    "ground_contact_bonus": 0.0,
    "terminal": -10.0
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
  }
}
```

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # Extract observations
    x, y = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    left_leg = obs[6]
    right_leg = obs[7]

    # Extract next_obs for terminal check
    nx, ny = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    nleft_leg = next_obs[6]
    nright_leg = next_obs[7]

    # 1. Distance shaping: encourage moving closer to pad (0,0)
    distance = np.sqrt(x**2 + y**2)
    distance_shaping = -0.5 * distance  # linear penalty, scaled modestly

    # 2. Velocity penalty: penalize high speed, especially vertical
    speed = np.sqrt(x_vel**2 + y_vel**2)
    vel_penalty = -0.3 * speed - 0.8 * max(y_vel, 0.0)  # extra penalty for upward velocity

    # 3. Angle penalty: penalize deviation from upright
    angle_penalty = -0.4 * abs(angle) - 0.1 * abs(ang_vel)

    # 4. Fuel efficiency: penalize main engine use (action == 2)
    fuel_penalty = -0.2 if action == 2 else 0.0

    # 5. Ground contact bonus: reward when both legs touch
    ground_bonus = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0

    # 6. Terminal component: large reward for success, large penalty for failure
    terminal = 0.0
    if done:
        # Check if landing was successful: both legs contact, near pad, low velocity
        both_legs = (nleft_leg > 0.5 and nright_leg > 0.5)
        near_pad = np.sqrt(nx**2 + ny**2) < 0.15
        low_speed = np.sqrt(nx_vel**2 + ny_vel**2) < 0.1
        good_angle = abs(angle) < 0.1
        if both_legs and near_pad and low_speed and good_angle:
            terminal = 20.0  # successful landing
        else:
            terminal = -10.0  # crash or out-of-bounds

    # Sum all components
    total_reward = distance_shaping + vel_penalty + angle_penalty + fuel_penalty + ground_bonus + terminal

    # Clip to reasonable bounds
    total_reward = max(min(total_reward, 100.0), -100.0)

    components = {
        "distance_shaping": distance_shaping,
        "velocity_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "fuel_efficiency": fuel_penalty,
        "ground_contact_bonus": ground_bonus,
        "terminal": terminal,
    }

    return float(total_reward), components

```