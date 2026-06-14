# LunarLander Reward Search — Environment Understanding

## Task
Design reward functions for LunarLander-v3 that train a PPO agent to land successfully
between the flags at the landing pad (x≈0, y≈0).

## Observation Space (8-dim vector, all normalized)
```
obs[0] = x_pos            # horizontal position, range [-2.5, 2.5], 0 = center of pad
obs[1] = y_pos            # vertical position, range [-2.5, 2.5], 0 = ground level
obs[2] = x_vel            # horizontal velocity, range [-10, 10]
obs[3] = y_vel            # vertical velocity, range [-10, 10], negative = moving down
obs[4] = angle            # lander tilt in radians, 0 = upright
obs[5] = ang_vel          # angular velocity, range [-10, 10]
obs[6] = left_leg_contact # 0 or 1 (boolean), 1 = leg touching ground
obs[7] = right_leg_contact# 0 or 1 (boolean), 1 = leg touching ground
```

## Action Space (4 discrete actions)
```
0 = do nothing (no engine)
1 = fire left orientation engine
2 = fire main engine (thrust downward)
3 = fire right orientation engine
```

## Goal
Land the lunar lander between the flags at the center of the pad (x ∈ [-0.2, 0.2]).
Successful landing requires:
- Both legs in contact with ground
- Close to pad center
- Low landing velocity
- Upright orientation

## Reward Function Interface
```python
def compute_reward(obs, action, next_obs, done, info) -> tuple[float, dict]:
    """
    obs:       np.ndarray (8,) — observation BEFORE action
    action:    int (0-3) — the action taken
    next_obs:  np.ndarray (8,) — observation AFTER action
    done:      bool — whether episode terminated
    info:      dict — additional info (empty for our purposes)

    Returns: (total_reward: float, components: dict[str, float])
    """
```

## What We CANNOT See
The official environment reward is COMPLETELY HIDDEN. We only see whether the
agent landed or crashed through the behavior metrics (landing_rate, final_distance,
episode_length). The absolute score we optimize for is `hidden_return_mean` — but
we never see how it's computed.

## What We CAN Observe (Metrics from training)
- `hidden_return_mean`: the official (hidden) environment score
- `generated_return_mean`: total of our reward function
- `generated_hidden_gap`: difference between the two
- `landing_rate`: fraction of eval episodes ending in successful landing
- `crash_rate`: fraction of eval episodes ending in crash
- `episode_length_mean`: average steps per episode
- `final_distance_mean`: average distance from pad at episode end
- `action_distribution`: which actions the agent uses
- `component_returns`: per-component breakdown of our reward

## Training Budget
- V1 Phase: 30k timesteps per candidate (quick screening)
- V2+ Phase: longer training for promising candidates
