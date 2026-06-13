Here is the environment understanding report and the structured JSON output.

## Environment Understanding Report

The environment is a 2D rocket landing simulation (Lunar Lander variant). The agent controls a lander with discrete actions (0: do nothing, 1: fire left engine, 2: fire main engine, 3: fire right engine). The goal is to land softly and upright on a landing pad located at coordinates (0,0) as fast as possible while minimizing fuel usage. Fuel is infinite, but fuel efficiency is part of the reward.

The observation (state) is an 8-dimensional vector:
- x position (normalized to [-1, 1])
- y position (normalized)
- linear velocity in x (normalized)
- linear velocity in y (normalized)
- angle (radians)
- angular velocity (scaled)
- left leg ground contact (0 or 1)
- right leg ground contact (0 or 1)

Termination occurs when:
- The lander crashes (body touches ground)
- The lander moves outside the viewport (|x| >= 1.0)
- The lander is not "awake" (physics engine idle)

The reward function and fitness score are computed internally using the state, engine powers (m_power, s_power), and termination flag.

```json
{
  "task_goal": "Land the rocket softly and upright on the landing pad at coordinates (0,0) as quickly as possible while minimizing fuel consumption.",
  "observations": [
    "normalized_x_position",
    "normalized_y_position",
    "normalized_x_velocity",
    "normalized_y_velocity",
    "angle_radians",
    "scaled_angular_velocity",
    "left_leg_ground_contact_bool",
    "right_leg_ground_contact_bool"
  ],
  "actions": {
    "type": "Discrete(4)",
    "mapping": {
      "0": "do_nothing",
      "1": "fire_left_orientation_engine",
      "2": "fire_main_engine",
      "3": "fire_right_orientation_engine"
    }
  },
  "termination": [
    "lander_crashes (body contacts ground)",
    "lander_outside_viewport (|x| >= 1.0)",
    "lander_not_awake (physics engine idle)"
  ],
  "public_reward_design_variables": [
    "state (8-dim observation vector)",
    "m_power (main engine throttle, 0.0 or 1.0 for discrete)",
    "s_power (side engine throttle, 0.0 or 1.0 for discrete)",
    "terminated (boolean flag)"
  ],
  "risks": [
    "The reward function is computed internally and not exposed; any reward design must reconstruct or approximate it from the public variables.",
    "Wind and turbulence are applied randomly when legs are not in ground contact, adding stochasticity that may need to be accounted for in reward shaping.",
    "The 'not awake' termination can occur unexpectedly (e.g., after a collision), making the episode end without a clear crash signal.",
    "The state normalization uses viewport dimensions and FPS, so raw physics values are not directly available; all spatial reasoning must use the normalized state."
}
```