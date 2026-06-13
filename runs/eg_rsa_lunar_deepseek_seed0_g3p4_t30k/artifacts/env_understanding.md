## Environment Understanding Report

This is a lunar lander environment with discrete actions (4 options) and continuous state observations. The goal is to land safely on a designated landing pad at coordinates (0,0) with minimal fuel usage and time.

### Key Observations
- The environment uses Box2D physics engine
- Wind and turbulence are applied when not in ground contact
- The lander has two legs with ground contact sensors
- Random initial force is applied to the lander's center of mass at start
- Fuel is infinite (not a constraint)

### State Construction
The 8-dimensional state vector is normalized:
- `state[0]`: Normalized x position (-1 to 1, where 0 is center)
- `state[1]`: Normalized y position (relative to helipad)
- `state[2]`: Normalized x velocity
- `state[3]`: Normalized y velocity
- `state[4]`: Raw angle (radians)
- `state[5]`: Normalized angular velocity
- `state[6]`: Left leg ground contact (0 or 1)
- `state[7]`: Right leg ground contact (0 or 1)

### Important Implementation Details
- Main engine power is always 1.0 when fired (discrete action 2)
- Side engines provide torque for orientation control
- Engine impulses include dispersion (randomness)
- The `m_power` and `s_power` variables track actual engine usage
- The reward function receives `state`, `m_power`, `s_power`, and `terminated` as inputs

```json
{
  "task_goal": "Land the lunar lander on the designated landing pad at coordinates (0,0) with minimal fuel consumption and minimal time, while maintaining stability and avoiding crashes",
  "observations": {
    "dimension": 8,
    "components": [
      {"index": 0, "name": "x_position", "range": [-1.0, 1.0], "description": "Normalized horizontal position (0=center)"},
      {"index": 1, "name": "y_position", "range": "normalized", "description": "Vertical position relative to helipad"},
      {"index": 2, "name": "x_velocity", "range": "normalized", "description": "Horizontal velocity"},
      {"index": 3, "name": "y_velocity", "range": "normalized", "description": "Vertical velocity"},
      {"index": 4, "name": "angle", "range": "radians", "description": "Lander orientation angle"},
      {"index": 5, "name": "angular_velocity", "range": "normalized", "description": "Rate of angle change"},
      {"index": 6, "name": "left_leg_contact", "range": [0.0, 1.0], "description": "Left leg ground contact indicator"},
      {"index": 7, "name": "right_leg_contact", "range": [0.0, 1.0], "description": "Right leg ground contact indicator"}
    ]
  },
  "actions": {
    "type": "discrete",
    "options": [
      {"value": 0, "name": "do_nothing", "description": "No engine firing"},
      {"value": 1, "name": "fire_left_engine", "description": "Fire left orientation engine"},
      {"value": 2, "name": "fire_main_engine", "description": "Fire main engine (full throttle)"},
      {"value": 3, "name": "fire_right_engine", "description": "Fire right orientation engine"}
    ]
  },
  "termination": {
    "conditions": [
      "Lander crashes (body contacts moon surface)",
      "Lander goes outside viewport (x coordinate > 1)",
      "Lander is not awake (physics engine sleeping)"
    ],
    "success_condition": "Not explicitly defined; landing at pad with both legs in contact is implied"
  },
  "public_reward_design_variables": [
    {"name": "state", "type": "8-element array", "source": "observation"},
    {"name": "m_power", "type": "float (0.0 or 1.0)", "description": "Main engine power usage"},
    {"name": "s_power", "type": "float (0.0 or 1.0)", "description": "Side engine power usage"},
    {"name": "terminated", "type": "boolean", "description": "Episode termination flag"}
  ],
  "risks": [
    "Reward function must handle both continuous and discrete action modes consistently",
    "The state normalization makes raw physics values inaccessible; rewards must use normalized values",
    "No explicit success signal in observations - must infer from leg contacts + position/velocity",
    "Wind and turbulence add stochastic elements that should be considered in reward design",
    "The angle observation is in radians while angular velocity is normalized differently",
    "Engine dispersion adds noise to actions that cannot be directly observed"
  ]
}
```