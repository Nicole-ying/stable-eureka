## Environment Understanding Report

### Task Goal
The lander must land safely on the landing pad at coordinates (0,0) as fast as possible while minimizing fuel consumption. The lander starts at the top center with random initial force. The optimal policy is to use discrete engine controls (on/off).

### Observations (8-dimensional vector)
1. **x-coordinate**: Normalized position (-1 to 1, where 0 is center)
2. **y-coordinate**: Normalized position (relative to helipad)
3. **x-velocity**: Normalized linear velocity in x direction
4. **y-velocity**: Normalized linear velocity in y direction
5. **angle**: Lander's rotation angle
6. **angular velocity**: Scaled angular velocity
7. **left leg contact**: Boolean (1.0 if ground contact, else 0.0)
8. **right leg contact**: Boolean (1.0 if ground contact, else 0.0)

### Actions (Discrete)
- **0**: Do nothing
- **1**: Fire left orientation engine (side engine)
- **2**: Fire main engine (downward thrust)
- **3**: Fire right orientation engine (side engine)

### Termination Conditions
1. **Crash**: Lander body contacts the moon surface
2. **Out of bounds**: |x-coordinate| >= 1.0 (outside viewport)
3. **Not awake**: Lander enters sleep state

### Reward Function Visible Inputs
The reward function receives: `obs`, `action`, `next_obs`, `done`, `info`
- `obs`/`next_obs`: The 8-dimensional state vector
- `action`: Discrete action (0-3)
- `done`: Boolean termination flag
- `info`: Dictionary (may contain additional information)

### Inferable Public Quantities
- **Main engine power (m_power)**: 1.0 when action=2, 0.0 otherwise
- **Side engine power (s_power)**: 1.0 when action=1 or 3, 0.0 otherwise
- **Ground contact**: From obs[6] and obs[7] (leg contact booleans)
- **Velocity magnitude**: Computable from obs[2] and obs[3]
- **Distance to pad**: Computable from obs[0] and obs[1] (target is 0,0)
- **Angle deviation**: From obs[4] (ideal angle is 0 for landing)

### Risks
1. **Wind and turbulence**: Random forces applied when not in ground contact (not directly observable in state)
2. **Engine dispersion**: Random impulse variations (±1/SCALE) affect actual thrust direction
3. **Infinite fuel**: No fuel constraints, but fuel efficiency is part of the goal
4. **The state is normalized**: Raw physics values are scaled, so reward design must account for normalized ranges
5. **Termination on crash**: Any body contact with ground ends episode, not just leg contact
6. **Side engine torque**: Position of side engine thrust depends on lander orientation, creating orientation-dependent torque

```json
{
  "task_goal": "Land the rocket safely on the landing pad at coordinates (0,0) as fast as possible with minimal fuel consumption",
  "observations": {
    "dimension": 8,
    "elements": [
      "x_coordinate_normalized",
      "y_coordinate_normalized",
      "x_velocity_normalized",
      "y_velocity_normalized",
      "angle",
      "angular_velocity_scaled",
      "left_leg_contact_boolean",
      "right_leg_contact_boolean"
    ]
  },
  "actions": {
    "type": "discrete",
    "space_size": 4,
    "mapping": {
      "0": "do_nothing",
      "1": "fire_left_orientation_engine",
      "2": "fire_main_engine",
      "3": "fire_right_orientation_engine"
    }
  },
  "termination": {
    "conditions": [
      "lander_crashes_body_contacts_moon",
      "x_coordinate_outside_viewport_abs_greater_than_1",
      "lander_not_awake"
    ]
  },
  "reward_function_visible_inputs": [
    "obs_8_dim_state_vector",
    "action_0_to_3",
    "next_obs_8_dim_state_vector",
    "done_boolean",
    "info_dict"
  ],
  "inferable_public_quantities": {
    "main_engine_power": "1.0_when_action_2_else_0.0",
    "side_engine_power": "1.0_when_action_1_or_3_else_0.0",
    "ground_contact": "from_obs_index_6_and_7",
    "velocity_magnitude": "sqrt(obs[2]^2_+_obs[3]^2)",
    "distance_to_pad": "sqrt(obs[0]^2_+_obs[1]^2)",
    "angle_deviation": "abs(obs[4])"
  },
  "risks": [
    "Wind_and_turbulence_random_forces_not_directly_observable",
    "Engine_dispersion_causes_random_thrust_variations",
    "Infinite_fuel_but_fuel_efficiency_is_reward_goal",
    "Normalized_state_requires_careful_reward_scaling",
    "Crash_termination_on_any_body_contact_not_just_legs",
    "Side_engine_torque_depends_on_lander_orientation"
  ]
}
```