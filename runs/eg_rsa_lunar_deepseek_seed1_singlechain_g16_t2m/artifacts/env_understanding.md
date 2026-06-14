## Environment Understanding Report

This is a Lunar Lander environment where the agent must learn to land a spacecraft on a designated landing pad. The environment features discrete actions (engine on/off) and continuous physics simulation.

### Task Goal
Land the spacecraft on the landing pad at coordinates (0,0) as fast as possible while minimizing fuel consumption. The lander starts at the top center with random initial forces.

### Observation Space (8-dimensional vector)
1. **x-coordinate**: Normalized horizontal position (range approximately [-1, 1])
2. **y-coordinate**: Normalized vertical position relative to landing pad
3. **x-velocity**: Normalized horizontal velocity
4. **y-velocity**: Normalized vertical velocity
5. **angle**: Lander orientation angle
6. **angular velocity**: Scaled angular velocity
7. **left leg contact**: Boolean (1.0 if left leg touches ground)
8. **right leg contact**: Boolean (1.0 if right leg touches ground)

### Action Space (Discrete, 4 actions)
- **0**: Do nothing
- **1**: Fire left orientation engine
- **2**: Fire main engine (downward thrust)
- **3**: Fire right orientation engine

### Termination Conditions
1. **Crash**: Lander body contacts the moon surface
2. **Out of bounds**: x-coordinate exceeds 1.0 (leaves viewport)
3. **Not awake**: Lander enters sleep state

### Reward Function Visible Inputs
The reward function receives only: `obs`, `action`, `next_obs`, `done`, `info`

### Inferable Public Quantities
From the observation vector, we can compute:
- **Distance to pad**: Euclidean distance from (obs[0], obs[1]) to (0,0)
- **Speed**: sqrt(obs[2]² + obs[3]²)
- **Angle deviation**: absolute value of obs[4]
- **Ground contact**: obs[6] and obs[7] indicate leg contact
- **Fuel usage**: Can be inferred from action == 2 (main engine firing)
- **Landing stability**: Combination of low speed, small angle, and ground contact

### Risks
- The reward function in step.py uses internal variables (`m_power`, `s_power`) that are NOT available as direct inputs to the reward function
- The `terminated` flag in step.py may differ from the `done` flag passed to the reward function
- Wind and turbulence effects are applied during the step but not directly observable in the state
- Fuel is infinite, but fuel minimization is still a goal - this must be inferred from action patterns
- The reward function must work with normalized observations

```json
{
    "task_goal": "Land the spacecraft on the landing pad at coordinates (0,0) as fast as possible while minimizing fuel consumption",
    "observations": [
        "x-coordinate (normalized, range ~[-1, 1])",
        "y-coordinate (normalized, relative to landing pad)",
        "x-velocity (normalized)",
        "y-velocity (normalized)",
        "angle (radians)",
        "angular velocity (scaled)",
        "left leg ground contact (boolean 0/1)",
        "right leg ground contact (boolean 0/1)"
    ],
    "actions": [
        "0: do nothing",
        "1: fire left orientation engine",
        "2: fire main engine",
        "3: fire right orientation engine"
    ],
    "termination": [
        "Lander crashes (body contacts moon)",
        "x-coordinate exceeds 1.0 (out of viewport)",
        "Lander is not awake"
    ],
    "reward_function_visible_inputs": [
        "obs (8-dimensional observation vector)",
        "action (discrete 0-3)",
        "next_obs (8-dimensional observation vector)",
        "done (boolean termination flag)",
        "info (dictionary, may contain additional data)"
    ],
    "inferable_public_quantities": [
        "distance to landing pad from obs[0], obs[1]",
        "linear speed from obs[2], obs[3]",
        "angular deviation from obs[4]",
        "ground contact status from obs[6], obs[7]",
        "fuel usage from action==2 frequency",
        "landing stability from combined metrics"
    ],
    "risks": [
        "Internal variables m_power and s_power used in step.py reward calculation are NOT available to the reward function",
        "Wind and turbulence effects are not directly observable in state",
        "Fuel is infinite but minimization is still a goal - must be inferred from actions",
        "terminated flag in step.py may differ from done flag passed to reward function",
        "Observations are normalized, so raw physics quantities must be computed carefully"
    ]
}
```