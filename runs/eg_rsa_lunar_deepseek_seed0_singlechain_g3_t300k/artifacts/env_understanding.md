## Environment Understanding Report

### Task Goal
The lander must land on the landing pad at coordinates (0,0) as fast as possible while minimizing fuel usage. The landing pad is located between two flags. The lander starts at the top center of the viewport with random initial forces.

### Observations (8-dimensional vector)
1. **x-coordinate**: Normalized horizontal position (range ~[-1, 1], termination at |x|>=1)
2. **y-coordinate**: Normalized vertical position relative to helipad
3. **x-velocity**: Normalized horizontal velocity
4. **y-velocity**: Normalized vertical velocity
5. **angle**: Lander's rotation angle (radians)
6. **angular velocity**: Scaled angular velocity
7. **left leg contact**: Boolean (1.0 if ground contact, else 0.0)
8. **right leg contact**: Boolean (1.0 if ground contact, else 0.0)

### Actions (Discrete: 0-3)
- **0**: Do nothing
- **1**: Fire left orientation engine (side engine)
- **2**: Fire main engine (downward thrust)
- **3**: Fire right orientation engine (side engine)

### Termination Conditions
1. Lander crashes (body contacts moon surface)
2. Lander goes outside viewport (|x-coordinate| >= 1.0)
3. Lander is not awake (physics body sleeping)

### Public Reward Design Variables
- `m_power`: Main engine power (0.0 when off, 1.0 when firing)
- `s_power`: Side engine power (0.0 when off, 1.0 when firing)
- `state`: The 8-dimensional observation vector
- `terminated`: Boolean termination flag
- `action`: The discrete action taken (0-3)

### Risks and Considerations
1. **Wind and turbulence**: Random forces applied to lander when not in ground contact, making hovering/landing challenging
2. **Engine dispersion**: Random noise (±1/SCALE) applied to engine impulse positions
3. **Coordinate system**: Observations are normalized, raw positions need denormalization for distance calculations
4. **Fuel is infinite**: No fuel constraint, but minimizing fuel usage is part of the goal
5. **Landing pad at (0,0)**: After normalization, the pad center corresponds to specific normalized coordinates
6. **Contact booleans**: Both legs must likely contact ground for successful landing
7. **Side engine asymmetry**: Side engine impulse position calculation has a known bug/artifact with orientation-dependent torque

```json
{
  "task_goal": "Land the rocket on the landing pad at coordinates (0,0) as fast as possible with minimal fuel usage",
  "observations": {
    "x": "Normalized horizontal position [-1, 1]",
    "y": "Normalized vertical position relative to helipad",
    "x_velocity": "Normalized horizontal velocity",
    "y_velocity": "Normalized vertical velocity",
    "angle": "Lander rotation angle",
    "angular_velocity": "Scaled angular velocity",
    "left_leg_contact": "Boolean, 1 if left leg touches ground",
    "right_leg_contact": "Boolean, 1 if right leg touches ground"
  },
  "actions": {
    "0": "Do nothing",
    "1": "Fire left orientation engine",
    "2": "Fire main engine",
    "3": "Fire right orientation engine"
  },
  "termination": [
    "Lander crashes (body contacts moon)",
    "|x-coordinate| >= 1.0 (outside viewport)",
    "Lander is not awake"
  ],
  "public_reward_design_variables": [
    "state (8-dim observation vector)",
    "m_power (main engine power, 0.0 or 1.0)",
    "s_power (side engine power, 0.0 or 1.0)",
    "terminated (boolean flag)",
    "action (discrete action 0-3)"
  ],
  "risks": [
    "Wind and turbulence forces applied when not in ground contact",
    "Engine impulse dispersion adds randomness to thrust direction",
    "Observations are normalized; raw positions require denormalization for accurate distance calculations",
    "No fuel constraint despite goal to minimize fuel usage",
    "Side engine impulse position has orientation-dependent torque artifact",
    "Successful landing requires both legs in ground contact at pad position"
  ]
}
```