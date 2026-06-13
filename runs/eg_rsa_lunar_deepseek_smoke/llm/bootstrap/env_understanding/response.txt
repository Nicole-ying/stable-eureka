## Environment Understanding Report

### Task Goal
The lander must land safely on the landing pad at coordinates (0,0) as quickly as possible while minimizing fuel consumption. The lander starts at the top center of the viewport with random initial force.

### Observations (8-dimensional vector)
1. **x-coordinate**: Normalized horizontal position of lander center
2. **y-coordinate**: Normalized vertical position relative to helipad
3. **x-velocity**: Normalized horizontal velocity
4. **y-velocity**: Normalized vertical velocity
5. **angle**: Current angle of the lander
6. **angular velocity**: Scaled angular velocity
7. **left leg contact**: Boolean (1.0 if left leg touches ground)
8. **right leg contact**: Boolean (1.0 if right leg touches ground)

### Actions (Discrete, 4 options)
- **0**: Do nothing
- **1**: Fire left orientation engine
- **2**: Fire main engine (full throttle)
- **3**: Fire right orientation engine

### Termination Conditions
1. Lander crashes (body contacts moon surface)
2. Lander goes outside viewport (|x-coordinate| >= 1.0)
3. Lander is not awake (physics simulation stops)

### Reward Function Visible Inputs
The reward function receives only: `state`, `m_power`, `s_power`, `terminated`

Where:
- `state` = 8-dimensional observation vector described above
- `m_power` = main engine power (0.0 or 1.0 for discrete)
- `s_power` = side engine power (0.0 or 1.0 for discrete)
- `terminated` = boolean termination flag

### Inferable Public Quantities
- Fuel usage can be inferred from `m_power` and `s_power` values
- Landing success can be inferred from leg contact booleans and position near (0,0)
- Speed/velocity components are directly observable
- Angle and angular velocity are directly observable

### Risks
1. **Hardware-specific constants**: The reward uses internal physics constants (SCALE, FPS, engine locations) that may not be directly accessible
2. **Wind/turbulence**: Applied randomly when not in ground contact, affecting trajectory unpredictably
3. **Engine dispersion**: Random noise in engine impulse direction (±1/SCALE) makes precise control challenging
4. **Normalized coordinates**: The state is normalized differently from raw physics coordinates, requiring careful interpretation

```json
{
  "task_goal": "Land the rocket safely on the landing pad at coordinates (0,0) as quickly as possible while minimizing fuel consumption",
  "observations": "8-dimensional vector: [x_position, y_position, x_velocity, y_velocity, angle, angular_velocity, left_leg_contact, right_leg_contact]",
  "actions": "Discrete 0-3: 0=do nothing, 1=fire left orientation engine, 2=fire main engine, 3=fire right orientation engine",
  "termination": "1) Lander crashes (body contacts moon), 2) |x-coordinate| >= 1.0 (outside viewport), 3) Lander not awake",
  "public_reward_design_variables": "state (8-dim observation), m_power (main engine power 0.0 or 1.0), s_power (side engine power 0.0 or 1.0), terminated (boolean)",
  "risks": "1) Internal physics constants not directly accessible, 2) Random wind/turbulence when airborne, 3) Random engine dispersion (±1/SCALE), 4) Normalized state coordinates differ from raw physics coordinates"
}
```