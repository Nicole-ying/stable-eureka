## Environment Understanding Report

### Task Goal
The lander must land safely on the landing pad at coordinates (0,0) as fast as possible while minimizing fuel usage. The environment is a classic rocket trajectory optimization problem with discrete actions (engine on/off). The landing pad is fixed at (0,0) and fuel is infinite.

### Observations (8-dimensional state vector)
1. **x_coord**: Normalized x position (range approximately [-1, 1])
2. **y_coord**: Normalized y position (relative to helipad)
3. **x_velocity**: Normalized linear velocity in x direction
4. **y_velocity**: Normalized linear velocity in y direction
5. **angle**: Lander's rotation angle
6. **angular_velocity**: Scaled angular velocity
7. **left_leg_contact**: Boolean (1.0 if left leg touches ground)
8. **right_leg_contact**: Boolean (1.0 if right leg touches ground)

### Actions (Discrete: 0-3)
- **0**: Do nothing
- **1**: Fire left orientation engine
- **2**: Fire main engine (downward thrust)
- **3**: Fire right orientation engine

### Termination Conditions
1. Lander crashes (body contacts moon surface)
2. Lander exits viewport (|x_coord| >= 1.0)
3. Lander is not awake (physics simulation deactivation)

### Reward Function Visible Inputs
The reward function can directly access:
- `state` (8-dimensional observation vector)
- `action` (discrete 0-3)
- The variables `m_power` and `s_power` are computed inside step() based on action
- `terminated` flag

### Inferable Public Quantities
From observation and action, we can infer:
- **m_power**: 1.0 if action==2 (main engine), 0.0 otherwise
- **s_power**: 1.0 if action==1 or action==3 (side engines), 0.0 otherwise
- **ground_contact**: Both legs contacting ground (state[6]==1.0 and state[7]==1.0)
- **Distance to pad**: Computed from x,y coordinates (state[0], state[1])
- **Velocity magnitude**: From x_vel and y_vel (state[2], state[3])
- **Angular deviation**: From angle (state[4]) and angular velocity (state[5])

### Risks
- The reward function must handle both continuous and discrete action modes (but input boundary restricts to discrete based on task description)
- Wind and turbulence are applied during flight (when no ground contact), affecting dynamics unpredictably
- Engine dispersion adds randomness to thrust application
- The "not awake" termination is physics-engine dependent and may occur unexpectedly
- Normalized coordinates require careful interpretation for distance calculations
- The helipad_y constant and LEG_DOWN/SCALE factors affect the y-coordinate normalization

```json
{
  "task_goal": "Land the rocket safely on the landing pad at (0,0) as fast as possible while minimizing fuel usage, with discrete engine controls (on/off)",
  "observations": {
    "x_coord": "Normalized x position (range ~[-1, 1])",
    "y_coord": "Normalized y position relative to helipad",
    "x_velocity": "Normalized x velocity",
    "y_velocity": "Normalized y velocity",
    "angle": "Lander rotation angle",
    "angular_velocity": "Scaled angular velocity",
    "left_leg_contact": "Boolean: 1.0 if left leg touches ground",
    "right_leg_contact": "Boolean: 1.0 if right leg touches ground"
  },
  "actions": {
    "0": "Do nothing",
    "1": "Fire left orientation engine",
    "2": "Fire main engine (downward thrust)",
    "3": "Fire right orientation engine"
  },
  "termination": {
    "crash": "Lander body contacts moon surface",
    "out_of_bounds": "|x_coord| >= 1.0",
    "not_awake": "Lander physics simulation deactivated"
  },
  "public_reward_design_variables": [
    "state (8-dim observation vector)",
    "action (0-3 discrete)",
    "m_power (inferable: 1.0 if action==2 else 0.0)",
    "s_power (inferable: 1.0 if action==1 or action==3 else 0.0)",
    "terminated (boolean flag)"
  ],
  "risks": [
    "Wind and turbulence affect flight dynamics unpredictably when not in ground contact",
    "Engine dispersion adds randomness to thrust application",
    "Not awake termination is physics-engine dependent and may occur unexpectedly",
    "Normalized coordinates require careful interpretation for distance calculations",
    "Helipad_y constant and LEG_DOWN/SCALE factors affect y-coordinate normalization"
  ]
}
```