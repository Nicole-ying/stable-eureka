## Environment Understanding Report

This is a 2D rocket landing simulation (Lunar Lander variant) where the agent must land a spacecraft safely on a designated landing pad at coordinates (0,0). The environment features discrete actions (on/off engine control) but also supports continuous mode. The lander starts at the top center with random initial force, and the goal is to land quickly with minimal fuel consumption. Wind and turbulence can optionally affect the lander when not in ground contact.

### Observations (8-dimensional vector):
1. **x position**: Normalized horizontal position (0 = center, -1/+1 = edges of viewport)
2. **y position**: Normalized vertical position (0 = helipad level, -1 = bottom, +1 = top)
3. **x velocity**: Normalized horizontal velocity
4. **y velocity**: Normalized vertical velocity
5. **angle**: Lander's rotation angle in radians
6. **angular velocity**: Normalized rotational velocity
7. **left leg contact**: Boolean (0 or 1) indicating left leg ground contact
8. **right leg contact**: Boolean (0 or 1) indicating right leg ground contact

### Actions (discrete, 4 options):
- **0**: Do nothing (no engine firing)
- **1**: Fire left orientation engine (rotates lander)
- **2**: Fire main engine (primary thrust downward)
- **3**: Fire right orientation engine (rotates lander opposite direction)

### Termination Conditions:
1. **Crash**: Lander body contacts the moon surface (game_over flag)
2. **Out of bounds**: x coordinate exceeds viewport bounds (|state[0]| >= 1.0)
3. **Not awake**: Lander enters sleep state (physics engine inactivity)

### Public Reward Design Variables:
- **state**: 8-dimensional observation vector available in compute_reward
- **m_power**: Main engine throttle (0.0 or 1.0 for discrete, 0.5-1.0 for continuous)
- **s_power**: Side engine throttle (0.0 or 1.0 for discrete, 0.5-1.0 for continuous)
- **terminated**: Boolean flag indicating episode end
- **individual_reward**: Dictionary for storing reward components
- **fitness_score**: Available via compute_fitness_score function

### Risks & Considerations:
- **Landing pad at (0,0)**: The target is fixed at the center, but the observation normalization makes this a coordinate transformation challenge
- **Infinite fuel**: Agent can hover indefinitely, but efficiency is still important for reward shaping
- **Wind/turbulence**: External forces applied when not in ground contact, creating unpredictable dynamics
- **Engine dispersion**: Random noise in thrust application requires robust control
- **Contact booleans**: Only indicate leg contact, not stable landing - a lander could tip over
- **Crash detection**: Any body contact with ground terminates episode, not just leg contact
- **Side engine asymmetry**: The side engine impulse position calculation uses different constants (17 vs SIDE_ENGINE_HEIGHT=14) creating orientation-dependent torque

```json
{
  "task_goal": "Land the rocket safely on the landing pad at coordinates (0,0) as quickly as possible with minimal fuel consumption, starting from a random initial position at the top center of the viewport",
  "observations": {
    "x_position": "Normalized horizontal position (-1 to 1, where 0 is center)",
    "y_position": "Normalized vertical position (0 at helipad level, -1 at bottom, +1 at top)",
    "x_velocity": "Normalized horizontal velocity",
    "y_velocity": "Normalized vertical velocity",
    "angle": "Lander rotation angle in radians",
    "angular_velocity": "Normalized rotational velocity (scaled by 20.0/FPS)",
    "left_leg_contact": "Boolean (0 or 1) indicating left leg ground contact",
    "right_leg_contact": "Boolean (0 or 1) indicating right leg ground contact"
  },
  "actions": {
    "0": "Do nothing - no engine firing",
    "1": "Fire left orientation engine - rotates lander clockwise",
    "2": "Fire main engine - primary downward thrust",
    "3": "Fire right orientation engine - rotates lander counterclockwise"
  },
  "termination": {
    "crash": "Lander body contacts moon surface (game_over flag set)",
    "out_of_bounds": "|normalized_x_position| >= 1.0 (lander exits viewport horizontally)",
    "not_awake": "Lander enters physics sleep state"
  },
  "public_reward_design_variables": [
    "state - 8-dimensional observation vector",
    "m_power - main engine throttle level (0.0 or 1.0 for discrete)",
    "s_power - side engine throttle level (0.0 or 1.0 for discrete)",
    "terminated - boolean episode termination flag",
    "individual_reward - dictionary for storing reward components",
    "fitness_score - available via compute_fitness_score function"
  ],
  "risks": [
    "Landing pad is at (0,0) but observations are normalized differently for x and y axes",
    "Infinite fuel means agent can hover indefinitely, requiring careful shaping for efficiency",
    "Wind and turbulence apply random forces when not in ground contact, creating unpredictable dynamics",
    "Engine thrust has random dispersion component, requiring robust control strategies",
    "Contact booleans only indicate leg contact, not stable landing - lander can tip over after landing",
    "Crash detection triggers on any body contact with ground, making gentle touchdowns critical",
    "Side engine impulse position uses inconsistent constants (17 vs 14) creating orientation-dependent torque",
    "State normalization factors differ between position, velocity, and angular velocity components"
  ]
}
```