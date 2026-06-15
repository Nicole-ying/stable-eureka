def compute_reward(self, state, m_power, s_power, terminated):
    # Extract state variables
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = state[4]
    angular_vel = state[5]
    leg1 = state[6]
    leg2 = state[7]

    # Initialize diagnostic variables
    distance_to_pad = 0.0
    vertical_velocity = 0.0
    angle_val = 0.0
    leg_contact_both = 0.0
    crash_or_out_of_bounds = 0.0
    hovering = 0.0
    objective_success_bonus = 0.0
    crash_penalty = 0.0
    distance_shaping = 0.0
    descending_progress_bonus = 0.0
    velocity_regularizer = 0.0
    angle_regularizer = 0.0
    fuel_penalty = 0.0
    hovering_penalty = 0.0
    stable_idle_bonus = 0.0

    # Compute distance to pad
    distance_to_pad = (x**2 + y**2)**0.5
    vertical_velocity = abs(vy)
    angle_val = abs(angle)
    leg_contact_both = 1.0 if (leg1 > 0.5 and leg2 > 0.5) else 0.0

    # Objective success bonus: both legs on ground, near pad, low velocity, upright
    if leg_contact_both > 0.5 and abs(x) < 0.1 and abs(y) < 0.1 and abs(vy) < 0.1 and abs(angle) < 0.1:
        objective_success_bonus = 100.0

    # Crash or out-of-bounds penalty
    if terminated and objective_success_bonus == 0.0:
        crash_or_out_of_bounds = 1.0
        crash_penalty = -100.0

    # Distance shaping (reduced weight)
    distance_shaping = -1.0 * distance_to_pad

    # Descending progress bonus: reward for reducing y (moving downward)
    if not hasattr(self, 'last_y'):
        self.last_y = y
    y_change = self.last_y - y
    if y_change > 0:
        descending_progress_bonus = 1.0 * y_change
    self.last_y = y

    # Velocity regularizer
    velocity_regularizer = -0.1 * vertical_velocity

    # Angle regularizer
    angle_regularizer = -0.05 * angle_val

    # Fuel penalty
    fuel_penalty = -0.01 * (m_power + s_power)

    # Hovering penalty: low speed, no ground contact, not terminated
    speed = (vx**2 + vy**2)**0.5
    if speed < 0.1 and leg_contact_both < 0.5 and not terminated:
        hovering = 1.0
        hovering_penalty = -0.1

    # Stable idle bonus: reward for using action 0 when stable
    # We need to know the action taken. Since we don't have it directly, we infer from m_power and s_power.
    # Action 0: m_power=0, s_power=0. Action 1: m_power=0, s_power>0. Action 2: m_power>0, s_power=0. Action 3: m_power=0, s_power>0 (but opposite direction).
    # In LunarLander-v2, action 0 is do nothing: m_power=0, s_power=0.
    if m_power == 0 and s_power == 0 and speed < 0.1 and angle_val < 0.1 and leg_contact_both < 0.5 and not terminated:
        stable_idle_bonus = 0.05

    # Total reward
    reward = objective_success_bonus + crash_penalty + distance_shaping + descending_progress_bonus + velocity_regularizer + angle_regularizer + fuel_penalty + hovering_penalty + stable_idle_bonus

    # Individual reward dictionary
    individual_reward = {
        'objective_success_bonus': objective_success_bonus,
        'crash_penalty': crash_penalty,
        'distance_shaping': distance_shaping,
        'descending_progress_bonus': descending_progress_bonus,
        'velocity_regularizer': velocity_regularizer,
        'angle_regularizer': angle_regularizer,
        'fuel_penalty': fuel_penalty,
        'hovering_penalty': hovering_penalty,
        'stable_idle_bonus': stable_idle_bonus,
        'distance_to_pad': distance_to_pad,
        'vertical_velocity': vertical_velocity,
        'angle': angle_val,
        'leg_contact_both': leg_contact_both,
        'crash_or_out_of_bounds': crash_or_out_of_bounds,
        'hovering': hovering
    }

    return float(reward), individual_reward