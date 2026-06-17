def compute_reward(self, state, m_power, s_power, terminated):
    # Extract state variables
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = state[4]
    angular_velocity = state[5]
    leg1_contact = state[6]
    leg2_contact = state[7]

    # Initialize diagnostic variables
    distance_to_pad = 0.0
    vertical_speed = 0.0
    horizontal_speed = 0.0
    angle_abs = 0.0
    angular_velocity_abs = 0.0
    leg_contact = 0.0
    m_power_used = 0.0
    s_power_used = 0.0
    crash_penalty = 0.0
    success_bonus = 0.0
    shaping_reward = 0.0
    fuel_penalty = 0.0
    orientation_penalty = 0.0
    leg_contact_bonus = 0.0

    # Compute diagnostics
    distance_to_pad = (x**2 + y**2) ** 0.5
    vertical_speed = abs(vy)
    horizontal_speed = abs(vx)
    angle_abs = abs(angle)
    angular_velocity_abs = abs(angular_velocity)
    leg_contact = leg1_contact + leg2_contact
    m_power_used = m_power
    s_power_used = s_power

    # Shaping reward: encourage reducing distance to pad and vertical speed
    shaping_reward = -distance_to_pad - vertical_speed

    # Fuel penalty
    fuel_penalty = -0.1 * m_power_used

    # Orientation penalty
    orientation_penalty = -0.1 * (angle_abs + angular_velocity_abs)

    # Leg contact bonus
    if leg1_contact == 1.0 and leg2_contact == 1.0:
        leg_contact_bonus = 10.0
    else:
        leg_contact_bonus = 0.0

    # Crash penalty
    if terminated:
        crash_penalty = -100.0
    else:
        crash_penalty = 0.0

    # Success bonus: both legs contact, upright, on pad, low speed
    if (leg1_contact == 1.0 and leg2_contact == 1.0 and
        abs(angle) < 0.1 and abs(x) < 0.1 and y < 0.1 and
        abs(vx) < 0.1 and abs(vy) < 0.1):
        success_bonus = 100.0
    else:
        success_bonus = 0.0

    # Total reward
    reward = shaping_reward + fuel_penalty + orientation_penalty + leg_contact_bonus + crash_penalty + success_bonus

    # Individual reward diagnostics (only schema component ids)
    individual_reward = {
        "success_bonus": success_bonus,
        "shaping_reward": shaping_reward,
        "fuel_penalty": fuel_penalty,
        "crash_penalty": crash_penalty,
        "orientation_penalty": orientation_penalty,
        "leg_contact_bonus": leg_contact_bonus
    }

    return float(reward), individual_reward