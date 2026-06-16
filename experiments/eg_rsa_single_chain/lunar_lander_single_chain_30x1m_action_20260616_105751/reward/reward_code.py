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
    angle_deviation = 0.0
    fuel_usage = 0.0
    success_flag = 0.0
    crash_flag = 0.0

    # Compute distance to pad (horizontal distance)
    distance_to_pad = abs(x)

    # Compute vertical speed magnitude
    vertical_speed = abs(vy)

    # Compute angle deviation from upright
    angle_deviation = abs(angle)

    # Compute fuel usage
    fuel_usage = m_power + s_power

    # Initialize reward components
    success_bonus = 0.0
    distance_penalty = 0.0
    vertical_speed_penalty = 0.0
    angle_penalty = 0.0
    fuel_penalty_main = 0.0
    fuel_penalty_side = 0.0
    crash_penalty = 0.0

    # Check for success: both legs contact, near pad, low velocities, upright
    both_legs_contact = (leg1_contact == 1.0 and leg2_contact == 1.0)
    near_pad = (abs(x) < 0.1)
    low_vx = (abs(vx) < 0.1)
    low_vy = (abs(vy) < 0.1)
    upright = (abs(angle) < 0.1)

    if both_legs_contact and near_pad and low_vx and low_vy and upright:
        success_bonus = 100.0
        success_flag = 1.0

    # Distance penalty (shaping)
    distance_penalty = -0.1 * distance_to_pad

    # Vertical speed penalty (shaping)
    vertical_speed_penalty = -0.1 * vertical_speed

    # Angle penalty (regularizer)
    angle_penalty = -0.02 * angle_deviation

    # Fuel penalties (regularizers)
    fuel_penalty_main = -0.03 * m_power
    fuel_penalty_side = -0.01 * s_power

    # Crash penalty (if terminated and not success)
    if terminated and success_flag == 0.0:
        crash_penalty = -100.0
        crash_flag = 1.0

    # Total reward
    reward = success_bonus + distance_penalty + vertical_speed_penalty + angle_penalty + fuel_penalty_main + fuel_penalty_side + crash_penalty

    # Individual reward diagnostics
    individual_reward = {
        'success_bonus': success_bonus,
        'distance_penalty': distance_penalty,
        'vertical_speed_penalty': vertical_speed_penalty,
        'angle_penalty': angle_penalty,
        'fuel_penalty_main': fuel_penalty_main,
        'fuel_penalty_side': fuel_penalty_side,
        'crash_penalty': crash_penalty,
        'distance_to_pad': distance_to_pad,
        'vertical_speed': vertical_speed,
        'angle_deviation': angle_deviation,
        'fuel_usage': fuel_usage,
        'success_flag': success_flag,
        'crash_flag': crash_flag
    }

    return float(reward), individual_reward