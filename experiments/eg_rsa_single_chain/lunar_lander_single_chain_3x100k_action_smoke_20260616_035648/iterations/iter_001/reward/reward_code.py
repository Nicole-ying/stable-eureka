def compute_reward(self, state, m_power, s_power, terminated):
    # Initialize diagnostics
    distance = 0.0
    vertical_speed = 0.0
    angle = 0.0
    main_fuel = 0.0
    orientation_fuel = 0.0
    success = 0.0
    crash_penalty = 0.0
    horizontal_correction_bonus = 0.0

    # Extract state components
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle_val = state[4]
    angular_vel = state[5]
    leg1 = state[6]
    leg2 = state[7]

    # Progress signals
    distance = -0.5 * (x**2 + y**2)**0.5
    vertical_speed = -0.3 * abs(vy)
    angle = -0.2 * abs(angle_val)

    # Regularizers
    main_fuel = -0.03 * m_power
    orientation_fuel = -0.01 * s_power

    # Action bonus for using right orientation engine when horizontal velocity is away from pad
    # vx * x > 0 means velocity is moving away from pad (positive x moving right when x>0, etc.)
    # This encourages using action 3 (right orientation engine) to correct drift toward pad
    if m_power == 0 and s_power > 0 and vx * x > 0:
        horizontal_correction_bonus = 0.5

    # Objective signal: successful landing
    both_legs = (leg1 > 0.5) and (leg2 > 0.5)
    near_pad = (abs(x) < 0.1) and (abs(y) < 0.1)
    low_speed = abs(vy) < 0.1
    upright = abs(angle_val) < 0.1
    if both_legs and near_pad and low_speed and upright and not terminated:
        success = 100.0

    # Terminal penalty for crash or out-of-bounds
    if terminated and not (both_legs and near_pad and low_speed and upright):
        crash_penalty = -100.0

    # Total reward
    reward = distance + vertical_speed + angle + main_fuel + orientation_fuel + horizontal_correction_bonus + success + crash_penalty

    # Diagnostics
    individual_reward = {
        'distance': distance,
        'vertical_speed': vertical_speed,
        'angle': angle,
        'main_fuel': main_fuel,
        'orientation_fuel': orientation_fuel,
        'horizontal_correction_bonus': horizontal_correction_bonus,
        'success': success,
        'crash_penalty': crash_penalty,
        'total': reward
    }

    return float(reward), individual_reward