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
    coast_bonus = 0.0

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
    current_distance = (x**2 + y**2)**0.5
    # Use internal state to store previous distance for progress-delta
    if not hasattr(self, '_prev_distance'):
        self._prev_distance = current_distance
    distance_delta = current_distance - self._prev_distance
    self._prev_distance = current_distance
    distance = -0.5 * distance_delta  # penalize only when distance increases
    vertical_speed = -0.3 * abs(vy)
    angle = -0.2 * abs(angle_val)

    # Regularizers
    main_fuel = -0.03 * m_power
    orientation_fuel = -0.01 * s_power

    # Action bonus for using right orientation engine when horizontal velocity is away from pad
    if m_power == 0 and s_power > 0 and vx * x > 0:
        horizontal_correction_bonus = 0.5

    # Action bonus for using action 0 (do nothing) when near pad with low speed and upright
    near_pad = (abs(x) < 0.1) and (abs(y) < 0.1)
    low_speed = abs(vy) < 0.1
    upright = abs(angle_val) < 0.1
    if m_power == 0 and s_power == 0 and near_pad and low_speed and upright:
        coast_bonus = 0.5

    # Objective signal: successful landing
    both_legs = (leg1 > 0.5) and (leg2 > 0.5)
    if both_legs and near_pad and low_speed and upright and not terminated:
        success = 100.0

    # Terminal penalty for crash or out-of-bounds
    if terminated and not (both_legs and near_pad and low_speed and upright):
        crash_penalty = -100.0

    # Total reward
    reward = distance + vertical_speed + angle + main_fuel + orientation_fuel + horizontal_correction_bonus + coast_bonus + success + crash_penalty

    # Diagnostics
    individual_reward = {
        'distance': distance,
        'vertical_speed': vertical_speed,
        'angle': angle,
        'main_fuel': main_fuel,
        'orientation_fuel': orientation_fuel,
        'horizontal_correction_bonus': horizontal_correction_bonus,
        'coast_bonus': coast_bonus,
        'success': success,
        'crash_penalty': crash_penalty,
        'total': reward
    }

    return float(reward), individual_reward