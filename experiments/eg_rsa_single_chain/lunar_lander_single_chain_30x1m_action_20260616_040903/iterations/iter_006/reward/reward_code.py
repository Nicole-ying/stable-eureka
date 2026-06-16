def compute_reward(self, state, m_power, s_power, terminated):
    # Unpack state
    x, y, vx, vy, angle, angular_velocity, leg1_contact, leg2_contact = state

    # Initialize diagnostics
    distance_to_pad = 0.0
    vertical_speed = 0.0
    angle_deviation = 0.0
    main_engine_firings = 0.0
    orientation_engine_firings = 0.0
    both_legs_contact = 0.0
    crash_occurred = 0.0
    success_flag = 0.0

    # Compute distance to pad (pad at (0,0))
    distance = (x**2 + y**2)**0.5
    distance_to_pad = distance

    # Vertical speed magnitude
    vertical_speed = abs(vy)

    # Angle deviation from upright
    angle_deviation = abs(angle)

    # Both legs contact flag
    both_legs_contact = 1.0 if (leg1_contact > 0.5 and leg2_contact > 0.5) else 0.0

    # Engine usage counts (approximate as binary per step)
    main_engine_firings = 1.0 if m_power > 0.5 else 0.0
    orientation_engine_firings = 1.0 if s_power > 0.5 else 0.0

    # Objective success bonus: one-shot at termination only
    success_bonus = 0.0
    if terminated and both_legs_contact > 0.5 and abs(x) < 0.1 and abs(y) < 0.1 and abs(vy) < 0.5 and abs(vx) < 0.5:
        success_bonus = 100.0
        success_flag = 1.0

    # Time bonus: one-shot at termination for early termination
    time_bonus = 0.0
    time_bonus_very_early = 0.0
    if terminated and success_flag > 0.5:
        if hasattr(self, 'episode_length'):
            ep_len = self.episode_length
        else:
            ep_len = 0
        if ep_len < 300:
            time_bonus_very_early = 100.0
        if ep_len < 500:
            time_bonus = 50.0
        elif ep_len < 750:
            time_bonus = 25.0

    # Progress rewards
    distance_reward = 0.5 * (1 - math.tanh(3 * distance))
    vertical_speed_reward = 0.3 * (1 - math.tanh(2 * abs(vy)))
    angle_reward = 0.1 * (1 - math.tanh(5 * abs(angle)))

    # Engine penalties conditioned on not being landed (to discourage hovering on pad)
    main_penalty = -0.03 * m_power * (1 - both_legs_contact)
    orientation_penalty = -0.01 * s_power * (1 - both_legs_contact)

    # Crash penalty
    crash_penalty = 0.0
    if terminated and not (both_legs_contact > 0.5 and abs(x) < 0.1 and abs(y) < 0.1):
        crash_penalty = -100.0
        crash_occurred = 1.0

    # Time penalty
    time_penalty = -0.01

    # Total reward
    reward = success_bonus + time_bonus + time_bonus_very_early + distance_reward + vertical_speed_reward + angle_reward + main_penalty + orientation_penalty + crash_penalty + time_penalty

    # Individual reward dictionary
    individual_reward = {
        'success_bonus': success_bonus,
        'time_bonus': time_bonus,
        'time_bonus_very_early': time_bonus_very_early,
        'distance_reward': distance_reward,
        'vertical_speed_reward': vertical_speed_reward,
        'angle_reward': angle_reward,
        'main_penalty': main_penalty,
        'orientation_penalty': orientation_penalty,
        'crash_penalty': crash_penalty,
        'time_penalty': time_penalty,
        'distance_to_pad': distance_to_pad,
        'vertical_speed': vertical_speed,
        'angle_deviation': angle_deviation,
        'main_engine_firings': main_engine_firings,
        'orientation_engine_firings': orientation_engine_firings,
        'both_legs_contact': both_legs_contact,
        'crash_occurred': crash_occurred,
        'success_flag': success_flag
    }

    return float(reward), individual_reward