def compute_reward(self, state, m_power, s_power, terminated):
    x, y, vx, vy, angle, angular_velocity, leg_left, leg_right = state
    distance = (x**2 + y**2)**0.5
    both_legs = leg_left > 0.5 and leg_right > 0.5
    success = both_legs and abs(x) < 0.1 and abs(y) < 0.1 and abs(vy) < 0.5 and abs(angle) < 0.1
    if success:
        objective_bonus = 100.0
    else:
        objective_bonus = 0.0
    progress = -0.3 * distance - 0.1 * abs(vy)
    main_engine_reward = 0.5 if m_power > 0 else 0.0
    regularizers = -0.03 * m_power - 0.02 * s_power - 0.05 * abs(angle)
    if terminated and not success:
        crash_penalty = -100.0
    else:
        crash_penalty = 0.0
    reward = objective_bonus + progress + main_engine_reward + regularizers + crash_penalty
    individual_reward = {
        'objective_bonus': objective_bonus,
        'progress': progress,
        'main_engine_reward': main_engine_reward,
        'regularizers': regularizers,
        'crash_penalty': crash_penalty,
        'distance': distance,
        'vy': vy,
        'leg_contact': both_legs,
        'angle': angle,
        'angular_velocity': angular_velocity,
        'm_power': m_power,
        's_power': s_power,
        'crash': 1.0 if (terminated and not success) else 0.0,
        'main_engine_used': 1.0 if m_power > 0 else 0.0
    }
    return float(reward), individual_reward