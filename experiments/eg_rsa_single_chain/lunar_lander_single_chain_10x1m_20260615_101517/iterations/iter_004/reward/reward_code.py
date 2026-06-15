def compute_reward(self, state, m_power, s_power, terminated):
    x, y, vx, vy, angle, angular_velocity, leg_left, leg_right = state
    distance = (x**2 + y**2)**0.5
    both_legs = leg_left > 0.5 and leg_right > 0.5
    success = both_legs and abs(x) < 0.1 and abs(y) < 0.1 and abs(vy) < 0.5 and abs(angle) < 0.1
    if success:
        objective_bonus = 200.0
    else:
        objective_bonus = 0.0
    progress = -1.0 * distance - 0.3 * abs(vy)
    regularizers = -0.03 * m_power - 0.02 * s_power - 0.05 * abs(angle)
    if terminated and not success:
        crash_penalty = -100.0
    else:
        crash_penalty = 0.0
    # Progress-conditioned main engine reward
    if not hasattr(self, 'prev_distance'):
        self.prev_distance = distance
    if m_power > 0 and distance < self.prev_distance:
        main_engine_reward = 2.0
    else:
        main_engine_reward = 0.0
    # Descent reward
    if m_power > 0 and vy < 0:
        if not hasattr(self, 'last_vy'):
            self.last_vy = vy
        descent_reward = 0.5 * (self.last_vy - vy)
        self.last_vy = vy
    else:
        descent_reward = 0.0
        if hasattr(self, 'last_vy'):
            del self.last_vy
    # Hovering penalty
    if abs(vy) < 0.1 and distance >= self.prev_distance:
        hovering_penalty = -0.5
    else:
        hovering_penalty = 0.0
    self.prev_distance = distance
    reward = objective_bonus + progress + regularizers + crash_penalty + main_engine_reward + descent_reward + hovering_penalty
    individual_reward = {
        'objective_bonus': objective_bonus,
        'progress': progress,
        'regularizers': regularizers,
        'crash_penalty': crash_penalty,
        'main_engine_reward': main_engine_reward,
        'descent_reward': descent_reward,
        'hovering_penalty': hovering_penalty,
        'distance': distance,
        'vy': vy,
        'leg_contact': both_legs,
        'angle': angle,
        'angular_velocity': angular_velocity,
        'm_power': m_power,
        's_power': s_power,
        'crash': 1.0 if (terminated and not success) else 0.0
    }
    return float(reward), individual_reward