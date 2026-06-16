def compute_reward(self, state, m_power, s_power, terminated):
    x, y, vx, vy, angle, angular_velocity, leg1, leg2 = state
    # Initialize diagnostics
    distance_to_pad = x*x + y*y
    angle_deviation = angle*angle
    vertical_speed = vy*vy
    lateral_speed = vx*vx
    main_engine_usage = m_power
    legs_contact = leg1 + leg2
    success_flag = 0.0
    crash_flag = 0.0
    controlled_descent_flag = 0.0
    touchdown_stability_flag = 0.0
    distance_improvement = 0.0

    # Check success condition
    success = (leg1 == 1.0 and leg2 == 1.0 and abs(x) < 0.15 and abs(y) < 0.15 and abs(vy) < 0.5)

    # Check crash (terminated without success)
    if terminated and not success:
        crash_flag = 1.0

    # Distance improvement bonus: reward decreasing distance (uncapped)
    if not hasattr(self, '_prev_distance'):
        self._prev_distance = distance_to_pad
    prev_dist = self._prev_distance
    if distance_to_pad < prev_dist:
        improvement = prev_dist - distance_to_pad
        distance_improvement = 0.2 * improvement
    self._prev_distance = distance_to_pad

    # Controlled descent bonus: one-shot event-gated
    if not hasattr(self, '_controlled_descent_given'):
        self._controlled_descent_given = False
    if not self._controlled_descent_given and m_power > 0 and distance_to_pad < 0.5 and abs(angle) < 0.2:
        controlled_descent_flag = 1.0
        self._controlled_descent_given = True

    # Touchdown stability bonus: one-shot event-gated
    if not hasattr(self, '_touchdown_stability_given'):
        self._touchdown_stability_given = False
    if not self._touchdown_stability_given and leg1 == 1.0 and leg2 == 1.0 and distance_to_pad < 0.5 and abs(vy) < 0.5 and abs(vx) < 0.5 and abs(angle) < 0.2:
        touchdown_stability_flag = 1.0
        self._touchdown_stability_given = True

    # Objective success bonus: one-shot event-gated
    if not hasattr(self, '_success_given'):
        self._success_given = False
    if success and not self._success_given:
        success_flag = 1.0
        self._success_given = True

    # Compute reward components
    objective_bonus = 100.0 if success_flag else 0.0
    crash_penalty = -100.0 if crash_flag else 0.0
    angle_penalty = -0.02 * angle_deviation
    velocity_penalty = -0.02 * vertical_speed - 0.01 * lateral_speed
    fuel_penalty = -0.01 * main_engine_usage
    controlled_descent_bonus = 10.0 if controlled_descent_flag else 0.0
    touchdown_stability_bonus = 20.0 if touchdown_stability_flag else 0.0

    reward = objective_bonus + crash_penalty + distance_improvement + angle_penalty + velocity_penalty + fuel_penalty + controlled_descent_bonus + touchdown_stability_bonus

    individual_reward = {
        'distance_to_pad': distance_to_pad,
        'angle_deviation': angle_deviation,
        'vertical_speed': vertical_speed,
        'lateral_speed': lateral_speed,
        'main_engine_usage': main_engine_usage,
        'legs_contact': legs_contact,
        'success_flag': success_flag,
        'crash_flag': crash_flag,
        'controlled_descent_flag': controlled_descent_flag,
        'touchdown_stability_flag': touchdown_stability_flag,
        'distance_improvement': distance_improvement,
        'objective_bonus': objective_bonus,
        'crash_penalty': crash_penalty,
        'angle_penalty': angle_penalty,
        'velocity_penalty': velocity_penalty,
        'fuel_penalty': fuel_penalty,
        'controlled_descent_bonus': controlled_descent_bonus,
        'touchdown_stability_bonus': touchdown_stability_bonus
    }

    return float(reward), individual_reward