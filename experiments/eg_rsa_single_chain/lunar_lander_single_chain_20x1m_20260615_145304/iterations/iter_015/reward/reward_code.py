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
    progress_flag = 0.0
    approach_flag = 0.0
    near_pad_flag = 0.0
    touchdown_stability_flag = 0.0

    # Check success condition
    success = (leg1 == 1.0 and leg2 == 1.0 and abs(x) < 0.15 and abs(y) < 0.15 and abs(vy) < 0.5)
    if success:
        success_flag = 1.0

    # Check crash (terminated without success)
    if terminated and not success:
        crash_flag = 1.0

    # One-shot progress bonus: first time distance<0.5 with any leg contact
    if not hasattr(self, '_progress_bonus_given'):
        self._progress_bonus_given = False
    if not self._progress_bonus_given and distance_to_pad < 0.5 and (leg1 == 1.0 or leg2 == 1.0):
        progress_flag = 1.0
        self._progress_bonus_given = True

    # One-shot approach bonus: first time distance<0.15 with any leg contact
    if not hasattr(self, '_approach_bonus_given'):
        self._approach_bonus_given = False
    if not self._approach_bonus_given and distance_to_pad < 0.15 and (leg1 == 1.0 or leg2 == 1.0):
        approach_flag = 1.0
        self._approach_bonus_given = True

    # Per-step near-pad bonus
    if distance_to_pad < 0.5:
        near_pad_flag = 1.0

    # Touchdown stability bonus: both legs contact with low velocities, small angle, near pad
    if leg1 == 1.0 and leg2 == 1.0 and abs(vy) < 0.5 and abs(vx) < 0.5 and abs(angle) < 0.2 and distance_to_pad < 0.04:
        touchdown_stability_flag = 1.0

    # Compute reward components
    objective_bonus = 100.0 if success else 0.0
    crash_penalty = -100.0 if crash_flag else 0.0
    progress_shaping = -0.08 * distance_to_pad - 0.02 * angle_deviation
    velocity_penalty = -0.02 * vertical_speed - 0.01 * lateral_speed
    fuel_penalty = -0.01 * main_engine_usage
    progress_bonus = 30.0 if progress_flag else 0.0
    approach_bonus = 50.0 if approach_flag else 0.0
    near_pad_bonus = 0.2 if near_pad_flag else 0.0
    touchdown_stability_bonus = 10.0 if touchdown_stability_flag else 0.0

    reward = objective_bonus + crash_penalty + progress_shaping + velocity_penalty + fuel_penalty + progress_bonus + approach_bonus + near_pad_bonus + touchdown_stability_bonus

    individual_reward = {
        'distance_to_pad': distance_to_pad,
        'angle_deviation': angle_deviation,
        'vertical_speed': vertical_speed,
        'lateral_speed': lateral_speed,
        'main_engine_usage': main_engine_usage,
        'legs_contact': legs_contact,
        'success_flag': success_flag,
        'crash_flag': crash_flag,
        'progress_flag': progress_flag,
        'approach_flag': approach_flag,
        'near_pad_flag': near_pad_flag,
        'touchdown_stability_flag': touchdown_stability_flag,
        'objective_bonus': objective_bonus,
        'crash_penalty': crash_penalty,
        'progress_shaping': progress_shaping,
        'velocity_penalty': velocity_penalty,
        'fuel_penalty': fuel_penalty,
        'progress_bonus': progress_bonus,
        'approach_bonus': approach_bonus,
        'near_pad_bonus': near_pad_bonus,
        'touchdown_stability_bonus': touchdown_stability_bonus
    }

    return float(reward), individual_reward