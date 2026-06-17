def compute_reward(self, state, m_power, s_power, terminated):
    # Initialize diagnostics
    distance_to_pad = 0.0
    vertical_speed = 0.0
    horizontal_speed = 0.0
    angle = 0.0
    fuel_used = 0.0
    crash = 0.0
    success = 0.0
    reward_distance = 0.0
    reward_vertical = 0.0
    reward_angle = 0.0
    reward_fuel = 0.0
    reward_time = 0.0
    reward_horizontal = 0.0
    reward_crash = 0.0
    reward_success = 0.0
    reward_noop = 0.0
    reward_main_engine = 0.0
    reward_near_pad = 0.0
    progress_delta = 0.0
    success_flag = 0.0
    objective_bonus = 0.0

    # Compute distance to pad (pad at (0,0))
    distance_to_pad = (state[0]**2 + state[1]**2)**0.5
    vertical_speed = state[3]
    horizontal_speed = state[2]
    angle = state[4]
    fuel_used = m_power + s_power

    # Success bonus: terminal-aligned, one-shot
    # Only pay when episode terminates with both legs contact, near pad, low speed, upright
    if terminated and state[6] == 1.0 and state[7] == 1.0 and distance_to_pad < 0.15 and abs(vertical_speed) < 0.5 and abs(angle) < 0.2:
        reward_success = 100.0
        success = 1.0
        success_flag = 1.0
        objective_bonus = 100.0

    # Progress delta reward: reward for reducing distance to pad
    # Use internal state to track previous distance
    if not hasattr(self, '_prev_distance'):
        self._prev_distance = distance_to_pad
    prev_distance = self._prev_distance
    progress_delta = max(prev_distance - distance_to_pad, 0.0)
    progress_delta = min(progress_delta, 5.0)  # cap at 5 per step
    reward_distance = progress_delta
    self._prev_distance = distance_to_pad

    # Vertical speed shaping: encourage low vertical speed
    reward_vertical = -abs(vertical_speed) * 1.0

    # Angle shaping: encourage upright orientation
    reward_angle = -abs(angle) * 2.0

    # Fuel penalty: mild penalty for engine use
    reward_fuel = -0.03 * m_power - 0.01 * s_power

    # Time penalty: small per-step penalty
    reward_time = -0.1

    # Horizontal velocity penalty: discourage drifting
    reward_horizontal = -abs(horizontal_speed) * 0.5

    # No-op penalty: penalize action 0 (do nothing)
    if m_power == 0.0 and s_power == 0.0:
        reward_noop = -0.5

    # Main engine bonus: encourage using main engine when far from pad
    if m_power > 0.0 and distance_to_pad > 0.5:
        reward_main_engine = 0.5

    # Near-pad bonus: encourage stable approach near the pad
    if distance_to_pad < 0.3 and abs(vertical_speed) < 0.5:
        reward_near_pad = 1.0

    # Crash penalty: only if terminated due to actual crash or out-of-bounds
    if terminated:
        # Out-of-bounds: abs(state[0]) >= 1.0
        if abs(state[0]) >= 1.0:
            reward_crash = -100.0
            crash = 1.0
        # Actual crash: both legs not in contact and not near pad (heuristic for game_over)
        elif state[6] == 0.0 and state[7] == 0.0 and distance_to_pad > 0.3:
            reward_crash = -100.0
            crash = 1.0
        # If terminated but not crash/out-of-bounds (e.g., timeout or success), no crash penalty

    # Total reward
    reward = reward_success + reward_distance + reward_vertical + reward_angle + reward_fuel + reward_time + reward_horizontal + reward_noop + reward_main_engine + reward_near_pad + reward_crash

    # Individual reward dictionary
    individual_reward = {
        'distance_to_pad': distance_to_pad,
        'vertical_speed': vertical_speed,
        'horizontal_speed': horizontal_speed,
        'angle': angle,
        'fuel_used': fuel_used,
        'crash': crash,
        'success': success,
        'reward_success': reward_success,
        'reward_distance': reward_distance,
        'reward_vertical': reward_vertical,
        'reward_angle': reward_angle,
        'reward_fuel': reward_fuel,
        'reward_time': reward_time,
        'reward_horizontal': reward_horizontal,
        'reward_noop': reward_noop,
        'reward_main_engine': reward_main_engine,
        'reward_near_pad': reward_near_pad,
        'reward_crash': reward_crash,
        'progress_delta': progress_delta,
        'success_flag': success_flag,
        'objective_bonus': objective_bonus
    }

    return float(reward), individual_reward