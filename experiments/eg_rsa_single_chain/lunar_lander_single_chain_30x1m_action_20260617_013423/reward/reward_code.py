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

    # Compute distance to pad (pad at (0,0))
    distance_to_pad = (state[0]**2 + state[1]**2)**0.5
    vertical_speed = state[3]
    horizontal_speed = state[2]
    angle = state[4]
    fuel_used = m_power + s_power

    # Success bonus: both legs contact, near pad, low speeds, upright
    if state[6] == 1.0 and state[7] == 1.0 and distance_to_pad < 0.1 and abs(vertical_speed) < 0.5 and abs(horizontal_speed) < 0.5 and abs(angle) < 0.1:
        reward_success = 100.0
        success = 1.0

    # Distance shaping: encourage moving closer to pad
    reward_distance = -distance_to_pad * 2.0

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

    # Crash penalty: large penalty if terminated due to crash or out-of-bounds
    if terminated:
        # Check if crash (game_over) or out-of-bounds
        # We can infer crash from terminated and not success (since success also sets terminated? Actually success does not set terminated, but landing sets legs contact but not terminated unless crash/out-of-bounds/not awake)
        # In LunarLander, successful landing does not terminate; termination only on crash/out-of-bounds/not awake.
        # So if terminated, it's a failure.
        reward_crash = -100.0
        crash = 1.0

    # Total reward
    reward = reward_success + reward_distance + reward_vertical + reward_angle + reward_fuel + reward_time + reward_horizontal + reward_crash

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
        'reward_crash': reward_crash
    }

    return float(reward), individual_reward