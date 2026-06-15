def compute_reward(self, state, m_power, s_power, terminated):
    # Unpack state components
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = state[4]
    ang_vel = state[5]
    leg_l = state[6]
    leg_r = state[7]

    reward = 0.0
    individual_reward = {}

    # --- Diagnostics (Numeric) ---
    distance_to_target = (x**2 + y**2)**0.5
    vertical_velocity_mag = abs(vy)
    fuel_consumption = m_power + s_power
    angle_deviation = abs(angle)

    individual_reward['distance_to_target'] = float(distance_to_target)
    individual_reward['vertical_velocity'] = float(vertical_velocity_mag)
    individual_reward['fuel_consumption'] = float(fuel_consumption)
    individual_reward['angle_deviation'] = float(angle_deviation)

    # --- Objective Feedback (Sparse) ---
    # Success: Both legs grounded, near pad center, low vertical velocity
    landed = (leg_l >= 0.9 and leg_r >= 0.9)
    on_pad = abs(x) < 0.5
    soft_touchdown = vertical_velocity_mag < 2.0

    if terminated:
        if landed and on_pad and soft_touchdown:
            reward += 100.0
            individual_reward['landing_success'] = True
        else:
            # Failure: Crash or Out of Bounds
            crash_penalty = -50.0
            out_of_bounds_penalty = -20.0
            
            if hasattr(self, 'game_over') and self.game_over:
                reward += crash_penalty
            elif abs(x) >= 1.0:
                reward += out_of_bounds_penalty
            else:
                # Bad landing (e.g., high velocity impact)
                reward -= 25.0
            individual_reward['landing_success'] = False
    
    # --- Reachable Progress Feedback (Dense Shaping) ---
    # Only apply shaping if not terminated to avoid double counting at end step
    if not terminated:
        progress_signal = -0.1 * distance_to_target - 0.05 * vertical_velocity_mag
        reward += progress_signal

    # --- Mild Regularizers ---
    fuel_penalty = -0.05 * fuel_consumption
    angle_stability = -0.02 * angle_deviation
    
    reward += fuel_penalty + angle_stability

    return float(reward), individual_reward