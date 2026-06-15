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

    # --- State Tracking for Progress Bonus ---
    if not hasattr(self, 'prev_distance'):
        self.prev_distance = distance_to_target
        self.prev_velocity = vertical_velocity_mag

    # --- Objective Feedback (Sparse) ---
    landed = (leg_l >= 0.9 and leg_r >= 0.9)
    on_pad = abs(x) < 0.5
    soft_touchdown = vertical_velocity_mag < 2.0

    if terminated:
        if landed and on_pad and soft_touchdown:
            reward += 100.0
            individual_reward['landing_success'] = True
        else:
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
        
        # Reset state tracking for next episode
        self.prev_distance = distance_to_target
        self.prev_velocity = vertical_velocity_mag
        return float(reward), individual_reward

    # --- Reachable Progress Feedback (Dense Shaping) ---
    # Positive reward for reducing distance, penalize high velocity to prevent crashing
    dist_reduction = max(0.0, self.prev_distance - distance_to_target)
    progress_bonus = 0.5 * dist_reduction
    
    # Velocity penalty scaled up to balance progress bonus (prevents fast falling)
    velocity_penalty = -0.2 * vertical_velocity_mag
    
    reward += progress_bonus + velocity_penalty
    individual_reward['progress_bonus'] = float(progress_bonus)
    individual_reward['velocity_penalty'] = float(velocity_penalty)

    # --- Mild Regularizers ---
    fuel_penalty = -0.05 * fuel_consumption
    angle_stability = -0.02 * angle_deviation
    
    reward += fuel_penalty + angle_stability
    individual_reward['fuel_penalty'] = float(fuel_penalty)
    individual_reward['angle_stability'] = float(angle_stability)

    # Update state tracking
    self.prev_distance = distance_to_target
    self.prev_velocity = vertical_velocity_mag

    return float(reward), individual_reward