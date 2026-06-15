def compute_reward(self, state, m_power, s_power, terminated):
    reward = 0.0
    info = {}
    
    # Extract state components
    x_pos = state[0]
    y_pos = state[1]
    vx = state[2]
    vy = state[3]
    angle = state[4]
    leg0 = state[6]
    leg1 = state[7]
    
    # Diagnostics
    info['fuel_step'] = float(m_power + s_power)
    info['dist_x'] = abs(x_pos)
    info['dist_y'] = abs(y_pos)
    info['vel_mag'] = (vx**2 + vy**2)**0.5
    info['angle_err'] = abs(angle)
    
    if terminated:
        # Terminal Evaluation
        is_safe_landing = False
        if leg1 == 1.0 and leg0 == 1.0:
            # Check velocity thresholds for safe landing (normalized approx)
            if abs(vx) < 1.5 and abs(vy) < 1.5 and abs(x_pos) < 0.5:
                is_safe_landing = True
        
        info['is_success'] = int(is_safe_landing)
        
        if is_safe_landing:
            reward += 100.0
        else:
            reward -= 100.0
    else:
        # Step-wise Shaping & Regularization
        # Reduced penalty for distance from target (x=0, y=0) to allow exploration
        dist_penalty = -0.2 * (abs(x_pos) + abs(y_pos))
        
        # Mild penalty for orientation deviation
        angle_penalty = -0.2 * abs(angle)
        
        # Fuel consumption penalty
        fuel_penalty = -0.1 * (m_power + s_power)
        
        # NEW: Conditional Thrust Incentive
        # Encourage Main Engine usage when upright to break 'do nothing' local optimum
        thrust_bonus = 0.0
        if m_power > 0 and abs(angle) < 0.5:
            thrust_bonus = 0.5
        
        reward += dist_penalty + angle_penalty + fuel_penalty + thrust_bonus
    
    info['thrust_bonus'] = float(thrust_bonus)
    return float(reward), info