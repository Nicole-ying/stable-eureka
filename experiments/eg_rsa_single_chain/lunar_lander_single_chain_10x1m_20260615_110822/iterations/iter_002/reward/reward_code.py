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
        # Mild penalty for distance from target (x=0, y=0) - Reduced weight to avoid dominance
        dist_penalty = -0.1 * (abs(x_pos) + abs(y_pos))
        
        # Mild penalty for orientation deviation - Reduced weight
        angle_penalty = -0.1 * abs(angle)
        
        # Fuel consumption penalty - Kept mild to discourage hovering but allow maneuvering
        fuel_penalty = -0.1 * (m_power + s_power)
        
        # Removed explicit velocity penalty to avoid conflict with gravity physics
        
        reward += dist_penalty + angle_penalty + fuel_penalty
    
    return float(reward), info