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
    
    # Calculate current distance (L1 norm for consistency)
    curr_dist = abs(x_pos) + abs(y_pos)
    
    # Diagnostics
    info['fuel_step'] = float(m_power + s_power)
    info['dist_x'] = abs(x_pos)
    info['dist_y'] = abs(y_pos)
    info['vel_mag'] = (vx**2 + vy**2)**0.5
    info['angle_err'] = abs(angle)
    
    # Stateful Progress Shaping: Reward reduction in distance to pad
    if not hasattr(self, '_prev_dist'):
        self._prev_dist = curr_dist
        dist_delta = 0.0
    else:
        dist_delta = max(0.0, self._prev_dist - curr_dist)
        self._prev_dist = curr_dist
    
    info['dist_delta'] = float(dist_delta)
    
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
        # Positive shaping for progress towards pad (offsets maneuvering costs)
        progress_reward = 0.5 * dist_delta
        
        # Mild penalty for distance from target (x=0, y=0) - kept to guide alignment
        dist_penalty = -0.3 * curr_dist
        
        # Mild penalty for orientation deviation
        angle_penalty = -0.2 * abs(angle)
        
        # Fuel consumption penalty
        fuel_penalty = -0.1 * (m_power + s_power)
        
        # Reduced velocity penalty to allow gravity-driven descent while penalizing high speed
        vel_penalty = -0.2 * abs(vy)
        
        reward += progress_reward + dist_penalty + angle_penalty + fuel_penalty + vel_penalty
    
    return float(reward), info