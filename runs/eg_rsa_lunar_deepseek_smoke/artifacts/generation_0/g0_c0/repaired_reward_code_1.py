def compute_reward(obs, action, next_obs, done, info):
    # Unpack observations - only first 8 elements
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = next_obs[:8]
    
    # Extract engine usage from action
    main_engine = 1.0 if action == 2 else 0.0
    side_engine = 1.0 if action in [1, 3] else 0.0
    
    # Terminal component: safe landing at pad
    at_pad_x = abs(x) < 0.1
    at_pad_y = abs(y) < 0.1
    both_legs_grounded = (left_leg > 0.5) and (right_leg > 0.5)
    low_speed = (abs(vx) < 0.1) and (abs(vy) < 0.1)
    stable_angle = abs(angle) < 0.1
    landing_conditions = at_pad_x and at_pad_y and both_legs_grounded and low_speed and stable_angle
    
    terminal_reward = 100.0 if landing_conditions else 0.0
    
    # Velocity penalty: penalize high speeds, especially vertical
    vel_penalty = -1.5 * (vx**2 + vy**2)
    
    # Angle penalty: penalize tilt and spin
    angle_penalty = -2.0 * (angle**2 + angular_vel**2)
    
    # Fuel efficiency: penalize engine usage
    fuel_penalty = -0.5 * main_engine - 0.2 * side_engine
    
    # Distance progress: guide toward pad (x=0, y=0)
    dist_to_pad = (x**2 + y**2)**0.5
    distance_penalty = -0.5 * dist_to_pad
    
    # Additional shaping: encourage being close to pad with legs down
    near_pad = dist_to_pad < 0.3
    legs_bonus = min(0.5 * (left_leg + right_leg), 1.0) if near_pad else 0.0
    
    # Combine all components
    total_reward = terminal_reward + vel_penalty + angle_penalty + fuel_penalty + distance_penalty + legs_bonus
    
    # Clamp total_reward to bounded range
    total_reward = max(min(total_reward, 1000.0), -1000.0)
    
    components = {
        'terminal': terminal_reward,
        'velocity_penalty': vel_penalty,
        'angle_penalty': angle_penalty,
        'fuel_efficiency': fuel_penalty,
        'distance_progress': distance_penalty
    }
    
    return float(total_reward), components
