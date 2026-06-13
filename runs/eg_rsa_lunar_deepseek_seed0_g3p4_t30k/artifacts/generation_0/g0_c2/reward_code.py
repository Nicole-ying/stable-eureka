def compute_reward(obs, action, next_obs, done, info):
    # Unpack observations
    x, y = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    left_leg, right_leg = obs[6], obs[7]
    
    # Next state for shaping deltas
    next_x, next_y = next_obs[0], next_obs[1]
    
    # ----- Progress: encourage moving toward landing pad (0,0) -----
    current_dist = np.sqrt(x**2 + y**2)
    next_dist = np.sqrt(next_x**2 + next_y**2)
    # Delta negative if moving closer
    dist_delta = current_dist - next_dist
    progress = 0.5 * dist_delta  # positive for moving closer
    
    # ----- Stability: penalize tilt, angular velocity, and high speeds -----
    # Angle penalty: upright is 0, max tilt ~ pi
    angle_penalty = -0.3 * abs(angle)
    # Angular velocity penalty
    ang_vel_penalty = -0.05 * abs(ang_vel)
    # Speed penalty: discourage high velocities, especially vertical
    speed = np.sqrt(x_vel**2 + y_vel**2)
    speed_penalty = -0.2 * speed
    # Extra vertical velocity penalty near ground (y close to 0)
    vertical_penalty = -0.5 * max(0, abs(y_vel) - 0.1) * (1.0 / (1.0 + 10.0 * abs(y + 0.5)))
    stability = angle_penalty + ang_vel_penalty + speed_penalty + vertical_penalty
    
    # ----- Effort: penalize engine usage -----
    # action 2 = main engine, action 1 or 3 = side engines
    main_engine = 1.0 if action == 2 else 0.0
    side_engine = 1.0 if action in [1, 3] else 0.0
    effort = -0.2 * main_engine - 0.05 * side_engine
    
    # ----- Terminal: success or failure -----
    terminal = 0.0
    if done:
        # Check if we have a successful landing: both legs contact, near center, low velocity
        both_legs = (left_leg > 0.5) and (right_leg > 0.5)
        near_pad = abs(x) < 0.1
        low_vertical_speed = abs(y_vel) < 0.1
        low_horizontal_speed = abs(x_vel) < 0.1
        upright = abs(angle) < 0.2
        
        if both_legs and near_pad and low_vertical_speed and low_horizontal_speed and upright:
            terminal = 100.0  # successful landing
        else:
            # Crash or out-of-bounds
            terminal = -50.0
    
    # Combine components
    total_reward = progress + stability + effort + terminal
    
    # Clamp total reward to bound
    total_reward = np.clip(total_reward, -1000.0, 1000.0)
    
    components = {
        "progress": progress,
        "stability": stability,
        "effort": effort,
        "terminal": terminal
    }
    
    return float(total_reward), components
