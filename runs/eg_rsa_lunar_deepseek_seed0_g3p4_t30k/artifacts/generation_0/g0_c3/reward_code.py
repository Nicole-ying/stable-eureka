def compute_reward(obs, action, next_obs, done, info):
    # Extract state components from next_obs (post-transition state)
    x = next_obs[0]
    y = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_leg = next_obs[6]
    right_leg = next_obs[7]
    
    # Progress: encourage moving toward landing pad (0,0) with gentle speed
    dist = np.sqrt(x**2 + y**2)
    # Use exponential shaping to give strong signal near pad, mild signal far away
    dist_reward = -0.5 * dist  # linear penalty for distance
    
    # Speed shaping: penalize high speeds, especially vertical speed near ground
    speed = np.sqrt(x_vel**2 + y_vel**2)
    # Heavier penalty for vertical speed (crash risk)
    vertical_penalty = 1.0 * abs(y_vel)
    horizontal_penalty = 0.3 * abs(x_vel)
    speed_penalty = -0.5 * speed - 0.5 * vertical_penalty - 0.2 * horizontal_penalty
    
    # Stability: penalize large angles and angular velocity
    angle_penalty = -1.0 * abs(angle) - 0.5 * abs(ang_vel)
    
    # Effort: penalize engine usage (fuel cost)
    # action is discrete: 0=do nothing, 1=left, 2=main, 3=right
    # Main engine (action 2) costs fuel and is heavy
    # Side engines (1,3) cost some fuel
    main_fuel_penalty = -0.3 if action == 2 else 0.0
    side_fuel_penalty = -0.1 if action in (1, 3) else 0.0
    effort_penalty = main_fuel_penalty + side_fuel_penalty
    
    # Terminal: reward safe landing when both legs contact ground
    # Landing pad is at (0,0), allow small x tolerance, require low vertical speed
    both_legs = left_leg > 0.5 and right_leg > 0.5
    near_pad = abs(x) < 0.15
    low_speed = abs(y_vel) < 0.1 and abs(x_vel) < 0.15
    upright = abs(angle) < 0.2
    
    terminal_reward = 0.0
    if done:
        # Check if termination was a successful landing or crash
        if both_legs and near_pad and low_speed and upright:
            terminal_reward = 100.0  # Successful landing
        else:
            # Crash or out-of-bounds: negative penalty
            terminal_reward = -50.0
            
    # Total reward
    total_reward = dist_reward + speed_penalty + angle_penalty + effort_penalty + terminal_reward
    
    # Clamp to absolute bound
    total_reward = np.clip(total_reward, -1000.0, 1000.0)
    
    # Components dict matching schema IDs
    components = {
        "progress": float(dist_reward + speed_penalty),  # task progress toward landing
        "stability": float(angle_penalty),               # upright orientation
        "effort": float(effort_penalty),                 # fuel efficiency
        "terminal": float(terminal_reward)               # landing success/failure
    }
    
    return float(total_reward), components
