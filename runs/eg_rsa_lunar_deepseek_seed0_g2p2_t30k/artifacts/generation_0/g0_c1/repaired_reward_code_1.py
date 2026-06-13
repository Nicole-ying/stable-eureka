def compute_reward(obs, action, next_obs, done, info):
    # Extract observations
    x = obs[0]
    y = obs[1]
    vel_x = obs[2]
    vel_y = obs[3]
    angle = obs[4]
    angular_vel = obs[5]
    leg_contact_0 = obs[6]
    leg_contact_1 = obs[7]
    
    # Extract next_obs for terminal checks
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vel_x = next_obs[2]
    next_vel_y = next_obs[3]
    next_angle = next_obs[4]
    next_leg_contact_0 = next_obs[6]
    next_leg_contact_1 = next_obs[7]
    
    # Extract action info (discrete: 0=do nothing, 1=left, 2=main, 3=right)
    main_engine_fired = 1.0 if action == 2 else 0.0
    side_engine_fired = 1.0 if action in [1, 3] else 0.0
    
    # ========== Component: distance_penalty ==========
    distance = (x**2 + y**2)**0.5
    distance_penalty = -0.5 * distance**2
    
    # ========== Component: velocity_penalty ==========
    vel_penalty_x = -0.1 * abs(vel_x)
    vel_penalty_y = -0.5 * abs(vel_y)
    velocity_penalty = vel_penalty_x + vel_penalty_y
    
    # ========== Component: angle_penalty ==========
    angle_penalty = -0.3 * abs(angle)
    
    # ========== Component: fuel_penalty ==========
    fuel_penalty = -0.02 * (main_engine_fired + side_engine_fired)
    
    # ========== Component: progress ==========
    next_distance = (next_x**2 + next_y**2)**0.5
    distance_improvement = distance - next_distance
    progress_reward = 0.5 * distance_improvement
    vel_improvement = abs(vel_y) - abs(next_vel_y)
    progress_reward += 0.3 * vel_improvement
    
    # ========== Component: stability ==========
    stability_reward = 0.0
    if abs(angle) < 0.2:
        stability_reward += 0.1
    if abs(angular_vel) < 0.1:
        stability_reward += 0.05
    if leg_contact_0 or leg_contact_1:
        stability_reward += 0.2
    
    # ========== Component: effort ==========
    if distance < 0.2 and abs(vel_y) < 0.1:
        effort_penalty = -0.05 * (main_engine_fired + side_engine_fired)
    else:
        effort_penalty = 0.0
    
    # ========== Component: terminal shaping ==========
    terminal_reward = 0.0
    if done:
        if (leg_contact_0 and leg_contact_1) and abs(angle) < 0.1 and abs(vel_y) < 0.1:
            terminal_reward = 10.0
        elif abs(vel_y) > 0.5 or abs(angle) > 0.5:
            terminal_reward = -10.0
        else:
            terminal_reward = -5.0
    
    # ========== Component: landing_bonus ==========
    landing_bonus = 0.0
    if (leg_contact_0 and leg_contact_1) and abs(angle) < 0.1 and abs(vel_y) < 0.1:
        landing_bonus = 15.0
    
    # ========== Component: crash_penalty ==========
    crash_penalty = 0.0
    if done and (abs(vel_y) > 0.5 or abs(angle) > 0.5):
        crash_penalty = -10.0
    
    # ========== Total Reward ==========
    total_reward = (
        distance_penalty +
        velocity_penalty +
        angle_penalty +
        fuel_penalty +
        progress_reward +
        stability_reward +
        effort_penalty +
        terminal_reward +
        landing_bonus +
        crash_penalty
    )
    
    # Clip total reward to be within [-1000, 1000] as per schema
    total_reward = max(-1000.0, min(1000.0, total_reward))
    
    # Components dict (all required IDs)
    components = {
        "landing_bonus": landing_bonus,
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "fuel_penalty": fuel_penalty,
        "crash_penalty": crash_penalty,
        "progress": progress_reward,
        "stability": stability_reward,
        "effort": effort_penalty,
        "terminal": terminal_reward
    }
    
    return float(total_reward), components