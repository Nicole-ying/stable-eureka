def compute_reward(obs, action, next_obs, done, info):
    # Extract relevant quantities from next_obs
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # --- Distance penalty: moderate linear penalty ---
    distance = np.sqrt(x**2 + y**2)
    distance_penalty = -2.0 * distance
    
    # --- Velocity penalty: gentle linear penalty ---
    velocity = np.sqrt(vx**2 + vy**2)
    velocity_penalty = -1.0 * velocity
    
    # --- Angle penalty: encourage upright orientation ---
    angle_penalty = -2.0 * abs(angle)
    
    # --- Progress shaping: reward reductions in distance and velocity ---
    # Estimate previous distance using obs (if available) or default to 0
    prev_x, prev_y = obs[0], obs[1]
    prev_distance = np.sqrt(prev_x**2 + prev_y**2)
    distance_progress = prev_distance - distance  # positive if getting closer
    prev_vx, prev_vy = obs[2], obs[3]
    prev_velocity = np.sqrt(prev_vx**2 + prev_vy**2)
    velocity_progress = prev_velocity - velocity  # positive if slowing down
    shaping_bonus = 1.0 * distance_progress + 0.5 * velocity_progress
    
    # --- Engine usage penalty: small penalty for main engine ---
    engine_usage_penalty = -0.1 if action == 2 else 0.0
    
    # --- Terminal reward: successful landing with relaxed conditions ---
    landing_condition = (
        left_contact > 0.5 and 
        right_contact > 0.5 and 
        distance < 0.2 and 
        abs(angle) < 0.2 and 
        abs(vy) < 0.2
    )
    terminal = 300.0 if landing_condition else 0.0
    
    # --- Crash penalty: moderate negative for failure ---
    crash_penalty = -50.0 if (done and not landing_condition) else 0.0
    
    # Sum up all components
    total_reward = (distance_penalty + velocity_penalty + angle_penalty +
                    shaping_bonus + engine_usage_penalty + terminal + crash_penalty)
    
    # Build components dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "shaping_bonus": shaping_bonus,
        "engine_usage_penalty": engine_usage_penalty,
        "terminal": terminal,
        "crash_penalty": crash_penalty,
    }
    
    return float(total_reward), components
