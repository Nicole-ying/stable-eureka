def compute_reward(obs, action, next_obs, done, info):
    # Extract relevant quantities from next_obs
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # Distance penalty: encourage moving toward the pad at (0,0)
    distance = np.sqrt(x**2 + y**2)
    distance_penalty = -distance
    
    # Velocity penalty: encourage gentle landing speed
    velocity = np.sqrt(vx**2 + vy**2)
    velocity_penalty = -velocity
    
    # Angle penalty: encourage upright orientation
    angle_penalty = -abs(angle)
    
    # Engine usage penalty: discourage fuel consumption
    # Main engine (action==2) costs fuel
    engine_usage_penalty = -0.3 if action == 2 else 0.0
    
    # Terminal reward: successful landing
    # Conditions: both legs on ground, close to pad, upright, low vertical speed
    landing_condition = (
        left_contact > 0.5 and 
        right_contact > 0.5 and 
        distance < 0.1 and 
        abs(angle) < 0.1 and 
        abs(vy) < 0.1
    )
    terminal = 100.0 if landing_condition else 0.0
    
    # Sum up all components
    total_reward = distance_penalty + velocity_penalty + angle_penalty + engine_usage_penalty + terminal
    
    # Build components dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "engine_usage_penalty": engine_usage_penalty,
        "terminal": terminal,
    }
    
    return float(total_reward), components
