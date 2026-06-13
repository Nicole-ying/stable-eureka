def compute_reward(obs, action, next_obs, done, info):
    # Extract relevant quantities from next_obs
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # Distance penalty: use squared distance with scaling for stronger gradient near the pad
    distance = np.sqrt(x**2 + y**2)
    distance_penalty = -10.0 * distance**2  # Stronger penalty far away, smoother near pad
    
    # Velocity penalty: penalize speed quadratically to encourage slowing down
    velocity = np.sqrt(vx**2 + vy**2)
    velocity_penalty = -5.0 * velocity**2  # Quadratic penalty for high speed
    
    # Angle penalty: penalize deviation from upright orientation
    angle_penalty = -2.0 * abs(angle)  # Linear penalty, moderate weight
    
    # Engine usage penalty: discourage all engine firings (fuel efficiency)
    # Action 2 is main engine (costly), actions 1 and 3 are side engines (less costly)
    if action == 2:
        engine_usage_penalty = -0.5  # Main engine is expensive
    elif action == 1 or action == 3:
        engine_usage_penalty = -0.1  # Side engines are cheaper
    else:
        engine_usage_penalty = 0.0   # No engine firing
    
    # Terminal reward: successful landing
    landing_condition = (
        left_contact > 0.5 and 
        right_contact > 0.5 and 
        distance < 0.15 and  # Slightly relaxed distance threshold
        abs(angle) < 0.15 and  # Slightly relaxed angle threshold
        abs(vy) < 0.15  # Slightly relaxed vertical velocity threshold
    )
    
    # Crash penalty: large negative reward when episode ends without successful landing
    # This helps the agent avoid crashing
    crash_penalty = 0.0
    if done and not landing_condition:
        crash_penalty = -100.0  # Strong penalty for crashing
    
    terminal = 150.0 if landing_condition else 0.0  # Increased terminal reward
    
    # Sum up all components
    total_reward = distance_penalty + velocity_penalty + angle_penalty + engine_usage_penalty + terminal + crash_penalty
    
    # Build components dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "engine_usage_penalty": engine_usage_penalty,
        "terminal": terminal + crash_penalty,  # Combine terminal outcomes into one component
    }
    
    return float(total_reward), components
