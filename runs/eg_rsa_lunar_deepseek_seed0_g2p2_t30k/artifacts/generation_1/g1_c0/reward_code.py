def compute_reward(obs, action, next_obs, done, info):
    # Unpack observation components (normalized)
    x = obs[0]
    y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    # --- Component 1: velocity_penalty ---
    # Penalize high speeds, especially vertical velocity (negative = moving down fast)
    # Scale reduced from parent to avoid overwhelming other signals
    vel_penalty = -1.5 * (abs(vy) + 0.3 * abs(vx))
    
    # --- Component 2: angle_penalty ---
    # Penalize tilt and spin, moderate magnitude
    angle_penalty = -2.0 * (abs(angle) + 0.5 * abs(ang_vel))
    
    # --- Component 3: distance_reward ---
    # Exponential shaping: near-zero far away, increasingly positive as agent approaches pad
    # This replaces the harsh linear negative distance with a smoother signal
    dist = np.sqrt(x**2 + y**2)
    # exp(-2*dist) gives ~1 when dist=0, ~0.14 when dist=1, ~0.02 when dist=2
    # Scale by 3 to make it comparable with other components
    distance_reward = 3.0 * np.exp(-2.0 * dist)
    
    # --- Component 4: fuel_efficiency_penalty ---
    # Increased penalties to discourage wasteful engine firing
    fuel_penalty = 0.0
    if action == 2:  # main engine
        fuel_penalty = -2.0
    elif action == 1 or action == 3:  # side engines
        fuel_penalty = -1.0
    
    # --- Component 5: ground_contact_bonus ---
    # Relaxed conditions: at least one leg contact and near pad
    # Also give smaller bonus for being near pad with one leg (partial progress)
    near_pad = (abs(x) < 0.3) and (y < 0.2)
    one_leg = (left_contact > 0.5) or (right_contact > 0.5)
    both_legs = (left_contact > 0.5) and (right_contact > 0.5)
    
    ground_contact_bonus = 0.0
    if both_legs and near_pad:
        ground_contact_bonus = 8.0  # full success posture
    elif one_leg and near_pad:
        ground_contact_bonus = 3.0  # partial progress
    
    # --- Component 6: terminal ---
    terminal_reward = 0.0
    if done:
        # Check for successful landing: both legs on ground, near pad, low speed, upright
        success = (
            both_legs and
            near_pad and
            abs(vy) < 0.15 and
            abs(vx) < 0.15 and
            abs(angle) < 0.15
        )
        if success:
            terminal_reward = 100.0
        else:
            # Reduced failure penalty to avoid masking shaping signals
            terminal_reward = -50.0
    
    # Sum all components
    total_reward = vel_penalty + angle_penalty + distance_reward + fuel_penalty + ground_contact_bonus + terminal_reward
    
    # Clamp total reward to absolute bound
    total_reward = np.clip(total_reward, -1000.0, 1000.0)
    
    # Build components dict
    components_dict = {
        "velocity_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "distance_reward": distance_reward,
        "fuel_efficiency_penalty": fuel_penalty,
        "ground_contact_bonus": ground_contact_bonus,
        "terminal": terminal_reward
    }
    
    return float(total_reward), components_dict
