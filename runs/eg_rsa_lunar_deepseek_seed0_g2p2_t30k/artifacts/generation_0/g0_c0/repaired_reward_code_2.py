import math

def compute_reward(obs, action, next_obs, done, info):
    # Extract observations
    x, y, vel_x, vel_y, angle, angular_vel, leg_0, leg_1 = obs
    
    # Extract next observations for terminal checks
    if next_obs is not None:
        nx, ny, nvel_x, nvel_y, nangle, nangular_vel, nleg_0, nleg_1 = next_obs
    else:
        nx, ny, nvel_x, nvel_y, nangle, nangular_vel, nleg_0, nleg_1 = obs
    
    # Extract action info from info dict if available (m_power, s_power)
    m_power = info.get('m_power', 0.0) if isinstance(info, dict) else 0.0
    s_power = info.get('s_power', 0.0) if isinstance(info, dict) else 0.0
    
    # If info is not a dict or missing keys, infer from action
    if m_power == 0.0 and s_power == 0.0:
        if action == 2:
            m_power = 1.0
        elif action in [1, 3]:
            s_power = 1.0
    
    # ---- Component Calculations ----
    
    # 1. landing_bonus: large positive if both legs contact, upright, low velocity, and done
    landing_bonus = 0.0
    if done and nleg_0 == 1.0 and nleg_1 == 1.0:
        if abs(nangle) < 0.1 and abs(nvel_y) < 0.1:
            landing_bonus = 100.0
        else:
            # Not a perfect landing, but legs are down - could be a crash or tilted landing
            landing_bonus = -50.0  # Penalize bad landings
    
    # 2. distance_penalty: penalize distance from pad (0,0)
    dist = math.sqrt(x*x + y*y)
    distance_penalty = -2.0 * dist
    
    # 3. velocity_penalty: penalize high velocities, especially vertical
    vel_pen = -1.0 * abs(vel_x) - 3.0 * abs(vel_y)
    
    # 4. angle_penalty: penalize tilt
    angle_pen = -2.0 * abs(angle)
    
    # 5. fuel_penalty: penalize engine usage
    fuel_pen = -0.5 * (m_power + s_power)
    
    # 6. crash_penalty: large negative if terminated with high velocity or angle (crash)
    crash_penalty = 0.0
    if done:
        # Check if it's a crash (not a good landing)
        is_good_landing = (nleg_0 == 1.0 and nleg_1 == 1.0 and abs(nangle) < 0.1 and abs(nvel_y) < 0.1)
        if not is_good_landing:
            # Check for clear crash indicators
            if abs(nvel_y) > 0.5 or abs(nangle) > 0.5 or abs(nx) >= 1.0:
                crash_penalty = -200.0
            else:
                # Early termination without crash (e.g., asleep)
                crash_penalty = -50.0
    
    # 7. progress: dense shaping for moving toward pad and reducing speed
    # Reward reduction in distance and velocity
    progress = 0.0
    if next_obs is not None:
        next_dist = math.sqrt(nx*nx + ny*ny)
        dist_change = dist - next_dist  # positive if moving closer
        progress = 5.0 * dist_change
    
    # 8. stability: reward for being upright and having low angular velocity
    stability = -0.5 * abs(angular_vel)  # penalize spinning
    
    # 9. effort: bounded penalty for unnecessary actions (fuel usage already penalized)
    # Also penalize doing nothing when far and high (should act)
    effort = 0.0
    if action == 0 and dist > 0.5:
        effort = -0.2  # small penalty for inaction when far
    
    # 10. terminal: bounded terminal shaping from done signal
    terminal = 0.0
    if done:
        if nleg_0 == 1.0 and nleg_1 == 1.0 and abs(nangle) < 0.1 and abs(nvel_y) < 0.1:
            terminal = 50.0  # extra bonus for successful landing
        else:
            terminal = -10.0  # penalty for termination without success
    
    # Total reward
    total_reward = (landing_bonus + distance_penalty + vel_pen + angle_pen + 
                    fuel_pen + crash_penalty + progress + stability + effort + terminal)
    
    # Clamp total reward to [-1000, 1000] as per schema
    total_reward = max(-1000.0, min(1000.0, total_reward))
    
    # Components dict
    components = {
        'landing_bonus': landing_bonus,
        'distance_penalty': distance_penalty,
        'velocity_penalty': vel_pen,
        'angle_penalty': angle_pen,
        'fuel_penalty': fuel_pen,
        'crash_penalty': crash_penalty,
        'progress': progress,
        'stability': stability,
        'effort': effort,
        'terminal': terminal
    }
    
    return float(total_reward), components