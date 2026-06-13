def compute_reward(obs, action, next_obs, done, info):
    # Unpack observations
    x, y = obs[0], obs[1]                # normalized position
    vx, vy = obs[2], obs[3]              # normalized velocity
    angle = obs[4]                        # raw angle in radians
    ang_vel = obs[5]                      # normalized angular velocity
    left_leg = obs[6]                     # 0 or 1
    right_leg = obs[7]                    # 0 or 1

    # Next obs for distance change (delta shaping)
    next_x, next_y = next_obs[0], next_obs[1]

    # --- Progress component: encourage moving toward landing pad (0,0) ---
    dist = math.sqrt(x * x + y * y)
    next_dist = math.sqrt(next_x * next_x + next_y * next_y)
    # Delta shaping: reward reduction in distance
    progress_delta = (next_dist - dist) * 5.0  # positive when moving closer
    progress_delta = max(-1.0, min(1.0, progress_delta))
    # Also add a small negative distance penalty to keep the signal dense
    dist_penalty = -0.3 * dist
    progress_reward = progress_delta + dist_penalty

    # --- Stability component: encourage upright orientation, low angular velocity, and gentle descent ---
    angle_penalty = -0.5 * abs(angle)
    ang_vel_penalty = -0.15 * abs(ang_vel)
    # Extra penalty for high vertical speed near the ground (y close to 0)
    ground_proximity_penalty = 0.0
    if y < 0.2:
        ground_proximity_penalty = -1.0 * max(0.0, abs(vy) - 0.1)
    stability_reward = angle_penalty + ang_vel_penalty + ground_proximity_penalty
    stability_reward = max(-2.0, min(0.0, stability_reward))

    # --- Effort component: penalize engine usage moderately ---
    # action: 0=do nothing, 1=left, 2=main, 3=right
    main_engine = 1.0 if action == 2 else 0.0
    side_engine = 1.0 if action in [1, 3] else 0.0
    effort_reward = -0.15 * main_engine - 0.05 * side_engine

    # --- Terminal component: reward successful landing or penalize crash/failure ---
    terminal_reward = 0.0
    if done:
        # Success condition: both legs on ground, near pad, low velocity, upright
        both_legs = (left_leg > 0.5 and right_leg > 0.5)
        near_pad = abs(x) < 0.1 and y < 0.05
        low_speed = abs(vx) < 0.1 and abs(vy) < 0.1
        upright = abs(angle) < 0.2

        if both_legs and near_pad and low_speed and upright:
            terminal_reward = 100.0
        else:
            # Moderate failure penalty (not too harsh)
            terminal_reward = -25.0

    # Combine all components
    total_reward = progress_reward + stability_reward + effort_reward + terminal_reward

    # Clamp total reward to safe bounds
    total_reward = max(-1000.0, min(1000.0, total_reward))

    components = {
        "progress": progress_reward,
        "stability": stability_reward,
        "effort": effort_reward,
        "terminal": terminal_reward,
    }

    return float(total_reward), components
