def compute_reward(obs, action, next_obs, done, info):
    # Unpack normalized observations
    x = obs[0]                # normalized x position
    y = obs[1]                # normalized y position
    vx = obs[2]               # normalized x velocity
    vy = obs[3]               # normalized y velocity
    angle = obs[4]            # raw angle in radians
    ang_vel = obs[5]          # normalized angular velocity
    left_leg = obs[6]         # 0 or 1
    right_leg = obs[7]        # 0 or 1

    # --- Progress: encourage moving toward landing pad (0,0) ---
    dist = math.sqrt(x * x + y * y)
    # Use exponential shaping so progress is bounded and smooth
    progress_reward = -0.8 * (1.0 - math.exp(-2.0 * dist))

    # --- Stability: encourage upright orientation and low angular velocity ---
    # Angle penalty grows quickly for large angles
    angle_penalty = -0.6 * min(abs(angle), 1.0)
    ang_vel_penalty = -0.3 * min(abs(ang_vel), 1.0)
    # Extra penalty for high speed close to ground (danger zone)
    ground_proximity = max(0.0, 1.0 - y / 0.3)  # 0 when high, 1 when at ground
    speed = math.sqrt(vx * vx + vy * vy)
    rough_landing_penalty = -0.8 * ground_proximity * min(speed, 1.0)
    stability_reward = angle_penalty + ang_vel_penalty + rough_landing_penalty
    # Clip stability to [-2.0, 0.0] to keep it bounded
    stability_reward = max(-2.0, min(0.0, stability_reward))

    # --- Effort: moderate penalty for engine usage ---
    # action: 0=do nothing, 1=left, 2=main, 3=right
    main_engine = 1.0 if action == 2 else 0.0
    side_engine = 1.0 if action in [1, 3] else 0.0
    effort_reward = -0.15 * main_engine - 0.05 * side_engine

    # --- Terminal: success bonus or failure penalty ---
    terminal_reward = 0.0
    if done:
        # Success condition: both legs on ground, near pad, low speed, upright
        both_legs = (left_leg > 0.5 and right_leg > 0.5)
        near_pad = (abs(x) < 0.12 and y < 0.1)
        low_speed = (abs(vx) < 0.1 and abs(vy) < 0.1)
        upright = abs(angle) < 0.25
        if both_legs and near_pad and low_speed and upright:
            terminal_reward = 100.0
        else:
            # Moderate failure penalty to avoid dominating shaping signals
            terminal_reward = -15.0

    # Combine all components
    total_reward = progress_reward + stability_reward + effort_reward + terminal_reward

    # Clamp to safe bounds
    total_reward = max(-1000.0, min(1000.0, total_reward))

    components = {
        "progress": progress_reward,
        "stability": stability_reward,
        "effort": effort_reward,
        "terminal": terminal_reward,
    }

    return float(total_reward), components
