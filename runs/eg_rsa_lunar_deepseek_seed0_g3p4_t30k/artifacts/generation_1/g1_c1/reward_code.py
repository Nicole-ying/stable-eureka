def compute_reward(obs, action, next_obs, done, info):
    # Unpack normalized observations
    x = obs[0]          # normalized x position, target 0
    y = obs[1]          # normalized y position, target ~0 (helipad)
    vx = obs[2]         # normalized x velocity
    vy = obs[3]         # normalized y velocity
    angle = obs[4]      # raw angle in radians, target 0
    ang_vel = obs[5]    # normalized angular velocity
    left_leg = obs[6]   # 0 or 1
    right_leg = obs[7]  # 0 or 1

    # --- Progress component: encourage moving toward landing pad (0,0) ---
    dist = math.sqrt(x * x + y * y)
    # Use a moderate negative distance penalty; exponential shaping for smooth gradient
    progress = -0.8 * dist

    # --- Stability component: encourage upright orientation and low angular velocity ---
    # Angle penalty: 0 when upright, -0.5 per radian (typical angle range ~[-π, π])
    angle_penalty = -0.5 * abs(angle)
    # Angular velocity penalty: encourage stable rotation
    ang_vel_penalty = -0.2 * abs(ang_vel)
    # When near ground (y < 0.2), add extra penalty for high speed to prevent crash
    near_ground = y < 0.2
    speed = math.sqrt(vx * vx + vy * vy)
    ground_speed_penalty = -0.6 * speed if near_ground else 0.0
    stability = angle_penalty + ang_vel_penalty + ground_speed_penalty
    # Clip stability to a reasonable range
    stability = max(-2.0, min(0.0, stability))

    # --- Effort component: penalize engine usage ---
    # action: 0=do nothing, 1=left, 2=main, 3=right
    main_engine = 1.0 if action == 2 else 0.0
    side_engine = 1.0 if action in [1, 3] else 0.0
    # Moderate penalties: -0.2 per main engine use, -0.05 per side engine use
    effort = -0.2 * main_engine - 0.05 * side_engine

    # --- Terminal component: reward successful landing or penalize crash ---
    terminal = 0.0
    if done:
        # Check for successful landing
        both_legs = (left_leg > 0.5 and right_leg > 0.5)
        near_pad = abs(x) < 0.1 and abs(y) < 0.1
        soft_landing = abs(vy) < 0.1 and abs(vx) < 0.1
        upright = abs(angle) < 0.2

        if both_legs and near_pad and soft_landing and upright:
            terminal = 100.0   # large success bonus
        else:
            terminal = -15.0   # moderate crash/out-of-bounds penalty

    # Combine all components
    total_reward = progress + stability + effort + terminal

    # Clamp total reward to safe bounds
    total_reward = max(-1000.0, min(1000.0, total_reward))

    components = {
        "progress": progress,
        "stability": stability,
        "effort": effort,
        "terminal": terminal,
    }

    return float(total_reward), components
