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
    speed = math.sqrt(vx * vx + vy * vy)
    # Stronger linear distance penalty: -1.0 * dist for strong gradient at all distances
    dist_penalty = -1.0 * dist
    # Speed penalty: penalize total speed and vertical speed separately
    speed_penalty = -0.5 * speed - 0.4 * abs(vy)
    # Extra vertical speed penalty when close to ground (y < 0.3)
    if y < 0.3:
        speed_penalty -= 1.0 * abs(vy)
    # Small positive reward for being close to pad with low speed (shaped landing bonus)
    landing_shaping = 0.0
    if dist < 0.2 and speed < 0.2:
        landing_shaping = 0.5 * (1.0 - dist / 0.2) * (1.0 - speed / 0.2)
    progress = dist_penalty + speed_penalty + landing_shaping
    progress = max(-5.0, min(0.5, progress))

    # --- Stability component: maintain upright orientation ---
    angle_penalty = -0.8 * abs(angle)
    ang_vel_penalty = -0.15 * abs(ang_vel)
    # When near ground (y < 0.3) and tilted significantly, add extra penalty
    if y < 0.3 and abs(angle) > 0.3:
        angle_penalty -= 0.6
    stability = angle_penalty + ang_vel_penalty
    stability = max(-3.0, min(0.0, stability))

    # --- Effort component: moderate penalty for engine usage ---
    # action: 0=do nothing, 1=left, 2=main, 3=right
    if action == 2:  # main engine
        effort = -0.12
    elif action in [1, 3]:  # side engines
        effort = -0.04
    else:
        effort = 0.0

    # --- Terminal component: success or failure ---
    terminal = 0.0
    if done:
        both_legs = (left_leg > 0.5 and right_leg > 0.5)
        near_pad = abs(x) < 0.12 and abs(y) < 0.08
        low_speed = abs(vx) < 0.1 and abs(vy) < 0.1
        upright = abs(angle) < 0.25

        if both_legs and near_pad and low_speed and upright:
            terminal = 100.0  # successful landing
        else:
            terminal = -15.0  # crash or out-of-bounds

    # Combine all components
    total_reward = progress + stability + effort + terminal
    total_reward = max(-1000.0, min(1000.0, total_reward))

    components = {
        "progress": progress,
        "stability": stability,
        "effort": effort,
        "terminal": terminal,
    }

    return float(total_reward), components
