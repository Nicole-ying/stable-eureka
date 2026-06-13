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

    # --- Progress component: strong shaping toward landing pad (0,0) ---
    dist = math.sqrt(x * x + y * y)
    speed = math.sqrt(vx * vx + vy * vy)
    # Linear distance penalty for strong gradient at all distances
    dist_penalty = -1.0 * dist
    # Speed penalty: penalize high speed, especially vertical speed near ground
    speed_penalty = -0.5 * speed - 0.3 * abs(vy)
    if y < 0.3:
        speed_penalty -= 1.0 * abs(vy)
    # Small survival bonus to encourage longer flights and exploration
    survival_bonus = 0.05
    progress = dist_penalty + speed_penalty + survival_bonus
    # Keep progress bounded but allow negative values to dominate
    progress = max(-6.0, min(1.0, progress))

    # --- Stability component: maintain upright orientation ---
    angle_penalty = -0.8 * abs(angle) - 0.2 * abs(ang_vel)
    # Extra penalty for large angles near ground (dangerous)
    if y < 0.3 and abs(angle) > 0.2:
        angle_penalty -= 0.8
    stability = max(-4.0, min(0.0, angle_penalty))

    # --- Effort component: moderate penalty for engine usage ---
    # action: 0=do nothing, 1=left, 2=main, 3=right
    if action == 2:  # main engine
        effort = -0.10
    elif action in [1, 3]:  # side engines
        effort = -0.03
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

    # Clip to absolute bound
    total_reward = max(min(total_reward, 1000.0), -1000.0)

    components = {
        "progress": progress,
        "stability": stability,
        "effort": effort,
        "terminal": terminal,
    }

    return float(total_reward), components
