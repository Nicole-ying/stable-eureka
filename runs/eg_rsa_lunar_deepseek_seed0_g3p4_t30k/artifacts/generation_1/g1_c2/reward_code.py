def compute_reward(obs, action, next_obs, done, info):
    # Unpack observations
    x = obs[0]          # normalized x position, target 0
    y = obs[1]          # normalized y position (relative to helipad)
    vx = obs[2]         # normalized x velocity
    vy = obs[3]         # normalized y velocity
    angle = obs[4]      # raw angle in radians, target 0
    ang_vel = obs[5]    # normalized angular velocity
    left_leg = obs[6]   # left leg contact (0 or 1)
    right_leg = obs[7]  # right leg contact (0 or 1)

    # Distance to landing pad (origin)
    dist = math.sqrt(x * x + y * y)
    # Speed magnitude
    speed = math.sqrt(vx * vx + vy * vy)

    # --- Progress component: dense shaping toward pad with soft landing ---
    # Exponential distance penalty: strongest far away, gentle near pad
    dist_penalty = -0.8 * dist
    # Speed penalty: penalize high speed, especially vertical speed near ground
    speed_penalty = -0.5 * speed - 0.3 * abs(vy)
    # Additional vertical speed penalty when close to ground
    if y < 0.3:
        speed_penalty -= 0.8 * abs(vy)
    progress_reward = dist_penalty + speed_penalty
    # Bound to reasonable range
    progress_reward = max(-5.0, min(0.0, progress_reward))

    # --- Stability component: maintain upright orientation ---
    angle_penalty = -0.6 * abs(angle) - 0.15 * abs(ang_vel)
    # Extra penalty for large angles near ground (dangerous)
    if y < 0.3 and abs(angle) > 0.3:
        angle_penalty -= 0.5
    stability_reward = max(-3.0, min(0.0, angle_penalty))

    # --- Effort component: moderate penalty for engine usage ---
    # action: 0=do nothing, 1=left, 2=main, 3=right
    if action == 2:  # main engine
        effort_reward = -0.15
    elif action in [1, 3]:  # side engines
        effort_reward = -0.05
    else:
        effort_reward = 0.0

    # --- Terminal component: success or failure ---
    terminal_reward = 0.0
    if done:
        both_legs = (left_leg > 0.5 and right_leg > 0.5)
        near_pad = abs(x) < 0.12 and abs(y) < 0.08
        low_speed = abs(vx) < 0.1 and abs(vy) < 0.1
        upright = abs(angle) < 0.25

        if both_legs and near_pad and low_speed and upright:
            terminal_reward = 100.0  # successful landing
        else:
            terminal_reward = -15.0  # crash or out-of-bounds

    # Combine all components
    total_reward = progress_reward + stability_reward + effort_reward + terminal_reward

    # Clip to absolute bound
    total_reward = max(min(total_reward, 1000.0), -1000.0)

    components = {
        "progress": progress_reward,
        "stability": stability_reward,
        "effort": effort_reward,
        "terminal": terminal_reward,
    }

    return float(total_reward), components
