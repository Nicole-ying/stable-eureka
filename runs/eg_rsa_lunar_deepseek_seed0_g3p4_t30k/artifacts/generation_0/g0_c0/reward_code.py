def compute_reward(obs, action, next_obs, done, info):
    # Extract state components (normalized as per environment)
    x = obs[0]  # normalized x position
    y = obs[1]  # normalized y position (relative to helipad)
    x_vel = obs[2]  # normalized x velocity
    y_vel = obs[3]  # normalized y velocity
    angle = obs[4]  # raw angle in radians
    ang_vel = obs[5]  # normalized angular velocity
    left_leg = obs[6]  # left leg contact
    right_leg = obs[7]  # right leg contact

    # Distance to landing pad (origin in normalized coordinates)
    dist = math.sqrt(x * x + y * y)

    # Linear velocity magnitude
    speed = math.sqrt(x_vel * x_vel + y_vel * y_vel)

    # --- Progress component: encourage moving toward pad and reducing speed ---
    # Negative distance penalty, scaled to be moderate
    progress_reward = -0.5 * dist - 0.3 * speed

    # --- Stability component: encourage upright orientation and low angular velocity ---
    angle_penalty = -0.4 * abs(angle) - 0.1 * abs(ang_vel)
    # Also penalize high speed when close to ground (y near 0) to prevent crash
    if y < 0.2:
        angle_penalty -= 0.5 * speed
    stability_reward = angle_penalty

    # --- Effort component: penalize main engine usage (action 2) ---
    # action is discrete: 0=do nothing, 1=left, 2=main, 3=right
    fuel_penalty = -0.3 if action == 2 else 0.0
    # Small penalty for side engines to encourage smooth control
    if action in [1, 3]:
        fuel_penalty -= 0.1
    effort_reward = fuel_penalty

    # --- Terminal component: reward successful landing or penalize crash ---
    terminal_reward = 0.0
    if done:
        # Check for successful landing: both legs contact, near pad, low vertical speed
        both_legs = (left_leg > 0.5 and right_leg > 0.5)
        near_pad = abs(x) < 0.1 and y < 0.05
        soft_landing = abs(y_vel) < 0.1
        if both_legs and near_pad and soft_landing:
            terminal_reward = 100.0  # large success bonus
        else:
            terminal_reward = -10.0  # crash or out-of-bounds penalty

    # Combine all components
    total_reward = progress_reward + stability_reward + effort_reward + terminal_reward

    # Clip total reward to bound
    total_reward = max(min(total_reward, 1000.0), -1000.0)

    components = {
        "progress": progress_reward,
        "stability": stability_reward,
        "effort": effort_reward,
        "terminal": terminal_reward,
    }

    return total_reward, components
