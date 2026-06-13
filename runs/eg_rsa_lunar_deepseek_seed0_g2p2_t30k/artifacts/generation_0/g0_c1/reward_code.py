def compute_reward(obs, action, next_obs, done, info):
    # Unpack observation components (normalized)
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angular_vel = obs[5]
    leg_left = obs[6]
    leg_right = obs[7]

    # ---------- 1. Velocity Penalty ----------
    # Penalize high vertical speed (especially negative = moving down) and horizontal speed.
    # Scale: 0-10 range roughly.
    vel_penalty = 3.0 * abs(vy) + 1.0 * abs(vx)

    # ---------- 2. Angle Penalty ----------
    # Penalize tilt and spin. Angle is in radians, angular_vel is scaled by 20/FPS.
    angle_penalty = 2.0 * abs(angle) + 1.0 * abs(angular_vel)

    # ---------- 3. Distance Reward ----------
    # Negative Euclidean distance from (x,y) to (0,0).
    # The space is roughly [-1,1] in x and [-0.5,1.5] in y, so typical distances < 2.
    dist = np.sqrt(x**2 + y**2)
    distance_reward = -4.0 * dist  # negative, so maximizing means closer to pad

    # ---------- 4. Fuel Efficiency Penalty ----------
    # Penalize each engine firing.
    fuel_penalty = 0.0
    if action == 2:  # main engine
        fuel_penalty += 0.5
    if action in [1, 3]:  # side engines
        fuel_penalty += 0.3

    # ---------- 5. Ground Contact Bonus ----------
    # Only give bonus if both legs contact ground AND lander is near the pad (|x|<0.2, y near 0).
    # Use next_obs for contact status after step.
    next_leg_left = next_obs[6]
    next_leg_right = next_obs[7]
    near_pad = (abs(next_obs[0]) < 0.2) and (next_obs[1] < 0.1)
    ground_bonus = 0.0
    if (next_leg_left > 0.5) and (next_leg_right > 0.5) and near_pad:
        ground_bonus = 5.0

    # ---------- 6. Terminal Component ----------
    terminal_reward = 0.0
    if done:
        # Check if it's a successful landing: both legs on ground, near pad, low speed
        successful = (
            (next_leg_left > 0.5) and (next_leg_right > 0.5) and
            (abs(next_obs[0]) < 0.2) and (next_obs[1] < 0.1) and
            (abs(next_obs[3]) < 0.1)  # vertical speed low
        )
        if successful:
            terminal_reward = 100.0
        else:
            # Crash or out-of-bounds
            terminal_reward = -100.0

    # ---------- Total Reward ----------
    total_reward = (
        -vel_penalty
        - angle_penalty
        + distance_reward
        - fuel_penalty
        + ground_bonus
        + terminal_reward
    )

    # Clamp total to absolute bound
    total_reward = np.clip(total_reward, -1000.0, 1000.0)

    components = {
        "velocity_penalty": -vel_penalty,
        "angle_penalty": -angle_penalty,
        "distance_reward": distance_reward,
        "fuel_efficiency_penalty": -fuel_penalty,
        "ground_contact_bonus": ground_bonus,
        "terminal": terminal_reward,
    }

    return float(total_reward), components
