def compute_reward(obs, action, next_obs, done, info):
    # Extract current observations
    x = obs[0]
    y = obs[1]
    vel_x = obs[2]
    vel_y = obs[3]
    angle = obs[4]
    angular_vel = obs[5]
    leg_contact_0 = obs[6]
    leg_contact_1 = obs[7]

    # Extract next observations for progress shaping
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vel_y = next_obs[3]
    next_angle = next_obs[4]

    # Action: 0=do nothing, 1=left, 2=main, 3=right
    main_engine = 1.0 if action == 2 else 0.0
    side_engine = 1.0 if action in [1, 3] else 0.0

    # ========== distance_penalty ==========
    distance = (x * x + y * y) ** 0.5
    distance_penalty = -0.1 * distance

    # ========== velocity_penalty ==========
    velocity_penalty = -0.3 * abs(vel_y) - 0.1 * abs(vel_x)

    # ========== angle_penalty ==========
    angle_penalty = -0.2 * abs(angle)

    # ========== fuel_penalty ==========
    fuel_penalty = -0.02 * (main_engine + side_engine)

    # ========== progress ==========
    next_distance = (next_x * next_x + next_y * next_y) ** 0.5
    distance_improvement = distance - next_distance
    progress_reward = 0.5 * distance_improvement
    vel_improvement = abs(vel_y) - abs(next_vel_y)
    progress_reward += 0.3 * vel_improvement

    # ========== stability ==========
    stability_reward = 0.0
    if abs(angle) < 0.2:
        stability_reward += 0.1
    if abs(angular_vel) < 0.1:
        stability_reward += 0.05
    if leg_contact_0 or leg_contact_1:
        stability_reward += 0.2

    # ========== effort ==========
    # Penalize engine use when close to pad and slow; reward inaction when stable
    if distance < 0.3 and abs(vel_y) < 0.2:
        effort_penalty = -0.05 * (main_engine + side_engine)
    else:
        effort_penalty = 0.0

    # ========== terminal shaping ==========
    terminal_reward = 0.0
    if done:
        both_legs = bool(leg_contact_0) and bool(leg_contact_1)
        upright = abs(angle) < 0.1
        low_speed = abs(vel_y) < 0.1
        if both_legs and upright and low_speed:
            terminal_reward = 100.0
        elif abs(vel_y) > 0.5 or abs(angle) > 0.5:
            terminal_reward = -10.0
        else:
            terminal_reward = -5.0

    # ========== landing_bonus ==========
    landing_bonus = 0.0
    both_legs = bool(leg_contact_0) and bool(leg_contact_1)
    upright = abs(angle) < 0.1
    low_speed = abs(vel_y) < 0.1
    if both_legs and upright and low_speed:
        landing_bonus = 100.0

    # ========== crash_penalty ==========
    crash_penalty = 0.0
    if done and (abs(vel_y) > 0.5 or abs(angle) > 0.5):
        crash_penalty = -10.0

    # ========== total reward ==========
    total_reward = (
        distance_penalty +
        velocity_penalty +
        angle_penalty +
        fuel_penalty +
        progress_reward +
        stability_reward +
        effort_penalty +
        terminal_reward +
        landing_bonus +
        crash_penalty
    )

    total_reward = max(-1000.0, min(1000.0, total_reward))

    components = {
        "landing_bonus": landing_bonus,
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "fuel_penalty": fuel_penalty,
        "crash_penalty": crash_penalty,
        "progress": progress_reward,
        "stability": stability_reward,
        "effort": effort_penalty,
        "terminal": terminal_reward
    }

    return float(total_reward), components