def compute_reward(self, state, m_power, s_power, terminated):
    # Extract state components
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = state[4]
    leg_l = state[6]
    leg_r = state[7]

    reward = 0.0
    individual_reward = {}

    # --- Active Term: Step Penalty (Increased Weight -0.5) ---
    step_penalty_val = -0.5
    reward += step_penalty_val
    individual_reward['step_penalty'] = float(step_penalty_val)

    # --- Active Term: Fuel Penalty (Maintained Weight 0.5) ---
    fuel_cost = m_power + s_power
    fuel_penalty_weight = 0.5
    reward -= fuel_cost * fuel_penalty_weight
    individual_reward['fuel_cost'] = float(fuel_cost)
    individual_reward['fuel_penalty'] = float(-fuel_cost * fuel_penalty_weight)

    # --- Active Term: Orientation Control (Maintained Weight 1.0) ---
    angle_penalty_weight = 1.0
    angle_penalty_val = abs(angle) * angle_penalty_weight
    reward -= angle_penalty_val
    individual_reward['orientation_penalty'] = float(angle_penalty_val)

    # --- Active Term: Position Shaping (X Alignment, Weight 0.1) ---
    position_shaping_weight = 0.1
    x_dist = abs(x)
    position_shaping_val = -x_dist * position_shaping_weight
    reward += position_shaping_val
    individual_reward['position_shaping'] = float(position_shaping_val)

    # --- Active Term: Velocity Stability (Thresholded) ---
    vel_mag = (vx**2 + vy**2)**0.5
    velocity_threshold = 0.6
    velocity_penalty_weight = 1.0
    if vel_mag > velocity_threshold:
        velocity_stability_val = -((vel_mag - velocity_threshold)**2) * velocity_penalty_weight
        reward += velocity_stability_val
    else:
        velocity_stability_val = 0.0
    individual_reward['velocity_stability'] = float(velocity_stability_val)

    # --- State-Based Success Detection (Tightened Velocity Threshold to 0.4) ---
    leg_contact = (leg_l == 1.0 or leg_r == 1.0)
    low_velocity = (abs(vx) < 0.4 and abs(vy) < 0.4)
    near_pad = (abs(x) < 0.5)
    is_success_state = leg_contact and low_velocity and near_pad

    # --- Sparse Success Bonus (Only on Termination) ---
    if terminated and is_success_state:
        reward += 100.0
        individual_reward['landing_success'] = 1.0
    else:
        individual_reward['landing_success'] = float(is_success_state)

    # --- Termination Handling (Failure Penalty) ---
    if terminated and not is_success_state:
        reward -= 100.0
        individual_reward['termination_failure_penalty'] = -100.0
    else:
        individual_reward['termination_failure_penalty'] = 0.0

    # --- Diagnostic Term: Leg Contact Status ---
    leg_contact_status = max(leg_l, leg_r)
    individual_reward['leg_contact_status'] = float(leg_contact_status)

    return float(reward), individual_reward