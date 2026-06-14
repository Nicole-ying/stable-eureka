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

    # --- Active Term: Step Penalty (Time Pressure) ---
    step_penalty_val = -0.2
    reward += step_penalty_val
    individual_reward['step_penalty'] = float(step_penalty_val)

    # --- Active Term: Fuel Penalty ---
    fuel_cost = m_power + s_power
    reward -= fuel_cost * 1.0
    individual_reward['fuel_cost'] = float(fuel_cost)

    # --- Active Term: Orientation Control ---
    angle_penalty = abs(angle) * 0.5
    reward -= angle_penalty
    individual_reward['orientation_penalty'] = float(angle_penalty)

    # --- Termination Handling (Success vs Failure) ---
    if terminated:
        # Check for successful landing conditions based on state inference
        leg_contact = (leg_l == 1.0 or leg_r == 1.0)
        low_velocity = (abs(vx) < 0.5 and abs(vy) < 0.5)
        near_pad = (abs(x) < 0.5)

        if leg_contact and low_velocity and near_pad:
            # Success: Large positive reward
            reward += 100.0
            individual_reward['landing_success'] = 1.0
        else:
            # Failure: Crash, Out of Bounds, or Bad Landing
            reward -= 100.0
            individual_reward['landing_success'] = 0.0
        
        return float(reward), individual_reward

    # --- Diagnostic Term: Leg Contact Status ---
    leg_contact_status = max(leg_l, leg_r)
    individual_reward['leg_contact_status'] = float(leg_contact_status)

    return float(reward), individual_reward