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

    # --- Active Term: Fuel Penalty ---
    fuel_cost = m_power + s_power
    reward -= fuel_cost * 1.0
    individual_reward['fuel_cost'] = float(fuel_cost)

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

    # --- Active Term: Position Shaping (Exploration) ---
    distance_to_pad = abs(x) + abs(y)
    reward -= distance_to_pad * 0.1
    individual_reward['distance_to_pad'] = float(distance_to_pad)

    # --- Active Term: Velocity Stability ---
    velocity_magnitude = abs(vx) + abs(vy)
    reward -= velocity_magnitude * 0.1
    individual_reward['velocity_magnitude'] = float(velocity_magnitude)

    # --- Diagnostic Term: Leg Contact Status ---
    leg_contact_status = max(leg_l, leg_r)
    individual_reward['leg_contact_status'] = float(leg_contact_status)

    return float(reward), individual_reward