def compute_reward(self, state, m_power, s_power, terminated):
    # State unpacking
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = state[4]
    leg_l = state[6]
    leg_r = state[7]

    # Diagnostics calculation
    dist_to_pad = abs(x)
    velocity_mag = (vx**2 + vy**2)**0.5
    fuel_cost = m_power + s_power
    is_contact = 1.0 if (leg_l == 1.0 and leg_r == 1.0) else 0.0

    # Success Condition: Terminated AND Legs Contact AND Near Center AND Stable Angle
    safe_landing = (terminated and 
                    leg_l == 1.0 and leg_r == 1.0 and 
                    abs(x) < 0.5 and 
                    abs(angle) < 0.2)

    # Failure Condition: Terminated AND NOT Safe Landing
    crash_or_fail = terminated and not safe_landing

    reward = 0.0
    
    if safe_landing:
        reward += 300.0
    elif crash_or_fail:
        reward -= 200.0
        
    # Step-wise shaping (only if not terminated)
    if not terminated:
        reward -= 0.1 * dist_to_pad
        reward -= 0.2 * velocity_mag
        reward -= 0.1 * abs(angle)
        
    # Regularizers (Fuel)
    reward -= 0.1 * fuel_cost

    individual_reward = {
        "dist_to_pad": dist_to_pad,
        "velocity_magnitude": velocity_mag,
        "fuel_consumption": fuel_cost,
        "leg_contact_status": is_contact,
        "angle_deviation": abs(angle),
        "is_safe_landing": 1.0 if safe_landing else 0.0,
        "is_crash_or_fail": 1.0 if crash_or_fail else 0.0
    }

    return float(reward), individual_reward