def compute_reward(self, state, m_power, s_power, terminated):
    import math
    
    # Component diagnostics dictionary for training aggregation
    individual_reward = {}
    total_reward = 0.0
    
    # --- Distance to pad shaping (state[0] should approach 0) ---
    distance_to_pad = -abs(state[0]) * 10.0
    if not terminated:
        total_reward += distance_to_pad
    individual_reward['distance_to_pad'] = float(distance_to_pad)
    
    # --- Velocity control (penalize high linear velocities) ---
    velocity_penalty = -(abs(state[2]) + abs(state[3])) * 5.0
    if not terminated:
        total_reward += velocity_penalty
    individual_reward['velocity_control'] = float(velocity_penalty)
    
    # --- Angle control (encourage upright orientation near pad) ---
    angle_abs = abs(state[4]) % math.pi
    angle_penalty = -angle_abs * 20.0 if not terminated else 0.0
    total_reward += angle_penalty
    individual_reward['angle_control'] = float(angle_penalty)
    
    # --- Angular velocity control (penalize high rotation rates) ---
    angular_velocity_penalty = -abs(state[5]) * 10.0 if not terminated else 0.0
    total_reward += angular_velocity_penalty
    individual_reward['angular_velocity_control'] = float(angular_velocity_penalty)
    
    # --- Leg contact verification (only matters at termination) ---
    leg_contact_score = max(state[6], state[7]) if terminated else 0.0
    individual_reward['leg_contact'] = float(leg_contact_score)
    
    # --- Fuel cost penalty (continuous consumption) ---
    fuel_cost_penalty = -(m_power + s_power) * 25.0
    total_reward += fuel_cost_penalty
    individual_reward['fuel_cost'] = float(fuel_cost_penalty)
    
    # --- Terminal success bonus (verified landing on pad) ---
    terminal_success_bonus = 0.0
    if terminated:
        horizontal_proximity = abs(state[0]) < 0.15
        legs_touched = state[6] == 1.0 or state[7] == 1.0
        velocity_safe = (abs(state[2]) + abs(state[3])) < 4.0
        angle_upright = abs(state[4]) % math.pi < math.pi / 4
        angular_velocity_low = abs(state[5]) < 0.6
        
        if horizontal_proximity and legs_touched:
            terminal_success_bonus = 180.0 * (1.0 - min(abs(state[0]), 0.3) / 0.3)
    total_reward += terminal_success_bonus
    individual_reward['terminal_success_bonus'] = float(terminal_success_bonus)
    
    # --- Unsafe termination penalty (crash or out of bounds) ---
    unsafe_terminal_penalty = 0.0
    if terminated:
        horizontal_out_of_bounds = abs(state[0]) >= 1.0
        legs_not_touched_on_pad = not (state[6] == 1.0 or state[7] == 1.0)
        
        # Heavy penalty for crash termination (game_over flag in step.py sets terminated=True when game_over is True)
        if horizontal_out_of_bounds:
            unsafe_terminal_penalty -= 250.0
        elif legs_not_touched_on_pad and abs(state[0]) > 0.15:
            # Sleeping off-pad without leg contact = sleep hacking attempt
            unsafe_terminal_penalty -= 300.0
    total_reward += unsafe_terminal_penalty
    individual_reward['unsafe_terminal'] = float(unsafe_terminal_penalty)
    
    # --- Time pressure (implicit via fuel cost, no explicit step penalty) ---
    time_pressure_component = 0.0
    individual_reward['time_pressure'] = float(time_pressure_component)
    
    # --- Success-like terminal diagnostic flag ---
    success_like_terminal_flag = 1.0 if terminated and abs(state[0]) < 0.15 and (state[6] == 1.0 or state[7] == 1.0) else 0.0
    individual_reward['success_like_terminal'] = float(success_like_terminal_flag)
    
    # --- Out of bounds diagnostic flag ---
    out_of_bounds_flag = 1.0 if terminated and abs(state[0]) >= 1.0 else 0.0
    individual_reward['out_of_bounds'] = float(out_of_bounds_flag)
    
    # Final reward with floor to prevent runaway penalties from dominating gradients
    total_reward = max(total_reward, -500.0)
    individual_reward['total_reward'] = float(total_reward)
    
    return float(total_reward), individual_reward