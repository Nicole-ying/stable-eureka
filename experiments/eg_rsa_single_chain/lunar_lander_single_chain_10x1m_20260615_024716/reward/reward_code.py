def compute_reward(self, state, m_power, s_power, terminated):
    # Initialize internal tracking attributes if not present (lazy init for episode start)
    if not hasattr(self, '_reward_step_count'):
        self._reset_reward_counters()
    
    # Increment step count
    self._reward_step_count += 1
    
    # Calculate fuel cost for this step
    current_fuel = m_power + s_power
    self._total_fuel_spent += current_fuel
    
    # Weights
    W_FUEL = 0.1
    W_TIME = 0.1
    W_VEL = 0.5
    W_ANG = 0.5
    R_SUCCESS = 100.0
    R_FAILURE = -100.0
    
    # Shaping Rewards (Dense)
    fuel_reward = -current_fuel * W_FUEL
    time_reward = -W_TIME
    
    # Velocity magnitude penalty (state[2]=vx, state[3]=vy)
    vel_mag = abs(state[2]) + abs(state[3])
    velocity_reward = -vel_mag * W_VEL
    
    # Orientation penalty (state[4]=angle)
    angle_deviation = abs(state[4])
    orientation_reward = -angle_deviation * W_ANG
    
    reward = fuel_reward + time_reward + velocity_reward + orientation_reward
    individual_reward = {
        'fuel_cost': float(fuel_reward),
        'time_penalty': float(time_reward),
        'velocity_penalty': float(velocity_reward),
        'orientation_penalty': float(orientation_reward),
        'step_count': int(self._reward_step_count),
        'total_fuel_spent': float(self._total_fuel_spent)
    }
    
    # Terminal Logic
    if terminated:
        leg_contact = (state[6] == 1.0) or (state[7] == 1.0)
        is_upright = angle_deviation < 0.15  # Approx threshold for upright in normalized space
        is_slow = vel_mag < 1.0               # Threshold for safe landing velocity
        
        if leg_contact and is_upright and is_slow:
            reward += R_SUCCESS
            individual_reward['terminal_bonus'] = float(R_SUCCESS)
            individual_reward['landing_velocity_norm'] = float(vel_mag)
            individual_reward['landing_angle_deviation'] = float(angle_deviation)
            individual_reward['success_flag'] = 1.0
        else:
            reward += R_FAILURE
            individual_reward['terminal_penalty'] = float(R_FAILURE)
            individual_reward['success_flag'] = 0.0
    
    return float(reward), individual_reward