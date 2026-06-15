def compute_reward(self, state, m_power, s_power, terminated):
    # Initialize internal state tracking (safe per environment_understanding)
    if not hasattr(self, '_step_counter'):
        self._step_counter = 0
    if not hasattr(self, '_cumulative_fuel'):
        self._cumulative_fuel = 0.0
    
    # Extract state components for readability
    x_pos = state[0]  # Normalized X position (0 = pad center)
    y_pos = state[1]  # Normalized Y height
    vx = state[2]     # Velocity X
    vy = state[3]     # Velocity Y
    angle = state[4]  # Lander angle in radians
    ang_vel = state[5]  # Angular velocity
    leg_contact_0 = state[6]
    leg_contact_1 = state[7]
    
    # === ACTIVE REWARD TERMS ===
    
    # 1. Step cost: small negative per step to encourage speed (prevents hovering)
    step_cost = -0.1
    
    # 2. Fuel penalty: light penalty on engine usage (allows exploration)
    fuel_penalty = -0.5 * (m_power + s_power)
    
    # 3. Position shaping: reward progress toward pad center
    # X error reduction: closer to 0 is better
    x_shaping = -abs(x_pos) * 2.0
    # Y descent: lower height is better (y_pos decreases as lander descends)
    y_shaping = -y_pos * 1.5
    position_shaping = x_shaping + y_shaping
    
    # 4. Orientation shaping: reward upright angle (near 0 radians)
    orientation_shaping = -abs(angle) * 3.0
    
    # 5. Velocity magnitude penalty: discourage uncontrolled free-fall during flight
    velocity_magnitude = (vx**2 + vy**2)**0.5
    velocity_magnitude_penalty = -1.0 * velocity_magnitude
    
    # === TERMINAL SIGNALS (Active in Iteration 1) ===
    
    # Terminal success detection conditions
    leg_contact = (leg_contact_0 == 1.0 or leg_contact_1 == 1.0)
    near_center = abs(x_pos) < 0.5
    low_velocity = abs(vx) < 0.5 and abs(vy) < 0.5
    upright = abs(angle) < 0.5
    is_successful_landing = terminated and leg_contact and near_center and low_velocity and upright
    
    # Terminal Success Reward (+100)
    terminal_success_reward = 100.0 if is_successful_landing else 0.0
    
    # Crash/Failure Penalty (-100) for non-successful terminations
    crash_penalty = -100.0 if terminated and not is_successful_landing else 0.0
    
    # === COMPUTE TOTAL REWARD (active terms only) ===
    reward = step_cost + fuel_penalty + position_shaping + orientation_shaping + velocity_magnitude_penalty + terminal_success_reward + crash_penalty
    
    # === UPDATE INTERNAL STATE FOR DIAGNOSTICS ===
    self._step_counter += 1
    self._cumulative_fuel += m_power + s_power
    
    # === BUILD INDIVIDUAL REWARD DICTIONARY ===
    individual_reward = {
        'step_cost': float(step_cost),
        'fuel_penalty': float(fuel_penalty),
        'position_shaping_x': float(x_shaping),
        'position_shaping_y': float(y_shaping),
        'orientation_shaping': float(orientation_shaping),
        'velocity_magnitude_penalty': float(velocity_magnitude_penalty),
        'terminal_success_reward': float(terminal_success_reward),
        'crash_penalty': float(crash_penalty),
        'cumulative_fuel': float(self._cumulative_fuel),
        'step_count': int(self._step_counter),
        'is_successful_landing': 1.0 if is_successful_landing else 0.0,
        'leg_contact': float(leg_contact_0 + leg_contact_1),
        'velocity_magnitude': float(velocity_magnitude)
    }
    
    return float(reward), individual_reward