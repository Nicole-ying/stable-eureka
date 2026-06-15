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
    
    # 5. Velocity Control (NEW ACTIVE): Penalize high velocity during flight to discourage free-fall
    vel_magnitude = (vx**2 + vy**2)**0.5
    velocity_control = -vel_magnitude * 1.5
    
    # 6. Terminal Success Reward (ACTIVE): Large bonus for safe landing at termination
    leg_contact = (leg_contact_0 == 1.0 or leg_contact_1 == 1.0)
    near_center = abs(x_pos) < 0.5
    low_velocity = vel_magnitude < 0.5
    upright = abs(angle) < 0.5
    
    terminal_success_reward = 0.0
    if terminated and leg_contact and near_center and low_velocity and upright:
        terminal_success_reward = 100.0
    
    # === COMPUTE TOTAL REWARD (active terms only) ===
    reward = step_cost + fuel_penalty + position_shaping + orientation_shaping + velocity_control + terminal_success_reward
    
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
        'velocity_control': float(velocity_control),
        'terminal_success_reward': float(terminal_success_reward),
        'cumulative_fuel': float(self._cumulative_fuel),
        'step_count': int(self._step_counter),
        'is_successful_landing': 1.0 if (terminated and leg_contact and near_center and low_velocity and upright) else 0.0,
        'leg_contact': float(leg_contact_0 + leg_contact_1),
        'velocity_magnitude': float(vel_magnitude)
    }
    
    return float(reward), individual_reward