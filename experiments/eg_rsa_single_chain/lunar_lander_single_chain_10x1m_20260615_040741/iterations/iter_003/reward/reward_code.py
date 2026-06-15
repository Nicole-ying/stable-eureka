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
    
    # 3. Position shaping: Penalize distance to target (0,0) instead of rewarding Y descent
    # This prevents free-fall exploitation where agent just falls to reduce Y error
    distance_to_target = (x_pos**2 + y_pos**2)**0.5
    position_shaping = -distance_to_target * 1.0
    
    # 4. Orientation shaping: reward upright angle (near 0 radians)
    orientation_shaping = -abs(angle) * 3.0
    
    # 5. Velocity penalty flight: Penalize high velocity magnitude continuously during flight
    # Discourages uncontrolled acceleration from gravity (free-fall exploitation)
    velocity_magnitude_sq = vx**2 + vy**2
    velocity_penalty_flight = -velocity_magnitude_sq * 0.5
    
    # === TERMINAL SUCCESS REWARD (Active in training loop) ===
    leg_contact = (leg_contact_0 == 1.0 or leg_contact_1 == 1.0)
    near_center = abs(x_pos) < 0.5
    low_velocity = abs(vx) < 0.5 and abs(vy) < 0.5
    upright = abs(angle) < 0.5
    
    terminal_success_reward = 0.0
    if terminated and leg_contact and near_center and low_velocity and upright:
        terminal_success_reward = 100.0
    
    # === COMPUTE TOTAL REWARD (active terms only) ===
    reward = step_cost + fuel_penalty + position_shaping + orientation_shaping + velocity_penalty_flight + terminal_success_reward
    
    # === UPDATE INTERNAL STATE FOR DIAGNOSTICS ===
    self._step_counter += 1
    self._cumulative_fuel += m_power + s_power
    
    # === BUILD INDIVIDUAL REWARD DICTIONARY ===
    individual_reward = {
        'step_cost': float(step_cost),
        'fuel_penalty': float(fuel_penalty),
        'position_shaping_distance': float(position_shaping),
        'orientation_shaping': float(orientation_shaping),
        'velocity_penalty_flight': float(velocity_penalty_flight),
        'terminal_success_reward': float(terminal_success_reward),
        'cumulative_fuel': float(self._cumulative_fuel),
        'step_count': int(self._step_counter),
        'is_successful_landing': 1.0 if (terminated and leg_contact and near_center and low_velocity and upright) else 0.0,
        'leg_contact': float(leg_contact_0 + leg_contact_1),
        'velocity_magnitude_sq': float(velocity_magnitude_sq)
    }
    
    return float(reward), individual_reward