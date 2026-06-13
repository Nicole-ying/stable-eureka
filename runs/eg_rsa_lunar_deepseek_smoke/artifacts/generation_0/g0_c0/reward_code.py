import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    # Unpack observations
    x, y = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    # Next state for progress computation
    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # Extract m_power and s_power from info
    m_power = info.get('m_power', 0.0)
    s_power = info.get('s_power', 0.0)
    
    # ===== Component Computations =====
    
    # 1. Distance to pad penalty (required: distance_to_pad)
    dist = np.sqrt(x**2 + y**2)
    distance_penalty = -1.0 * dist
    
    # 2. Vertical velocity penalty (required: vertical_velocity_penalty)
    # Penalize high vertical speed, more when close to ground
    vert_speed_penalty = -0.5 * abs(y_vel) - 2.0 * abs(y_vel) * max(0, 0.3 - y)
    
    # 3. Horizontal velocity penalty (required: horizontal_velocity_penalty)
    horiz_speed_penalty = -0.3 * abs(x_vel)
    
    # 4. Angle penalty (required: angle_penalty)
    angle_penalty = -0.5 * abs(angle) - 0.2 * abs(ang_vel)
    # Extra angle penalty when near ground
    if y < 0.2:
        angle_penalty -= 2.0 * abs(angle)
    
    # 5. Fuel efficiency penalty (optional: fuel_efficiency_penalty)
    fuel_penalty = -0.03 * m_power - 0.02 * s_power
    
    # 6. Time penalty (optional: time_penalty)
    time_penalty = -0.1
    
    # 7. Progress shaping (required: progress)
    # Reward moving closer to pad
    prev_dist = np.sqrt(x**2 + y**2)
    curr_dist = np.sqrt(next_x**2 + next_y**2)
    dist_improvement = prev_dist - curr_dist
    progress_reward = 2.0 * dist_improvement
    
    # Reward reducing vertical speed when near ground
    if y < 0.2 and y > -0.1:
        vert_speed_improvement = abs(y_vel) - abs(next_y_vel)
        progress_reward += 1.0 * vert_speed_improvement
    
    # Reward reducing angle when near ground
    if y < 0.3:
        angle_improvement = abs(angle) - abs(next_angle)
        progress_reward += 0.5 * angle_improvement
    
    # 8. Stability shaping (required: stability)
    # Encourage upright orientation and low angular velocity
    stability_reward = -0.3 * abs(angle) - 0.1 * abs(ang_vel)
    # Reward both legs contacting ground stably
    if left_contact and right_contact:
        stability_reward += 0.5 * (1.0 - abs(angle))
    
    # 9. Effort shaping (required: effort)
    # Penalize unnecessary engine use; reward efficient control
    effort_penalty = -0.02 * m_power - 0.01 * s_power
    # Small reward for using side engines to correct angle when needed
    if abs(angle) > 0.2 and (action == 1 or action == 3):
        effort_penalty += 0.1 * (abs(angle) - abs(next_angle))
    
    # 10. Terminal shaping (required: terminal)
    terminal_reward = 0.0
    if done:
        # Check if it's a successful landing
        both_legs_contact = (left_contact == 1.0 and right_contact == 1.0)
        near_pad = (abs(x) < 0.15 and abs(y) < 0.15)
        low_vertical_speed = abs(y_vel) < 0.1
        low_horizontal_speed = abs(x_vel) < 0.1
        upright = abs(angle) < 0.15
        
        if both_legs_contact and near_pad and low_vertical_speed and low_horizontal_speed and upright:
            # Successful landing
            terminal_reward = 100.0
        elif abs(x) >= 1.0:
            # Out of bounds
            terminal_reward = -50.0
        else:
            # Crash or other termination
            terminal_reward = -30.0
    
    # 11. Landing bonus (required: landing_bonus)
    landing_bonus = 0.0
    both_legs_contact = (left_contact == 1.0 and right_contact == 1.0)
    near_pad = (abs(x) < 0.15 and abs(y) < 0.15)
    low_vertical_speed = abs(y_vel) < 0.1
    low_horizontal_speed = abs(x_vel) < 0.1
    upright = abs(angle) < 0.15
    
    if both_legs_contact and near_pad and low_vertical_speed and low_horizontal_speed and upright:
        landing_bonus = 50.0
    
    # 12. Crash penalty (required: crash_penalty)
    crash_penalty = 0.0
    if done and terminal_reward < 0:
        crash_penalty = terminal_reward  # Already captured in terminal
    
    # ===== Combine all components =====
    total_reward = (distance_penalty + vert_speed_penalty + horiz_speed_penalty +
                    angle_penalty + fuel_penalty + time_penalty +
                    progress_reward + stability_reward + effort_penalty +
                    terminal_reward + landing_bonus)
    
    # Clamp total reward to bound
    total_reward = np.clip(total_reward, -1000.0, 1000.0)
    
    # Build components dictionary
    components = {
        'landing_bonus': float(landing_bonus),
        'crash_penalty': float(crash_penalty),
        'distance_to_pad': float(distance_penalty),
        'vertical_velocity_penalty': float(vert_speed_penalty),
        'horizontal_velocity_penalty': float(horiz_speed_penalty),
        'angle_penalty': float(angle_penalty),
        'fuel_efficiency_penalty': float(fuel_penalty),
        'time_penalty': float(time_penalty),
        'progress': float(progress_reward),
        'stability': float(stability_reward),
        'effort': float(effort_penalty),
        'terminal': float(terminal_reward),
    }
    
    return float(total_reward), components