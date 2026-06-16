def compute_reward(self, state, m_power, s_power, terminated):
    # Extract state variables
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    angle = state[4]
    angular_velocity = state[5]
    leg1_contact = state[6]
    leg2_contact = state[7]

    # Initialize diagnostic variables
    distance_to_pad = 0.0
    vertical_speed = 0.0
    angle_deviation = 0.0
    fuel_usage = 0.0
    success_flag = 0.0
    crash_flag = 0.0
    distance_progress = 0.0
    vertical_speed_progress = 0.0
    progress_bonus_flag = 0.0
    action_0_bonus_flag = 0.0
    descent_bonus_flag = 0.0
    near_ground_stability_bonus_flag = 0.0

    # Compute distance to pad (horizontal distance)
    distance_to_pad = abs(x)

    # Compute vertical speed magnitude
    vertical_speed = abs(vy)

    # Compute angle deviation from upright
    angle_deviation = abs(angle)

    # Compute fuel usage
    fuel_usage = m_power + s_power

    # Initialize reward components
    success_bonus = 0.0
    angle_penalty = 0.0
    fuel_penalty_main = 0.0
    fuel_penalty_side = 0.0
    altitude_penalty = 0.0
    descent_bonus = 0.0
    near_ground_stability_bonus = 0.0
    progress_bonus = 0.0
    action_0_bonus = 0.0
    crash_penalty = 0.0

    # Check for success: both legs contact, near pad, low velocities, upright
    both_legs_contact = (leg1_contact == 1.0 and leg2_contact == 1.0)
    near_pad = (abs(x) < 0.1)
    low_vx = (abs(vx) < 0.1)
    low_vy = (abs(vy) < 0.1)
    upright = (abs(angle) < 0.1)

    if both_legs_contact and near_pad and low_vx and low_vy and upright:
        success_bonus = 500.0
        success_flag = 1.0

    # Progress-delta shaping for distance
    # Use internal state to track previous distance
    if not hasattr(self, 'prev_distance'):
        self.prev_distance = distance_to_pad
    if not hasattr(self, 'prev_vertical_speed'):
        self.prev_vertical_speed = vertical_speed
    if not hasattr(self, 'progress_bonus_paid'):
        self.progress_bonus_paid = False
    if not hasattr(self, 'action_0_bonus_paid'):
        self.action_0_bonus_paid = False
    if not hasattr(self, 'descent_bonus_paid'):
        self.descent_bonus_paid = False
    if not hasattr(self, 'near_ground_stability_bonus_paid'):
        self.near_ground_stability_bonus_paid = False

    distance_delta = self.prev_distance - distance_to_pad
    distance_progress = 0.5 * distance_delta
    self.prev_distance = distance_to_pad

    # Progress-delta shaping for vertical speed
    vertical_speed_delta = self.prev_vertical_speed - vertical_speed
    vertical_speed_progress = 0.5 * vertical_speed_delta
    self.prev_vertical_speed = vertical_speed

    # Angle penalty (regularizer)
    angle_penalty = -0.02 * angle_deviation

    # Fuel penalties (regularizers)
    fuel_penalty_main = -0.03 * m_power
    fuel_penalty_side = -0.01 * s_power

    # Altitude penalty: mild per-step penalty for being at high altitude (y > 0.5)
    if y > 0.5:
        altitude_penalty = -0.01 * (y - 0.5)

    # Descent bonus: one-shot event-gated bonus for first descending below y < 0.3
    if y < 0.3 and not self.descent_bonus_paid:
        descent_bonus = 50.0
        descent_bonus_flag = 1.0
        self.descent_bonus_paid = True

    # Near-ground stability bonus: one-shot event-gated bonus for first achieving stable state near ground
    near_ground_stable = (y < 0.3) and (abs(vx) < 0.1) and (abs(vy) < 0.1) and (abs(angle) < 0.1)
    if near_ground_stable and not self.near_ground_stability_bonus_paid:
        near_ground_stability_bonus = 30.0
        near_ground_stability_bonus_flag = 1.0
        self.near_ground_stability_bonus_paid = True

    # Progress bonus: one-shot event-gated bonus for achieving stable near-pad state with both legs contacting
    stable_near_pad = (abs(x) < 0.1) and (abs(vx) < 0.1) and (abs(vy) < 0.1) and (abs(angle) < 0.1) and both_legs_contact
    if stable_near_pad and not self.progress_bonus_paid:
        progress_bonus = 1.0
        progress_bonus_flag = 1.0
        self.progress_bonus_paid = True

    # Action 0 bonus: one-shot event-gated bonus for using action 0 when stable near-pad state is achieved
    if m_power == 0 and s_power == 0 and stable_near_pad and not self.action_0_bonus_paid:
        action_0_bonus = 1.0
        action_0_bonus_flag = 1.0
        self.action_0_bonus_paid = True

    # Crash penalty (if terminated and not success)
    if terminated and success_flag == 0.0:
        crash_penalty = -100.0
        crash_flag = 1.0

    # Total reward
    reward = success_bonus + distance_progress + vertical_speed_progress + angle_penalty + fuel_penalty_main + fuel_penalty_side + altitude_penalty + descent_bonus + near_ground_stability_bonus + progress_bonus + action_0_bonus + crash_penalty

    # Individual reward diagnostics
    individual_reward = {
        'success_bonus': success_bonus,
        'distance_progress': distance_progress,
        'vertical_speed_progress': vertical_speed_progress,
        'angle_penalty': angle_penalty,
        'fuel_penalty_main': fuel_penalty_main,
        'fuel_penalty_side': fuel_penalty_side,
        'altitude_penalty': altitude_penalty,
        'descent_bonus': descent_bonus,
        'near_ground_stability_bonus': near_ground_stability_bonus,
        'progress_bonus': progress_bonus,
        'action_0_bonus': action_0_bonus,
        'crash_penalty': crash_penalty,
        'distance_to_pad': distance_to_pad,
        'vertical_speed': vertical_speed,
        'angle_deviation': angle_deviation,
        'fuel_usage': fuel_usage,
        'success_flag': success_flag,
        'crash_flag': crash_flag,
        'progress_bonus_flag': progress_bonus_flag,
        'action_0_bonus_flag': action_0_bonus_flag,
        'descent_bonus_flag': descent_bonus_flag,
        'near_ground_stability_bonus_flag': near_ground_stability_bonus_flag
    }

    return float(reward), individual_reward