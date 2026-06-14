"""
Iter 3: Pure positive rewards. No velocity or angle penalties.
- Remove velocity_penalty and angle_penalty entirely
- Big controlled descent bonus when using main engine near ground
- Landing quality bonus for soft touchdowns
- Shaping guides to pad, terminal rewards landing
"""
import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_leg = next_obs[6]
    right_leg = next_obs[7]

    dist = np.sqrt(x**2 + y**2)
    height = max(y, 0.0)

    # Distance shaping: positive gradient toward pad
    shaping = 2.0 * np.exp(-2.0 * dist)

    # Stability bonus: reward being near pad with low speed and upright
    # This replaces velocity/angle penalties with positive framing
    speed = np.sqrt(vx**2 + vy**2)
    stability = 0.5 * np.exp(-2.0 * dist) * np.exp(-1.0 * speed) * np.exp(-2.0 * abs(angle))

    # Controlled descent: BIG bonus for using main engine when falling near ground
    is_falling = float(vy < -0.1)
    is_low = float(height < 1.5)
    is_main = float(action == 2)
    controlled_descent = is_falling * is_low * is_main * 3.0

    # Mild step penalty (only penalty remaining)
    step_penalty = -0.02

    # Terminal with landing quality bonus
    terminal = 0.0
    if done:
        both_legs = left_leg > 0.5 and right_leg > 0.5
        near_pad = abs(x) < 0.3 and y < 0.25
        slow = abs(vx) < 0.5 and abs(vy) < 0.5
        upright = abs(angle) < 0.3
        if both_legs and near_pad and slow and upright:
            # Quality multiplier: cleaner landing = bigger bonus
            pos_quality = np.exp(-3.0 * dist)
            vel_quality = np.exp(-2.0 * speed)
            ang_quality = np.exp(-2.0 * abs(angle))
            quality = pos_quality * vel_quality * ang_quality
            terminal = 1200.0 * quality

    total = shaping + stability + controlled_descent + step_penalty + terminal
    return float(total), {
        "shaping": shaping,
        "stability": stability,
        "controlled_descent": controlled_descent,
        "step_penalty": step_penalty,
        "terminal": terminal,
    }
