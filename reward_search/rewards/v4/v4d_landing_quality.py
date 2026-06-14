"""
v4d_landing_quality: v3a + quality-dependent terminal bonus.
Terminal scales with landing quality: faster/better landings get more reward.
Rewards precision without being binary.
Also adds tiny step penalty for urgency.
200k steps.
"""
import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_leg = next_obs[6]
    right_leg = next_obs[7]

    dist = np.sqrt(x**2 + y**2)
    shaping = 2.0 * np.exp(-2.0 * dist)
    speed = np.sqrt(vx**2 + vy**2)
    velocity_penalty = -0.3 * speed
    angle_penalty = -0.3 * abs(angle)
    step_penalty = -0.02  # tiny urgency

    terminal = 0.0
    if done:
        both_legs = left_leg > 0.5 and right_leg > 0.5
        if both_legs:
            # Quality score: closer to pad, slower, more upright = better
            pos_quality = np.exp(-3.0 * dist)
            vel_quality = np.exp(-2.0 * speed)
            ang_quality = np.exp(-2.0 * abs(angle))
            quality = pos_quality * vel_quality * ang_quality
            terminal = 1000.0 * quality  # up to 1000 for perfect landing
        else:
            terminal = -10.0  # very mild non-landing penalty

    total = shaping + velocity_penalty + angle_penalty + step_penalty + terminal
    return float(total), {
        "shaping": shaping, "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty, "step_penalty": step_penalty,
        "terminal": terminal,
    }
