"""
v2c_step_urgency: v2b + small per-step penalty to create urgency.
-0.05 per step means ~-5 to -10 total over a typical episode.
This creates mild pressure to finish the episode (by landing)
without being harsh enough to cause crashes.
Hypothesis: A tiny step penalty tips the balance toward landing.
"""
import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_leg = next_obs[6]
    right_leg = next_obs[7]

    dist = np.sqrt(x**2 + y**2)

    # Distance shaping
    shaping = 3.0 * np.exp(-2.0 * dist)

    # Gentle velocity penalty
    speed = np.sqrt(vx**2 + vy**2)
    velocity_penalty = -0.3 * speed

    # Gentle angle penalty
    angle_penalty = -0.3 * abs(angle)

    # Tiny per-step penalty for urgency
    step_penalty = -0.05

    # Terminal: big bonus for landing, 0 for timeout/crash
    terminal = 0.0
    if done:
        near_pad = abs(x) < 0.15 and y < 0.15
        both_legs = left_leg > 0.5 and right_leg > 0.5
        slow = abs(vx) < 0.3 and abs(vy) < 0.3
        upright = abs(angle) < 0.2

        if both_legs and near_pad and slow and upright:
            terminal = 1000.0

    total = shaping + velocity_penalty + angle_penalty + step_penalty + terminal
    return float(total), {
        "shaping": shaping,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "step_penalty": step_penalty,
        "terminal": terminal,
    }
