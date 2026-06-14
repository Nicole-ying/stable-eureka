"""
v3b_minimal_shaping: Even less shaping (1.5) + bigger terminal (1000).
The terminal bonus should be the overwhelmingly dominant signal.
Shaping is just a gentle nudge toward the pad.
100k steps.
"""
import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_leg = next_obs[6]
    right_leg = next_obs[7]

    dist = np.sqrt(x**2 + y**2)

    # Minimal shaping — just enough to find the pad
    shaping = 1.5 * np.exp(-2.0 * dist)

    speed = np.sqrt(vx**2 + vy**2)
    velocity_penalty = -0.3 * speed
    angle_penalty = -0.3 * abs(angle)

    terminal = 0.0
    if done:
        near_pad = abs(x) < 0.3 and y < 0.25
        both_legs = left_leg > 0.5 and right_leg > 0.5
        slow = abs(vx) < 0.5 and abs(vy) < 0.5
        upright = abs(angle) < 0.3

        if both_legs and near_pad and slow and upright:
            terminal = 1000.0  # Massive bonus

    total = shaping + velocity_penalty + angle_penalty + terminal
    return float(total), {
        "shaping": shaping,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "terminal": terminal,
    }
