"""
v2a_no_crash_penalty: v1_distance_shaping + NO crash penalty.
Same shaping structure but terminal=0 for non-landing instead of -50.
This removes the fear of attempting a landing.
Hypothesis: Agent will attempt more landings without the crash penalty.
"""
import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_leg = next_obs[6]
    right_leg = next_obs[7]

    dist = np.sqrt(x**2 + y**2)

    # Distance shaping (same as v1_distance_shaping)
    shaping = 3.0 * np.exp(-2.0 * dist)

    # Gentle velocity penalty
    speed = np.sqrt(vx**2 + vy**2)
    velocity_penalty = -0.3 * speed

    # Gentle angle penalty
    angle_penalty = -0.3 * abs(angle)

    # Terminal: +500 for landing, 0 for anything else (NO CRASH PENALTY)
    terminal = 0.0
    if done:
        near_pad = abs(x) < 0.15 and y < 0.15
        both_legs = left_leg > 0.5 and right_leg > 0.5
        slow = abs(vx) < 0.3 and abs(vy) < 0.3
        upright = abs(angle) < 0.2

        if both_legs and near_pad and slow and upright:
            terminal = 500.0

    total = shaping + velocity_penalty + angle_penalty + terminal
    return float(total), {
        "shaping": shaping,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "terminal": terminal,
    }
