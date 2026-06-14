"""
v2b_big_terminal: v1_distance_shaping + BIG terminal bonus + no crash penalty.
Terminal = 1000 for landing (was 500). No crash penalty.
The landing bonus should dominate all per-step shaping rewards.
Hypothesis: A massive terminal bonus makes landing the unambiguous goal.
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

    # Terminal: MASSIVE bonus for landing, 0 otherwise
    terminal = 0.0
    if done:
        near_pad = abs(x) < 0.15 and y < 0.15
        both_legs = left_leg > 0.5 and right_leg > 0.5
        slow = abs(vx) < 0.3 and abs(vy) < 0.3
        upright = abs(angle) < 0.2

        if both_legs and near_pad and slow and upright:
            terminal = 1000.0  # Doubled from v1

    total = shaping + velocity_penalty + angle_penalty + terminal
    return float(total), {
        "shaping": shaping,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "terminal": terminal,
    }
