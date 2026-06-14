"""
v1_distance_shaping: Continuous distance-based shaping.
Uses exp(-α*dist) to create a smooth gradient toward the landing pad.
+ terminal bonus for landing.

Hypothesis: The exponential decay gives strong gradient near the pad
without overwhelming penalties far away.
"""
import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_leg = next_obs[6]
    right_leg = next_obs[7]

    # Distance to pad
    dist = np.sqrt(x**2 + y**2)

    # Shaping: positive reward for being near pad, decays exponentially
    # At dist=0: +3.0, at dist=0.5: +1.1, at dist=1.0: +0.15, at dist=2.0: ~0
    shaping = 3.0 * np.exp(-2.0 * dist)

    # Gentle velocity penalty to discourage excessive speed
    speed = np.sqrt(vx**2 + vy**2)
    velocity_penalty = -0.3 * speed

    # Gentle angle penalty
    angle_penalty = -0.3 * abs(angle)

    # Terminal reward
    terminal = 0.0
    if done:
        near_pad = abs(x) < 0.15 and y < 0.15
        both_legs = left_leg > 0.5 and right_leg > 0.5
        slow = abs(vx) < 0.3 and abs(vy) < 0.3
        upright = abs(angle) < 0.2

        if both_legs and near_pad and slow and upright:
            terminal = 500.0
        else:
            terminal = -50.0  # Penalty for crashing, but not too harsh

    total = shaping + velocity_penalty + angle_penalty + terminal

    return float(total), {
        "shaping": shaping,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "terminal": terminal,
    }
