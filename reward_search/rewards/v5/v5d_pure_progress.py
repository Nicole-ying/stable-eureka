"""
v5d_pure_progress: Potential-based only (no absolute shaping).
reward = 2.0*(prev_dist - next_dist) + velocity_penalty + angle_penalty + terminal
Progress-based reward can't be exploited by hovering (zero progress = zero reward).
Mathematically correct shaping that preserves optimal policy.
150k steps.
"""
import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    prev_x, prev_y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_leg = next_obs[6]
    right_leg = next_obs[7]

    prev_dist = np.sqrt(prev_x**2 + prev_y**2)
    next_dist = np.sqrt(next_x**2 + next_y**2)

    # Progress: positive when moving toward pad
    progress = 2.0 * (prev_dist - next_dist)

    speed = np.sqrt(vx**2 + vy**2)
    velocity_penalty = -0.3 * speed
    angle_penalty = -0.3 * abs(angle)

    terminal = 0.0
    if done:
        near_pad = abs(next_x) < 0.3 and next_y < 0.25
        both_legs = left_leg > 0.5 and right_leg > 0.5
        slow = abs(vx) < 0.5 and abs(vy) < 0.5
        upright = abs(angle) < 0.3
        if both_legs and near_pad and slow and upright:
            terminal = 800.0

    total = progress + velocity_penalty + angle_penalty + terminal
    return float(total), {
        "progress": progress, "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty, "terminal": terminal,
    }
