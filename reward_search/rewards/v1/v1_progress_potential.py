"""
v1_progress_potential: Potential-based reward shaping.
reward = potential(next_obs) - potential(obs)
where potential = -distance_to_pad (negative, so reducing distance gives +reward)

This is mathematically correct: the optimal policy under shaped reward
is the same as under the original potential function.

Also includes small auxiliary penalties and terminal bonus.
"""
import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    # Potential: negative distance to pad
    # Higher (less negative) = better
    prev_x, prev_y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    prev_dist = np.sqrt(prev_x**2 + prev_y**2)
    next_dist = np.sqrt(next_x**2 + next_y**2)

    # Potential-based shaping: reward for reducing distance
    # When moving toward pad: next_dist < prev_dist → positive reward
    potential_old = -2.0 * prev_dist
    potential_new = -2.0 * next_dist
    progress = potential_new - potential_old

    # Small auxiliary penalties for stability
    # (these don't change the optimal policy, just speed up learning)
    velocity_penalty = -0.1 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = -0.1 * abs(next_obs[4])

    # Penalize main engine usage (fuel efficiency)
    fuel_penalty = -0.1 if action == 2 else 0.0

    # Terminal reward
    terminal = 0.0
    if done:
        near_pad = abs(next_x) < 0.15 and next_y < 0.15
        both_legs = next_obs[6] > 0.5 and next_obs[7] > 0.5
        slow = abs(next_obs[2]) < 0.3 and abs(next_obs[3]) < 0.3
        upright = abs(next_obs[4]) < 0.2

        if both_legs and near_pad and slow and upright:
            terminal = 300.0
        else:
            terminal = -100.0

    total = progress + velocity_penalty + angle_penalty + fuel_penalty + terminal

    return float(total), {
        "progress": progress,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "fuel_penalty": fuel_penalty,
        "terminal": terminal,
    }
