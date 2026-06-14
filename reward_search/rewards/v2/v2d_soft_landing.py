"""
v2d_soft_landing: v1_distance_shaping + softer landing conditions + bigger bonus.
Relaxed conditions make landing easier to achieve early in training.
Terminal=800, no crash penalty.
Hypothesis: Easier landing conditions help the agent discover landing sooner,
then it can refine toward cleaner landings.
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

    # Terminal: generous landing conditions + big bonus
    terminal = 0.0
    if done:
        # Softer conditions than v1:
        # position within 0.3 (was 0.15), velocity within 0.5 (was 0.3), angle within 0.3 (was 0.2)
        near_pad = abs(x) < 0.3 and y < 0.25
        both_legs = left_leg > 0.5 and right_leg > 0.5
        slow = abs(vx) < 0.5 and abs(vy) < 0.5
        upright = abs(angle) < 0.3

        if both_legs and near_pad and slow and upright:
            terminal = 800.0

    total = shaping + velocity_penalty + angle_penalty + terminal
    return float(total), {
        "shaping": shaping,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "terminal": terminal,
    }
