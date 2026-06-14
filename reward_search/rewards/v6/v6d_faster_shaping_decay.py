"""
v6d: v5a + faster shaping decay (exp(-3*dist) instead of exp(-2*dist)).
Shaping drops off faster with distance → less incentive to loiter at intermediate distances.
Terminal=1000, step=-0.03. 250k steps.
"""
import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_leg = next_obs[6]
    right_leg = next_obs[7]
    dist = np.sqrt(x**2 + y**2)
    shaping = 2.0 * np.exp(-3.0 * dist)  # faster decay
    speed = np.sqrt(vx**2 + vy**2)
    velocity_penalty = -0.3 * speed
    angle_penalty = -0.3 * abs(angle)
    step_penalty = -0.03
    terminal = 0.0
    if done:
        near_pad = abs(x) < 0.3 and y < 0.25
        both_legs = left_leg > 0.5 and right_leg > 0.5
        slow = abs(vx) < 0.5 and abs(vy) < 0.5
        upright = abs(angle) < 0.3
        if both_legs and near_pad and slow and upright:
            terminal = 1000.0
    total = shaping + velocity_penalty + angle_penalty + step_penalty + terminal
    return float(total), {"shaping":shaping,"velocity_penalty":velocity_penalty,"angle_penalty":angle_penalty,"step_penalty":step_penalty,"terminal":terminal}
