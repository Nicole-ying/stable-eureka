"""
v6c: v5a + tighter landing conditions for cleaner landings.
pos<0.2, y<0.2, vel<0.4, angle<0.2. Terminal=1000.
Tighter conditions → better landing quality → better hidden score.
250k steps.
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
    step_penalty = -0.03
    terminal = 0.0
    if done:
        near_pad = abs(x) < 0.2 and y < 0.2
        both_legs = left_leg > 0.5 and right_leg > 0.5
        slow = abs(vx) < 0.4 and abs(vy) < 0.4
        upright = abs(angle) < 0.2
        if both_legs and near_pad and slow and upright:
            terminal = 1000.0
    total = shaping + velocity_penalty + angle_penalty + step_penalty + terminal
    return float(total), {"shaping":shaping,"velocity_penalty":velocity_penalty,"angle_penalty":angle_penalty,"step_penalty":step_penalty,"terminal":terminal}
