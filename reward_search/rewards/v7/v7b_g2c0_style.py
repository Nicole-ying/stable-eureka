"""
v7b: g2_c0 style from the successful singlechain experiment.
shaping=3.0*exp(-3.0*dist), vel=-0.5*(|vx|+|vy|), angle=-0.5*(|a|+0.5*|ω|)
terminal=500, tighter landing conditions. No step penalty.
This config previously achieved hidden=+267.8 (albeit with different PPO params).
100k steps.
"""
import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_leg = next_obs[6]
    right_leg = next_obs[7]
    dist = np.sqrt(x**2 + y**2)
    shaping = 3.0 * np.exp(-3.0 * dist)
    velocity_penalty = -0.5 * (abs(vx) + abs(vy))
    angle_penalty = -0.5 * (abs(angle) + 0.5 * abs(ang_vel))
    terminal = 0.0
    if done:
        near_pad = abs(x) < 0.15 and y < 0.15
        both_legs = left_leg > 0.5 and right_leg > 0.5
        slow = abs(vx) < 0.3 and abs(vy) < 0.3
        upright = abs(angle) < 0.2
        if both_legs and near_pad and slow and upright:
            terminal = 500.0
    total = shaping + velocity_penalty + angle_penalty + terminal
    return float(total), {"shaping":shaping,"velocity_penalty":velocity_penalty,"angle_penalty":angle_penalty,"terminal":terminal}
