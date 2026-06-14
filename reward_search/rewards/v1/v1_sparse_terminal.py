"""
v1_sparse_terminal: Pure sparse reward.
Only reward is at episode end:
  +100 for successful landing
  -100 for crash/failure
  0 during flight

Hypothesis: With enough training, the agent can learn from sparse rewards alone.
Risk: May not learn at all if exploration never finds a landing.
"""
import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    total = 0.0
    components = {}

    if done:
        # Landing condition
        x, y = next_obs[0], next_obs[1]
        vx, vy = next_obs[2], next_obs[3]
        angle = next_obs[4]
        left_leg = next_obs[6]
        right_leg = next_obs[7]

        near_pad = abs(x) < 0.2 and y < 0.2
        both_legs_down = left_leg > 0.5 and right_leg > 0.5
        slow = abs(vx) < 0.5 and abs(vy) < 0.5
        upright = abs(angle) < 0.3

        if both_legs_down and near_pad and slow and upright:
            total = 100.0
            components["terminal"] = 100.0
        else:
            total = -100.0
            components["terminal"] = -100.0
    else:
        components["terminal"] = 0.0

    return float(total), components
