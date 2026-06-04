from __future__ import annotations


LUNARLANDER_FDRE_HRDC = """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, av, left, right = [float(v) for v in next_obs]
    px, py = float(obs[0]), float(obs[1])
    progress = (abs(px) + abs(py)) - (abs(x) + abs(y))
    centered = max(0.0, 1.0 - abs(x))
    slow = max(0.0, 1.0 - abs(vx) - abs(vy))
    upright = max(0.0, 1.0 - abs(angle) - 0.5 * abs(av))
    contact = float(left) + float(right)
    both = 1.0 if left > 0.5 and right > 0.5 else 0.0
    land = 1.0 if original_reward > 70.0 else 0.0
    crash = 1.0 if original_reward < -70.0 else 0.0
    fuel = 0.012 if int(action) == 2 else (0.004 if int(action) in (1, 3) else 0.0)
    p = max(0.0, min(1.0, training_progress))
    early = 0.04 * progress + 0.04 * centered + 0.03 * upright
    late = 0.02 * progress + 0.05 * centered + 0.06 * slow + 0.06 * upright + 0.08 * contact + 0.12 * both
    shaping = (1.0 - p) * early + p * late
    return float(original_reward + shaping + 4.0 * land - 4.0 * crash - fuel)
"""


LUNARLANDER_FDRE_CANDIDATES = [
    (
        "hrdc_progress_contact",
        """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, av, left, right = [float(v) for v in next_obs]
    px, py = float(obs[0]), float(obs[1])
    progress = (abs(px) + abs(py)) - (abs(x) + abs(y))
    stable = max(0.0, 1.0 - (abs(vx) + abs(vy) + abs(angle)))
    contact = float(left) + float(right)
    fuel = 0.02 if int(action) == 2 else (0.005 if int(action) in (1, 3) else 0.0)
    shaping = 0.05 * progress + 0.05 * stable + 0.05 * contact - fuel
    return float(original_reward + shaping)
""",
    ),
    (
        "hrdc_balanced_stage",
        LUNARLANDER_FDRE_HRDC,
    ),
    (
        "hrdc_contact_stage",
        """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, av, left, right = [float(v) for v in next_obs]
    px, py = float(obs[0]), float(obs[1])
    progress = (abs(px) + abs(py)) - (abs(x) + abs(y))
    centered = max(0.0, 1.0 - abs(x))
    slow = max(0.0, 1.0 - abs(vx) - 1.5 * abs(vy))
    upright = max(0.0, 1.0 - abs(angle) - 0.5 * abs(av))
    contact = float(left) + float(right)
    both = 1.0 if left > 0.5 and right > 0.5 else 0.0
    fuel = 0.015 if int(action) == 2 else (0.004 if int(action) in (1, 3) else 0.0)
    p = max(0.0, min(1.0, training_progress))
    shaping = (1.0 - p) * (0.04 * progress + 0.03 * centered + 0.03 * upright) + p * (0.03 * centered + 0.05 * slow + 0.05 * upright + 0.08 * contact + 0.12 * both)
    return float(original_reward + shaping - fuel)
""",
    ),
    (
        "hrdc_terminal_guard",
        """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, av, left, right = [float(v) for v in next_obs]
    px, py = float(obs[0]), float(obs[1])
    progress = (abs(px) + abs(py)) - (abs(x) + abs(y))
    stable = max(0.0, 1.0 - abs(vx) - abs(vy) - abs(angle))
    contact = float(left) + float(right)
    land = 1.0 if original_reward > 70.0 else 0.0
    crash = 1.0 if original_reward < -70.0 else 0.0
    return float(original_reward + 0.03 * progress + 0.04 * stable + 0.04 * contact + 4.0 * land - 4.0 * crash)
""",
    ),
]


LUNARLANDER_NO_DIAGNOSTIC = """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    fuel_cost = 0.02 if int(action) != 0 else 0.0
    return float(original_reward - fuel_cost)
"""


LUNARLANDER_NO_DYNAMIC_WEIGHTS = """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, av, left, right = [float(v) for v in next_obs]
    prev_x, prev_y = float(obs[0]), float(obs[1])
    progress = (abs(prev_x) + abs(prev_y)) - (abs(x) + abs(y))
    centered = max(0.0, 1.0 - abs(x))
    stable = max(0.0, 1.0 - (abs(vx) + abs(vy) + abs(angle) + 0.5 * abs(av)))
    contact = float(left) + float(right)
    fuel_cost = 0.02 if int(action) == 2 else (0.005 if int(action) in (1, 3) else 0.0)
    shaping = 0.035 * progress + 0.04 * centered + 0.04 * stable + 0.025 * contact - fuel_cost
    return float(original_reward + shaping)
"""


ACROBOT_FDRE_HRDC = """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    c1, s1, c2, s2, d1, d2 = [float(v) for v in obs]
    nc1, ns1, nc2, ns2, nd1, nd2 = [float(v) for v in next_obs]
    height = -c1 - (c1 * c2 - s1 * s2)
    next_height = -nc1 - (nc1 * nc2 - ns1 * ns2)
    progress = next_height - height
    velocity = abs(nd1) + abs(nd2)
    goal_bonus = 2.0 if next_height > 1.0 else 0.0
    action_cost = 0.01 if int(action) != 1 else 0.0
    p = max(0.0, min(1.0, training_progress))
    swing_reward = 0.60 * progress + 0.04 * velocity + 0.60 * next_height
    stabilize_reward = 0.25 * progress + 0.02 * velocity + 0.90 * next_height + goal_bonus
    shaping = (1.0 - p) * swing_reward + p * stabilize_reward
    return float(original_reward + shaping - action_cost)
"""
