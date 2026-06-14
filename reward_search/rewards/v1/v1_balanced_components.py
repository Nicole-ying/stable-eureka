"""
v1_balanced_components: Multiple balanced reward components.
- Distance: exponential shaping toward pad
- Velocity: penalty for high speed
- Angle: penalty for tilt
- Contact: bonus for leg contact near pad
- Fuel: small penalty for engine use
- Terminal: large bonus for successful landing, penalty for crash

Each component scaled to ~10-100 range per episode.
Terminal dominates (500) to make landing the clear goal.
"""
import numpy as np

def compute_reward(obs, action, next_obs, done, info):
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_leg = next_obs[6]
    right_leg = next_obs[7]

    # --- Distance shaping (smooth gradient to pad) ---
    dist = np.sqrt(x**2 + y**2)
    distance_reward = 2.0 * np.exp(-2.5 * dist)  # max 2.0 at pad, decays fast

    # --- Velocity penalty (smooth landing) ---
    velocity_penalty = -0.4 * (abs(vx) + abs(vy))

    # --- Angle penalty (stay upright) ---
    angle_penalty = -0.5 * abs(angle)

    # --- Leg contact bonus (encourage touching ground near pad) ---
    near_pad = abs(x) < 0.25 and y < 0.2
    leg_contact = (left_leg > 0.5) or (right_leg > 0.5)
    contact_bonus = 2.0 if (leg_contact and near_pad) else 0.0

    # --- Fuel efficiency ---
    fuel_penalty = -0.05 if action == 2 else 0.0  # main engine costs fuel

    # --- Terminal ---
    terminal = 0.0
    if done:
        both_legs = left_leg > 0.5 and right_leg > 0.5
        slow = abs(vx) < 0.3 and abs(vy) < 0.3
        upright = abs(angle) < 0.15
        on_pad = abs(x) < 0.15 and y < 0.15

        if both_legs and on_pad and slow and upright:
            terminal = 500.0  # Successful landing
        else:
            terminal = -50.0  # Crash or out of bounds

    total = distance_reward + velocity_penalty + angle_penalty + contact_bonus + fuel_penalty + terminal
    total = float(np.clip(total, -1000.0, 1000.0))

    return total, {
        "distance_reward": distance_reward,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "contact_bonus": contact_bonus,
        "fuel_penalty": fuel_penalty,
        "terminal": terminal,
    }
