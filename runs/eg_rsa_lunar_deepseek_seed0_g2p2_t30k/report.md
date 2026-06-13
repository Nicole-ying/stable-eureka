# EG-RSA Reward Search Run Report
best_candidate: g1_c0
schema_version: eg_rsa_reward_schema_v1_8c07c5a7b5
env_alias: Env-90b964d9
status: ok
selection_score_private_eval: -47.99154506341481
private_eval_return: -47.99154506341481
generated_reward_return: -159.60337955301102
repair_attempts: 0
repair_success: False
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: ['g0_c0', 'g0_c1']

## Reflection / Feedback Context
**Reflection on Generation 0**

**1. What worked:**
- Both candidates produced valid reward functions that ran without errors (num_ok=2).
- Candidate g0_c0 achieved a significantly better private_eval_return (-357.7) compared to g0_c1 (-581.7), suggesting its reward structure is more aligned with the hidden evaluator's criteria.
- The component breakdown shows g0_c0 had lower velocity penalty (-236.7 vs -440.2) and distance penalty (-435.8 vs -875.3), indicating it encouraged gentler movement and closer proximity to the pad.
- g0_c0 achieved some ground contact bonus (3.33) while g0_c1 got none (0.0), suggesting g0_c0's landing conditions were more achievable.
- g0_c0 used less engine firing (action_mean 1.03 vs 1.55) and had shorter episodes (74.3 vs 94.0), both desirable for fuel efficiency and stable landings.

**2. What failed:**
- Both candidates have very negative private_eval_returns (-357.7 and -581.7), far from any positive landing reward. Neither achieved successful landings (both terminal rewards = -100).
- The generated_private_gap is large negative for both (-479.3 and -920.3), meaning the reward functions significantly underestimated the true return. This suggests the shaping rewards are too pessimistic or misaligned.
- g0_c0's ground_contact_bonus was only 3.33 total across the episode, meaning it rarely achieved both-legs-near-pad condition. The conditions may be too strict.
- Both candidates' distance_reward is too negative (-435.8 and -875.3) - the agent is spending too much time far from pad or the shaping is too harsh.
- Fuel efficiency penalty is small relative to other components, yet action_mean is >1.0, meaning engines are fired frequently - the penalty may not be strong enough to discourage wasteful firing.
- The terminal reward of -100 for non-landings dominates the total, but since no landings occurred, the shaping components are failing to guide the agent toward landing conditions.

**3. What to try next:**
- **Reduce shaping penalty magnitudes**: The distance_reward and velocity_penalty are too negative, overwhelming the positive signals. Try scaling them down by 2-3x to give the agent more room to explore.
- **Make ground_contact_bonus easier to achieve**: Relax the near_pad condition (e.g., |x|<0.3, y<0.2) and/or reduce the both_legs requirement to at least one leg. Consider giving smaller bonuses for partial progress.
- **Increase fuel penalty**: The current penalties (-0.5 for main, -0.2 for side) are negligible compared to other components. Try -2.0 for main, -1.0 for side to discourage random firing.
- **Add positive distance shaping**: Instead of pure negative distance, try a sigmoid or exponential that gives near-zero reward far from pad and increasingly positive reward as agent approaches pad (e.g., -3*dist → -3*(dist-0.5) or exp(-dist)).
- **Reconsider terminal reward structure**: A flat -100 for failure is harsh. Try -50 for crash, -20 for timeout, or make the penalty proportional to distance from pad at termination.
- **Adjust velocity penalty**: Penalize vertical velocity more than horizontal (since landing requires slow descent), but reduce overall magnitude. Try -1.0*abs(vy) - 0.3*abs(vx).
- **Increase episode length**: Current mean length ~74-94 steps may be too short for the agent to learn to land. The environment may allow longer episodes; consider not penalizing length.

**4. Lessons supported or contradicted:**
- **Supported**: "Large negative shaping rewards can dominate and prevent learning" - both candidates show this with very negative distance and velocity components.
- **Supported**: "Terminal reward of -100 for failure can mask all shaping signals" - the terminal -100 dominates the total, making the agent's cumulative reward heavily negative regardless of partial progress.
- **Supported**: "Strict landing conditions (both legs + near pad) are hard to achieve without proper shaping" - g0_c0 got only 3.33 ground bonus, g0_c1 got 0.
- **Contradicted**: "Using next_obs for ground contact is better than obs" - g0_c1 used next_obs and got worse results than g0_c0 which used obs. However, this may be due to other differences.
- **Contradicted**: "Higher action penalty reduces engine usage" - g0_c1 had higher fuel penalty (0.5 main, 0.3 side vs 0.5/0.2) but still had higher action_mean (1.55 vs 1.03). The penalty alone doesn't deter firing if other components demand it.
- **Needs investigation**: The large negative generated_private_gap suggests the reward functions are structurally misaligned with the hidden evaluator. This may indicate the evaluator uses different scaling, different component weights, or additional unobserved factors.

## Diagnostics
```json
{
  "generated_private_gap": -111.61183448959622,
  "action_mean": 0.0,
  "action_std": 0.0,
  "episode_length_mean": 75.0,
  "component_returns": {
    "velocity_penalty": -113.03857617899779,
    "angle_penalty": -43.81218312191777,
    "distance_reward": 46.24737974790447,
    "fuel_efficiency_penalty": 0.0,
    "ground_contact_bonus": 1.0,
    "terminal": -50.0
  }
}
```

## Prompt paths
```json
{
  "reward_coder": {
    "system": "runs/eg_rsa_lunar_deepseek_seed0_g2p2_t30k/llm/generation_1/g1_c0/reward_coder/system.txt",
    "user": "runs/eg_rsa_lunar_deepseek_seed0_g2p2_t30k/llm/generation_1/g1_c0/reward_coder/user.txt",
    "response": "runs/eg_rsa_lunar_deepseek_seed0_g2p2_t30k/llm/generation_1/g1_c0/reward_coder/response.txt"
  }
}
```

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # Unpack observation components (normalized)
    x = obs[0]
    y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    # --- Component 1: velocity_penalty ---
    # Penalize high speeds, especially vertical velocity (negative = moving down fast)
    # Scale reduced from parent to avoid overwhelming other signals
    vel_penalty = -1.5 * (abs(vy) + 0.3 * abs(vx))
    
    # --- Component 2: angle_penalty ---
    # Penalize tilt and spin, moderate magnitude
    angle_penalty = -2.0 * (abs(angle) + 0.5 * abs(ang_vel))
    
    # --- Component 3: distance_reward ---
    # Exponential shaping: near-zero far away, increasingly positive as agent approaches pad
    # This replaces the harsh linear negative distance with a smoother signal
    dist = np.sqrt(x**2 + y**2)
    # exp(-2*dist) gives ~1 when dist=0, ~0.14 when dist=1, ~0.02 when dist=2
    # Scale by 3 to make it comparable with other components
    distance_reward = 3.0 * np.exp(-2.0 * dist)
    
    # --- Component 4: fuel_efficiency_penalty ---
    # Increased penalties to discourage wasteful engine firing
    fuel_penalty = 0.0
    if action == 2:  # main engine
        fuel_penalty = -2.0
    elif action == 1 or action == 3:  # side engines
        fuel_penalty = -1.0
    
    # --- Component 5: ground_contact_bonus ---
    # Relaxed conditions: at least one leg contact and near pad
    # Also give smaller bonus for being near pad with one leg (partial progress)
    near_pad = (abs(x) < 0.3) and (y < 0.2)
    one_leg = (left_contact > 0.5) or (right_contact > 0.5)
    both_legs = (left_contact > 0.5) and (right_contact > 0.5)
    
    ground_contact_bonus = 0.0
    if both_legs and near_pad:
        ground_contact_bonus = 8.0  # full success posture
    elif one_leg and near_pad:
        ground_contact_bonus = 3.0  # partial progress
    
    # --- Component 6: terminal ---
    terminal_reward = 0.0
    if done:
        # Check for successful landing: both legs on ground, near pad, low speed, upright
        success = (
            both_legs and
            near_pad and
            abs(vy) < 0.15 and
            abs(vx) < 0.15 and
            abs(angle) < 0.15
        )
        if success:
            terminal_reward = 100.0
        else:
            # Reduced failure penalty to avoid masking shaping signals
            terminal_reward = -50.0
    
    # Sum all components
    total_reward = vel_penalty + angle_penalty + distance_reward + fuel_penalty + ground_contact_bonus + terminal_reward
    
    # Clamp total reward to absolute bound
    total_reward = np.clip(total_reward, -1000.0, 1000.0)
    
    # Build components dict
    components_dict = {
        "velocity_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "distance_reward": distance_reward,
        "fuel_efficiency_penalty": fuel_penalty,
        "ground_contact_bonus": ground_contact_bonus,
        "terminal": terminal_reward
    }
    
    return float(total_reward), components_dict

```