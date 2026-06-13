# EG-RSA Reward Search Run Report
best_candidate: g1_c2
schema_version: eg_rsa_reward_schema_v1_0f5ab68dae
env_alias: Env-90b964d9
status: ok
selection_score_private_eval: -84.16028288271336
private_eval_return: -84.16028288271336
generated_reward_return: -153.80860537142675
repair_attempts: 0
repair_success: False
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: ['g0_c0', 'g0_c1']

## Reflection / Feedback Context
Based on the structured evidence from Generation 0, here is my analysis:

## 1. What worked

- **All 4 candidates were valid** (no validation errors, status "ok"), indicating the reward code generation produced syntactically correct functions.
- **Candidate g0_c0 scored best** with a private_eval_return of -122.83, which, while still negative, is substantially better than the others. Its diagnostics show a **generated_private_gap of only 54.09**, the smallest gap among all candidates, suggesting the reward shaping is more aligned with the hidden evaluator.
- **g0_c0 had the best component returns**: progress (-50.64) and stability (-8.11) are the least negative among all candidates. Its effort is 0.0 (no penalty applied), and terminal is -10.0 (less harsh than -50.0 used by others).
- **g0_c0's action distribution is reasonable**: action_mean=0.0 (mostly no-op) and action_std=0.0, combined with episode_length_mean=64.33, suggests the lander is not firing engines aggressively and episodes end relatively quickly (likely crashing or landing).
- **All candidates include terminal rewards** for success (+100) and failure (negative), which is a standard and necessary pattern for sparse reward tasks.

## 2. What failed

- **All candidates have negative private_eval_return**, meaning none of them achieve a positive total reward under the hidden evaluator. The best score is -122.83, which is far from a successful landing.
- **g0_c0's terminal reward is only -10** for failure, while others use -50. This is less punitive but still insufficient to guide the agent toward success.
- **g0_c3 performed worst** with private_eval_return = -620.11, massive generated_private_gap (276.68), and action_mean=2.0 (always main engine), indicating the reward function completely failed to discourage constant main engine firing.
- **g0_c2 has a huge generated_private_gap (728.62)**, meaning the candidate's own internal reward calculation is wildly different from the hidden evaluator's. This suggests fundamental misalignment in reward structure.
- **g0_c1 and g0_c3 have large generated_private_gaps** (172.29 and 276.68 respectively), indicating poor alignment with the hidden evaluator.
- **No successful landings occurred** in any candidate's evaluation (terminal component returns are all negative, ranging from -10 to -50).
- **g0_c3's effort component is -29.6**, heavily penalizing engine usage, yet the agent still fired main engine constantly (action_mean=2.0), suggesting the penalty was insufficient or the progress/stability signals were too weak to overcome the crash penalty.

## 3. What to try next

- **Reduce the generated_private_gap**: The key issue is that all candidates have large gaps between their own reward calculation and the hidden evaluator. Focus on making the reward function's internal logic match the hidden evaluator's priorities. Since g0_c0 has the smallest gap, analyze its structure more carefully and iterate from it.
- **Adjust reward scales**: The current rewards are too small relative to the terminal penalties. Try larger shaping rewards (e.g., progress reward of -1.0 * dist instead of -0.5) to provide stronger guidance before termination.
- **Rethink the terminal reward structure**: A failure penalty of -10 (as in g0_c0) may be too weak to discourage crashes, while -50 may dominate the total reward and obscure shaping signals. Consider a moderate failure penalty like -25 or -30.
- **Improve progress signal**: Instead of just penalizing distance, explicitly reward *reduction in distance* (delta shaping) as attempted by g0_c1 and g0_c2, but with higher scaling factors. The delta approach can provide more informative gradients.
- **Add velocity shaping**: The hidden evaluator likely cares about soft landing. Add explicit rewards for low vertical speed near the ground (e.g., exponential reward for |vy| < threshold when y is small).
- **Balance effort penalties**: g0_c0 has 0 effort penalty, which may be too lenient. But g0_c3's heavy penalty (-0.3 per main engine use) led to constant firing anyway. Try moderate penalties (-0.1 to -0.2) that don't dominate the total reward.
- **Consider sparse + dense hybrid**: Keep the terminal success reward high (+100) but also provide dense shaping that smoothly guides the agent toward the pad with appropriate velocity control.
- **Test with more episodes**: The current episode lengths are short (64-133 steps), suggesting the lander crashes quickly. The reward function should encourage longer, stable flights that eventually lead to successful landings.

## 4. Which lessons seem supported or contradicted

**Supported lessons:**
- **Terminal rewards alone are insufficient** for learning in Lunar Lander. All candidates had terminal rewards but still failed, supporting the need for dense shaping.
- **Large generated_private_gap correlates with poor performance**. The candidate with smallest gap (g0_c0) scored best, supporting the idea that alignment with the hidden evaluator is critical.
- **Action distribution matters**: g0_c3 (constant main engine) and g0_c2 (mostly main engine) performed worst, supporting the importance of discouraging constant engine firing.
- **Progress and stability components are key**: The best candidate had the best progress and stability scores, suggesting these components are most aligned with the hidden evaluator's priorities.

**Contradicted lessons:**
- **Heavy effort penalties do not necessarily reduce engine usage**. g0_c3 had the highest effort penalty but still had action_mean=2.0 (constant main engine), contradicting the idea that penalizing fuel consumption effectively shapes behavior.
- **Complex shaping is not always better**: g0_c2 had the most elaborate shaping (delta distance, multiple velocity penalties, vertical penalty near ground) but performed worst, contradicting the assumption that more detailed shaping automatically improves learning.
- **Smaller failure penalties (g0_c0 = -10) led to better scores** than larger penalties (-50), contradicting the intuition that harsh failure penalties are necessary to avoid crashes. The agent may need more exploration before harsh penalties help.

## Diagnostics
```json
{
  "generated_private_gap": -69.64832248871339,
  "action_mean": 0.07111111111111111,
  "action_std": 0.25701035190757265,
  "episode_length_mean": 75.0,
  "component_returns": {
    "progress": -129.51752754372572,
    "stability": -9.02441116103437,
    "effort": -0.26666666666666666,
    "terminal": -15.0
  }
}
```

## Prompt paths
```json
{
  "reward_coder": {
    "system": "runs/eg_rsa_lunar_deepseek_seed0_g3p4_t30k/llm/generation_1/g1_c2/reward_coder/system.txt",
    "user": "runs/eg_rsa_lunar_deepseek_seed0_g3p4_t30k/llm/generation_1/g1_c2/reward_coder/user.txt",
    "response": "runs/eg_rsa_lunar_deepseek_seed0_g3p4_t30k/llm/generation_1/g1_c2/reward_coder/response.txt"
  }
}
```

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # Unpack observations
    x = obs[0]          # normalized x position, target 0
    y = obs[1]          # normalized y position (relative to helipad)
    vx = obs[2]         # normalized x velocity
    vy = obs[3]         # normalized y velocity
    angle = obs[4]      # raw angle in radians, target 0
    ang_vel = obs[5]    # normalized angular velocity
    left_leg = obs[6]   # left leg contact (0 or 1)
    right_leg = obs[7]  # right leg contact (0 or 1)

    # Distance to landing pad (origin)
    dist = math.sqrt(x * x + y * y)
    # Speed magnitude
    speed = math.sqrt(vx * vx + vy * vy)

    # --- Progress component: dense shaping toward pad with soft landing ---
    # Exponential distance penalty: strongest far away, gentle near pad
    dist_penalty = -0.8 * dist
    # Speed penalty: penalize high speed, especially vertical speed near ground
    speed_penalty = -0.5 * speed - 0.3 * abs(vy)
    # Additional vertical speed penalty when close to ground
    if y < 0.3:
        speed_penalty -= 0.8 * abs(vy)
    progress_reward = dist_penalty + speed_penalty
    # Bound to reasonable range
    progress_reward = max(-5.0, min(0.0, progress_reward))

    # --- Stability component: maintain upright orientation ---
    angle_penalty = -0.6 * abs(angle) - 0.15 * abs(ang_vel)
    # Extra penalty for large angles near ground (dangerous)
    if y < 0.3 and abs(angle) > 0.3:
        angle_penalty -= 0.5
    stability_reward = max(-3.0, min(0.0, angle_penalty))

    # --- Effort component: moderate penalty for engine usage ---
    # action: 0=do nothing, 1=left, 2=main, 3=right
    if action == 2:  # main engine
        effort_reward = -0.15
    elif action in [1, 3]:  # side engines
        effort_reward = -0.05
    else:
        effort_reward = 0.0

    # --- Terminal component: success or failure ---
    terminal_reward = 0.0
    if done:
        both_legs = (left_leg > 0.5 and right_leg > 0.5)
        near_pad = abs(x) < 0.12 and abs(y) < 0.08
        low_speed = abs(vx) < 0.1 and abs(vy) < 0.1
        upright = abs(angle) < 0.25

        if both_legs and near_pad and low_speed and upright:
            terminal_reward = 100.0  # successful landing
        else:
            terminal_reward = -15.0  # crash or out-of-bounds

    # Combine all components
    total_reward = progress_reward + stability_reward + effort_reward + terminal_reward

    # Clip to absolute bound
    total_reward = max(min(total_reward, 1000.0), -1000.0)

    components = {
        "progress": progress_reward,
        "stability": stability_reward,
        "effort": effort_reward,
        "terminal": terminal_reward,
    }

    return float(total_reward), components

```