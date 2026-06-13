# Environment Memory

env_alias: Env-90b964d9
latest_generation: 1

## Latest reflection
**Reflection on Generation 1**

**1. What worked:**
- Candidate g1_c0 achieved a significantly better private_eval_return (-47.99) compared to g1_c1 (-127.70), and both improved dramatically over the best from Generation 0 (-357.7). This is a ~7x improvement in just one generation.
- **Exponential distance reward** (g1_c0) was a clear success: distance_reward = +46.25 (positive!) vs -220.90 for g1_c1 which used linear negative distance. This confirms that positive shaping for proximity works much better than negative shaping for distance.
- **Reduced shaping penalties** in g1_c0 (velocity_penalty -1.5, angle_penalty -2.0) produced much lower cumulative penalties (-113.0 and -43.8) compared to g1_c1 (-141.0 and -40.5), allowing the positive signals to dominate.
- **Relaxed ground_contact_bonus** in g1_c0 (partial bonus for one leg, expanded near_pad region) yielded +1.0 total bonus, while g1_c1 got 0.0. This is a small but meaningful improvement.
- **Reduced terminal failure penalty** (-50 instead of -100) helped both candidates avoid the massive negative terminal signal that dominated Generation 0.
- **Episode length increased** to 75 steps for g1_c0 (up from ~74 in g0_c0), suggesting the agent is surviving longer, which is a positive sign.

**2. What failed:**
- Neither candidate achieved a successful landing (both terminal rewards = -50). The private_eval_return is still negative, meaning the hidden evaluator does not consider the behavior successful.
- g1_c0's **generated_private_gap is -111.6**, still large and negative. This means the reward function significantly underestimates the true return. The shaping rewards are still too pessimistic or mis-scaled relative to the hidden evaluator.
- **Fuel penalty had zero impact**: both candidates show fuel_efficiency_penalty = 0.0, meaning the engine usage penalties are never triggered, or they are triggered but exactly offset by other components. This is suspicious and may indicate the fuel penalty code is not firing as intended (e.g., action values may be continuous floats, not discrete integers 0-3).
- **Action_mean = 0.0 for both candidates**: This is unusual - it suggests the agent is taking no actions at all (action 0 = do nothing), or the action logging is broken. If the agent is indeed doing nothing, it would explain why it never lands (just drifts until timeout/crash) and why fuel penalty is zero.
- **g1_c1 still performed poorly** (-127.70) despite similar changes. Its linear distance penalty (-4.0*dist) was too harsh, and its stricter success criteria (abs(vy)<0.1 vs 0.15) made landing even harder. This shows that small parameter differences matter greatly.

**3. What to try next:**
- **Investigate the action_mean=0.0 issue**: If the agent is truly taking no actions, the reward function may be too punitive for engine usage, or the fuel penalty implementation is buggy. Verify whether actions are continuous (e.g., [main, side]) or discrete (0,1,2,3). If continuous, the current `if action == 2` logic will never trigger, making fuel penalty always zero. This is a critical bug to fix.
- **Further reduce or eliminate the terminal failure penalty**: The -50 for failure still dominates. Try -20 or even 0, relying solely on shaping to guide the agent. The hidden evaluator likely rewards successful landings, not punishes failures.
- **Make distance reward more generous**: g1_c0's exponential distance reward (3.0 * exp(-2*dist)) worked well. Try increasing the scaling factor (e.g., 5.0) to give more positive signal, or use a slower decay (exp(-1.5*dist)) to provide positive shaping over a wider range.
- **Add a positive reward for low velocity near pad**: The current velocity_penalty is always negative. Consider a small positive reward when the agent is both near the pad and moving slowly (e.g., +2.0 if dist<0.3 and |vy|<0.1), to explicitly incentivize the pre-landing state.
- **Ensure fuel penalty actually applies**: If actions are continuous, change the fuel penalty to use action magnitude: `fuel_penalty = -2.0 * abs(action[1])` for main engine, or similar. Match the expected action space of the environment.
- **Consider removing the angle_penalty or making it very small**: The angle penalty may be counterproductive if the agent needs to tilt to fire side engines. Try -0.5*abs(angle) or even 0, and let the terminal success condition handle uprightness.

**4. Lessons supported or contradicted:**
- **Supported**: "Exponential distance reward works better than linear negative distance" - g1_c0 (exponential) vastly outperformed g1_c1 (linear). Strong confirmation.
- **Supported**: "Relax ground contact conditions with partial bonuses helps" - g1_c0's partial bonus (one leg) yielded some bonus, while g1_c1's strict both-legs requirement yielded zero.
- **Supported**: "Reduce terminal failure penalty from -100 to -50 improves results" - both candidates improved dramatically over Generation 0.
- **Supported**: "Scale down shaping penalties to avoid overwhelming positive signals" - g1_c0's lower penalties (-1.5, -2.0) outperformed g1_c1's higher penalties (-2.0, -3.0).
- **Contradicted**: "Increase fuel penalties to discourage engine firing" - both candidates had increased fuel penalties (-2 main, -1 side) but showed zero fuel penalty impact. This suggests the fuel penalty mechanism is broken (likely due to action space mismatch), not that the penalty magnitude is wrong.
- **Contradicted (or needs investigation)**: "Action-dependent discrete fuel penalties work" - the evidence shows fuel_penalty=0.0 despite the code having discrete checks. This strongly suggests the action space is continuous, not discrete, making the current implementation ineffective.
- **Needs investigation**: The action_mean=0.0 anomaly. If the agent is truly taking no actions, the reward function may be inadvertently punishing all action-taking, leading to a do-nothing policy. This could explain why no landings occur despite improved shaping.

## Recent environment lessons
- failure_mode: Scale down shaping penalties by 2-3x to allow the agent to explore and receive positive feedback. Use moderate magnitudes (e.g., -1.0 for velocity, -1.5 for distance) and consider adding positive shaping near the goal.
- failure_mode: Reduce the failure penalty (e.g., -50 for crash, -20 for timeout) or make it proportional to distance from pad at termination. Consider using a success bonus (e.g., +100) without a large failure penalty.
- failure_mode: Relax the conditions: allow bonus for at least one leg contact, expand the near-pad region (e.g., |x|<0.3, y<0.2), or give partial bonuses for progress (e.g., small bonus for being near pad, larger for both legs).
- reward_pattern: Increase fuel penalties significantly (e.g., -2.0 for main, -1.0 for side) to make the agent more conservative with engine usage, or tie fuel penalty to the magnitude of velocity change to encourage efficient control.
- repair_rule: Investigate the hidden evaluator's likely structure by analyzing successful reward functions from other envs or tasks. Consider adding positive shaping for proximity to pad and low speed, and ensure all components are scaled to be comparable in magnitude.
- general: Prefer using obs for simplicity and consistency, unless there is clear evidence that next_obs provides a meaningful advantage (e.g., for contact states that change after the step).
- failure_mode: Verify the action space format. If continuous, change fuel penalty to use action magnitude (e.g., fuel_penalty = -2.0 * abs(action[1]) for main engine). If discrete, ensure action values are integers.
- reward_pattern: Prefer exponential distance reward (e.g., 3.0 * exp(-2.0 * dist)) over linear negative distance. Increase scaling or use slower decay if more positive signal is needed.
- failure_mode: Further reduce or eliminate the terminal failure penalty (try -20 or 0). Rely on positive shaping signals to guide the agent toward successful landings. The hidden evaluator likely rewards success, not punishes failure.
- reward_pattern: Use relaxed conditions with partial bonuses (e.g., one-leg bonus 3.0, both-legs bonus 8.0) to provide shaping for progress toward landing. Expand near_pad region to make bonuses achievable.
- reward_pattern: Keep shaping penalty magnitudes moderate (e.g., velocity_penalty -1.5, angle_penalty -2.0). Avoid high penalties that overwhelm positive signals.
- failure_mode: Investigate whether the reward function inadvertently punishes all actions. Consider adding a small positive reward for being near pad with low velocity (e.g., +2.0 if dist<0.3 and |vy|<0.1). Ensure fuel penalty is correctly implemented for the action space.
- failure_mode: Use relaxed success criteria (vy<0.15, vx<0.15, angle<0.15) and positive exponential distance shaping. Avoid strict thresholds that make success nearly impossible during early learning.
- general: Further reduce shaping penalty magnitudes (try velocity_penalty -1.0, angle_penalty -1.0) or increase positive shaping (distance_reward scaling 5.0). Aim to make the generated reward more aligned with the hidden evaluator's scale.