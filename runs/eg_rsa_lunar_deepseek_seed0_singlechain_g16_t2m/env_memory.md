# Environment Memory

env_alias: Env-90b964d9
latest_generation: 9

## Latest reflection
## Reflection on Generation 9

### 1. What worked

- **Action mean is now healthy (0.13)**: The action mean dropped from **1.9983 (g8)** to **0.1304 (g9)** — a **93.5% reduction**. The throttle penalty (-0.2 per main engine activation) and survival cost (-0.05 per step) successfully broke the degenerate full-throttle behavior. The agent is now using engines sparingly.
- **Action std is reasonable (0.54)**: The action standard deviation of 0.54 indicates the agent is exploring different actions rather than stuck in a single action. This is a dramatic improvement from the near-zero variance in g8.
- **Fuel efficiency penalty is being applied meaningfully**: At **-4.06**, the fuel penalty is small in absolute terms but present. The agent used main engine only ~20 times per episode (4.06 / 0.2 = ~20 main engine activations), which is vastly more fuel-efficient than the g8 behavior.
- **Episode length dropped significantly**: From **1000 (g8)** to **75.9 (g9)** — a **92.4% reduction**. The agent is now landing or crashing quickly rather than hovering for the full episode. The survival cost created urgency.
- **Velocity penalty is having an effect**: At **-81.85**, the velocity penalty is now 8.5x the shaping reward — the strongest penalty component. This is actually exceeding the target, suggesting the agent is moving with high velocity (possibly crashing).
- **Generated return is low but not catastrophic**: At **-77.1**, the generated return is negative but the agent is actually behaving more naturally (short episodes, varied actions) rather than reward-hacking.

### 2. What failed

- **Private_eval_return dropped catastrophically**: From **-11.75 (g8)** to **-115.0 (g9)** — a **10x worsening**. This is the worst private return ever observed. The agent is now actively failing at the task.
- **The generated-private gap remains large**: At **37.89**, the gap is much smaller than g8 (2834) but still 33% of the private return magnitude. The gap direction reversed — now the generated return is *less negative* than private return, meaning the generated reward is overestimating performance.
- **No terminal reward earned (0.0)**: The terminal bonus of 1000 was never earned. The agent either crashes before reaching the landing condition or never achieves the landing criteria. The terminal condition may be too strict or the agent is dying before reaching the pad.
- **Shaping reward collapse**: From **1127.92 (g8)** to **9.62 (g9)** — a **99.1% reduction**. The shaping multiplier reduction (2.0 → 1.5) alone doesn't explain this; the agent is spending very little time near the pad (episode length 75.9 vs 1000). The agent may be crashing before getting close to the pad.
- **Angle penalty nearly zero (-0.81)**: The angle penalty is negligible, suggesting the agent is maintaining good orientation but the short episodes prevent any meaningful penalty accumulation.
- **The agent may be crashing**: Episode length of 75.9 with no terminal reward and high velocity penalty (-81.85) suggests the agent is crashing into the ground or water, not landing successfully. The survival cost (-0.05 per step) creates urgency but may be causing the agent to descend too quickly and crash.

### 3. What to try next

- **Reduce the survival cost**: The -0.05 per step creates -3.8 per episode (75.9 steps), which is small but may be encouraging overly aggressive descent. Reduce to **-0.02** or **-0.01** to maintain some urgency without causing crashes.
- **Reduce the velocity penalty**: The -0.8 multiplier on velocity is too aggressive, especially combined with the survival cost. Reduce to **-0.4** or **-0.3** to allow controlled descent without punishing all movement.
- **Make the terminal condition more achievable**: The current terminal condition requires `abs(obs[0]) < 0.15, abs(obs[1]) < 0.15, abs(obs[2]) < 0.3, abs(obs[3]) < 0.3, abs(obs[4]) < 0.2`. Widen these tolerances (e.g., `abs(obs[0]) < 0.3, abs(obs[1]) < 0.3, abs(obs[2]) < 0.5, abs(obs[3]) < 0.5, abs(obs[4]) < 0.3`) to make landing achievable for the current policy.
- **Add a descent reward**: Add `0.3 * (obs[3] < -0.1) * np.exp(-1.0 * np.sqrt(obs[0]**2 + obs[1]**2))` to reward controlled descent toward the pad, providing an alternative to the shaping reward that requires proximity.
- **Increase shaping multiplier slightly**: The shaping collapse from 1127 to 9.6 is too extreme. Increase from 1.5 to **2.0** but with a slower decay (e.g., `np.exp(-2.0 * np.sqrt(obs[0]**2 + obs[1]**2))`) to provide a longer-range signal that guides the agent toward the pad before it crashes.
- **Keep the throttle penalty but reduce it slightly**: The -0.2 per main engine activation is working well (action mean 0.13). Reduce to **-0.15** to allow slightly more main engine use for controlled landing while still preventing full-throttle behavior.
- **Add a "leg contact" or "ground contact" reward**: Add a small bonus (e.g., 0.5) when the agent is on the ground with low velocity, bridging to the terminal bonus. This helps the agent learn that being near the ground is good.

### 4. Which lessons seem supported or contradicted

- **Supported**: "Add a strong throttle penalty (e.g., -0.2 per main engine activation) to break degenerate full-throttle behavior." — The action mean dropped from 1.9983 to 0.1304, confirming this works. The throttle penalty was the right intervention.
- **Supported**: "Add a time penalty or survival cost to create urgency to land quickly." — Episode length dropped from 1000 to 75.9, confirming the survival cost creates urgency. However, the effect was too strong — the agent is now crashing.
- **Supported**: "Reduce the terminal bonus further (e.g., to 1000) and ensure the soft landing bonus is separate and smaller." — The terminal bonus was reduced to 1000 and the soft landing bonus was removed. However, no terminal reward was earned, suggesting the condition is too strict.
- **Supported**: "Add validation criteria that monitor action mean (reject if > 1.5) and generated-private gap (reject if > 500)." — The action mean is now healthy (0.13), and the gap is small (37.89). These criteria would have caught g8 and prevented its acceptance.
- **Contradicted (partially)**: "The 20-40% penalty-to-shaping ratio guideline is conservative when terminal bonuses are large and achievable." — In g9, the velocity penalty is 850% of shaping reward (far above 20-40%), and the agent is failing. The guideline may need to be revisited when shaping is very small.
- **New lesson emerging**: "The survival cost and throttle penalty, while effective at breaking degenerate behavior, can overshoot and cause the agent to crash before reaching the pad. Reduce penalties incrementally and ensure the agent can still learn a successful landing policy."
- **New lesson emerging**: "When reshaping a reward function from degenerate to healthy, the agent may need a gentler gradient to learn the new behavior. Start with moderate penalties and gradually increase them as the policy improves, rather than applying full-strength penalties immediately."
- **New lesson emerging**: "The terminal condition tolerances must be achievable given the current policy's behavior. If the agent is crashing before reaching the pad, either widen the tolerances or add intermediate rewards (e.g., descent reward, proximity bonus) to guide the agent toward the landing zone."
- **New lesson emerging**: "A sudden collapse in shaping reward (99% reduction) combined with no terminal reward indicates the agent is not reaching the pad at all. This suggests the penalties are too strong relative to the positive signal, causing the agent to crash before learning controlled descent."

## Recent environment lessons
- general: Target penalty components to be 20-40% of the shaping reward magnitude. If a penalty is preventing the agent from achieving the terminal condition, reduce it iteratively until the agent can complete the task while still avoiding extreme behaviors.
- failure_mode: Add a throttle penalty (e.g., -0.1 * action[0]**2) to discourage excessive thrust, and monitor action mean and std as key health indicators.
- failure_mode: Reduce the terminal bonus (e.g., to 1500 or 1000) and add penalties on excessive thrust and rewards for controlled descent to balance the reward structure.
- failure_mode: Make incremental changes (e.g., 25% adjustment per generation) and monitor behavioral metrics like action mean/std and episode length to detect regime shifts early.
- general: Consider that the 20-40% penalty-to-shaping ratio guideline is conservative when terminal bonuses are large and achievable; adjust targets based on actual behavioral outcomes.
- general: Add validation criteria that monitor action mean and std thresholds (e.g., action mean < 1.5) to flag degenerate policies before accepting candidates.
- failure_mode: Add a strong throttle penalty (e.g., -0.2 * (action[0]**2 + action[1]**2)) or increase the fuel penalty to at least 10-20% of the terminal bonus magnitude. Target a per-episode penalty of -100 to -200.
- failure_mode: Make the soft landing bonus conditional on being very close to the pad (e.g., |x|<0.1 and |y|<0.1) so it bridges to landing rather than competing with it. Alternatively, implement it as a separate component for independent monitoring.
- failure_mode: Add a hard validation threshold to reject candidates when the generated-private gap exceeds 1000 or the private return is below -10. Do not accept such candidates without repair.
- failure_mode: Reduce the terminal bonus further (e.g., to 1000) and add a time penalty or survival cost (e.g., -0.1 per step) to create urgency to land quickly. Ensure the soft landing bonus is separate and smaller.
- failure_mode: Add validation criteria that monitor action mean (reject if > 1.5) and generated-private gap (reject if > 500) to flag degenerate policies before accepting candidates.
- general: Make incremental changes per generation (e.g., adjust one or two components at a time by no more than 25-50%) and monitor behavioral metrics like action mean/std and component returns to isolate effects.
- general: Implement the soft landing bonus as a separate component (e.g., 'soft_landing') to allow independent monitoring, scaling, and diagnosis of its effect on behavior.
- failure_mode: Apply penalties incrementally rather than at full strength. Start with a lower survival cost (-0.01 to -0.02) and a slightly reduced throttle penalty (-0.15), and gradually increase them as the policy improves. Ensure the positive signal (shaping) remains strong enough to guide the agent toward the pad.
- reward_pattern: Widen terminal condition tolerances (e.g., position < 0.3, velocity < 0.5, angle < 0.3) to make landing achievable for the current policy. Add intermediate rewards such as a descent reward or ground contact bonus to bridge the gap between survival and successful landing.
- general: Phase in penalties over multiple generations: start with moderate penalties and increase them as the policy adapts. Use validation criteria to detect if the policy is crashing (e.g., episode length < 50, velocity penalty > -100) and adjust accordingly.
- general: Monitor shaping reward as a leading indicator of whether the agent is reaching the pad. If shaping reward drops below 10% of its previous value, the agent is likely crashing before descent. Reduce penalties or add a longer-range proximity signal to guide the agent.
- failure_mode: When the gap reverses sign after a redesign, check whether the generated reward is rewarding behaviors that the hidden evaluator penalizes. Add a penalty for crashing (e.g., negative terminal reward when done but landing conditions not met) to align the generated signal with the hidden evaluator.
- mutation_rule: Use a smaller survival cost (-0.01 to -0.02) combined with a descent reward that rewards moving downward toward the pad. This creates urgency without punishing the agent for taking the time needed to land safely.
- general: Keep the velocity penalty-to-shaping ratio below 2:1 to ensure the agent still has incentive to move toward the pad. If the ratio exceeds 5:1, reduce the velocity multiplier or increase the shaping multiplier.