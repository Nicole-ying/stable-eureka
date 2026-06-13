# Environment Memory

env_alias: Env-90b964d9
latest_generation: 2

## Latest reflection
## Reflection on Generation 2

### 1. What worked

- **g2_c0 is the best candidate so far** with private_eval_return = -100.89, representing a slight improvement over g1_c2 (-84.16 → -100.89 is worse, but g2_c0 is the best in this generation). Wait — actually -100.89 is *worse* than -84.16. This generation **regressed** — no candidate beat g1_c2's -84.16.
- **g2_c0 has a negative generated_private_gap** (-36.77), continuing the pattern that conservative/pessimistic shaping correlates with better performance. This is the most negative gap in this generation.
- **g2_c0 has moderate action diversity** (action_mean=0.11, action_std=0.32), indicating the agent is exploring multiple actions rather than collapsing.
- **g2_c0 has a landing_shaping bonus** that provides positive reinforcement near the pad — a new feature not present in g1_c2. This is a creative addition.
- **All 4 candidates passed validation** with no errors, maintaining the 100% validation rate.
- **g2_c3 has very high action diversity** (action_mean=0.94, action_std=1.29), suggesting the agent is actively using engines. However, this didn't translate to better performance.

### 2. What failed

- **All candidates performed worse than g1_c2 (-84.16)**. The best (g2_c0 at -100.89) is ~17 points worse. This is a significant regression, indicating that the changes made in this generation were net harmful.
- **No candidate achieved a positive private_eval_return** or even broke -80. The best score is still deeply negative.
- **g2_c0 has a very large generated_private_gap magnitude** (-36.77) compared to g1_c2 (-69.65). Wait — -36.77 is *smaller* magnitude than -69.65. The gap *shrank* (from -69.65 to -36.77), which should be good according to the lesson, but performance *worsened*. This **contradicts** the lesson that smaller absolute gap = better performance.
- **g2_c1 performed worst** at -423.67 — a catastrophic regression. Its action_mean=1.45 and action_std=1.09 indicate heavy main engine firing, and its progress component (-251.35) is the most negative ever seen. The survival bonus (+0.05) clearly didn't help.
- **g2_c2 performed poorly** at -244.63 with action_std=0.40 — moderate diversity but terrible returns. Its stability component (-48.67) is the most negative stability ever, suggesting the angle penalties overwhelmed the agent.
- **g2_c3** at -195.79 has action_mean=0.94 and action_std=1.29 — the agent is firing engines actively but crashing. Its effort penalty (-1.15 total) is high but didn't prevent engine overuse.
- **The landing_shaping bonus in g2_c0** only applies when dist < 0.2 AND speed < 0.2 — a very narrow window. The agent almost never reaches this state, so the bonus is effectively unused.
- **Episode lengths are still short** (55-90 steps), with no successful landings.

### 3. What to try next

- **Revert to g1_c2's exact structure as the baseline** and make *minimal* changes. The regression from g1_c2 (-84.16) to g2_c0 (-100.89) shows that increasing distance penalty from -0.8 to -1.0 and angle penalty from -0.6 to -0.8 was harmful. **Try decreasing** these coefficients instead — perhaps the progress signal is *too strong* and overwhelming other components.
- **Test smaller distance penalties**: Try -0.6*dist or even -0.4*dist. The fact that g1_c2 (-0.8*dist) outperformed g2_c0 (-1.0*dist) suggests that stronger isn't always better. The optimal may be lower than -0.8.
- **Reduce angle penalty back to -0.6** (g1_c2's value). The increase to -0.8 in all g2 candidates correlated with worse performance. The progress-stability balance may have been better at the original ratio.
- **Remove the landing_shaping bonus** — it's unused and adds unnecessary complexity. The narrow condition (dist < 0.2 AND speed < 0.2) is almost never reached.
- **Add a continuous altitude reward** instead: reward the agent for maintaining y > 0.3 (high altitude) with a small positive bonus. This could prevent the "crash immediately" behavior and give the agent time to learn controlled descent.
- **Consider a survival bonus** (+0.05 or +0.1 per timestep) that doesn't depend on being near the pad. g2_c1 tried this but with too many other changes — isolate this variable.
- **Target a generated_private_gap of -50 to -60** (between g1_c2's -69.65 and g2_c0's -36.77). The gap shrinking from -69.65 to -36.77 correlated with *worse* performance, suggesting there's an optimal gap magnitude that's not zero.
- **Monitor the progress:stability ratio**: g1_c2 had -129.52:-9.02 (14.4:1). g2_c0 has -118.55:-3.85 (30.8:1) — the ratio *increased* despite increasing angle penalty. This means progress became relatively even more dominant. The issue may be that progress scaling increased (from -0.8 to -1.0) but stability scaling increased less (from -0.6 to -0.8), making the ratio worse.

### 4. Which lessons seem supported or contradicted

**Supported lessons:**
- **Negative generated_private_gap correlates with better performance within a generation** — g2_c0 (gap=-36.77) is the best, while g2_c1 (gap=+117.65) is the worst. The pattern holds: pessimistic shaping > optimistic shaping.
- **Moderate failure penalty (-10 to -30) is fine** — all candidates used -15, consistent with the lesson. Performance varied widely despite identical terminal penalties.
- **Action diversity matters** — g2_c0 (action_std=0.32) and g2_c3 (action_std=1.29) have non-zero diversity and outperform g2_c1 (action_std=1.09, but action_mean=1.45 — near main engine) and g2_c2 (action_std=0.40 but terrible returns).
- **Validation is not a quality signal** — all passed, performance ranged from -100 to -423.
- **Linear distance penalties are fine** — all candidates used linear -1.0*dist, consistent with the lesson. But performance regressed, suggesting the *coefficient* matters more than the formulation.
- **Targeted complexity can help** — g2_c0's landing_shaping bonus didn't help (too narrow), but the general approach of adding specialized components is sound.

**Contradicted lessons:**
- **"Smaller generated_private_gap = better performance" is contradicted** — g1_c2 had gap=-69.65 and scored -84.16. g2_c0 has gap=-36.77 (smaller magnitude) but scored -100.89 (worse). The relationship is NOT monotonic — there's an optimal gap magnitude that's not zero. This is a critical new insight.
- **"Stronger progress gradient helps" is contradicted** — increasing distance penalty from -0.8 to -1.0 made performance worse (-84.16 → -100.89). The lesson that "linear outperforms exponential" still holds (g1_c2 > g1_c3), but "stronger is better" does not.
- **"Increase stability scaling to balance ratio" is contradicted** — increasing angle penalty from -0.6 to -0.8 actually *worsened* the progress:stability ratio (from 14.4:1 to 30.8:1) because progress increased more. The ratio matters more than absolute values.
- **"Survival bonus helps prevent do-nothing" is partially contradicted** — g2_c1 added +0.05 survival bonus but scored -423.67 (worst ever). The bonus alone doesn't help if other components are poorly scaled.
- **"Add landing_shaping for precise approach" is contradicted** — g2_c0's landing_shaping bonus was too narrow to ever activate. The condition dist < 0.2 AND speed < 0.2 is almost never reached with current behavior.

**New insights:**
- **There is a U-shaped relationship between progress scaling and performance**: -0.8*dist (g1_c2) outperforms both -1.0*dist (g2_c0) and -0.5*dist (g0 candidates). The optimal coefficient may be near -0.8.
- **The progress:stability ratio should be monitored, not just individual coefficients**: g1_c2 had 14.4:1 and scored -84.16. g2_c0 had 30.8:1 and scored -100.89. A ratio of ~15:1 may be near-optimal.
- **Component returns from the candidate's perspective (generated) vs. evaluator's perspective (private) provide different signals**: g1_c2 had generated progress=-129.52 and private progress component is hidden, but the gap analysis suggests the evaluator values progress more than the candidate thinks.
- **The "do nothing" failure mode (action_std=0.0) and "always main engine" failure mode (action_mean≈2.0) are distinct problems**: g2_c1 has action_mean=1.45 (near main engine) and terrible performance. The fix for engine overuse is different from the fix for inaction.

## Recent environment lessons
- reward_pattern: Focus on making the progress and stability components more aligned with the hidden evaluator. Use g0_c0's structure as a baseline and try increasing the scaling factors on these components while monitoring generated_private_gap.
- general: Do not rely on validation success as an indicator of reward quality. Use private_eval_return and generated_private_gap as the primary metrics for evaluating candidate effectiveness.
- reward_pattern: When optimizing reward functions, target a negative generated_private_gap by ensuring shaping components are conservative but directional, rather than optimistic. Use black-box selection feedback to monitor and reduce generated/private mismatch.
- failure_mode: Add a small positive survival bonus per timestep (e.g., +0.05 or +0.1) and consider phase-based rewards that incentivize altitude maintenance at high altitude to prevent 'do nothing' behavior.
- failure_mode: Strengthen progress and stability components to provide stronger gradient for controlled descent, rather than relying on effort penalties to shape behavior. Ensure shaping signals dominate over terminal penalties.
- mutation_rule: Use linear distance penalties (e.g., -coeff * dist) rather than exponential or capped formulations to maintain strong gradient throughout the episode.
- reward_pattern: Prioritize progress shaping as the primary signal, but consider gradually increasing stability scaling (e.g., from -0.6*abs(angle) to -1.0*abs(angle)) to provide more balanced guidance without undermining progress dominance.
- general: Do not rely on validation success as a signal of reward quality. Use private_eval_return and generated_private_gap as primary quality metrics, and focus on component balance and shaping alignment.
- reward_pattern: Maintain failure penalties in the moderate range (-10 to -30). Avoid harsh penalties (e.g., -50 or lower) that can dominate the reward signal and discourage exploration.
- general: Add specialized shaping components that target observed failure modes (e.g., ground-proximity velocity penalties, altitude-dependent angle penalties) rather than relying on minimal component sets. Ensure each component is properly scaled.
- prompt_rule: When generating reward candidates, explicitly instruct the LLM to produce reward functions that are conservative (slightly pessimistic) rather than optimistic, to achieve negative generated_private_gap. Use examples where conservative shaping outperforms optimistic shaping.
- repair_rule: If action_std is 0.0, add positive rewards for engagement (survival bonus, altitude maintenance) rather than just adjusting penalties. If action_mean is near 2.0, strengthen progress/stability shaping to incentivize controlled flight.
- general: Keep distance penalty coefficient near -0.8 rather than increasing it. There is a U-shaped relationship where -0.8 outperforms both -1.0 and -0.5.
- general: Monitor the progress:stability ratio when adjusting coefficients. Target a ratio near 15:1 rather than independently increasing stability penalties.
- general: Target a generated_private_gap of -50 to -60 rather than trying to minimize the absolute gap. Use black-box selection feedback to maintain this range.
- failure_mode: Either widen the activation conditions (e.g., dist < 0.5, speed < 0.5) or use a continuous altitude reward that activates earlier in the trajectory.
- failure_mode: Isolate survival bonus as a single variable change. Test it with the proven g1_c2 baseline rather than combining it with multiple other adjustments.
- reward_pattern: Focus on shaping component coefficients rather than terminal penalty values. Keep terminal at -15 but tune progress and stability scaling independently.
- failure_mode: Increase main engine effort penalty (e.g., -0.20) and add a condition-based vertical speed penalty that activates when the main engine is fired at low altitude.
- reward_pattern: Target a progress:stability ratio of approximately 15:1. This can be achieved by keeping distance penalty at -0.8 and angle penalty at -0.6, or by adjusting both coefficients proportionally.