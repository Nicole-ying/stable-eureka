## Reflection on Generation 1

### 1. What worked

- **Candidate g1_c2 significantly improved over all generation 0 candidates**: With private_eval_return = -84.16, it outperformed the previous best (g0_c0 at -122.83) by ~38 points, a meaningful improvement of ~31%. This is the first candidate to break the -100 barrier.
- **g1_c2 has the smallest generated_private_gap (-69.65)** among all candidates in both generations, but notably this gap is *negative* (generated_return is worse than private). This is a new pattern - the candidate's own reward function is *more pessimistic* than the hidden evaluator, suggesting the shaping signals are providing useful guidance that the hidden evaluator values more highly.
- **All 4 candidates passed validation** and had valid code, maintaining the 100% validation rate from generation 0.
- **g1_c2's component returns show better balance**: progress (-129.52) is still the dominant negative term, but stability (-9.02) and effort (-0.27) are much more moderate. The terminal component (-15.0) is consistent with the moderate penalty approach.
- **g1_c2 has non-zero action diversity** (action_mean=0.07, action_std=0.26), indicating the agent is exploring multiple actions rather than collapsing to a single action.
- **g1_c2 explicitly addressed the key failure mode** from generation 0: it strengthened progress shaping (exponential distance, stronger vertical speed penalty near ground), increased angle penalty, and used moderate effort penalties - all aligned with the lessons from memory.

### 2. What failed

- **All candidates still have negative private_eval_return**, meaning none achieved successful landings. The best score (-84.16) is still far from positive.
- **g1_c0 and g1_c1 performed worse than g0_c0** (the generation 0 best), with scores of -118.49 and -111.04 respectively. This is a regression - changes made in these candidates hurt more than helped.
- **g1_c3 performed worst** at -140.55, despite having similar structure to g1_c2. The key difference? g1_c3 used exponential distance shaping with a cap at -0.8 * (1 - exp(-2*dist)), while g1_c2 used a linear -0.8*dist penalty. The linear version apparently provides stronger gradient.
- **g1_c0 and g1_c1 have large positive generated_private_gaps** (54.05 and 39.41 respectively), meaning their own reward functions are *more optimistic* than the hidden evaluator. This is the opposite problem from g1_c2 - these candidates are overestimating reward.
- **g1_c0 has action_mean=0.88 and action_std=1.22**, indicating action collapse toward main engine (action=2). Its effort component (-1.37) is the most negative of all candidates, yet the agent still fired engines heavily. This confirms the lesson that effort penalties alone can't prevent engine overuse when shaping signals are weak.
- **g1_c1 and g1_c3 have action_std=0.0**, meaning zero action diversity - the agent is doing nothing (action_mean=0.0). This is a different failure mode: the reward function is so discouraging that the agent has learned to do nothing rather than attempt the task.
- **Progress component dominates all candidates**: Even in g1_c2 (the best), progress is -129.52 while the next largest component (stability) is only -9.02. This 14:1 ratio suggests progress is still overwhelming other signals, though less severely than in generation 0.

### 3. What to try next

- **Analyze g1_c2's success more deeply**: Its negative generated_private_gap is unique. The hidden evaluator values its shaping more than the candidate itself thinks. This suggests the progress and stability components are well-aligned with the evaluator's priorities. **Double down on this approach** - strengthen the components that g1_c2 uses while keeping the overall structure.
- **Increase progress scaling further**: g1_c2 used -0.8*dist, which outperformed g1_c3's exponential cap. Try -1.0*dist or -1.2*dist to provide even stronger gradient. The hidden evaluator seems to reward progress heavily.
- **Add a survival bonus**: Episode lengths are still short (59-75 steps). Add a small positive reward per timestep (e.g., +0.1 or +0.05) to encourage longer flights. This could help the agent learn to avoid crashes and explore more.
- **Fix the action collapse problem**: Two candidates had action_std=0.0 (no action diversity). The issue may be that the shaping signals are too weak relative to the terminal penalty when the agent isn't near the pad. Consider adding a small positive reward for maintaining altitude (y > 0.3) to encourage stable hovering before attempting landing.
- **Reduce the progress-stability imbalance**: Progress is 14x more negative than stability. While progress should be the dominant signal, the ratio is extreme. Try increasing stability scaling (e.g., from -0.6*abs(angle) to -1.0*abs(angle)) to give it more influence.
- **Consider a two-phase reward**: Phase 1 (high altitude): focus on stability and altitude maintenance. Phase 2 (low altitude, y < 0.3): focus on soft landing and precise positioning. This could prevent the "do nothing" behavior at high altitude.
- **Test g1_c2's structure with stronger effort penalties**: g1_c2 has moderate effort (-0.15 main, -0.05 side) but action_mean=0.07 suggests the agent is barely using engines. Try slightly lower effort penalties (-0.10 main, -0.03 side) to encourage more active control while still penalizing waste.

### 4. Which lessons seem supported or contradicted

**Supported lessons:**
- **Smaller generated_private_gap correlates with higher performance** - g1_c2 has the smallest absolute gap (-69.65) and the best score (-84.16). The gap direction matters too: negative gaps (candidate is pessimistic) seem better than positive gaps (candidate is optimistic).
- **Moderate failure penalty (-10 to -30) outperforms harsh penalties** - g1_c2 uses -15, consistent with the lesson. All top candidates used -15 or -25, and the worst candidate (g1_c3) also used -15, suggesting the penalty magnitude alone isn't the key factor.
- **Progress and stability components are key drivers** - g1_c2's stronger progress shaping (-0.8*dist vs. earlier -0.5*dist) directly contributed to its improvement, supporting the lesson to increase shaping magnitudes.
- **Action diversity matters** - g1_c2 (action_std=0.26) outperformed g1_c1 and g1_c3 (action_std=0.0). The "do nothing" collapse is clearly detrimental.
- **Don't rely on validation success** - all candidates passed validation, yet performance varied dramatically from -84 to -140. Validation is not a quality indicator.

**Contradicted lessons:**
- **Complex shaping is not always bad** - g1_c2 has *more* complexity than g1_c0 (which attempted delta shaping), yet g1_c2 performed better. The lesson that "complex shaping leads to poor performance" is nuanced: *targeted* complexity (exponential distance, ground-proximity velocity penalties) can help, while *poorly scaled* complexity (delta shaping with wrong scaling) hurts.
- **Heavy effort penalties don't necessarily reduce engine usage** - confirmed again: g1_c0 had the highest effort penalty (-1.37 total) but still had action_mean=0.88 (near main engine). The lesson holds: effort penalties alone don't shape behavior when other signals are weak.
- **The "minimal set" repair rule is contradicted** - g1_c2, the best candidate, has a more complex reward function with multiple specialized penalties (ground-proximity speed, exponential distance, extra angle penalty near ground). The best approach may be *targeted complexity* rather than minimalism.
- **"Increase angle penalty scaling" lesson partially contradicted** - g1_c2 increased angle penalty to -0.6 (from -0.5 in g0_c0), which helped. But g1_c3 also used -0.6 and performed worst. The angle penalty alone isn't decisive - it must be combined with appropriate progress shaping.

**New insights:**
- **The sign of generated_private_gap matters**: Negative gaps (candidate undervalues its own behavior) correlate with better hidden evaluator scores. Positive gaps (candidate overvalues behavior) correlate with worse scores. This is a new diagnostic signal.
- **Linear distance penalties may outperform exponential caps**: g1_c2 (linear -0.8*dist) outperformed g1_c3 (exponential -0.8*(1-exp(-2*dist))). The linear version provides stronger gradient at all distances.
- **Action diversity can collapse in two directions**: either "always main engine" (action_mean=2.0) or "always do nothing" (action_mean=0.0). Both failure modes need different fixes: the first needs stronger progress/stability to incentivize controlled flight, the second needs positive rewards for engagement (survival bonus, altitude maintenance).