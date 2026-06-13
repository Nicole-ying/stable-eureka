# Environment Memory

env_alias: Env-90b964d9
latest_generation: 1

## Latest reflection
## Reflection on Generation 1

### 1. What Worked
- **Significant improvement in selection score**: The best candidate (g1_c0) improved from -684.8 to -191.85 — a ~3.5x improvement. The generated return also improved from -92.4 to -191.85, but more importantly, the gap between generated and private return shrank from 288 to 99.4.
- **Action diversity achieved**: Action_mean = 0.33 with std = 0.94 for g1_c0, showing the agent now explores different actions (main engine, side engines, no action) instead of the previous single-action policy. This is a dramatic improvement from action_mean=2.0, std=0.0.
- **Component balance improved**: Distance penalty dropped from -306 to -6.4 — the linear penalty successfully prevented dominance. Velocity penalty (-22.5) and angle penalty (-5.7) are now in a reasonable range.
- **Positive progress shaping working**: Progress component gave +1.1 reward, indicating the agent is moving toward the pad and reducing speed.
- **Stability reward meaningful**: +11.2 from stability, suggesting the agent maintains upright posture.
- **Shorter episodes**: Episode length dropped from 105 to 66.7 — the agent is terminating faster, which could indicate more decisive behavior.
- **No import errors**: Both candidates passed validation successfully.

### 2. What Failed
- **Still no successful landing**: Landing bonus = 0.0, crash penalty = -50.0, terminal = -20.0. The agent crashes or times out without achieving a proper landing.
- **Generated-private gap still significant**: 99.4 points gap for g1_c0. While improved, the reward function still doesn't fully align with the hidden evaluator.
- **Candidate g1_c1 regressed severely**: -774.2 selection score, with a massive 655.7 gap. This candidate had action_mean = 1.63 (mostly main engine) and episode length 216 (very long episodes). The progress component was actually negative (-1.4), meaning the agent moved away from the pad.
- **Crash penalty dominates terminal outcomes**: -50 crash + -20 terminal = -70 total for non-landing terminations. This is still double-counting terminal outcomes despite the lesson about consolidation.
- **Effort penalty never triggered**: effort = 0.0 for both candidates. The threshold `dist < 0.3 and abs(vel_y) < 0.2` is still too strict — the agent never reaches this state.
- **Fuel penalty negligible**: -0.07 for g1_c0, -3.45 for g1_c1. The g1_c1 candidate fired engines much more (action_mean=1.63 vs 0.33) but the penalty was still small compared to other components.

### 3. What to Try Next
- **Fix terminal double-counting**: The code for g1_c0 still has separate `terminal_reward`, `landing_bonus`, and `crash_penalty` variables that all activate on `done`. Consolidate into a single terminal value: e.g., +200 for perfect landing, -10 for crash, -5 for timeout/other.
- **Increase landing bonus further**: The previous generation had +10, this one aimed for +150 but the code was truncated. Ensure the landing bonus is truly large (e.g., +200 to +500) and is the *only* terminal component.
- **Make crash penalty less punishing**: -50 crash penalty may be discouraging the agent from attempting risky maneuvers that could lead to learning. Reduce to -10 or consolidate into a single terminal value of -10 for non-landing.
- **Fix effort threshold**: Use a continuous function instead of hard thresholds. For example: `effort_penalty = -0.1 * (main + side) * exp(-distance/0.5) * exp(-abs(vel_y)/0.3)` — this smoothly penalizes engine use when close and slow, without a hard cutoff.
- **Strengthen progress shaping**: The current progress gives positive rewards for moving toward pad, reducing speed, and becoming upright. Increase the coefficients: try 0.5 for distance improvement, 0.8 for speed reduction, 0.3 for angle correction.
- **Add explicit reward for using side engines to correct angle**: The agent still doesn't use side engines effectively. Add a component like: `angle_correction_reward = 0.2 * (abs(angle) - abs(next_angle))` when side engine is fired — this directly rewards using side engines to improve posture.
- **Reduce episode length penalty**: The agent terminates at 66 steps on average. If the environment has a max step limit, the agent may be timing out rather than crashing. Consider adding a small per-step survival penalty (e.g., -0.1 per step) to encourage faster landing.
- **Apply lessons more aggressively**: The gap is still 99.4. The landing bonus needs to be large enough to overcome the cumulative negative reward over an episode. If expected non-terminal reward is -50 per episode, landing bonus should be +200 or more.

### 4. Lesson Support/Contradiction
- **Supported**: "Normalize component magnitudes" — Switching from quadratic (-0.5*d²) to linear (-0.1*d) distance penalty dramatically improved balance. Distance penalty went from -306 to -6.4.
- **Supported**: "Ensure different actions produce measurably different rewards" — Action diversity improved from std=0.0 to std=0.94, and the agent now uses multiple action types.
- **Supported**: "Consolidate terminal logic into a single component" — g1_c0 still has separate landing_bonus, crash_penalty, and terminal_reward. The -70 total for non-landing termination is likely suboptimal. This lesson was partially followed but needs full implementation.
- **Partially contradicted**: "Set landing bonus to at least 10x expected cumulative penalty" — The landing bonus was increased (aimed for +150), but the cumulative penalty was also reduced significantly (-53 total for g1_c0 vs -684 for g0_c1). The ratio improved but still no landing occurred. The lesson needs refinement: the bonus must be high enough relative to the *actual* path taken, not just the expected total.
- **Supported**: "Use progressive thresholds or continuous functions instead of hard thresholds" — The effort penalty threshold (dist < 0.3) was loosened from dist < 0.2, but still never triggered. This confirms hard thresholds are fragile.
- **New insight**: The g1_c1 candidate shows that simply copying the same structure with minor tweaks can produce wildly different results. The difference between g1_c0 and g1_c1 was subtle (mostly coefficient differences), yet g1_c1 regressed severely. This suggests the reward landscape is highly sensitive to parameter choices.
- **Emerging lesson**: When improving a reward function, make changes one at a time and verify each change improves the score. The jump from g0_c1 (-684) to g1_c0 (-191) was a big improvement, but the landing bonus is still not being achieved. The next iteration should focus specifically on making landing achievable.

## Recent environment lessons
- failure_mode: Normalize component magnitudes so that the maximum possible contribution of any single component is at most 2-3x the others. Use linear penalties (e.g., -0.1*distance) instead of quadratic (-0.5*distance²) to avoid exponential dominance. Verify balance by computing expected ranges for each component.
- failure_mode: Increase the terminal landing bonus dramatically (e.g., +100 or more) to make landing the dominant objective. Reduce or remove continuous penalties that don't directly relate to landing success. Test that the reward function's optimal policy would also score well on the private evaluator.
- failure_mode: Ensure that different actions produce measurably different rewards. Add action-dependent shaping: e.g., reward using side engines to reduce angular velocity, penalize firing main engine when far from landing zone. Remove or reduce components that don't depend on action choice.
- failure_mode: Set thresholds based on empirical observation of the agent's trajectory. Use progressive thresholds that start loose and tighten as the agent improves. Alternatively, use continuous functions (e.g., exp(-distance)) instead of hard thresholds. Monitor component activation rates during training.
- failure_mode: Consolidate all terminal logic into a single component. Use a single `if done:` block that assigns one terminal value based on the landing quality. Avoid multiple components that each check `done`.
- reward_pattern: Set terminal landing bonus to at least 10x the expected cumulative penalty over an episode. For example, if expected total penalty is -500, set landing bonus to +500 or more. Alternatively, reduce cumulative penalties so that the net reward for a successful landing is clearly positive.
- mutation_rule: In the prompt, explicitly state 'Do not use any import statements. All math operations must use built-in Python operators (e.g., **0.5 for sqrt).' Add a validation check before generation that rejects any code with import statements.
- general: Start with a minimal reward function: 1) Large terminal bonus for successful landing, 2) Small penalty for distance from pad, 3) Small penalty for high velocity. Add complexity only after observing the agent's behavior. Use the private eval return as the primary metric for success.
- reward_pattern: Use linear or sub-linear penalties for distance and other large-magnitude state variables to prevent any single component from dominating the reward signal. Scale coefficients so that each component stays within a similar range (e.g., -10 to +10) during normal operation.
- failure_mode: Consolidate all terminal logic into a single component: assign +200 for a perfect landing, -10 for a crash, and -5 for timeout. This avoids double-counting and provides a clean, interpretable terminal signal.
- failure_mode: Use continuous functions instead of hard thresholds for conditional rewards. For example: effort_penalty = -0.1 * (main + side) * exp(-distance/0.5) * exp(-abs(vel_y)/0.3) smoothly penalizes engine use when close and slow, providing a gradient even when the agent is far from the target state.
- reward_pattern: Set the landing bonus to at least 10x the expected cumulative non-terminal penalty along a successful landing trajectory (e.g., +200 to +500). Also, reduce crash penalties to avoid discouraging exploration. Test that the bonus is actually achievable by checking if the agent ever reaches the landing state.
- general: Make only one change at a time between generations and validate its impact before adding the next change. This isolates the effect of each modification and prevents regression from compounded parameter interactions.
- repair_rule: Strengthen progress shaping coefficients (e.g., 0.5 for distance improvement, 0.8 for speed reduction, 0.3 for angle correction) and add explicit rewards for using side engines to correct angle (e.g., angle_correction_reward = 0.2 * (abs(angle) - abs(next_angle)) when side engine is fired). This directly incentivizes the actions needed for landing.
- prompt_rule: In the prompt, explicitly state: 'Use exactly one variable for terminal outcomes. Do not use separate landing_bonus, crash_penalty, or terminal_reward variables. Assign a single value to a single variable when done is True.' This reduces ambiguity and ensures the LLM follows the consolidation rule strictly.