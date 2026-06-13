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