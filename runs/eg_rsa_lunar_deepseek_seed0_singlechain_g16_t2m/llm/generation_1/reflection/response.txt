**Reflection on Generation 1**

**1. What worked**

- **Shaping reward dominates positively**: The shaping component returned +7691, which is now vastly larger than penalties (-12.5 velocity, -13.4 angle). The 15x multiplier successfully overcame the previous issue of penalties dominating.
- **Velocity and angle penalties are now negligible**: At -12.5 and -13.4 respectively, these penalties are ~600x smaller than the shaping reward, effectively removing their influence on the total reward.
- **Action mean is high (1.93)**: The agent is actively moving (mean action ~2.0) rather than taking no-ops. This indicates the reward function successfully incentivizes engine use.
- **Episode length is long (901 steps)**: The agent sustains flight for nearly the entire 1000-step horizon, suggesting it doesn't crash early and explores the state space.
- **Generated return is very high (7665)**: The agent accumulates massive reward over the episode, confirming the shaping reward strongly drives behavior.
- **Validation passed cleanly**: No repair attempts, no validation errors.

**2. What failed**

- **Private eval return is catastrophically low (4.63)**: Despite the massive generated return, the hidden evaluator gives almost no reward. This is the worst possible outcome: the agent learned something that looks great in the shaped reward but is completely misaligned with the true objective.
- **Massive generated-private gap (7660.75)**: The gap increased from +75.8 in gen0 to +7660.8 in gen1. This is a **10,000x increase** in misalignment. The reward function is now perfectly optimized for the wrong thing.
- **No terminal successes (terminal = 0.0)**: The agent never achieved a successful landing, despite 901-step episodes. This means the shaping reward is so dominant that the agent can collect huge reward simply by hovering near the pad without ever attempting to land.
- **Shaping reward is pathologically large**: At 15x exponential, the shaping reward (~8.5 per step when near pad) completely eclipses all other signals. The agent can earn ~7650 over 900 steps by just staying near the pad, with no incentive to actually land.
- **The fix overshot completely**: The previous lesson said "increase shaping 5-10x" but 15x was excessive. The ratio of shaping:penalties went from 0.3:1 (too weak) to 600:1 (too strong), creating a new failure mode.

**3. What to try next**

- **Reduce shaping multiplier dramatically**: Try 2x-5x instead of 15x. The shaping should be a gentle guide, not the dominant reward. A reasonable target is shaping ~50-100 total per episode, not 7000+.
- **Increase velocity penalty back to a meaningful level**: The -0.1 is too weak. Try -0.3 to -0.5 so the agent is penalized for excessive speed but not paralyzed. The goal is to make the agent move *deliberately* toward the pad, not just hover.
- **Add a survival/small negative per-step reward**: Consider -0.1 per step to create pressure to finish the episode. This prevents the agent from collecting infinite shaping reward by hovering.
- **Make terminal bonus truly dominant**: Increase terminal bonus to 500-1000 and ensure it's the largest single reward available. This creates a clear "finish the game" incentive that overcomes per-step shaping.
- **Add intermediate landing rewards**: Give small rewards for reducing altitude (obs[1] becoming more negative), slowing descent velocity (obs[3] near 0 when close to ground), or getting leg contact (obs[6], obs[7]). This creates a staircase of milestones toward landing.
- **Restructure component balance**: Target a reward composition where:
  - Shaping: ~20-50 per episode (gentle guidance)
  - Velocity penalty: ~-10 to -30 per episode (meaningful but not paralyzing)
  - Angle penalty: ~-5 to -10 per episode (encourages stable orientation)
  - Terminal: 500-1000 (dominant finish signal)
  - Per-step survival: -0.1 (creates urgency)
- **Consider the possibility that the hidden evaluator cares about landing success above all else**: The private return of 4.63 (vs generated 7665) strongly suggests the evaluator gives negligible reward for anything except successful landing. The reward function must make landing the only way to get high total reward.

**4. Lessons supported or contradicted**

- **Supported**: "Component scaling ratios matter more than individual component values." The ratio went from 0.3:1 (too low) to 600:1 (too high), confirming balance is critical.
- **Supported**: "Large generated-private gap indicates reward misspecification." The gap exploded to 7660, confirming the reward function is optimizing for the wrong objective.
- **Contradicted**: "Increase shaping reward magnitude to 10-50x." The 15x was far too aggressive and caused catastrophic misalignment. A more moderate 2-5x may be appropriate.
- **Contradicted**: "Remove fuel penalty to avoid discouraging engine use." The agent now uses engines freely (action mean 1.93) but without productive purpose. Engine use needs to be *directed* toward landing, not just encouraged.
- **New lesson emerging**: "Shaping rewards that dominate the reward signal can create policies that collect large per-step reward without ever completing the task, leading to massive generated-private gap."
- **New lesson emerging**: "When shaping rewards are very large, the terminal bonus must be even larger to incentivize task completion. Alternatively, add per-step negative reward to create urgency."
- **New lesson emerging**: "The hidden evaluator likely gives reward primarily or exclusively for successful landing. All components should be designed to guide toward landing, not to provide alternative sources of reward."