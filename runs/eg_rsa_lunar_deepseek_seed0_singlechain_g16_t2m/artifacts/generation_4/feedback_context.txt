## Reflection on Generation 3

### 1. What worked

- **The agent is still achieving terminal reward (250.0)**: The landing tolerances remain achievable, with terminal component providing 250.0 total reward. This confirms the landing capability established in gen2 is preserved.
- **Action std decreased slightly**: From 0.92 in gen2 to 0.87, suggesting the policy is becoming more deterministic, which is a positive sign for convergence.
- **The survival penalty was successfully implemented**: `fuel_efficiency` is now -0.1 per step, totaling -100.0 over the 1000-step episode. This was the key missing piece from gen2 and is now active.

### 2. What failed

- **Catastrophic drop in private_eval_return**: From 267.8 in gen2 down to 28.7 in gen3 — a **9.3x decrease**. This is a severe regression, undoing most of the progress from gen2.
- **Episode length increased dramatically**: From 289.5 steps in gen2 to 1000 steps in gen3 (the maximum). The agent is surviving the full episode without landing, indicating the reward structure now discourages landing rather than encouraging it.
- **Shaping reward exploded**: From 370.6 in gen2 to 1444.7 in gen3 — a **3.9x increase** despite reducing the shaping multiplier from 3.0 to 2.0. This is because the agent now survives for 1000 steps instead of 289, accumulating shaping for much longer.
- **Generated-private gap exploded**: From 534.5 in gen2 to 1484.8 in gen3 — a **2.8x increase**. The gap is now ~52x the private return (28.7), indicating severe misalignment.
- **The survival penalty backfired**: The -0.1 per step penalty (-100 total over 1000 steps) is insufficient to overcome the shaping reward (1444.7). Instead of creating urgency, the agent simply endures the penalty while collecting shaping.
- **Terminal reward dropped**: From 500.0 in gen2 to 250.0 in gen3. The agent is landing less frequently or the terminal condition is being met only some of the time.
- **The combined effect of changes was catastrophic**: Reducing shaping from 3x to 2x while adding -0.1 survival penalty created a reward structure where the agent prefers to hover at the center (collecting ~1.44 shaping per step, net ~1.34 after penalty) rather than attempting to land.

### 3. What to try next

- **Increase survival penalty magnitude dramatically**: The -0.1 per step is negligible compared to shaping. Try -0.5 or -1.0 per step to create real urgency. Target: survival penalty of -300 to -500 over a full episode.
- **Reduce shaping multiplier further**: Try 1.0x or even 0.5x. The exponential decay means shaping is still high near the center. With 1000-step episodes, even 1.0x shaping would accumulate ~700-800 per episode, still too high.
- **Increase terminal bonus**: Try 1000 instead of 500 to make landing more attractive. Combined with stronger survival penalty, this could tip the balance toward landing.
- **Consider a step limit in the environment**: If the agent can survive indefinitely without landing, the reward structure will always favor hovering. Ensure episodes terminate (with penalty) after some maximum steps.
- **Add a descending reward**: Add a component that rewards reducing altitude when near the pad, e.g., `0.2 * max(0, -obs[1]) * exp(-2 * sqrt(obs[0]^2 + obs[1]^2))`. This gives the agent a clear gradient toward the landing zone.
- **Monitor the ratio of terminal to per-step rewards**: The terminal bonus (250 average) is only 0.17x the shaping (1444.7) and 0.25x the survival penalty (-100). Target terminal bonus being 3-5x the largest per-step component total.

### 4. Which lessons seem supported or contradicted

- **Supported**: "Add a separate per-step survival penalty component (e.g., -0.1 * 1.0) to penalize long episodes." — The penalty was added as planned, confirming this lesson can be implemented correctly.
- **Supported**: "Monitor generated-private gap as the primary diagnostic." — The gap of 1484.8 (52x private return) correctly identified severe misalignment, consistent with this lesson.
- **Supported**: "Use episode length as a diagnostic: if length is very high (>500 steps), it may indicate the agent is avoiding termination." — Episode length of 1000 confirmed the agent is avoiding landing, validating this diagnostic.
- **Contradicted (partially)**: "When reducing an over-dominant component, a 3-5x reduction is a reasonable step." — Reducing shaping from 3x to 2x (1.5x reduction) was insufficient because the agent exploited the longer episode to accumulate more shaping. The lesson needs refinement: "Reduction must consider the agent's ability to extend episode length; if the agent can survive longer, even reduced per-step shaping can accumulate to dominate."
- **Contradicted**: "Target a generated-private gap of less than 0.5x the private return." — This target was not achieved; the gap increased to 52x. The lesson is correct in principle but the specific changes failed to move toward it.
- **New lesson emerging**: "A survival penalty of -0.1 per step is too weak relative to shaping of ~1.4 per step. The penalty must be large enough that the net per-step reward (shaping + penalty) is negative when the agent is not actively landing, forcing the agent to seek the terminal bonus."
- **New lesson emerging**: "When adding a per-step penalty, ensure it is strong enough to offset all positive per-step components combined. If shaping provides +1.44 per step, the survival penalty must be at least -1.5 per step to make hovering net negative."
- **New lesson emerging**: "Reducing one component (shaping from 3x to 2x) while adding a small penalty (-0.1) can create a worse outcome than the original if the agent exploits the longer episode duration. Always simulate the expected per-step and total-episode reward balance before implementing."
- **New lesson emerging**: "Terminal bonus must be significantly larger than the maximum per-step reward accumulated over a full episode. With shaping at 1444.7 and penalty at -100, terminal bonus of 500 is only 0.37x the net per-step total. Target terminal bonus of 3-5x net per-step total."