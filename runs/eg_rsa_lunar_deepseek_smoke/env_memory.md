# Environment Memory

env_alias: Env-90b964d9
latest_generation: 0

## Latest reflection
**1. What worked.**  
- The code structure is clear and modular, breaking reward into interpretable components (distance, velocity, angle, fuel, ground contact, terminal).  
- The agent survives for 56 steps on average (episode_length_mean=56), meaning it can hover or move without immediate crash.  
- The generated return (−87.21) is far higher than the private eval return (−480.57), suggesting the reward function is *more optimistic* than the true hidden metric. This indicates the shaping terms are providing positive feedback that the private evaluator does not share.  
- The terminal component correctly distinguishes success (20.0) from failure (−10.0), and the ground contact bonus (1.0) is a reasonable sparse signal.

**2. What failed.**  
- **Massive private eval gap**: `generated_minus_private = +393.36` means the reward function severely overestimates performance relative to the hidden evaluator. The private return is −480.57, which is extremely low.  
- **Action mean = 3.0, action std = 0.0**: The agent always selects action 3 (likely "do nothing" or "hover" depending on action space). It never fires main engine (action 2) or uses other actions. This suggests the shaping terms discourage any movement and the agent learns a trivial policy (stay still, collect small negative rewards, avoid terminal penalties).  
- **All component returns are negative except zero-value ones**: distance_shaping −24.9, velocity_penalty −20.2, angle_penalty −32.0, terminal −10.0. Fuel efficiency and ground contact bonus are 0 (agent never fires main engine, never both legs touch). The agent is penalized for being far from origin, moving, and being angled, but receives no positive reinforcement.  
- **Ground contact bonus never triggers**: Both legs never contact simultaneously (left_leg, right_leg likely 0 or low). The reward for landing is sparse and conditional on a tight success check (distance < 0.15, speed < 0.1, angle < 0.1, both legs). The agent never achieves this.  
- **Fuel penalty only applies to action 2**: Since the agent never takes action 2, fuel penalty is always 0, but the agent also never uses the engine to correct trajectory.

**3. What to try next.**  
- **Fix the action starvation**: The reward function is too punitive for movement. Reduce velocity penalty magnitude (e.g., from 0.3 to 0.05) and angle penalty magnitude (e.g., from 0.4 to 0.1). The agent must be allowed to move and correct without immediate heavy penalty.  
- **Add positive shaping for progress**: Instead of only negative penalties, reward *reduction* in distance or angle compared to previous step. Use `next_obs` to compute delta distance and delta angle, giving small positive reward when the agent moves closer to the pad or upright.  
- **Redesign terminal condition**: The success criteria (distance < 0.15, speed < 0.1, angle < 0.1, both legs) is too strict for early learning. Relax the success threshold (e.g., distance < 0.5, speed < 0.5, angle < 0.3) or provide intermediate rewards for partial success (e.g., both legs contact = +5 regardless of position).  
- **Make ground contact bonus easier to achieve**: Lower leg threshold from 0.5 to 0.1 or use continuous leg contact signal (e.g., `left_leg + right_leg`). Reward any leg contact, not just both.  
- **Encourage engine use**: Remove or reduce fuel penalty, or replace with a small positive reward for firing main engine when it helps (e.g., when moving toward pad or slowing descent).  
- **Scale all components down**: The sum of negative components (−87) plus terminal (−10) gives −97 before clipping. The private return is −480, meaning the hidden evaluator penalizes much more harshly. Try making all shaping terms 10× smaller (e.g., distance_shaping = −0.05 * distance) to match the private scale better.

**4. Which lessons seem supported or contradicted.**  
- **Supported**:  
  - *Sparse terminal-only rewards fail to guide exploration.* The agent never reaches the success state and learns a do-nothing policy.  
  - *Overly aggressive negative shaping can paralyze the agent.* High velocity/angle penalties caused action std=0.  
  - *Large gap between generated and private return indicates misaligned reward components.* The shaping terms are not valued by the hidden evaluator.

- **Contradicted**:  
  - *"Add ground contact bonus to encourage landing"* – In this case, the bonus threshold (both legs > 0.5) was never reached, so it provided zero guidance. The lesson should specify *easy-to-achieve* bonuses.  
  - *"Fuel penalty prevents wasteful engine use"* – Here it prevented all engine use, which is worse. The lesson should include *careful scaling* and *conditional reward* (e.g., penalize only if engine use doesn't improve state).

**Summary**: The agent is stuck in a local minimum of doing nothing. The reward function must be restructured to provide *positive* feedback for progress, reduce negative penalties, and make success conditions reachable. The private evaluator likely rewards final state quality (landing success) much more than per-step shaping, so the terminal reward should be the dominant signal, with shaping serving only as gentle guidance.

## Recent environment lessons
- failure_mode: Restructure reward to include positive shaping for progress (e.g., reduction in distance or angle), reduce penalty magnitudes, and relax success thresholds to be achievable during early learning.
- reward_pattern: Ensure shaping terms are small relative to the terminal reward and directly correlate with the hidden metric; consider using delta-based positive rewards instead of static penalties.
- failure_mode: Lower leg contact thresholds (e.g., > 0.1) or use a continuous signal (e.g., sum of leg values) to make the bonus easier to achieve; reward any leg contact initially.
- failure_mode: Avoid penalties that implicitly discourage exploration; instead, provide small positive rewards for engine use when it improves the state (e.g., reduces distance to pad or slows descent).
- failure_mode: Use relaxed success thresholds initially (e.g., distance<0.5, speed<0.5, angle<0.3) and gradually tighten as training progresses; or provide intermediate rewards for partial success.
- general: Ensure at least one component provides positive reward for desirable behavior (e.g., progress toward goal, leg contact, successful steps); balance positive and negative signals to drive exploration.