# Reward Search Journal — Living Document

This is my thinking log. I update it after every iteration.
It captures: what I tried, what I observed, what I learned, what I'll try next.

---

## Initial State (2026-06-14)

### What I know about LunarLander-v3
- 8-dim observation: x, y, vx, vy, angle, ang_vel, left_leg, right_leg
- 4 discrete actions: 0=nothing, 1=left engine, 2=main engine, 3=right engine
- Goal: land between flags at (0,0) with both legs down, upright, low speed
- The official reward is HIDDEN. I only see hidden_return_mean as a black-box score.
- Random agent gets hidden ≈ -200 to -550

### PPO Config (RL Zoo)
- n_envs=16, n_steps=1024, batch_size=64, n_epochs=4
- gamma=0.999, gae_lambda=0.98, ent_coef=0.01
- total_timesteps=2,000,000

### Starting hypothesis (from previous batch experiments)
- Exponential distance shaping: `alpha * exp(-beta * dist)` works as a guide
- Mild step penalty (~-0.03) prevents infinite hovering
- Terminal bonus must dominate per-step shaping to make landing the clear goal
- Soft landing conditions help the agent discover landing early
- Best config so far: shaping=2.0*exp(-2.0*dist), vel_pen=-0.3*speed, ang_pen=-0.3*|angle|, step_pen=-0.03, terminal=800

### Metrics I'll Track
1. **hidden_return_mean** — the ultimate score (target > 200)
2. **landing_rate / crash_rate / timeout_rate** — what the agent actually does
3. **episode_length_mean** — are we hovering, crashing early, or landing efficiently?
4. **final_distance_mean** — how close to the pad at episode end
5. **component_returns** — which reward terms dominate
6. **action_distribution** — is the agent using engines or doing nothing?
7. **step_distance_mean** — average distance from pad across all steps
8. **generated_hidden_gap** — alignment between our reward and hidden evaluator

---

## Iter 0 — 2026-06-14 — Initial Probe

### Plan
Start with v5a formula (best from batch: hidden=+5.5 at 150k steps, 1 env).
Now with proper RL Zoo config (n_envs=16, 2M steps), this should perform much better.

### Reward design rationale
- **shaping=2.0*exp(-2.0*dist)**: Smooth gradient toward pad. At pad: +2.0/step. At dist=1: +0.27/step.
- **velocity_penalty=-0.3*speed**: Mild speed discouragement. At speed=1: -0.3/step.
- **angle_penalty=-0.3*|angle|**: Stay upright. At 10° tilt: -0.05/step.
- **step_penalty=-0.03**: Tiny urgency. Over 200 steps: only -6 total. Not enough to force crash, enough to prevent infinite hover.
- **terminal=800**: Landing bonus. At 200 steps, per-step net at pad ≈ 2.0-0.15-0.05-0.03 = 1.77. 200*1.77 = 354 < 800. Landing wins!
- **Soft conditions** (pos<0.3, vel<0.5, angle<0.3): Easy to discover landing.

### What I'm watching for
- Does the agent land at all? (landing_rate)
- Does it hover? (timeout_rate, ep_len > 500)
- Does it crash? (crash_rate, ep_len < 50)
- Is the hidden score positive? (target > 0 first, then > 200)

### Results
- **hidden=-118.1** → Worse than expected. Landings are low quality.
- **landing=35%, crash=65%, timeout=0%** → Agent terminates every episode (no hover!)
- **ep_len=66.8** → Short episodes. Agent either lands or crashes quickly.
- **CRITICAL: action 2 (main engine) = 0.0%!** Agent NEVER fires main engine.
  Only uses side engines (37%+36%) and do-nothing (27%).
  Without main engine, the agent can't control vertical descent → explains high crash rate.
- **terminal=400** → ~50% of episodes got the 800 bonus (reward fn's loose conditions).
  But train.py only counts 35% (stricter x<0.2 vs x<0.3).
- **final_distance=0.231** → Close but not centered. Many landings off to the side.

### Diagnosis
1. **Main engine avoidance is the root problem.** The velocity_penalty punishes speed,
   and firing main engine creates speed. Agent learned to avoid it entirely.
2. **Step penalty works** (0% timeout) — episodes end one way or another.
3. **Landing precision is poor** — agent gets close but not centered, because
   it can't control descent without main engine.
4. **The shaping gradient near pad is too weak** — exp(-2.0*dist) at dist=0.1 = 0.82,
   at dist=0 = 1.0. Difference is only 0.18/step. Agent has little incentive to center.

### Decision for Iter 1
1. **Fix main engine avoidance**: Instead of penalizing ALL speed, penalize only
   HORIZONTAL speed (vx). Vertical speed (vy) should be rewarded when near ground
   (controlled descent). This removes the disincentive for main engine.
2. **Sharper shaping**: Change exp(-2.0*dist) → exp(-3.0*dist) for stronger gradient.
3. **Add vertical control reward**: bonus for low |vy| when y < 0.5 (descending).
4. **Keep**: step_penalty=-0.03, terminal=800, soft conditions.




### Iter 1 Results
- **hidden=-116.1** → No improvement. Changes had ZERO effect.
- **action_2 = 0.0% STILL** → Deeper problem: agent found side-engine-only strategy.
- **side engine usage 97%** → Tilting wildly instead of using main engine.
- **horizontal_penalty=-12** → Average |vx| very high. Oscillating horizontally.
- Some landings ARE clean (ep1: x=0.054, vx=0, legs both) — 35% success rate.

### Deeper Diagnosis (Iter 1→2)
The agent converged on a local optimum: tilt + side engines to roughly maneuver.
Main engine is never explored because:
1. Gravity = free descent. Agent doesn't "need" main engine to reach ground.
2. Side engines alone work ~35% of the time for "landing" (by soft conditions).
3. Once PPO finds this strategy, entropy noise is too low to escape.
4. The reward doesn't differentiate main engine from side engines.

### Decision for Iter 2
Need to EXPLICITLY break the side-engine local optimum:
1. **Main engine bonus**: reward action==2 when falling (vy < 0) — directly incentivize controlled descent
2. **Bigger terminal=1200**: make careful landing worth more than cheap side-engine approach
3. **Softer step_penalty=-0.02**: give more time for careful descent
4. **Return to exp(-2*dist)**: sharper decay reduced overall guidance too much


### Iter 2 Results
- **hidden=-111.8** → Same ballpark. landing=40% (slight improvement).
- **action_2 = 0.0% AGAIN** → controlled_descent bonus (+1) too weak vs velocity_penalty.
- The velocity_penalty (-0.3*speed) punishes acceleration from main engine more than the +1 bonus rewards it.
- Agent has settled into a stable local optimum that 3 iterations couldn't break.

### RADICAL DECISION for Iter 3
The velocity penalty is the ROOT CAUSE of main engine avoidance. Time to try:
1. **ZERO velocity penalty** → agent free to use main engine without punishment
2. **Big main engine bonus (+2.0)** when falling near ground
3. **Landing quality bonus** → reward smooth touchdowns proportionally
4. **Pure positive rewards** → no penalties except mild step_penalty
