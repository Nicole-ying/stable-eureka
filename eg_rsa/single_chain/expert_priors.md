# Expert Reward Design Priors

These are transferable reward-design principles accumulated from RL research and
engineering practice. They apply to any environment — they are NOT environment-specific.

---

## 1. Reward Function Architecture

Every reward function decomposes into four categories. Their roles are different;
do NOT confuse them.

### Category 1: Terminal Objective Signal
- **What**: Fires only at episode end, marks task success or failure.
- **Properties**: Sparse, unbiased, cannot be farmed. Hard to discover via random exploration.
- **Rule**: Use the environment's own termination flag (e.g. `lander.awake == False`,
  `game_over == True`). Do NOT invent your own success condition by combining
  multiple state variables — that almost always gets the boundary conditions wrong.

### Category 2: Progress Signal
- **What**: Fires per-step, marks "you are getting closer to the goal".
- **Properties**: Dense, guides exploration. Has bias, can be farmed.
- **Rule**: Progress signals must be *target-aligned* (moving toward the objective,
  not just producing some side effect). If dominated by progress signal, the agent
  optimizes the proxy instead of the objective.
- **Design trade-off**: A strong progress signal speeds up learning but risks reward
  hacking. A weak progress signal is safe but leaves the agent blind.

### Category 3: Regularization Signal
- **What**: Small penalty on undesirable behavior, per-step.
- **Properties**: Fine-tuning only. If scale is too large, it suppresses necessary behavior.
- **Rule**: Do NOT use regularization to replace progress or objective signals.
  If the agent never performs an action, a penalty on that action does nothing.
  If the agent performs the action too much, a mild penalty can curb it.

### Category 4: Diagnostic Signal
- **What**: Does NOT affect total reward. Exposed in individual_reward for post-hoc analysis.
- **Properties**: Unbiased, no behavioral impact. Pure observation.
- **Rule**: Always include diagnostics for key state/action indicators. They are
  invisible to the agent but visible to future revision analysis.

---

## 2. Design Decision Tree

Ask these questions in order:

### Q1: Is the true success signal sparse (hard to discover randomly)?
- **Yes** → You MUST add progress signals to create a path. But progress must point
  toward the true objective, not some correlated-but-wrong proxy.
- **No** → A pure terminal signal may suffice.

### Q2: Which mathematical form should the progress signal take?

| Form | Expression | When to use |
|------|-----------|-------------|
| Delta reward (positive only) | `scale * max(0, prev - curr)` | Task is "reduce X" (speed, distance). Gentle, only rewards improvement. |
| Delta reward (signed) | `scale * (prev - curr)` | Task is "reduce X", agent needs both reward and penalty signal. |
| Proximity reward | `scale * exp(-|curr|)` or `scale / (1 + |curr|)` | Task is "stay close to target". Smooth gradient near target. |
| Threshold gate | `scale if condition else 0` | Discrete sub-goal milestone. Must be one-shot to prevent farming. |

### Q3: Is a one-shot mechanism needed?
- If a positive reward could fire multiple times per episode for the wrong reason →
  add one-shot gating: `if condition and not self._flag: reward = scale; self._flag = True`.
- Classic cases: leg contact, reaching a zone, achieving a threshold.

### Q4: How many conditions should gate a single reward term?
- **≤ 2 conditions**: acceptable.
- **3+ conditions**: the combined trigger probability is too low. Split into multiple
  simpler signals or relax the thresholds.

---

## 3. Component Scale Reasoning

When choosing a scale for each component, estimate the *per-episode expected total*:

```
E[total_per_episode] ≈ scale × expected_steps × expected_value_per_step
```

- **Objective signal**: should be the single largest contributor when it fires.
  Typical per-episode target: ~100 (for normalized reward ranges).
- **Progress signal**: should be smaller than objective, larger than regularization.
  Typical: 1/10 to 1/2 of objective scale per episode.
- **Regularization**: should be visible but not dominant. Typical: 1/100 to 1/10
  of objective scale per episode.

If a regularization term dominates the episode return, the agent is NOT optimizing
the task — it is minimizing a penalty.

---

## 4. Common Failure Syndromes

These patterns appear across environments. When you see them in evidence, the
diagnosis and treatment are well-established:

| Syndrome | Signature | Diagnosis | Treatment |
|----------|-----------|-----------|-----------|
| **Do-nothing / passive policy** | One action dominates (90%+), success=0 | All terms are negative; any action incurs cost | Convert absolute state penalties to delta progress rewards. Add positive signal for necessary actions. |
| **Reward farming** | One positive component is abnormally large, but success=0 | A repeatable positive term fires every step without leading to the objective | Make it one-shot, event-gated, or terminal-aligned. |
| **Hovering / timeout** | Long episodes, no crash, no success | Agent found a stable state that avoids penalties but doesn't complete the task | The penalty structure is rewarding stasis. Add time-pressure or progress-requirement signals. |
| **Early termination** | Short episodes (len < 50), success=0 | Total penalties exceed crash penalty; agent prefers early death | Reduce dense penalty magnitudes, or scale the crash penalty larger. |
| **Sparse blindness** | Success component is always 0.0 | Success condition is too strict to ever trigger randomly | Relax success thresholds, or add intermediate milestone rewards. |
| **False success** | success_rate is high but fitness/evaluation is poor | Agent is triggering a proxy-success condition, not the true one | Replace proxy condition with the environment's true termination flag. |

---

## 5. Editing / Tuning Principles

### How many terms to change at once
- **1-2 terms per iteration**. Changing more makes it impossible to attribute
  the effect. Multi-term edits that regress are untrustworthy.

### Before changing anything, check the evidence
- Which component dominates the episode return? → Address that one first.
- Which component is always zero? → It isn't participating. Adjust its condition
  or remove it.
- Which actions are unused? → If an unused action is necessary for success,
  add positive incentive for it. If it's unnecessary, ignore it.

### Scale adjustments
- **Increase**: if the behavior this component targets never appears.
- **Decrease**: if this component contributes > 50% of episode return.
- **Keep**: if the component fires but does not dominate.

### Structural changes (changing mathematical form)
- Absolute penalty → delta progress: when the component dominates and suppresses
  exploration.
- Repeatable → one-shot: when a positive component appears abnormally large.
- Gated → ungated: when a component never fires because the conditions are too strict.

---

## 6. What NOT to Do

- **Do not use the agent's generated reward as a framework objective.**
  The framework optimizes for *task behavior*, not for a high generated_reward number.
  A high generated_reward with zero success is a red flag, not a success.

- **Do not treat peak checkpoint fitness as verified reward quality.**
  Transient peaks happen. Final checkpoint behavior is what matters.

- **Do not preserve a reward design merely because one checkpoint had high fitness.**
  That fitness may be noise or a transient policy that collapsed later.

- **Do not add more than one new component per iteration during early search.**
  Each new component needs evidence to evaluate. Adding several at once creates
  confusion.

- **Do not write success conditions by hand.**
  Use the terminal classification flags provided by the environment wrapper
  (success_like_terminal, crash_terminal, etc.) rather than guessing from
  state variables.
