# Clean Reward Search Run Report
best_candidate: g1_c1
schema_version: clean_reward_schema_v1_b3cacbe4c2
env_alias: Env-90b964d9
status: ok
selection_score_private_eval: -96.8937218640101
private_eval_return: -96.8937218640101
generated_reward_return: -11.42313063982971
repair_attempts: 0
repair_success: False
judge_score: 0.0
judge_reason: deepseek_text_only_judge_skipped
parents: ['g0_c0', 'g0_c1']

## Reflection
## Analysis of Past Clean Candidates

**g0_c0 (score -135.55):**
- Generated reward returned -11.64, but private evaluation returned -135.55 — large negative gap indicates the reward signal was poorly aligned with the true objective, producing overly optimistic or misdirected shaping
- Validation passed with no errors, so structure was correct but behavioral outcome was poor
- Likely failure mode: reward function created positive local incentives that conflicted with long-term objective

**g0_c1 (score -2226.00):**
- Generated reward returned +517.77 (highly positive), yet private evaluation returned -2226.00 — extreme divergence shows the reward signal was catastrophically misaligned
- Validation passed, so schema compliance alone does not guarantee good behavior
- Failure mode: reward function exploited some unintended behavior or produced massive positive feedback for actions that actually harmed the objective

## What to Keep
- Schema structure is validated and functional
- Both candidates passed validation — no structural issues

## What to Change
- Reward components must be conservative — avoid large positive signals unless they demonstrably align with private evaluation
- Need better balance between immediate reward and long-term outcome
- Avoid single-component dominance that can amplify misalignment

## Next Schema-Preserving Mutation Hypotheses (max 5)

1. **Penalize over-optimistic shaping**: Add a penalty term that activates when generated reward deviates significantly from a running mean, forcing reward to stay grounded
2. **Inverse scaling on positive spikes**: Apply a saturating nonlinearity (e.g., tanh) to reward components to cap extreme positive values and prevent runaway exploitation
3. **Two-term loss with consistency regularization**: Combine a main reward term with a secondary term that penalizes large differences between consecutive reward predictions
4. **Negative bias initialization**: Start all reward component weights with slight negative bias, requiring positive evidence to overcome — prevents accidental positive loops
5. **Delayed reward mixing**: Blend immediate generated reward with a moving average of past rewards to smooth out spikes and reduce short-term exploitation

## Reward code
```python
def compute_reward(obs, action, next_obs, done, info):
    # Unpack observations: [x, y, vx, vy, angle, angular_vel, contact1, contact2]
    x, y, vx, vy, angle, ang_vel, c1, c2 = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nc1, nc2 = next_obs
    
    # Progress: encourage movement in positive x direction (likely goal direction)
    dx = nx - x
    progress = np.clip(dx * 2.0, -1.0, 1.0)
    
    # Stability: penalize high velocity and angular deviation from upright (angle=0)
    speed_penalty = -0.1 * (abs(nvx) + abs(nvy))
    angle_penalty = -0.2 * abs(nangle)
    stability = np.clip(speed_penalty + angle_penalty, -1.0, 0.0)
    
    # Effort: penalize changing action (proxy for unnecessary control)
    # Use action value directly as discrete actions have magnitude
    effort = -0.05 * float(action)  # small penalty for larger action indices
    
    # Terminal: handle done signal
    if done:
        # Check for success signal in info (common pattern)
        if info.get('success', False) or info.get('goal_reached', False):
            terminal = 1.0
        else:
            terminal = -0.5  # penalty for failure/truncation
    else:
        terminal = 0.0
    
    total_reward = progress + stability + effort + terminal
    components = {
        'progress': float(progress),
        'stability': float(stability),
        'effort': float(effort),
        'terminal': float(terminal)
    }
    return float(total_reward), components
```