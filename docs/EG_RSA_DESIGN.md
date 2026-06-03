# EG-RSA Design and Implementation Manual

## 1. Method Positioning

**EG-RSA** stands for **Experience-Guided Reward Search Agent**.

EG-RSA reformulates automated reward design as a **sequential reward editing process** rather than one-shot reward generation or population-level reward sampling. Instead of asking an LLM to freely generate complete Python reward functions, EG-RSA maintains a componentized reward schema and iteratively edits it using reward-component attribution, reward-task misalignment diagnostics, and structured experience memory.

The key idea is:

> EG-RSA does not rely on unconstrained LLM reward-code generation. It constrains the LLM to produce structured edit decisions over a reward schema, while trusted program logic compiles, diagnoses, edits, and records the reward function.

## 2. Difference from Stable-Eureka

Stable-Eureka follows a population-style reward search loop:

```text
LLM generates multiple complete reward-code samples
-> each candidate reward is trained in parallel
-> oracle fitness_score selects the best reward
-> the best reward reflection is used in the next iteration
```

This design is powerful, but it has several limitations for our target setting:

1. It depends on an oracle `fitness_score` or ground-truth evaluator during reward search.
2. It trains multiple candidates per iteration, which can be expensive.
3. It lets the LLM write full Python reward code, which can be unstable.
4. Reward hacking is difficult to detect from scalar returns alone.
5. Historical experience is mainly stored as text reflection, not as reusable reward-editing knowledge.

EG-RSA changes the search loop to:

```text
current reward schema
-> compile schema into reward function
-> train one policy with fixed sufficient budget
-> collect trajectories, reward components, task metrics, and event flags
-> perform reward attribution and reward-task misalignment diagnostics
-> retrieve structured memory cards
-> LLM outputs JSON edit_plan only
-> trusted edit operators update the reward schema
-> store FailureMode-Attribution-EditOperator-Outcome memory
```

## 3. Core System Modules

### 3.1 Task Metric Specification

Task metrics are diagnostic signals used to judge whether the current reward aligns with the true task objective. They are not the reward optimized by the policy.

We should avoid claiming that the system fully automatically creates task metrics. A more rigorous formulation is:

> Task metrics are instantiated from general diagnostic templates using observation semantics and task descriptions. The LLM may assist variable mapping, but the system does not assume fully automatic metric discovery.

For LunarLander, a task metric specification may include:

```yaml
observation_mapping:
  x: obs[0]
  y: obs[1]
  vx: obs[2]
  vy: obs[3]
  angle: obs[4]
  angular_velocity: obs[5]
  left_leg_contact: obs[6]
  right_leg_contact: obs[7]

task_metrics:
  progress:
    type: distance_to_target
    target: [0.0, 0.0]

  stability:
    type: velocity_angle_stability
    variables: [vx, vy, angle, angular_velocity]

  success:
    type: stable_contact
    conditions:
      both_legs_contact: true
      abs_vx_lt: 0.1
      abs_vy_lt: 0.1
      abs_angle_lt: 0.1
      duration_steps: 30

  energy:
    type: action_cost
```

Paper wording:

> During reward search, the policy is trained only with the generated reward. The environment oracle reward is not used for reward selection or editing. We report the oracle reward only as a post-hoc evaluation metric.

### 3.2 Componentized Reward Schema

EG-RSA represents rewards as structured schemas instead of unconstrained code.

Example:

```json
{
  "version": 0,
  "components": [
    {
      "name": "r_distance",
      "type": "distance_penalty",
      "weight": 1.0,
      "inputs": ["x", "y"],
      "params": {"target": [0.0, 0.0]},
      "clip": [-2.0, 0.0],
      "enabled": true
    },
    {
      "name": "r_velocity",
      "type": "velocity_penalty",
      "weight": 0.5,
      "inputs": ["vx", "vy"],
      "enabled": true
    }
  ],
  "event_rules": []
}
```

Principles:

1. The LLM must not directly emit Python reward code.
2. The LLM may propose or edit reward schema JSON.
3. The program compiles the schema into `compute_reward`.
4. Every component must be separately logged for attribution.

### 3.3 Reward Compiler

The reward compiler converts `reward_schema.json` into executable reward code compatible with the environment.

Current implementation:

```text
eg_rsa/reward/safe_compiler.py
```

The safe compiler embeds the schema through `json.loads(...)`, avoiding invalid Python literals such as `true`, `false`, or `null`.

### 3.4 Policy Training

The policy is trained with the generated reward, not with the oracle environment reward.

Use the phrase:

```text
fixed sufficient training budget
```

instead of:

```text
complete training
```

Recommended budgets:

```text
CartPole-v1:       100k steps
LunarLander-v3:    500k to 1M steps
BipedalWalker-v3:  1M to 3M steps
```

### 3.5 Trajectory Recorder

Reward hacking cannot be diagnosed from scalar returns alone. EG-RSA needs step-level trajectory data:

```json
{
  "episode_id": 0,
  "steps": [
    {
      "obs": [...],
      "action": 2,
      "reward": 0.31,
      "components": {
        "r_distance": -0.2,
        "r_contact": 1.0
      },
      "task_metrics": {
        "progress": 0.4,
        "success": 0.0
      },
      "events": {
        "contact": true,
        "near_goal": false
      }
    }
  ],
  "summary": {
    "episode_reward": 132.4,
    "progress_score": 0.5,
    "success": 0.0,
    "episode_length": 800
  }
}
```

This enables detection of repeated events, high reward with low progress, shaping-goal mismatch, and other generic misalignment patterns.

### 3.6 Reward Attribution

Reward attribution estimates which component dominates the total reward.

Current implementation:

```text
eg_rsa/diagnostics/attribution.py
```

Tracked statistics:

```text
sum
abs_sum
mean
std
min
max
ratio
trigger_rate
dominant_component
dominant_component_ratio
```

### 3.7 Reward-Task Misalignment Diagnostics

EG-RSA should not predefine environment-specific reward-hacking rules. It detects generic reward-task misalignment patterns:

1. High reward but low task progress.
2. Single reward component dominance.
3. Repeated event triggering.
4. High shaping reward but low final success.

Current implementation:

```text
eg_rsa/diagnostics/hack_detectors.py
```

Paper wording:

> Rather than enumerating environment-specific reward hacking behaviors, EG-RSA detects reward-task misalignment through generic diagnostic patterns, including high reward with low task progress, single-component dominance, repeated event triggering, and shaping-goal mismatch.

### 3.8 Experience Memory

Memory stores reusable reward-debugging experience, not raw reward code.

Memory card format:

```json
{
  "memory_id": "memory_0007",
  "env_family": "landing_control",
  "failure_modes": [
    "repeated_event_exploitation",
    "shaping_goal_mismatch"
  ],
  "reward_attribution": {
    "dominant_component": "r_contact",
    "dominant_component_ratio": 0.76
  },
  "edit_plan": [
    {
      "operator": "convert_to_one_time_event",
      "target": "r_contact"
    }
  ],
  "outcome": {
    "hack_score_delta": -0.41,
    "success_rate_delta": 0.23
  },
  "lesson": "When an event reward dominates but success remains low, convert it to one-time or duration-conditioned reward."
}
```

Current implementation:

```text
eg_rsa/memory/memory_card.py
eg_rsa/memory/memory_store.py
```

### 3.9 Operator-Constrained Reward Editing

The LLM must output a JSON edit plan, not Python code.

Supported operators in the first implementation:

```text
increase_weight
decrease_weight
clip_component
disable_component
add_component
convert_to_one_time_event
add_duration_condition
reshape_sparse_to_dense
```

Example:

```json
{
  "diagnosis": "r_contact dominates the reward while success remains low.",
  "edit_plan": [
    {
      "operator": "decrease_weight",
      "target": "r_contact",
      "factor": 0.5
    },
    {
      "operator": "convert_to_one_time_event",
      "target": "r_contact"
    }
  ]
}
```

## 4. EG-RSA Algorithm

```text
Algorithm: EG-RSA

Input:
    Environment E
    Task description D
    Task metric specification M
    Initial reward schema R0
    Experience memory H
    Training budget B
    Iteration number K

for k = 0, 1, ..., K-1:

    1. Compile reward schema Rk into executable reward function.

    2. Train policy pi_k using generated reward Rk for B timesteps.

    3. Roll out pi_k and collect trajectories:
       tau_k = {(s_t, a_t, r_t, components_t, metrics_t, events_t)}

    4. Compute reward-component attribution:
       A_k = Attribution(tau_k)

    5. Detect reward-task misalignment:
       F_k = Diagnostics(tau_k, A_k, M)

    6. Retrieve relevant memory:
       C_k = Retrieve(H, F_k, A_k)

    7. Generate edit plan:
       P_k = LLM_EditAgent(Rk, A_k, F_k, C_k, allowed_operators)

    8. Validate edit plan:
       if P_k invalid:
           repair or fallback to rule-based edit

    9. Apply edit operators:
       R_{k+1} = ApplyOperators(Rk, P_k)

    10. Store memory card:
       H <- H union {(F_k, A_k, P_k, outcome_k)}

Return the best reward schema according to task metrics.
```

The best reward is selected according to task metrics and diagnostics, not oracle reward.

## 5. Paper Contributions

### Contribution 1: Sequential Reward Editing

We propose a sequential reward editing paradigm for automated reward design. Instead of generating multiple complete reward functions per iteration and selecting the best candidate, EG-RSA maintains a componentized reward schema and locally edits problematic components based on post-training diagnostics.

### Contribution 2: Structured Experience Memory

We introduce a structured reward memory that stores reward-debugging experience as failure mode, reward attribution, edit operator, and outcome tuples. This memory enables the agent to reuse previously successful editing strategies rather than merely storing reward code or textual reflections.

### Contribution 3: Reward-Component Attribution

We design a reward-component attribution mechanism that quantifies the contribution, trigger frequency, variance, and dominance of each reward component. The attribution result guides operator-constrained local edits, reducing the instability of unconstrained LLM reward generation.

### Contribution 4: Reward-Task Misalignment Diagnostics

We propose generic reward-task misalignment diagnostics to detect potential reward hacking without manually enumerating environment-specific exploit behaviors. The diagnostics capture cross-environment patterns such as high reward with low task progress, single-component dominance, repeated event triggering, and shaping-goal mismatch.

## 6. Dangerous Claims to Avoid

Avoid writing:

```text
fully automatic task metric discovery
zero human prior
complete reward-hacking prevention
no evaluation signal at all
LLM automatically designs perfect rewards
complete training for every reward
```

Use instead:

```text
not using environment oracle reward during search
task-level diagnostics from observation semantics and general templates
reward-hacking detection and mitigation
fixed sufficient training budget
operator-constrained reward editing
```

## 7. Experimental Plan

### Environments

```text
1. CartPole-v1
   Purpose: fast pipeline validation.

2. LunarLander-v3
   Purpose: main experiment for reward-task misalignment and reward-hacking repair.

3. BipedalWalker-v3
   Purpose: continuous-control generalization.
```

Meta-World is deferred because the debugging cost is high.

### Baselines

```text
B0. Oracle PPO
    Trained with official environment reward. Upper-bound reference only.

B1. LLM One-shot Reward
    Generate one reward once, no iteration.

B2. LLM Free Rewriting
    LLM freely rewrites reward code each iteration. No schema, no memory, no attribution, no operator constraints.

B3. Sequential Editing w/o Memory
    Attribution and edit operators are used, but no memory retrieval.

B4. Sequential Editing w/o Attribution
    Memory and operators are used, but no reward-component statistics are given.

B5. Sequential Editing w/o Hack Detector
    Memory and attribution are used, but no reward-task misalignment diagnostics are used.

B6. EG-RSA Full
    Full method.
```

### Metrics

```text
post-hoc official reward
task success rate
failure rate
hack score
high_reward_low_progress ratio
component dominance ratio
repeated event count
shaping-goal mismatch rate
iterations to target performance
invalid reward / invalid JSON rate
number of reward edits
wall-clock time / environment interaction steps
```

The official reward is only used for post-hoc reporting, not for reward search.

## 8. Implementation Roadmap

Current completed phase:

```text
Phase 0: Minimal mock EG-RSA loop
```

Existing files:

```text
eg_rsa/reward/schema.py
eg_rsa/reward/safe_compiler.py
eg_rsa/reward/operators.py
eg_rsa/diagnostics/attribution.py
eg_rsa/diagnostics/hack_detectors.py
eg_rsa/memory/memory_card.py
eg_rsa/memory/memory_store.py
eg_rsa/runner.py
train_eg_rsa.py
configs/eg_rsa_minimal.yml
```

Next phases:

```text
Phase 1: LunarLander adapter
  - obs -> obs_map
  - obs/action -> task_metrics
  - obs/action -> events
  - obs/action -> state_flags

Phase 2: trajectory recorder
  - rollout trained policy
  - record obs/action/reward/components/task_metrics/events per step
  - save trajectories.jsonl

Phase 3: SB3 training integration
  - reward_schema -> environment wrapper or generated env
  - train with fixed budget
  - save model.zip
  - generate real trajectories

Phase 4: LLM edit agent
  - prompt with current_schema, diagnostics, memory, allowed operators
  - parse JSON edit_plan
  - validate and execute operators

Phase 5: full EG-RSA runner
  - compile
  - train
  - record
  - diagnose
  - retrieve
  - edit
  - update memory
```

## 9. Coding Principles

1. Do not break the original `stable_eureka` package.
2. Keep EG-RSA under `eg_rsa/`.
3. Make the mock loop work before connecting real training.
4. Never let the LLM directly write executable reward code in the main EG-RSA path.
5. Do not use `fitness_score` or oracle reward for reward search.
6. Save all intermediate artifacts as JSON for debugging and paper figures.
7. Keep each module independently testable.
