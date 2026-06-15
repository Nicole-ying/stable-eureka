# EG-RSA Single-Chain Expert-Brain Reward Search

This folder implements a single-chain EG-RSA reward-search path. It does not use Eureka-style multi-candidate sampling. Each iteration trains exactly one reward function, reads measured evidence, and generates one revised reward.

The current design is intentionally not a heavy multi-agent workflow. Expert reward-design knowledge is placed into static priors, task-model prompts, prescriptive memory, evidence summaries, and controller decisions.

## Bootstrap pipeline

The bootstrap path now uses two LLM calls before PPO training:

```text
task_description.txt + step.py
  -> TaskModelAgent
  -> task_model.json
  -> ExpertRewardDesignerAgent
  -> expert_blueprint + reward_schema + reward_code
  -> RewardGuard: static validation + runtime safety + smoke test + repair if needed
  -> append compute_reward into copied env_code/env.py
  -> PPO training
  -> evals.json + component_metrics.json + trajectory_summary.json + reward_trace.json
```

Run bootstrap only:

```bash
python -m eg_rsa.run_single_chain --config eg_rsa/single_chain/configs/lunar_lander.yaml
```

## Iterative pipeline

```text
trained parent artifacts
  -> EvidenceBuilder
  -> iteration_evidence.json
  -> MemoryManager retrieves compact memories
  -> ExpertMemoryContext builds prescriptive constraints
  -> ReflectionAgent reads evidence + expert context
  -> RewardRevisionAgent reads reflection + expert context
  -> RewardGuard before PPO
  -> PPO training for one new candidate
  -> SearchController accepts, rejects, or selects next parent
  -> MemoryManager records measured transition
```

Run bootstrap plus iterative search:

```bash
python -m eg_rsa.run_single_chain_iterative --config eg_rsa/single_chain/configs/lunar_lander.yaml
```

Continue from an existing run directory:

```bash
python -m eg_rsa.run_single_chain_iterative \
  --config eg_rsa/single_chain/configs/lunar_lander.yaml \
  --bootstrap-run-dir experiments/eg_rsa_single_chain/lunar_lander_single_chain_YYYYMMDD_HHMMSS
```

The runner creates:

```text
<run_dir>/
  task_model.json
  expert_reward_design_priors.json
  expert_reward_design_blueprint.json
  reward_guard_summary.json
  iteration_evidence.json
  best_iteration_evidence.json
  memory/reward_memory.jsonl
  search/
    expert_memory_context.json
    reflection_decision.json
    reflection_raw.txt
    expert_memory_context_after_reflection.json
    revised_reward_schema_and_code.json
    reward_revision_raw.txt
  iterations/iter_001/
    reward/reward_schema.json
    reward/reward_code.py
    reward/validation_report.json
    reward/runtime_safety_report.json
    reward/smoke_test_report.json
    reward_guard_summary.json
    env_code/env.py
    training/final_eval.json
    training/evals.json
    training/component_metrics.json
    training/trajectory_summary.json
    training/reward_trace.json
    iteration_evidence.json
    controller_decision.json
    memory_transition.json
```

## Design choices

- Bootstrap uses `TaskModelAgent` plus `ExpertRewardDesignerAgent`, not separate environment, target, architect, and reward agents.
- Expert knowledge is static framework prior plus task-specific reasoning inside the reward designer output.
- `ExpertMemoryContext` turns repeated measured failures into hard constraints for the next revision.
- `RewardGuard` prevents invalid generated reward code from reaching expensive PPO training.
- `SearchController` prevents a worse candidate from becoming the next edit parent; a selected parent is not retrained, only used as the next reward-edit base.
- Future reward revisions are not artificially limited to a fixed number of changed terms. The LLM may propose small, large, or restructuring edits, but must explain the causal hypothesis, expected behavior change, risks, and validation metrics.
