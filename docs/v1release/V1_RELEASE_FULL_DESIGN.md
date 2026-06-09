# v1release Full Design

## Project Name

Memory-driven Agentic Reward Search with Tool-augmented Semantic Diagnostics

中文名称：

基于记忆与工具调用的 Agent 化奖励函数自进化搜索框架

---

## 0. Why v1release is needed

The current EG-RSA system has shown useful results on LunarLander, but the experiment history also exposes its limitations.

Current v0 is better described as:

LLM-assisted constrained reward schema optimizer.

It can adjust a human-designed reward schema, but it is not yet a full agent system.

Main v0 limitations:

1. The LLM mostly outputs edit_plan instead of choosing actions and tools.
2. Memory mostly records logs instead of distilling reusable experience.
3. Rollback restores a previous schema but does not fully explain what went wrong.
4. Tool use is fixed by Runner, not actively selected by the LLM.
5. Reward search is strongly limited by a human-designed initial schema.
6. Structural search has limited safe fallback paths.
7. Environment transfer still depends on manually written task_description, diagnostic_spec and initial_schema.

v1release is designed to transform EG-RSA from a restricted edit pipeline into an agentic reward-search system.

---

## 1. v1 Core Innovations

v1 introduces five core innovations:

1. Agent does not only generate reward edits. It chooses the next search action.
2. Memory does not only store logs. It distills good edit, weak edit, bad edit and policy lessons.
3. Tool use supports trajectory inspection, schema diff, scale audit and candidate testing.
4. Reward search space opens by phase: constrained tuning, refinement, and free schema surgery.
5. Environment Semantic Translator reduces dependence on manually designed schemas.

---

# Part I. AgentActionController

## 1.1 Purpose

AgentActionController is the decision center of v1.

In v0, Runner decides the workflow and the LLM only fills edit_plan.

In v1, the LLM should choose one of several actions:

- apply_local_edit
- apply_structural_edit
- free_rewrite_schema
- continue_training
- evaluate_more
- rollback_replan
- run_tool
- early_stop

The controller converts the current state into an agent decision.

## 1.2 Inputs

AgentActionController receives:

- task_description
- current_reward_schema
- diagnostic_report
- semantic_outcome
- attribution_report
- retrieved_episodic_memory
- retrieved_lesson_memory
- retrieved_policy_memory
- latest_outcome_decision
- available_tools
- current_search_phase

## 1.3 Output

The controller outputs:

- action
- confidence
- reason_summary
- required_tools
- edit_intent
- memory_items_to_write
- safety_requirements
- fallback_action

Example:

{
  "action": "run_tool",
  "tool_name": "scale_audit",
  "reason_summary": "A new dense penalty is proposed after success. Check whether it will dominate terminal reward.",
  "fallback_action": "continue_training"
}

## 1.4 Execution rule

Runner must not blindly map every LLM output into edit_plan.

Runner should execute:

1. If action is run_tool, call the specified tool and return results to the agent.
2. If action is apply_local_edit, call EditAgent with local operators.
3. If action is apply_structural_edit, call StructuralSearchAgent plus ScaleAudit.
4. If action is free_rewrite_schema, call FreeRewriteAgent plus schema validator.
5. If action is continue_training, load current checkpoint and continue training.
6. If action is rollback_replan, restore best schema and immediately replan.
7. If action is evaluate_more, run multi-seed evaluation.
8. If action is early_stop, stop only after justification is written.

## 1.5 Acceptance criteria

AgentActionController is successful when:

- LLM can choose non-edit actions.
- no_edit no longer means workflow dead end.
- continue_training loads checkpoint.
- rollback triggers replan, not empty retraining.
- tool calls appear in logs before risky edits.

---

# Part II. Three-Layer Memory

## 2.1 Purpose

Memory is the center of v1. It must become an experience pool, not a log file.

v1 has three layers:

1. EpisodicMemory
2. LessonMemory
3. PolicyMemory

---

## 2.2 EpisodicMemory

EpisodicMemory stores raw facts.

Each card records:

- run_id
- iteration
- schema_hash
- edit_plan
- before_metrics
- after_metrics
- semantic_outcome
- attribution
- outcome_decision
- checkpoint_path
- posthoc_eval_for_report_only
- timestamp

Use:

- reconstruct exact history
- debug a transition
- support later lesson extraction

---

## 2.3 LessonMemory

LessonMemory distills one transition into a reusable lesson.

Each card records:

- lesson_type
- related_components
- related_edit_operators
- situation_signature
- evidence_summary
- mechanism_summary
- future_guidance
- confidence
- applicability

Lesson types:

- effective_edit
- weak_positive_edit
- neutral_edit
- harmful_scale_change
- structural_scale_mismatch
- terminal_condition_mismatch
- dense_goal_mismatch
- training_variance_uncertain

Example:

A structural component caused a new reward term to dominate after a previously successful schema. Future structural edits should pass scale audit and start from small magnitude.

---

## 2.4 PolicyMemory

PolicyMemory stores high-level strategy.

Each card records:

- situation_signature
- preferred_actions
- actions_requiring_audit
- actions_to_avoid
- phase_hint
- evidence_count
- confidence
- environment_family
- last_updated

Example:

Situation:
success_rate high, episode_length high, hack_score zero.

Preferred actions:
continue_training first, then small energy penalty adjustment.

Actions requiring audit:
new dense stagnation penalty.

## 2.5 Retrieval

Before each AgentActionController call, retrieve memory by:

- failure mode
- semantic situation
- component names
- edit operators
- search phase
- environment family
- schema similarity

The prompt must show memory in separate sections:

- raw episodes
- distilled lessons
- policy guidance

## 2.6 Acceptance criteria

Memory is successful when:

- every measured transition writes EpisodicMemory
- every outcome writes LessonMemory
- repeated patterns update PolicyMemory
- prompts use memory to change future actions
- repeated harmful edits are not proposed without explicit justification

---

# Part III. Tool Registry

## 3.1 Purpose

Tool Registry makes LLM a tool-using agent.

In v0, Runner calls fixed diagnostics.

In v1, the agent can request tools before deciding edits.

---

## 3.2 Tool categories

### Read-only evidence tools

- TrajectoryInspector
- SchemaDiffTool
- ComponentAttributionTool
- MemoryRetriever

### Safety tools

- ScaleAuditTool
- RewardDominanceChecker
- TriggerConditionChecker

### Proposal tools

- LocalEditProposalTool
- StructuralProposalTool
- FreeRewriteProposalTool

### Experiment tools

- CandidateSweeper
- MultiSeedEvaluator
- RolloutGifRecorder

---

## 3.3 TrajectoryInspector

Inputs:

- iteration_id
- episode_id or top_k failure episodes
- metrics of interest

Outputs:

- event timeline
- component reward timeline
- contact timeline
- terminal event trigger time
- failure or success summary

Use:

The agent uses this before deciding whether a problem is reward misalignment, insufficient training or policy instability.

---

## 3.4 SchemaDiffTool

Inputs:

- schema_before
- schema_after

Outputs:

- added components
- removed components
- changed weights
- changed clips
- changed event conditions
- changed one_time flags
- estimated risk

Use:

The agent uses this to understand what changed between iterations and which edit may explain the outcome.

---

## 3.5 ScaleAuditTool

Purpose:

Prevent new components from overwhelming existing terminal incentives.

Inputs:

- proposed edit_plan
- current schema
- trajectory statistics
- terminal reward scale
- per-step horizon estimate

Outputs:

- max_possible_episode_contribution
- expected_episode_contribution
- terminal_ratio
- dominance_risk
- recommended_safe_weight
- audit_pass

Example rule:

If a new dense penalty may accumulate more than a safe fraction of terminal reward, the agent must reduce its weight or run candidate sweep.

---

## 3.6 CandidateSweeper

Purpose:

Allow the agent to test multiple candidate edits before committing.

Inputs:

- candidate edit list
- small training budget
- seeds
- evaluation config

Outputs:

- candidate ranking
- semantic score
- stability
- risk report

Use:

When the agent is uncertain between weight values, it should run candidate sweep rather than commit one large edit.

---

## 3.7 RolloutGifRecorder

Purpose:

Generate visual evidence after training.

Inputs:

- model_path
- env_id
- seed
- output_path

Outputs:

- policy.gif
- return
- episode_length
- final state summary

Use:

GIF is reporting-only and must not enter reward search decisions unless explicitly marked as visual diagnostic.

---

# Part IV. OutcomeLessonBuilder

## 4.1 Purpose

OutcomeLessonBuilder converts every transition into a lesson.

It is called after each measured outcome.

## 4.2 Inputs

- schema_before
- schema_after
- edit_plan
- before_metrics
- after_metrics
- attribution_delta
- semantic_delta
- outcome_decision

## 4.3 Outputs

- lesson_type
- evidence_summary
- related_components
- related_edits
- mechanism_summary
- future_guidance
- confidence

## 4.4 Important behavior

If an edit improves success, write what worked.

If an edit degrades success, write why it likely degraded.

If an edit is uncertain, write what additional evidence is needed.

If continue_training improves score, write that reward may be aligned and policy optimization was the bottleneck.

## 4.5 Acceptance criteria

- every transition has a lesson
- every rollback has an explanation
- future prompts retrieve lessons
- repeated mistakes become less frequent

---

# Part V. Phase-Based Reward Search

## 5.1 Purpose

v1 must not use one search mode for all stages.

Reward search should change phase based on semantic outcome.

---

## 5.2 Phase 1: Constrained Search

Condition:

- success rate low
- terminal reward rarely triggers
- hack risk present
- reward structure still unstable

Allowed actions:

- local edit
- coupled rebalancing
- add duration condition
- weight adjustment
- limited structural proposal

Goal:

Find a reward that can produce successful behavior.

---

## 5.3 Phase 2: Refinement Search

Condition:

- success rate moderate or high
- hack score low
- terminal evidence exists
- main issue is speed, smoothness or stability

Allowed actions:

- continue_training
- small weight tuning
- multi-seed evaluation
- scale-audited penalty changes
- candidate sweep

Goal:

Improve stability and efficiency without destroying the reward balance.

---

## 5.4 Phase 3: Schema Surgery

Condition:

- local operators cannot express needed change
- repeated evidence shows structural limitation
- ScaleAudit and memory checks pass

Allowed actions:

- free_rewrite_schema
- add new structural component
- modify reward schema more freely
- generate new metric if supported by diagnostic spec

Requirements:

- schema validator
- scale audit
- small-budget smoke test
- memory risk check
- rollback safety

Goal:

Expand reward search space beyond the initial human schema.

---

## 5.5 Phase transition rules

Phase is determined by:

- success_episode_rate
- terminal_reward_paid_episode_rate
- hack_score
- reward_repetition_risk
- episode_length
- score stability
- memory guidance

Example:

If success_rate >= 0.7 and hack_score = 0, do not enter aggressive structural search unless local refinement fails.

---

# Part VI. Environment Semantic Translator

## 6.1 Purpose

Reduce dependence on manually designed task_description, diagnostic_spec and initial_schema.

## 6.2 Inputs

- env_id
- env.py if available
- observation space
- action space
- task description
- optional masked original reward for analysis only
- few-shot examples from LunarLander

## 6.3 Outputs

- generated_task_description.txt
- generated_diagnostic_spec.yml
- generated_initial_schema.json
- validation_report.json
- smoke_test_report.json

## 6.4 Pipeline

1. Read environment metadata.
2. Infer observation semantics.
3. Infer action semantics.
4. Generate task metrics.
5. Generate events.
6. Generate initial reward schema.
7. Validate all metrics and events.
8. Run signal smoke test.
9. Run short budget RL test.
10. Save environment semantic package.

## 6.5 Validation

The translator output must pass:

- all referenced metrics exist
- all events are executable
- reward components compile
- signals are non-constant
- terminal event is not always true
- dense terms are bounded
- initial schema trains without runtime error

## 6.6 Acceptance criteria

The system can create a new environment semantic package with minimal manual work.

For BipedalWalker, it should produce:

- forward progress metric
- torso stability metric
- fall event
- energy penalty
- gait stability proxy if possible
- initial schema using these signals

---

# Part VII. v1 Call Flow

## 7.1 Main loop

For each iteration:

1. Train or continue policy.
2. Record trajectories.
3. Run semantic outcome analyzer.
4. Run attribution analyzer.
5. Retrieve memory.
6. AgentActionController selects next action.
7. If needed, call tools.
8. Generate edit or training action.
9. Validate and audit.
10. Execute next action.
11. Measure outcome next iteration.
12. Build lessons.
13. Update memory.
14. Save reports.

---

## 7.2 Rollback flow

When a candidate degrades:

1. OutcomeAcceptor recommends rollback.
2. Restore best schema.
3. OutcomeLessonBuilder explains the transition.
4. PolicyMemory updates future guidance.
5. AgentActionController immediately replans.
6. The next candidate must cite retrieved lesson.

Rollback is not the end of exploration. It is a learning signal.

---

## 7.3 Continue training flow

When agent chooses continue_training:

1. Keep schema unchanged.
2. Load previous model checkpoint.
3. Continue training additional timesteps.
4. Evaluate semantic and posthoc reports.
5. Write lesson: optimization bottleneck or plateau.
6. If plateau, choose refinement action.

---

## 7.4 Structural edit flow

When agent proposes structural change:

1. Run SchemaDiffTool.
2. Run ScaleAuditTool.
3. Check memory for similar proposals.
4. If risky, run CandidateSweeper.
5. Apply only if safe and justified.
6. If fails, write structural lesson.

---

# Part VIII. Implementation Order

## Stage 1: Stabilize current loop

- ensure checkpoint continuation works
- add GIF recording script
- ensure summary always writes
- ensure no runner truncation or long-file remote overwrite

## Stage 2: OutcomeLessonBuilder

- build transition lesson
- write lesson files
- connect lesson retrieval to prompts

## Stage 3: Tool Registry

- implement SchemaDiffTool
- implement ScaleAuditTool
- implement TrajectoryInspector
- implement GIF recorder

## Stage 4: AgentActionController

- define action schema
- route actions in runner
- support run_tool before edit
- support evaluate_more

## Stage 5: Phase-Based Search

- implement phase detector
- switch edit freedom by phase
- add refinement mode

## Stage 6: Environment Semantic Translator

- generate semantic package
- validate package
- run smoke test
- test on BipedalWalker

---

# Part IX. Release Criteria

v1release is successful when:

1. LLM chooses actions, not only edits.
2. Rollback produces reusable lessons.
3. Memory changes future actions.
4. Tool calls appear before risky structural edits.
5. Success-phase search becomes conservative and scale-audited.
6. Free rewrite is allowed only after validation.
7. New environment semantic package can be generated and tested.
8. The framework can explain why a reward edit helped or hurt.

---

# Part X. Positioning

v0 contribution:

LLM-guided reward schema optimization with semantic diagnostics.

v1 contribution:

Memory-driven agentic reward search with tool-augmented semantic diagnostics.

The paper should honestly state:

- v0 depends on human semantic schema.
- v1 reduces this dependence with Environment Semantic Translator.
- posthoc official reward remains report-only.
- memory and tools are internal decision support, not oracle reward leakage.

