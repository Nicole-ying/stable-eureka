# EG-RSA Open Issues and Next-Step Notes

This note records issues found after the DeepSeek + memory mechanism tests. It is not a paper claim; it is a development checklist to prevent confusing mechanism validation with task-solving performance.

## 1. No-edit is a control decision, not a root-cause solution

A valid `no_edit` decision prevents over-editing, but it does not solve the underlying task if the current reward remains weak. If an iteration chooses `no_edit`, the next iteration may train the same schema again. This is useful for confirming stability, but repeated no-edit iterations can waste compute.

Open question:
- Should EG-RSA stop early after repeated no-edit decisions?
- Should EG-RSA switch from edit mode to exploration / longer-training mode?
- Should EG-RSA ask for a structural search proposal when task proxies remain poor but reward-hack evidence is weak?

## 2. Detector false positives must be explicitly modeled

The repeated-event detector can remain active even after an event reward is converted to one-time. In that case, event toggles may indicate poor task behavior, but not necessarily reward hacking, because repeated toggles no longer produce repeated reward.

Needed distinction:
- reward hack: repeated behavior increases reward;
- task failure: repeated behavior happens, but does not produce additional reward;
- detector false positive: detector fires but reward structure no longer supports exploitation.

The Diagnostic Analyst should classify these cases before the Reward Editor proposes changes.

## 3. Lesson quality must be judged, not merely stored

A lesson card existing in `lesson_cards.jsonl` does not mean it is useful. Each lesson needs quality assessment:

- outcome quality: did task proxies improve and hack risk decrease?
- applicability: does the current failure pattern match the lesson condition?
- novelty: is the lesson already absorbed by the current schema?
- risk: could applying the lesson again cause over-editing?
- evidence strength: how many trials support the lesson?

The Memory Reflector should explicitly classify retrieved lessons as:

- reusable now;
- already applied;
- not applicable;
- weak or failed;
- conflicting.

## 4. Mechanism validation is not performance validation

The current tests validated several mechanisms:

- DeepSeek edit agent can emit structured role outputs;
- no-edit decisions can be respected;
- raw memory can be measured with before/after/delta;
- lesson cards can be generated and retrieved;
- lesson cards can be supplied to the next edit prompt.

However, these tests do not prove that EG-RSA solves LunarLander. Official posthoc returns remain far below solved-level performance, so the reward search strategy is still incomplete.

## 5. Reward search remains too conservative

The system can fix obvious repeated event bonuses, but it still struggles when the reward needs new structural signals rather than local edits. Future work should add generic, non-environment-specific search actions such as:

- progress-delta components;
- stagnation penalties;
- event penalties/bonuses from configured event spaces;
- conditioned event rewards based on configured metric/event predicates.

These additions must be implemented with real execution semantics before being exposed to the LLM.

## 6. No official reward leakage

Official environment reward may be used only for posthoc evaluation and reporting. It must not be used in:

- LLM prompts;
- memory cards or lesson cards used for decision-making;
- edit selection;
- replay or candidate scoring.

Task proxies and diagnostic metrics are allowed, but they must be described as proxy feedback, not as oracle rewards.

## 7. Immediate engineering TODOs

1. Fix summary output so legal no-edit is displayed as `no_edit`, not `invalid_or_fallback`.
2. Strengthen Memory Reflector prompt to classify each retrieved lesson as reusable / already applied / not applicable / weak / conflicting.
3. Consider storing no-edit decisions as lessons when they prevent over-editing.
4. Verify `duration_steps` with a focused experiment or unit test because it now has real wrapper semantics.
5. Add early-stop or mode-switch logic for repeated no-edit decisions to avoid retraining the same schema without purpose.
