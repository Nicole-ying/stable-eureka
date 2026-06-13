# EG-RSA Final Framework

## Goal

EG-RSA follows Eureka-style task input while adding reward-search structure, repair, diagnostics, reflection, and three-level memory.

## Input Boundary

The LLM receives the same task cognition source as Eureka:

1. envs/<task>/task_description.txt
2. envs/<task>/step.py

The framework does not synthesize an extra Gym-space-derived observation/action range table.

Allowed:

- LLM reasoning over observation/action semantics from the task files
- environment understanding generated from task files
- reward schema and search plan generated from task files
- parent reward code
- training feedback
- STM, MTM, and LTM lessons

Forbidden in generated reward code:

- env_reward
- fitness_score
- compute_fitness_score
- official reward formula
- hidden evaluator implementation
- expert reward template

## LLM Agents

1. EnvUnderstandingAgent
   - Input: task_description plus step.py
   - Output: artifacts/env_understanding.md and artifacts/env_understanding.json

2. SchemaPlannerAgent
   - Input: task files plus environment understanding
   - Output: reward_schema.txt and clean_plan.txt

3. RewardCoderAgent
   - Input: task files, environment understanding, schema, search plan, feedback, memory context, parent reward code
   - Output: reward_code.py, rationale, raw LLM response

4. RepairAgent
   - Trigger: validator failure
   - Output: repaired reward code

5. ReflectionAgent
   - Trigger: generation end
   - Input: structured evidence, environment memory, retrieved lessons
   - Output: reflection_report.md

6. LessonExtractorAgent
   - Trigger: generation end
   - Output: environment lessons and cross-environment lessons

## Memory

### STM: Candidate Memory

Files:

- memory.jsonl
- memory.csv
- candidate artifacts under artifacts/generation_*

Used for:

- parent reward selection
- candidate-level lesson retrieval

### MTM: Environment Memory

Files:

- env_lessons.jsonl
- env_memory.md

Updated after every generation.

Used in the next generation as environment-level memory context.

### LTM: Cross-Environment Memory

File:

- runs/ltm_lessons.jsonl

Updated when a lesson is marked reusable beyond the current environment.

Retrieved in later runs as cross-task memory.

## Artifacts

Every LLM call is logged:

- llm/<stage>/system.txt
- llm/<stage>/user.txt
- llm/<stage>/response.txt
- llm/<stage>/budget.json

budget.json stores character counts and estimated token counts.

## Runtime Flow

task_description.txt plus step.py
  -> EnvUnderstandingAgent
  -> SchemaPlannerAgent
  -> MemoryRetriever
  -> RewardCoderAgent
  -> Validator
  -> RepairAgent if needed
  -> RL training and private evaluation
  -> EvidencePacker
  -> ReflectionAgent
  -> LessonExtractorAgent
  -> STM, MTM, and LTM update
  -> next generation


## Eureka Step Policy

EG-RSA uses the same task-code input policy as Eureka.

The LLM receives:

1. task_description.txt
2. step.py exactly as provided in the Eureka-style envs/<task>/ directory

The step.py may contain hook calls such as compute_reward or compute_fitness_score. These hook calls are part of the Eureka-provided environment code and are allowed in the input.

The hidden implementations of the official reward and fitness evaluator are not provided. Generated reward code must not reconstruct or imitate those hidden implementations.
