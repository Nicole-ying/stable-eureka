# Source Map

## Project Identity
- **Name**: EG-RSA (Experience-Guided Reward Search Agent)
- **Tagline**: LLM-assisted reward design re-framed as a sequential, history-aware, risk-audited reward search process — not one-shot generation.

## Key Source Files

### Design & Writing (authoritative)

| File | Role |
|---|---|
| `docs/EG_RSA_DESIGN.md` | Complete method design, algorithm, contributions, dangerous claims to avoid |
| `paper/README_writing_plan.md` | Paper structure, title proposals, innovation claims, writing logic per section |
| `paper/related_work_notes.md` | Related work organization |
| `paper/related_work_draft.md` | Related work draft prose |
| `paper/references_eg_rsa.bib` | BibTeX reference bank |
| `docs/v1release/V1_RELEASE_FULL_DESIGN.md` | Full v1 release design |
| `docs/eg_rsa_framework_v0_1.md` | Earlier framework design |
| `docs/eg_rsa_open_issues.md` | Known issues |

### Core Implementation

| File | Role |
|---|---|
| `eg_rsa/runner.py` | Main EG-RSA loop (44KB) |
| `eg_rsa/reward/schema.py` | Reward schema representation |
| `eg_rsa/reward/safe_compiler.py` | Schema → executable reward compilation |
| `eg_rsa/reward/operators.py` | Edit operators (increase_weight, etc.) |
| `eg_rsa/reward/candidate_evaluator.py` | Candidate evaluation |
| `eg_rsa/reward/edit_plan_validator.py` | Edit plan validation |
| `eg_rsa/reward/edit_decision_gate.py` | Edit decision gating |
| `eg_rsa/reward/outcome_acceptor.py` | Outcome acceptance |
| `eg_rsa/diagnostics/attribution.py` | Reward-component attribution |
| `eg_rsa/diagnostics/hack_detectors.py` | Reward-task misalignment detection (28KB) |
| `eg_rsa/diagnostics/semantic_outcome.py` | Semantic outcome evaluation |
| `eg_rsa/diagnostics/trajectory_recorder.py` | Step-level trajectory recording |
| `eg_rsa/diagnostics/task_metrics.py` | Task metric computation |
| `eg_rsa/memory/memory_card.py` | Memory card data structure |
| `eg_rsa/memory/memory_store.py` | Memory store |
| `eg_rsa/memory/lesson_store.py` | Outcome lesson storage |
| `eg_rsa/llm/edit_agent.py` | LLM edit agent |
| `eg_rsa/llm/structural_search_agent.py` | Structural search agent |
| `eg_rsa/llm/reflection_agent.py` | Reflection agent |
| `eg_rsa/tools/behavior_risk_audit.py` | Behavior risk audit (28KB) |
| `eg_rsa/tools/scale_audit.py` | Scale audit tool |
| `eg_rsa/tools/outcome_lesson_builder.py` | Outcome lesson building |
| `stable_eureka/stable_eureka.py` | Original Stable-Eureka implementation |

### Experiment Data

Key experiment directories under `experiments/`:
- `eg_rsa_lunar_lander_v1_role_attrib_10x1m/` — Main role attribution experiment (10 runs × 1M steps)
- `eg_rsa_lunar_lander_v1_role_attrib_10x1m_audit_relaxed/` — Relaxed audit variant (NEW)
- `eg_rsa_lunar_lander_v1_active_smoke_3x100k_v5/` — Latest active smoke test
- `eg_rsa_lunar_lander_aligned_balanced_10x1m_gpu/` — GPU-scale aligned balanced run
- `eg_rsa_lunar_lander_semantic_10x1m_gpu/` — GPU-scale semantic run

Each experiment contains: `summary.json`, `run_history.json`, `best_reward_schema.json`, `structural_context.json`, `experiment_mode.json`

### Configs

Active config: `configs/eg_rsa_deepseek_v1_role_attrib_10x1m.yml` (MODIFIED, uncommitted)
Other configs cover: ablation studies, semantic variants, aligned variants, smoke tests.

### Environments

- LunarLander-v3 (primary): `envs/lunar_lander/`
- BipedalWalker-v3: `envs/bipedal_walker/`
- MountainCarContinuous: `envs/mountain_car_continuous/`

### Figures

- `img/stable-eureka_workflow.png` — Workflow diagram
- `img/lunar_lander.png`, `img/bipedal_walker.png` — Environment screenshots
- `img/tensorboard_eval.png` — Evaluation curves
