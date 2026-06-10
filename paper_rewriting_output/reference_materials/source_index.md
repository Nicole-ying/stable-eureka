# Source Index

## Project: EG-RSA (Experience-Guided Reward Search Agent)

| Source ID | Type | Title/Name | Origin/URL/Path | Why Included | Local File/Note | Used For |
|---|---|---|---|---|---|---|
| DSG-01 | design_doc | EG-RSA Design and Implementation Manual | `docs/EG_RSA_DESIGN.md` | Authoritative method description, architecture, contributions, forbidden claims | Local | Method, claims, architecture |
| WRT-01 | writing_plan | EG-RSA Paper Writing Plan | `paper/README_writing_plan.md` | Paper structure, title proposals, innovation framing, writing logic per section | Local | Structure, tone, positioning |
| RW-01 | related_work_draft | Related Work Draft Prose | `paper/related_work_draft.md` | Full Related Work section draft with citations | Local | Related Work section |
| RW-02 | related_work_notes | Related Work Literature Matrix | `paper/related_work_notes.md` | Structured citation-to-claim mapping for all references | Local | Citation anchoring, positioning |
| BIB-01 | bibliography | EG-RSA Reference Bank | `paper/references_eg_rsa.bib` | 12 BibTeX entries covering RL, reward design, LLM agents, safety | Local | All citations |
| DSN-02 | design_doc | V1 Release Full Design | `docs/v1release/V1_RELEASE_FULL_DESIGN.md` | Full v1 feature set and release design | Local | Method completeness |
| DSN-03 | design_doc | EG-RSA Framework v0.1 | `docs/eg_rsa_framework_v0_1.md` | Earlier framework iteration | Local | Evolution of design |
| ISS-01 | issues | EG-RSA Open Issues | `docs/eg_rsa_open_issues.md` | Known limitations, open problems | Local | Discussion, limitations |
| EXP-01 | experiment | Main Role Attribution Experiment (10×1M) | `experiments/eg_rsa_lunar_lander_v1_role_attrib_10x1m/` | 10-iteration EG-RSA run with full trajectory, memory, audit, attribution | Local | Primary experimental evidence |
| EXP-02 | experiment | Relaxed Audit Variant | `experiments/eg_rsa_lunar_lander_v1_role_attrib_10x1m_audit_relaxed/` | 4-iteration run with relaxed audit to escape deadlock | Local | Audit deadlock case study |
| EXP-03 | experiment | Active Smoke Test v5 (3×100K) | `experiments/eg_rsa_lunar_lander_v1_active_smoke_3x100k_v5/` | Smoke test with active diagnostics | Local | Validation |
| EXP-04 | experiment | GPU-Scale Aligned Balanced (10×1M) | `experiments/eg_rsa_lunar_lander_aligned_balanced_10x1m_gpu/` | GPU-scale aligned balanced run | Local | Ablation comparison |
| EXP-05 | experiment | GPU-Scale Semantic (10×1M) | `experiments/eg_rsa_lunar_lander_semantic_10x1m_gpu/` | GPU-scale semantic run | Local | Ablation comparison |
| CFG-01 | config | Active Config: Role Attribution 10×1M | `configs/eg_rsa_deepseek_v1_role_attrib_10x1m.yml` | Active experiment configuration (MODIFIED, uncommitted) | Local | Reproducibility |
| IMP-01 | code | EG-RSA Runner (44KB) | `eg_rsa/runner.py` | Main EG-RSA loop implementation | Local | Method reference |
| IMP-02 | code | Hack Detectors (28KB) | `eg_rsa/diagnostics/hack_detectors.py` | Reward-task misalignment diagnostics | Local | Audit mechanism reference |
| IMP-03 | code | Behavior Risk Audit (28KB) | `eg_rsa/tools/behavior_risk_audit.py` | Risk audit implementation | Local | Risk audit mechanism |
| IMP-04 | code | Reward Schema | `eg_rsa/reward/schema.py` | Reward schema data structures | Local | Schema design |
| IMP-05 | code | Safe Compiler | `eg_rsa/reward/safe_compiler.py` | Schema→executable reward compilation | Local | Compiler design |
| IMP-06 | code | Attribution Module | `eg_rsa/diagnostics/attribution.py` | Reward-component attribution | Local | Attribution mechanism |
| IMP-07 | code | Semantic Outcome | `eg_rsa/diagnostics/semantic_outcome.py` | Semantic outcome evaluation | Local | Evaluation mechanism |
| FIG-01 | figure | Stable-Eureka Workflow Diagram | `img/stable-eureka_workflow.png` | Workflow visualization | Local | Figure reference |
| FIG-02 | figure | Lunar Lander Screenshot | `img/lunar_lander.png` | Environment visualization | Local | Figure reference |
| FIG-03 | figure | Evaluation Curves | `img/tensorboard_eval.png` | Training evaluation curves | Local | Figure reference |
| ENV-01 | environment | LunarLander-v3 | `envs/lunar_lander/` | Primary experiment environment | Local | Environment description |
