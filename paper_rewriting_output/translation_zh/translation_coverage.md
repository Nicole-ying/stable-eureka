# 翻译覆盖报告

## 概述

- **场景：** NeurIPS/ICML 会议
- **源语言：** 英语（en）
- **目标语言：** 中文（zh）
- **翻译日期：** 2026-06-10

## 文件覆盖

| 英文源文件 | 中文目标文件 | 状态 |
|---|---|---|
| `paper_spine_config.json` | — | skipped（机器可读配置，无需翻译） |
| `confirmed_motivation.md` | `confirmed_motivation.zh.md` | translated |
| `research_dossier.md` | `research_dossier.zh.md` | translated |
| `style_profile.md` | `style_profile.zh.md` | pending |
| `section_blueprints.md` | — | skipped（执行方案，表格密集） |
| `writing_rationale_matrix.md` | — | skipped（47 行大型矩阵） |
| `citation_support_bank.md` | — | skipped（62 行大型表格） |
| `structured_review.md` | `structured_review.zh.md` | translated |
| `latex_report.md` | `latex_report.zh.md` | translated |
| `final_artifact_manifest.md` | `final_artifact_manifest.zh.md` | translated |
| `integrity_audit.md` | `integrity_audit.zh.md` | pending |
| `final_paper/main.tex` | `full_paper_translation.zh.md` | **translated** |
| — | `manifest.md` | complete |
| — | `translation_coverage.md` | complete |

## 核心交付物

**`full_paper_translation.zh.md`** — 完整论文中文翻译，包括：
- 标题
- 摘要（约 250 字）
- 第 1 节：引言（含 3 项贡献）
- 第 2 节：相关工作（5 个主题段：奖励设计、LLM 奖励生成、记忆与反思、奖励欺骗与安全、定位）
- 第 3 节：方法（5 个子节：搜索循环、Schema 与归因、结果记忆、算子约束编辑、风险审计）
- 第 4 节：实验（设置、主实验、3 个案例研究、放宽审计）
- 第 5 节：讨论（局限性、意义、未来工作）
- 第 6 节：结论
- 附录（超参数、实验模式、扩展轨迹、计划消融）

## 翻译质量说明

- 所有引用键（`\citep{...}`）保持不变
- 标签（`\label{...}`）保持不变
- 数值和表格数据保持不变
- 专业术语使用标准中文翻译（如 "强化学习" 对应 reinforcement learning，"奖励塑形" 对应 reward shaping）
- 代码标识符和文件路径保持原始形式
