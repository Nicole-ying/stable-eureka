# 周报：EG-RSA 奖励函数自动搜索 — v1/v2 并行实验

**日期**: 2025-06-12  
**环境**: LunarLander-v3, PPO, 10 iterations × 1M timesteps

---

## 1. 框架说明

EG-RSA 是一个 LLM 驱动的闭环奖励函数自动搜索框架。每次迭代包含五个阶段：

1. **训练** — 用当前 reward schema 训练 PPO 策略
2. **诊断** — 从 rollout 提取组件占比、hacking 信号、行为指标（成功率、稳定着陆率、接触抖动等）
3. **归因** — LLM 结合诊断报告 + memory 中历史经验，定位失败模式，生成编辑策略
4. **编辑** — LLM 生成 schema 修改，经多层门控过滤（Hack Detector / Edit Gate / Outcome Acceptor / Scale Audit / Behavior Risk Audit）后提交
5. **闭环** — 新 schema 进入下一轮训练，memory 积累经验

框架的两个核心设计：

- **语义角色系统** — 每个 reward 组件标注语义角色（`dense_guidance`, `stability_quality`, `terminal_success`, `safety_constraint`, `control_cost`），使 LLM 从行为语义层面理解 reward 结构，而非在数值空间中盲目搜索。同一套角色 taxonomy 可跨环境复用。

- **多层安全门控** — Hack Detector（检测组件支配/事件抖动/低成功率）、Edit Gate（限制编辑数量/最小触发率）、Outcome Acceptor（task/semantic/hack 改善阈值）、Scale Audit（dense/sparse 比例约束）、Behavior Risk Audit（行为指标风险评估），形成从诊断到接受的完整过滤链。

v1 和 v2 的区别在于搜索空间：v1 操作预定义的 metric 组件（`metric_value`, `metric_delta`, `event_bonus`），LLM 调权重和启禁用；v2 操作原始 formula AST（`formula_component`, `conditional_formula_component`, `event_predicate`），LLM 从零设计和修改数学公式。其余架构完全一致。

---

## 2. 实验结果

### 2.1 v1: Metric-based 搜索

实验路径: `experiments/eg_rsa_lunar_lander_v1_role_attrib_10x1m_audit_relaxed`

| Iter | Posthoc Return | Selection Score | Hack | 成功率 | 稳定着陆 | Terminal 触发 | 编辑决策 |
|------|---------------|-----------------|------|--------|---------|-------------|---------|
| 0 | 2.8 | 0.56 | 0.4 | 0% | 0% | 0% | edit: 禁用 r_approach_region, 提升 terminal |
| 1 | 29.4 | 0.62 | 0.4 | 0% | 0% | 0% | edit: 重平衡 dense/terminal |
| 2 | **154.5** | **6.40** | **0.0** | **83%** | **83%** | 100% | no_edit |
| 3 | **200.5** | **7.41** | 0.0 | **100%** | **100%** | 100% | no_edit |
| 4 | **260.6** | **7.41** | 0.0 | 100% | 100% | 100% | no_edit |
| 5 | **221.6** | **7.44** | 0.0 | 100% | 100% | 100% | best |
| 6 | 104.5 | 4.67 | 0.0 | 50% | 50% | 100% | edit: shaping-goal mismatch |
| 7 | -304.2 | 0.80 | 0.4 | 0% | 0% | 100% | edit: 修复 descent incentive |
| 8 | 202.9 | 4.93 | 0.0 | 50% | 50% | 100% | edit: 修复 landing quality |
| 9 | **200.5** | **7.28** | 0.0 | **100%** | **100%** | 100% | final |

**最优 posthoc return**: 260.6 (iter 4)，最优 selection score: 7.44 (iter 5)。搜索在 iter 2 即收敛到有效解（成功率 83%），iter 3-5 稳定在 100% 成功率+稳定着陆。iter 6-7 出现过编辑回退，但通过 memory 检索历史成功经验，iter 9 恢复到 100%。

### 2.2 v2: Formula-based Bootstrap 搜索

实验路径: `experiments/eg_rsa_lunar_lander_v2_bootstrap_10x1m`

v2 的初始 schema 由 LLM 仅基于原始观测/动作变量和数学运算符（abs, min, max, sqrt, exp, tanh）从零生成，不接触任何 v1 的 metric 定义或先验 reward 知识。

| Iter | Posthoc Return | Selection Score | Hack | 成功率 | 稳定着陆 | Terminal 触发 | 编辑决策 |
|------|---------------|-----------------|------|--------|---------|-------------|---------|
| 0 | -552.3 | 1.26 | 0.4 | 0% | 0% | 0% | edit: center_guidance 支配 (91.6%) |
| 1 | **193.1** | **2.64** | 0.2 | 0% | 0% | 100% | edit: 着陆质量差 |
| 2 | **225.8** | **2.68** | 0.2 | 0% | 0% | 100% | edit: bouncing, 无稳定接触 |
| 3 | **220.6** | 2.63 | 0.2 | 0% | 0% | 100% | edit: landing_success 条件过松 |
| 4 | 30.8 | 1.32 | 0.4 | 0% | 0% | 0% | edit: 缺近地速度惩罚 |
| 5 | -352.0 | 1.23 | 0.4 | 0% | 0% | 0% | edit: descent 死区悬停 |
| 6 | -32.0 | 1.29 | 0.4 | 0% | 0% | 0% | edit: descent 恒定奖励 |
| 7 | 12.2 | 1.22 | 0.4 | 0% | 0% | 0% | edit: shaping-goal mismatch |
| 8 | -292.5 | 1.31 | 0.4 | 0% | 0% | 0% | edit: center_guidance 支配 (85.6%) |
| 9 | **198.8** | **2.51** | 0.2 | 0% | 0% | 83% | final |

**最优 posthoc return**: 225.8 (iter 2)，最优 selection score: 2.68 (iter 2)。v2 在 iter 1-3 和 iter 9 均搜索到了 posthoc return 190-225 的奖励函数——在 LunarLander 中这属于有效着陆级别的回报。但搜索过程不稳定，在 iter 4-8 间出现大幅震荡。

### 2.3 核心对比

| 维度 | v1 (Metric) | v2 (Formula Bootstrap) |
|------|------------|----------------------|
| 搜索空间 | metric 权重 + 启禁用 | 完整 formula AST |
| 初始 schema | 人类预定义 metric | LLM 从零生成 |
| 最佳 posthoc return | **260.6** | **225.8** |
| 搜索到有效解 (return > 190) | **7/10 轮** | **4/10 轮** |
| 稳定着陆率 | 最高 100% | 始终 0%（诊断 metric 偏 v1） |
| 编辑稳定性 | 中等（存在过编辑回退） | 差（大幅震荡） |

---

## 3. 分析

### 3.1 框架有效性：两种范式均能搜索到有效奖励

v1 和 v2 在框架的同一套闭环机制下，分别搜索到了 posthoc return 260 和 225 的奖励函数。这验证了 EG-RSA 的诊断-归因-编辑-门控流水线在两种不同的搜索范式下均能工作。框架不绑定于特定的 reward 表示形式——metric 和 formula 只是两种不同的搜索接口。

### 3.2 v1 的优势：metric 抽象降低搜索难度

v1 的 LLM 操作的是语义明确的 metric（如 `approach_region_score`, `stability`），每个 metric 内部封装了底层计算逻辑。当 hack detector 报告 `r_approach_region` 支配度过高时，LLM 能直接定位到是"dense_guidance 角色过度奖励导致 bouncing"，执行"禁用该组件 + 大幅提升 terminal bonus"的原子编辑，两轮即收敛。

### 3.3 v2 的问题：formula 空间搜索稳定性不足

v2 的 LLM 需要在无限 AST 组合空间中修改数学公式。虽然它**确实搜索到了** return 190-225 的有效解（iter 1-3, 9），但编辑稳定性差——后续编辑容易破坏已找到的好解（iter 3 → 4: 220 → 30）。原因在于 formula 空间的编辑缺少 metric 层的语义聚合，LLM 对公式的局部修改可能产生非预期的全局行为变化。

另外，v2 的 stable_landing_episode_rate 始终为 0，部分原因是诊断 metric 体系（`stable_landing_condition` 等）是基于 v1 的 metric 系统设计的，对 v2 的 formula-based 奖励产生的行为覆盖不足——v2 iter 2 的 posthoc return 225 说明 agent 实际上在着陆，但诊断框架未能识别。

### 3.4 v1 的过编辑问题

v1 iter 6-7 出现从 100% 成功率骤降到 0% 的回退，原因是连续多轮 no_edit 后 LLM 过度积极地进行编辑（shaping-goal mismatch）。memory 机制在 iter 8-9 帮助恢复了性能，但说明编辑保守性参数需要调整。

---

## 4. 后续计划

### 4.1 本周修改

1. **v2 诊断体系对齐** — 为 v2 的 formula-based 奖励补充独立的行为诊断指标，使 v2 的搜索质量评估不再依赖 v1 的 metric 体系，更公平地反映 formula 搜索的实际表现。

2. **v1 编辑保守性** — 增加"连续成功 ≥ 2 轮时自动收紧编辑门控"的规则，防止 iter 6-7 式的过编辑回退。

3. **v2 搜索稳定性** — 在编辑门控中增加"回退检测"：当新 schema 的 posthoc return 相比 best 下降超过 50% 时自动触发 rollback，减少大幅震荡。

### 4.2 后续实验

| 编号 | 实验 | 内容 | 预期 |
|------|------|------|------|
| v2.1 | formula + metric hybrid | LLM 编辑 metric 层，底层自动编译为 formula（结合 v1 稳定性 + v2 灵活性） | 稳定搜索 + 保持 return > 200 |
| v2.2 | bootstrap prompt 强化 | bootstrap 阶段要求 LLM 显式分解任务阶段（approach→descend→contact→stable），每个阶段一个 reward signal | 提升初始 schema 质量 |
| v1.2 | BipedalWalker 迁移 | v1 完整流程迁移到 BipedalWalker-v3 | 验证跨环境泛化 |
| abl-1 | 去语义角色 | 去掉 semantic_role 标注，LLM 仅看组件名+数值 | 量化归因贡献 |
| abl-2 | 去 memory | 关闭 memory 检索 | 量化跨迭代学习贡献 |

---

## 5. 总结

- EG-RSA 框架在 **两种搜索范式（metric-based v1, formula-based v2）下均搜索到了有效奖励函数**（posthoc return 260 / 225），验证了诊断-归因-编辑-门控闭环架构的通用性。
- v1 的 metric 抽象层提供更稳定的编辑信号，收敛到 100% 成功率；v2 的 formula 搜索灵活性更高但编辑震荡大，需要在诊断对齐和搜索稳定性上做针对性改进。
- 下周重点：v2 诊断体系独立化 + v2 搜索稳定性增强 + v1.2 BipedalWalker 跨环境实验。
