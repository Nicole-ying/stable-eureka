# EG-RSA V1 / V2 实验记录分析

> 目的：给论文实验部分和方法迭代说明提供素材。正文不直接写“V1/V2 开发史”，但这些分析可以帮助我们提炼最终 V2 的动机和设计选择。

---

## 1. V1 实验：`eg_rsa_lunar_lander_v1_role_attrib_10x1m_audit_relaxed`

### 1.1 V1 做了什么？

V1 的核心不是 LLM bootstrap，而是从一个较强的人工/半人工初始 reward schema 出发，引入：

1. role attribution：分析不同 reward component 的贡献比例；
2. semantic outcome：记录 success、safe contact、stable landing 等语义指标；
3. edit agent：根据诊断提出修改；
4. audit relaxed：较宽松地允许 edit 进入下一轮；
5. continuation：当 reward appears aligned 时继续训练而不是频繁修改。

V1 的作用是证明：**基于 reward component attribution 的诊断式 reward editing 是有用的。**

---

### 1.2 V1 关键结果

从 summary 可以看到：

- Iter 0：posthoc return = 2.83，success rate = 0，failure mode 为 repeated_event_exploitation + shaping_goal_mismatch。
- Iter 1：posthoc return = 29.36，仍未成功，但 edit agent 诊断出 dense region reward loitering。
- Iter 2：posthoc return = 154.51，success rate = 0.833，terminal reward paid = 1.0。
- Iter 3：posthoc return = 200.46，success rate = 1.0。
- Iter 4：posthoc return = 260.59，success rate = 1.0。
- Iter 5：posthoc return = 221.57，success rate = 1.0。
- Iter 6：posthoc return = 104.48，success rate = 0.5。
- Iter 7：posthoc return = -304.24，出现明显退化，dominant component 为 `r_landing_region_entry_once`，比例 0.878。
- Iter 8：posthoc return = 202.93，success rate = 0.5。
- Iter 9：posthoc return = 200.51，success rate = 1.0。

结论：V1 能够在 Iter 2-5 达到较好的 LunarLander 表现，但后续仍存在 reward edit 导致退化的问题。

---

### 1.3 V1 的优点

1. **成功率指标更对齐**：V1 的 success_episode_rate 在 Iter 2 后明显变为非零，并多轮达到 1.0。
2. **诊断有效**：系统能识别 dense reward loitering、shaping-goal mismatch 和 component dominance。
3. **训练闭环有效**：Iter 0/1 到 Iter 2/3 出现明显行为改善。
4. **no-edit / continuation 策略有价值**：当 reward appears aligned，继续训练比频繁 edit 更稳。

---

### 1.4 V1 的不足

1. **初始 schema 依赖较强设计**：V1 更像从人工较好 reward schema 出发，而不是 LLM 从 primitive interface 自主 bootstrap。
2. **reward schema 表达不够结构化**：相比 V2 的 AST schema，V1 更难做到公式级安全校验和跨环境迁移。
3. **退化仍存在**：Iter 7 出现从高回报到 -304 的严重退化，说明 memory/rollback/trust 对编辑稳定性的约束不足。
4. **不适合作为最终论文主线**：V1 更像方法开发中的中间阶段，可作为 internal evidence，不建议在会议正文中作为主要方法版本展开。

---

## 2. V2 实验：`eg_rsa_lunar_lander_v2_bootstrap_10x1m`

### 2.1 V2 做了什么提升？

V2 相比 V1 的主要提升：

1. **LLM bootstrap**：从 primitive interface 自动生成初始 reward schema。
2. **AST-first reward schema**：将公式和条件表示为受限 AST，而不是自由字符串公式。
3. **safe compiler / validator**：奖励 schema 可校验、可编译、可审计。
4. **ReflectionAgent + EditAgent**：将诊断反思和具体 edit 计划分离。
5. **三层 memory**：raw memory card、distilled lesson、outcome lesson。
6. **trajectory-grounded diagnostics**：使用 component attribution、semantic outcome、failure modes 指导 edit。

---

### 2.2 V2 关键结果

10×1M 结果显示：

- Iter 0：posthoc return = -552.34，success rate = 0，策略悬停；dominant component 为 `r_center_guidance`，ratio = 0.916。
- Iter 1：posthoc return = 193.06，success rate = 0，策略开始出现着陆行为；dominant component 变为 `r_landing_success`，ratio = 0.512。
- Iter 2：posthoc return = 225.78，为本实验最优；success rate 仍为 0。
- Iter 3：posthoc return = 220.59。
- Iter 4-8：出现明显震荡和退化，最低到 -352.00 / -292.47。
- Iter 9：posthoc return = 198.80，恢复较高回报。

---

### 2.3 V2 的价值

V2 最有价值的不是 success rate，而是：

1. **从不完美 bootstrap 出发**：初始 schema 漏掉 descent guidance，导致悬停。
2. **诊断能抓住问题**：Iter 0 识别 `r_center_guidance` dominance。
3. **一轮 edit 改变策略模式**：Iter 1 从 -552 到 193，说明 reward edit 对行为有实质影响。
4. **AST schema 跑通完整闭环**：LLM 生成/编辑不再依赖自由字符串公式。
5. **暴露真实挑战**：后续震荡说明 reward self-evolution 需要更强的 memory evaluation 和稳定性设计。

---

### 2.4 V2 的不足

1. **success proxy 未对齐**：official posthoc return 可达 225，但 internal success_rate 始终为 0，说明 diagnostic success predicate 过严或不匹配。
2. **bootstrap prompt 有 task-specific bias**：当前 primitive interface 和 AST grammar 包含 LunarLander 风格变量，如 `x`、`left_contact/right_contact`、`main_engine`。
3. **memory 是软证据**：memory 被读写并进入 prompt，但不能稳定保留历史最优 schema。
4. **迭代不稳定**：Iter 4-8 出现退化。

这些问题不应该作为论文正文的主贡献，但应该作为下一步实验和限制说明。

---

## 3. V1 到 V2 的方法提升

| 维度 | V1 | V2 |
|---|---|---|
| 初始奖励 | 较强人工/半人工 schema | LLM bootstrap schema |
| 表达方式 | reward components + roles | AST-first structured reward schema |
| 公式安全 | 较弱 | 强校验、可编译 |
| 诊断 | role attribution + semantic outcome | attribution + semantic outcome + ReflectionAgent |
| memory | 初步 memory | raw memory + distilled lesson + outcome lesson |
| 泛化潜力 | 较弱 | 更适合跨任务，但需要输入边界升级 |
| 当前稳定性 | 表现更好，但仍有退化 | 更自动，但震荡更明显 |

---

## 4. 论文应该怎么使用 V1/V2？

论文正文建议只呈现最终框架 EG-RSA，不写“V1/V2 开发史”。

但内部叙事可以借鉴：

- V1 证明 role attribution 和 semantic diagnostics 有价值。
- V2 证明 LLM bootstrap + AST schema + memory-driven edit loop 可跑通。
- 最终论文应吸收 V1 的稳定 continuation/no-edit 思想和 V2 的 AST schema/bootstrap 思想，形成一个统一、干净的框架。

---

## 5. 实验补充建议

### 5.1 必做

1. 修正 LunarLander success metric。
2. 重新跑 V2 3-5 seeds。
3. 做 one-shot bootstrap baseline。
4. 做 w/o memory、w/o reflection、w/o attribution 消融。
5. 增加 BipedalWalker smoke + long run。

### 5.2 可选

1. 与 Eureka-like code reward baseline 对比。
2. 与 Text2Reward-style one-shot dense reward baseline 对比。
3. 对 prompt bias 做 ablation：
   - task-specific AST examples；
   - task-neutral AST examples；
   - raw env extracted primitive interface。

---

