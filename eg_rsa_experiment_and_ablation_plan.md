# EG-RSA 后续实验、消融与 Baseline 计划

> 目的：把论文从“一个 LunarLander case study”推进到更像会议论文的实验设计。

---

## 1. 当前必须补的实验问题

### 1.1 修正 LunarLander success metric

当前 V2 出现 official posthoc return 高，但 internal success_rate 始终为 0 的问题。需要把 success 拆开：

1. diagnostic_success：由 stable_landing_condition 触发。
2. terminal_reward_paid：终端 reward 是否支付。
3. official_success_proxy：posthoc_return >= 200。
4. safe_landing_proxy：速度、姿态、双腿接触综合判断。

论文正文中要避免只用一个 success rate。

---

## 2. Baseline 设计

### Baseline 1: Official Environment Reward

用 Gymnasium 官方 reward 训练 PPO，作为环境上限/参考。

### Baseline 2: Manual Reward Schema

使用人工设计 reward schema，评估人工 reward engineering 的表现。

### Baseline 3: LLM One-shot AST Schema

只用 LLM bootstrap 的初始 AST schema，不做后续 edit。

### Baseline 4: LLM Iterative Edit without Memory

保留 diagnostics 和 edit agent，但关闭 memory。

### Baseline 5: LLM Iterative Edit without Reflection

直接让 EditAgent 根据 diagnostic report 生成 edit，不经过 ReflectionAgent。

### Baseline 6: Code Reward Generation Baseline

如果时间允许，实现 Eureka-like code reward generation baseline。可先做弱版本：LLM 直接生成 reward.py，然后 validator 执行训练。

### Baseline 7: EG-RSA Full

完整系统：AST schema + diagnostics + reflection + memory + validator + audit。

---

## 3. 消融实验

建议最小消融矩阵：

| Setting | AST | Diagnostics | Reflection | Memory | Expected purpose |
|---|---|---|---|---|---|
| One-shot bootstrap | ✓ | ✗ | ✗ | ✗ | 测初始 LLM reward 能力 |
| w/o Memory | ✓ | ✓ | ✓ | ✗ | 测 memory 是否稳定搜索 |
| w/o Reflection | ✓ | ✓ | ✗ | ✓ | 测策略反思是否必要 |
| w/o Attribution | ✓ | partial | ✓ | ✓ | 测 component attribution 价值 |
| String Formula Schema | ✗ | ✓ | ✓ | ✓ | 测 AST schema 的稳定性 |
| Full EG-RSA | ✓ | ✓ | ✓ | ✓ | 完整方法 |

---

## 4. BipedalWalker 泛化计划

目标：验证 EG-RSA 不只是 LunarLander-specific。

### 4.1 输入边界

论文正文不要声称已经完成 raw env parsing。下一步计划是：

```text
task description + env.py / step()
        ↓
自动/半自动 primitive interface extraction
        ↓
AST bootstrap
        ↓
EG-RSA self-evolution
```

### 4.2 BipedalWalker 实验阶段

1. 手动或半自动生成 primitive interface smoke。
2. 用 task-neutral AST prompt，移除 LunarLander 示例。
3. 运行 3×100k smoke。
4. 运行 5×1M 或 10×1M long run。
5. 跑至少 3 seeds。
6. 对比 official reward、one-shot bootstrap、w/o memory、full EG-RSA。

---

## 5. 评价指标

### 5.1 Training-free / schema-level

- schema validity rate
- AST validation pass rate
- edit validation pass rate
- unsafe formula rejection count
- number of committed edits

### 5.2 Policy-level

- official posthoc return mean/std
- episode length
- environment success proxy
- diagnostic success
- terminal reward paid rate

### 5.3 Reward-search-level

- best iteration index
- improvement from bootstrap
- regression count
- component dominance ratio
- failure mode count
- memory retrieval hit rate
- memory-use audit score

---

## 6. 论文实验表格建议

### Table 1: Main LunarLander result

| Method | Return mean | Return std | Diagnostic success | Terminal paid | Best iter |
|---|---:|---:|---:|---:|---:|
| Official reward PPO | | | | | |
| Manual schema | | | | | |
| LLM one-shot AST | | | | | |
| EG-RSA w/o memory | | | | | |
| EG-RSA full | | | | | |

### Table 2: Ablation

| Variant | Valid edit rate | Best return | Regression count | Notes |
|---|---:|---:|---:|---|
| w/o reflection | | | | |
| w/o memory | | | | |
| w/o attribution | | | | |
| string formula schema | | | | |
| AST schema full | | | | |

### Figure 1: Framework flowchart

用第三章中的 Mermaid 图转成 PDF/PNG。

### Figure 2: Reward evolution curve

横轴 iteration，纵轴：

- posthoc return
- semantic score
- dominant component ratio

### Figure 3: Component attribution stacked bar

展示每轮 reward payment ratio。

---

## 7. 最小可行补实验路线

如果时间有限，建议按这个顺序：

1. 修 success metric。
2. 重跑 V2 LunarLander 3 seeds × 10×1M。
3. 跑 one-shot AST baseline。
4. 跑 w/o memory。
5. 跑 BipedalWalker 3×100k smoke。
6. 如果 Bipedal smoke 有希望，再上 5×1M。

