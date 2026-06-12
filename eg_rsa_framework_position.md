# EG-RSA 框架定位说明：奖励函数自进化搜索，而不是一次性奖励生成

> 版本：draft-0.1  
> 用途：用于统一论文前三章的核心叙事、方法边界、创新点和后续实验计划。  
> 注意：本文档区分“论文正文已经完成的贡献”和“下一步计划”。当前论文正文不把 raw `env.py/step()` 自动解析写成已完成贡献；该方向只放入未来工作和下一步实验计划。

---

## 1. 我们到底想做什么？

本课题的核心目标不是简单让大模型生成一个奖励函数，而是研究：

> **如何让强化学习奖励函数在训练反馈、行为诊断和历史经验的驱动下进行自我演化搜索。**

也就是说，我们关注的是 reward design 的动态闭环：

```text
初始奖励函数并不一定正确
        ↓
训练 RL 策略，收集轨迹
        ↓
诊断 reward hacking / shaping mismatch / component dominance
        ↓
LLM agent 基于诊断和经验记忆提出 reward edit
        ↓
校验、执行、再训练
        ↓
奖励函数逐轮演化
```

这和 Eureka/Text2Reward 这类“一次性或候选式奖励生成”不同。我们的研究重点是：

```text
Reward Generation  →  Reward Self-Evolution
一次生成奖励函数       多轮诊断、反思、记忆和结构化编辑
```

---

## 2. 论文应该怎么定位？

建议把会议小论文定位为：

> **An agentic reward self-evolution framework for reinforcement learning, where an LLM does not directly optimize the policy but iteratively edits a structured executable reward schema using rollout diagnostics and experience memory.**

中文表述：

> 本文提出一种基于大模型智能体的强化学习奖励函数自进化搜索框架。该框架将奖励函数表示为结构化、可校验的 AST reward schema，通过训练轨迹诊断、组件贡献归因、语义结果分析和经验记忆，引导 LLM agent 逐轮编辑奖励函数，从而将初始不完善奖励逐步演化为更符合任务目标的奖励结构。

---

## 3. 当前正文应该声称什么？

当前正文可以声称：

1. **Reward Schema IR**：我们不让 LLM 直接写任意 Python reward code，而是让其生成/编辑结构化 reward schema，并用 AST 表达公式和事件条件。
2. **Agentic Reward Editing Loop**：我们构建 ReflectionAgent、EditAgent、Validator、Audit 和 Memory 的闭环，使奖励函数可以根据训练反馈逐轮演化。
3. **Trajectory-grounded Diagnostics**：我们不用官方环境奖励指导 edit，而是使用自定义奖励组件归因、语义代理指标和轨迹行为统计进行诊断。
4. **Experience Memory over Reward Edits**：我们保存 reward edit 的 before/after、failure mode、attribution 和 outcome，用于后续迭代中的经验检索和反思。
5. **Empirical Case Study**：在 LunarLander 上，V2 从 LLM bootstrap 生成的不完善初始 schema 出发，在多轮迭代中发现并修正 reward shaping 缺陷，显著改变策略行为模式。

---

## 4. 当前正文不应该声称什么？

当前正文不要声称：

1. 已经实现完全 Eureka-like 的 raw env.py / step 函数自动解析。
2. 已经在 BipedalWalker 上完成泛化验证。
3. 当前 memory 已经是强全局搜索记忆或能严格防止退化。
4. 当前 success metric 与官方环境成功判据完全一致。
5. 当前方法已经稳定优于 Eureka/Text2Reward/人工 reward。

这些可以放入实验计划、讨论或未来工作，但不能写成已完成贡献。

---

## 5. 关于 Eureka-like 输入边界：放入下一步计划，不写入正文贡献

你的目标是最终调整到真正 Eureka-like 输入：

```text
输入：task description + env.py/step 函数
        ↓
自动抽取 observation/action variables、事件、可用状态量
        ↓
生成 primitive_interface.json
        ↓
LLM bootstrap / edit reward schema
```

这非常重要，但目前 V2 实验使用的是人工整理过的 `primitive_interface.json`。因此论文正文应表述为：

> In the current implementation, EG-RSA operates on a primitive task interface that exposes observation/action variables and safe formula variables. Moving from this interface-conditioned setting to fully raw environment-code-conditioned reward self-evolution is an important next step.

中文：

> 当前实现基于 primitive interface 条件下的奖励函数自进化；从人工整理接口进一步发展到直接读取环境源码与 step 函数，是后续工作重点。

这句话可以放在“Limitations and Future Work”，不要放在 Introduction 的贡献列表里。

---

## 6. 当前 V2 和 Eureka/Text2Reward 的区别

| 维度 | Eureka / Text2Reward 类工作 | EG-RSA-V2 当前定位 |
|---|---|---|
| 输入 | 任务描述、环境代码/接口 | 当前：primitive interface；下一步：raw env.py/step |
| LLM 输出 | 奖励代码或 dense reward code | 结构化 AST reward schema / edit plan |
| 反馈 | reward candidate performance / human feedback / trajectory feedback | reward attribution + semantic outcome + failure diagnostics + memory |
| 搜索方式 | 候选生成或 in-context improvement | reward edit transition over schema |
| 安全性 | 多依赖代码执行与筛选 | AST validator + schema compiler + operator gate |
| 记忆 | 通常较弱或任务内 prompt context | raw memory card + distilled lesson + outcome lesson |
| 核心贡献 | 自动奖励生成 | 奖励函数自进化搜索 |

---

## 7. 可以考虑的论文标题

### 推荐主标题

**EG-RSA: Experience-Guided Reward Self-Evolution with Large Language Model Agents**

优点：
- 明确强调 Experience-Guided。
- 突出 Reward Self-Evolution。
- LLM Agents 放在后面，避免变成普通 LLM reward generation。

### 备选标题

1. **Reward Self-Evolution for Reinforcement Learning via LLM Agents and Structured Reward Schemas**
2. **From Reward Generation to Reward Self-Evolution: Diagnosing and Editing Reinforcement Learning Rewards with LLM Agents**
3. **Experience-Guided Reward Schema Adaptation for Reinforcement Learning**
4. **LLM-Agentic Reward Search with Trajectory Diagnostics and Structured Memory**
5. **Towards Self-Evolving Reward Functions for Reinforcement Learning**

如果是会议小论文，我建议用第 1 个或第 2 个。第 2 个叙事性更强，适合 workshop；第 1 个更像正式方法名。

---

## 8. 我们的创新点该如何凝练？

不要写成“我们用了 LLM、memory、reflection、audit、AST、diagnostics”，那会像堆模块。建议凝练为三点：

### Innovation 1: Reward self-evolution instead of one-shot reward generation

我们把 reward design 建模为一个跨迭代的搜索过程：每一次 reward edit 都有 before/after outcome，并可被记忆、诊断和复用。

### Innovation 2: Structured AST reward schema for safe and editable reward search

LLM 不直接写任意 Python，而是输出受限 AST reward schema 和 edit operators。这样奖励函数既可执行，又能被审计、归因和局部编辑。

### Innovation 3: Trajectory-grounded diagnostic and memory loop

系统用组件归因、语义结果、failure mode 和经验记忆来指导下一轮 edit，而不是只根据最终官方奖励挑选 candidate。

这三点足够支撑会议论文的核心贡献。

---

## 9. 当前实验结果应该怎么服务论文叙事？

V2 LunarLander 10×1M 的实验不要写成“完美解决 LunarLander”，而应该写成：

1. **Bootstrap flaw**：LLM 初始 schema 语义覆盖合理，但缺少下降引导，导致悬停。
2. **Diagnostic correction**：EG-RSA 识别 `r_center_guidance` dominance 和 shaping-goal mismatch。
3. **Behavior shift**：一轮 edit 后策略从悬停负分变为可着陆高分行为。
4. **Search instability**：后续迭代出现退化，说明 reward self-evolution 需要更好的 memory evaluation 和 retrieval。
5. **Research value**：该实验揭示了 reward self-evolution 的真实挑战，而不是只展示一次性成功结果。

---

## 10. 下一步实验计划

这里可以写入项目计划，但不要作为当前论文正文已完成贡献。

### 10.1 输入边界升级：Eureka-like raw env input

目标：

```text
task_description + env.py/step 函数
        ↓
自动抽取 primitive interface
        ↓
LLM bootstrap AST schema
```

需要实现：

1. env parser / perception agent。
2. observation/action variable extractor。
3. step function semantic summarizer。
4. primitive_interface 自动生成与校验。
5. 移除 BootstrapAgent 中的 LunarLander-specific AST 示例。

### 10.2 BipedalWalker 泛化

目标：验证 EG-RSA 不是 LunarLander-specific。

流程：

1. 自动或半自动生成 BipedalWalker primitive interface。
2. 使用 task-neutral AST bootstrap prompt。
3. 跑 3×100k smoke。
4. 跑 5×1M 或 10×1M。
5. 和 manual reward、Eureka-like one-shot、no-memory、no-reflection 做对比。

### 10.3 消融实验

建议消融：

1. w/o memory。
2. w/o reflection。
3. w/o AST schema，改为字符串 formula 或 code reward。
4. w/o attribution diagnostics。
5. w/o outcome lessons。
6. one-shot bootstrap only。
7. random or heuristic edit baseline。

### 10.4 Baselines

建议 baseline：

1. Environment official reward。
2. Manual reward schema。
3. LLM one-shot reward schema。
4. LLM iterative edit without memory。
5. Eureka-style code reward generation。
6. Text2Reward-style dense reward generation。
7. EG-RSA full。

