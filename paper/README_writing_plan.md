# EG-RSA 论文写作路线

## 1. 推荐论文题目

推荐标题：

From Reward Generation to Reward Search: An Experience-Guided Framework for LLM-Assisted Reinforcement Learning

中文理解：

从奖励生成到奖励搜索：一种面向 LLM 辅助强化学习的历史经验驱动框架

备选标题：

1. EG-RSA: Experience-Guided Reward Schema Adaptation for LLM-Assisted Reinforcement Learning
2. Experience-Guided Reward Search with Semantic Attribution and Risk-Aware Editing
3. Toward Reliable LLM-Based Reward Design: Experience Memory, Role Attribution, and Risk-Aware Schema Editing

## 2. 论文核心定位

不要写成：

我们已经提出了一个稳定自动发现最优 reward 的系统。

应该写成：

我们提出了一个历史经验驱动的 reward search 框架，将 LLM 奖励设计从一次性 reward generation 扩展为带有记忆、归因、风险审计和回滚的闭环搜索过程。

核心表达：

Existing LLM-based reward design methods mainly show that LLMs can generate reward functions. EG-RSA further studies how LLMs can search, revise, and validate reward schemas in a history-aware and risk-aware manner.

## 3. 论文创新点

### Contribution 1: From reward generation to reward search

已有方法通常是：

    task description -> LLM -> reward function -> RL training

EG-RSA 是：

    initial reward schema
    -> RL training
    -> trajectory feedback
    -> failure diagnosis
    -> semantic role attribution
    -> experience retrieval
    -> operator-constrained edit
    -> validation and rollback
    -> outcome lesson update

可以写成：

We formulate LLM-assisted reward design as an experience-guided reward search problem rather than a one-shot reward generation problem.

### Contribution 2: Reward semantic role attribution

EG-RSA 给 reward component 加 semantic role：

    dense_guidance
    stability_quality
    terminal_success
    safety_constraint
    control_cost

作用是解释：

    当前策略为什么失败？
    哪个 reward component 主导了 agent 行为？
    agent 是真的接近任务成功，还是只是在 exploiting dense shaping？

可以写成：

EG-RSA introduces semantic role attribution to identify which reward roles dominate policy behavior and to guide structured reward edits.

### Contribution 3: Outcome lesson memory

EG-RSA 不只保存 best reward，还保存：

    effective_edit_lesson
    regression_lesson
    schema_diff
    metric_delta
    failure_modes
    rollback_decision

可以写成：

EG-RSA stores effective and regressive reward edits as structured outcome lessons, allowing future edit planning to reuse prior search experience.

### Contribution 4: Risk-aware operator-constrained editing

EG-RSA 不让 LLM 直接 free rewrite reward code，而是输出 edit plan：

    increase_weight
    decrease_weight
    add_component
    add_event_rule
    disable_component

并经过：

    scale audit
    behavior risk audit
    repair
    outcome acceptor
    rollback

可以写成：

EG-RSA constrains LLM reward modification to auditable operators and uses risk-aware auditing and rollback to reduce reward hacking and unsafe edits.

## 4. 当前实验应该怎么定位

当前实验不适合写成最终性能主结果，而适合写成：

    mechanism verification + failure-driven case study

尤其是：

1. iteration 0 -> 1 证明 semantic attribution + outcome memory 可以产生有效编辑。
2. iteration 1 -> 2 证明 regression lesson + rollback 有意义。
3. strict audit 导致 deadlock，证明过强安全审计会抑制必要探索。
4. relaxed audit 修复实验可以作为下一阶段主实验。

## 5. 论文结构建议

1. Introduction
2. Related Work
   2.1 Reward Design and Reward Shaping in Reinforcement Learning
   2.2 LLM-Assisted Reward Generation
   2.3 Memory and Reflection in Language Agents
   2.4 Reward Hacking and Risk-Aware Reward Optimization
   2.5 Positioning of EG-RSA
3. Method
   3.1 Overview
   3.2 Reward Schema Representation
   3.3 Semantic Role Attribution
   3.4 Experience Memory and Outcome Lesson
   3.5 Operator-Constrained Reward Editing
   3.6 Risk-Aware Audit and Rollback
4. Experiments
   4.1 Environment and Setup
   4.2 Main EG-RSA Experiment
   4.3 Effective Edit Case Study
   4.4 Audit-Induced Deadlock Case Study
   4.5 Relaxed Audit Policy
   4.6 Ablation Plan
5. Results and Analysis
6. Discussion
7. Conclusion and Future Work

## 6. Introduction 写作逻辑

Introduction 可以围绕这条逻辑写：

1. 强化学习依赖 reward。
2. reward 设计难，容易 reward hacking。
3. LLM 可以生成 reward，但现有方法多是 generation/refinement。
4. 真正需要的是 history-aware reward search。
5. EG-RSA 通过 reward schema、semantic role attribution、outcome lesson、operator-constrained editing 和 risk-aware rollback 实现这一点。

## 7. Related Work 写作逻辑

Related Work 不要写成文献堆砌，而要写成：

1. 经典文献说明 reward design 是重要问题。
2. LLM reward design 文献说明方向相近。
3. LLM memory 文献说明历史经验机制有依据。
4. reward hacking 文献说明风险审计有必要。
5. 最后明确 EG-RSA 与它们不同。
