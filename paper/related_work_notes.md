# Related Work Notes for EG-RSA

这个文件不是论文正文，而是文献矩阵。后面你找新文献，就按这个格式继续加。

## A. 强化学习与奖励设计基础

### Sutton & Barto, Reinforcement Learning: An Introduction

- 引用键：sutton2018reinforcement
- 类型：经典教材
- 引用目的：
  - 支撑 reinforcement learning、reward、return、policy、value function 等基础概念。
- 和 EG-RSA 的关系：
  - EG-RSA 不改底层 RL 问题定义，而是研究 reward function 如何设计和搜索。
- 可以写进正文：
  - Reinforcement learning optimizes policies through reward signals, making the reward function a central interface between task intent and policy learning.

### Schulman et al., PPO

- 引用键：schulman2017ppo
- 类型：底层 RL 算法
- 引用目的：
  - 说明实验使用 PPO 作为 policy optimizer。
- 和 EG-RSA 的关系：
  - EG-RSA 关注 reward search，PPO 是被 reward 驱动的底层训练器。
- 可以写进正文：
  - In our experiments, PPO is used as the underlying policy optimization algorithm, while EG-RSA focuses on the orthogonal problem of reward design.

## B. Reward Shaping / IRL / 自动奖励设计

### Ng et al., Policy Invariance under Reward Transformations

- 引用键：ng1999policy
- 类型：reward shaping 经典文献
- 引用目的：
  - 说明 reward shaping 是经典问题。
  - 说明设计额外 reward signal 需要注意不改变最优策略。
- 和 EG-RSA 的关系：
  - 传统 shaping 通常依赖人工设计 potential 或专家知识。
  - EG-RSA 通过训练反馈和历史经验自动修改 reward schema。
- 可以写进正文：
  - Classical reward shaping studies how additional reward signals affect policy learning and policy invariance.

### Ng & Russell, Algorithms for Inverse Reinforcement Learning

- 引用键：ng2000irl
- 类型：IRL 经典文献
- 引用目的：
  - 说明另一条自动 reward 设计路线是从专家行为中反推 reward。
- 和 EG-RSA 的区别：
  - IRL 通常依赖 expert demonstrations。
  - EG-RSA 不依赖专家示范，而是从训练失败、轨迹反馈和历史编辑经验中搜索 reward。
- 可以写进正文：
  - Unlike inverse reinforcement learning, EG-RSA does not assume access to expert demonstrations.

### Abbeel & Ng, Apprenticeship Learning via IRL

- 引用键：abbeel2004apprenticeship
- 类型：从专家示范学习 reward / policy
- 引用目的：
  - 支撑 learning from demonstration / apprenticeship learning 方向。
- 和 EG-RSA 的区别：
  - EG-RSA 是 self-improvement over reward schemas，不是 imitation from expert demonstrations。

## C. LLM-Assisted Reward Generation

### Eureka

- 引用键：ma2023eureka
- 类型：最核心相关工作
- 核心做法：
  - 使用 LLM 生成 reward code。
  - 通过 iterative / evolutionary optimization over reward code 改进 reward。
- 和 EG-RSA 相同：
  - 都使用 LLM 辅助 reward design。
  - 都通过 RL training feedback 迭代改进 reward。
- 和 EG-RSA 不同：
  - Eureka 更强调 free-form reward code generation 和 in-context evolutionary optimization。
  - EG-RSA 更强调 reward schema、semantic role attribution、outcome lesson memory、operator-constrained edit、risk audit 和 rollback。
- 可以写进正文：
  - Eureka demonstrates that LLMs can act as reward designers by generating executable reward code and improving it through iterative feedback. EG-RSA follows this motivation but explicitly models reward design as structured schema search with memory and risk-aware editing.

### Text2Reward

- 引用键：xie2023text2reward
- 类型：LLM 生成 dense reward
- 核心做法：
  - 从自然语言目标生成 dense reward functions。
  - 支持 human feedback refinement。
- 和 EG-RSA 相同：
  - 都希望降低人工 reward engineering 成本。
- 和 EG-RSA 不同：
  - Text2Reward 重点是从文本生成 dense reward。
  - EG-RSA 重点是初始 reward 失败之后，如何诊断、归因、检索经验、结构化编辑。
- 可以写进正文：
  - Text2Reward focuses on generating shaped dense rewards from language descriptions, whereas EG-RSA focuses on history-aware adaptation of reward schemas after initial reward failures.

### Auto MC-Reward

- 引用键：li2023automcreward
- 类型：LLM + dense reward + trajectory analyzer
- 核心做法：
  - Reward Designer
  - Reward Critic
  - Trajectory Analyzer
- 和 EG-RSA 相同：
  - 都使用 trajectory feedback refine reward。
- 和 EG-RSA 不同：
  - EG-RSA 明确存储 outcome lesson 和 regression lesson。
  - EG-RSA 引入 semantic role attribution 和 risk-aware operator edit。
- 可以写进正文：
  - Auto MC-Reward also uses trajectory analysis to refine rewards, but EG-RSA further stores edit outcomes as reusable lessons and audits edits at the semantic-role level.

### CARD

- 引用键：sun2024card
- 类型：很接近的 LLM reward design 工作
- 核心做法：
  - Coder + Evaluator
  - Dynamic feedback
  - Trajectory Preference Evaluation
  - 减少反复 RL training 的成本
- 和 EG-RSA 相同：
  - 都是 iterative LLM-driven reward design。
- 和 EG-RSA 不同：
  - CARD 关注 dynamic feedback 和 training/query efficiency。
  - EG-RSA 关注历史经验记忆、语义角色归因、风险审计、回滚和结构化 reward schema search。
- 可以写进正文：
  - CARD improves LLM reward design using dynamic feedback and trajectory preference evaluation, while EG-RSA focuses on storing and reusing reward-edit experience across iterations.

## D. LLM Agent Memory and Reflection

### Reflexion

- 引用键：shinn2023reflexion
- 类型：LLM agent 记忆/反思
- 核心做法：
  - 不更新模型参数。
  - 通过 verbal feedback 和 episodic memory 改进后续行为。
- 和 EG-RSA 相同：
  - 都强调 trial-and-error 后的经验复用。
- 和 EG-RSA 不同：
  - Reflexion 存语言反思。
  - EG-RSA 存结构化 reward edit outcome lesson。
- 可以写进正文：
  - Inspired by memory-based language agents, EG-RSA stores past reward-search outcomes. Unlike general verbal reflection, EG-RSA records schema diffs, metric deltas, failure modes, and rollback decisions.

### Voyager

- 引用键：wang2023voyager
- 类型：LLM lifelong embodied agent
- 核心做法：
  - automatic curriculum
  - skill library
  - iterative prompting
- 和 EG-RSA 的关系：
  - Voyager 存 skill code。
  - EG-RSA 存 reward edit knowledge。
- 可以写进正文：
  - Similar to skill libraries in lifelong LLM agents, EG-RSA maintains a reusable memory; however, the stored objects are reward-edit lessons rather than action skills.

### Generative Agents

- 引用键：park2023generative
- 类型：LLM agent memory/reflection/planning
- 引用目的：
  - 支撑 memory stream、reflection、planning 对 agent 行为的重要性。
- 和 EG-RSA 的关系：
  - EG-RSA 把 memory/reflection 思想迁移到 reward search 过程。

## E. Reward Hacking / Reward Misspecification / Safety

### Concrete Problems in AI Safety

- 引用键：amodei2016concrete
- 类型：AI safety 经典文献
- 引用目的：
  - 支撑 reward hacking、safe exploration、scalable oversight 等安全问题。
- 和 EG-RSA 的关系：
  - EG-RSA 的 hack detector、behavior risk audit、rollback 都是为了降低 reward hacking 风险。
- 可以写进正文：
  - Reward hacking motivates the need for risk-aware reward search, since agents may exploit proxy rewards while failing the intended objective.

### The Effects of Reward Misspecification

- 引用键：pan2022effects
- 类型：reward misspecification / reward hacking 系统研究
- 引用目的：
  - 支撑 proxy reward 与 true reward 可能不一致。
- 和 EG-RSA 的关系：
  - EG-RSA 区分 internal semantic outcome 和 post-hoc oracle evaluation。
- 可以写进正文：
  - This motivates EG-RSA's separation between internal semantic feedback and post-hoc oracle evaluation.

## F. Survey

### Survey on LLM-Enhanced Reinforcement Learning

- 引用键：cao2024survey
- 类型：survey
- 引用目的：
  - 说明 LLM-enhanced RL 包含 LLM-as-reward-designer 方向。
- 和 EG-RSA 的关系：
  - EG-RSA 位于 LLM-as-reward-designer，但进一步关注 reward search 和 experience memory。
