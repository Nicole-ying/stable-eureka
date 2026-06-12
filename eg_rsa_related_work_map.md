# EG-RSA 相关工作版图与参考文献补充

> 目的：给论文前两章提供文献结构，而不是机械堆砌。相关工作必须服务于 EG-RSA 的核心问题：奖励函数为什么难设计，LLM 如何生成奖励，为什么一次生成不够，为什么需要 agentic self-evolution、结构化 schema、诊断和记忆。

---

## 1. 相关工作主线设计

建议 Related Work 不要按“谁都讲一点”的方式写，而是围绕一条逻辑链：

```text
Reward design is hard
        ↓
Classical reward shaping / IRL / reward machines provide structure but require human design or demonstrations
        ↓
LLMs enable automatic reward generation, but often produce code-level rewards and rely on candidate evaluation
        ↓
Agentic LLM systems introduce reflection, memory and tool feedback, but are rarely specialized for RL reward evolution
        ↓
EG-RSA connects these threads: structured reward schema + rollout diagnostics + experience memory + iterative reward editing
```

---

## 2. 必须覆盖的文献类别

### 2.1 Classical reward shaping and reward misspecification

这部分用于说明：奖励函数不是普通工程细节，而是 RL 成败的核心。

应提到：

1. Potential-Based Reward Shaping, Ng et al. 1999。
2. Inverse Reinforcement Learning, Ng and Russell 2000。
3. Reward hacking / specification gaming, Amodei et al. 2016；Krakovna et al. specification gaming examples。
4. Inverse Reward Design, Hadfield-Menell et al. 2017。
5. Reward Machines, Icarte et al. 2020。

写作重点：

- PBRS 提供理论保证，但要求人类设计 potential function。
- IRL 从专家行为恢复 reward，但需要 demonstrations。
- Reward machines 暴露 reward 结构，但通常仍需要人工 specification。
- Reward hacking 说明 proxy reward 被优化后会偏离真实目标。

### 2.2 LLM-based reward design

这部分是 EG-RSA 的直接上游。

必须提到：

1. Text2Reward：LLM 根据语言目标生成 dense reward code。
2. Eureka：LLM 进行 reward code evolutionary optimization。
3. DrEureka：LLM 同时设计 reward 和 domain randomization，用于 sim-to-real。
4. Auto MC-Reward：Minecraft 中 LLM reward designer + reward critic + trajectory analyzer。

写作重点：

- 这些工作证明 LLM 能自动生成有效 reward。
- 但多数工作仍以 code reward / candidate generation 为中心。
- reward 的历史 edit 经验、结构化可审计表达、跨迭代 outcome memory 没有成为核心对象。
- EG-RSA 的切入点是 reward self-evolution，而不是 one-shot reward synthesis。

### 2.3 LLM agents, reflection, memory, and self-improvement

这部分说明为什么我们用 agentic loop 而不是单个 LLM call。

必须提到：

1. ReAct：LLM interleave reasoning and acting。
2. Reflexion：verbal reinforcement learning + episodic memory。
3. Self-Refine：LLM 自反馈迭代修改。
4. Voyager：Minecraft 中自动课程、skill library、environment feedback。
5. Generative Agents：observation, memory, reflection architecture。
6. AutoGen：multi-agent conversation framework。
7. EvoAgent / agent evolution 类工作：agent framework / skill / multi-agent evolution。

写作重点：

- 这些工作说明 reflection 和 memory 能增强 LLM agent 的持续改进能力。
- 但它们通常不是为 RL reward design 设计的。
- EG-RSA 把 memory 的对象从“任务解法/skill/code”转成“reward edit transition”。

### 2.4 Structured reward representations

这部分支撑 AST schema 的必要性。

可以提到：

1. Reward Machines。
2. Programmatic reward / code-as-reward 类工作。
3. Typed DSL / program synthesis for reward specification。
4. Safe code generation / constrained generation 相关工作。

写作重点：

- 直接生成 Python reward code 灵活但不安全、不稳定、难编辑。
- 结构化 reward IR 牺牲部分自由度，换来可验证、可归因和可演化。
- EG-RSA 的 AST schema 是为了让 LLM 做 semantic design，而不是无限制写代码。

---

## 3. 推荐引用列表

### 3.1 Classical RL and reward design

- Sutton and Barto. Reinforcement Learning: An Introduction.
- Ng, Harada, Russell. Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping. ICML 1999.
- Ng and Russell. Algorithms for Inverse Reinforcement Learning. ICML 2000.
- Ziebart et al. Maximum Entropy Inverse Reinforcement Learning. AAAI 2008.
- Hadfield-Menell et al. Inverse Reward Design. NeurIPS 2017.
- Amodei et al. Concrete Problems in AI Safety. 2016.
- Krakovna et al. Specification Gaming: The Flip Side of AI Ingenuity. 2020.
- Icarte et al. Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning. JAIR / arXiv 2020.

### 3.2 LLM-based reward design

- Xie et al. Text2Reward: Reward Shaping with Language Models for Reinforcement Learning. 2023.
- Ma et al. Eureka: Human-Level Reward Design via Coding Large Language Models. ICLR 2024.
- Ma et al. DrEureka: Language Model Guided Sim-To-Real Transfer. 2024.
- Li et al. Auto MC-Reward: Automated Dense Reward Design with Large Language Models for Minecraft. 2023.

### 3.3 Agentic LLM and self-improvement

- Yao et al. ReAct: Synergizing Reasoning and Acting in Language Models. 2022.
- Shinn et al. Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS 2023.
- Madaan et al. Self-Refine: Iterative Refinement with Self-Feedback. 2023.
- Wang et al. Voyager: An Open-Ended Embodied Agent with Large Language Models. 2023.
- Park et al. Generative Agents: Interactive Simulacra of Human Behavior. 2023.
- Wu et al. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. 2023.
- Yuan et al. EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms. 2024.
- Feng et al. EvoAgent: Agent Autonomous Evolution with Continual World Model for Long-Horizon Tasks. 2025.

### 3.4 RLHF / reward modeling / alignment background

- Christiano et al. Deep Reinforcement Learning from Human Preferences. 2017.
- Ouyang et al. Training Language Models to Follow Instructions with Human Feedback. 2022.
- Casper et al. Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback. TMLR 2023.
- Skalse et al. Defining and Characterizing Reward Hacking. 2022.

---

## 4. Related Work 建议结构

### 2.1 Reward Design and Reward Misspecification in Reinforcement Learning

重点讲 reward shaping、IRL、reward hacking、reward machines。

### 2.2 Large Language Models for Automatic Reward Generation

重点讲 Text2Reward、Eureka、DrEureka、Auto MC-Reward。最后引出不足：reward code 生成和候选评估很强，但 reward edit history / structured IR / memory-driven evolution 不足。

### 2.3 Language Agents with Reflection and Memory

重点讲 ReAct、Reflexion、Self-Refine、Voyager、AutoGen、EvoAgent。最后引出：这些方法启发了 agentic loop，但没有解决 RL reward self-evolution。

### 2.4 Our Position: Reward Self-Evolution over Structured Schemas

用一小段总结：EG-RSA 位于 LLM reward generation 和 LLM agent self-improvement 的交叉点。

---

## 5. 不建议的写法

不要写成：

```text
某某提出了 A。某某提出了 B。某某提出了 C。
```

应该写成：

```text
已有工作解决了 reward generation 的第一步，但没有把 reward design 作为可记忆、可诊断、可编辑的跨迭代搜索过程。
```

这才会自然引到 EG-RSA。

---

## 6. 关于 EVALOE / EvoAgent

目前没有确认到名为 “EVALOE” 的可靠 reward-design 论文条目。更可能需要补的是 EvoAgent / agent evolution 相关工作，尤其是：

- EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms.
- EvoAgent: Agent Autonomous Evolution with Continual World Model for Long-Horizon Tasks.

如果你说的 EVALOE 是另一个具体论文名，需要再给我准确标题或截图。我在当前文献版图中先把它归入 “agent evolution / autonomous agent self-improvement” 方向处理。

