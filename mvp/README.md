# MVP: 全自动闭环奖励函数自进化 Agent

## 你关心的问题（本版已整改）

1. **“你真的跑通过吗？”**
   - 提供了 `mock` provider，可在无云端 API 的情况下本地跑通闭环。
2. **“能切 Ollama 吗？”**
   - `model.provider` 支持 `openai | ollama | mock`，可 CLI 覆盖。
3. **“为什么一个 config 里看起来有多个环境？”**
   - 一次运行只会用一个 `rl.env_id`；
   - 多环境通过多个 YAML 配置文件管理（每个环境一份）。
4. **“prompt 是否通用且可迁移？”**
   - Prompt 采用“通用模板 + 任务规格注入（TaskSpec）”模式，迁移到新环境只需新增 spec。
5. **“是否参考 Eureka 的迭代记录思想？”**
   - 已记录 parent_ids、reflection_summary、候选分数与原因，便于复现实验与回溯。

---

## 1) Prompt 设计（显式文件）

- `mvp/prompts/planner_system.txt`
- `mvp/prompts/reward_coder_system.txt`
- `mvp/prompts/reflection_system.txt`
- `mvp/prompts/vision_judge_system.txt`

## 2) 任务与环境（可迁移）

- `mvp/task_specs.py`：定义每个环境的 `objective / obs/action hints / success/failure / judge rubric`
- 当前预置：`LunarLander-v3`, `BipedalWalker-v3`
- 新环境迁移：新增 `TaskSpec` 条目即可

## 3) 配置模式

每次运行只用一个环境配置：

- `mvp/configs/lunar_lander.yaml`
- `mvp/configs/bipedal_walker.yaml`
- `mvp/configs/cartpole_mock.yaml`（本地闭环 smoke test）

支持 CLI 覆盖：

```bash
python run_mvp.py --config mvp/configs/lunar_lander.yaml --provider ollama
python run_mvp.py --config mvp/configs/cartpole_mock.yaml
```

## 4) 自进化闭环

每一代：

1. 读 top-k 历史候选
2. Reflection 生成 mutation 指导
3. RewardCoder 基于父代代码做变异
4. RL 训练 + 回放
5. Judge 打分
6. 写入记忆，进入下一代

## 5) 输出

- `runs/*/memory.jsonl`
- `runs/*/videos/*.gif`
- `runs/*/checkpoints/*.zip`
- `runs/*/report.md`
- `runs/*/memory.csv`（可视化友好，含 generation/score/error_type 等列）


## 6) 实验停止条件

默认停止：
- 跑满 `evolution.generations`；

可选提前停止：
- 达到 `evolution.target_score`；
- 连续 `evolution.max_stagnation_generations` 代无提升。

## 7) 记忆跨次运行关系

- 记忆写在 `workspace/memory.jsonl`；
- 如果两次运行使用同一个 `workspace`，第二次会继续读取并追加（有关联）；
- 如果换了 `workspace`，就相当于新实验。
