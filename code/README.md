# LLM Reward Evolver

本目录实现了一个最小可行版的强化学习奖励函数自进化框架，覆盖文档里的两个创新点：

- **FDRE**：LLM 生成奖励函数、训练智能体、根据评估反馈迭代优化。
- **HRDC**：奖励函数按子目标分解，并在代码内按训练阶段动态组合权重。

## 快速 dry-run

不安装 RL 依赖也可以先验证 LLM 奖励函数生成、校验和文件输出：

```bash
python scripts/run_experiment.py --config config.example.json --dry-run
```

## 真实训练

安装依赖后运行：

```bash
python -m pip install -r requirements.txt
python scripts/run_experiment.py --config config.example.json
```

## 三组对比实验

一次性运行原始奖励 baseline、LLM 单次生成、FDRE 迭代优化：

```bash
python scripts/run_experiment.py --config config.example.json --suite
```

输出会写入对应目录下的 `suite_summary.json`、`suite_summary.md` 和 `paper_experiment_draft.md`，包含平均得分、成功率、平均 episode 长度、中断状态和奖励函数错误次数。

更复杂环境的配置已放在：

- `configs/mountaincar_stress.json`
- `configs/acrobot_stress.json`
- `configs/lunarlander_stress.json`
- `configs/lunarlander.json`
- `configs/bipedalwalker.json`

如需使用 DeepSeek 兼容接口：

```bash
set DEEPSEEK_API_KEY=你的key
python scripts/check_deepseek.py
python scripts/run_experiment.py --config configs/cartpole_stress.json --suite
```

当前默认实验配置已经切换为 DeepSeek 真模型，模型名为 `deepseek-chat`。代码不会保存 API Key，只从环境变量 `DEEPSEEK_API_KEY` 读取。

如需使用本地开源模型（Ollama）：

```bash
ollama run qwen2.5-coder:7b
python scripts/run_experiment.py --config config.example.json --llm-provider ollama --llm-model qwen2.5-coder:7b
```

也可以通过环境变量切换：

- `OLLAMA_HOST`
- `OLLAMA_MODEL`
- `OLLAMA_TEMPERATURE`

推荐主实验优先使用 `configs/lunarlander.json` 或 `configs/bipedalwalker.json`，更容易拉开 baseline、单次生成和 FDRE 的差异。

## 主要入口

- `scripts/run_experiment.py`：命令行入口。
- `src/llm_reward_evolver/evolver.py`：FDRE 迭代主流程。
- `src/llm_reward_evolver/reward.py`：LLM 奖励代码校验与执行。
- `src/llm_reward_evolver/wrappers.py`：Gymnasium 奖励替换包装器。
- `src/llm_reward_evolver/prompts.py`：分层 prompt，避免把具体技巧写死。
