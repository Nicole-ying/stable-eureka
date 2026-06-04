# 代码交付说明

本目录为 FDRE-HRDC 实验代码快照，包含核心源码、实验脚本、配置文件、测试文件和依赖声明。

## 目录

- `src/llm_reward_evolver/`：奖励函数生成、自进化、反馈诊断、训练封装和报告生成模块。
- `scripts/run_experiment.py`：实验入口，支持 `mock`、`ollama`、`deepseek` 调用方式。
- `scripts/generate_publication_figures.py`：论文版单图生成脚本。
- `configs/lunarlander_paper.json`：LunarLander-v3 对比实验配置。
- `tests/`：基础奖励函数测试。

## 常用命令

```powershell
python scripts/run_experiment.py --config configs/lunarlander_paper.json --llm-provider ollama --llm-model qwen2.5-coder:7b --suite
python scripts/generate_publication_figures.py
python -m pytest tests
```

当前交付数据对应 LunarLander-v3：1M 最终评估平均得分 `226.04 +/- 6.31`，三个 seed 均超过 200 分。
