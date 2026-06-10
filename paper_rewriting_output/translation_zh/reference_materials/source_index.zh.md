# 源索引（中文）

## 项目: EG-RSA（经验引导的奖励搜索智能体）

| 源 ID | 类型 | 标题/名称 | 来源 | 为何包含 | 用途 |
|---|---|---|---|---|---|
| DSG-01 | 设计文档 | EG-RSA 设计与实现手册 | `docs/EG_RSA_DESIGN.md` | 权威方法描述、架构、贡献、禁用声称 | 方法、论断、架构 |
| WRT-01 | 写作计划 | EG-RSA 论文写作计划 | `paper/README_writing_plan.md` | 论文结构、标题提案、创新框架、逐节写作逻辑 | 结构、语调、定位 |
| RW-01 | 相关文献草稿 | 相关文献草稿 | `paper/related_work_draft.md` | 含引用的完整相关文献部分草稿 | 相关文献部分 |
| RW-02 | 相关文献笔记 | 相关文献矩阵 | `paper/related_work_notes.md` | 所有参考文献的结构化引用到论断映射 | 引用锚定、定位 |
| BIB-01 | 参考文献 | EG-RSA 参考文献库 | `paper/references_eg_rsa.bib` | 12 条 BibTeX 条目涵盖 RL、奖励设计、LLM 智能体、安全 | 所有引用 |
| DSN-02 | 设计文档 | V1 发布完整设计 | `docs/v1release/V1_RELEASE_FULL_DESIGN.md` | 完整 v1 功能集和发布设计 | 方法完整性 |
| ISS-01 | 问题 | EG-RSA 待解决问题 | `docs/eg_rsa_open_issues.md` | 已知局限、待解决问题 | 讨论、局限 |
| EXP-01 | 实验 | 主角色归因实验（10×1M） | `experiments/eg_rsa_lunar_lander_v1_role_attrib_10x1m/` | 含完整轨迹、记忆、审计、归因的 10 轮 EG-RSA 运行 | 主要实验证据 |
| EXP-02 | 实验 | 放宽审计实验 | `experiments/eg_rsa_lunar_lander_v1_role_attrib_10x1m_audit_relaxed/` | 放宽审计死锁验证运行 | 死锁解除证据 |
| CFG-01 | 配置 | EG-RSA DeepSeek v1 配置 | `configs/eg_rsa_deepseek_v1_role_attrib_10x1m.yml` | 实验超参数、LLM 设置、审计规则 | 可复现性 |
| ENV-01 | 环境 | LunarLander-v3 | Gymnasium 套件 | 标准连续控制基准 | 实验环境 |
| FIG-01 | 图表 | 工作流图 | `img/stable-eureka_workflow.png` | EG-RSA 搜索循环可视化 | 图 1 |
