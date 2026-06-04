# FDRE-HRDC 公式化交付说明

## 1. 任务与交付范围

本交付围绕 LunarLander-v3 环境验证 FDRE-HRDC 奖励函数自进化方法。训练算法采用 PPO，评价指标统一使用环境原始奖励，而不是训练过程中加入 shaping 后的奖励。因此，最终分数可以直接与 LunarLander-v3 的 solved 标准进行比较。

设马尔可夫决策过程为：

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r^{env},\gamma),
$$

其中 $s_t\in\mathcal{S}$ 为状态，$a_t\in\mathcal{A}$ 为动作，$P(s_{t+1}|s_t,a_t)$ 为状态转移，$r_t^{env}$ 为环境原始奖励。本文方法不改变环境评价口径，只在训练阶段构造辅助奖励：

$$
r_t^{train}=r_t^{FDRE},\qquad R^{eval}=\sum_{t=0}^{T} r_t^{env}.
$$

最终报告中的所有对比实验、消融实验和 1M 稳定性验证均使用 $R^{eval}$ 作为得分。

## 2. 方法定义

FDRE-HRDC 将大模型奖励函数生成、训练反馈诊断、奖励安全检查、HRDC 分层奖励建模和候选奖励筛选组成闭环。给定当前转移样本：

$$
\tau_t=(s_t,a_t,s_{t+1},r_t^{env},p_t),
$$

其中 $p_t\in[0,1]$ 表示训练进度，FDRE-HRDC 生成训练奖励函数：

$$
r_t^{FDRE}=f_\theta(s_t,a_t,s_{t+1},r_t^{env},p_t).
$$

大模型在第 $k$ 轮生成候选函数集合：

$$
\mathcal{F}_k=\left\{f_{\theta_1}^{(k)},f_{\theta_2}^{(k)},\ldots,f_{\theta_m}^{(k)}\right\}.
$$

系统对候选函数进行语义检查、运行检查和短程训练反馈，根据综合目标选择最优候选：

$$
f^*=\arg\max_{f\in\cup_k\mathcal{F}_k}J(f).
$$

## 3. HRDC 分层奖励结构

在 LunarLander-v3 中，状态记为：

$$
s=(x,y,v_x,v_y,\alpha,\dot\alpha,l,r),
$$

其中 $x,y$ 表示相对位置，$v_x,v_y$ 表示速度，$\alpha$ 表示机身角度，$\dot\alpha$ 表示角速度，$l,r$ 表示左右支架接触状态。

HRDC 将奖励分解为可解释子目标：

$$
\Phi=\left\{\phi_{prog},\phi_{center},\phi_{slow},\phi_{upright},\phi_{contact},\phi_{both},\phi_{fuel},\phi_{terminal}\right\}.
$$

各子项定义如下：

$$
\phi_{prog}=(|x_t|+|y_t|)-(|x_{t+1}|+|y_{t+1}|),
$$

$$
\phi_{center}=\max(0,1-|x_{t+1}|),
$$

$$
\phi_{slow}=\max(0,1-|v_{x,t+1}|-|v_{y,t+1}|),
$$

$$
\phi_{upright}=\max(0,1-|\alpha_{t+1}|-0.5|\dot\alpha_{t+1}|),
$$

$$
\phi_{contact}=l_{t+1}+r_{t+1},\qquad
\phi_{both}=\mathbb{I}(l_{t+1}>0.5\land r_{t+1}>0.5).
$$

动作代价定义为：

$$
c(a_t)=
\begin{cases}
0.012, & a_t=2,\\
0.004, & a_t\in\{1,3\},\\
0, & a_t=0.
\end{cases}
$$

终止状态保护项定义为：

$$
\phi_{terminal}=4\mathbb{I}(r_t^{env}>70)-4\mathbb{I}(r_t^{env}<-70).
$$

## 4. 动态权重机制

FDRE-HRDC 的关键点不是固定叠加 shaping 项，而是根据训练阶段调整奖励关注点。定义阶段权重：

$$
w_i(p_t)=(1-p_t)w_i^{early}+p_t w_i^{late}.
$$

早期阶段更强调靠近目标、保持居中和姿态稳定：

$$
\Phi_{early}=0.04\phi_{prog}+0.04\phi_{center}+0.03\phi_{upright}.
$$

后期阶段更强调低速、接触和双腿稳定着陆：

$$
\Phi_{late}
=0.02\phi_{prog}
+0.05\phi_{center}
+0.06\phi_{slow}
+0.06\phi_{upright}
+0.08\phi_{contact}
+0.12\phi_{both}.
$$

最终训练奖励为：

$$
r_t^{FDRE}
=r_t^{env}
+(1-p_t)\Phi_{early}
+p_t\Phi_{late}
+\phi_{terminal}
-c(a_t).
$$

该形式保证训练过程中仍保留原始任务奖励，同时用可解释子目标增强探索、降落稳定性和终止状态区分。

## 5. 候选奖励选择目标

对每个候选奖励函数 $f$，系统记录其原始环境得分、成功率、episode 长度、奖励函数错误和训练中断情况。综合评分定义为：

$$
J(f)=\bar R(f)+\alpha S(f)-\beta L(f)-\gamma E(f)-\eta I(f),
$$

其中：

$$
\bar R(f)=\frac1n\sum_{j=1}^{n}R_j^{eval}(f),
$$

$$
S(f)=\frac1n\sum_{j=1}^{n}\mathbb{I}(R_j^{eval}\ge 200),
$$

$$
E(f)=\text{reward error count},\qquad I(f)=\text{interruption indicator}.
$$

因此，FDRE-HRDC 不是只追求单次分数，而是在分数、成功率、稳定性和可运行性之间进行综合筛选。

## 6. 实验设置

实验环境为 LunarLander-v3。主要实验分为两组：

1. 100k timesteps：用于验证样本效率、奖励自进化过程和消融实验差异。
2. 1M timesteps：用于验证最终策略是否稳定达到 solved 标准。

对比方法包括 PPO Baseline、LLM Once、FDRE-HRDC、w/o Diagnostic 和 w/o Dynamic。评价均采用原始环境奖励：

$$
R^{eval}=\sum_t r_t^{env}.
$$

多 seed 均值与方差定义为：

$$
\mu_R=\frac1n\sum_{j=1}^{n}R_j,\qquad
\sigma_R=\sqrt{\frac1n\sum_{j=1}^{n}(R_j-\mu_R)^2}.
$$

## 7. 主要结果

1M timesteps 最终验证中，FDRE-HRDC 三个 seed 的原始环境得分为：

$$
R_{42}=223.40,\qquad R_{43}=219.98,\qquad R_{44}=234.74.
$$

因此：

$$
\mu_R=226.04,\qquad \sigma_R=6.31,\qquad \min_jR_j=219.98>200.
$$

这说明 FDRE-HRDC 在 1M 训练预算下不是仅平均分达标，而是三个随机种子全部超过 LunarLander-v3 的 200 分 solved 标准。

100k timesteps 对比实验中：

$$
R_{FDRE}=159.80\pm8.10,
$$

$$
R_{PPO}=-4.57\pm41.13,\qquad R_{LLM}=-58.01\pm51.44,
$$

$$
R_{w/o\ diag}=27.91\pm21.78,\qquad R_{w/o\ dyn}=19.16\pm38.15.
$$

相对提升为：

$$
\Delta_{FDRE-PPO}=164.37,\qquad \Delta_{FDRE-LLM}=217.81,
$$

$$
\Delta_{FDRE-diag}=131.89,\qquad \Delta_{FDRE-dyn}=140.64.
$$

## 8. 稳定性结果

奖励函数执行错误次数为：

$$
E=0.
$$

训练因奖励函数错误中断次数为：

$$
I=0.
$$

这表明当前实现中的奖励函数生成、注入、运行检查和候选选择流程可以稳定完成实验，不存在因为奖励函数运行错误导致实验中断的问题。

## 9. 交付结论

FDRE-HRDC 的创新点体现在闭环奖励自进化、训练反馈诊断、HRDC 分层奖励建模和阶段化动态权重四个方面。实验结果显示，该方法在 100k 设置下具有明显样本效率优势，在消融实验中能够体现诊断反馈和动态权重的贡献；在 1M 设置下，三个 seed 均超过 200 分 solved 标准，平均得分达到 $226.04\pm6.31$。因此，当前交付结果可以支撑“FDRE-HRDC 能够提升复杂控制任务中奖励函数设计质量、训练效率和最终策略稳定性”的论文式结论。
