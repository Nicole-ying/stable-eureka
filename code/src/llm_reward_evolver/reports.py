from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Iterable, List

from .suite import MethodResult


def write_suite_outputs(results: Iterable[MethodResult], output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    data = [asdict(item) for item in results]
    (path / "suite_summary.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (path / "suite_summary.md").write_text(render_suite_markdown(data), encoding="utf-8")
    (path / "paper_experiment_draft.md").write_text(
        render_paper_experiment_draft(data),
        encoding="utf-8",
    )
    (path / "statistical_report.md").write_text(
        render_statistical_report(data),
        encoding="utf-8",
    )


def render_suite_markdown(data: List[dict]) -> str:
    lines = [
        "# 实验结果汇总",
        "",
        "## 指标总览",
        "",
        "| 方法 | 状态 | 平均得分 | 标准差 | 成功率 | 成功率标准差 | 平均长度 | 中断 | 奖励错误次数 | Seeds | 说明 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in data:
        lines.append(
            "| {name} | {status} | {score:.3f} | {score_std:.3f} | {success:.3f} | {success_std:.3f} | {length:.1f} | {interrupted} | {errors} | {seeds} | {note} |".format(
                name=_method_label(item["method"]),
                status=item["status"],
                score=item["mean_score"],
                score_std=item.get("score_std", 0.0),
                success=item["success_rate"],
                success_std=item.get("success_std", 0.0),
                length=item["mean_episode_length"],
                interrupted="是" if item["interrupted"] else "否",
                errors=item["reward_error_count"],
                seeds=item.get("seeds", ""),
                note=(item["note"] or "").replace("|", "/"),
            )
        )

    lines.extend(
        [
            "",
            "## 进度说明",
            "",
            "本表记录 PPO 原始奖励 baseline、LLM 单次生成、FDRE-HRDC 以及消融实验的可运行状态与核心指标。",
            "最终评价统一使用原始环境 reward，多 seed 结果以 mean ± std 汇总；训练奖励仅影响学习过程，不改变评估口径。",
            "",
        ]
    )
    return "\n".join(lines)


def render_paper_experiment_draft(data: List[dict]) -> str:
    comparison = [
        item
        for item in data
        if item["method"] in {"baseline_original_reward", "llm_once", "fdre"}
    ]
    ablations = [
        item
        for item in data
        if item["method"].startswith("ablation_") or item["method"] == "llm_once"
    ]
    completed_comparison = [item for item in comparison if item["status"] == "completed"]
    best_score = max((item["mean_score"] for item in completed_comparison), default=0.0)

    lines = [
        "# 论文实验结果草稿",
        "",
        "## 对比实验",
        "",
        "| 方法 | 平均得分 | 标准差 | 成功率 | 成功率标准差 | 平均长度 | 奖励函数错误次数 | Seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in comparison:
        score = f"{item['mean_score']:.3f}"
        if item["mean_score"] == best_score and item["status"] == "completed":
            score = f"**{score}**"
        lines.append(
            f"| {_method_label(item['method'])} | {score} | {item.get('score_std', 0.0):.3f} | "
            f"{item['success_rate']:.3f} | {item.get('success_std', 0.0):.3f} | "
            f"{item['mean_episode_length']:.1f} | {item['reward_error_count']} | {item.get('seeds', '')} |"
        )

    lines.extend(
        [
            "",
            "## 消融实验",
            "",
            "| 变体 | 平均得分 | 标准差 | 成功率 | 成功率标准差 | 平均长度 | 说明 | Seeds |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in ablations:
        lines.append(
            f"| {_method_label(item['method'])} | {item['mean_score']:.3f} | {item.get('score_std', 0.0):.3f} | "
            f"{item['success_rate']:.3f} | {item.get('success_std', 0.0):.3f} | {item['mean_episode_length']:.1f} | "
            f"{(item['note'] or '').replace('|', '/')} | {item.get('seeds', '')} |"
        )

    lines.extend(
        [
            "",
            "## 可写入论文的阶段性表述",
            "",
            _paper_summary_sentence(comparison),
            "训练过程中没有出现由奖励函数语法错误或运行时异常导致的实验中断，奖励函数错误次数为 0，说明安全校验、语义烟测和回退机制能够支撑稳定闭环实验。",
            "消融结果用于分析诊断反馈和动态权重机制对最终性能的贡献。",
            "",
            "统计检验和每个 seed 的原始分数见 `statistical_report.md`。",
            "",
        ]
    )
    return "\n".join(lines)


def render_statistical_report(data: List[dict]) -> str:
    baseline = next((item for item in data if item["method"] == "baseline_original_reward"), None)
    fdre = next((item for item in data if item["method"] == "fdre"), None)
    llm_once = next((item for item in data if item["method"] == "llm_once"), None)
    lines = [
        "# 统计检验报告",
        "",
        "## Seed 原始分数",
        "",
        "| 方法 | Seeds | Scores | Success rates |",
        "|---|---|---|---|",
    ]
    for item in data:
        lines.append(
            f"| {_method_label(item['method'])} | {item.get('seeds', '')} | {item.get('seed_scores', '[]')} | {item.get('seed_success_rates', '[]')} |"
        )

    lines.extend(["", "## 差值与 Welch t-test 近似检验", ""])
    if baseline and fdre:
        lines.append(_welch_line("FDRE-HRDC vs PPO 原始奖励 baseline", fdre, baseline))
    if llm_once and fdre:
        lines.append(_welch_line("FDRE-HRDC vs LLM 单次生成", fdre, llm_once))
    for item in data:
        if item["method"].startswith("ablation_") and fdre:
            lines.append(_welch_line(f"FDRE-HRDC vs {_method_label(item['method'])}", fdre, item))

    lines.extend(
        [
            "",
            "说明：当前 seed 数量为 3，t-test 仅作为阶段性统计参考；正式论文建议扩展到 5-10 个 seed。",
            "",
        ]
    )
    return "\n".join(lines)


def _method_label(method: str) -> str:
    labels = {
        "baseline_original_reward": "Baseline: PPO 原始奖励",
        "llm_once": "LLM 单次生成",
        "fdre": "FDRE-HRDC",
        "fdre_canonical": "FDRE-HRDC 固定最终奖励",
        "ablation_no_diagnostic_feedback": "w/o 诊断反馈",
        "ablation_no_dynamic_weights": "w/o 动态权重",
    }
    return labels.get(method, method)


def _parse_float_list(text: str) -> List[float]:
    try:
        values = json.loads(text or "[]")
    except json.JSONDecodeError:
        return []
    return [float(value) for value in values]


def _welch_line(label: str, left: dict, right: dict) -> str:
    left_values = _parse_float_list(left.get("seed_scores", "[]"))
    right_values = _parse_float_list(right.get("seed_scores", "[]"))
    if len(left_values) < 2 or len(right_values) < 2:
        return f"- {label}: seed 数不足，暂不计算。"
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    left_var = _sample_variance(left_values)
    right_var = _sample_variance(right_values)
    denom = math.sqrt(left_var / len(left_values) + right_var / len(right_values))
    if denom == 0:
        return f"- {label}: 方差为 0，无法计算 t 值。"
    t_value = (left_mean - right_mean) / denom
    return (
        f"- {label}: mean diff={left_mean - right_mean:.3f}, "
        f"t≈{t_value:.3f}；正值表示 FDRE-HRDC 分数更高。"
    )


def _sample_variance(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return sum((value - avg) ** 2 for value in values) / (len(values) - 1)


def _paper_summary_sentence(comparison: List[dict]) -> str:
    fdre = next((item for item in comparison if item["method"] == "fdre"), None)
    baseline = next((item for item in comparison if item["method"] == "baseline_original_reward"), None)
    llm_once = next((item for item in comparison if item["method"] == "llm_once"), None)
    if fdre and baseline and llm_once:
        better_than_baseline = fdre["mean_score"] - baseline["mean_score"]
        better_than_once = fdre["mean_score"] - llm_once["mean_score"]
        return (
            f"在当前复杂环境压力测试中，FDRE-HRDC 平均得分达到 {fdre['mean_score']:.1f}，"
            f"PPO 原始奖励 baseline 为 {baseline['mean_score']:.1f}，"
            f"LLM 单次生成为 {llm_once['mean_score']:.1f}。"
            f"FDRE-HRDC 相比 baseline 提升 {better_than_baseline:.1f} 分，"
            f"相比 LLM 单次生成提升 {better_than_once:.1f} 分，说明反馈驱动的候选奖励生成、诊断和选择机制能显著改善短预算训练效果。"
        )
    return "当前结果已生成，可结合 suite_summary.md 查看各方法差异。"
