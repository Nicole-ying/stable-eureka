from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "deliverables" / "完整交付包"
FIG = PACK / "figures"
DATA = PACK / "data"
CODE = PACK / "code"

SUMMARY_1M = ROOT / "outputs" / "lunarlander_1m_stability" / "fdre_hrdc_stable_1m_summary.json"
SUMMARY_100K = ROOT / "outputs" / "lunarlander_paper" / "suite_summary.json"
HISTORY_ROOT = ROOT / "outputs" / "lunarlander_paper"

BLUE = "#2f5f9f"
ORANGE = "#d9822b"
GREEN = "#2e8b57"
RED = "#c94c4c"
PURPLE = "#7b5ea7"
GRAY = "#59616f"
DARK = "#1f2937"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def setup() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    CODE.mkdir(parents=True, exist_ok=True)
    for old_png in FIG.glob("*.png"):
        old_png.unlink()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#d0d7e2",
            "grid.color": "#e7ecf3",
            "grid.linewidth": 0.8,
        }
    )


def method_label(method: str) -> str:
    return {
        "baseline_original_reward": "PPO Baseline",
        "llm_once": "LLM Once",
        "fdre": "FDRE-HRDC",
        "fdre_canonical": "FDRE-HRDC",
        "ablation_no_diagnostic_feedback": "w/o Diagnostic",
        "ablation_no_dynamic_weights": "w/o Dynamic",
    }.get(method, method)


def lookup(summary: list[dict]) -> dict[str, dict]:
    return {item["method"]: item for item in summary}


def parse_list(value) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str):
        return [float(v) for v in json.loads(value)]
    return []


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_histories(method: str) -> dict[int, list[dict]]:
    histories: dict[int, list[dict]] = {}
    for path in sorted((HISTORY_ROOT / method).glob("seed_*/history.json")):
        try:
            seed = int(path.parent.name.split("_")[-1])
        except ValueError:
            continue
        histories[seed] = load_json(path)
    return histories


def history_mean(method: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    histories = load_histories(method)
    if not histories:
        return np.array([]), np.array([])
    max_iter = max(len(records) for records in histories.values())
    grid = np.full((len(histories), max_iter), np.nan)
    for row, records in enumerate(histories.values()):
        for rec in records:
            grid[row, int(rec["iteration"])] = float(rec[metric])
    return np.arange(max_iter), np.nanmean(grid, axis=0)


def plot_1m_scores(stable: dict) -> None:
    scores = np.array(stable["seed_scores"], dtype=float)
    seeds = [str(record["seed"]) for record in stable["records"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor="white")
    bars = ax.bar(seeds, scores, color=GREEN, width=0.58)
    ax.axhline(200, color=RED, linestyle="--", linewidth=1.5, label="Solved threshold")
    ax.axhline(stable["mean_score"], color=DARK, linewidth=1.2, label=f"Mean = {stable['mean_score']:.1f}")
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in scores], padding=3, fontsize=10)
    ax.set_title("1M Evaluation Scores across Seeds")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Original environment score")
    ax.set_ylim(185, max(scores) + 18)
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)
    save(fig, "01_1M_seed_scores.png")


def plot_1m_mean(stable: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2), facecolor="white")
    bar = ax.bar(["FDRE-HRDC"], [stable["mean_score"]], yerr=[stable["score_std"]], capsize=6, color=BLUE, width=0.42)
    ax.axhline(200, color=RED, linestyle="--", linewidth=1.5, label="Solved threshold")
    ax.bar_label(bar, labels=[f"{stable['mean_score']:.1f} ± {stable['score_std']:.1f}"], padding=5, fontsize=10)
    ax.set_title("1M Mean Performance")
    ax.set_ylabel("Original environment score")
    ax.set_ylim(185, 240)
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)
    save(fig, "02_1M_mean_score.png")


def plot_1m_success(stable: dict) -> None:
    success = np.array(stable["seed_success_rates"], dtype=float)
    seeds = [str(record["seed"]) for record in stable["records"]]
    fig, ax = plt.subplots(figsize=(5.4, 4.2), facecolor="white")
    ax.plot(seeds, success, color=GREEN, marker="o", linewidth=2.4)
    ax.fill_between(seeds, success, 0, color=GREEN, alpha=0.12)
    for x, y in zip(seeds, success):
        ax.text(x, y + 0.035, f"{y:.2f}", ha="center", fontsize=10)
    ax.set_title("1M Success Rate across Seeds")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.0)
    clean_axis(ax)
    save(fig, "03_1M_success_rate.png")


def plot_100k_score(summary: list[dict]) -> None:
    data = lookup(summary)
    methods = [
        "baseline_original_reward",
        "llm_once",
        "ablation_no_diagnostic_feedback",
        "ablation_no_dynamic_weights",
        "fdre",
    ]
    means = np.array([data[m]["mean_score"] for m in methods], dtype=float)
    stds = np.array([data[m]["score_std"] for m in methods], dtype=float)
    colors = [GRAY, PURPLE, RED, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor="white")
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, width=0.62)
    ax.axhline(0, color=DARK, linewidth=0.9)
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in means], padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in methods], rotation=15, ha="right")
    ax.set_title("100k Score Comparison")
    ax.set_ylabel("Original environment score")
    clean_axis(ax)
    save(fig, "04_100k_score_comparison.png")


def plot_100k_success(summary: list[dict]) -> None:
    data = lookup(summary)
    methods = [
        "baseline_original_reward",
        "llm_once",
        "ablation_no_diagnostic_feedback",
        "ablation_no_dynamic_weights",
        "fdre",
    ]
    success = np.array([data[m]["success_rate"] for m in methods], dtype=float)
    colors = [GRAY, PURPLE, RED, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(6.8, 4.3), facecolor="white")
    y = np.arange(len(methods))
    bars = ax.barh(y, success, color=colors, height=0.58)
    ax.set_yticks(y)
    ax.set_yticklabels([method_label(m) for m in methods])
    ax.set_xlim(0, 1.0)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in success], padding=4, fontsize=10)
    ax.set_title("100k Success Rate Comparison")
    ax.set_xlabel("Success rate")
    clean_axis(ax)
    save(fig, "05_100k_success_rate.png")


def plot_ablation_gain(summary: list[dict]) -> None:
    data = lookup(summary)
    fdre = float(data["fdre"]["mean_score"])
    refs = [
        ("PPO Baseline", float(data["baseline_original_reward"]["mean_score"]), GRAY),
        ("LLM Once", float(data["llm_once"]["mean_score"]), PURPLE),
        ("w/o Diagnostic", float(data["ablation_no_diagnostic_feedback"]["mean_score"]), RED),
        ("w/o Dynamic", float(data["ablation_no_dynamic_weights"]["mean_score"]), ORANGE),
    ]
    gains = [fdre - score for _, score, _ in refs]
    fig, ax = plt.subplots(figsize=(6.8, 4.3), facecolor="white")
    bars = ax.barh(np.arange(len(refs)), gains, color=[c for _, _, c in refs], height=0.58)
    ax.set_yticks(np.arange(len(refs)))
    ax.set_yticklabels([name for name, _, _ in refs])
    ax.bar_label(bars, labels=[f"+{v:.1f}" for v in gains], padding=4, fontsize=10)
    ax.set_xlabel("FDRE-HRDC score gain at 100k")
    ax.set_title("Ablation Gain of FDRE-HRDC")
    ax.set_xlim(0, max(gains) * 1.16)
    clean_axis(ax)
    save(fig, "06_ablation_score_gain.png")


def plot_seed_distribution(summary: list[dict], stable: dict) -> None:
    data = lookup(summary)
    methods = [
        "baseline_original_reward",
        "llm_once",
        "ablation_no_diagnostic_feedback",
        "ablation_no_dynamic_weights",
        "fdre",
    ]
    values = [parse_list(data[m]["seed_scores"]) for m in methods] + [stable["seed_scores"]]
    names = [method_label(m) for m in methods] + ["FDRE-HRDC 1M"]
    colors = [GRAY, PURPLE, RED, ORANGE, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor="white")
    parts = ax.violinplot(values, showmeans=False, showmedians=False, widths=0.7)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color("#8a94a6")
    for i, vals in enumerate(values, start=1):
        jitter = np.linspace(-0.07, 0.07, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=colors[i - 1], s=42, zorder=3, edgecolor="white", linewidth=0.7)
        ax.plot([i - 0.16, i + 0.16], [np.mean(vals), np.mean(vals)], color=DARK, linewidth=2.0)
    ax.axhline(200, color=RED, linestyle="--", linewidth=1.4, label="Solved threshold")
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("Score by seed")
    ax.set_title("Seed-Level Score Distribution")
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)
    save(fig, "07_seed_level_distribution.png")


def plot_reward_iteration_score(summary: list[dict]) -> None:
    specs = [
        ("fdre", "FDRE-HRDC", BLUE),
        ("ablation_no_diagnostic_feedback", "w/o Diagnostic", RED),
        ("ablation_no_dynamic_weights", "w/o Dynamic", ORANGE),
        ("llm_once", "LLM Once", PURPLE),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.6), facecolor="white")
    for method, name, color in specs:
        xs, ys = history_mean(method, "score")
        if len(xs):
            ax.plot(xs, ys, marker="o", linewidth=2.2, color=color, label=name)
    final = float(lookup(summary)["fdre"]["mean_score"])
    ax.scatter([3], [final], color=GREEN, s=80, zorder=4, label="Selected final reward")
    ax.text(3, final + 20, f"{final:.1f}", ha="center", color=GREEN, fontsize=10)
    ax.axvspan(2.72, 3.28, color=GREEN, alpha=0.06)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["iter 0", "iter 1", "iter 2", "selected"])
    ax.set_title("Reward Evolution Score Trajectory")
    ax.set_xlabel("Reward generation iteration")
    ax.set_ylabel("Mean score across seeds")
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)
    save(fig, "08_reward_iteration_score.png")


def plot_reward_iteration_success(summary: list[dict]) -> None:
    specs = [
        ("fdre", "FDRE-HRDC", BLUE),
        ("ablation_no_diagnostic_feedback", "w/o Diagnostic", RED),
        ("ablation_no_dynamic_weights", "w/o Dynamic", ORANGE),
        ("llm_once", "LLM Once", PURPLE),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.6), facecolor="white")
    for method, name, color in specs:
        xs, ys = history_mean(method, "success_rate")
        if len(xs):
            ax.plot(xs, ys, marker="o", linewidth=2.2, color=color, label=name)
    final = float(lookup(summary)["fdre"]["success_rate"])
    ax.scatter([3], [final], color=GREEN, s=80, zorder=4, label="Selected final reward")
    ax.text(3, final + 0.035, f"{final:.2f}", ha="center", color=GREEN, fontsize=10)
    ax.axvspan(2.72, 3.28, color=GREEN, alpha=0.06)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["iter 0", "iter 1", "iter 2", "selected"])
    ax.set_ylim(-0.02, 0.92)
    ax.set_title("Reward Evolution Success-Rate Trajectory")
    ax.set_xlabel("Reward generation iteration")
    ax.set_ylabel("Mean success rate")
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)
    save(fig, "09_reward_iteration_success.png")


def plot_stability_matrix(summary: list[dict], stable: dict) -> None:
    data = lookup(summary)
    rows = [
        ("PPO baseline 100k", data["baseline_original_reward"]["interrupted"], data["baseline_original_reward"]["reward_error_count"]),
        ("LLM once 100k", data["llm_once"]["interrupted"], data["llm_once"]["reward_error_count"]),
        ("FDRE-HRDC 100k", data["fdre"]["interrupted"], data["fdre"]["reward_error_count"]),
        ("FDRE-HRDC 1M", stable["interrupted"], stable["reward_error_count"]),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.9), facecolor="white")
    ax.axis("off")
    ax.set_title("Execution Stability Check", fontsize=13, fontweight="bold", pad=12)
    headers = ["Experiment", "Interrupted", "Reward errors", "Status"]
    xs = [0.12, 0.50, 0.70, 0.88]
    for x, h in zip(xs, headers):
        ax.text(x, 0.88, h, transform=ax.transAxes, fontsize=10, fontweight="bold", color=DARK, ha="center")
    for i, (name, interrupted, errors) in enumerate(rows):
        y = 0.72 - i * 0.17
        rect = FancyBboxPatch(
            (0.025, y - 0.052),
            0.95,
            0.10,
            boxstyle="round,pad=0.01,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor="#f8fafc" if i % 2 == 0 else "#ffffff",
            edgecolor="#e1e7ef",
            linewidth=1.0,
        )
        ax.add_patch(rect)
        ok = (not interrupted) and int(errors) == 0
        ax.text(xs[0], y, name, transform=ax.transAxes, ha="center", va="center", fontsize=9.5, color=DARK)
        ax.text(xs[1], y, str(int(bool(interrupted))), transform=ax.transAxes, ha="center", va="center", fontsize=10, color=GREEN if not interrupted else RED, fontweight="bold")
        ax.text(xs[2], y, str(int(errors)), transform=ax.transAxes, ha="center", va="center", fontsize=10, color=GREEN if int(errors) == 0 else RED, fontweight="bold")
        ax.text(xs[3], y, "PASS" if ok else "CHECK", transform=ax.transAxes, ha="center", va="center", fontsize=10, color=GREEN if ok else RED, fontweight="bold")
    save(fig, "10_stability_matrix.png")


def plot_evidence_chain(summary: list[dict], stable: dict) -> None:
    data = lookup(summary)
    points = [
        ("PPO\nBaseline", float(data["baseline_original_reward"]["mean_score"]), GRAY),
        ("LLM\nOnce", float(data["llm_once"]["mean_score"]), PURPLE),
        ("FDRE\n100k", float(data["fdre"]["mean_score"]), BLUE),
        ("FDRE\n1M", float(stable["mean_score"]), GREEN),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.5), facecolor="white")
    xs = np.arange(len(points))
    scores = [p[1] for p in points]
    ax.plot(xs, scores, color=DARK, linewidth=1.2, alpha=0.5)
    for x, (_, score, color) in zip(xs, points):
        ax.scatter([x], [score], s=150, color=color, edgecolor="white", linewidth=1.3, zorder=3)
        ax.text(x, score + 17, f"{score:.1f}", ha="center", fontsize=10, color=color, fontweight="bold")
    ax.axhline(200, color=RED, linestyle="--", linewidth=1.4, label="Solved threshold")
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in points])
    ax.set_ylabel("Original environment score")
    ax.set_title("Evidence Chain to Solved Policy")
    ax.set_ylim(min(scores) - 42, max(scores) + 42)
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax)
    save(fig, "11_evidence_chain.png")


def plot_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.4), facecolor="white")
    ax.axis("off")
    nodes = [
        ("LLM reward\nproposal", 0.10, BLUE),
        ("Safety\ncheck", 0.30, GREEN),
        ("PPO feedback\ndiagnosis", 0.50, ORANGE),
        ("Dynamic HRDC\nweights", 0.70, PURPLE),
        ("Reward\nselection", 0.90, DARK),
    ]
    y = 0.55
    for text, x, color in nodes:
        rect = FancyBboxPatch(
            (x - 0.075, y - 0.13),
            0.15,
            0.26,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            alpha=0.96,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", transform=ax.transAxes, fontsize=9, color="white", fontweight="bold")
    for i in range(len(nodes) - 1):
        ax.annotate(
            "",
            xy=(nodes[i + 1][1] - 0.085, y),
            xytext=(nodes[i][1] + 0.085, y),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", color="#7d8797", lw=1.6),
        )
    ax.annotate(
        "",
        xy=(0.15, 0.31),
        xytext=(0.85, 0.31),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.25", color="#7d8797", lw=1.6),
    )
    ax.text(0.50, 0.10, "Closed-loop reward self-evolution", ha="center", transform=ax.transAxes, fontsize=10, color=GRAY)
    ax.set_title("FDRE-HRDC Method Overview", fontsize=13, fontweight="bold", pad=10)
    save(fig, "12_method_overview.png")


def copy_code_snapshot() -> None:
    if CODE.exists():
        shutil.rmtree(CODE)
    CODE.mkdir(parents=True, exist_ok=True)
    for folder in ["src", "scripts", "configs", "tests"]:
        src = ROOT / folder
        if src.exists():
            shutil.copytree(src, CODE / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    for filename in ["README.md", "requirements.txt", "pyproject.toml", "config.example.json"]:
        src = ROOT / filename
        if src.exists():
            shutil.copy2(src, CODE / filename)

    text = """# 代码交付说明

本目录为 FDRE-HRDC 实验代码快照，包含核心源码、实验脚本、配置文件、测试文件和依赖声明。

## 常用命令

```powershell
python scripts/run_experiment.py --config configs/lunarlander_paper.json --llm-provider ollama --llm-model qwen2.5-coder:7b --suite
python scripts/generate_final_delivery_pack.py
python -m pytest tests
```
"""
    (CODE / "code_delivery.md").write_text(text, encoding="utf-8")


def write_documents(stable: dict, summary: list[dict]) -> None:
    data = lookup(summary)
    (DATA / "fdre_hrdc_stable_1m_summary.json").write_text(json.dumps(stable, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATA / "lunarlander_100k_suite_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# FDRE-HRDC Delivery Package

This package contains paper-ready single-panel figures, source data, source code, and the formal method document.

## Key Results

FDRE-HRDC solves LunarLander-v3 under 1M timesteps. All three seeds exceed 200, with mean score `{stable['mean_score']:.2f} +/- {stable['score_std']:.2f}` and mean success rate `{stable['success_rate']:.2f} +/- {stable['success_std']:.2f}`. At 100k timesteps, FDRE-HRDC reaches `{data['fdre']['mean_score']:.2f} +/- {data['fdre']['score_std']:.2f}`, outperforming PPO baseline, LLM once, and both ablation variants.

## Contents

- `figures/`: paper-ready single-panel figures.
- `data/`: JSON summaries.
- `code/`: complete source-code snapshot.
- `FDRE-HRDC_formula_delivery.md`: formal method and result description.
- `FDRE-HRDC???????.md`: Chinese formal method and result description.
"""
    (PACK / "README.md").write_text(readme, encoding="utf-8")

    figures = "\n".join(f"- `figures/{p.name}`" for p in sorted(FIG.glob("*.png")))
    index = f"""# Delivery Index

## Figures

{figures}

## Data

- `data/fdre_hrdc_stable_1m_summary.json`
- `data/lunarlander_100k_suite_summary.json`

## Code

- `code/src/llm_reward_evolver/`
- `code/scripts/`
- `code/configs/`
- `code/tests/`
- `code/requirements.txt`
- `code/pyproject.toml`
- `code/code_delivery.md`
"""
    (PACK / "delivery_index.md").write_text(index, encoding="utf-8")

def main() -> None:
    setup()
    stable = load_json(SUMMARY_1M)
    summary = load_json(SUMMARY_100K)

    plot_1m_scores(stable)
    plot_1m_mean(stable)
    plot_1m_success(stable)
    plot_100k_score(summary)
    plot_100k_success(summary)
    plot_ablation_gain(summary)
    plot_seed_distribution(summary, stable)
    plot_reward_iteration_score(summary)
    plot_reward_iteration_success(summary)
    plot_stability_matrix(summary, stable)
    plot_evidence_chain(summary, stable)
    plot_method_overview()
    copy_code_snapshot()
    write_documents(stable, summary)
    print(PACK)


if __name__ == "__main__":
    main()
