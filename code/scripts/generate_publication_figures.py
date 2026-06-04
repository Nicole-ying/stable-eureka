from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "deliverables" / "完整交付包"
FIG = PACK / "figures"
SUMMARY_1M = ROOT / "outputs" / "lunarlander_1m_stability" / "fdre_hrdc_stable_1m_summary.json"
SUMMARY_100K = ROOT / "outputs" / "lunarlander_paper" / "suite_summary.json"
HISTORY_ROOT = ROOT / "outputs" / "lunarlander_paper"

BLUE = "#3B6EA8"
ACCENT = "#2E8B57"
MUTED = "#B8C0CC"
MUTED_DARK = "#7D8794"
ORANGE = "#E6A15C"
RED = "#D97B7B"
PURPLE = "#A88AC2"
GRAY = "#9AA3AE"
DARK = "#252B33"
GRID = "#EEF1F5"
SPINE = "#C7CED8"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def setup() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for p in FIG.glob("*.png"):
        p.unlink()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 360,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.titleweight": "normal",
            "axes.edgecolor": SPINE,
            "axes.linewidth": 0.8,
            "grid.color": GRID,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.85,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(colors=DARK, width=0.8, length=3)
    ax.set_axisbelow(True)


def panel_label(ax, text: str) -> None:
    return None


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def label(method: str) -> str:
    return {
        "baseline_original_reward": "PPO\nbaseline",
        "llm_once": "LLM\nonce",
        "ablation_no_diagnostic_feedback": "w/o\ndiagnostic",
        "ablation_no_dynamic_weights": "w/o\ndynamic",
        "fdre": "FDRE-HRDC",
    }[method]


def lookup(summary):
    return {item["method"]: item for item in summary}


def parse_scores(value):
    return json.loads(value) if isinstance(value, str) else value


def history_mean(method: str, metric: str):
    paths = sorted((HISTORY_ROOT / method).glob("seed_*/history.json"))
    if not paths:
        return np.array([]), np.array([])
    rows = []
    for p in paths:
        records = load(p)
        rows.append([float(r[metric]) for r in records])
    width = max(len(r) for r in rows)
    grid = np.full((len(rows), width), np.nan)
    for i, row in enumerate(rows):
        grid[i, : len(row)] = row
    return np.arange(width), np.nanmean(grid, axis=0)


def plot_1m_seed_scores(stable):
    scores = np.array(stable["seed_scores"], dtype=float)
    seeds = [str(r["seed"]) for r in stable["records"]]
    fig, ax = plt.subplots(figsize=(3.6, 2.55))
    bars = ax.bar(seeds, scores, width=0.52, color=ACCENT, alpha=0.88)
    ax.axhline(200, color="#B94A48", lw=0.9, ls="--", label="threshold")
    ax.axhline(stable["mean_score"], color=DARK, lw=0.9, label="mean")
    ax.bar_label(bars, [f"{v:.1f}" for v in scores], fontsize=7, padding=2)
    panel_label(ax, "1M scores by seed")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Original score")
    ax.set_ylim(185, 242)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    clean(ax)
    save(fig, "01_1M_seed_scores.png")


def plot_1m_mean(stable):
    fig, ax = plt.subplots(figsize=(2.75, 2.55))
    bar = ax.bar(["FDRE-HRDC"], [stable["mean_score"]], yerr=[stable["score_std"]], width=0.36, color=BLUE, alpha=0.88, capsize=3, error_kw={"elinewidth": 0.9, "capthick": 0.9})
    ax.axhline(200, color="#B94A48", lw=0.9, ls="--")
    ax.bar_label(bar, [f"{stable['mean_score']:.1f} +/- {stable['score_std']:.1f}"], fontsize=7, padding=3)
    panel_label(ax, "1M mean score")
    ax.set_ylabel("Original score")
    ax.set_ylim(185, 240)
    clean(ax)
    save(fig, "02_1M_mean_score.png")


def plot_1m_success(stable):
    success = np.array(stable["seed_success_rates"], dtype=float)
    seeds = [str(r["seed"]) for r in stable["records"]]
    fig, ax = plt.subplots(figsize=(3.2, 2.45))
    ax.plot(seeds, success, marker="o", ms=4, lw=1.4, color=ACCENT)
    ax.fill_between(seeds, success, 0, color=ACCENT, alpha=0.07)
    for x, y in zip(seeds, success):
        ax.text(x, y + 0.035, f"{y:.2f}", ha="center", fontsize=7)
    panel_label(ax, "1M success rate")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.02)
    clean(ax)
    save(fig, "03_1M_success_rate.png")


def plot_100k_score(summary):
    data = lookup(summary)
    methods = ["baseline_original_reward", "llm_once", "ablation_no_diagnostic_feedback", "ablation_no_dynamic_weights", "fdre"]
    vals = np.array([data[m]["mean_score"] for m in methods], dtype=float)
    errs = np.array([data[m]["score_std"] for m in methods], dtype=float)
    colors = [MUTED, MUTED, "#D9A0A0", "#E9BE86", BLUE]
    fig, ax = plt.subplots(figsize=(4.2, 2.85))
    x = np.arange(len(methods))
    bars = ax.bar(x, vals, yerr=errs, capsize=2.2, color=colors, alpha=0.9, width=0.55, error_kw={"elinewidth": 0.85, "capthick": 0.85})
    ax.axhline(0, color=DARK, lw=0.75)
    ax.bar_label(bars, [f"{v:.1f}" for v in vals], fontsize=6.4, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in methods])
    panel_label(ax, "100k score comparison")
    ax.set_ylabel("Original score")
    clean(ax)
    save(fig, "04_100k_score_comparison.png")


def plot_100k_success(summary):
    data = lookup(summary)
    methods = ["baseline_original_reward", "llm_once", "ablation_no_diagnostic_feedback", "ablation_no_dynamic_weights", "fdre"]
    vals = np.array([data[m]["success_rate"] for m in methods], dtype=float)
    colors = [MUTED, MUTED, "#D9A0A0", "#E9BE86", BLUE]
    fig, ax = plt.subplots(figsize=(3.9, 2.65))
    y = np.arange(len(methods))
    bars = ax.barh(y, vals, color=colors, alpha=0.9, height=0.48)
    ax.set_yticks(y)
    ax.set_yticklabels([label(m).replace("\n", " ") for m in methods])
    ax.bar_label(bars, [f"{v:.2f}" for v in vals], fontsize=6.8, padding=3)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Success rate")
    panel_label(ax, "100k success rate")
    clean(ax)
    save(fig, "05_100k_success_rate.png")


def plot_ablation_gain(summary):
    data = lookup(summary)
    fdre = float(data["fdre"]["mean_score"])
    refs = [
        ("PPO baseline", float(data["baseline_original_reward"]["mean_score"]), MUTED_DARK),
        ("LLM once", float(data["llm_once"]["mean_score"]), MUTED_DARK),
        ("w/o diagnostic", float(data["ablation_no_diagnostic_feedback"]["mean_score"]), "#D9A0A0"),
        ("w/o dynamic", float(data["ablation_no_dynamic_weights"]["mean_score"]), "#E9BE86"),
    ]
    gains = [fdre - r[1] for r in refs]
    fig, ax = plt.subplots(figsize=(3.9, 2.55))
    bars = ax.barh(np.arange(len(refs)), gains, color=[r[2] for r in refs], alpha=0.9, height=0.48)
    ax.set_yticks(np.arange(len(refs)))
    ax.set_yticklabels([r[0] for r in refs])
    ax.bar_label(bars, [f"+{v:.1f}" for v in gains], fontsize=6.8, padding=3)
    ax.set_xlim(0, max(gains) * 1.14)
    ax.set_xlabel("Score gain")
    panel_label(ax, "Ablation gain")
    clean(ax)
    save(fig, "06_ablation_score_gain.png")


def plot_seed_distribution(summary, stable):
    data = lookup(summary)
    methods = ["baseline_original_reward", "llm_once", "ablation_no_diagnostic_feedback", "ablation_no_dynamic_weights", "fdre"]
    values = [parse_scores(data[m]["seed_scores"]) for m in methods] + [stable["seed_scores"]]
    names = [label(m).replace("\n", " ") for m in methods] + ["FDRE 1M"]
    colors = [MUTED, MUTED, "#D9A0A0", "#E9BE86", BLUE, ACCENT]
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    parts = ax.violinplot(values, showmeans=False, showmedians=False, widths=0.65)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.16)
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color(SPINE)
    for i, vals in enumerate(values, start=1):
        vals = np.array(vals, dtype=float)
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.055, 0.055, len(vals)), vals, s=18, color=colors[i - 1], edgecolor="white", lw=0.4, zorder=3)
        ax.plot([i - 0.13, i + 0.13], [vals.mean(), vals.mean()], color=DARK, lw=1.2)
    ax.axhline(200, color="#B94A48", ls="--", lw=0.9)
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("Score")
    panel_label(ax, "Seed-level distribution")
    clean(ax)
    save(fig, "07_seed_level_distribution.png")


def plot_reward_iteration_score(summary):
    specs = [("fdre", "FDRE-HRDC", BLUE), ("ablation_no_diagnostic_feedback", "w/o diagnostic", "#D9A0A0"), ("ablation_no_dynamic_weights", "w/o dynamic", "#E9BE86"), ("llm_once", "LLM once", MUTED_DARK)]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for method, name, color in specs:
        xs, ys = history_mean(method, "score")
        if len(xs):
            lw = 1.6 if method == "fdre" else 1.05
            alpha = 0.95 if method == "fdre" else 0.75
            ax.plot(xs, ys, marker="o", ms=3.6, lw=lw, color=color, alpha=alpha, label=name)
    final = float(lookup(summary)["fdre"]["mean_score"])
    ax.scatter([3], [final], color=ACCENT, s=38, zorder=4, label="selected")
    ax.text(3, final + 20, f"{final:.1f}", ha="center", fontsize=7, color=ACCENT)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0", "1", "2", "selected"])
    ax.set_xlabel("Reward generation iteration")
    ax.set_ylabel("Mean score")
    panel_label(ax, "Reward evolution score")
    ax.legend(frameon=False, loc="lower right")
    clean(ax)
    save(fig, "08_reward_iteration_score.png")


def plot_reward_iteration_success(summary):
    specs = [("fdre", "FDRE-HRDC", BLUE), ("ablation_no_diagnostic_feedback", "w/o diagnostic", "#D9A0A0"), ("ablation_no_dynamic_weights", "w/o dynamic", "#E9BE86"), ("llm_once", "LLM once", MUTED_DARK)]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for method, name, color in specs:
        xs, ys = history_mean(method, "success_rate")
        if len(xs):
            lw = 1.6 if method == "fdre" else 1.05
            alpha = 0.95 if method == "fdre" else 0.75
            ax.plot(xs, ys, marker="o", ms=3.6, lw=lw, color=color, alpha=alpha, label=name)
    final = float(lookup(summary)["fdre"]["success_rate"])
    ax.scatter([3], [final], color=ACCENT, s=38, zorder=4, label="selected")
    ax.text(3, final + 0.035, f"{final:.2f}", ha="center", fontsize=7, color=ACCENT)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0", "1", "2", "selected"])
    ax.set_ylim(-0.02, 0.92)
    ax.set_xlabel("Reward generation iteration")
    ax.set_ylabel("Mean success rate")
    panel_label(ax, "Reward evolution success")
    ax.legend(frameon=False, loc="upper left")
    clean(ax)
    save(fig, "09_reward_iteration_success.png")


def plot_stability(summary, stable):
    data = lookup(summary)
    rows = [
        ("PPO baseline", data["baseline_original_reward"]["interrupted"], data["baseline_original_reward"]["reward_error_count"]),
        ("LLM once", data["llm_once"]["interrupted"], data["llm_once"]["reward_error_count"]),
        ("FDRE 100k", data["fdre"]["interrupted"], data["fdre"]["reward_error_count"]),
        ("FDRE 1M", stable["interrupted"], stable["reward_error_count"]),
    ]
    fig, ax = plt.subplots(figsize=(4.2, 2.45))
    ax.axis("off")
    panel_label(ax, "Execution stability")
    xs = [0.20, 0.54, 0.75, 0.91]
    headers = ["Experiment", "Stop", "Error", "Status"]
    for x, h in zip(xs, headers):
        ax.text(x, 0.88, h, transform=ax.transAxes, ha="center", fontsize=7.5, color=DARK, weight="bold")
    for i, (name, interrupted, errors) in enumerate(rows):
        y = 0.72 - 0.17 * i
        ax.add_patch(plt.Rectangle((0.04, y - 0.045), 0.92, 0.09, transform=ax.transAxes, color="#F7F9FC", ec=GRID, lw=0.6))
        ok = (not interrupted) and int(errors) == 0
        ax.text(xs[0], y, name, transform=ax.transAxes, ha="center", va="center", fontsize=7.2)
        ax.text(xs[1], y, str(int(bool(interrupted))), transform=ax.transAxes, ha="center", va="center", fontsize=7.2, color=ACCENT if not interrupted else RED)
        ax.text(xs[2], y, str(int(errors)), transform=ax.transAxes, ha="center", va="center", fontsize=7.2, color=ACCENT if int(errors) == 0 else RED)
        ax.text(xs[3], y, "PASS" if ok else "CHECK", transform=ax.transAxes, ha="center", va="center", fontsize=7.2, color=ACCENT if ok else RED)
    save(fig, "10_stability_matrix.png")


def plot_evidence_chain(summary, stable):
    data = lookup(summary)
    labels = ["PPO\nbaseline", "LLM\nonce", "FDRE\n100k", "FDRE\n1M"]
    scores = [float(data["baseline_original_reward"]["mean_score"]), float(data["llm_once"]["mean_score"]), float(data["fdre"]["mean_score"]), float(stable["mean_score"])]
    colors = [MUTED, MUTED_DARK, BLUE, ACCENT]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.arange(len(scores))
    ax.plot(x, scores, color=SPINE, lw=1.1)
    ax.scatter(x, scores, s=46, color=colors, edgecolor="white", lw=0.6, zorder=3)
    for xi, yi, c in zip(x, scores, colors):
        ax.text(xi, yi + 16, f"{yi:.1f}", ha="center", fontsize=7, color=c)
    ax.axhline(200, color="#B94A48", ls="--", lw=0.9, label="threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Original score")
    panel_label(ax, "Evidence chain")
    ax.set_ylim(min(scores) - 40, max(scores) + 42)
    ax.legend(frameon=False, loc="upper left")
    clean(ax)
    save(fig, "11_evidence_chain.png")


def plot_method():
    fig, ax = plt.subplots(figsize=(4.8, 1.9))
    ax.axis("off")
    nodes = [("LLM reward", 0.11, BLUE), ("Safety check", 0.31, ACCENT), ("PPO feedback", 0.51, ORANGE), ("Dynamic HRDC", 0.71, PURPLE), ("Selection", 0.90, DARK)]
    for text, x, color in nodes:
        ax.add_patch(plt.Rectangle((x - 0.07, 0.48), 0.14, 0.20, transform=ax.transAxes, facecolor=color, edgecolor=color, alpha=0.92))
        ax.text(x, 0.58, text, transform=ax.transAxes, ha="center", va="center", fontsize=6.8, color="white")
    for i in range(len(nodes) - 1):
        ax.annotate("", xy=(nodes[i + 1][1] - 0.078, 0.58), xytext=(nodes[i][1] + 0.078, 0.58), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.0, color=SPINE))
    ax.annotate("", xy=(0.16, 0.40), xytext=(0.84, 0.40), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.25", lw=1.0, color=SPINE))
    ax.text(0.5, 0.18, "Closed-loop reward self-evolution", transform=ax.transAxes, ha="center", fontsize=7.5, color=DARK)
    panel_label(ax, "FDRE-HRDC workflow")
    save(fig, "12_method_overview.png")


def main():
    setup()
    stable = load(SUMMARY_1M)
    summary = load(SUMMARY_100K)
    plot_1m_seed_scores(stable)
    plot_1m_mean(stable)
    plot_1m_success(stable)
    plot_100k_score(summary)
    plot_100k_success(summary)
    plot_ablation_gain(summary)
    plot_seed_distribution(summary, stable)
    plot_reward_iteration_score(summary)
    plot_reward_iteration_success(summary)
    plot_stability(summary, stable)
    plot_evidence_chain(summary, stable)
    plot_method()
    print(FIG)


if __name__ == "__main__":
    main()
