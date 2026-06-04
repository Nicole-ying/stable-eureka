from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary_path = ROOT / "outputs" / "lunarlander_1m_stability" / "fdre_hrdc_stable_1m_summary.json"
    output_dir = ROOT / "deliverables" / "figures_lunarlander"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    scores = [float(v) for v in data["seed_scores"]]
    seeds = [str(record["seed"]) for record in data["records"]]
    mean_score = float(data["mean_score"])
    score_std = float(data["score_std"])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(seeds, scores, color=["#4C78A8", "#59A14F", "#F28E2B"], width=0.55)
    ax.axhline(200.0, color="#E15759", linewidth=1.6, linestyle="--", label="Solved threshold = 200")
    ax.errorbar(
        [len(seeds) - 0.05],
        [mean_score],
        yerr=[score_std],
        fmt="o",
        color="#222222",
        capsize=5,
        label=f"Mean = {mean_score:.1f} ± {score_std:.1f}",
    )
    ax.set_title("FDRE-HRDC 1M Evaluation on LunarLander-v3")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Original Environment Score")
    ax.set_ylim(min(scores + [200]) - 25, max(scores + [200]) + 25)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "fdre_1m_solved_threshold.png", bbox_inches="tight")
    plt.close(fig)
    print(output_dir / "fdre_1m_solved_threshold.png")


if __name__ == "__main__":
    main()
