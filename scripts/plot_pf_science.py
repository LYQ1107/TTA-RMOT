#!/usr/bin/env python3
"""Static figures from all-six sealed metrics; no model or raw GT access."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ("PF_RANK", "PF_IDENTITY", "PF_HYBRID")


def render(publish):
    publish = Path(publish)
    evidence = json.loads((publish/"video_uncertainty.json").read_text())
    if evidence["status"] != "COMPLETE_6_METHODS":
        raise ValueError("scientific summary requires all six complete methods, not partial results")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios":[1, 1.7]})
    colors = ("#4477AA", "#228833", "#CC6677")
    for index, (method, color) in enumerate(zip(PRIMARY, colors)):
        comparison = evidence["comparisons"][method+"_vs_ZERO_UPDATE"]
        low, high = comparison["paired_video_cluster_CI95"]
        axes[0].hlines(index, low, high, color=color, linewidth=2)
        axes[0].scatter([comparison["delta_hota"]], [index], color=color, s=40, zorder=3)
        rows = comparison["per_video"]
        axes[1].plot(range(len(rows)), [r["delta_hota"] for r in rows], marker="o", markersize=3,
                     linewidth=1, label=method, color=color)
    axes[0].set_yticks(range(len(PRIMARY)))
    axes[0].set_yticklabels(PRIMARY)
    axes[0].axvline(0, linestyle="--", linewidth=.8, color="black")
    axes[0].set_xlabel("Pooled delta HOTA (percentage points)")
    axes[0].set_title("Paired 25-video bootstrap, 95% intervals")
    axes[0].grid(axis="x", alpha=.2)
    axes[1].axhline(0, linestyle="--", linewidth=.8, color="black")
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels([r["video"].removeprefix("dancetrack") for r in rows], rotation=90, fontsize=8)
    axes[1].set_xlabel("Video ID (all 17 expressions kept together)")
    axes[1].set_ylabel("Per-video delta HOTA (percentage points)")
    axes[1].set_title("Video heterogeneity; not additive contributions")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("V2 to Dance / 25 videos, 425 expressions / matched Frozen HOTA 8.2180")
    fig.tight_layout(rect=(0, 0, 1, .94))
    output = publish/"figures"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output/("scientific_summary."+suffix), dpi=170)
    plt.close(fig)
    print(json.dumps(dict(status="COMPLETE_6_METHODS", figure=str(output/"scientific_summary.png"))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publish-root", type=Path, default=ROOT/"results/positive_free/full")
    args = p.parse_args()
    render(args.publish_root)


if __name__ == "__main__": main()
