"""
Generate paper-ready plots from multi_run_results.json.

Outputs are written to:
    research_visualizations/paper_graphs/

Run:
    python generate_paper_graphs.py
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(PROJECT_DIR, "multi_run_results.json")
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "research_visualizations", "paper_graphs")

IID_NAME = "IID_Distribution"
DP_NAME = "With_Differential_Privacy"

MODALITY_LABELS = {
    "facial": "Facial",
    "behavioral": "Behavioral",
    "fusion": "Fusion",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "client_accuracy_variance": "Client variance",
}

IID_COLOR = "#2563EB"
DP_COLOR = "#DC2626"
FACIAL_COLOR = "#0F766E"
BEHAVIORAL_COLOR = "#7C3AED"
FUSION_COLOR = "#D97706"
GRID_COLOR = "#D1D5DB"
TEXT_COLOR = "#111827"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.45,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_results(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fig(fig: plt.Figure, out_dir: str, filename: str) -> None:
    path = os.path.join(out_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def _aggregate(results: Dict, modality: str, config: str, metric: str) -> Tuple[float, float]:
    stats = results["aggregate"][modality][config][metric]
    return float(stats["mean"]), float(stats["std"])


def plot_fusion_accuracy(results: Dict, out_dir: str) -> None:
    """Plot the fusion model's accuracy only."""
    x = np.arange(1)
    width = 0.34

    iid_means = []
    iid_stds = []
    dp_means = []
    dp_stds = []
    mean, std = _aggregate(results, "fusion", IID_NAME, "accuracy")
    iid_means.append(mean)
    iid_stds.append(std)
    mean, std = _aggregate(results, "fusion", DP_NAME, "accuracy")
    dp_means.append(mean)
    dp_stds.append(std)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars_iid = ax.bar(
        x - width / 2,
        iid_means,
        width,
        yerr=iid_stds,
        capsize=5,
        label="IID",
        color=IID_COLOR,
        alpha=0.9,
    )
    bars_dp = ax.bar(
        x + width / 2,
        dp_means,
        width,
        yerr=dp_stds,
        capsize=5,
        label="Non-IID + DP",
        color=DP_COLOR,
        alpha=0.9,
    )

    ax.set_title("Fusion Accuracy Across Runs")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Fusion"])
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.bar_label(bars_iid, fmt="%.1f", padding=3, fontsize=9)
    ax.bar_label(bars_dp, fmt="%.1f", padding=3, fontsize=9)

    save_fig(fig, out_dir, "fig1_fusion_accuracy_errorbars.png")


def _per_seed_values(results: Dict, modality: str, config: str, metric: str) -> List[float]:
    return [
        float(run[metric])
        for run in results["per_run_results"][modality][config]
    ]


def plot_fusion_per_seed(results: Dict, out_dir: str) -> None:
    seeds = [int(seed) for seed in results["seeds"]]
    iid = _per_seed_values(results, "fusion", IID_NAME, "accuracy")
    dp = _per_seed_values(results, "fusion", DP_NAME, "accuracy")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(seeds, iid, marker="o", linewidth=2.4, color=IID_COLOR, label="Fusion IID")
    ax.plot(seeds, dp, marker="s", linewidth=2.4, color=DP_COLOR, label="Fusion Non-IID + DP")

    best_iid_idx = int(np.argmax(iid))
    best_dp_idx = int(np.argmax(dp))
    ax.scatter([seeds[best_iid_idx]], [iid[best_iid_idx]], s=110, color=IID_COLOR, edgecolor="black", zorder=5)
    ax.scatter([seeds[best_dp_idx]], [dp[best_dp_idx]], s=110, color=DP_COLOR, edgecolor="black", zorder=5)
    ax.annotate(
        f"Best IID: {iid[best_iid_idx]:.2f}%",
        (seeds[best_iid_idx], iid[best_iid_idx]),
        xytext=(-15, 18),
        textcoords="offset points",
        color=IID_COLOR,
        fontsize=9,
        fontweight="bold",
    )
    ax.annotate(
        f"Best DP: {dp[best_dp_idx]:.2f}%",
        (seeds[best_dp_idx], dp[best_dp_idx]),
        xytext=(-15, -24),
        textcoords="offset points",
        color=DP_COLOR,
        fontsize=9,
        fontweight="bold",
    )

    ax.set_title("Fusion Accuracy Across Random Seeds")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(seeds)
    ax.set_ylim(60, 100)
    ax.legend(frameon=False)

    save_fig(fig, out_dir, "fig2_fusion_accuracy_by_seed.png")


def plot_fusion_client_variance(results: Dict, out_dir: str) -> None:
    """Plot fusion client-accuracy variance only."""
    x = np.arange(1)
    width = 0.34

    iid_means = []
    iid_stds = []
    dp_means = []
    dp_stds = []
    mean, std = _aggregate(results, "fusion", IID_NAME, "client_accuracy_variance")
    iid_means.append(mean)
    iid_stds.append(std)
    mean, std = _aggregate(results, "fusion", DP_NAME, "client_accuracy_variance")
    dp_means.append(mean)
    dp_stds.append(std)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(
        x - width / 2,
        iid_means,
        width,
        yerr=iid_stds,
        capsize=5,
        label="IID",
        color=IID_COLOR,
        alpha=0.9,
    )
    ax.bar(
        x + width / 2,
        dp_means,
        width,
        yerr=dp_stds,
        capsize=5,
        label="Non-IID + DP",
        color=DP_COLOR,
        alpha=0.9,
    )

    ax.set_title("Fusion Client Accuracy Variance")
    ax.set_ylabel("Variance of client accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(["Fusion"])
    ax.legend(frameon=False)

    save_fig(fig, out_dir, "fig3_fusion_client_accuracy_variance.png")


def plot_fusion_metrics(results: Dict, out_dir: str) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.34

    iid_means = []
    iid_stds = []
    dp_means = []
    dp_stds = []
    for metric in metrics:
        mean, std = _aggregate(results, "fusion", IID_NAME, metric)
        iid_means.append(mean)
        iid_stds.append(std)
        mean, std = _aggregate(results, "fusion", DP_NAME, metric)
        dp_means.append(mean)
        dp_stds.append(std)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(
        x - width / 2,
        iid_means,
        width,
        yerr=iid_stds,
        capsize=5,
        label="Fusion IID",
        color=IID_COLOR,
        alpha=0.9,
    )
    ax.bar(
        x + width / 2,
        dp_means,
        width,
        yerr=dp_stds,
        capsize=5,
        label="Fusion Non-IID + DP",
        color=DP_COLOR,
        alpha=0.9,
    )

    ax.set_title("Fusion Classification Metrics")
    ax.set_ylabel("Metric value (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics])
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)

    save_fig(fig, out_dir, "fig4_fusion_metrics_errorbars.png")


def plot_architecture_diagram(out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.16, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            color=TEXT_COLOR,
            fontsize=10,
            fontweight="bold",
            wrap=True,
        )

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#374151"),
        )

    box(0.05, 0.66, 0.17, 0.16, "Facial image\n224 x 224", FACIAL_COLOR)
    box(0.28, 0.66, 0.20, 0.16, "MobileNetV2\nfacial encoder", FACIAL_COLOR)
    box(0.54, 0.66, 0.16, 0.16, "1280-D\nembedding", FACIAL_COLOR)

    box(0.05, 0.31, 0.17, 0.16, "Behavioral video\n16 frames", BEHAVIORAL_COLOR)
    box(0.28, 0.31, 0.20, 0.16, "Frame MobileNetV2\n+ TCN", BEHAVIORAL_COLOR)
    box(0.54, 0.31, 0.16, 0.16, "256-D\nembedding", BEHAVIORAL_COLOR)

    box(0.76, 0.48, 0.18, 0.18, "Late fusion\nclassifier", FUSION_COLOR)
    box(0.76, 0.17, 0.18, 0.13, "FedAvg server\naggregates clients", "#475569")

    arrow(0.22, 0.74, 0.28, 0.74)
    arrow(0.48, 0.74, 0.54, 0.74)
    arrow(0.70, 0.74, 0.76, 0.60)

    arrow(0.22, 0.39, 0.28, 0.39)
    arrow(0.48, 0.39, 0.54, 0.39)
    arrow(0.70, 0.39, 0.76, 0.54)
    arrow(0.85, 0.48, 0.85, 0.30)

    ax.text(
        0.5,
        0.94,
        "Federated Multi-Modal ASD Detection Architecture",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.text(
        0.5,
        0.06,
        "Each client trains locally; raw facial/video data remain on-client. Only model weights are aggregated.",
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
    )

    save_fig(fig, out_dir, "fig5_architecture_diagram.png")


def write_summary_table(results: Dict, out_dir: str) -> None:
    path = os.path.join(out_dir, "paper_graph_values.md")
    lines = [
        "# Paper Graph Values",
        "",
        "Values are mean +/- std across saved runs.",
        "",
        "| Model | Config | Accuracy | Precision | Recall | F1 | Client variance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config in [IID_NAME, DP_NAME]:
        values = {}
        for metric in ["accuracy", "precision", "recall", "f1", "client_accuracy_variance"]:
            mean, std = _aggregate(results, "fusion", config, metric)
            values[metric] = f"{mean:.2f} +/- {std:.2f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    "Fusion",
                    "IID" if config == IID_NAME else "Non-IID + DP",
                    values["accuracy"],
                    values["precision"],
                    values["recall"],
                    values["f1"],
                    values["client_accuracy_variance"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Fusion Per Seed",
            "",
            "| Seed | IID accuracy | Non-IID + DP accuracy |",
            "|---:|---:|---:|",
        ]
    )
    for seed, iid, dp in zip(
        results["seeds"],
        _per_seed_values(results, "fusion", IID_NAME, "accuracy"),
        _per_seed_values(results, "fusion", DP_NAME, "accuracy"),
    ):
        lines.append(f"| {seed} | {iid:.2f} | {dp:.2f} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper plots from multi-run results.")
    parser.add_argument("--results", default=DEFAULT_RESULTS, help="Path to multi_run_results.json")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    args = parser.parse_args()

    results = load_results(args.results)
    ensure_dir(args.out_dir)

    plot_fusion_accuracy(results, args.out_dir)
    plot_fusion_per_seed(results, args.out_dir)
    plot_fusion_client_variance(results, args.out_dir)
    plot_fusion_metrics(results, args.out_dir)
    plot_architecture_diagram(args.out_dir)
    write_summary_table(results, args.out_dir)


if __name__ == "__main__":
    main()
