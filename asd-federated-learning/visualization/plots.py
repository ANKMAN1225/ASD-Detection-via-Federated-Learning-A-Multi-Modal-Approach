"""
All matplotlib/seaborn plotting functions for federated learning results.
"""

from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: save only, no display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tabulate import tabulate

import os

# ── Style setup ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# Save to project folder
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
SAVE_DIR = os.path.join(_PROJECT_DIR, "research_visualizations")

IID_COLOR = "#2563EB"
DP_COLOR = "#DC2626"
BG = "#F9FAFB"
CLIENT_COLORS_IID = ["#1D4ED8", "#2563EB", "#3B82F6", "#60A5FA", "#93C5FD"]
CLIENT_COLORS_DP = ["#991B1B", "#DC2626", "#EF4444", "#F87171", "#FCA5A5"]


def _get_metrics(targets, preds):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(targets, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        targets, preds, average="weighted"
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def _ensure_save_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)


def _set_comm_round_ticks(ax: plt.Axes, rounds: List[int], max_ticks: int = 12) -> None:
    """Thin x-axis ticks for communication rounds charts.

    With large `num_rounds` (e.g., 100), labeling every round makes the plot unreadable.
    """
    rounds = list(rounds)
    if not rounds:
        return

    if len(rounds) <= max_ticks:
        ax.set_xticks(rounds)
        return

    stride = int(np.ceil(len(rounds) / max_ticks))
    ticks = rounds[::stride]
    if ticks[-1] != rounds[-1]:
        ticks.append(rounds[-1])

    ax.set_xticks(ticks)


# Module-level data (populated by _extract_data)
iid_data = iid_rounds_data = iid_final = None
iid_rounds = iid_loss = iid_clients = iid_preds = iid_targets = []
iid_metrics = {}
iid_eval_acc_log: list = []
dp_data = dp_rounds_data = dp_final = None
dp_rounds = dp_loss = dp_clients = dp_preds = dp_targets = []
dp_metrics = {}
dp_eval_acc_log: list = []


def _extract_data(results: Dict[str, Any]) -> bool:
    """Extract IID and DP data from results. Returns True if both exist."""
    global iid_data, iid_rounds_data, iid_final, iid_rounds, iid_loss
    global iid_clients, iid_preds, iid_targets, iid_metrics, iid_eval_acc_log
    global dp_data, dp_rounds_data, dp_final, dp_rounds, dp_loss
    global dp_clients, dp_preds, dp_targets, dp_metrics, dp_eval_acc_log

    exp = results.get("experimental_results", {})
    if "IID_Distribution" not in exp or "With_Differential_Privacy" not in exp:
        return False

    def _get_exp_details(data):
        # Prefer facial, fallback to behavioral
        if "facial_experiment" in data:
            key = "facial_experiment"
        elif "behavioral_experiment" in data:
            key = "behavioral_experiment"
        else:
            return None
        
        rounds_data = data[key].get("round_metrics", [])
        final = data[key].get("final_metrics", {})
        eval_acc_log = data[key].get("eval_accuracy_log", data[key].get("val_accuracy_log", []))
        
        # If history is missing, try to use final metrics as a single round point
        if not rounds_data and final:
            rounds_data = [{"avg_loss": 0.0, "round": 1}] # dummy loss
        
        rounds = list(range(1, len(rounds_data) + 1))
        loss = [r.get("avg_loss", 0.0) for r in rounds_data]
        clients = final.get("client_metrics", [])
        
        # Safely extract predictions and targets
        def _to_int(x):
            if hasattr(x, "item"): return int(x.item())
            return int(x)
            
        preds = [_to_int(p) for p in final.get("predictions", [])]
        targets = [_to_int(t) for t in final.get("targets", [])]
        metrics = _get_metrics(targets, preds) if targets else {}
        
        return rounds_data, final, eval_acc_log, rounds, loss, clients, preds, targets, metrics

    iid_iid = _get_exp_details(exp["IID_Distribution"])
    if not iid_iid: return False
    iid_rounds_data, iid_final, iid_eval_acc_log, iid_rounds, iid_loss, iid_clients, iid_preds, iid_targets, iid_metrics = iid_iid

    dp_dp = _get_exp_details(exp["With_Differential_Privacy"])
    if not dp_dp: return False
    dp_rounds_data, dp_final, dp_eval_acc_log, dp_rounds, dp_loss, dp_clients, dp_preds, dp_targets, dp_metrics = dp_dp

    return True


# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Training Loss Comparison
# ══════════════════════════════════════════════════════════════════

def _fig1_loss():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor(BG)

    ax.plot(iid_rounds, iid_loss, color=IID_COLOR, lw=2.2, marker="o",
            markersize=5, label="Exp 1 — IID Distribution")
    ax.fill_between(iid_rounds,
                    [l * 0.97 for l in iid_loss],
                    [l * 1.03 for l in iid_loss],
                    color=IID_COLOR, alpha=0.12)

    ax.plot(dp_rounds, dp_loss, color=DP_COLOR, lw=2.2, marker="s",
            markersize=5, linestyle="--", label="Exp 3 — Differential Privacy")
    ax.fill_between(dp_rounds,
                    [l * 0.97 for l in dp_loss],
                    [l * 1.03 for l in dp_loss],
                    color=DP_COLOR, alpha=0.12)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Average Training Loss")
    ax.set_title("Figure 1: Federated Training Loss per Round")
    _set_comm_round_ticks(ax, iid_rounds, max_ticks=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig1_loss.png"))
    plt.close()
    print("Fig 1 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Global Accuracy over Rounds
# ══════════════════════════════════════════════════════════════════

def _fig2_accuracy(results):
    """Plot real per-round global accuracy from logged evaluation data."""
    if not iid_eval_acc_log or not dp_eval_acc_log:
        print(
            "WARNING: Figure 2 skipped because eval_accuracy_log is missing. "
            "Re-run experiments to generate per-round accuracy logs."
        )
        return

    iid_r = [entry["round"] for entry in iid_eval_acc_log]
    iid_acc = [entry["accuracy"] for entry in iid_eval_acc_log]
    dp_r = [entry["round"] for entry in dp_eval_acc_log]
    dp_acc = [entry["accuracy"] for entry in dp_eval_acc_log]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor(BG)

    ax.plot(iid_r, iid_acc, color=IID_COLOR, lw=2.2, marker="o",
            markersize=5, label="Exp 1 — IID Distribution")
    ax.plot(dp_r, dp_acc, color=DP_COLOR, lw=2.2, marker="s",
            markersize=5, linestyle="--", label="Exp 3 — Differential Privacy")

    # Annotate final accuracy value
    ax.annotate(f"{iid_acc[-1]:.1f}%", xy=(iid_r[-1], iid_acc[-1]),
                xytext=(-30, 8), textcoords="offset points",
                fontsize=9, color=IID_COLOR, fontweight="bold")
    ax.annotate(f"{dp_acc[-1]:.1f}%", xy=(dp_r[-1], dp_acc[-1]),
                xytext=(-30, -15), textcoords="offset points",
                fontsize=9, color=DP_COLOR, fontweight="bold")

    # Mark the peak accuracy point for each experiment
    iid_peak_idx = int(np.argmax(iid_acc))
    dp_peak_idx = int(np.argmax(dp_acc))
    ax.scatter([iid_r[iid_peak_idx]], [iid_acc[iid_peak_idx]],
               color=IID_COLOR, s=80, zorder=5,
               label=f"IID Best: {iid_acc[iid_peak_idx]:.1f}% @ round {iid_r[iid_peak_idx]}")
    ax.scatter([dp_r[dp_peak_idx]], [dp_acc[dp_peak_idx]],
               color=DP_COLOR, s=80, zorder=5, marker="^",
               label=f"DP Best: {dp_acc[dp_peak_idx]:.1f}% @ round {dp_r[dp_peak_idx]}")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Accuracy (%)")
    ax.set_title("Figure 2: Global Model Accuracy over Federated Rounds")
    all_r = sorted(set(iid_r + dp_r))
    _set_comm_round_ticks(ax, all_r, max_ticks=12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig2_accuracy.png"))
    plt.close()
    print("Fig 2 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Per-Client Accuracy IID
# ══════════════════════════════════════════════════════════════════

def _fig3_per_client_iid():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor(BG)

    for idx, client in enumerate(iid_clients):
        ax.bar(idx + 1, client["accuracy"],
               color=CLIENT_COLORS_IID[idx % len(CLIENT_COLORS_IID)],
               alpha=0.85, edgecolor="white", width=0.5,
               label=f"Client {client.get('client_id', idx)}")
        ax.text(idx + 1, client["accuracy"] + 0.3,
                f"{client['accuracy']:.1f}%",
                ha="center", va="bottom", fontsize=9,
                color=CLIENT_COLORS_IID[idx % 5])

    ax.axhline(iid_final["global_accuracy"], color="black", lw=2,
               linestyle="-.", label=f"Global Avg ({iid_final['global_accuracy']:.1f}%)")

    ax.set_xlabel("Client")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Figure 3: Per-Client Final Accuracy — Exp 1 (IID)")
    ax.set_xticks(range(1, len(iid_clients) + 1))
    ax.set_xticklabels([f"Client {i+1}" for i in range(len(iid_clients))])
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig3_per_client_iid.png"))
    plt.close()
    print("Fig 3 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Per-Client Accuracy DP
# ══════════════════════════════════════════════════════════════════

def _fig4_per_client_dp():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor(BG)

    for idx, client in enumerate(dp_clients):
        ax.bar(idx + 1, client["accuracy"],
               color=CLIENT_COLORS_DP[idx % len(CLIENT_COLORS_DP)],
               alpha=0.85, edgecolor="white", width=0.5)
        ax.text(idx + 1, client["accuracy"] + 0.3,
                f"{client['accuracy']:.1f}%",
                ha="center", va="bottom", fontsize=9,
                color=CLIENT_COLORS_DP[idx % 5])

    ax.axhline(dp_final["global_accuracy"], color="black", lw=2,
               linestyle="-.", label=f"Global Avg ({dp_final['global_accuracy']:.1f}%)")

    ax.set_xlabel("Client")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Figure 4: Per-Client Final Accuracy — Exp 3 (Differential Privacy)")
    ax.set_xticks(range(1, len(dp_clients) + 1))
    ax.set_xticklabels([f"Client {i+1}" for i in range(len(dp_clients))])
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig4_per_client_dp.png"))
    plt.close()
    print("Fig 4 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — Confusion Matrices
# ══════════════════════════════════════════════════════════════════

def _fig5_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")
    labels = ["Autistic", "Non-Autistic"]

    for ax, targets, preds, title, cmap in zip(
        axes,
        [iid_targets, dp_targets],
        [iid_preds, dp_preds],
        ["Exp 1 — IID Distribution", "Exp 3 — Differential Privacy"],
        ["Blues", "Reds"]
    ):
        cm = confusion_matrix(targets, preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, annot=False, cmap=cmap,
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.8})

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.5,
                        f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                        ha="center", va="center", fontsize=11,
                        color="white" if cm_norm[i, j] > 0.5 else "#1f2937",
                        fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    fig.suptitle("Figure 5: Confusion Matrices — Final Round",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig5_confusion.png"))
    plt.close()
    print("Fig 5 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 6 — Radar / Spider Chart
# ══════════════════════════════════════════════════════════════════

def _fig6_radar():
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    iid_vals = [iid_metrics["accuracy"], iid_metrics["precision"],
                iid_metrics["recall"], iid_metrics["f1"]]
    dp_vals = [dp_metrics["accuracy"], dp_metrics["precision"],
               dp_metrics["recall"], dp_metrics["f1"]]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    iid_v = iid_vals + iid_vals[:1]
    dp_v = dp_vals + dp_vals[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    ax.plot(angles, iid_v, color=IID_COLOR, lw=2.2, label="Exp 1 — IID")
    ax.fill(angles, iid_v, color=IID_COLOR, alpha=0.18)
    ax.plot(angles, dp_v, color=DP_COLOR, lw=2.2, linestyle="--", label="Exp 3 — DP")
    ax.fill(angles, dp_v, color=DP_COLOR, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title("Figure 6: Performance Metrics Radar Chart",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig6_radar.png"))
    plt.close()
    print("Fig 6 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 7 — IID vs DP Grouped Metric Bar
# ══════════════════════════════════════════════════════════════════

def _fig7_metric_bar():
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    iid_vals = [iid_metrics["accuracy"], iid_metrics["precision"],
                iid_metrics["recall"], iid_metrics["f1"]]
    dp_vals = [dp_metrics["accuracy"], dp_metrics["precision"],
               dp_metrics["recall"], dp_metrics["f1"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor(BG)

    b1 = ax.bar(x - width/2, iid_vals, width, color=IID_COLOR,
                alpha=0.85, label="Exp 1 — IID", edgecolor="white")
    b2 = ax.bar(x + width/2, dp_vals, width, color=DP_COLOR,
                alpha=0.85, label="Exp 3 — DP", edgecolor="white")

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=8, color=IID_COLOR)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=8, color=DP_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Figure 7: Classification Metrics — IID vs. Differential Privacy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig7_metric_bar.png"))
    plt.close()
    print("Fig 7 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 8 — Client Sample Distribution
# ══════════════════════════════════════════════════════════════════

def _fig8_sample_distribution():
    iid_samples = [c["samples"] for c in iid_clients]
    dp_samples = [c["samples"] for c in dp_clients]

    x = np.arange(len(iid_clients))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor(BG)

    ax.bar(x - width/2, iid_samples, width, color=IID_COLOR,
           alpha=0.85, label="Exp 1 — IID", edgecolor="white")
    ax.bar(x + width/2, dp_samples, width, color=DP_COLOR,
           alpha=0.85, label="Exp 3 — DP", edgecolor="white")

    ax.set_xlabel("Client")
    ax.set_ylabel("Number of Test Samples")
    ax.set_title("Figure 8: Test Sample Distribution per Client")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Client {i+1}" for i in range(len(iid_clients))])
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig8_sample_distribution.png"))
    plt.close()
    print("Fig 8 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 9 — Loss Convergence Rate
# ══════════════════════════════════════════════════════════════════

def _fig9_loss_delta():
    iid_delta = [abs(iid_loss[i] - iid_loss[i-1]) for i in range(1, len(iid_loss))]
    dp_delta = [abs(dp_loss[i] - dp_loss[i-1]) for i in range(1, len(dp_loss))]
    iid_rounds = list(range(2, len(iid_loss) + 1))
    dp_rounds = list(range(2, len(dp_loss) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor(BG)

    ax.plot(iid_rounds, iid_delta, color=IID_COLOR, lw=2.2, marker="o",
            markersize=5, label="Exp 1 — IID")
    ax.plot(dp_rounds, dp_delta, color=DP_COLOR, lw=2.2, marker="s",
            markersize=5, linestyle="--", label="Exp 3 — DP")
    ax.axhline(0.01, color="gray", lw=1.2, linestyle=":",
               label="Convergence threshold (0.01)")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("|Loss Change| from Previous Round")
    ax.set_title("Figure 9: Loss Convergence Rate per Round")
    all_rounds = sorted(set(iid_rounds + dp_rounds))
    _set_comm_round_ticks(ax, all_rounds, max_ticks=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig9_convergence_rate.png"))
    plt.close()
    print("Fig 9 saved")


# ══════════════════════════════════════════════════════════════════
# FIGURE 10 — Fairness: Client Accuracy Spread
# ══════════════════════════════════════════════════════════════════

def _fig10_fairness():
    iid_accs = [c["accuracy"] for c in iid_clients]
    dp_accs = [c["accuracy"] for c in dp_clients]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    for ax, accs, color, title in zip(
        axes,
        [iid_accs, dp_accs],
        [IID_COLOR, DP_COLOR],
        ["Exp 1 — IID Distribution", "Exp 3 — Differential Privacy"]
    ):
        ax.set_facecolor(BG)
        clients = [f"C{i+1}" for i in range(len(accs))]
        mean = np.mean(accs)
        std = np.std(accs)

        ax.bar(clients, accs, color=color, alpha=0.8, edgecolor="white")
        ax.axhline(mean, color="black", lw=2, linestyle="-.",
                   label=f"Mean: {mean:.1f}%")
        ax.fill_between(range(len(accs)), mean - std, mean + std,
                        color=color, alpha=0.15, label=f"±1 Std Dev: {std:.2f}%")

        for i, acc in enumerate(accs):
            ax.text(i, acc + 0.5, f"{acc:.1f}%", ha="center", fontsize=9, color=color)

        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Client")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)

    fig.suptitle("Figure 10: Fairness Analysis — Client Accuracy Spread",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig10_fairness.png"))
    plt.close()
    print("Fig 10 saved")


# ══════════════════════════════════════════════════════════════════
# ADDITIONAL FIGURES — IID vs DP Grouped Bar, AUC-ROC, Fairness Box
# ══════════════════════════════════════════════════════════════════

def _fig11_grouped_metrics_pct():
    """Grouped bar chart: IID Baseline vs DP-Enabled, 0–100% scale."""
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    iid_vals = np.array([
        iid_metrics["accuracy"], iid_metrics["precision"],
        iid_metrics["recall"], iid_metrics["f1"]
    ]) * 100
    dp_vals = np.array([
        dp_metrics["accuracy"], dp_metrics["precision"],
        dp_metrics["recall"], dp_metrics["f1"]
    ]) * 100

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor(BG)

    b1 = ax.bar(x - width/2, iid_vals, width, color=IID_COLOR,
                alpha=0.85, label="IID Baseline", edgecolor="white")
    b2 = ax.bar(x + width/2, dp_vals, width, color=DP_COLOR,
                alpha=0.85, label="DP-Enabled", edgecolor="white")

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9,
                color=IID_COLOR, fontweight="bold")
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9,
                color=DP_COLOR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("IID Baseline vs DP-Enabled: Classification Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig11_grouped_metrics_pct.png"))
    plt.close()
    print("Fig 11 (grouped bar) saved")


def _fig12_auc_roc():
    """AUC-ROC curves for IID and DP experiments."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor(BG)

    for targets, preds, color, label in [
        (iid_targets, iid_preds, IID_COLOR, "IID Baseline"),
        (dp_targets, dp_preds, DP_COLOR, "DP-Enabled"),
    ]:
        y_true = np.array(targets)
        y_pred = np.array(preds)
        # Map hard predictions to score-like values for ROC (pred 0 -> 0.1, pred 1 -> 0.9)
        y_scores = np.where(np.array(preds) == 1, 0.9, 0.1).astype(float)

        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{label} (AUC = {roc_auc:.3f})")
            ax.fill_between(fpr, tpr, alpha=0.15, color=color)
        except Exception as e:
            print(f"  WARNING: AUC-ROC for {label} skipped: {e}")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("AUC-ROC Curves: IID vs DP-Enabled")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig12_auc_roc.png"))
    plt.close()
    print("Fig 12 (AUC-ROC) saved")


def _fig13_fairness_boxplot():
    """Fairness & client variance: box plot of per-client accuracy (IID vs DP)."""
    iid_accs = [c["accuracy"] for c in iid_clients]
    dp_accs = [c["accuracy"] for c in dp_clients]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor(BG)

    data = [iid_accs, dp_accs]
    labels = ["IID Baseline", "DP-Enabled"]
    colors = [IID_COLOR, DP_COLOR]
    # Use tick_labels (Matplotlib 3.9+) or labels (legacy)
    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for whisker in bp["whiskers"]:
        whisker.set_color("black")
    for cap in bp["caps"]:
        cap.set_color("black")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_ylabel("Client Accuracy (%)")
    ax.set_title("Fairness & Client Variance: Per-Client Accuracy Distribution")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "fig13_fairness_boxplot.png"))
    plt.close()
    print("Fig 13 (fairness box plot) saved")


def generate_research_visualizations(results: Dict[str, Any]) -> None:
    """
    Generate all research figures from experimental results (including IID vs DP grouped bar, AUC-ROC, fairness box plot).
    Saves to research_visualizations/ in the project folder. No display.
    Requires IID_Distribution and With_Differential_Privacy experiments.
    """
    _ensure_save_dir()

    if not _extract_data(results):
        print("WARNING: Cannot generate research visualizations: need IID_Distribution "
              "and With_Differential_Privacy in experimental_results.")
        return

    print("Generating visualizations from experiment data...")
    print("=" * 55)

    _fig1_loss()
    _fig2_accuracy(results)
    _fig3_per_client_iid()
    _fig4_per_client_dp()
    _fig5_confusion()
    _fig6_radar()
    _fig7_metric_bar()
    _fig8_sample_distribution()
    _fig9_loss_delta()
    _fig10_fairness()
    _fig11_grouped_metrics_pct()
    _fig12_auc_roc()
    _fig13_fairness_boxplot()

    print(f"\nAll 13 figures saved to: {SAVE_DIR}")


def print_detailed_results(results: Dict[str, Any]) -> None:
    """Print detailed analysis of experimental results."""
    print("\n" + "=" * 80)
    print("DETAILED EXPERIMENTAL RESULTS ANALYSIS")
    print("=" * 80)

    if "experimental_results" not in results or not results["experimental_results"]:
        print("WARNING: No experimental results found!")
        return

    experimental_results = results["experimental_results"]
    filtered_results = {
        k: v for k, v in experimental_results.items()
        if ("IID" in k or "Privacy" in k) and "Non" not in k
    }

    print("\nExperiment Summary")
    print("-" * 50)
    print(f"Total Experiments Conducted: {len(filtered_results)}")
    print(f"Successful: {len([r for r in filtered_results.values() if r is not None])}")

    for exp_name, exp_data in filtered_results.items():
        if exp_data is None:
            continue
        print(f"\nExperiment: {exp_name}")
        print("=" * (15 + len(exp_name)))

        config = exp_data.get("config", {})
        print(f"\nConfiguration:")
        print(f"   - Clients: {config.get('num_clients', 'N/A')}")
        print(f"   - Rounds: {config.get('num_rounds', 'N/A')}")
        dp_str = "Enabled" if config.get("use_differential_privacy") else "Disabled"
        print(f"   - Differential Privacy: {dp_str}")

        facial_exp = exp_data.get("facial_experiment", {})
        round_metrics = facial_exp.get("round_metrics", [])
        if round_metrics:
            print(
                "\nTraining: "
                f"Initial Loss {round_metrics[0]['avg_loss']:.4f} -> "
                f"Final {round_metrics[-1]['avg_loss']:.4f}"
            )

        final_metrics = facial_exp.get("final_metrics", {})
        if final_metrics:
            print(
                f"\nFinal Global Accuracy: {final_metrics.get('global_accuracy', 0):.2f}%"
            )


def generate_research_summary(results: Dict[str, Any]) -> None:
    """Generate research paper-style summary."""
    print("\n" + "=" * 80)
    print("RESEARCH PAPER SUMMARY")
    print("=" * 80)

    exp = results.get("experimental_results", {})
    filtered = {
        k: v for k, v in exp.items()
        if ("IID" in k or "Privacy" in k) and "Non" not in k
    }

    all_acc = []
    for exp_data in filtered.values():
        if exp_data and "facial_experiment" in exp_data:
            fm = exp_data["facial_experiment"].get("final_metrics", {})
            if fm:
                all_acc.append(fm.get("global_accuracy", 0))

    if all_acc:
        print(
            "\nMean accuracy across experiments: "
            f"{np.mean(all_acc):.2f}% +/- {np.std(all_acc):.2f}%"
        )


def create_comparison_table(results: Dict[str, Any]) -> None:
    """Create detailed comparison table."""
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON TABLE")
    print("=" * 80)

    exp = results.get("experimental_results", {})
    filtered = {
        k: v for k, v in exp.items()
        if ("IID" in k or "Privacy" in k) and "Non" not in k
    }

    table_data = []
    for exp_name, exp_data in filtered.items():
        if exp_data and "facial_experiment" in exp_data:
            fe = exp_data["facial_experiment"]
            fm = fe.get("final_metrics", {})
            rm = fe.get("round_metrics", [])
            config = exp_data.get("config", {})
            acc = fm.get("global_accuracy", 0)
            rounds = len(rm)
            privacy = "DP" if config.get("use_differential_privacy") else "None"
            table_data.append([exp_name.replace("_", " "), f"{acc:.2f}", rounds, privacy])

    if table_data:
        print(tabulate(table_data, headers=["Experiment", "Accuracy (%)", "Rounds", "Privacy"], tablefmt="grid"))
    else:
        print("No data available.")


def run_complete_analysis(
    results: Dict[str, Any],
    save_graphs: bool = True,
) -> None:
    """Run complete analysis pipeline: print results, generate figures, summary, table."""
    print("\nSTARTING COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 80)

    print_detailed_results(results)
    if save_graphs:
        generate_research_visualizations(results)
    generate_research_summary(results)
    create_comparison_table(results)
    print("\nAnalysis complete!")
