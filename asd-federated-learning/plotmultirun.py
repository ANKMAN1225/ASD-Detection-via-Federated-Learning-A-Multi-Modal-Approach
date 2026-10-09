import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Paths
# ==========================================================

json_path = r"D:\WORK\VScode\Capstone\asd-federated-learning\multi_run_results.json"

output_dir = r"D:\WORK\VScode\Capstone\asd-federated-learning\research_visualizations\paper_graphs\new plots multirun"

os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# Load JSON
# ==========================================================

with open(json_path, "r") as f:
    data = json.load(f)

modalities = ["facial", "behavioral", "fusion"]
labels = ["Facial", "Behavioral", "Fusion"]

# ==========================================================
# Figure 1
# Mean Accuracy + 95% Confidence Interval
# ==========================================================

iid_means = []
dp_means = []

iid_ci = []
dp_ci = []

for m in modalities:
    iid = data["aggregate"][m]["IID_Distribution"]["accuracy"]
    dp = data["aggregate"][m]["With_Differential_Privacy"]["accuracy"]

    iid_means.append(iid["mean"])
    dp_means.append(dp["mean"])

    iid_ci.append(iid["ci_half_width"])
    dp_ci.append(dp["ci_half_width"])

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8,5))

plt.bar(
    x-width/2,
    iid_means,
    width,
    yerr=iid_ci,
    capsize=5,
    label="IID"
)

plt.bar(
    x+width/2,
    dp_means,
    width,
    yerr=dp_ci,
    capsize=5,
    label="Differential Privacy"
)

plt.xticks(x, labels)
plt.ylabel("Accuracy (%)")
plt.title("Mean Accuracy with 95% Confidence Intervals")
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "Figure1_MeanAccuracy_CI.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ==========================================================
# Figure 2
# Accuracy Boxplots
# ==========================================================

fig, axes = plt.subplots(1, 3, figsize=(13,4.5))

for ax, modality, label in zip(axes, modalities, labels):

    iid = [
        x["accuracy"]
        for x in data["per_run_results"][modality]["IID_Distribution"]
    ]

    dp = [
        x["accuracy"]
        for x in data["per_run_results"][modality]["With_Differential_Privacy"]
    ]

    ax.boxplot(
        [iid, dp],
        tick_labels=["IID", "DP"]
    )

    ax.set_title(label)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "Figure2_Boxplots.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Figure 3
# Client Variance
# ==========================================================

iid_var = []
dp_var = []

for m in modalities:

    iid_var.append(
        data["aggregate"][m]["IID_Distribution"]["client_accuracy_variance"]["mean"]
    )

    dp_var.append(
        data["aggregate"][m]["With_Differential_Privacy"]["client_accuracy_variance"]["mean"]
    )

plt.figure(figsize=(8,5))

plt.bar(
    x-width/2,
    iid_var,
    width,
    label="IID"
)

plt.bar(
    x+width/2,
    dp_var,
    width,
    label="Differential Privacy"
)

plt.xticks(x, labels)

plt.ylabel("Client Accuracy Variance")

plt.title("Average Client Accuracy Variance")

plt.legend()

plt.grid(axis='y', alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "Figure3_ClientVariance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Figure 4
# Per Run Accuracy
# ==========================================================

runs = np.arange(1, 6)

fig, axes = plt.subplots(1,3, figsize=(13,4.5))

for ax, modality, label in zip(axes, modalities, labels):

    iid = [
        x["accuracy"]
        for x in data["per_run_results"][modality]["IID_Distribution"]
    ]

    dp = [
        x["accuracy"]
        for x in data["per_run_results"][modality]["With_Differential_Privacy"]
    ]

    ax.plot(
        runs,
        iid,
        marker="o",
        linewidth=2,
        label="IID"
    )

    ax.plot(
        runs,
        dp,
        marker="s",
        linewidth=2,
        label="DP"
    )

    ax.set_title(label)

    ax.set_xlabel("Run")

    ax.set_ylabel("Accuracy (%)")

    ax.set_xticks(runs)

    ax.grid(alpha=0.3)

    ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "Figure4_PerRunAccuracy.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("="*60)
print("All plots saved successfully.")
print(output_dir)
print("="*60)