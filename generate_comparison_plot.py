import matplotlib.pyplot as plt
import os
import numpy as np

# Comparison data extracted from PDF and current project results
models = [
    "Sarker et al.",
    "Lakkapragada et al.",
    "Ali et al.",
    "Our Model (Federated Learning)",
    "VGG-16-LSTM "
]

accuracy = [
    69.0,    # Sarker
    84.0,    # Lakkapragada
    85.6,    # Ali
    86.0,    # Current Project (Our Model)
    88.0     # VGG-16-LSTM (Corrected)
]

# Sort by accuracy for better visualization
sorted_indices = np.argsort(accuracy)
models = [models[i] for i in sorted_indices]
accuracy = [accuracy[i] for i in sorted_indices]

# Colors: highlight "Our Model"
colors = ['#E5E7EB' if "Our Model" not in m and "LRCN (Paper)" not in m else ('#2563EB' if "Our Model" in m else '#10B981') for m in models]

plt.figure(figsize=(10, 6))
bars = plt.barh(models, accuracy, color=colors, alpha=0.9, edgecolor='white', linewidth=1)

# Style setup
plt.title("Comparison of ASD Detection Models", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Accuracy (%)", fontsize=12)
plt.xlim(0, 110)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Remove spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add values to bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
             va='center', fontweight='bold' if width >= 86.0 else 'normal')

plt.tight_layout()

# Save the plot
save_path = r"d:\WORK\VScode\Capstone\asd-federated-learning\research_visualizations\model_comparison.png"
plt.savefig(save_path, dpi=300)
print(f"Graph saved to {save_path}")
