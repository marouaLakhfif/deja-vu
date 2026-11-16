from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

RAW_PATH = RESULTS_DIR / "train_test_pixel_raw_metrics.csv"
NORMALIZED_PATH = RESULTS_DIR / "train_test_pixel_normalized_metrics.csv"
RESNET_PATH = RESULTS_DIR / "resnet18_retrieval_metrics.csv"

raw_metrics = pd.read_csv(RAW_PATH)
normalized_metrics = pd.read_csv(NORMALIZED_PATH)
resnet_metrics = pd.read_csv(RESNET_PATH)

print("Raw pixel metrics:")
print(raw_metrics)

print("\nNormalized pixel metrics:")
print(normalized_metrics)

print("\nResNet-18 metrics:")
print(resnet_metrics)
print("\nResNet-18 metrics:")
print(resnet_metrics)

raw_overall = raw_metrics["label_agreement_at_5"].mean() * 100

normalized_overall = (
    normalized_metrics["label_agreement_at_5"].mean() * 100
)

resnet_overall = resnet_metrics.loc[
    resnet_metrics["label"] == "OVERALL",
    "percentage"
].iloc[0]

print("\nOverall label agreement@5:")
print(f"Raw pixels: {raw_overall:.1f}%")
print(f"Normalized pixels: {normalized_overall:.1f}%")
print(f"ResNet-18: {resnet_overall:.1f}%")

methods = [
    "Raw pixels",
    "Normalized pixels",
    "ResNet-18"
]

scores = [
    raw_overall,
    normalized_overall,
    resnet_overall
]

plt.figure(figsize=(8, 5))

bars = plt.bar(methods, scores)

plt.title("Comparison of Image Retrieval Methods")
plt.ylabel("Overall Label Agreement@5 (%)")
plt.ylim(0, 100)

plt.bar_label(bars, fmt="%.1f%%", padding=3)

plt.tight_layout()

CHART_PATH = RESULTS_DIR / "method_comparison.png"
plt.savefig(CHART_PATH, dpi=300)
plt.show()

print("Chart saved to:", CHART_PATH)