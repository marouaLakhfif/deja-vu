from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    PROJECT_ROOT / "results" / "resnet18_retrieval_results.csv"
)

df = pd.read_csv(RESULTS_PATH)

# Select one NORMAL query
query_row = df[df["query_label"] == "NORMAL"].iloc[0]
query_path = query_row["query_path"]

neighbors = df[df["query_path"] == query_path].sort_values("rank")

# Load query image
query_image_path = PROJECT_ROOT / query_path

with Image.open(query_image_path) as img:
    query_image = img.convert("L")

# Create visualization
fig, axes = plt.subplots(1, 6, figsize=(16, 4))

axes[0].imshow(query_image, cmap="gray")
axes[0].set_title("QUERY\nNORMAL")

for i, (_, row) in enumerate(neighbors.iterrows(), start=1):

    image_path = PROJECT_ROOT / row["path"]

    with Image.open(image_path) as img:
        image = img.convert("L")

    axes[i].imshow(image, cmap="gray")

    axes[i].set_title(
        f"Rank {row['rank']}\n"
        f"{row['label']}\n"
        f"Match: {row['label_match']}"
    )

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output_path = (
    PROJECT_ROOT / "results" / "resnet18_retrieval_example.png"
)

plt.savefig(output_path, dpi=150)
plt.show()

print("Visualization saved to:", output_path)