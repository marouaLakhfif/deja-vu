from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = PROJECT_ROOT / "results" / "train_test_pixel_normalized_results.csv"

# Load retrieval results
df = pd.read_csv(RESULTS_PATH)

print(df.head())
print(df.columns.tolist())
# Select the first query image
query_path = df["query_path"].iloc[0]

# Get all retrieved neighbors for this query
neighbors = df[df["query_path"] == query_path]

print("Query image:", query_path)
print("Number of neighbors:", len(neighbors))
print(neighbors[["path", "label", "rank", "distance", "label_match"]])
# Load the original query image
query_image_path = PROJECT_ROOT / query_path

with Image.open(query_image_path) as img:
    query_image = img.convert("L")

# Load the five retrieved neighbor images
neighbor_images = []

for _, row in neighbors.iterrows():
    image_path = PROJECT_ROOT / row["path"]

    with Image.open(image_path) as img:
        neighbor_images.append(img.convert("L"))
        # Create a figure with one query and five neighbors
fig, axes = plt.subplots(1, 6, figsize=(18, 4))

# Display the query image
axes[0].imshow(query_image, cmap="gray")
axes[0].set_title(f"Query\n{df['query_label'].iloc[0]}")
axes[0].axis("off")

# Display the five retrieved neighbors
for i, (_, row) in enumerate(neighbors.iterrows()):
    axes[i + 1].imshow(neighbor_images[i], cmap="gray")

    axes[i + 1].set_title(
        f"Neighbor {row['rank']}\n"
        f"{row['label']}\n"
        f"Distance: {row['distance']:.2f}"
    )

    axes[i + 1].axis("off")

plt.tight_layout()
plt.show()