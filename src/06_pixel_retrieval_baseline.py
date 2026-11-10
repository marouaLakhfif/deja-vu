
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path.cwd()
RESULTS_DIR = PROJECT_ROOT / "results"

GALLERY_PATH = RESULTS_DIR / "gallery_images.csv"
QUERY_PATH = RESULTS_DIR / "query_images.csv"

IMAGE_SIZE = (64, 64)
TOP_K = 5
NUM_QUERIES = 50


def image_hash(relative_path):
    """Return a hash identifying the exact file contents."""
    path = PROJECT_ROOT / relative_path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_pixel_features(relative_path):
    """Resize an image and convert it into a normalized pixel vector."""
    path = PROJECT_ROOT / relative_path

    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize(IMAGE_SIZE)

        pixels = np.asarray(img, dtype=np.float32) / 255.0

    return pixels.flatten()


# Load the gallery and query lists.
gallery_df = pd.read_csv(GALLERY_PATH)
query_df = pd.read_csv(QUERY_PATH)

# Remove repeated exact copies from the query evaluation.
query_df["hash"] = query_df["path"].apply(image_hash)
query_df = query_df.drop_duplicates(subset="hash").copy()

# Use a reproducible sample for the first experiment.
#query_df = query_df.sample(
#    n=min(NUM_QUERIES, len(query_df)),
#    random_state=42
#).reset_index(drop=True)
# Select NORMAL queries only for this experiment.
normal_queries = query_df[query_df["label"] == "NORMAL"]

query_df = normal_queries.sample(
    n=min(NUM_QUERIES, len(normal_queries)),
    random_state=42
).reset_index(drop=True)
print(f"Gallery images: {len(gallery_df)}")
print(f"Unique query images evaluated: {len(query_df)}")

# Convert all gallery images into pixel vectors.
gallery_features = np.stack([
    extract_pixel_features(path)
    for path in gallery_df["path"]
])

print(f"Feature matrix shape: {gallery_features.shape}")

# Store retrieval results for each query.
all_results = []
first_query_matches = None

for query_index, query in query_df.iterrows():
    query_features = extract_pixel_features(query["path"])

    # Euclidean distance: smaller means more similar pixel vectors.
    distances = np.linalg.norm(
        gallery_features - query_features,
        axis=1
    )

    nearest_indices = np.argsort(distances)[:TOP_K]

    matches = gallery_df.iloc[nearest_indices].copy()
    matches["distance"] = distances[nearest_indices]
    matches["query_path"] = query["path"]
    matches["query_label"] = query["label"]

    all_results.append(matches)

    if query_index == 0:
        first_query_matches = matches.copy()

# Combine and save the retrieval results.
results_df = pd.concat(all_results, ignore_index=True)
results_df.to_csv(
    RESULTS_DIR / "pixel_baseline_results.csv",
    index=False
)

# Calculate label agreement at K=5.
results_df["label_match"] = (
    results_df["label"] == results_df["query_label"]
)

# Count matching labels among the top 5 for each query.
per_query = (
    results_df.groupby(["query_path", "query_label"])["label_match"]
    .sum()
    .reset_index(name="matching_labels_at_5")
)

print("\nNORMAL query results: matching labels among top 5")
print(
    per_query["matching_labels_at_5"]
    .value_counts()
    .sort_index()
    .to_string()
)

# Inspect the queries with the fewest matching labels.
worst_queries = per_query.sort_values(
    "matching_labels_at_5"
).head(5)

print("\nFive queries with the fewest matching labels:")
print(
    worst_queries[
        ["query_path", "matching_labels_at_5"]
    ].to_string(index=False)
)


precision_at_5 = results_df.groupby("query_path")[
    "label_match"
].mean().mean()

print(f"\nMean label agreement at 5: {precision_at_5:.3f}")

# Evaluate label agreement separately for each query class.
print("\nQueries evaluated per class:")
print(query_df["label"].value_counts())

print("\nMean label agreement at 5 by query class:")
per_class_agreement = results_df.groupby("query_label")[
    "label_match"
].mean()

print(per_class_agreement.round(3))

print("Saved results/pixel_baseline_results.csv")


# Visualize a NORMAL query with zero matching labels.
worst_query_path = (
    "data\\raw\\chest_xray\\train\\NORMAL\\IM-0764-0001.jpeg"
)

worst_matches = results_df[
    results_df["query_path"] == worst_query_path
].sort_values("distance").head(TOP_K)

fig, axes = plt.subplots(1, TOP_K + 1, figsize=(15, 4))

# Display the query.
with Image.open(PROJECT_ROOT / worst_query_path) as img:
    axes[0].imshow(img.convert("L"), cmap="gray")

axes[0].set_title("QUERY\nNORMAL")
axes[0].axis("off")

# Display the five nearest gallery images.
for rank, (_, match) in enumerate(worst_matches.iterrows(), start=1):
    with Image.open(PROJECT_ROOT / match["path"]) as img:
        axes[rank].imshow(img.convert("L"), cmap="gray")

    axes[rank].set_title(
        f"Rank {rank}\n{match['label']}\n"
        f"Distance: {match['distance']:.2f}"
    )
    axes[rank].axis("off")

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "pixel_baseline_worst_case.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("Saved results/pixel_baseline_worst_case.png")
print("\nRetrieved images for the worst query:")
print(
    worst_matches[["path", "label", "distance"]]
    .to_string(index=False)
)
