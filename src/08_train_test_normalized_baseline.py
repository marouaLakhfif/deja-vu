
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path.cwd()
RESULTS_DIR = PROJECT_ROOT / "results"
METADATA_PATH = RESULTS_DIR / "week1_metadata.csv"

IMAGE_SIZE = (64, 64)
TOP_K = 5


def image_hash(relative_path):
    """Identify files with exactly identical contents."""
    path = PROJECT_ROOT / relative_path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_pixel_features(relative_path):
    """Resize an image and standardize its pixel intensities."""
    path = PROJECT_ROOT / relative_path

    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize(IMAGE_SIZE)
        pixels = np.asarray(img, dtype=np.float32) / 255.0

    # Calculate the mean brightness and standard deviation.
    mean = pixels.mean()
    std = pixels.std()

    # Standardize pixel intensities.
    if std > 1e-8:
        pixels = (pixels - mean) / std
    else:
        pixels = pixels - mean

    return pixels.flatten()


# Load metadata and preserve the official dataset splits.
df = pd.read_csv(METADATA_PATH)
train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

print(f"Train gallery images: {len(train_df)}")
print(f"Test query images: {len(test_df)}")

# Check for byte-identical files across train and test.
train_hashes = set(train_df["path"].apply(image_hash))
test_hashes = set(test_df["path"].apply(image_hash))
overlap = train_hashes & test_hashes

print(f"Exact-hash overlap between train and test: {len(overlap)}")

# Extract features for every gallery image.
print("\nExtracting gallery features...")
gallery_features = np.stack([
    extract_pixel_features(path)
    for path in train_df["path"]
])

print(f"Gallery feature matrix: {gallery_features.shape}")

# Search the gallery for every test image.
all_results = []

print("\nRetrieving nearest neighbors...")
for query_index, query in test_df.iterrows():
    query_features = extract_pixel_features(query["path"])

    distances = np.linalg.norm(
        gallery_features - query_features,
        axis=1
    )

    nearest_indices = np.argsort(distances)[:TOP_K]
    matches = train_df.iloc[nearest_indices].copy()

    matches["rank"] = np.arange(1, TOP_K + 1)
    matches["distance"] = distances[nearest_indices]
    matches["query_path"] = query["path"]
    matches["query_label"] = query["label"]

    all_results.append(matches)

results_df = pd.concat(all_results, ignore_index=True)

# Evaluate label agreement, not clinical accuracy.
results_df["label_match"] = (
    results_df["label"] == results_df["query_label"]
)

per_query = results_df.groupby(
    ["query_path", "query_label"]
)["label_match"].mean().reset_index(name="label_agreement_at_5")

print("\nQueries per class:")
print(test_df["label"].value_counts())

print("\nMean label agreement at 5 by query class:")
per_class = per_query.groupby("query_label")[
    "label_agreement_at_5"
].mean()
print(per_class.round(3))

print(f"\nOverall mean label agreement at 5: "
      f"{per_query['label_agreement_at_5'].mean():.3f}")

results_df.to_csv(
    RESULTS_DIR / "train_test_pixel_normalized_results.csv",
    index=False
)

per_query.to_csv(
    RESULTS_DIR / "train_test_pixel_normalized_metrics.csv",
    index=False
)
print("\nSaved:")
print("results/train_test_pixel_normalized_results.csv")
print("results/train_test_pixel_normalized_metrics.csv")
