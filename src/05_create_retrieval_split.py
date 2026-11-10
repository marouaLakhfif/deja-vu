
from pathlib import Path
import hashlib
import pandas as pd

PROJECT_ROOT = Path.cwd()
METADATA_PATH = PROJECT_ROOT / "results" / "week1_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "results"

# Load metadata and keep only the official training split.
df = pd.read_csv(METADATA_PATH)
train_df = df[df["split"] == "train"].copy()

# Calculate a hash for each image file.
def get_hash(relative_path):
    image_path = PROJECT_ROOT / relative_path
    return hashlib.sha256(image_path.read_bytes()).hexdigest()

train_df["hash"] = train_df["path"].apply(get_hash)

# Each hash group contains byte-identical files.
unique_groups = list(train_df.groupby("hash"))

# Shuffle groups reproducibly.
import random

rng = random.Random(42)
rng.shuffle(unique_groups)

# Assign whole groups to query or gallery.
query_groups = []
gallery_groups = []

for _, group in unique_groups:
    if len(query_groups) < int(len(train_df) * 0.2):
        query_groups.append(group)
    else:
        gallery_groups.append(group)

query_df = pd.concat(query_groups).copy()
gallery_df = pd.concat(gallery_groups).copy()
# Check that exact duplicate images do not cross between sets.
query_hashes = set(query_df["hash"])
gallery_hashes = set(gallery_df["hash"])

overlap = query_hashes & gallery_hashes

print(f"\nExact-hash overlap: {len(overlap)}")
assert len(overlap) == 0, "Exact duplicates crossed between sets!"
print("Split verification passed.")
# Remove helper hashes from the output.
query_df = query_df.drop(columns=["hash"])
gallery_df = gallery_df.drop(columns=["hash"])

query_df.to_csv(OUTPUT_DIR / "query_images.csv", index=False)
gallery_df.to_csv(OUTPUT_DIR / "gallery_images.csv", index=False)

print(f"Training images: {len(train_df)}")
print(f"Query images: {len(query_df)}")
print(f"Gallery images: {len(gallery_df)}")
print("\nQuery labels:")
print(query_df["label"].value_counts())
print("\nGallery labels:")
print(gallery_df["label"].value_counts())
