
from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image

# 1. Load metadata
PROJECT_ROOT = Path.cwd()
METADATA_PATH = PROJECT_ROOT / "results" / "week1_metadata.csv"

df = pd.read_csv(METADATA_PATH)

# 2. Check whether image files exist
df["path_exists"] = df["path"].apply(
    lambda p: (PROJECT_ROOT / p).is_file()
)

print("File existence results:")
print(df["path_exists"].value_counts(dropna=False))

missing = df[~df["path_exists"]]

print(f"\nMissing files: {len(missing)}")

if not missing.empty:
    print(missing["path"].head(10).to_string(index=False))

# 3. Check whether images can be opened
readable = 0
unreadable = []

for path in df.loc[df["path_exists"], "path"]:
    img_path = PROJECT_ROOT / path

    try:
        with Image.open(img_path) as img:
            img.verify()
        readable += 1

    except Exception as e:
        unreadable.append({
            "path": str(path),
            "error": str(e)
        })

print(f"\nReadable images: {readable}")
print(f"Unreadable images: {len(unreadable)}")

if unreadable:
    print("\nUnreadable image examples:")
    for item in unreadable[:10]:
        print(item["path"], ":", item["error"])

# 4. Calculate SHA-256 hashes for existing files
hashes = {}

for path in df.loc[df["path_exists"], "path"]:
    img_path = PROJECT_ROOT / path

    try:
        digest = hashlib.sha256(
            img_path.read_bytes()
        ).hexdigest()

        hashes.setdefault(digest, []).append(str(path))

    except OSError as e:
        print(f"Could not hash {path}: {e}")

# 5. Identify duplicate files
duplicate_groups = [
    paths for paths in hashes.values()
    if len(paths) > 1
]

print(f"\nDuplicate groups: {len(duplicate_groups)}")

for paths in duplicate_groups[:10]:
    print(paths)

# 6. Check whether duplicates occur across splits
print("\nDuplicate groups by split:")

for paths in duplicate_groups:
    splits = set(
        df.loc[df["path"].isin(paths), "split"]
    )

    if len(splits) > 1:
        print("CROSS-SPLIT:", paths, "| Splits:", splits)

# Summarize duplicate groups by split.
duplicate_group_counts = {}
duplicate_file_count = 0

for paths in duplicate_groups:
    duplicate_file_count += len(paths)

    splits = {
        df.loc[df["path"] == path, "split"].iloc[0]
        for path in paths
    }

    for split in splits:
        duplicate_group_counts[split] = (
            duplicate_group_counts.get(split, 0) + 1
        )

print("\nDuplicate group counts by split:")
print(duplicate_group_counts)

print(f"\nFiles involved in duplicate groups: {duplicate_file_count}")
from collections import Counter

group_sizes = Counter(len(paths) for paths in duplicate_groups)

print("\nDuplicate group sizes:")
for size, count in sorted(group_sizes.items()):
    print(f"{count} groups contain {size} files")