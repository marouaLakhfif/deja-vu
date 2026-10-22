from pathlib import Path
import pandas as pd
from PIL import Image

RAW = Path("data/raw/chest_xray")
SPLITS = ["train", "test", "val"]
LABELS = ["NORMAL", "PNEUMONIA"]

rows = []
errors = []

for split in SPLITS:
    for label in LABELS:
        folder = RAW / split / label
        for img_path in folder.iterdir():
            if not img_path.is_file():
                continue
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode
                rows.append({
                    "path": str(img_path),
                    "split": split,
                    "label": label,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "bytes": img_path.stat().st_size,
                })
            except Exception as e:
                errors.append({"path": str(img_path), "error": str(e)})

df = pd.DataFrame(rows)
print(f"Read: {len(df)} images, errors: {len(errors)}")
print()
print("Counts per split and label:")
print(df.groupby(["split", "label"]).size())
print()
print("Dimension distribution (top 5):")
print(df.groupby(["width", "height"]).size().sort_values(ascending=False).head())
print()
print("Pixel mode:")
print(df["mode"].value_counts())

df.to_csv("results/week1_metadata.csv", index=False)
print("\nSaved to results/week1_metadata.csv")