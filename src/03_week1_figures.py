from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("results/week1_metadata.csv")
results = Path("results")

# Figure 1: class distribution per split
counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
counts = counts.loc[["train", "val", "test"]]

fig, ax = plt.subplots(figsize=(7, 4))
counts.plot(kind="bar", ax=ax)
ax.set_title("Images per split and label")
ax.set_ylabel("Image count")
ax.set_xlabel("Split")
for container in ax.containers:
    ax.bar_label(container, fontsize=8)
plt.tight_layout()
plt.savefig(results / "week1_class_distribution.png", dpi=120)
plt.close()
print("Saved week1_class_distribution.png")

# Figure 2: sample grid, 4 per class
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for row, label in enumerate(["NORMAL", "PNEUMONIA"]):
    sample = df[df["label"] == label].sample(4, random_state=42)
    for col, (_, row_data) in enumerate(sample.iterrows()):
        with Image.open(row_data["path"]) as img:
            axes[row, col].imshow(img.convert("L"), cmap="gray")
        axes[row, col].set_title(
            f"{label}\n{row_data['width']}x{row_data['height']}", fontsize=9
        )
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig(results / "week1_samples.png", dpi=120)
plt.close()
print("Saved week1_samples.png")