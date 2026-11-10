from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "results" / "week1_metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "resnet18_test_embeddings.npz"


# Load image metadata
df = pd.read_csv(METADATA_PATH)

# Select two images from each split and label group
sample = (
    df.groupby(["split", "label"], sort=False)
    .head(2)
    .reset_index(drop=True)
)


# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load pretrained ResNet-18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Remove the classification layer
model.fc = torch.nn.Identity()
model.eval()


# Extract features
all_features = []

with torch.no_grad():
    for relative_path in sample["path"]:
        image_path = PROJECT_ROOT / relative_path

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            tensor = preprocess(image).unsqueeze(0)

        features = model(tensor)
        all_features.append(features.squeeze(0).numpy())


# Convert features into a NumPy matrix
embeddings = np.array(all_features)


np.savez_compressed(
    OUTPUT_PATH,
    embeddings=embeddings,
    paths=sample["path"].to_numpy(dtype=str),
    labels=sample["label"].to_numpy(dtype=str),
    splits=sample["split"].to_numpy(dtype=str)
)


print("Feature extraction completed!")
print("Number of images:", len(sample))
print("Embedding matrix shape:", embeddings.shape)
print("Saved to:", OUTPUT_PATH)