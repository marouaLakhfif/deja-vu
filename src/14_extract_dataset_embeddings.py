from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "results" / "week1_metadata.csv"

GALLERY_OUTPUT = PROJECT_ROOT / "results" / "resnet18_gallery_embeddings.npz"
QUERY_OUTPUT = PROJECT_ROOT / "results" / "resnet18_query_embeddings.npz"


# Settings
BATCH_SIZE = 32
DEVICE = torch.device("cpu")


# Load metadata
df = pd.read_csv(METADATA_PATH)

gallery_df = df[df["split"] == "train"].reset_index(drop=True)
query_df = df[df["split"] == "test"].reset_index(drop=True)


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

model.fc = torch.nn.Identity()
model = model.to(DEVICE)
model.eval()


def extract_embeddings(split_df, split_name):
    feature_batches = []

    with torch.no_grad():

        for start in range(0, len(split_df), BATCH_SIZE):

            batch_df = split_df.iloc[start:start + BATCH_SIZE]

            batch_images = []

            for relative_path in batch_df["path"]:
                image_path = PROJECT_ROOT / relative_path

                with Image.open(image_path) as img:
                    image = img.convert("RGB")
                    tensor = preprocess(image)

                batch_images.append(tensor)

            batch_tensor = torch.stack(batch_images).to(DEVICE)

            features = model(batch_tensor)

            feature_batches.append(features.cpu().numpy())

            processed = min(start + BATCH_SIZE, len(split_df))

            print(
                f"{split_name}: "
                f"{processed}/{len(split_df)} images processed"
            )

    embeddings = np.concatenate(feature_batches, axis=0)

    return embeddings


# Extract gallery embeddings
gallery_embeddings = extract_embeddings(gallery_df, "Gallery")

# Save gallery data
np.savez_compressed(
    GALLERY_OUTPUT,
    embeddings=gallery_embeddings,
    paths=gallery_df["path"].to_numpy(dtype=str),
    labels=gallery_df["label"].to_numpy(dtype=str)
)


# Extract query embeddings
query_embeddings = extract_embeddings(query_df, "Queries")

# Save query data
np.savez_compressed(
    QUERY_OUTPUT,
    embeddings=query_embeddings,
    paths=query_df["path"].to_numpy(dtype=str),
    labels=query_df["label"].to_numpy(dtype=str)
)


print("\nFull dataset feature extraction completed!")
print("Gallery embeddings:", gallery_embeddings.shape)
print("Query embeddings:", query_embeddings.shape)
print("Gallery saved to:", GALLERY_OUTPUT)
print("Queries saved to:", QUERY_OUTPUT)