from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
    / "test"
    / "NORMAL"
    / "IM-0001-0001.jpeg"
)


# ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Open and convert the image
with Image.open(IMAGE_PATH) as img:
    image = img.convert("RGB")

    # Apply preprocessing
    tensor = preprocess(image)

# Add batch dimension
tensor = tensor.unsqueeze(0)

print("Image preprocessing successful!")
print("Tensor shape:", tensor.shape)
print("Tensor data type:", tensor.dtype)
print("Minimum value:", tensor.min().item())
print("Maximum value:", tensor.max().item())

from torchvision import models

# Load the pretrained ResNet-18 model.
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Remove the final classification layer.
model.fc = torch.nn.Identity()

# Set the model to evaluation mode.
model.eval()

# Extract image features without calculating gradients.
with torch.no_grad():
    features = model(tensor)

print("Feature extraction successful!")
print("Feature shape:", features.shape)