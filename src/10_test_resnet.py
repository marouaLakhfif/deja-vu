import torch
from torchvision import models

# Load pretrained ResNet-18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Remove the classification layer
model.fc = torch.nn.Identity()

# Set evaluation mode
model.eval()

print("ResNet-18 loaded successfully!")
print("Feature extraction mode enabled.")
print("Output features:", model.fc)