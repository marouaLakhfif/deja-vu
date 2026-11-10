from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_PATH = (
    PROJECT_ROOT / "results" / "resnet18_test_embeddings.npz"
)

# Load the saved file
saved_data = np.load(EMBEDDINGS_PATH)

# Access each stored array
embeddings = saved_data["embeddings"]
paths = saved_data["paths"]
labels = saved_data["labels"]
splits = saved_data["splits"]

# Inspect the data
print("Available arrays:", saved_data.files)
print("Embeddings shape:", embeddings.shape)
print("Paths shape:", paths.shape)
print("Labels shape:", labels.shape)
print("Splits shape:", splits.shape)

print("\nFirst image:")
print("Path:", paths[0])
print("Label:", labels[0])
print("Split:", splits[0])

print("\nFirst 10 features:")
print(embeddings[0][:10])