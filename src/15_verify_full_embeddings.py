from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GALLERY_PATH = PROJECT_ROOT / "results" / "resnet18_gallery_embeddings.npz"
QUERY_PATH = PROJECT_ROOT / "results" / "resnet18_query_embeddings.npz"

gallery = np.load(GALLERY_PATH)
queries = np.load(QUERY_PATH)

print("GALLERY")
print("Embeddings:", gallery["embeddings"].shape)
print("Paths:", gallery["paths"].shape)
print("Labels:", gallery["labels"].shape)

print("\nQUERIES")
print("Embeddings:", queries["embeddings"].shape)
print("Paths:", queries["paths"].shape)
print("Labels:", queries["labels"].shape)

print("\nGallery label distribution:")
print({
    label: int(np.sum(gallery["labels"] == label))
    for label in np.unique(gallery["labels"])
})

print("\nQuery label distribution:")
print({
    label: int(np.sum(queries["labels"] == label))
    for label in np.unique(queries["labels"])
})