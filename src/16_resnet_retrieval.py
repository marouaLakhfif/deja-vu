from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

GALLERY_PATH = PROJECT_ROOT / "results" / "resnet18_gallery_embeddings.npz"
QUERY_PATH = PROJECT_ROOT / "results" / "resnet18_query_embeddings.npz"

RESULTS_PATH = PROJECT_ROOT / "results" / "resnet18_retrieval_results.csv"
METRICS_PATH = PROJECT_ROOT / "results" / "resnet18_retrieval_metrics.csv"


# Load saved embeddings
gallery = np.load(GALLERY_PATH)
queries = np.load(QUERY_PATH)

gallery_embeddings = gallery["embeddings"]
gallery_paths = gallery["paths"]
gallery_labels = gallery["labels"]

query_embeddings = queries["embeddings"]
query_paths = queries["paths"]
query_labels = queries["labels"]


# Retrieval settings
TOP_K = 5

all_results = []


# Process each query image
for i, query_vector in enumerate(query_embeddings):

    # Calculate distances to all gallery images
    distances = np.linalg.norm(
        gallery_embeddings - query_vector,
        axis=1
    )

    # Find the five closest images
    nearest_indices = np.argsort(distances)[:TOP_K]

    # Save the retrieved neighbors
    for rank, gallery_index in enumerate(nearest_indices, start=1):

        all_results.append({
            "query_path": query_paths[i],
            "query_label": query_labels[i],
            "path": gallery_paths[gallery_index],
            "label": gallery_labels[gallery_index],
            "rank": rank,
            "distance": distances[gallery_index],
            "label_match": (
                query_labels[i] == gallery_labels[gallery_index]
            )
        })


# Convert results into a DataFrame
results_df = pd.DataFrame(all_results)

results_df.to_csv(RESULTS_PATH, index=False)


# Calculate label agreement@5
metrics = []

for label in np.unique(query_labels):

    label_results = results_df[
        results_df["query_label"] == label
    ]

    agreement = label_results["label_match"].mean()

    metrics.append({
        "label": label,
        "agreement_at_5": agreement,
        "percentage": agreement * 100
    })


overall_agreement = results_df["label_match"].mean()

metrics.append({
    "label": "OVERALL",
    "agreement_at_5": overall_agreement,
    "percentage": overall_agreement * 100
})


metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(METRICS_PATH, index=False)


# Display results
print("ResNet retrieval completed!")
print("Number of queries:", len(query_embeddings))
print("Gallery size:", len(gallery_embeddings))
print("Results shape:", results_df.shape)

print("\nLabel agreement@5:")
print(metrics_df.to_string(index=False))

print("\nSaved results to:", RESULTS_PATH)
print("Saved metrics to:", METRICS_PATH)