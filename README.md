# Chest X-Ray Image Retrieval

## Project Overview

This project explores how Python-based image processing and visualization can support image retrieval using a chest X-ray dataset.

The project begins with a pixel-based retrieval baseline and investigates how different feature representations affect retrieval results.

It compares raw pixel features, normalized pixel features and pretrained ResNet-18 image embeddings.

The focus is on image processing, data handling, retrieval and visualization rather than training a deep learning model from scratch.

## Dataset

The dataset contains 5,856 images divided into three splits:

| Split | Images |
|---|---:|
| Train | 5,216 |
| Test | 624 |
| Validation | 16 |
| **Total** | **5,856** |

The training set contains:
- NORMAL: 1,341 images (25.7%)
- PNEUMONIA: 3,875 images (74.3%)

The validation split contains only 8 images per class, so the official test set is used for the retrieval experiments.

## Week 1: Dataset Exploration

- Counted images by split and class.
- Built an image metadata CSV using Python and Pandas.
- Collected image dimensions, pixel modes and file sizes.
- Checked image readability and missing files.
- Identified groups of byte-identical images.
- Created a class-distribution chart and sample-image visualization.

### Data quality findings
- 5,856 readable images.
- Zero missing or unreadable files.
- 30 groups of byte-identical images.
- Zero exact-hash overlap between the official training and test sets.

## Week 2: Pixel-Based Image Retrieval

The retrieval system uses:
- Grayscale images resized to 64 × 64 pixels.
- Flattened pixel arrays containing 4,096 features.
- Euclidean distance to identify the five nearest neighbors.
- Training images as the gallery and test images as queries.

Two feature representations were evaluated:

| Method | NORMAL | PNEUMONIA | Overall |
|---|---:|---:|---:|
| Raw pixels | 37.9% | 97.6% | 75.3% |
| Normalized pixels | 43.5% | 96.9% | 76.9% |

## Week 3: Pretrained ResNet-18 Embeddings

A pretrained ResNet-18 model was explored as a feature extractor.

- The final classification layer was replaced with an identity layer.
- Each image was represented by a 512-dimensional feature vector.
- The pretrained model was used without additional training or fine-tuning.
- Euclidean distance was used to retrieve the five nearest gallery images for each test query.

### Results comparison

| Method | NORMAL | PNEUMONIA | Overall |
|---|---:|---:|---:|
| Raw pixels | 37.9% | 97.6% | 75.3% |
| Normalized pixels | 43.5% | 96.9% | 76.9% |
| ResNet-18 embeddings | 52.7% | 97.2% | 80.5% |

ResNet-18 embeddings achieved the highest overall label agreement among the three evaluated methods.

## Evaluation Metric

The evaluation metric is **mean label agreement@5**: the proportion of the five retrieved images that share the query's class label, averaged across queries.

All three methods use:
- The same training gallery.
- The same test queries.
- Euclidean distance.
- The five nearest neighbors.

The methods differ in their feature-extraction pipelines.

These results measure label agreement, not clinical similarity or diagnostic accuracy.

## Current Limitations

- The dataset has an imbalanced class distribution.
- Pixel-based features are sensitive to image brightness, cropping and alignment.
- ResNet-18 was pretrained on ImageNet rather than medical images.
- Patient-level separation has not been verified.
- Zero exact-hash overlap between the official training and test sets does not establish patient-level separation.

## Project Structure

```text
DejaVu/
├── data/
│   └── raw/
│       └── chest_xray/
├── notes/
├── notebooks/
├── results/
├── src/
├── .gitignore
├── README.md
└── PROGRESS.md
```

## Current Work

Week 4 focuses on:
- Understanding the retrieval implementation.
- Reviewing feature vectors, Euclidean distance and nearest-neighbor selection.
- Improving retrieval visualizations and comparison charts.
- Exploring aspect-ratio-preserving preprocessing if time permits.

## Project Direction

The project prioritizes Python-based data processing, image retrieval and visualization. Pretrained image embeddings are used as a comparison experiment, without training or fine-tuning a deep learning model.