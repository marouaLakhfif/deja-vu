# Learning and Development Log

## Week 1: Dataset Exploration and Visualization

### Completed
- Counted images across train, test and validation splits.
- Built image metadata using Python, pathlib, Pandas and Pillow.
- Inspected image dimensions, pixel modes and file sizes.
- Practiced file handling and error handling.
- Exported dataset metadata and class counts.
- Generated a class-distribution chart.
- Created a sample-image grid showing NORMAL and PNEUMONIA images with their dimensions.
- Completed Lessons 1–10.

### Data quality findings
- 5,856 readable images.
- Zero missing or unreadable files.
- 30 groups of byte-identical images.
- No exact-hash overlap between the official training and test sets.

### Learning
Practiced Python paths, image processing, DataFrames, filtering, grouping, CSV export, error handling and Matplotlib visualization.

## Week 2: Image Retrieval Baseline

### Completed
- Established the training set as the retrieval gallery.
- Used the test set as retrieval queries.
- Converted images to grayscale and resized them to 64 × 64.
- Extracted flattened pixel features.
- Implemented Euclidean distance for nearest-neighbor retrieval.
- Retrieved the top five neighbors for each test image.
- Evaluated mean label agreement@5 separately by class.
- Compared raw pixel features with per-image normalized pixel features.

### Results

| Method | NORMAL | PNEUMONIA | Overall |
|---|---:|---:|---:|
| Raw pixels | 37.9% | 97.6% | 75.3% |
| Normalized pixels | 43.5% | 96.9% | 76.9% |

### Observations
Normalization improved label agreement for NORMAL queries by 5.6 percentage points and reduced it for PNEUMONIA queries by 0.7 percentage points.

The results show that the feature representation affects retrieval differently across the two classes.

### Limitations
- The dataset has an imbalanced class distribution.
- Pixel-based features are sensitive to image structure, cropping and alignment.
- Label agreement does not establish clinical similarity or diagnostic accuracy.
- Patient-level separation has not been verified.

## Week 3: Pretrained Image Embeddings

### Completed
- Explored pretrained ResNet-18 as a feature extractor.
- Replaced the final classification layer with an identity layer to extract 512-dimensional feature vectors.
- Used the pretrained model without additional training or fine-tuning.
- Extracted embeddings separately for the training gallery and test queries.
- Implemented Euclidean-distance nearest-neighbor retrieval using the extracted embeddings.
- Retrieved the five closest gallery images for each test query.
- Evaluated label agreement@5 by class and overall.
- Generated visual examples of retrieved neighbors.

### Results

| Method | NORMAL | PNEUMONIA | Overall |
|---|---:|---:|---:|
| Normalized pixels | 43.5% | 96.9% | 76.9% |
| ResNet-18 embeddings | 52.7% | 97.2% | 80.5% |

### Observations
ResNet-18 embeddings achieved higher overall label agreement than the normalized pixel baseline, increasing from 76.9% to 80.5%.

The improvement was more noticeable for NORMAL queries. The results suggest that pretrained image embeddings can provide a different and potentially more useful feature representation for this retrieval task.

### Limitations
- ResNet-18 was pretrained on ImageNet rather than medical images.
- The dataset has an imbalanced class distribution.
- Label agreement does not establish clinical similarity or diagnostic accuracy.
- Patient-level separation has not been verified.

## Week 4: Understanding and Improving the Retrieval System

### In Progress
- Reviewing the retrieval implementation step by step.
- Studying feature vectors and their relationship to image representations.
- Understanding Euclidean distance calculations.
- Learning how NumPy sorting selects the nearest neighbors.
- Reviewing how retrieval results are stored and evaluated using Pandas.
- Improving retrieval visualizations and comparison charts.

### Planned
- Finish reviewing the retrieval implementation.
- Inspect successful and unsuccessful retrieval examples.
- Compare retrieval methods through visualizations.
- Investigate aspect-ratio-preserving preprocessing if time permits.

## Learning Principles
- Understand the code and methods rather than only running them.
- Record completed work separately from planned work.
- Do not mark a task complete until it has actually been done.
- Prioritize a clear, manageable and reproducible project.