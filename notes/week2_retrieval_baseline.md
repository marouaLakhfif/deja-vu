# Week 2: Image Retrieval Baseline

## Objective
Implement and evaluate a simple image retrieval system using pixel-based features.

## Experimental setup
- Gallery: 5,216 training images
- Queries: 624 test images
- Image size: 64 × 64
- Color mode: Grayscale
- Feature representation: Flattened pixel arrays (4,096 features)
- Distance metric: Euclidean distance
- Number of retrieved neighbors: 5
- Exact-hash overlap between training and test sets: 0

## Experiments

### Experiment 1: Raw pixel features
Pixel intensities were scaled to the range [0, 1].

Results:
- NORMAL: 37.9%
- PNEUMONIA: 97.6%
- Overall: 75.3%

### Experiment 2: Normalized pixel features
Each image's pixel intensities were standardized using its own mean and standard deviation.

Results:
- NORMAL: 43.5%
- PNEUMONIA: 96.9%
- Overall: 76.9%

## Observations
Normalization improved mean label agreement for NORMAL queries by 5.6 percentage points. However, it slightly reduced agreement for PNEUMONIA queries by 0.7 percentage points.

The results indicate that normalization affects the two image classes differently.

## Limitations
- The dataset has an imbalanced class distribution.
- Pixel-based features are sensitive to image structure, cropping, and alignment.
- Label agreement does not measure clinical similarity or diagnostic accuracy.
- Patient-level separation has not been verified.

## Next steps
1. Visualize query images and their retrieved neighbors.
2. Investigate examples where retrieval performs poorly.
3. Explore pretrained image embeddings as a feature-extraction method.
4. Compare all methods using the same evaluation protocol.