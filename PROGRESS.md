# Project Progress

## Déjà Vu: Exploring Image Embeddings for Chest X-Ray Retrieval and Visualization

### Week 1: Dataset Exploration and Visualization

**Completed**
- Counted images by dataset split and label using `pathlib`.
- Built a metadata table with `pandas`.
- Extracted image dimensions, pixel modes, and file sizes using Pillow.
- Handled unreadable images with `try` and `except`.
- Exported metadata to CSV.
- Created a class-distribution bar chart with Matplotlib.
- Created a sample grid of NORMAL and PNEUMONIA images.

**Concepts studied**
- File paths and directory iteration
- Lists, dictionaries, and `append()`
- Image properties and file handling
- Pandas DataFrames, filtering, and `groupby()`
- CSV export
- Matplotlib figures, axes, and image grids

**Current learning point**

Understanding how images are opened, converted to grayscale, placed in a grid, and displayed with Matplotlib.

**Next steps**
- Finish understanding the image-grid visualization code.
- Check image metadata for duplicates and data-quality issues.
- Build a simple image-retrieval baseline.
- Explore pretrained CNN embeddings if time permits.

