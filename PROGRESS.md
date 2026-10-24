## Learning and Development Log

### Week 1: Dataset Exploration and Visualization

**Project work completed**
- Counted images across dataset splits and labels.
- Built image metadata using Python, pathlib, pandas and Pillow.
- Inspected image dimensions, pixel modes and file sizes.
- Practiced handling unreadable images and recording errors.
- Grouped dataset counts by split and label.
- Exported metadata and dataset counts.
- Generated a class-distribution chart and a sample-grid visualization.

**Learning progress**
- Lessons 1–9 covered file handling, metadata collection, image properties, DataFrames, filtering, error handling, grouping, CSV export and Matplotlib.
- Detailed explanations and examples are being collected in `notes/week1_learning_notes.md`.
- Lesson 10, displaying and annotating images in the grid, remains in progress.

**Next steps**
1. Complete Lesson 10 and understand each line of the image-grid code.
2. Review the dataset for data-quality issues and possible duplicates.
3. Build a simple image-retrieval baseline using pixel features and cosine similarity.
4. Evaluate retrieval results with Precision@5 and inspect failure cases.
5. Explore frozen pretrained CNN embeddings if time permits.

**Documentation principle:** Record completed work separately from planned work. Do not mark a task complete until it has actually been done.

## Week 1: Dataset Exploration and Visualization

### Completed
- Explored the dataset's train, validation, and test splits.
- Counted images by split and class using Pandas.
- Collected image metadata, including dimensions, modes, and file sizes.
- Practiced handling errors during image processing.
- Exported metadata to CSV and class counts to a text file.
- Created a class-distribution chart.
- Created a grid of sample X-ray images for visual inspection.
- Studied Python paths, lists, dictionaries, Pillow, Pandas, Matplotlib, and image visualization.

### Key observation
The dataset is imbalanced, with 3,875 PNEUMONIA images and 1,341 NORMAL images in the training split. The validation split contains only 16 images, so its suitability for evaluation is limited.

### Next steps
- Inspect image dimensions, modes, unreadable files, and possible duplicates.
- Build a simple image-retrieval baseline.
- Evaluate retrieval with Precision@5.
- Consider comparing pixel features with frozen pretrained CNN embeddings.

### Learning status
Lessons 1–10 completed. Continue learning by understanding each code section and testing it rather than copying code without explanation.
