# Week 1: Learning Notes

## Project: Déjà Vu
**Goal:** Explore chest X-ray images, understand the dataset, and eventually build an image retrieval system.

## Dataset facts

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

- The training dataset is imbalanced: approximately 74% PNEUMONIA and 26% NORMAL.
- The validation split is very small.
- Retrieval evaluation must account for class imbalance.

## Python lessons

### Lesson 1: Paths and folders
- `Path` represents a filesystem path.
- `.iterdir()` lists immediate contents, not nested contents.
- `.is_file()` checks whether a path is a file.
- `sum(1 for ...)` counts items satisfying a condition.
- `groupby(...).size()` counts rows in each group.

### Lesson 2: Lists and dictionaries
- `rows = []` creates an empty list.
- `rows.append({...})` adds a dictionary to the list.
- Dictionaries store information as key-value pairs.
- Each dictionary can represent one image's metadata.

### Lesson 3: Opening images
- `Image.open(img_path)` opens an image with Pillow.
- `img.size` returns `(width, height)`.
- `img.mode` describes the image's pixel mode.
- `"L"` means grayscale.
- `with` manages the image file's lifetime and closes it when the block ends.

### Lesson 4: DataFrames and filtering
- `pd.DataFrame(rows)` converts a list of dictionaries into a DataFrame.
- `img_path.stat().st_size` returns the file size in bytes.
- `df["label"] == "PNEUMONIA"` creates a Boolean condition.
- `df[df["label"] == "PNEUMONIA"]` selects matching rows.

### Lesson 5: Handling errors
- `try` contains code that might fail.
- `except Exception as e` catches an exception and gives access to the error.
- `str(e)` converts the error to a readable string.
- An `errors` list can record failed image operations while processing continues.

### Lesson 6: Grouping and exporting
- `df.groupby(["split", "label"]).size()` counts each split-label combination.
- `df["mode"].value_counts()` counts occurrences of each image mode.
- `df.to_csv(path, index=False)` exports a CSV without an extra index column.

### Lesson 7: Reshaping grouped data
- `.unstack(fill_value=0)` moves one grouping level into columns.
- `fill_value=0` fills missing combinations with zero.
- `.loc[["train", "val", "test"]]` selects rows in a specified order.

### Lesson 8: Plotting with Matplotlib
- `fig` represents the entire figure.
- `ax` represents a plotting area.
- `plt.subplots()` creates a figure and axes.
- `.plot(kind="bar", ax=ax)` draws a bar chart.
- `ax.set_title()` and `ax.set_xlabel()` / `ax.set_ylabel()` add labels.
- `ax.bar_label()` adds values to bars.
- `plt.tight_layout()` adjusts spacing.
- `plt.savefig()` saves the chart.
- `plt.close()` closes the figure; it does not delete the saved file.

### Lesson 9: Creating an image grid
- `plt.subplots(2, 4)` creates eight plotting areas.
- `enumerate()` provides an index and its corresponding item.
- `.sample(4, random_state=42)` selects four reproducible random rows.
- `.iterrows()` yields a row index and row data.
- `axes[row, col]` selects a specific subplot.

### Lesson 10: Displaying images in the grid
**Status: Not started yet.**

Topics to learn:
- Opening an image using its saved path.
- Converting images to grayscale.
- Displaying images with `imshow()`.
- Adding titles with labels and dimensions.
- Hiding axes and saving the finished grid.

## Next steps
1. Complete Lesson 10.
2. Inspect image dimensions, modes, unreadable files, and possible duplicates.
3. Build a simple image-retrieval baseline.
4. Evaluate retrieval results using Precision@5.
5. Compare with frozen pretrained CNN embeddings if time allows.

## Learning rule
For each lesson:
- Understand what each line does.
- Predict the result before running the code.
- Complete a short quiz.
- Record confusing concepts and mistakes.
