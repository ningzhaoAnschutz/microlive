# FRAP Analyses

This folder contains notebooks and modules for **Fluorescence Recovery After Photobleaching (FRAP)** analysis.

## Installation

Ensure you have the `microlive` environment activated:

```bash
conda activate microlive
```

## Files

### Notebooks

| Notebook | Description |
|----------|-------------|
| `FRAP_processing.ipynb` | Main FRAP data processing pipeline |
| `FRAP_representative_images.ipynb` | Generate representative images and videos |
| `FRAP_interpreatation_final.ipynb` | Statistical analysis and trajectory plotting |

### Modules

| Module | Description |
|--------|-------------|
| `frap_utilities.py` | Consolidated plotting functions |
| `FRAP_line_command.py` | Command-line FRAP processing script |

## Usage

### Using the Plotting Module

Import the plotting functions in your notebook:

```python
from frap_utilities import (
    plot_images_frap_all_channels_representative,
    plot_kymograph,
    plot_merged_image,
    plot_combined_cell_and_kymograph,
    save_video_as_avi,
    compose_pngs,
    plot_FRAP_trajectories,
)
```

### Available Functions

#### `plot_images_frap_all_channels_representative()`

Plot representative frames for all color channels with optional mask-based cropping.

```python
plot_images_frap_all_channels_representative(
    image_TZXYC,                    # 5D image array (T, Z, Y, X, C)
    list_selected_frames=[0, 11, 40, 100, 139],
    cmap_list=[green_colormap, magenta_colormap],
    coordinates_roi=coordinates_roi,
    masks_TXY=masks_TXY,
    results_folder=results_folder,
)
```

#### `plot_kymograph()`

Generate kymograph from ROI coordinates across time.

```python
plot_kymograph(
    image_TZXYC,
    coordinates_roi,
    list_selected_frames,
    x_title_list=['0s', '10s', '30s'],
    length_kymograph_line=50,
    cmap_list=[green_colormap, magenta_colormap],
    results_folder=results_folder,
)
```

#### `plot_merged_image()`

Create merged RGB image with FRAP line overlay.

```python
plot_merged_image(
    image_TZXYC,
    coordinates_roi,
    length_kymograph_line=50,
    cmap_list_imagej=[green_colormap, magenta_colormap],
    masks_TXY=masks_TXY,
    gamma=0.6,
    clip_limit=0.001,
    results_folder=results_folder,
)
```

#### `plot_combined_cell_and_kymograph()`

Create publication-quality combined figure with cell images and kymographs.

```python
plot_combined_cell_and_kymograph(
    image_TZXYC,
    coordinates_roi,
    list_selected_frames,
    x_title_list,
    masks_TXY=masks_TXY,
    cmap_list=[green_colormap, magenta_colormap],
    y_label_list=['GFP', 'mCherry'],
    results_folder=results_folder,
)
```

#### `save_video_as_avi()`

Export multi-channel AVI video with consistent cropping.

```python
save_video_as_avi(
    image_TZXYC,
    avi_name=results_folder / 'frap_movie',
    frame_values=frame_values,
    masks_TXY=masks_TXY,
    cmap_list=[green_colormap, magenta_colormap],
    fps=5,
)
```

#### `compose_pngs()`

Combine two PNG images side-by-side.

```python
compose_pngs(
    png_path1='image1.png',
    png_path2='image2.png',
    output_png='combined.png',
    spacing=10,
)
```

#### `plot_FRAP_trajectories()`

Plot FRAP recovery trajectories with mean overlay.

```python
plot_FRAP_trajectories(
    df_list,                        # DataFrame with FRAP data
    selected_dataset='sfGFP',
    apply_min_max_normalization=True,
    display_cell_count=True,
    results_folder=results_folder,
)
```

## Workflow

### 1. Process FRAP Data

Run `FRAP_processing.ipynb` to:

- Load LIF files
- Segment nuclei with Cellpose
- Detect FRAP ROI
- Quantify fluorescence recovery

### 2. Generate Representative Images

Run `FRAP_representative_images.ipynb` to:

- Create time-course montages
- Generate kymographs
- Export AVI videos
- Create publication figures

### 3. Analyze Results

Run `FRAP_interpreatation_final.ipynb` to:

- Plot FRAP trajectories
- Calculate statistics
- Compare conditions

## Dependencies

- `microlive>=1.0.17` - Microscopy analysis library
- `matplotlib` - Plotting
- `numpy` - Array operations
- `scipy` - Image processing
- `opencv-python` - Video export
- `PIL` - Image manipulation

---

*Authors: Rhiannon Sears, Luis Aguilera*
