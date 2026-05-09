# LIF Export Tools

Batch export and filter tools for converting Leica LIF microscopy files into OME-TIFF and AVI formats.

---

## Files

| File | Purpose |
|---|---|
| `export_lif_data.py` | Core library — all export and filter functions |
| `export_lif_batch.ipynb` | Notebook — batch export TIF + AVI from a folder of LIF files |
| `filter_exported_data.ipynb` | Notebook — keep only exported files whose series were processed in the GUI |

---

## Quick Start

### 1. Batch Export

Open `export_lif_batch.ipynb`, set the `data_folder` path, and run all cells:

```python
from export_lif_data import batch_export

batch_export(
    input_folder="/path/to/lif_files",
    output_folder="/path/to/lif_files/exported_data",
    export_tif=True,
    export_video=True,
    show_timestamp=True,
    show_scalebar=True,
)
```

### 2. Filter by Processed Results

Open `filter_exported_data.ipynb`, set the paths, and run:

```python
from export_lif_data import filter_exported_data

filter_exported_data(
    exported_folder="/path/to/exported_data/AlfaTag",
    results_folder="/path/to/AlfaTag/HT_Analysis_GUI",
    output_folder="/path/to/exported_filtered/AlfaTag",
    dry_run=False,
)
```

---

## Output Folder Structure

```
exported_data/
├── <subfolder>/                        # mirrors input structure
│   ├── <lif_name>/
│   │   ├── tif/
│   │   │   ├── <lif_name>_<series>.ome.tif        # full Z-stack
│   │   │   └── <lif_name>_<series>_maxZ.ome.tif   # max projection (if enabled)
│   │   └── avi/
│   │       └── <lif_name>_<series>.avi
```

---

## Functions Reference

### `batch_export()`

Iterates through a folder (recursively), finds all `.lif` files, and exports TIF + AVI for every series.

| Parameter | Default | Description |
|---|---|---|
| `input_folder` | *required* | Path to folder containing `.lif` files |
| `output_folder` | *required* | Root output directory |
| `export_tif` | `True` | Export OME-TIFF for each series |
| `export_video` | `True` | Export AVI video for each series |
| `show_timestamp` | `False` | Overlay timestamp on video frames |
| `show_scalebar` | `False` | Overlay scale bar on video frames |
| `fps` | `10` | Video frame rate |
| `channel` | `0` | Channel index for video rendering |
| `min_percentile` | `0.1` | Min percentile for intensity rescaling (AVI only) |
| `max_percentile` | `99.9` | Max percentile for intensity rescaling (AVI only) |
| `sigma` | `0.7` | Smoothing Gaussian sigma (AVI only, 0 = disabled) |
| `low_sigma` | `0.15` | Low-pass Gaussian sigma (AVI only, 0 = disabled) |
| `dpi` | `150` | Rendering DPI for video |
| `recursive` | `True` | Search subdirectories for LIF files |
| `max_projection` | `False` | Export max-Z projection TIF instead of full Z-stack |

### `filter_exported_data()`

Copies only exported files whose series have a matching `results_*` folder in the GUI analysis output.

| Parameter | Default | Description |
|---|---|---|
| `exported_folder` | *required* | Path to batch-exported data |
| `results_folder` | *required* | Path to GUI analysis output with `results_*` folders |
| `output_folder` | `None` | Destination (defaults to `exported_filtered/` sibling) |
| `results_prefix` | `"results_"` | Prefix for results folders |
| `dry_run` | `False` | If `True`, preview without copying files |

**Returns:** `(kept, skipped)` file counts.

---

## TIF Export Details

- **Raw data**: No rescaling, no clipping, no filters applied
- **Format**: OME-TIFF with full metadata (pixel sizes, time interval, channels)
- **Shape**: `(T, C, Z, Y, X)` — standard OME axis order
- **Max projection**: When `max_projection=True`, Z is collapsed via `np.max` (Z=1). Filenames get a `_maxZ` suffix

## AVI Export Details

The AVI rendering pipeline matches the MicroLive GUI display settings:

1. **Max-Z projection** per frame
2. **Percentile rescaling** (default: 0.1–99.9)
3. **Low-pass Gaussian filter** (default σ=0.15)
4. **Smoothing Gaussian filter** (default σ=0.7)
5. **Grayscale colormap** rendering
6. **Optional overlays**: timestamp (upper-left) and scale bar (lower-right)

---

## Filter Matching Logic

The filter matches exported filenames against `results_*` folder names:

```
results_<lif_stem>_<series_name>     →  stem = <lif_stem>_<series_name>
<lif_stem>_<series_name>.ome.tif     →  stem = <lif_stem>_<series_name>     ✓ match
<lif_stem>_<series_name>_maxZ.ome.tif → stem = <lif_stem>_<series_name>     ✓ match (_maxZ stripped)
<lif_stem>_<series_name>.avi         →  stem = <lif_stem>_<series_name>     ✓ match
```

---

## CLI Usage

The script can also be run from the command line:

```bash
# Full export with overlays:
python export_lif_data.py \
    --input_folder /data/lif_files \
    --output_folder /data/exports \
    --show_timestamp --show_scalebar --fps 15

# Max-Z projection TIF only:
python export_lif_data.py \
    --input_folder /data/lif_files \
    --output_folder /data/exports \
    --no_video --max_projection

# AVI only, custom percentiles:
python export_lif_data.py \
    --input_folder /data/lif_files \
    --output_folder /data/exports \
    --no_tif --min_percentile 0.5 --max_percentile 99.9
```

---

## Notes

- **macOS resource forks** (`._*.lif`) are automatically filtered out
- **String metadata**: LIF `CycleTime` values that are strings (e.g., `'60'`) are safely cast to `float`
- **Dependencies**: Requires `microlive`, `tifffile`, `opencv-python`, `scipy`, `matplotlib`, `matplotlib-scalebar`
- **Environment**: Must be run in the `microlive` conda environment
