# %% [markdown]
# # 🔬 Multi-Channel Montage — Run
#
# Interactive viewer and export for multi-channel LIF montages.
# All code lives in `montage.py` — this file is only user inputs.
#
# **Workflow:** run each cell top-to-bottom.

# %%
import sys
from pathlib import Path

# Ensure montage.py is importable from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from montage import (
    read_lif_metadata, print_series_table, load_series,
    resolve_config, launch_viewer, export_from_viewer,
    export_all_from_viewer,
)

# %% [markdown]
# ---
# ## 1. LIF File Path

# %%
lif_path = "/Volumes/LaCie/Final coTFT LIF Files/20250827 pRS032_012_JF646 6hr Tfx.lif"

meta = read_lif_metadata(lif_path)
print_series_table(meta, lif_path)

# %% [markdown]
# ---
# ## 2. Select Series & Options
#
# Set `SERIES_INDEX` to a number to load one series, or `None` to
# batch-export **all** series (skips the interactive viewer).

# %%[100, 400, 50, 350]
SERIES_INDEX  = None               # ← series index, or None = all series
channel_order = None            # ← None = natural, or e.g. [1, 0]
panel_titles  = ["Folding Spots", "Nascent Chain Spots", "Merge"]  # ← None = metadata names

# Crop ROI in pixels, or None = full image.
# Format: [x_min, x_max, y_min, y_max]
#          x = columns (left→right), y = rows (top→bottom)
# Example for a 512×512 image:
#   CROP = [100, 400, 50, 350]
#          ↑     ↑    ↑    ↑
#          x_min x_max y_min y_max  → crops to 300 wide × 300 tall
# Percentiles are computed on the cropped region only.
CROP = [50, 512, 30, 512]

if SERIES_INDEX is not None:
    data = load_series(meta, SERIES_INDEX)
    channel_order, panel_titles, n_panels = resolve_config(
        data, channel_order, panel_titles)
else:
    print(f"⏩ All {len(meta['list_names'])} series will be exported in §4.")

# %% [markdown]
# ---
# ## 3. Interactive Viewer
# *(Skipped automatically when SERIES_INDEX is None)*

# %%
if SERIES_INDEX is not None:
    wdg = launch_viewer(data, channel_order, panel_titles, n_panels, crop=CROP)
else:
    wdg = None
    print("⏩ Viewer skipped — batch mode (SERIES_INDEX = None)")

# %% [markdown]
# ---
# ## 4. Export
# Uses the current widget settings (colors, sigma, percentiles, overlays).
#
# When `SERIES_INDEX = None`, exports all series with the default
# settings specified below.

# %%
OUTPUT_FOLDER      = str(Path(lif_path).parent / "montage_export")
EXPORT_AVI         = True
EXPORT_TIF         = False
SHOW_SCALEBAR      = True
SHOW_TIMESTAMP     = True
FPS                = 10
DPI                = 150
PANEL_WIDTH_INCHES = 3.0

# Per-channel contrast: float (same for all) or list (one per channel).
# Example: MIN_PERCENTILE = [0.1, 0.5]   → ch0 at 0.1%, ch1 at 0.5%
#          MAX_PERCENTILE = [99.95, 99.5]
MIN_PERCENTILE = [0.001, 0.05]
MAX_PERCENTILE = [99.95, 99.9]

if SERIES_INDEX is not None:
    # ── Single series: override widgets with the flags above ──
    wdg['w_scalebar'].value  = SHOW_SCALEBAR
    wdg['w_timestamp'].value = SHOW_TIMESTAMP
    # Override per-channel percentile widgets
    _min_list = (MIN_PERCENTILE if isinstance(MIN_PERCENTILE, list)
                 else [MIN_PERCENTILE] * len(wdg['w_min_pcts']))
    _max_list = (MAX_PERCENTILE if isinstance(MAX_PERCENTILE, list)
                 else [MAX_PERCENTILE] * len(wdg['w_max_pcts']))
    for w, v in zip(wdg['w_min_pcts'], _min_list):
        w.value = v
    for w, v in zip(wdg['w_max_pcts'], _max_list):
        w.value = v
    export_from_viewer(
        data, lif_path, channel_order, panel_titles, n_panels, wdg,
        output_folder=OUTPUT_FOLDER,
        export_avi=EXPORT_AVI,
        export_tif=EXPORT_TIF,
        fps=FPS,
        dpi=DPI,
        panel_width_inches=PANEL_WIDTH_INCHES,
        crop=CROP,
    )
else:
    # ── All series: batch export ──
    export_all_from_viewer(
        meta=meta,
        lif_path=lif_path,
        channel_order=channel_order,
        panel_titles=panel_titles,
        output_folder=OUTPUT_FOLDER,
        export_avi=EXPORT_AVI,
        export_tif=EXPORT_TIF,
        show_scalebar=SHOW_SCALEBAR,
        show_timestamp=SHOW_TIMESTAMP,
        fps=FPS,
        dpi=DPI,
        panel_width_inches=PANEL_WIDTH_INCHES,
        min_percentile=MIN_PERCENTILE,
        max_percentile=MAX_PERCENTILE,
        crop=CROP,
    )
