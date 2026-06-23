#!/usr/bin/env python3
"""
run_filter_and_rename.py
========================
Read LIF files, filter series by checking which ones were processed in
the GUI (i.e. have a ``tracking_*.csv`` in the corresponding
``results_*`` folder), and export only those series as sequentially
renamed TIF and optionally AVI files.

Workflow
--------
  1. Scan ``results_folder`` for ``results_*`` directories that contain
     a ``tracking_*.csv`` file → build the set of "processed" stems.
  2. Read each ``.lif`` file in ``lif_folder`` using ``mi.ReadLif``.
  3. For each series inside a LIF, compute the stem as
     ``{lif_stem}_{series_name}`` (matching the GUI convention from
     ``get_default_export_filename``).
  4. If the stem is in the processed set → export as TIF (and
     optionally AVI) with sequential naming
     (e.g. ``UTag_HT_001.ome.tif``).
  5. Write an ``export_manifest.csv`` mapping table.

Usage
-----
    1. Edit the configuration section below with your paths and prefixes.
    2. Run:  python run_filter_and_rename.py
    3. First run is a dry run (preview). Set DRY_RUN = False to export.

Author: Auto-generated from MicroLive export pipeline.
"""

import sys
import os
import re
import logging
from pathlib import Path

# Auto-relaunch in microlive conda environment if not already active
import platform
if platform.system() == 'Windows':
    # Auto-detect microlive environment in common Windows installation paths
    home = os.path.expanduser('~')
    common_paths = [
        os.path.join(home, 'anaconda3', 'envs', 'microlive'),
        os.path.join(home, 'miniconda3', 'envs', 'microlive'),
        os.path.join(home, 'miniforge3', 'envs', 'microlive'),
        os.path.join(home, 'AppData', 'Local', 'anaconda3', 'envs', 'microlive'),
        os.path.join(home, 'AppData', 'Local', 'miniconda3', 'envs', 'microlive'),
        r'C:\ProgramData\anaconda3\envs\microlive',
        r'C:\ProgramData\miniconda3\envs\microlive'
    ]
    MICROLIVE_ENV = next((p for p in common_paths if os.path.exists(os.path.join(p, 'python.exe'))), None)
    if not MICROLIVE_ENV:
        MICROLIVE_ENV = os.path.join(home, 'anaconda3', 'envs', 'microlive') # fallback
    _python_exe = os.path.join(MICROLIVE_ENV, 'python.exe')
else:
    MICROLIVE_ENV = '/opt/anaconda3/envs/microlive'
    _python_exe = os.path.join(MICROLIVE_ENV, 'bin', 'python')

if sys.prefix != MICROLIVE_ENV:
    if os.path.exists(_python_exe):
        print(f"Relaunching script using the microlive environment Python...\n")
        if platform.system() == 'Windows':
            os.execv(_python_exe, [_python_exe] + sys.argv)
        else:
            os.execl(_python_exe, _python_exe, *sys.argv)
    else:
        print(f"Warning: microlive environment not found at {MICROLIVE_ENV}.")
        print(f"  Expected Python at: {_python_exe}")
        print(f"  Run: conda activate microlive && python -c \"import sys; print(sys.prefix)\"")
        print(f"  Then update MICROLIVE_ENV in this script.\n")

# Ensure export_lif_data is importable from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import microlive.microscopy as mi
from export_lif_data import export_ome_tif, export_avi, _sanitize, _results_stem

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION — Edit these paths and options
# ============================================================

# Base directory containing all tag data
# BASE = "/Volumes/LaCie/UTag_paper_data/Harringtonine"
BASE = "/Volumes/LaCie/test_export/UTag"

# Define each tag with its LIF folder, results folder, and rename prefix.
TAG_CONFIGS = {
    "UTag": {
        "lif_folder":      f"{BASE}/LIFs",
        "results_folder":  f"{BASE}/HT_Analysis_GUI",
        "rename_prefix":   "UTag_HT",
    },
}

# Output directory for filtered/renamed files (per tag)
OUTPUT_BASE = f"{BASE}/exported_filtered"

# Sequential numbering options
START_INDEX = 1       # First sequential number
ZERO_PAD    = 3       # Digits: 3 → 001, 002, ...;  4 → 0001, 0002, ...

# Export options
EXPORT_TIF   = True   # Export OME-TIFF for each matched series
EXPORT_VIDEO = True   # Export AVI video for each matched series

# Max-Z projection: set True to collapse Z-stacks during export.
MAX_PROJECTION = True

# Video options (only used when EXPORT_VIDEO = True)
VIDEO_FPS           = 10
VIDEO_CHANNEL       = 0
VIDEO_SHOW_TIMESTAMP = False
VIDEO_SHOW_SCALEBAR  = False
VIDEO_MIN_PERCENTILE = 0.1
VIDEO_MAX_PERCENTILE = 99.9
VIDEO_SIGMA          = 0.7
VIDEO_LOW_SIGMA      = 0.15
VIDEO_DPI            = 150

# Dry run: set True to preview without exporting files.
# Set False to actually export and write the manifest.
DRY_RUN = False

# ============================================================
# END OF CONFIGURATION
# ============================================================


def _build_processed_stems(results_folder, results_prefix="results_"):
    """Scan results_folder for results_* dirs containing tracking_*.csv.

    Returns a set of stems (folder name minus the results_ prefix) that
    have been processed (i.e. have tracking data).
    """
    results_path = Path(results_folder)
    processed = set()

    if not results_path.is_dir():
        log.warning("Results folder not found: %s", results_path)
        return processed

    for d in sorted(results_path.iterdir()):
        if not d.is_dir() or not d.name.startswith(results_prefix):
            continue
        stem = d.name[len(results_prefix):]
        # Check if this folder contains a tracking_*.csv
        tracking_files = list(d.glob("tracking_*.csv"))
        if tracking_files:
            processed.add(stem)
            log.debug("  ✓ Processed: %s  (tracking: %s)", stem, tracking_files[0].name)
        else:
            log.debug("  ✗ No tracking CSV: %s", stem)

    return processed


def _process_tag(tag_name, cfg, output_base, start_index, zero_pad,
                 export_tif, export_video, max_projection, dry_run,
                 video_kwargs):
    """Process a single tag: read LIFs, filter by tracking, export & rename."""
    lif_folder_path = Path(cfg["lif_folder"])
    results_folder = cfg["results_folder"]
    rename_prefix = cfg.get("rename_prefix", tag_name)
    output_folder = Path(output_base) / tag_name

    if not lif_folder_path.is_dir():
        log.error("LIF folder not found: %s", lif_folder_path)
        return 0, 0, None

    # Step 1: Build set of processed stems
    processed_stems = _build_processed_stems(results_folder)
    if not processed_stems:
        log.warning("No processed results found in %s", results_folder)
        return 0, 0, None

    log.info("=" * 60)
    log.info("Filtered LIF Export: %s", tag_name)
    log.info("  LIF folder    : %s", lif_folder_path)
    log.info("  Results folder : %s", results_folder)
    log.info("  Output folder  : %s", output_folder)
    log.info("  Processed stems: %d", len(processed_stems))
    log.info("  Rename prefix  : %s", rename_prefix)
    log.info("  Export TIF     : %s", export_tif)
    log.info("  Export Video   : %s", export_video)
    log.info("  Max-Z proj     : %s", max_projection)
    log.info("  Dry run        : %s", dry_run)
    log.info("=" * 60)

    # Step 2: Find all LIF files
    lif_files = sorted(
        f for f in lif_folder_path.glob("*.lif")
        if not f.name.startswith('._')
    )
    if not lif_files:
        log.warning("No .lif files found in %s", lif_folder_path)
        return 0, 0, None

    log.info("  Found %d LIF file(s)", len(lif_files))

    kept = 0
    skipped = 0
    seq_counter = start_index
    manifest_records = []

    # Step 3: Process each LIF file
    for lif_file in lif_files:
        log.info("\n  📂 Reading %s …", lif_file.name)
        try:
            reader = mi.ReadLif(
                path=str(lif_file),
                show_metadata=False,
                save_tif=False,
                save_png=False,
                format='TZYXC',
                lazy=True,
            )
            (
                _list_images,
                list_names,
                pixel_xy_um,
                pixel_z_um,
                channel_names,
                _num_channels,
                list_time_intervals,
                bit_depth,
                _laser_lines,
                _intensities,
                _wave_ranges,
            ) = reader.read()
        except Exception as exc:
            log.error("  ❌ Failed to read %s: %s", lif_file.name, exc)
            continue

        voxel_yx_nm = pixel_xy_um * 1000.0
        voxel_z_nm = pixel_z_um * 1000.0

        # Derive the LIF stem the same way the GUI does
        lif_stem = _results_stem(lif_file.name)

        num_scenes = len(list_names)
        log.info("    %d series found.  voxel YX=%.1f nm  Z=%.1f nm",
                 num_scenes, voxel_yx_nm, voxel_z_nm)

        for idx in range(num_scenes):
            series_name = _sanitize(list_names[idx])
            # Build the stem exactly as the GUI does:
            #   {lif_filename_without_extension}_{series_name}
            stem = f"{lif_stem}_{series_name}"

            try:
                time_interval = float(list_time_intervals[idx])
            except (TypeError, ValueError):
                time_interval = 0.0

            if stem in processed_stems:
                seq_str = str(seq_counter).zfill(zero_pad)
                new_base = f"{rename_prefix}_{seq_str}"

                log.info("    ✓ KEEP  [%d/%d] %s → %s",
                         idx + 1, num_scenes, stem, new_base)

                if not dry_run:
                    # Read the scene data
                    image_stack = reader.read_scene(idx)  # (T, Z, Y, X, C)

                    # Export TIF
                    if export_tif:
                        tif_folder = output_folder / "tif"
                        tif_folder.mkdir(parents=True, exist_ok=True)
                        tif_suffix = "_maxZ" if max_projection else ""
                        tif_name = f"{new_base}{tif_suffix}.ome.tif"
                        tif_path = tif_folder / tif_name
                        try:
                            export_ome_tif(
                                image_stack=image_stack,
                                out_path=tif_path,
                                voxel_yx_nm=voxel_yx_nm,
                                voxel_z_nm=voxel_z_nm,
                                time_interval=time_interval,
                                bit_depth=bit_depth,
                                channel_names=channel_names,
                                max_projection=max_projection,
                            )
                        except Exception as exc:
                            log.error("      TIF export failed: %s", exc)

                    # Export AVI
                    if export_video:
                        avi_folder = output_folder / "avi"
                        avi_folder.mkdir(parents=True, exist_ok=True)
                        avi_name = f"{new_base}.avi"
                        avi_path = avi_folder / avi_name
                        try:
                            export_avi(
                                image_stack=image_stack,
                                out_path=avi_path,
                                voxel_yx_nm=voxel_yx_nm,
                                time_interval=time_interval,
                                **video_kwargs,
                            )
                        except Exception as exc:
                            log.error("      AVI export failed: %s", exc)

                manifest_records.append({
                    'Seq': seq_counter,
                    'New_Name': new_base,
                    'Original_Stem': stem,
                    'LIF_File': lif_file.name,
                    'Series_Name': series_name,
                    'Series_Index': idx,
                })

                seq_counter += 1
                kept += 1
            else:
                log.info("    ✗ SKIP  [%d/%d] %s  (no tracking data)",
                         idx + 1, num_scenes, stem)
                skipped += 1

    # Step 4: Build manifest DataFrame
    manifest_df = None
    if manifest_records:
        manifest_df = pd.DataFrame(manifest_records)
        if not dry_run:
            output_folder.mkdir(parents=True, exist_ok=True)
            manifest_path = output_folder / "export_manifest.csv"
            manifest_df.to_csv(manifest_path, index=False)
            log.info("  Manifest saved to: %s", manifest_path)

    log.info("-" * 60)
    log.info("  Kept: %d  |  Skipped: %d  |  Total: %d",
             kept, skipped, kept + skipped)
    log.info("=" * 60)

    return kept, skipped, manifest_df


def main():
    print("=" * 70)
    print("  Filtered LIF Export & Sequential Rename")
    print(f"  Dry Run: {DRY_RUN}")
    print(f"  Export TIF: {EXPORT_TIF}")
    print(f"  Export AVI: {EXPORT_VIDEO}")
    print(f"  Max-Z Projection: {MAX_PROJECTION}")
    print("=" * 70)

    video_kwargs = dict(
        show_timestamp=VIDEO_SHOW_TIMESTAMP,
        show_scalebar=VIDEO_SHOW_SCALEBAR,
        fps=VIDEO_FPS,
        channel=VIDEO_CHANNEL,
        min_percentile=VIDEO_MIN_PERCENTILE,
        max_percentile=VIDEO_MAX_PERCENTILE,
        sigma=VIDEO_SIGMA,
        low_sigma=VIDEO_LOW_SIGMA,
        dpi=VIDEO_DPI,
    )

    for tag_name, cfg in TAG_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"  {tag_name}")
        print(f"{'=' * 60}")

        kept, skipped, manifest_df = _process_tag(
            tag_name=tag_name,
            cfg=cfg,
            output_base=OUTPUT_BASE,
            start_index=START_INDEX,
            zero_pad=ZERO_PAD,
            export_tif=EXPORT_TIF,
            export_video=EXPORT_VIDEO,
            max_projection=MAX_PROJECTION,
            dry_run=DRY_RUN,
            video_kwargs=video_kwargs,
        )

        # Print summary
        print(f"  → {tag_name}: {kept} kept, {skipped} skipped")

        # Display the mapping table
        if manifest_df is not None and not manifest_df.empty:
            print(f"\n  Mapping table ({len(manifest_df)} entries):")
            print(manifest_df.to_string(index=False))

    print("\n" + "=" * 70)
    if DRY_RUN:
        print("  DRY RUN complete — no files were exported.")
        print("  Set DRY_RUN = False in this script to execute.")
    else:
        print("  Done! All tags filtered and exported.")
    print("=" * 70)


if __name__ == "__main__":
    main()
