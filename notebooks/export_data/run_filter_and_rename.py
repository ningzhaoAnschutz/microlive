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
  3. Match each processed stem to the longest corresponding LIF-series
     stem. Result suffixes such as ``_cell1`` and ``_cell2`` therefore
     map back to one source series.
  4. Export each matched source series once as TIF (and optionally AVI)
     with sequential naming
     (e.g. ``UTag_HT_001.ome.tif``).
  5. Write an ``export_manifest.csv`` mapping table.

Usage
-----
    1. Edit the configuration section below with your paths and prefixes.
    2. Run:  python run_filter_and_rename.py
    3. First run is a dry run (preview). Set DRY_RUN = False to export.

Author: Auto-generated from MicroLive export pipeline.
"""

import logging
import os
import platform
import sys
from pathlib import Path

# Auto-relaunch in microlive conda environment if not already active
if platform.system() == 'Windows':
    # Auto-detect microlive environment in common Windows installation paths
    user_home_dir = Path.home()
    common_env_dirs = [
        user_home_dir / 'anaconda3' / 'envs' / 'microlive',
        user_home_dir / 'miniconda3' / 'envs' / 'microlive',
        user_home_dir / 'miniforge3' / 'envs' / 'microlive',
        user_home_dir / 'AppData' / 'Local' / 'anaconda3' / 'envs' / 'microlive',
        user_home_dir / 'AppData' / 'Local' / 'miniconda3' / 'envs' / 'microlive',
        Path(r'C:\ProgramData\anaconda3\envs\microlive'),
        Path(r'C:\ProgramData\miniconda3\envs\microlive'),
    ]
    MICROLIVE_ENV_DIR = next(
        (env_dir for env_dir in common_env_dirs if (env_dir / 'python.exe').is_file()),
        user_home_dir / 'anaconda3' / 'envs' / 'microlive',
    )
    _python_executable_path = MICROLIVE_ENV_DIR / 'python.exe'
else:
    MICROLIVE_ENV_DIR = Path('/opt/anaconda3/envs/microlive')
    _python_executable_path = MICROLIVE_ENV_DIR / 'bin' / 'python'

if Path(sys.prefix).resolve() != MICROLIVE_ENV_DIR.resolve():
    if _python_executable_path.is_file():
        print("Relaunching script using the microlive environment Python...\n")
        if platform.system() == 'Windows':
            os.execv(
                str(_python_executable_path),
                [str(_python_executable_path), *sys.argv],
            )
        else:
            os.execl(
                str(_python_executable_path),
                str(_python_executable_path),
                *sys.argv,
            )
    else:
        print(f"Warning: microlive environment not found at {MICROLIVE_ENV_DIR}.")
        print(f"  Expected Python at: {_python_executable_path}")
        print(f"  Run: conda activate microlive && python -c \"import sys; print(sys.prefix)\"")
        print(f"  Then update MICROLIVE_ENV_DIR in this script.\n")

# Ensure export_lif_data is importable from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import microlive.microscopy as mi
from export_lif_data import _results_stem, _sanitize, export_avi, export_ome_tif

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
# BASE_DIR = Path("/Volumes/LaCie/UTag_paper_data/Harringtonine")
BASE_DIR = Path("/Volumes/LaCie/test_export/UTag")

# Define each tag with its LIF folder, results folder, and rename prefix.
TAG_CONFIGS = {
    "UTag": {
        "lif_folder": BASE_DIR / "LIFs",
        "results_folder": BASE_DIR / "HT_Analysis_GUI",
        "rename_prefix": "UTag_HT",
    },
}

# Output directory for filtered/renamed files (per tag)
OUTPUT_BASE_DIR = BASE_DIR / "exported_filtered"

# Sequential numbering options
START_INDEX = 1  # First sequential number
ZERO_PAD = 3  # Digits: 3 → 001, 002, ...;  4 → 0001, 0002, ...

# Export options
EXPORT_TIF = True  # Export OME-TIFF for each matched series
EXPORT_VIDEO = True   # Export AVI video for each matched series

# Max-Z projection: set True to collapse Z-stacks during export.
MAX_PROJECTION = True

# Video options (only used when EXPORT_VIDEO = True)
VIDEO_FPS = 10
VIDEO_CHANNEL = 0
VIDEO_SHOW_TIMESTAMP = False
VIDEO_SHOW_SCALEBAR = False
VIDEO_MIN_PERCENTILE = 0.1
VIDEO_MAX_PERCENTILE = 99.9
VIDEO_SIGMA = 0.7
VIDEO_LOW_SIGMA = 0.15
VIDEO_DPI = 150

# Dry run: set True to preview without exporting files.
# Set False to actually export and write the manifest.
DRY_RUN = True

# ============================================================
# END OF CONFIGURATION
# ============================================================


def _build_processed_stems(results_dir, results_prefix="results_"):
    """Scan a results directory for processed image stems.

    Args:
        results_dir: Directory containing ``results_*`` subdirectories.
        results_prefix: Prefix used for each results subdirectory.

    Returns:
        Set of folder-name stems that contain at least one ``tracking_*.csv``.
    """
    results_path = Path(results_dir)
    processed_stems = set()

    if not results_path.is_dir():
        log.warning("Results folder not found: %s", results_path)
        return processed_stems

    for result_dir in sorted(results_path.iterdir()):
        if not result_dir.is_dir() or not result_dir.name.startswith(results_prefix):
            continue
        stem = result_dir.name[len(results_prefix):]
        # Check if this folder contains a tracking_*.csv
        tracking_files = list(result_dir.glob("tracking_*.csv"))
        if tracking_files:
            processed_stems.add(stem)
            log.debug("  ✓ Processed: %s  (tracking: %s)", stem, tracking_files[0].name)
        else:
            log.debug("  ✗ No tracking CSV: %s", stem)

    return processed_stems


def _group_results_by_scene(processed_stems, scene_stems):
    """Group exact and suffixed result stems under their source LIF scene.

    Args:
        processed_stems: Result-folder stems containing tracking data.
        scene_stems: Canonical stems for scenes present in the LIF file.

    Returns:
        Tuple containing the scene-to-results mapping and all matched results.
    """
    unique_scene_stems = sorted(set(scene_stems), key=lambda stem: (-len(stem), stem))
    grouped = {stem: [] for stem in unique_scene_stems}
    matched_results = set()

    for result_stem in sorted(processed_stems):
        candidates = []
        for scene_stem in unique_scene_stems:
            if result_stem == scene_stem:
                candidates.append(scene_stem)
                continue
            if not result_stem.startswith(scene_stem):
                continue
            suffix = result_stem[len(scene_stem):]
            if suffix and not suffix[0].isalnum():
                candidates.append(scene_stem)

        if not candidates:
            continue

        canonical_scene = max(candidates, key=len)
        grouped[canonical_scene].append(result_stem)
        matched_results.add(result_stem)

    return grouped, matched_results


def _process_tag(tag_name, config, output_base_dir, start_index, zero_pad,
                 should_export_tif, should_export_video, should_max_project,
                 is_dry_run, video_options):
    """Read, filter, and export the processed LIF scenes for one dataset."""
    lif_dir = Path(config["lif_folder"])
    results_dir = Path(config["results_folder"])
    rename_prefix = config.get("rename_prefix", tag_name)
    output_dir = Path(output_base_dir) / tag_name

    if not lif_dir.is_dir():
        log.error("LIF folder not found: %s", lif_dir)
        return 0, 0, None

    # Step 1: Build set of processed stems
    processed_stems = _build_processed_stems(results_dir)
    if not processed_stems:
        log.warning("No processed results found in %s", results_dir)
        return 0, 0, None

    log.info("=" * 60)
    log.info("Filtered LIF Export: %s", tag_name)
    log.info("  LIF folder    : %s", lif_dir)
    log.info("  Results folder : %s", results_dir)
    log.info("  Output folder  : %s", output_dir)
    log.info("  Processed stems: %d", len(processed_stems))
    log.info("  Rename prefix  : %s", rename_prefix)
    log.info("  Export TIF     : %s", should_export_tif)
    log.info("  Export Video   : %s", should_export_video)
    log.info("  Max-Z proj     : %s", should_max_project)
    log.info("  Dry run        : %s", is_dry_run)
    log.info("=" * 60)

    # Step 2: Find all LIF files
    lif_files = sorted(
        lif_path for lif_path in lif_dir.glob("*.lif")
        if not lif_path.name.startswith('._')
    )
    if not lif_files:
        log.warning("No .lif files found in %s", lif_dir)
        return 0, 0, None

    log.info("  Found %d LIF file(s)", len(lif_files))

    kept_count = 0
    skipped_count = 0
    sequence_counter = start_index
    manifest_records = []
    matched_processed_stems = set()

    # Step 3: Process each LIF file
    for lif_path in lif_files:
        log.info("\n  📂 Reading %s …", lif_path.name)
        try:
            reader = mi.ReadLif(
                path=str(lif_path),
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
            log.error("  ❌ Failed to read %s: %s", lif_path.name, exc)
            continue

        voxel_yx_nm = pixel_xy_um * 1000.0
        voxel_z_nm = pixel_z_um * 1000.0

        # Derive the LIF stem the same way the GUI does
        lif_stem = _results_stem(lif_path.name)

        num_scenes = len(list_names)
        log.info("    %d series found.  voxel YX=%.1f nm  Z=%.1f nm",
                 num_scenes, voxel_yx_nm, voxel_z_nm)

        scene_records = []
        for idx, raw_series_name in enumerate(list_names):
            series_name = _sanitize(raw_series_name)
            scene_records.append({
                'index': idx,
                'series_name': series_name,
                'stem': f"{lif_stem}_{series_name}",
            })

        results_by_scene, matched_in_lif = _group_results_by_scene(
            processed_stems,
            [record['stem'] for record in scene_records],
        )
        matched_processed_stems.update(matched_in_lif)

        for scene_record in scene_records:
            idx = scene_record['index']
            series_name = scene_record['series_name']
            stem = scene_record['stem']
            matching_results = results_by_scene.get(stem, [])

            try:
                time_interval = float(list_time_intervals[idx])
            except (TypeError, ValueError):
                time_interval = 0.0

            if matching_results:
                sequence_text = str(sequence_counter).zfill(zero_pad)
                new_base = f"{rename_prefix}_{sequence_text}"

                log.info(
                    "    ✓ KEEP  [%d/%d] %s → %s  (%d matching result folder%s)",
                    idx + 1,
                    num_scenes,
                    stem,
                    new_base,
                    len(matching_results),
                    "" if len(matching_results) == 1 else "s",
                )

                if not is_dry_run:
                    # Read the scene data
                    image_stack = reader.read_scene(idx)  # (T, Z, Y, X, C)

                    # Export TIF
                    if should_export_tif:
                        tif_dir = output_dir / "tif"
                        tif_dir.mkdir(parents=True, exist_ok=True)
                        tif_suffix = "_maxZ" if should_max_project else ""
                        tif_name = f"{new_base}{tif_suffix}.ome.tif"
                        tif_path = tif_dir / tif_name
                        try:
                            export_ome_tif(
                                image_stack=image_stack,
                                out_path=tif_path,
                                voxel_yx_nm=voxel_yx_nm,
                                voxel_z_nm=voxel_z_nm,
                                time_interval=time_interval,
                                bit_depth=bit_depth,
                                channel_names=channel_names,
                                max_projection=should_max_project,
                            )
                        except Exception as exc:
                            log.error("      TIF export failed: %s", exc)

                    # Export AVI
                    if should_export_video:
                        avi_dir = output_dir / "avi"
                        avi_dir.mkdir(parents=True, exist_ok=True)
                        avi_name = f"{new_base}.avi"
                        avi_path = avi_dir / avi_name
                        try:
                            export_avi(
                                image_stack=image_stack,
                                out_path=avi_path,
                                voxel_yx_nm=voxel_yx_nm,
                                time_interval=time_interval,
                                **video_options,
                            )
                        except Exception as exc:
                            log.error("      AVI export failed: %s", exc)

                manifest_records.append({
                    'Seq': sequence_counter,
                    'New_Name': new_base,
                    'Original_Stem': stem,
                    'LIF_File': lif_path.name,
                    'Series_Name': series_name,
                    'Series_Index': idx,
                    'Matching_Results_Count': len(matching_results),
                    'Matching_Results_Folders': ';'.join(
                        f"results_{result_stem}" for result_stem in matching_results
                    ),
                })

                sequence_counter += 1
                kept_count += 1
            else:
                log.info("    ✗ SKIP  [%d/%d] %s  (no tracking data)",
                         idx + 1, num_scenes, stem)
                skipped_count += 1

    unmatched_processed_stems = sorted(processed_stems - matched_processed_stems)
    if unmatched_processed_stems:
        log.warning(
            "  %d processed result folder(s) did not match any LIF scene:",
            len(unmatched_processed_stems),
        )
        for unmatched_stem in unmatched_processed_stems:
            log.warning("    results_%s", unmatched_stem)

    # Step 4: Build manifest DataFrame
    manifest_df = None
    if manifest_records:
        manifest_df = pd.DataFrame(manifest_records)
        if not is_dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = output_dir / "export_manifest.csv"
            manifest_df.to_csv(manifest_path, index=False)
            log.info("  Manifest saved to: %s", manifest_path)

    log.info("-" * 60)
    log.info("  Kept: %d  |  Skipped: %d  |  Total: %d",
             kept_count, skipped_count, kept_count + skipped_count)
    log.info("=" * 60)

    return kept_count, skipped_count, manifest_df


def main():
    """Run all configured filtered LIF exports."""
    print("=" * 70)
    print("  Filtered LIF Export & Sequential Rename")
    print(f"  Dry Run: {DRY_RUN}")
    print(f"  Export TIF: {EXPORT_TIF}")
    print(f"  Export AVI: {EXPORT_VIDEO}")
    print(f"  Max-Z Projection: {MAX_PROJECTION}")
    print("=" * 70)

    video_options = dict(
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

    for tag_name, config in TAG_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"  {tag_name}")
        print(f"{'=' * 60}")

        kept_count, skipped_count, manifest_df = _process_tag(
            tag_name=tag_name,
            config=config,
            output_base_dir=OUTPUT_BASE_DIR,
            start_index=START_INDEX,
            zero_pad=ZERO_PAD,
            should_export_tif=EXPORT_TIF,
            should_export_video=EXPORT_VIDEO,
            should_max_project=MAX_PROJECTION,
            is_dry_run=DRY_RUN,
            video_options=video_options,
        )

        # Print summary
        print(f"  → {tag_name}: {kept_count} kept, {skipped_count} skipped")

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
