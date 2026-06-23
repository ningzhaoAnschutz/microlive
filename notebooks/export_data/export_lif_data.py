#!/usr/bin/env python3
"""
export_lif_data.py
==================
Batch export TIF and AVI files from Leica LIF files.

This script mirrors the GUI's "Export TIF" and "Export Video" functionality
(Import tab) but runs headlessly, iterating over an entire folder of LIF
files. For each LIF file and each series (scene) inside it, the script:

  1. Exports the full image stack as an OME-TIFF (identical to the GUI's
     "Export TIF" button).
  2. Exports a grayscale AVI video with optional timestamp overlay and
     scale bar (extends the GUI's "Export Video" button).

The output folder structure mirrors the input:
    <output_root>/
        <subfolder>/
            <lif_filename>/
                tif/
                    <lif_filename>_<series_name>.ome.tif
                avi/
                    <lif_filename>_<series_name>.avi

Usage
-----
    python export_lif_data.py --input_folder /path/to/lif_files \
                              --output_folder /path/to/output \
                              --show_timestamp \
                              --show_scalebar \
                              --fps 10

Author: Luis Aguilera (auto-generated from MicroLive GUI logic)
"""

import argparse
import os
import shutil
import sys
import re
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import cv2
from scipy.ndimage import gaussian_filter

# MicroLive imports — ensure the package is importable
try:
    import microlive.microscopy as mi
    from matplotlib_scalebar.scalebar import ScaleBar
except ImportError as exc:
    sys.exit(
        f"Could not import MicroLive or matplotlib-scalebar: {exc}\n"
        "Make sure the 'microlive' environment is activated."
    )

import matplotlib
matplotlib.use("Agg")  # headless backend — no GUI required
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Sanitize a string for safe use as a filename component."""
    return re.sub(r'[^\w\-_. ]', '_', name)


def _results_stem(filename: str) -> str:
    """Derive the base stem from a LIF filename, matching the GUI convention.

    The GUI truncates at the *first* period (``name.split('.')[0]``),
    while ``Path.stem`` strips only the last extension.  For
    ``sample.v2.lif`` these differ: ``sample`` vs ``sample.v2``.
    We follow the GUI to ensure ``results_*`` folder names match.
    """
    return _sanitize(filename.split('.')[0])


def _format_time(value_s: float, reference_s: float) -> str:
    """Format a time value exactly as the MicroLive GUI does.

    The unit (µs, ms, s) is chosen based on *reference_s* so that all frames
    in a movie share the same unit.
    """
    if reference_s < 0.001:
        return f"{value_s * 1e6:.2f} µs"
    elif reference_s < 1:
        return f"{value_s * 1000:.2f} ms"
    elif reference_s < 10:
        return f"{value_s:.2f} s"
    else:
        return f"{value_s:.1f} s"


# ---------------------------------------------------------------------------
# TIF export  (mirrors GUI  _export_ome_tif)
# ---------------------------------------------------------------------------

def export_ome_tif(
    image_stack: np.ndarray,
    out_path: Path,
    voxel_yx_nm: float,
    voxel_z_nm: float,
    time_interval: float,
    bit_depth: int,
    channel_names: list,
    max_projection: bool = False,
):
    """Export a 5-D image stack as OME-TIFF.

    Parameters
    ----------
    image_stack : np.ndarray
        Shape ``(T, Z, Y, X, C)`` — the MicroLive internal format.
    out_path : Path
        Full output file path (should end with ``.ome.tif``).
    voxel_yx_nm : float
        XY pixel size in nanometres.
    voxel_z_nm : float
        Z pixel size in nanometres.
    time_interval : float
        Time between frames in seconds.
    bit_depth : int
        Significant bit depth (e.g. 8, 12, 16).
    channel_names : list[str]
        Channel name labels.
    max_projection : bool
        If True, collapse Z via max projection before saving (Z=1).
    """
    # Optional max-Z projection (raw uint16 values preserved)
    if max_projection:
        image_stack = np.max(image_stack, axis=1, keepdims=True)  # (T, 1, Y, X, C)

    # Transpose from TZYXC → TCZYX (OME-TIFF standard)
    temp_image = np.moveaxis(image_stack, 4, 1)
    shape = temp_image.shape

    if bit_depth is None or bit_depth == 0:
        bit_depth = 16

    physical_size_x = float(voxel_yx_nm) / 1000.0 if voxel_yx_nm else 1.0
    physical_size_z = float(voxel_z_nm) / 1000.0 if voxel_z_nm else 1.0
    channel_metadata = {'Name': channel_names} if channel_names else {}

    tifffile.imwrite(
        str(out_path),
        temp_image.astype(np.uint16),
        shape=shape,
        dtype='uint16',
        imagej=False,
        metadata={
            'axes': 'TCZYX',
            'PhysicalSizeX': physical_size_x,
            'PhysicalSizeY': physical_size_x,
            'PhysicalSizeZ': physical_size_z,
            'TimeIncrement': float(time_interval),
            'TimeIncrementUnit': 's',
            'SignificantBits': bit_depth,
            'Channel': channel_metadata,
        },
    )
    log.info("  TIF  → %s", out_path.name)


# ---------------------------------------------------------------------------
# AVI export  (extends GUI  export_displayed_video)
# ---------------------------------------------------------------------------

def export_avi(
    image_stack: np.ndarray,
    out_path: Path,
    voxel_yx_nm: float = None,
    time_interval: float = None,
    show_timestamp: bool = False,
    show_scalebar: bool = False,
    fps: int = 10,
    min_percentile: float = 0.1,
    max_percentile: float = 99.9,
    sigma: float = 0.7,
    low_sigma: float = 0.15,
    channel: int = 0,
    dpi: int = 150,
):
    """Export a grayscale AVI video from a 5-D image stack.

    Parameters
    ----------
    image_stack : np.ndarray
        Shape ``(T, Z, Y, X, C)``.
    out_path : Path
        Output ``.avi`` file path.
    voxel_yx_nm : float, optional
        XY pixel size in nm.  Required for scale bar.
    time_interval : float, optional
        Frame interval in seconds.  Required for timestamp.
    show_timestamp : bool
        Overlay a time label (upper-left).
    show_scalebar : bool
        Overlay a matplotlib-scalebar (lower-right).
    fps : int
        Frames per second for the output video.
    min_percentile, max_percentile : float
        Intensity rescaling percentiles (GUI defaults: 0.1, 99.9).
    sigma : float
        Smoothing Gaussian filter sigma (GUI default: 0.7). Set to 0 to disable.
    low_sigma : float
        Low-pass Gaussian filter sigma (GUI default: 0.15). Set to 0 to disable.
    channel : int
        Channel index to render (grayscale).
    dpi : int
        Rendering DPI (controls output resolution).
    """
    T, Z, Y, X, C = image_stack.shape

    # Pre-compute figure dimensions from pixel size
    fig_w = X / dpi
    fig_h = Y / dpi
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi)
    canvas = FigureCanvas(fig)

    frames = []
    for t in range(T):
        fig.clear()
        ax = fig.add_axes([0, 0, 1, 1])  # fill the entire figure

        # Max-Z projection for the selected channel
        frame_channel = image_stack[t, :, :, :, channel]  # (Z, Y, X)
        data_img = np.max(frame_channel, axis=0)  # (Y, X)

        # Percentile-based rescaling (matches GUI RemoveExtrema + normalize)
        vmin = np.percentile(data_img, min_percentile)
        vmax = np.percentile(data_img, max_percentile)
        if vmax <= vmin:
            vmax = vmin + 1
        normalized = np.clip((data_img.astype(float) - vmin) / (vmax - vmin), 0, 1)

        # Apply Gaussian filters (matches GUI display_sigma / low_display_sigma)
        if low_sigma > 0:
            normalized = gaussian_filter(normalized, sigma=low_sigma)
        if sigma > 0:
            normalized = gaussian_filter(normalized, sigma=sigma)

        ax.imshow(normalized, cmap='gray', vmin=0, vmax=1, aspect='equal')
        ax.axis('off')

        # Scale bar
        if show_scalebar and voxel_yx_nm is not None and voxel_yx_nm > 0:
            microns_per_pixel = voxel_yx_nm / 1000.0
            scalebar = ScaleBar(
                microns_per_pixel,
                units='um',
                length_fraction=0.2,
                location='lower right',
                box_color='black',
                color='white',
                font_properties={'size': 10},
            )
            ax.add_artist(scalebar)

        # Timestamp
        if show_timestamp and time_interval is not None:
            time_val = float(t) * float(time_interval)
            ts = _format_time(time_val, reference_s=time_interval)
            ax.text(
                0.05, 0.95, ts,
                transform=ax.transAxes,
                verticalalignment='top',
                color='white',
                fontsize=12,
                bbox=dict(facecolor='black', alpha=0.5, pad=2),
            )

        # Render figure to numpy array
        canvas.draw()
        buf = canvas.buffer_rgba()
        arr = np.asarray(buf)
        frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        frames.append(frame_bgr)

    if not frames:
        log.warning("  No frames to write for %s", out_path.name)
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height), isColor=True)
    for frame in frames:
        out.write(frame)
    out.release()
    log.info("  AVI  → %s  (%d frames, %dx%d)", out_path.name, len(frames), width, height)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_lif_file(
    lif_path: Path,
    output_root: Path,
    export_tif: bool = True,
    export_video: bool = True,
    show_timestamp: bool = False,
    show_scalebar: bool = False,
    fps: int = 10,
    channel: int = 0,
    min_percentile: float = 0.1,
    max_percentile: float = 99.9,
    sigma: float = 0.7,
    low_sigma: float = 0.15,
    dpi: int = 150,
    max_projection: bool = False,
    input_root: Path = None,
):
    """Process a single LIF file: read all series and export TIF / AVI.

    Parameters
    ----------
    lif_path : Path
        Path to the ``.lif`` file.
    output_root : Path
        Root output directory.  A subfolder named after the LIF file is
        created automatically.
    export_tif : bool
        If True, export OME-TIFF for each series.
    export_video : bool
        If True, export AVI for each series.
    show_timestamp, show_scalebar : bool
        Overlays for the AVI export.
    fps : int
        AVI frame rate.
    channel : int
        Channel index for AVI rendering.
    min_percentile, max_percentile : float
        Intensity rescaling percentiles for AVI.
    sigma : float
        Smoothing Gaussian filter sigma. Set to 0 to disable.
    low_sigma : float
        Low-pass Gaussian filter sigma. Set to 0 to disable.
    dpi : int
        Rendering DPI for AVI.
    max_projection : bool
        If True, export max-Z projection TIF instead of full Z-stack.
    input_root : Path, optional
        Root input directory for computing relative LIF paths in the
        manifest.  When None, only the LIF filename is recorded.

    Returns
    -------
    list[dict]
        Manifest records for each series processed.
    """
    log.info("Reading %s …", lif_path.name)
    reader = mi.ReadLif(
        path=str(lif_path),
        show_metadata=False,
        save_tif=False,
        save_png=False,
        format='TZYXC',
        lazy=True,
    )
    (
        _list_images,  # not used — we read scene-by-scene below
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

    voxel_yx_nm = pixel_xy_um * 1000.0
    voxel_z_nm = pixel_z_um * 1000.0

    # Use _results_stem (split at first '.') to match the GUI convention
    lif_stem = _results_stem(lif_path.name)

    # Create output subfolder: <output_root>/<lif_stem>/
    lif_folder = output_root / lif_stem
    lif_folder.mkdir(parents=True, exist_ok=True)

    # Compute relative LIF path for manifest
    if input_root is not None:
        try:
            lif_rel = str(lif_path.relative_to(input_root))
        except ValueError:
            lif_rel = lif_path.name
    else:
        lif_rel = lif_path.name

    num_scenes = len(list_names)
    log.info("  %d series found.  voxel YX=%.1f nm  Z=%.1f nm  bit_depth=%d",
             num_scenes, voxel_yx_nm, voxel_z_nm, bit_depth)

    manifest_records = []

    for idx in range(num_scenes):
        series_name = _sanitize(list_names[idx])
        # Cast to float — LIF metadata may return CycleTime as a string
        try:
            time_interval = float(list_time_intervals[idx])
        except (TypeError, ValueError):
            time_interval = 0.0
        log.info("  [%d/%d] %s  (dt=%.4f s)", idx + 1, num_scenes, series_name, time_interval)

        # Read scene data (lazy — only loads this scene)
        image_stack = reader.read_scene(idx)  # shape: (T, Z, Y, X, C)

        # Build filename: <lif_stem>_<series_name>
        base_name = f"{lif_stem}_{series_name}"

        # Track what was exported for the manifest
        tif_filename = ""
        avi_filename = ""

        # --- OME-TIFF export ---
        if export_tif:
            tif_folder = lif_folder / "tif"
            tif_folder.mkdir(parents=True, exist_ok=True)
            tif_suffix = "_maxZ" if max_projection else ""
            tif_name = f"{base_name}{tif_suffix}.ome.tif"
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
                tif_filename = tif_name
            except Exception as exc:
                log.error("    TIF export failed for %s: %s", series_name, exc)

        # --- AVI video export ---
        if export_video:
            avi_folder = lif_folder / "avi"
            avi_folder.mkdir(parents=True, exist_ok=True)
            avi_name = f"{base_name}.avi"
            avi_path = avi_folder / avi_name
            try:
                export_avi(
                    image_stack=image_stack,
                    out_path=avi_path,
                    voxel_yx_nm=voxel_yx_nm,
                    time_interval=time_interval,
                    show_timestamp=show_timestamp,
                    show_scalebar=show_scalebar,
                    fps=fps,
                    min_percentile=min_percentile,
                    max_percentile=max_percentile,
                    sigma=sigma,
                    low_sigma=low_sigma,
                    channel=channel,
                    dpi=dpi,
                )
                avi_filename = avi_name
            except Exception as exc:
                log.error("    AVI export failed for %s: %s", series_name, exc)

        # Record manifest entry (only includes successfully exported files)
        manifest_records.append({
            'LIF_Relative_Path': lif_rel,
            'Series_Index': idx,
            'Series_Name': list_names[idx],  # raw, unsanitized
            'Original_Stem': base_name,
            'TIF_Filename': tif_filename,
            'AVI_Filename': avi_filename,
            'Time_Interval_s': time_interval,
        })

    log.info("  Done with %s ✓", lif_path.name)
    return manifest_records


def _write_manifest_csv(records: list, output_path: Path):
    """Write manifest records to CSV atomically.

    Writes to a temporary file first, then atomically replaces the
    target to avoid partial manifests on crash.
    """
    if not records:
        return
    df = pd.DataFrame(records)
    manifest_path = output_path / "export_manifest.csv"
    # Atomic write: temp file → os.replace
    fd, tmp_path = tempfile.mkstemp(
        suffix=".csv", prefix=".manifest_", dir=str(output_path)
    )
    try:
        os.close(fd)
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, str(manifest_path))
        log.info("  Manifest written: %s  (%d records)", manifest_path.name, len(records))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def batch_export(
    input_folder: str,
    output_folder: str,
    export_tif: bool = True,
    export_video: bool = True,
    show_timestamp: bool = False,
    show_scalebar: bool = False,
    fps: int = 10,
    channel: int = 0,
    min_percentile: float = 0.1,
    max_percentile: float = 99.9,
    sigma: float = 0.7,
    low_sigma: float = 0.15,
    dpi: int = 150,
    recursive: bool = True,
    max_projection: bool = False,
):
    """Iterate through a folder of LIF files and export TIF + AVI.

    An ``export_manifest.csv`` is written to the output folder after
    processing, listing every exported file and its source identity.

    Parameters
    ----------
    input_folder : str
        Directory containing ``.lif`` files (searched recursively by default).
    output_folder : str
        Root output directory.  Subfolder structure mirrors the input.
    export_tif : bool
        Export OME-TIFF for each series.
    export_video : bool
        Export AVI video for each series.
    show_timestamp : bool
        Overlay timestamp on each video frame.
    show_scalebar : bool
        Overlay scale bar on each video frame.
    fps : int
        Video frame rate.
    channel : int
        Channel index for video (0 = first channel).
    min_percentile, max_percentile : float
        Intensity rescaling percentiles for the video.
    sigma : float
        Smoothing Gaussian sigma for video. Set to 0 to disable.
    low_sigma : float
        Low-pass Gaussian sigma for video. Set to 0 to disable.
    dpi : int
        Rendering DPI for the video.
    recursive : bool
        If True, search subdirectories for LIF files.
    max_projection : bool
        If True, export max-Z projection TIF instead of full Z-stack.
    """
    input_path = Path(input_folder).resolve()
    output_path = Path(output_folder).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.is_dir():
        log.error("Input folder does not exist: %s", input_path)
        return

    # Find all .lif files (exclude macOS ._ resource fork files)
    pattern = '**/*.lif' if recursive else '*.lif'
    lif_files = sorted(
        f for f in input_path.glob(pattern)
        if not f.name.startswith('._')
    )

    if not lif_files:
        log.warning("No .lif files found in %s", input_path)
        return

    log.info("=" * 60)
    log.info("Batch LIF Export")
    log.info("  Input folder  : %s", input_path)
    log.info("  Output folder : %s", output_path)
    log.info("  LIF files     : %d", len(lif_files))
    log.info("  Export TIF    : %s", export_tif)
    log.info("  Export Video  : %s", export_video)
    log.info("  Timestamp     : %s", show_timestamp)
    log.info("  Scale bar     : %s", show_scalebar)
    log.info("  FPS           : %d", fps)
    log.info("  Channel       : %d", channel)
    log.info("  Percentiles   : [%.1f, %.1f]", min_percentile, max_percentile)
    log.info("  DPI           : %d", dpi)
    log.info("  Max-Z proj    : %s", max_projection)
    log.info("=" * 60)

    all_manifest_records = []

    for i, lif_file in enumerate(lif_files, 1):
        # Preserve subfolder structure relative to input_folder
        relative_parent = lif_file.parent.relative_to(input_path)
        dest_root = output_path / relative_parent

        log.info("[%d/%d] %s", i, len(lif_files), lif_file.relative_to(input_path))
        try:
            records = process_lif_file(
                lif_path=lif_file,
                output_root=dest_root,
                export_tif=export_tif,
                export_video=export_video,
                show_timestamp=show_timestamp,
                show_scalebar=show_scalebar,
                fps=fps,
                channel=channel,
                min_percentile=min_percentile,
                max_percentile=max_percentile,
                sigma=sigma,
                low_sigma=low_sigma,
                dpi=dpi,
                max_projection=max_projection,
                input_root=input_path,
            )
            all_manifest_records.extend(records)
        except Exception as exc:
            log.error("  FAILED: %s", exc, exc_info=True)

    # Write export manifest
    _write_manifest_csv(all_manifest_records, output_path)

    log.info("=" * 60)
    log.info("Batch export complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch export TIF and AVI from Leica LIF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Export both TIF and AVI with timestamp + scale bar:
  python export_lif_data.py \\
      --input_folder  /data/microscope/lif_files \\
      --output_folder /data/exports \\
      --show_timestamp --show_scalebar --fps 15

  # Export only TIF files:
  python export_lif_data.py \\
      --input_folder  /data/lif_files \\
      --output_folder /data/tif_only \\
      --no_video

  # Export only AVI, second channel, custom percentiles:
  python export_lif_data.py \\
      --input_folder  /data/lif_files \\
      --output_folder /data/videos \\
      --no_tif --channel 1 \\
      --min_percentile 0.5 --max_percentile 99.9
""",
    )

    parser.add_argument(
        "--input_folder", type=str, required=True,
        help="Path to folder containing .lif files.",
    )
    parser.add_argument(
        "--output_folder", type=str, required=True,
        help="Path to output folder (created if needed).",
    )

    # Export toggles
    parser.add_argument(
        "--no_tif", action="store_true", default=False,
        help="Skip OME-TIFF export.",
    )
    parser.add_argument(
        "--no_video", action="store_true", default=False,
        help="Skip AVI video export.",
    )

    # Video overlays
    parser.add_argument(
        "--show_timestamp", action="store_true", default=False,
        help="Overlay timestamp on each video frame.",
    )
    parser.add_argument(
        "--show_scalebar", action="store_true", default=False,
        help="Overlay scale bar on each video frame.",
    )

    # Video parameters
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Video frame rate (default: 10).",
    )
    parser.add_argument(
        "--channel", type=int, default=0,
        help="Channel index to render in the video (default: 0).",
    )
    parser.add_argument(
        "--min_percentile", type=float, default=0.1,
        help="Min percentile for intensity rescaling (default: 0.1).",
    )
    parser.add_argument(
        "--max_percentile", type=float, default=99.9,
        help="Max percentile for intensity rescaling (default: 99.9).",
    )
    parser.add_argument(
        "--sigma", type=float, default=0.7,
        help="Smoothing Gaussian sigma (default: 0.7). Set to 0 to disable.",
    )
    parser.add_argument(
        "--low_sigma", type=float, default=0.15,
        help="Low-pass Gaussian sigma (default: 0.15). Set to 0 to disable.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Rendering DPI for AVI (default: 150).",
    )

    # TIF options
    parser.add_argument(
        "--max_projection", action="store_true", default=False,
        help="Export max-Z projection instead of full Z-stack.",
    )

    # Search options
    parser.add_argument(
        "--no_recursive", action="store_true", default=False,
        help="Do not search subdirectories for .lif files.",
    )

    args = parser.parse_args()

    batch_export(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        export_tif=not args.no_tif,
        export_video=not args.no_video,
        show_timestamp=args.show_timestamp,
        show_scalebar=args.show_scalebar,
        fps=args.fps,
        channel=args.channel,
        min_percentile=args.min_percentile,
        max_percentile=args.max_percentile,
        sigma=args.sigma,
        low_sigma=args.low_sigma,
        dpi=args.dpi,
        recursive=not args.no_recursive,
        max_projection=args.max_projection,
    )


# ---------------------------------------------------------------------------
# Filter exported data based on processed results
# ---------------------------------------------------------------------------

def _extract_stem(fname: str):
    """Extract the matching stem from an exported filename.

    Strips ``.ome.tif`` / ``.avi`` extensions and the optional ``_maxZ``
    suffix to produce a stem that can be matched against ``results_*``
    folder names from the GUI.

    Returns
    -------
    tuple[str, str] or None
        ``(stem, extension)`` where extension is ``'.ome.tif'`` or
        ``'.avi'``, or None if the filename is not a recognised export.
    """
    if fname.endswith(".ome.tif"):
        stem = fname[:-len(".ome.tif")]
        if stem.endswith("_maxZ"):
            stem = stem[:-len("_maxZ")]
        return stem, ".ome.tif"
    elif fname.endswith(".avi"):
        stem = fname[:-len(".avi")]
        return stem, ".avi"
    return None


def filter_exported_data(
    exported_folder: str,
    results_folder: str,
    output_folder: str = None,
    results_prefix: str = "results_",
    dry_run: bool = False,
    rename_prefix: str = None,
    start_index: int = 1,
    zero_pad: int = 3,
    max_projection: bool = False,
):
    """Copy only exported files whose series were processed in the GUI.

    Scans ``results_folder`` for directories matching ``results_<stem>`` and
    keeps only the TIF/AVI files in ``exported_folder`` whose filename stem
    matches one of those processed stems.

    When ``rename_prefix`` is provided, kept files are copied with
    sequential names (e.g. ``AlfaTag_HT_001.ome.tif``) using a single
    global counter.  An ``export_manifest.csv`` mapping table is written
    to the output folder and a :class:`pandas.DataFrame` is returned.

    When ``max_projection`` is True, full Z-stack OME-TIFFs are
    max-projected (collapse Z axis via ``np.max``) before being saved
    to the output folder.  Files that are already max-projected
    (``_maxZ`` suffix) are copied as-is.

    Parameters
    ----------
    exported_folder : str
        Path to the batch-exported data (e.g. ``exported_data/``).
    results_folder : str
        Path to the GUI analysis output containing ``results_*`` folders.
    output_folder : str, optional
        Destination for filtered files.  Defaults to ``exported_filtered/``
        as a sibling of ``exported_folder``.
    results_prefix : str
        Prefix used by the GUI for results folders (default: ``results_``).
    dry_run : bool
        If True, only log what would be copied without writing files.
    rename_prefix : str, optional
        If provided, rename kept files using this prefix plus a
        sequential number (e.g. ``AlfaTag_HT_001``).  A TIF and its
        corresponding AVI for the same series share the same number.
        An ``export_manifest.csv`` is saved to ``output_folder``.
    start_index : int
        Starting number for sequential naming (default: 1).
    zero_pad : int
        Number of digits for the sequential number (default: 3).
    max_projection : bool
        If True, apply max-Z projection to kept TIF files during copy.
        Already-projected files (``_maxZ`` suffix) are copied unchanged.

    Returns
    -------
    tuple[int, int, pandas.DataFrame | None]
        ``(kept, skipped, manifest_df)``.  ``manifest_df`` is a DataFrame
        mapping sequential names to original stems when ``rename_prefix``
        is set, otherwise ``None``.
    """
    exported_path = Path(exported_folder).resolve()
    results_path = Path(results_folder).resolve()

    if output_folder is None:
        output_path = exported_path.parent / "exported_filtered"
    else:
        output_path = Path(output_folder).resolve()

    if not exported_path.is_dir():
        log.error("Exported folder not found: %s", exported_path)
        return 0, 0, None
    if not results_path.is_dir():
        log.error("Results folder not found: %s", results_path)
        return 0, 0, None

    # Validate rename_prefix if provided
    if rename_prefix is not None:
        rename_prefix = _sanitize(rename_prefix).strip()
        if not rename_prefix:
            raise ValueError("rename_prefix is empty after sanitization.")

    # Build set of processed stems from results_* folders
    processed_stems = set()
    for d in results_path.iterdir():
        if d.is_dir() and d.name.startswith(results_prefix):
            stem = d.name[len(results_prefix):]
            processed_stems.add(stem)

    if not processed_stems:
        log.warning("No %s* folders found in %s", results_prefix, results_path)
        return 0, 0, None

    log.info("=" * 60)
    log.info("Filter Exported Data")
    log.info("  Exported folder : %s", exported_path)
    log.info("  Results folder  : %s", results_path)
    log.info("  Output folder   : %s", output_path)
    log.info("  Processed stems : %d", len(processed_stems))
    log.info("  Dry run         : %s", dry_run)
    if rename_prefix:
        log.info("  Rename prefix   : %s", rename_prefix)
        log.info("  Start index     : %d", start_index)
        log.info("  Zero padding    : %d digits", zero_pad)
    if max_projection:
        log.info("  Max-Z projection: True")
    log.info("=" * 60)

    # ── Pass 1: identify all KEEP files and group by stem ──
    # Group files by their matching stem so that TIF + AVI pairs
    # for the same series share a single sequential number.
    keep_files = []     # list of (src_file, stem, extension)
    skipped = 0

    for src_file in sorted(exported_path.rglob("*")):
        if not src_file.is_file():
            continue

        parsed = _extract_stem(src_file.name)
        if parsed is None:
            continue  # skip non-export files

        stem, ext = parsed

        if stem in processed_stems:
            keep_files.append((src_file, stem, ext))
            log.info("  ✓ KEEP   %s", src_file.relative_to(exported_path))
        else:
            log.info("  ✗ SKIP   %s", src_file.relative_to(exported_path))
            skipped += 1

    kept = len(keep_files)

    # ── Pass 2: assign sequential numbers and copy/rename ──
    manifest_records = []

    if rename_prefix is not None:
        # Build ordered list of unique stems to assign sequential numbers.
        # Preserve the sorted order from the file walk.
        seen_stems = []
        stem_to_seq = {}
        for _, stem, _ in keep_files:
            if stem not in stem_to_seq:
                seq = start_index + len(seen_stems)
                stem_to_seq[stem] = seq
                seen_stems.append(stem)

        for src_file, stem, ext in keep_files:
            seq = stem_to_seq[stem]
            seq_str = str(seq).zfill(zero_pad)
            new_base = f"{rename_prefix}_{seq_str}"
            new_filename = f"{new_base}{ext}"

            # Determine the LIF subfolder name for provenance
            try:
                rel_to_exported = src_file.relative_to(exported_path)
                lif_subfolder = rel_to_exported.parts[0] if len(rel_to_exported.parts) > 1 else ""
            except ValueError:
                lif_subfolder = ""

            # Determine file type
            file_type = "tif" if ext == ".ome.tif" else "avi"

            if not dry_run:
                output_path.mkdir(parents=True, exist_ok=True)
                dst_file = output_path / new_filename

                # Apply max projection if requested for TIF files
                if max_projection and ext == ".ome.tif" and "_maxZ" not in src_file.name:
                    _copy_with_max_projection(src_file, dst_file)
                else:
                    shutil.copy2(str(src_file), str(dst_file))

                log.info("  📝 %s → %s", src_file.name, new_filename)

            manifest_records.append({
                'Seq': seq,
                'New_Name': new_base,
                'New_Filename': new_filename,
                'Original_Stem': stem,
                'Original_Filename': src_file.name,
                'Source_LIF_Folder': lif_subfolder,
                'File_Type': file_type,
            })
    else:
        # No renaming — copy with original names (legacy behavior)
        for src_file, stem, ext in keep_files:
            rel = src_file.relative_to(exported_path)
            dst_file = output_path / rel
            if not dry_run:
                dst_file.parent.mkdir(parents=True, exist_ok=True)

                # Apply max projection if requested for TIF files
                if max_projection and ext == ".ome.tif" and "_maxZ" not in src_file.name:
                    _copy_with_max_projection(src_file, dst_file)
                else:
                    shutil.copy2(str(src_file), str(dst_file))

    # ── Write manifest CSV ──
    manifest_df = None
    if rename_prefix is not None and manifest_records:
        manifest_df = pd.DataFrame(manifest_records)
        if not dry_run:
            _write_manifest_csv(manifest_records, output_path)
        log.info("  Manifest: %d records", len(manifest_records))

    log.info("-" * 60)
    log.info("  Kept: %d  |  Skipped: %d  |  Total: %d", kept, skipped, kept + skipped)
    if dry_run:
        log.info("  (DRY RUN — no files were copied)")
    log.info("=" * 60)
    return kept, skipped, manifest_df


def _copy_with_max_projection(src_path: Path, dst_path: Path):
    """Read an OME-TIFF, apply max-Z projection, and save.

    If the file is already single-Z (Z=1), it is copied directly.
    The max projection collapses the Z dimension via ``np.max``.
    """
    try:
        data = tifffile.imread(str(src_path))
    except Exception as exc:
        log.warning("    Could not read %s for max projection, copying as-is: %s",
                    src_path.name, exc)
        shutil.copy2(str(src_path), str(dst_path))
        return

    # Determine axes layout — exported OME-TIFFs are TCZYX
    if data.ndim == 5:
        # (T, C, Z, Y, X) → max over Z axis (index 2)
        if data.shape[2] <= 1:
            # Already single-Z, just copy
            shutil.copy2(str(src_path), str(dst_path))
            return
        projected = np.max(data, axis=2, keepdims=True)
    elif data.ndim == 4:
        # Could be (T, Z, Y, X) or (T, C, Y, X) — assume Z is axis 1
        if data.shape[1] <= 1:
            shutil.copy2(str(src_path), str(dst_path))
            return
        projected = np.max(data, axis=1, keepdims=True)
    elif data.ndim == 3:
        # (Z, Y, X) — max over axis 0
        if data.shape[0] <= 1:
            shutil.copy2(str(src_path), str(dst_path))
            return
        projected = np.max(data, axis=0, keepdims=True)
    else:
        # 2D or unexpected — just copy
        shutil.copy2(str(src_path), str(dst_path))
        return

    # Write the projected data, preserving dtype
    tifffile.imwrite(str(dst_path), projected.astype(data.dtype))
    log.info("    Max-Z projected: %s → shape %s", src_path.name, projected.shape)


if __name__ == "__main__":
    main()
