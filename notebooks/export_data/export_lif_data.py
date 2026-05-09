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
import shutil
import sys
import re
import logging
from pathlib import Path

import numpy as np
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

    # Create output subfolder: <output_root>/<lif_stem>/
    lif_folder = output_root / _sanitize(lif_path.stem)
    lif_folder.mkdir(parents=True, exist_ok=True)

    num_scenes = len(list_names)
    log.info("  %d series found.  voxel YX=%.1f nm  Z=%.1f nm  bit_depth=%d",
             num_scenes, voxel_yx_nm, voxel_z_nm, bit_depth)

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
        base_name = f"{_sanitize(lif_path.stem)}_{series_name}"

        # --- OME-TIFF export ---
        if export_tif:
            tif_folder = lif_folder / "tif"
            tif_folder.mkdir(parents=True, exist_ok=True)
            tif_suffix = "_maxZ" if max_projection else ""
            tif_path = tif_folder / f"{base_name}{tif_suffix}.ome.tif"
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

        # --- AVI video export ---
        if export_video:
            avi_folder = lif_folder / "avi"
            avi_folder.mkdir(parents=True, exist_ok=True)
            avi_path = avi_folder / f"{base_name}.avi"
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

    log.info("  Done with %s ✓", lif_path.name)


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

    for i, lif_file in enumerate(lif_files, 1):
        # Preserve subfolder structure relative to input_folder
        relative_parent = lif_file.parent.relative_to(input_path)
        dest_root = output_path / relative_parent

        log.info("[%d/%d] %s", i, len(lif_files), lif_file.relative_to(input_path))
        try:
            process_lif_file(
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
            )
        except Exception as exc:
            log.error("  FAILED: %s", exc, exc_info=True)

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

def filter_exported_data(
    exported_folder: str,
    results_folder: str,
    output_folder: str = None,
    results_prefix: str = "results_",
    dry_run: bool = False,
):
    """Copy only exported files whose series were processed in the GUI.

    Scans ``results_folder`` for directories matching ``results_<stem>`` and
    keeps only the TIF/AVI files in ``exported_folder`` whose filename stem
    matches one of those processed stems.

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

    Returns
    -------
    tuple[int, int]
        ``(kept, skipped)`` file counts.
    """
    exported_path = Path(exported_folder).resolve()
    results_path = Path(results_folder).resolve()

    if output_folder is None:
        output_path = exported_path.parent / "exported_filtered"
    else:
        output_path = Path(output_folder).resolve()

    if not exported_path.is_dir():
        log.error("Exported folder not found: %s", exported_path)
        return 0, 0
    if not results_path.is_dir():
        log.error("Results folder not found: %s", results_path)
        return 0, 0

    # Build set of processed stems from results_* folders
    processed_stems = set()
    for d in results_path.iterdir():
        if d.is_dir() and d.name.startswith(results_prefix):
            stem = d.name[len(results_prefix):]
            processed_stems.add(stem)

    if not processed_stems:
        log.warning("No %s* folders found in %s", results_prefix, results_path)
        return 0, 0

    log.info("=" * 60)
    log.info("Filter Exported Data")
    log.info("  Exported folder : %s", exported_path)
    log.info("  Results folder  : %s", results_path)
    log.info("  Output folder   : %s", output_path)
    log.info("  Processed stems : %d", len(processed_stems))
    log.info("  Dry run         : %s", dry_run)
    log.info("=" * 60)

    kept = 0
    skipped = 0

    # Walk all files in the exported folder
    for src_file in sorted(exported_path.rglob("*")):
        if not src_file.is_file():
            continue

        # Extract stem: remove .ome.tif or .avi extension (and _maxZ suffix)
        fname = src_file.name
        if fname.endswith(".ome.tif"):
            stem = fname[:-len(".ome.tif")]
            if stem.endswith("_maxZ"):
                stem = stem[:-len("_maxZ")]
        elif fname.endswith(".avi"):
            stem = fname[:-len(".avi")]
        else:
            continue  # skip non-export files

        # Check if this stem was processed
        if stem in processed_stems:
            # Compute destination preserving subfolder structure
            rel = src_file.relative_to(exported_path)
            dst_file = output_path / rel
            if not dry_run:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_file), str(dst_file))
            log.info("  ✓ KEEP   %s", rel)
            kept += 1
        else:
            log.info("  ✗ SKIP   %s", src_file.relative_to(exported_path))
            skipped += 1

    log.info("-" * 60)
    log.info("  Kept: %d  |  Skipped: %d  |  Total: %d", kept, skipped, kept + skipped)
    if dry_run:
        log.info("  (DRY RUN — no files were copied)")
    log.info("=" * 60)
    return kept, skipped


if __name__ == "__main__":
    main()
