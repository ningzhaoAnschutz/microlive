#!/usr/bin/env python3
"""
export_multichannel_montage.py
==============================
Library and CLI for exporting multi-channel microscopy movies as
subplot montages (grayscale per channel + color merge).

Can be used as:
  1. **Importable module** — from notebooks or scripts.
  2. **CLI tool** — ``python export_multichannel_montage.py --help``
  3. **Interactive viewer** — via ``launch_viewer()`` in Jupyter.

Overlays (scale bar, timestamp) use image metadata matching the
MicroLive GUI exactly.

Author: Luis Aguilera
"""

import argparse
import re
import sys
import logging
from pathlib import Path

import numpy as np
import tifffile
import cv2
from scipy.ndimage import gaussian_filter

# MicroLive imports
try:
    import microlive.microscopy as mi
    from matplotlib_scalebar.scalebar import ScaleBar
except ImportError as exc:
    sys.exit(
        f"Could not import MicroLive or matplotlib-scalebar: {exc}\n"
        "Make sure the 'microlive' environment is activated."
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.backends.backend_agg
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
# Constants
# ---------------------------------------------------------------------------
COLOR_MAP = {
    'Green':   (0.0, 1.0, 0.0),
    'Magenta': (1.0, 0.0, 1.0),
    'Cyan':    (0.0, 1.0, 1.0),
    'Red':     (1.0, 0.0, 0.0),
    'Yellow':  (1.0, 1.0, 0.0),
    'Blue':    (0.0, 0.0, 1.0),
    'White':   (1.0, 1.0, 1.0),
}
COLOR_OPTIONS = list(COLOR_MAP.keys())
_DEFAULT_COLORS = ['Green', 'Magenta', 'Cyan', 'Red', 'Yellow', 'Blue', 'White']


def _expand_per_channel(val, C, name="value"):
    """Expand a scalar or list to a per-channel list of length C.

    Parameters
    ----------
    val : float or list
        A single value (applied to all channels) or a list of length C.
    C : int
        Number of channels.
    name : str
        Name for error messages.

    Returns
    -------
    list of float, length C.
    """
    if isinstance(val, (int, float)):
        return [float(val)] * C
    val = list(val)
    if len(val) != C:
        raise ValueError(
            f"{name} must be a scalar or a list of length {C}, "
            f"got {len(val)} values.")
    return [float(v) for v in val]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Sanitize a string for safe use as a filename component."""
    return re.sub(r'[^\w\-_. ]', '_', name)


def _results_stem(filename: str) -> str:
    """Derive the base stem from a LIF filename (split at first '.')."""
    return _sanitize(filename.split('.')[0])


def format_time_interval(value_s, reference_interval):
    """Format time exactly as the MicroLive GUI does.

    The unit (µs, ms, s) is chosen based on *reference_interval* so
    every frame in a movie uses the same unit.

    Parameters
    ----------
    value_s : float
        Time value in seconds.
    reference_interval : float
        The experiment's frame interval in seconds.  Determines the
        display unit.

    Returns
    -------
    str
    """
    if value_s is None or reference_interval is None:
        return "N/A"
    try:
        val = float(value_s)
        ref = float(reference_interval)
        if ref < 0.001:
            return f"{val * 1e6:.2f} µs"
        elif ref < 1:
            return f"{val * 1000:.2f} ms"
        elif ref < 10:
            return f"{val:.2f} s"
        else:
            return f"{val:.1f} s"
    except (TypeError, ValueError):
        return "N/A"


def normalize_frame(data_2d: np.ndarray, min_pct: float, max_pct: float,
                    sigma: float = 0.0, low_sigma: float = 0.0) -> np.ndarray:
    """Percentile-normalize a 2D image to [0, 1] with optional Gaussian
    smoothing.

    Parameters
    ----------
    data_2d : np.ndarray
        2D image (Y, X).
    min_pct, max_pct : float
        Percentile bounds for contrast rescaling.
    sigma : float
        Smoothing Gaussian sigma (0 = disabled).
    low_sigma : float
        Low-pass Gaussian sigma (0 = disabled).

    Returns
    -------
    np.ndarray
        Normalized image in [0, 1], shape (Y, X).
    """
    vmin = np.percentile(data_2d, min_pct)
    vmax = np.percentile(data_2d, max_pct)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((data_2d.astype(np.float64) - vmin) / (vmax - vmin), 0.0, 1.0)
    if low_sigma > 0:
        norm = gaussian_filter(norm, sigma=low_sigma)
    if sigma > 0:
        norm = gaussian_filter(norm, sigma=sigma)
    return norm


# ---------------------------------------------------------------------------
# LIF I/O
# ---------------------------------------------------------------------------

def read_lif_metadata(lif_path: str) -> dict:
    """Read a LIF file and return metadata without loading image data.

    Returns
    -------
    dict
        Keys: reader, list_names, pixel_xy_um, pixel_z_um,
        channel_names, num_channels, list_time_intervals, bit_depth.
    """
    reader = mi.ReadLif(
        path=str(lif_path),
        show_metadata=False,
        save_tif=False,
        save_png=False,
        format='TZYXC',
        lazy=True,
    )
    (
        _, list_names, pixel_xy_um, pixel_z_um,
        channel_names, num_channels, list_time_intervals,
        bit_depth, _, _, _,
    ) = reader.read()
    return {
        'reader': reader,
        'list_names': list_names,
        'pixel_xy_um': pixel_xy_um,
        'pixel_z_um': pixel_z_um,
        'channel_names': channel_names,
        'num_channels': num_channels,
        'list_time_intervals': list_time_intervals,
        'bit_depth': bit_depth,
    }


def print_series_table(meta: dict, lif_path: str = ""):
    """Print a formatted table of all series in a LIF file.

    Parameters
    ----------
    meta : dict
        Return value of :func:`read_lif_metadata`.
    lif_path : str, optional
        Original file path (used only for the header line).
    """
    if lif_path:
        print(f"LIF file : {Path(lif_path).name}")
    print(f"Channels : {meta['num_channels']}  →  {meta['channel_names']}")
    print(f"Pixel XY : {meta['pixel_xy_um']:.4f} µm")
    print()
    print(f"{'Idx':>4}  {'Series Name':<40}  {'dt (s)':>10}")
    print("-" * 60)
    for i, name in enumerate(meta['list_names']):
        try:
            dt = float(meta['list_time_intervals'][i])
        except (TypeError, ValueError):
            dt = 0.0
        print(f"{i:>4}  {name:<40}  {dt:>10.4f}")
    print(f"\nTotal: {len(meta['list_names'])} series")


def load_series(meta: dict, series_index: int) -> dict:
    """Load a series and pre-compute max-Z projections.

    Parameters
    ----------
    meta : dict
        Return value of :func:`read_lif_metadata`.
    series_index : int
        0-based index into the series list.

    Returns
    -------
    dict
        Keys: image_stack, maxz, T, Z, Y, X, C, series_name,
        channel_names, voxel_yx_nm, time_interval.
    """
    reader = meta['reader']
    image_stack = reader.read_scene(series_index)
    T, Z, Y, X, C = image_stack.shape
    s_name = meta['list_names'][series_index]
    ch_names = meta['channel_names']
    vox = meta['pixel_xy_um'] * 1000.0
    try:
        dt = float(meta['list_time_intervals'][series_index])
    except (TypeError, ValueError):
        dt = 0.0

    print(f"Loaded series [{series_index}]: '{s_name}'")
    print(f"  Shape : T={T}, Z={Z}, Y={Y}, X={X}, C={C}")
    print(f"  dt    : {dt:.4f} s")
    print(f"  Pixel : {vox:.1f} nm/px  ({meta['pixel_xy_um']:.4f} µm/px)")
    print(f"  Channels: {ch_names}")

    print("\nPre-computing max-Z projections …")
    maxz = np.zeros((T, Y, X, C), dtype=image_stack.dtype)
    for t in range(T):
        for c in range(C):
            maxz[t, :, :, c] = np.max(image_stack[t, :, :, :, c], axis=0)
    print("Done ✓")

    return {
        'image_stack': image_stack, 'maxz': maxz,
        'T': T, 'Z': Z, 'Y': Y, 'X': X, 'C': C,
        'series_name': s_name, 'channel_names': ch_names,
        'voxel_yx_nm': vox, 'time_interval': dt,
    }


def resolve_config(data: dict, channel_order=None, panel_titles=None):
    """Validate and apply defaults for channel_order and panel_titles.

    Parameters
    ----------
    data : dict
        Return value of :func:`load_series`.
    channel_order : list[int], optional
        Permutation of ``[0, …, C-1]``.  ``None`` → natural order.
    panel_titles : list[str], optional
        One title per panel (C channels + merge).  ``None`` → metadata
        names + ``"Merge"``.

    Returns
    -------
    (channel_order, panel_titles, n_panels)
    """
    C = data['C']
    ch_names = data['channel_names']

    if channel_order is None:
        channel_order = list(range(C))
    else:
        assert len(channel_order) == C, \
            f"channel_order must have {C} entries, got {len(channel_order)}"
        assert set(channel_order) == set(range(C)), \
            f"channel_order must be a permutation of {list(range(C))}"

    n_panels = C + 1

    if panel_titles is None:
        panel_titles = [
            ch_names[ci] if ci < len(ch_names) else f"Ch{ci}"
            for ci in channel_order
        ]
        panel_titles.append("Merge")
    else:
        assert len(panel_titles) == n_panels, \
            f"panel_titles must have {n_panels} entries, got {len(panel_titles)}"

    print(f"Channel order : {channel_order}")
    print(f"Panel titles  : {panel_titles}")
    return channel_order, panel_titles, n_panels


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _render_single_frame(maxz, t, Y, X, C, channel_order, colors,
                         panel_titles, n_panels, ch_names, voxel_yx_nm,
                         time_interval, show_scalebar, show_timestamp,
                         min_pcts, max_pcts, sigma, low_sigma,
                         fig_w, fig_h, dpi, crop=None):
    """Render one montage frame to an RGB uint8 array (internal).

    ``min_pcts`` and ``max_pcts`` are lists of length C (one per
    channel position in *channel_order*).

    ``crop`` is ``[x_min, x_max, y_min, y_max]`` or ``None``.
    """
    # Apply crop
    if crop is not None:
        x0, x1, y0, y1 = crop
        frame_data = maxz[t, y0:y1, x0:x1, :]
    else:
        frame_data = maxz[t, :, :, :]
    cY, cX = frame_data.shape[0], frame_data.shape[1]

    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor='black')
    canvas = FigureCanvas(fig)

    # Normalize each channel with its own percentile range
    normed = {}
    for pos, ci in enumerate(channel_order):
        d = frame_data[:, :, ci].astype(np.float64)
        normed[ci] = normalize_frame(d, min_pcts[pos], max_pcts[pos],
                                     sigma, low_sigma)

    # Grayscale panels
    for pos, ci in enumerate(channel_order):
        ax = fig.add_subplot(1, n_panels, pos + 1)
        ax.imshow(normed[ci], cmap='gray', vmin=0, vmax=1, aspect='equal')
        title = panel_titles[pos] if pos < len(panel_titles) else ""
        if title:
            ax.set_title(title, color='white', fontsize=10, pad=4)
        ax.axis('off')

    # Merge panel
    ax_m = fig.add_subplot(1, n_panels, n_panels)
    rgb = np.zeros((cY, cX, 3), dtype=np.float64)
    for pos, ci in enumerate(channel_order):
        color_name = colors[pos] if pos < len(colors) else 'White'
        c_rgb = COLOR_MAP.get(color_name, (1, 1, 1))
        rgb[:, :, 0] += normed[ci] * c_rgb[0]
        rgb[:, :, 1] += normed[ci] * c_rgb[1]
        rgb[:, :, 2] += normed[ci] * c_rgb[2]
    rgb = np.clip(rgb, 0, 1)
    ax_m.imshow(rgb, aspect='equal')
    merge_title = panel_titles[-1] if len(panel_titles) == n_panels else ""
    if merge_title:
        ax_m.set_title(merge_title, color='white', fontsize=10, pad=4)
    ax_m.axis('off')

    # Overlays
    all_axes = fig.axes
    if show_scalebar and voxel_yx_nm is not None and voxel_yx_nm > 0:
        um_px = voxel_yx_nm / 1000.0
        for ax in all_axes:
            ax.add_artist(ScaleBar(
                um_px, units='um', length_fraction=0.2,
                location='lower right', box_color='black',
                color='white', font_properties={'size': 10}))
    if show_timestamp and time_interval is not None and time_interval > 0:
        ts = format_time_interval(t * time_interval, time_interval)
        all_axes[0].text(5, 5, ts, color='white', fontsize=12,
                         backgroundcolor='black', va='top', ha='left')

    fig.subplots_adjust(left=0.01, right=0.99, top=0.90,
                        bottom=0.05, wspace=0.05)
    canvas.draw()
    buf = canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return arr


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_montage_from_data(
    data: dict,
    lif_path: str,
    channel_order: list,
    panel_titles: list,
    n_panels: int,
    colors: list,
    output_folder: str,
    export_avi: bool = True,
    export_tif: bool = True,
    show_timestamp: bool = False,
    show_scalebar: bool = False,
    fps: int = 10,
    min_percentile=0.1,
    max_percentile=99.95,
    sigma: float = 0.7,
    low_sigma: float = 0.15,
    dpi: int = 150,
    panel_width_inches: float = 3.0,
    crop=None,
):
    """Export a montage movie from pre-loaded data.

    Parameters
    ----------
    data : dict
        Return value of :func:`load_series`.
    lif_path : str
        Original LIF path (used for output naming).
    channel_order : list[int]
        Channel display order.
    panel_titles : list[str]
        Panel titles (len = n_panels).
    n_panels : int
        Number of panels (C + 1).
    colors : list[str]
        Color name per channel position.
    output_folder : str
        Destination directory.
    export_avi, export_tif : bool
        Which formats to write.
    show_timestamp, show_scalebar : bool
        Overlay flags.
    min_percentile : float or list[float]
        Min percentile(s) for contrast.  A single float applies to all
        channels; a list of length C sets per-channel values.
    max_percentile : float or list[float]
        Max percentile(s) for contrast (same rules).
    crop : list[int] or None
        ``[x_min, x_max, y_min, y_max]`` pixel ROI.  ``None`` = full image.
    fps, sigma, low_sigma, dpi, panel_width_inches : numeric
        Rendering parameters.
    """
    T = data['T']
    Y, X, C = data['Y'], data['X'], data['C']
    maxz = data['maxz']
    ch_names = data['channel_names']
    voxel_yx_nm = data['voxel_yx_nm']
    time_interval = data['time_interval']

    # Expand scalars to per-channel lists
    min_pcts = _expand_per_channel(min_percentile, C, 'min_percentile')
    max_pcts = _expand_per_channel(max_percentile, C, 'max_percentile')

    # Effective dimensions after crop
    if crop is not None:
        x0, x1, y0, y1 = crop
        eY, eX = y1 - y0, x1 - x0
    else:
        eY, eX = Y, X

    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    lif_stem = _sanitize(Path(lif_path).name.split('.')[0])
    s_name   = _sanitize(data['series_name'])
    base     = f"{lif_stem}_{s_name}_montage"

    pw = panel_width_inches
    aspect = eY / eX
    fig_w = pw * n_panels
    fig_h = pw * aspect

    print(f"Exporting {T} frames …")
    print(f"  Sigma      : {sigma}")
    print(f"  Low sigma  : {low_sigma}")
    print(f"  Crop       : {crop if crop else 'None (full image)'}")
    if crop:
        print(f"  Crop size  : {eX}×{eY}  (from {X}×{Y})")
    for pos, ci in enumerate(channel_order):
        ch_label = ch_names[ci] if ci < len(ch_names) else f"Ch{ci}"
        print(f"  {ch_label:12s}: color={colors[pos]}, "
              f"min_pct={min_pcts[pos]}, max_pct={max_pcts[pos]}")
    print(f"  Ch. order  : {channel_order}")
    print(f"  Titles     : {panel_titles}")
    print(f"  Scale bar  : {show_scalebar}")
    print(f"  Timestamp  : {show_timestamp}")
    print()

    rendered = []
    for t in range(T):
        if t == 0 or t == T - 1 or (t + 1) % 10 == 0:
            print(f"  Frame {t + 1:>4d} / {T}")
        arr = _render_single_frame(
            maxz, t, Y, X, C, channel_order, colors,
            panel_titles, n_panels, ch_names, voxel_yx_nm,
            time_interval, show_scalebar, show_timestamp,
            min_pcts, max_pcts, sigma, low_sigma,
            fig_w, fig_h, dpi, crop=crop)
        rendered.append(arr)

    if export_avi and rendered:
        avi_path = out_dir / f"{base}.avi"
        h, w = rendered[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(str(avi_path), fourcc, fps,
                                 (w, h), isColor=True)
        for frm in rendered:
            writer.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"\n✅ AVI → {avi_path}")
        print(f"   {len(rendered)} frames, {w}×{h}, {fps} fps")

    if export_tif and rendered:
        tif_path = out_dir / f"{base}.tif"
        stack = np.stack(rendered, axis=0)
        tifffile.imwrite(str(tif_path), stack, metadata={'axes': 'TYXS'})
        print(f"✅ TIF → {tif_path}")
        print(f"   {len(rendered)} pages, {stack.shape[2]}×{stack.shape[1]}")

    print("\nDone ✓")


def export_montage(
    lif_path: str,
    series_index: int,
    output_folder: str,
    colors: list = None,
    channel_order: list = None,
    panel_titles: list = None,
    **kwargs,
):
    """Export a montage movie directly from a LIF file (convenience).

    Reads the LIF, loads the series, resolves config, and exports.
    All extra ``kwargs`` are forwarded to
    :func:`export_montage_from_data`.
    """
    meta = read_lif_metadata(lif_path)
    data = load_series(meta, series_index)

    if colors is None:
        colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
                  for i in range(data['C'])]
    # Normalise to title-case for COLOR_MAP lookup
    colors = [c.title() if isinstance(c, str) else c for c in colors]

    channel_order, panel_titles, n_panels = resolve_config(
        data, channel_order, panel_titles)

    export_montage_from_data(
        data=data,
        lif_path=lif_path,
        channel_order=channel_order,
        panel_titles=panel_titles,
        n_panels=n_panels,
        colors=colors,
        output_folder=output_folder,
        **kwargs,
    )


def export_montage_all_series(lif_path: str, output_folder: str, **kwargs):
    """Export montage movies for ALL series in a LIF file."""
    meta = read_lif_metadata(lif_path)
    n = len(meta['list_names'])
    log.info("Exporting montage for all %d series …", n)
    results = []
    for idx in range(n):
        try:
            export_montage(
                lif_path=lif_path,
                series_index=idx,
                output_folder=output_folder,
                **kwargs,
            )
            results.append(idx)
        except Exception as exc:
            log.error("  Series %d failed: %s", idx, exc, exc_info=True)
    log.info("Batch complete: %d / %d series exported.", len(results), n)


# ---------------------------------------------------------------------------
# Interactive viewer (requires ipywidgets — Jupyter only)
# ---------------------------------------------------------------------------

def launch_viewer(data: dict, channel_order: list, panel_titles: list,
                  n_panels: int, crop=None) -> dict:
    """Build an ipywidgets interactive montage viewer.

    Call from a Jupyter notebook / VSCode interactive Python.

    Parameters
    ----------
    data : dict
        Return value of :func:`load_series`.
    channel_order, panel_titles : list
        Return values of :func:`resolve_config`.
    n_panels : int
        Number of panels.
    crop : list[int] or None
        ``[x_min, x_max, y_min, y_max]`` pixel ROI.  ``None`` = full.

    Returns
    -------
    dict
        Widget references (pass to :func:`export_from_viewer`).
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    T = data['T']
    Y, X, C = data['Y'], data['X'], data['C']
    maxz = data['maxz']
    ch_names = data['channel_names']
    voxel_yx_nm = data['voxel_yx_nm']
    time_interval = data['time_interval']

    _style = {'description_width': '110px'}
    _sl = widgets.Layout(width='420px')

    w_frame = widgets.IntSlider(
        min=0, max=max(T - 1, 0), value=0,
        description='Frame:', style=_style, layout=_sl,
        continuous_update=False)
    w_sigma = widgets.FloatSlider(
        min=0.0, max=5.0, value=0.7, step=0.05,
        description='Sigma:', style=_style, layout=_sl,
        continuous_update=False)
    w_low_sigma = widgets.FloatSlider(
        min=0.0, max=2.0, value=0.15, step=0.05,
        description='Low sigma:', style=_style, layout=_sl,
        continuous_update=False)

    # Per-channel min/max percentile sliders
    w_min_pcts = []
    w_max_pcts = []
    for pos, ci in enumerate(channel_order):
        ch_label = ch_names[ci] if ci < len(ch_names) else f"Ch{ci}"
        w_min = widgets.FloatSlider(
            min=0.0, max=50.0, value=0.1, step=0.1,
            description=f'{ch_label} min%:', style=_style, layout=_sl,
            continuous_update=False)
        w_max = widgets.FloatSlider(
            min=50.0, max=100.0, value=99.95, step=0.05,
            description=f'{ch_label} max%:', style=_style, layout=_sl,
            continuous_update=False)
        w_min_pcts.append(w_min)
        w_max_pcts.append(w_max)

    w_scalebar = widgets.Checkbox(
        value=False, description='Show Scalebar', style=_style)
    w_timestamp = widgets.Checkbox(
        value=False, description='Show Time Stamp', style=_style)

    w_colors = []
    for pos, ci in enumerate(channel_order):
        ch_label = ch_names[ci] if ci < len(ch_names) else f"Ch{ci}"
        w = widgets.Dropdown(
            options=COLOR_OPTIONS,
            value=_DEFAULT_COLORS[pos % len(_DEFAULT_COLORS)],
            description=f'{ch_label} color:', style=_style,
            layout=widgets.Layout(width='280px'))
        w_colors.append(w)

    plot_out = widgets.Output()

    def _render(**kw):
        frame     = kw['frame']
        sigma     = kw['sigma']
        low_sigma = kw['low_sigma']
        show_sb   = kw['show_sb']
        show_ts   = kw['show_ts']
        colors    = [kw[f'c{i}'] for i in range(C)]
        min_pcts  = [kw[f'mn{i}'] for i in range(C)]
        max_pcts  = [kw[f'mx{i}'] for i in range(C)]

        with plot_out:
            clear_output(wait=True)

            # Apply crop
            if crop is not None:
                cx0, cx1, cy0, cy1 = crop
                fdata = maxz[frame, cy0:cy1, cx0:cx1, :]
            else:
                fdata = maxz[frame, :, :, :]
            cY, cX = fdata.shape[0], fdata.shape[1]

            aspect = cY / cX
            pw = 4.0
            fig, axes = plt.subplots(
                1, n_panels, figsize=(pw * n_panels, pw * aspect + 0.8),
                facecolor='black')
            if n_panels == 1:
                axes = [axes]

            normed = {}
            for pos, ci in enumerate(channel_order):
                d = fdata[:, :, ci].astype(np.float64)
                normed[ci] = normalize_frame(d, min_pcts[pos], max_pcts[pos],
                                             sigma, low_sigma)

            for pos, ci in enumerate(channel_order):
                ax = axes[pos]
                ax.imshow(normed[ci], cmap='gray', vmin=0, vmax=1,
                          aspect='equal')
                title = panel_titles[pos] if pos < len(panel_titles) else ""
                if title:
                    ax.set_title(title, color='white', fontsize=12,
                                 fontweight='bold', pad=6)
                ax.axis('off')
                ax.set_facecolor('black')

            ax_m = axes[-1]
            rgb = np.zeros((cY, cX, 3), dtype=np.float64)
            legend_parts = []
            for pos, ci in enumerate(channel_order):
                c_rgb = COLOR_MAP.get(colors[pos], (1, 1, 1))
                rgb[:, :, 0] += normed[ci] * c_rgb[0]
                rgb[:, :, 1] += normed[ci] * c_rgb[1]
                rgb[:, :, 2] += normed[ci] * c_rgb[2]
                ch_label = (ch_names[ci] if ci < len(ch_names)
                            else f"Ch{ci}")
                legend_parts.append(f"{colors[pos]}: {ch_label}")
            rgb = np.clip(rgb, 0, 1)
            ax_m.imshow(rgb, aspect='equal')
            mt = panel_titles[-1] if len(panel_titles) == n_panels else ""
            if mt:
                ax_m.set_title(mt, color='white', fontsize=12,
                               fontweight='bold', pad=6)
            ax_m.axis('off')
            ax_m.set_facecolor('black')
            ax_m.text(0.5, -0.02, '  |  '.join(legend_parts),
                      transform=ax_m.transAxes, ha='center', va='top',
                      color='white', fontsize=8,
                      bbox=dict(facecolor='black', alpha=0.7, pad=3,
                                boxstyle='round,pad=0.3'))

            if show_sb and voxel_yx_nm > 0:
                um_px = voxel_yx_nm / 1000.0
                for ax in axes:
                    ax.add_artist(ScaleBar(
                        um_px, units='um', length_fraction=0.2,
                        location='lower right', box_color='black',
                        color='white', font_properties={'size': 10}))
            if show_ts and time_interval > 0:
                ts = format_time_interval(
                    frame * time_interval, time_interval)
                axes[0].text(5, 5, ts, color='white', fontsize=12,
                             backgroundcolor='black', va='top', ha='left')

            fig.subplots_adjust(left=0.01, right=0.99, top=0.88,
                                bottom=0.08, wspace=0.05)
            plt.show()

    wd = {'frame': w_frame, 'sigma': w_sigma, 'low_sigma': w_low_sigma,
          'show_sb': w_scalebar, 'show_ts': w_timestamp}
    for i, w in enumerate(w_colors):
        wd[f'c{i}'] = w
    for i, w in enumerate(w_min_pcts):
        wd[f'mn{i}'] = w
    for i, w in enumerate(w_max_pcts):
        wd[f'mx{i}'] = w
    interactive_out = widgets.interactive_output(_render, wd)

    # Build per-channel contrast section
    _contrast_widgets = []
    for pos, ci in enumerate(channel_order):
        ch_label = ch_names[ci] if ci < len(ch_names) else f"Ch{ci}"
        _contrast_widgets.append(
            widgets.HTML(f"<b style='color:#aaa; font-size:11px'>{ch_label}</b>"))
        _contrast_widgets.append(w_min_pcts[pos])
        _contrast_widgets.append(w_max_pcts[pos])

    ctrl_display = widgets.VBox([
        widgets.HTML("<h4 style='margin:4px 0'>📐 Display</h4>"),
        w_frame, w_sigma, w_low_sigma,
        widgets.HTML("<h4 style='margin:4px 0'>📊 Per-Channel Contrast</h4>"),
        *_contrast_widgets])
    ctrl_colors = widgets.VBox([
        widgets.HTML("<h4 style='margin:4px 0'>🎨 Merge Colors</h4>"),
        *w_colors])
    ctrl_overlays = widgets.VBox([
        widgets.HTML(
            "<h4 style='margin:4px 0'>🏷 Overlays (from metadata)</h4>"),
        w_scalebar, w_timestamp,
        widgets.HTML(
            f"<div style='color:#999; font-size:11px; margin-top:4px'>"
            f"Pixel size: {voxel_yx_nm:.1f} nm/px<br>"
            f"Time interval: {time_interval:.4f} s</div>")])
    controls = widgets.VBox(
        [ctrl_display, ctrl_colors, ctrl_overlays],
        layout=widgets.Layout(width='440px', padding='8px'))
    display(widgets.HBox([controls, plot_out]))
    display(interactive_out)

    return {
        'w_frame': w_frame, 'w_sigma': w_sigma,
        'w_low_sigma': w_low_sigma,
        'w_min_pcts': w_min_pcts, 'w_max_pcts': w_max_pcts,
        'w_scalebar': w_scalebar,
        'w_timestamp': w_timestamp, 'w_colors': w_colors,
    }


def export_from_viewer(
    data: dict,
    lif_path: str,
    channel_order: list,
    panel_titles: list,
    n_panels: int,
    wdg: dict,
    output_folder: str,
    export_avi: bool = True,
    export_tif: bool = True,
    fps: int = 10,
    dpi: int = 150,
    panel_width_inches: float = 3.0,
    crop=None,
):
    """Export using the current widget settings from :func:`launch_viewer`.

    Reads sigma, percentiles, colors, and overlay flags directly from
    the widget dict.
    """
    colors    = [w.value for w in wdg['w_colors']]
    min_pcts  = [w.value for w in wdg['w_min_pcts']]
    max_pcts  = [w.value for w in wdg['w_max_pcts']]
    export_montage_from_data(
        data=data,
        lif_path=lif_path,
        channel_order=channel_order,
        panel_titles=panel_titles,
        n_panels=n_panels,
        colors=colors,
        output_folder=output_folder,
        export_avi=export_avi,
        export_tif=export_tif,
        show_timestamp=wdg['w_timestamp'].value,
        show_scalebar=wdg['w_scalebar'].value,
        fps=fps,
        min_percentile=min_pcts,
        max_percentile=max_pcts,
        sigma=wdg['w_sigma'].value,
        low_sigma=wdg['w_low_sigma'].value,
        dpi=dpi,
        panel_width_inches=panel_width_inches,
        crop=crop,
    )


def export_all_from_viewer(
    meta: dict,
    lif_path: str,
    output_folder: str,
    channel_order=None,
    panel_titles=None,
    export_avi: bool = True,
    export_tif: bool = True,
    fps: int = 10,
    dpi: int = 150,
    panel_width_inches: float = 3.0,
    show_timestamp: bool = False,
    show_scalebar: bool = False,
    min_percentile: float = 0.1,
    max_percentile: float = 99.9,
    sigma: float = 0.7,
    low_sigma: float = 0.15,
    crop=None,
):
    """Batch-export montage movies for ALL series in a LIF file.

    Called from ``montage_run.py`` when ``SERIES_INDEX = None``.
    Uses default rendering settings (no interactive widgets).

    Parameters
    ----------
    meta : dict
        Return value of :func:`read_lif_metadata`.
    lif_path : str
        Original LIF file path.
    output_folder : str
        Destination directory.
    channel_order, panel_titles : optional
        Passed to :func:`resolve_config` for each series.
    """
    n = len(meta['list_names'])
    print(f"\n{'='*60}")
    print(f"  Batch export: {n} series from {Path(lif_path).name}")
    print(f"{'='*60}\n")

    ok = 0
    for idx in range(n):
        print(f"\n── Series {idx}/{n-1}: {meta['list_names'][idx]} ──")
        try:
            d = load_series(meta, idx)
            co, pt, np_ = resolve_config(d, channel_order, panel_titles)
            colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
                      for i in range(d['C'])]
            export_montage_from_data(
                data=d,
                lif_path=lif_path,
                channel_order=co,
                panel_titles=pt,
                n_panels=np_,
                colors=colors,
                output_folder=output_folder,
                export_avi=export_avi,
                export_tif=export_tif,
                show_timestamp=show_timestamp,
                show_scalebar=show_scalebar,
                fps=fps,
                min_percentile=min_percentile,
                max_percentile=max_percentile,
                sigma=sigma,
                low_sigma=low_sigma,
                dpi=dpi,
                panel_width_inches=panel_width_inches,
                crop=crop,
            )
            ok += 1
        except Exception as exc:
            print(f"  ⚠ Series {idx} failed: {exc}")

    print(f"\n{'='*60}")
    print(f"  Batch complete: {ok} / {n} series exported ✓")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export multi-channel LIF movies as montage panels "
                    "(grayscale per channel + color merge).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
  # List all series:
  python export_multichannel_montage.py \\
      --lif_path /path/to/file.lif --list_series

  # Export series 0, ch0=green, ch1=magenta:
  python export_multichannel_montage.py \\
      --lif_path /path/to/file.lif \\
      --series 0 \\
      --colors Green Magenta \\
      --output_folder /path/to/output

  # Swap channel order and use custom titles:
  python export_multichannel_montage.py \\
      --lif_path /path/to/file.lif \\
      --series 0 \\
      --colors Magenta Green \\
      --channel_order 1 0 \\
      --titles "DUSP1 mRNA" "GR" "Merge" \\
      --show_timestamp --show_scalebar

  # Export all series:
  python export_multichannel_montage.py \\
      --lif_path /path/to/file.lif \\
      --series all \\
      --colors Green Magenta
""",
    )
    parser.add_argument("--lif_path", type=str, required=True)
    parser.add_argument("--list_series", action="store_true")
    parser.add_argument("--series", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--colors", type=str, nargs='+', default=None)
    parser.add_argument("--channel_order", type=int, nargs='+', default=None)
    parser.add_argument("--titles", type=str, nargs='+', default=None)
    parser.add_argument("--no_avi", action="store_true")
    parser.add_argument("--no_tif", action="store_true")
    parser.add_argument("--show_timestamp", action="store_true")
    parser.add_argument("--show_scalebar", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min_percentile", type=float, default=0.1)
    parser.add_argument("--max_percentile", type=float, default=99.9)
    parser.add_argument("--sigma", type=float, default=0.7)
    parser.add_argument("--low_sigma", type=float, default=0.15)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--panel_width", type=float, default=3.0)

    args = parser.parse_args()

    if args.list_series:
        meta = read_lif_metadata(args.lif_path)
        print_series_table(meta, args.lif_path)
        return

    if args.series is None:
        parser.error("Specify --series INDEX or --series all "
                     "(or use --list_series).")

    output_folder = args.output_folder or str(
        Path(args.lif_path).parent / "montage_export")

    export_kwargs = dict(
        output_folder=output_folder,
        colors=args.colors,
        channel_order=args.channel_order,
        panel_titles=args.titles,
        export_avi=not args.no_avi,
        export_tif=not args.no_tif,
        show_timestamp=args.show_timestamp,
        show_scalebar=args.show_scalebar,
        fps=args.fps,
        min_percentile=args.min_percentile,
        max_percentile=args.max_percentile,
        sigma=args.sigma,
        low_sigma=args.low_sigma,
        dpi=args.dpi,
        panel_width_inches=args.panel_width,
    )

    if args.series.lower() == "all":
        export_montage_all_series(
            lif_path=args.lif_path, **export_kwargs)
    else:
        try:
            idx = int(args.series)
        except ValueError:
            parser.error(f"--series must be an integer or 'all'.")
        export_montage(
            lif_path=args.lif_path, series_index=idx, **export_kwargs)


if __name__ == "__main__":
    main()
