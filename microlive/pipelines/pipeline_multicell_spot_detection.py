"""Multi-cell spot detection pipeline for fixed-cell microscopy analysis.

This module provides an automated batch pipeline for processing fixed-cell
microscopy datasets. It segments cells (cytosol + nucleus), detects diffraction-
limited spots in one or more channels, quantifies spots per compartment per
cell, and emits both machine-readable tables and visual outputs for manual
verification.

The pipeline supports:
    - Single TIFF files, Leica LIF files, Zeiss CZI files, or folders
      containing any mix of the three
    - Cellpose-based segmentation (cytosol and/or nucleus)
    - Multi-channel BigFISH 3D spot detection
    - Per-cell quantification (spots/clusters in nucleus vs cytosol)
    - Per-image visual outputs + per-cell cropped views
    - Batch summary CSVs, PDF report, and a reproducible metadata sidecar

Example:
    >>> from pathlib import Path
    >>> from microlive.pipelines.pipeline_multicell_spot_detection import (
    ...     pipeline_multicell_spot_detection,
    ... )
    >>> df_spots, df_cells, summary = pipeline_multicell_spot_detection(
    ...     data_path=Path("~/data/Vo_2023/ROI003.tif"),
    ...     input_axis_order=['Z', 'Y', 'X', 'C'],
    ...     pixel_size_xy_nm=160,
    ...     voxel_size_z_nm=500,
    ...     channels_spots=[1, 3],
    ...     channel_cytosol=2,
    ...     channel_nucleus=0,
    ...     diameter_cytosol=120,
    ...     diameter_nucleus=60,
    ... )

Authors:
    Luis U. Aguilera, Ning Zhao

License:
    GPL v3
"""
from microlive.imports import *

# =============================================================================
# Constants
# =============================================================================

DEFAULT_PIXEL_SIZE_XY_NM = 160
DEFAULT_VOXEL_SIZE_Z_NM = 500
DEFAULT_TIME_INTERVAL_SEC = 1.0
DEFAULT_YX_SPOT_SIZE_PX = 5
DEFAULT_Z_SPOT_SIZE_PX = 3
DEFAULT_CLUSTER_RADIUS_NM = 500
DEFAULT_MAX_SPOTS_FOR_THRESHOLD = 100000

# Cached plot colormaps for visualization; cycles if N channels > len()
DEFAULT_CHANNEL_COLORMAPS = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'Greys']

TIFF_SUFFIXES = {'.tif', '.tiff'}
LIF_SUFFIXES = {'.lif'}
CZI_SUFFIXES = {'.czi'}


# =============================================================================
# Input loading
# =============================================================================

def load_images(data_path, input_axis_order, pixel_size_xy_nm, voxel_size_z_nm):
    """Load one or more images from a TIFF, LIF, CZI file, or folder.

    Args:
        data_path: Path to a .tif/.tiff, .lif, or .czi file, or a directory
            that contains any mix of these formats.
        input_axis_order: Axis order for TIFF inputs (e.g. ['Z', 'Y', 'X', 'C']).
            Ignored for LIF and CZI. Required for TIFF unless file is already TZYXC.
        pixel_size_xy_nm: XY pixel size in nm (user-provided, authoritative for TIFF).
        voxel_size_z_nm: Z voxel size in nm.

    Returns:
        tuple: (images, names, pixel_xy_nm, voxel_z_nm, channel_names)
            - images: list of np.ndarray, each [T, Z, Y, X, C]
            - names: list of str, one per image
            - pixel_xy_nm: float (from LIF/CZI metadata if available, else user-supplied)
            - voxel_z_nm: float
            - channel_names: list[str] or None
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    # --- LIF file ---
    if data_path.is_file() and data_path.suffix.lower() in LIF_SUFFIXES:
        return load_bioio(mi.ReadLif, data_path, pixel_size_xy_nm, voxel_size_z_nm)

    # --- CZI file ---
    if data_path.is_file() and data_path.suffix.lower() in CZI_SUFFIXES:
        return load_bioio(mi.ReadCzi, data_path, pixel_size_xy_nm, voxel_size_z_nm)

    # --- Single TIFF ---
    if data_path.is_file() and data_path.suffix.lower() in TIFF_SUFFIXES:
        img = load_single_tiff(data_path, input_axis_order)
        return [img], [data_path.stem], pixel_size_xy_nm, voxel_size_z_nm, None

    # --- Folder: mix of TIFF / LIF / CZI accepted ---
    if data_path.is_dir():
        all_suffixes = TIFF_SUFFIXES | LIF_SUFFIXES | CZI_SUFFIXES
        file_paths = sorted([p for p in data_path.iterdir()
                             if p.suffix.lower() in all_suffixes])
        if not file_paths:
            raise FileNotFoundError(
                f"No TIFF/LIF/CZI files found in directory: {data_path}"
            )
        images, names = [], []
        pixel_xy_nm, voxel_z_nm = pixel_size_xy_nm, voxel_size_z_nm
        channel_names = None
        for p in file_paths:
            suf = p.suffix.lower()
            # Include the file extension in the fallback name so that
            # ``sample.tif`` and ``sample.lif`` in the same folder land in
            # distinct output sub-folders instead of overwriting each other.
            base = f"{p.stem}_{p.suffix.lstrip('.').lower()}"
            if suf in TIFF_SUFFIXES:
                images.append(load_single_tiff(p, input_axis_order))
                names.append(base)
            else:
                reader_cls = mi.ReadLif if suf in LIF_SUFFIXES else mi.ReadCzi
                sub_images, sub_names, sub_xy, sub_z, sub_chn = load_bioio(
                    reader_cls, p, pixel_size_xy_nm, voxel_size_z_nm
                )
                images.extend(sub_images)
                # Multi-scene files: prefix with stem + '__<scene>'; single
                # scene: use the extension-qualified base.
                names.extend([f"{p.stem}__{n}" if len(sub_names) > 1 else base
                              for n in sub_names])
                # First multi-scene file wins for pixel size + channel names
                if channel_names is None and sub_chn is not None:
                    channel_names = sub_chn
                    pixel_xy_nm, voxel_z_nm = sub_xy, sub_z
        return images, names, pixel_xy_nm, voxel_z_nm, channel_names

    raise ValueError(
        f"Unsupported data_path: {data_path}. Expected a .tif/.tiff, .lif, "
        "or .czi file, or a directory containing any of these."
    )


_UNSAFE_FS_CHARS = '/\\:*?"<>|'


def sanitize_name(name):
    """Replace filesystem-unsafe characters so the string is safe as a folder
    name. Scene names in LIF/CZI files can legitimately contain '/', ':' etc.,
    which would otherwise create nested directories or fail on Windows.
    """
    s = str(name)
    for ch in _UNSAFE_FS_CHARS:
        s = s.replace(ch, '_')
    return s.strip() or 'unnamed'


def load_bioio(reader_cls, path, pixel_size_xy_nm, voxel_size_z_nm):
    """Shared loader for LIF/CZI readers - both return the same 11-tuple."""
    reader = reader_cls(path, show_metadata=False, save_tif=False,
                        save_png=False, format='TZYXC')
    result = reader.read()
    list_images, list_names = result[0], result[1]
    list_names = [sanitize_name(n) for n in list_names]
    pixel_xy_um, voxel_z_um = result[2], result[3]
    channel_names = result[4]
    # LIF/CZI metadata reports in microns; convert to nm
    pixel_xy_nm = float(pixel_xy_um) * 1000.0 if pixel_xy_um else pixel_size_xy_nm
    voxel_z_nm = float(voxel_z_um) * 1000.0 if voxel_z_um else voxel_size_z_nm
    return list_images, list_names, pixel_xy_nm, voxel_z_nm, channel_names


def load_single_tiff(tiff_path, input_axis_order):
    """Read a single TIFF and reshape to [T, Z, Y, X, C] via ConvertFormat."""
    raw = tifffile.imread(str(tiff_path))
    if input_axis_order is None:
        # Best-effort fallback for common ZYXC layout with a warning
        if raw.ndim == 4:
            warnings.warn(
                f"input_axis_order not provided for {tiff_path.name}; "
                "assuming ['Z', 'Y', 'X', 'C']."
            )
            input_axis_order = ['Z', 'Y', 'X', 'C']
        elif raw.ndim == 5:
            warnings.warn(
                f"input_axis_order not provided for {tiff_path.name}; "
                "assuming ['T', 'Z', 'Y', 'X', 'C']."
            )
            input_axis_order = ['T', 'Z', 'Y', 'X', 'C']
        else:
            raise ValueError(
                f"Cannot infer axis order for {tiff_path.name} with shape "
                f"{raw.shape}; please provide input_axis_order."
            )
    converter = mi.ConvertFormat(
        image=raw,
        original_order=list(input_axis_order),
        desired_order=['T', 'Z', 'Y', 'X', 'C'],
    )
    return converter.convert()


# =============================================================================
# Segmentation
# =============================================================================

def segment_cells(image_ZYXC, channel_cytosol, channel_nucleus,
                   diameter_cytosol, diameter_nucleus,
                   model_cyto_segmentation, model_nuc_segmentation,
                   pretrained_model_cyto, pretrained_model_nuc,
                   remove_cells_touching_border, min_cell_area_px):
    """Run cytosol and/or nucleus segmentation on a single image.

    Returns:
        tuple: (mask_cytosol, mask_nucleus, mask_cytosol_no_nuclei)
            Each is a labeled [Y, X] uint mask, 0 = background.
            If a compartment wasn't requested, its mask is returned as None
            (but mask_cytosol is always populated - either from the cytosol
            channel or, if only nucleus was segmented, by copying the nuclei).
    """
    if channel_cytosol is None and channel_nucleus is None:
        raise ValueError(
            "At least one of channel_cytosol or channel_nucleus must be set."
        )

    mask_cytosol = None
    mask_nucleus = None

    if channel_cytosol is not None:
        seg_cyto = mi.CellSegmentation(
            image_ZYXC,
            channels_cytosol=channel_cytosol,
            channels_nucleus=None,
            diameter_cytosol=diameter_cytosol,
            model_cyto_segmentation=model_cyto_segmentation,
            pretrained_model_cyto_segmentation=pretrained_model_cyto,
            selection_metric=None,
            running_in_pipeline=True,
            show_plot=False,
        )
        mask_cytosol, _, _ = seg_cyto.calculate_masks()

    if channel_nucleus is not None:
        seg_nuc = mi.CellSegmentation(
            image_ZYXC,
            channels_cytosol=None,
            channels_nucleus=channel_nucleus,
            diameter_nucleus=diameter_nucleus,
            model_nuc_segmentation=model_nuc_segmentation,
            pretrained_model_nuc_segmentation=pretrained_model_nuc,
            selection_metric=None,
            running_in_pipeline=True,
            show_plot=False,
        )
        _, mask_nucleus, _ = seg_nuc.calculate_masks()

    # Synchronize labels so cell IDs align across cytosol and nucleus masks
    if mask_cytosol is not None and mask_nucleus is not None \
            and np.max(mask_cytosol) > 0 and np.max(mask_nucleus) > 0:
        mask_cytosol, mask_nucleus = mi.CellSegmentation.synchronize_masks(
            mask_cytosol, mask_nucleus
        )

    # Fallback: if cytosol wasn't segmented, use nucleus as the cell boundary
    if mask_cytosol is None and mask_nucleus is not None:
        mask_cytosol = mask_nucleus.copy()

    # Decide which cell labels to drop based on the cytosol mask, then apply
    # the drop + a single shared relabel to BOTH masks so the labels stay
    # aligned (the defining invariant for downstream per-cell quantification).
    labels_to_drop = set()
    if mask_cytosol is not None and np.max(mask_cytosol) > 0:
        if remove_cells_touching_border:
            labels_to_drop.update(border_labels(mask_cytosol))
        if min_cell_area_px > 0:
            labels_to_drop.update(small_labels(mask_cytosol, min_cell_area_px))

    if labels_to_drop:
        mask_cytosol, mask_nucleus = drop_labels_and_relabel(
            [mask_cytosol, mask_nucleus], labels_to_drop
        )

    # Derive cytosol-minus-nucleus
    mask_cytosol_no_nuclei = None
    if mask_cytosol is not None:
        mask_cytosol_no_nuclei = mask_cytosol.copy()
        if mask_nucleus is not None:
            mask_cytosol_no_nuclei[mask_nucleus > 0] = 0

    return mask_cytosol, mask_nucleus, mask_cytosol_no_nuclei


def border_labels(mask):
    """Return the set of labels that touch any image border pixel."""
    labels = set()
    labels.update(np.unique(mask[0, :]))
    labels.update(np.unique(mask[-1, :]))
    labels.update(np.unique(mask[:, 0]))
    labels.update(np.unique(mask[:, -1]))
    labels.discard(0)
    return {int(v) for v in labels}


def small_labels(mask, min_area_px):
    """Return the set of labels whose pixel area is below min_area_px."""
    labels = set()
    unique, counts = np.unique(mask, return_counts=True)
    for lbl, count in zip(unique, counts):
        if lbl == 0:
            continue
        if count < min_area_px:
            labels.add(int(lbl))
    return labels


def drop_labels_and_relabel(masks, labels_to_drop):
    """Drop a set of labels from every mask and apply a shared sequential relabel.

    Using the same mapping for every mask is what keeps ``mask_cytosol == k``
    and ``mask_nucleus == k`` referring to the *same* cell after filtering.
    The new label ordering is derived from the first non-None mask.

    Args:
        masks: iterable of labeled masks (may contain None entries).
        labels_to_drop: iterable of labels to zero out in every mask.

    Returns:
        list: filtered + relabeled masks, in the same order as the input
        (None entries are preserved as None).
    """
    drop_arr = np.asarray(sorted(labels_to_drop), dtype=np.int64)

    cleaned = []
    for m in masks:
        if m is None:
            cleaned.append(None)
            continue
        c = m.copy()
        if drop_arr.size:
            c[np.isin(c, drop_arr)] = 0
        cleaned.append(c)

    primary = next((m for m in cleaned if m is not None), None)
    if primary is None:
        return cleaned

    old_labels = sorted(int(v) for v in np.unique(primary) if v != 0)
    mapping = {old: new for new, old in enumerate(old_labels, start=1)}

    # O(N) palette lookup: build a mapping array so relabeling is a single
    # advanced-index operation instead of one boolean-mask per label.
    max_label = int(np.max(primary))
    palette = np.zeros(max_label + 1, dtype=primary.dtype)
    for old, new in mapping.items():
        palette[old] = new

    result = []
    for m in cleaned:
        if m is None:
            result.append(None)
        else:
            result.append(palette[m])
    return result


# =============================================================================
# Detection + quantification
# =============================================================================

def normalize_thresholds(threshold_for_spot_detection, n_spot_channels):
    """Broadcast a scalar/list threshold spec to a list of length n_spot_channels.

    Accepted forms:
        - ``None``: auto-detect every channel.
        - scalar (int/float): use that value for every channel.
        - list[Optional[float]]: per-channel; ``None`` entries auto-detect,
          numeric entries are fixed.

    Returns a list of length ``n_spot_channels`` whose entries are floats
    (fixed threshold) or ``None`` (flag for auto-detection later).
    """
    if threshold_for_spot_detection is None:
        return [None] * n_spot_channels
    if isinstance(threshold_for_spot_detection, (int, float, np.integer, np.floating)):
        return [float(threshold_for_spot_detection)] * n_spot_channels
    thresholds = list(threshold_for_spot_detection)
    if len(thresholds) != n_spot_channels:
        raise ValueError(
            f"threshold_for_spot_detection has {len(thresholds)} entries but "
            f"channels_spots has {n_spot_channels}."
        )
    # Keep None as-is; coerce numeric entries to float
    return [None if t is None else float(t) for t in thresholds]


@contextlib.contextmanager
def suppress_plt_show():
    """Temporarily neutralize ``plt.show()`` so threshold plots don't block.

    ``mi.Utilities().calculate_threshold_for_spot_detection`` calls
    ``plt.show()`` unconditionally when ``show_plot=True`` - convenient in a
    notebook, but blocking in a batch script. This context manager replaces
    ``plt.show`` with a no-op for the duration of the call; figures are still
    rendered in memory and saved via ``plot_name``.
    """
    original = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.show = original


def _map_filtered_threshold_to_raw(image_raw, filtered_threshold, spot_radius_px):
    """Remap a LoG-filtered-space threshold to raw-intensity space.

    ``mi.SpotDetection`` treats ``threshold_for_spot_detection`` as a raw-
    intensity value and internally maps it to the filtered image via percentile
    (see microscopy.py BigFISH.__init__). Auto-thresholds computed from
    ``detection.get_elbow_values`` live in filtered space, so without this
    inverse remap the two mappings don't cancel and the effective detection
    threshold collapses to the permissive end of the filtered range.

    The percentile conventions mirror SpotDetection's internal remap so the
    round-trip is approximately identity.
    """
    try:
        rna_filtered = stack.log_filter(image_raw, spot_radius_px)
    except ValueError:
        rna_filtered = stack.remove_background_gaussian(image_raw, spot_radius_px)

    filtered_positive = rna_filtered[rna_filtered > 0]
    if filtered_positive.size == 0:
        return float(filtered_threshold)
    f_min = np.percentile(filtered_positive, 1)
    f_max = np.percentile(filtered_positive, 99.9)
    if f_max <= f_min:
        return float(filtered_threshold)

    raw_positive = image_raw[image_raw > 0]
    if raw_positive.size == 0:
        return float(filtered_threshold)
    # SpotDetection uses the full raw image for the max percentile and the
    # >0 subset for min; match that here so the round-trip is consistent.
    r_max = np.percentile(image_raw, 99.9)
    r_min = np.percentile(raw_positive, 1)

    pct = np.clip((filtered_threshold - f_min) / (f_max - f_min), 0.01, 0.99)
    return float(r_min + pct * (r_max - r_min))


def compute_kneedle_threshold(
    image_TZYXC, channel, list_voxels, yx_spot_size_in_px, z_spot_size_in_px,
    save_path=None,
):
    """Compute a spot detection threshold using the Kneedle algorithm.

    Log-compresses the spot-count curve with ``np.log1p`` before running
    Kneedle. On linear Y, the noise floor produces a near-vertical drop that
    pulls the geometric "knee" premature (too permissive); the log transform
    flattens that spike so the knee lands at the true signal/noise transition.

    The returned value is remapped from LoG-filtered space to raw-intensity
    space via ``_map_filtered_threshold_to_raw`` so it can be passed directly
    to ``mi.SpotDetection.threshold_for_spot_detection``, which expects a
    raw-intensity threshold.

    Args:
        image_TZYXC: [T, Z, Y, X, C] image array.
        channel: channel index to threshold.
        list_voxels: [voxel_z_nm, voxel_yx_nm].
        yx_spot_size_in_px, z_spot_size_in_px: spot-size hints.
        save_path: if set, saves a diagnostic PNG.

    Returns:
        float: threshold in raw-intensity space.
    """
    voxel_z, voxel_yx = list_voxels[0], list_voxels[1]
    voxel_size = (voxel_z, voxel_yx, voxel_yx)
    spot_radius_px = detection.get_object_radius_pixel(
        voxel_size_nm=voxel_size,
        object_radius_nm=(voxel_z * (z_spot_size_in_px // 2),
                          voxel_yx * (yx_spot_size_in_px // 2),
                          voxel_yx * (yx_spot_size_in_px // 2)),
        ndim=3,
    )
    spot_radius_nm = detection.get_object_radius_nm(
        voxel_size_nm=voxel_size,
        object_radius_px=spot_radius_px, ndim=3,
    )

    image_single = image_TZYXC[0, :, :, :, channel]
    thresholds, count_spots, _ = detection.get_elbow_values(
        image_single, voxel_size=voxel_size, spot_radius=spot_radius_nm,
    )

    # log1p-compress counts so the near-vertical noise-floor drop doesn't
    # skew the kneedle diagonal. Smooth, then normalize both axes to [0,1].
    y = gaussian_filter1d(np.log1p(count_spots.astype(float)), sigma=5)
    x = thresholds.astype(float)

    x_n = (x - x[0]) / (x[-1] - x[0] + 1e-12)
    y_n = (y - y[-1]) / (y[0] - y[-1] + 1e-12)

    # Perpendicular distance from each point to the line y = 1 - x
    distances = np.abs(x_n + y_n - 1.0) / np.sqrt(2)

    # Search only the first 75% to avoid tail artifacts
    search_end = len(x) * 3 // 4
    idx_knee = int(np.argmax(distances[:search_end]))
    filtered_threshold = float(thresholds[idx_knee])

    raw_threshold = _map_filtered_threshold_to_raw(
        image_single, filtered_threshold, spot_radius_px
    )

    if save_path is not None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].plot(thresholds, count_spots, color='gray', linewidth=2)
        axes[0].plot(thresholds[idx_knee], count_spots[idx_knee],
                     'o', color='green', markersize=10,
                     label=f'knee (raw thr={raw_threshold:.1f})')
        axes[0].set_yscale('log')
        axes[0].set_title(f'Number Spots (log) - Channel {channel}')
        axes[0].legend()
        axes[1].plot(thresholds[:search_end], distances[:search_end],
                     color='dodgerblue', linewidth=2)
        axes[1].axvline(thresholds[idx_knee], color='green', ls='--', lw=1.5)
        axes[1].set_title(f'Kneedle distance - Channel {channel}')
        axes[1].set_ylabel('Distance from diagonal')
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)

    return raw_threshold


def resolve_thresholds(
    image_TZYXC, channels_spots, threshold_spec,
    yx_spot_size_in_px, z_spot_size_in_px, list_voxels,
    max_spots_for_threshold, save_plot_prefix=None,
    threshold_sensitivity=1.0,
    threshold_method='elbow',
):
    """Turn a normalized threshold spec into a list of concrete floats.

    For channels where the user did not supply a numeric threshold (``None``
    entries), an automatic method is used to compute the value.

    Args:
        image_TZYXC: [T, Z, Y, X, C] image for auto-thresholding.
        channels_spots: list of channel indices.
        threshold_spec: list of length ``len(channels_spots)``; entries are
            float (fixed) or None (auto-detect).
        yx_spot_size_in_px, z_spot_size_in_px: spot-size hints.
        list_voxels: [voxel_z_nm, voxel_yx_nm].
        max_spots_for_threshold: cap for the elbow picker.
        save_plot_prefix: if set, saves a diagnostic PNG per auto channel.
        threshold_sensitivity: multiplier applied to auto-detected values.
            Only affects auto-detected channels. Defaults to 1.0.
        threshold_method: algorithm for auto-detection. Options:
            - ``'elbow'`` (default): derivative-valley method from
              ``mi.Utilities().calculate_threshold_for_spot_detection``.
              Works well for live-cell data.
            - ``'kneedle'``: Kneedle algorithm - finds the true knee point
              by maximizing distance from the curve to a diagonal line.
              More sensitive; recommended for FISH data.

    Returns:
        list[float]: thresholds aligned with ``channels_spots``.
    """
    if all(t is not None for t in threshold_spec):
        return [float(t) for t in threshold_spec]

    if threshold_method == 'kneedle':
        resolved = []
        for ch, t in zip(channels_spots, threshold_spec):
            if t is not None:
                resolved.append(float(t))
                continue
            save_path = (f"{save_plot_prefix}_channel_{ch}.png"
                         if save_plot_prefix else None)
            auto_val = compute_kneedle_threshold(
                image_TZYXC, ch, list_voxels,
                yx_spot_size_in_px, z_spot_size_in_px,
                save_path=save_path,
            )
            resolved.append(auto_val * threshold_sensitivity)
        return resolved

    # Default: elbow method via microscopy engine
    channels_needing_auto = [ch for ch, t in zip(channels_spots, threshold_spec)
                             if t is None]
    save_plots = save_plot_prefix is not None

    with suppress_plt_show():
        auto_thresholds = mi.Utilities().calculate_threshold_for_spot_detection(
            image_TZYXC,
            list_spot_size_px=[z_spot_size_in_px, yx_spot_size_in_px],
            list_voxels=list_voxels,
            channels_spots=channels_needing_auto,
            max_spots_for_threshold=max_spots_for_threshold,
            show_plot=save_plots,
            plot_name=str(save_plot_prefix) if save_plots else None,
        )
    plt.close('all')

    auto_iter = iter(auto_thresholds)
    resolved = []
    for t in threshold_spec:
        if t is None:
            auto_val = float(next(auto_iter)) * threshold_sensitivity
            resolved.append(auto_val)
        else:
            resolved.append(float(t))
    return resolved


def detect_and_quantify(
    image_TZYXC, mask_cytosol, mask_nucleus, mask_cytosol_no_nuclei,
    channels_spots, channel_cytosol, channel_nucleus,
    list_voxels, yx_spot_size_in_px, z_spot_size_in_px,
    cluster_radius_nm, maximum_spots_cluster,
    threshold_for_spot_detection, use_maximum_projection,
    use_log_filter_for_spot_detection,
    image_counter=0,
):
    """Detect spots and build a per-cell summary DataFrame.

    Returns:
        tuple: (df_spots, df_cells)
            df_spots has one row per spot (from mi.SpotDetection).
            df_cells has one row per cell with spot/cluster counts split by
            compartment and channel.
    """
    # SpotDetection expects a single ZYXC volume
    image_ZYXC = image_TZYXC[0]

    spot_detector = mi.SpotDetection(
        image=image_ZYXC,
        channels_spots=list(channels_spots),
        channels_cytosol=channel_cytosol,
        channels_nucleus=channel_nucleus,
        masks_complete_cells=mask_cytosol,
        masks_nuclei=mask_nucleus,
        masks_cytosol_no_nuclei=mask_cytosol_no_nuclei,
        list_voxels=list_voxels,
        yx_spot_size_in_px=yx_spot_size_in_px,
        z_spot_size_in_px=z_spot_size_in_px,
        cluster_radius_nm=cluster_radius_nm,
        threshold_for_spot_detection=threshold_for_spot_detection,
        show_plot=False,
        save_files=False,
        use_maximum_projection=use_maximum_projection,
        use_log_filter_for_spot_detection=use_log_filter_for_spot_detection,
        image_counter=image_counter,
    )
    df_spots, _, thresholds_used = spot_detector.get_dataframe()
    if df_spots is None:
        df_spots = pd.DataFrame()

    # Purge dummy rows injected by the engine for cells with zero spots.
    # These carry spot_id = -1 and coordinates at (-1, -1).
    if not df_spots.empty and 'spot_id' in df_spots.columns:
        df_spots = df_spots[df_spots['spot_id'] >= 0].reset_index(drop=True)

    # Optional cluster-size cap. Singletons carry ``cluster_size=1`` (not 0),
    # so filtering on cluster_size alone would drop them too; gate on
    # ``is_cluster`` instead so the cap applies only to real clusters.
    if maximum_spots_cluster is not None and not df_spots.empty \
            and 'cluster_size' in df_spots.columns \
            and 'is_cluster' in df_spots.columns:
        is_cluster = df_spots['is_cluster'] == 1
        df_spots = df_spots[
            ~is_cluster | (df_spots['cluster_size'] <= maximum_spots_cluster)
        ].reset_index(drop=True)

    df_cells = build_cell_summary(df_spots, mask_cytosol, mask_nucleus,
                                   channels_spots)
    return df_spots, df_cells, thresholds_used


def build_cell_summary(df_spots, mask_cytosol, mask_nucleus, channels_spots):
    """Count spots and clusters per cell, split by compartment and channel."""
    rows = []
    if mask_cytosol is None or np.max(mask_cytosol) == 0:
        return pd.DataFrame()

    cell_ids = [v for v in np.unique(mask_cytosol) if v != 0]

    def spots_for_cell(mask_label):
        """Return df_spots rows for the given mask label (0-indexed cell_id)."""
        if df_spots.empty:
            return df_spots
        return df_spots[df_spots['cell_id'] == mask_label - 1]

    for mask_label in cell_ids:
        cell_area = int(np.sum(mask_cytosol == mask_label))
        nuc_area = (int(np.sum(mask_nucleus == mask_label))
                    if mask_nucleus is not None else 0)
        # Clamp: synchronize_masks aligns labels but not geometry, so a nucleus
        # can have pixels outside its cytosol label (protrusion/border).
        cyto_area = max(0, cell_area - nuc_area)
        sub = spots_for_cell(mask_label)

        row = {
            'cell_id': int(mask_label - 1),
            'cell_area_px': cell_area,
            'nuc_area_px': nuc_area,
            'cyto_area_px': cyto_area,
        }

        has_nuc_col = 'is_nuc' in sub.columns
        has_cluster_col = 'is_cluster' in sub.columns
        has_size_col = 'cluster_size' in sub.columns

        total_spots_all_ch = 0
        total_clusters_all_ch = 0
        total_mrna_all_ch = 0

        for ch in channels_spots:
            ch_sub = sub[sub['spot_type'] == ch] if 'spot_type' in sub.columns else sub
            if has_cluster_col:
                singles = ch_sub[ch_sub['is_cluster'] == 0]
                clusters = ch_sub[ch_sub['is_cluster'] == 1]
            else:
                singles = ch_sub
                clusters = ch_sub.iloc[0:0]
            if has_nuc_col:
                n_s_nuc = int((singles['is_nuc'] == 1).sum())
                n_s_cyto = int((singles['is_nuc'] == 0).sum())
                n_c_nuc = int((clusters['is_nuc'] == 1).sum())
                n_c_cyto = int((clusters['is_nuc'] == 0).sum())
                rna_in_cl_nuc = (int(clusters.loc[clusters['is_nuc'] == 1,
                                                  'cluster_size'].sum())
                                 if has_size_col else 0)
                rna_in_cl_cyto = (int(clusters.loc[clusters['is_nuc'] == 0,
                                                   'cluster_size'].sum())
                                  if has_size_col else 0)
            else:
                n_s_nuc, n_s_cyto = 0, len(singles)
                n_c_nuc, n_c_cyto = 0, len(clusters)
                rna_in_cl_nuc = 0
                rna_in_cl_cyto = (int(clusters['cluster_size'].sum())
                                  if has_size_col else 0)

            # Total mRNA = single-molecule spots + sum of cluster_size over
            # clusters (the engine reports how many mRNA live inside each
            # cluster). Without this, a cluster of 50 would be counted as 0.
            mrna_nuc = n_s_nuc + rna_in_cl_nuc
            mrna_cyto = n_s_cyto + rna_in_cl_cyto

            row[f'spots_in_nuc_ch_{ch}'] = n_s_nuc
            row[f'spots_in_cyto_ch_{ch}'] = n_s_cyto
            row[f'clusters_in_nuc_ch_{ch}'] = n_c_nuc
            row[f'clusters_in_cyto_ch_{ch}'] = n_c_cyto
            row[f'total_spots_ch_{ch}'] = n_s_nuc + n_s_cyto
            row[f'total_clusters_ch_{ch}'] = n_c_nuc + n_c_cyto
            row[f'mrna_in_nuc_ch_{ch}'] = mrna_nuc
            row[f'mrna_in_cyto_ch_{ch}'] = mrna_cyto
            row[f'total_mrna_ch_{ch}'] = mrna_nuc + mrna_cyto
            total_spots_all_ch += n_s_nuc + n_s_cyto
            total_clusters_all_ch += n_c_nuc + n_c_cyto
            total_mrna_all_ch += mrna_nuc + mrna_cyto

        row['total_spots'] = total_spots_all_ch
        row['total_clusters'] = total_clusters_all_ch
        row['total_mrna'] = total_mrna_all_ch
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Visualization
# =============================================================================

def render_raw_channels(image_TZYXC, channel_names, save_path, z_index=None):
    """Save a figure with each channel + a merged RGB."""
    if z_index is None:
        z_index = image_TZYXC.shape[1] // 2
    img_slice = image_TZYXC[0, z_index, :, :, :]
    n_ch = img_slice.shape[-1]
    names = channel_names or [f'Channel {i}' for i in range(n_ch)]
    cmaps = [DEFAULT_CHANNEL_COLORMAPS[i % len(DEFAULT_CHANNEL_COLORMAPS)]
             for i in range(n_ch)]
    vmaxes = [np.percentile(img_slice[:, :, i], 99.5) for i in range(n_ch)]

    fig, axes = plt.subplots(1, n_ch + 1, figsize=(4 * (n_ch + 1), 4))
    if n_ch + 1 == 1:
        axes = np.array([axes])
    for i in range(n_ch):
        name = names[i] if names and i < len(names) else f'Channel {i}'
        axes[i].imshow(img_slice[:, :, i], cmap=cmaps[i],
                       vmin=0, vmax=vmaxes[i] if vmaxes[i] > 0 else 1)
        axes[i].set_title(str(name), fontsize=10, fontweight='bold')
        axes[i].axis('off')

    # Standard fluorescence-merge convention: ch0 → Blue (DAPI-like short
    # wavelength), ch1 → Green, last channel → Red (long wavelength, typically
    # the smFISH / Cy5 channel in multichannel setups).
    merged = np.zeros((*img_slice.shape[:2], 3), dtype=np.float32)
    if n_ch >= 1:
        merged[:, :, 2] = img_slice[:, :, 0] / max(vmaxes[0], 1)
    if n_ch >= 2:
        merged[:, :, 1] = img_slice[:, :, 1] / max(vmaxes[1], 1)
    if n_ch >= 3:
        r_idx = n_ch - 1
        merged[:, :, 0] = img_slice[:, :, r_idx] / max(vmaxes[r_idx], 1)
    merged = np.clip(merged, 0, 1)
    axes[-1].imshow(merged)
    axes[-1].set_title('Merged', fontsize=10, fontweight='bold')
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def render_segmentation(image_TZYXC, mask_cytosol, mask_nucleus,
                         mask_cytosol_no_nuclei, channel_cytosol,
                         channel_nucleus, save_path):
    """Save a 3-panel figure: cytosol, nucleus, cytosol-no-nucleus overlays."""
    z_mid = image_TZYXC.shape[1] // 2
    if channel_cytosol is not None:
        cyto_bg = np.max(image_TZYXC[0, :, :, :, channel_cytosol], axis=0)
    else:
        cyto_bg = np.max(image_TZYXC[0, :, :, :, channel_nucleus], axis=0)
    if channel_nucleus is not None:
        nuc_bg = image_TZYXC[0, z_mid, :, :, channel_nucleus]
    else:
        nuc_bg = cyto_bg

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    n_cyto = int(np.max(mask_cytosol)) if mask_cytosol is not None else 0
    n_nuc = int(np.max(mask_nucleus)) if mask_nucleus is not None else 0
    n_cno = int(np.max(mask_cytosol_no_nuclei)) if mask_cytosol_no_nuclei is not None else 0

    axes[0].imshow(cyto_bg, cmap='gray')
    if mask_cytosol is not None and n_cyto > 0:
        axes[0].imshow(np.ma.masked_where(mask_cytosol == 0, mask_cytosol),
                       cmap='tab10', alpha=0.4)
    axes[0].set_title(f'Cytosol ({n_cyto} cells)')
    axes[0].axis('off')

    axes[1].imshow(nuc_bg, cmap='gray')
    if mask_nucleus is not None and n_nuc > 0:
        axes[1].imshow(np.ma.masked_where(mask_nucleus == 0, mask_nucleus),
                       cmap='tab10', alpha=0.4)
    axes[1].set_title(f'Nucleus ({n_nuc} nuclei)')
    axes[1].axis('off')

    axes[2].imshow(cyto_bg, cmap='gray')
    if mask_cytosol_no_nuclei is not None and n_cno > 0:
        axes[2].imshow(np.ma.masked_where(mask_cytosol_no_nuclei == 0,
                                          mask_cytosol_no_nuclei),
                       cmap='tab10', alpha=0.4)
    axes[2].set_title(f'Cytosol (no nuclei) ({n_cno} regions)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def render_spots_overlay(image_TZYXC, df_spots, mask_cytosol, mask_nucleus,
                          channels_spots, save_path,
                          visualization_z_slice=None, marker_size=15):
    """Save per-channel spot overlays with mask contours and cell labels."""
    use_mip = visualization_z_slice is None
    n_ch = len(channels_spots)
    fig, axes = plt.subplots(1, n_ch, figsize=(8 * n_ch, 8), squeeze=False)
    axes = axes[0]

    for i, ch in enumerate(channels_spots):
        ax = axes[i]
        if use_mip:
            bg = np.max(image_TZYXC[0, :, :, :, ch], axis=0)
            title = f'Ch {ch} - Max projection'
            spots_to_plot = df_spots if not df_spots.empty else df_spots
        else:
            # Clamp so batches containing mixed Z depths don't IndexError
            max_z = image_TZYXC.shape[1] - 1
            z = min(max(int(visualization_z_slice), 0), max_z)
            bg = image_TZYXC[0, z, :, :, ch]
            title = f'Ch {ch} - Z = {z}'
            spots_to_plot = df_spots[df_spots['z'] == z] if not df_spots.empty else df_spots
        if not df_spots.empty and 'spot_type' in df_spots.columns:
            spots_to_plot = spots_to_plot[spots_to_plot['spot_type'] == ch]

        vmax = np.percentile(bg, 99.5)
        ax.imshow(bg, cmap='gray', vmin=0, vmax=vmax if vmax > 0 else 1)

        if not spots_to_plot.empty:
            singles = spots_to_plot[spots_to_plot.get('is_cluster', 0) == 0] \
                if 'is_cluster' in spots_to_plot.columns else spots_to_plot
            clusters = spots_to_plot[spots_to_plot.get('is_cluster', 0) == 1] \
                if 'is_cluster' in spots_to_plot.columns else spots_to_plot.iloc[0:0]
            if not singles.empty:
                ax.scatter(singles['x'], singles['y'], s=marker_size,
                           facecolors='none', edgecolors='red',
                           marker='s', linewidth=1, label='Spots')
            if not clusters.empty:
                ax.scatter(clusters['x'], clusters['y'], s=marker_size * 1.8,
                           facecolors='none', edgecolors='cyan',
                           marker='s', linewidth=1.5, label='Clusters')

        if mask_cytosol is not None:
            for lbl in np.unique(mask_cytosol):
                if lbl == 0:
                    continue
                ax.contour(mask_cytosol == lbl, levels=[0.5],
                           colors='lime', linewidths=1, alpha=0.7)
                if mask_nucleus is not None and np.any(mask_nucleus == lbl):
                    ax.contour(mask_nucleus == lbl, levels=[0.5],
                               colors='blue', linewidths=1, alpha=0.7)
            for prop in regionprops(mask_cytosol):
                y, x = prop.centroid
                ax.text(x, y, str(int(prop.label - 1)), color='lime',
                        fontsize=11, fontweight='bold', ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.5,
                                  edgecolor='none', pad=1))

        handles, labels = ax.get_legend_handles_labels()
        handles.extend([Line2D([0], [0], color='lime', lw=2),
                        Line2D([0], [0], color='blue', lw=2)])
        labels.extend(['Cytosol mask', 'Nucleus mask'])
        ax.legend(handles, labels, loc='upper right', fontsize=8)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def render_per_cell_crops(image_TZYXC, df_spots, mask_cytosol, mask_nucleus,
                           channels_spots, save_dir, crop_padding_px=10,
                           crop_mode='max_proj', pixel_size_um=None):
    """Save one PNG per cell showing cropped channels, spots, contours, and RGB.

    Each cell image contains one grayscale panel per spot channel plus an
    RGB-composite panel. Spots are drawn as circles (red = singletons,
    cyan = clusters) with a count legend. Mask contours are shown in
    magenta (cytosol) and blue (nucleus). An optional scale bar is added
    if ``pixel_size_um`` is provided.

    Args:
        image_TZYXC: [T, Z, Y, X, C] image array.
        df_spots: spots DataFrame for this image.
        mask_cytosol: labeled [Y, X] cytosol mask.
        mask_nucleus: labeled [Y, X] nucleus mask.
        channels_spots: list of channel indices.
        save_dir: directory to save per-cell PNGs.
        crop_padding_px: padding around each cell's bbox. Defaults to 10.
        crop_mode: 'max_proj' or 'z_slice'. Defaults to 'max_proj'.
        pixel_size_um: pixel size in micrometers for scale bar. None = no bar.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if mask_cytosol is None or np.max(mask_cytosol) == 0:
        return []

    saved = []
    for prop in regionprops(mask_cytosol):
        cell_id = int(prop.label - 1)
        min_y, min_x, max_y, max_x = prop.bbox
        min_y = max(0, min_y - crop_padding_px)
        min_x = max(0, min_x - crop_padding_px)
        max_y = min(image_TZYXC.shape[2], max_y + crop_padding_px)
        max_x = min(image_TZYXC.shape[3], max_x + crop_padding_px)

        if crop_mode == 'z_slice':
            z = image_TZYXC.shape[1] // 2
            crop_by_ch = [image_TZYXC[0, z, min_y:max_y, min_x:max_x, c]
                          for c in channels_spots]
        else:
            crop_by_ch = [np.max(image_TZYXC[0, :, min_y:max_y, min_x:max_x, c],
                                 axis=0) for c in channels_spots]

        cell_spots = df_spots[df_spots['cell_id'] == cell_id] \
            if not df_spots.empty else df_spots
        if crop_mode == 'z_slice' and not cell_spots.empty \
                and 'z' in cell_spots.columns:
            cell_spots = cell_spots[cell_spots['z'] == z]

        # Number of panels: per-channel + RGB composite
        n_ch = len(channels_spots)
        n_panels = n_ch + 1
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4),
                                 squeeze=False)
        axes = axes[0]

        # Grayscale per-channel panels with spots + contours + legend
        for i, (ch, crop) in enumerate(zip(channels_spots, crop_by_ch)):
            vmax = np.percentile(crop, 99.5) if crop.size else 1
            axes[i].imshow(crop, cmap='gray', vmin=0,
                           vmax=vmax if vmax > 0 else 1)

            if not cell_spots.empty:
                ch_spots = cell_spots[cell_spots['spot_type'] == ch] \
                    if 'spot_type' in cell_spots.columns else cell_spots
                if not ch_spots.empty:
                    xs = ch_spots['x'].values - min_x
                    ys = ch_spots['y'].values - min_y
                    is_cluster = (ch_spots['is_cluster'].values
                                  if 'is_cluster' in ch_spots.columns
                                  else np.zeros(len(ch_spots), dtype=int))
                    singles_x = xs[is_cluster == 0]
                    singles_y = ys[is_cluster == 0]
                    clusters_x = xs[is_cluster == 1]
                    clusters_y = ys[is_cluster == 1]
                    n_singles = int(len(singles_x))
                    n_clusters = int(len(clusters_x))
                    if n_singles > 0:
                        axes[i].scatter(
                            singles_x, singles_y, s=30,
                            facecolors='none', edgecolors='red',
                            marker='o', linewidth=1,
                            label=f'Spots: {n_singles}')
                    if n_clusters > 0:
                        axes[i].scatter(
                            clusters_x, clusters_y, s=60,
                            facecolors='none', edgecolors='cyan',
                            marker='o', linewidth=1.5,
                            label=f'Clusters: {n_clusters}')
                    if n_singles > 0 or n_clusters > 0:
                        legend = axes[i].legend(
                            loc='upper right', fontsize=7,
                            facecolor='white', framealpha=0.8)

            if mask_cytosol is not None:
                m_cyto = (mask_cytosol[min_y:max_y, min_x:max_x] == prop.label)
                axes[i].contour(m_cyto, levels=[0.5], colors='magenta',
                                linewidths=1)
            if mask_nucleus is not None:
                m_nuc = (mask_nucleus[min_y:max_y, min_x:max_x] == prop.label)
                axes[i].contour(m_nuc, levels=[0.5], colors='blue',
                                linewidths=1)

            if pixel_size_um is not None and ScaleBar is not None:
                sb = ScaleBar(pixel_size_um, units='um',
                              length_fraction=0.25, location='lower right',
                              box_color='k', color='w')
                axes[i].add_artist(sb)

            axes[i].set_title(f'Cell {cell_id} - Ch {ch}', fontsize=10)
            axes[i].axis('off')

        # RGB composite panel
        rgb_shape = crop_by_ch[0].shape
        rgb = np.zeros((*rgb_shape, 3), dtype=np.float64)
        color_map = [(0, 1, 0), (0, 0, 1), (1, 0, 0)]  # Green, Blue, Red
        for ch_idx, crop in enumerate(crop_by_ch):
            vmax = np.percentile(crop, 99.5) if crop.size else 1
            vmax = max(vmax, 1)
            normalized = crop.astype(np.float64) / vmax
            color = color_map[ch_idx % len(color_map)]
            for c_dim in range(3):
                rgb[:, :, c_dim] += normalized * color[c_dim]
        rgb = np.clip(rgb, 0, 1)
        axes[-1].imshow(rgb)
        axes[-1].set_title(f'Cell {cell_id} - RGB', fontsize=10)
        axes[-1].axis('off')

        plt.tight_layout()
        out = save_dir / f'cell_{cell_id:03d}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        saved.append(out)

    return saved


def render_all_cells_gallery(
    image_TZYXC, df_spots, mask_cytosol, mask_nucleus,
    channels_spots, save_path, crop_padding_px=10,
    crop_mode='max_proj', num_columns=10, pixel_size_um=None,
):
    """Save a single PNG grid showing all cells in paired rows.

    Row 1 (even): max-projection crop of each cell (image only).
    Row 2 (odd): same crop with detected spots, mask contours, and legend.

    This is the most impactful visualization for batch QC, showing every
    cell at a glance with its detection results side-by-side.

    Args:
        image_TZYXC: [T, Z, Y, X, C] image array.
        df_spots: spots DataFrame for this image.
        mask_cytosol: labeled [Y, X] cytosol mask.
        mask_nucleus: labeled [Y, X] nucleus mask.
        channels_spots: list of channel indices (first is used for display).
        save_path: output file path.
        crop_padding_px: padding around each cell's bbox.
        crop_mode: 'max_proj' or 'z_slice'.
        num_columns: cells per row. Defaults to 10.
        pixel_size_um: pixel size in micrometers for scale bar.
    """
    if mask_cytosol is None or np.max(mask_cytosol) == 0:
        return

    props = regionprops(mask_cytosol)
    n_cells = len(props)
    if n_cells == 0:
        return

    num_rows = int(np.ceil(n_cells / num_columns)) * 2
    fig_height = max(4, num_rows * 3)
    fig_width = num_columns * 3
    fig, axes = plt.subplots(num_rows, num_columns,
                             figsize=(fig_width, fig_height))
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    # Clear all axes
    for r in range(num_rows):
        for c in range(num_columns):
            axes[r, c].axis('off')
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    display_ch = channels_spots[0]

    for cell_idx, prop in enumerate(props):
        cell_id = int(prop.label - 1)
        row_img = (cell_idx // num_columns) * 2
        row_det = row_img + 1
        col = cell_idx % num_columns

        min_y, min_x, max_y, max_x = prop.bbox
        min_y = max(0, min_y - crop_padding_px)
        min_x = max(0, min_x - crop_padding_px)
        max_y = min(image_TZYXC.shape[2], max_y + crop_padding_px)
        max_x = min(image_TZYXC.shape[3], max_x + crop_padding_px)

        if crop_mode == 'z_slice':
            z = image_TZYXC.shape[1] // 2
            crop = image_TZYXC[0, z, min_y:max_y, min_x:max_x, display_ch]
        else:
            crop = np.max(image_TZYXC[0, :, min_y:max_y, min_x:max_x,
                                      display_ch], axis=0)

        vmax = np.percentile(crop, 99.5) if crop.size else 1
        vmax = max(vmax, 1)

        # --- Row 1: image only ---
        ax_img = axes[row_img, col]
        ax_img.imshow(crop, cmap='gray', vmin=0, vmax=vmax)
        ax_img.set_title(f'Cell {cell_id}', fontsize=8)
        if pixel_size_um is not None and ScaleBar is not None:
            sb = ScaleBar(pixel_size_um, units='um',
                          length_fraction=0.25, location='lower right',
                          box_color='k', color='w',
                          font_properties={'size': 6})
            ax_img.add_artist(sb)

        # --- Row 2: image + spots + contours ---
        ax_det = axes[row_det, col]
        ax_det.imshow(crop, cmap='gray', vmin=0, vmax=vmax)
        ax_det.set_title(f'Cell {cell_id} - Detection', fontsize=8)

        # Get spots for this cell
        cell_spots = df_spots[df_spots['cell_id'] == cell_id] \
            if not df_spots.empty else df_spots
        if crop_mode == 'z_slice' and not cell_spots.empty \
                and 'z' in cell_spots.columns:
            cell_spots = cell_spots[cell_spots['z'] == z]
        if not cell_spots.empty and 'spot_type' in cell_spots.columns:
            cell_spots = cell_spots[cell_spots['spot_type'] == display_ch]

        if not cell_spots.empty:
            xs = cell_spots['x'].values - min_x
            ys = cell_spots['y'].values - min_y
            is_cluster = (cell_spots['is_cluster'].values
                          if 'is_cluster' in cell_spots.columns
                          else np.zeros(len(cell_spots), dtype=int))
            singles_mask = is_cluster == 0
            clusters_mask = is_cluster == 1
            n_singles = int(singles_mask.sum())
            n_clusters = int(clusters_mask.sum())

            if n_singles > 0:
                ax_det.scatter(xs[singles_mask], ys[singles_mask], s=15,
                               facecolors='none', edgecolors='red',
                               marker='o', linewidth=0.7,
                               label=f'Spots: {n_singles}')
            if n_clusters > 0:
                ax_det.scatter(xs[clusters_mask], ys[clusters_mask], s=35,
                               facecolors='none', edgecolors='cyan',
                               marker='o', linewidth=1.2,
                               label=f'Clusters: {n_clusters}')
            if n_singles > 0 or n_clusters > 0:
                ax_det.legend(loc='upper right', fontsize=5,
                              facecolor='white', framealpha=0.8)

        # Mask contours
        if mask_cytosol is not None:
            m_cyto = (mask_cytosol[min_y:max_y, min_x:max_x] == prop.label)
            ax_det.contour(m_cyto, levels=[0.5], colors='magenta',
                           linewidths=0.8)
        if mask_nucleus is not None:
            m_nuc = (mask_nucleus[min_y:max_y, min_x:max_x] == prop.label)
            ax_det.contour(m_nuc, levels=[0.5], colors='blue',
                           linewidths=0.8)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def render_fov_overview(
    image_TZYXC, mask_cytosol, mask_nucleus,
    channel_nucleus, channels_spots,
    save_path, pixel_size_um=None,
    max_percentile=99.5,
):
    """Save a full field-of-view RGB composite with cell ID labels.

    Maps channels to RGB colors and overlays cell ID labels at each
    nucleus centroid. Cytosol contours are drawn in magenta, nucleus
    contours in blue. Provides spatial context for which cells were
    segmented and where they are in the field.

    Args:
        image_TZYXC: [T, Z, Y, X, C] image array.
        mask_cytosol: labeled [Y, X] cytosol mask.
        mask_nucleus: labeled [Y, X] nucleus mask.
        channel_nucleus: nucleus channel index (mapped to blue).
        channels_spots: spot channel indices (mapped to green, red).
        save_path: output file path.
        pixel_size_um: pixel size in micrometers for scale bar.
        max_percentile: upper percentile for intensity normalization.
    """
    n_channels = image_TZYXC.shape[-1]
    Y, X = image_TZYXC.shape[2], image_TZYXC.shape[3]
    rgb = np.zeros((Y, X, 3), dtype=np.float64)

    # Build max-projection per channel and assign to RGB
    spot_colors = [(0, 1, 0), (1, 0, 0)]  # Green, Red for spot channels
    for i, ch in enumerate(channels_spots):
        mip = np.max(image_TZYXC[0, :, :, :, ch], axis=0).astype(np.float64)
        vmax = np.percentile(mip, max_percentile)
        vmax = max(vmax, 1)
        normalized = np.clip(mip / vmax, 0, 1)
        color = spot_colors[i % len(spot_colors)]
        for c_dim in range(3):
            rgb[:, :, c_dim] += normalized * color[c_dim]

    if channel_nucleus is not None:
        mip_nuc = np.max(image_TZYXC[0, :, :, :, channel_nucleus],
                         axis=0).astype(np.float64)
        vmax_nuc = np.percentile(mip_nuc, max_percentile)
        vmax_nuc = max(vmax_nuc, 1)
        normalized_nuc = np.clip(mip_nuc / vmax_nuc, 0, 1)
        rgb[:, :, 2] += normalized_nuc  # Blue channel

    rgb = np.clip(rgb, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(rgb)

    # Draw mask contours and cell ID labels
    if mask_cytosol is not None:
        for prop in regionprops(mask_cytosol):
            lbl = prop.label
            ax.contour(mask_cytosol == lbl, levels=[0.5],
                       colors='magenta', linewidths=0.6, alpha=0.7)
    if mask_nucleus is not None:
        for prop in regionprops(mask_nucleus):
            lbl = prop.label
            ax.contour(mask_nucleus == lbl, levels=[0.5],
                       colors='blue', linewidths=0.6, alpha=0.7)
            # Cell ID label at nucleus centroid
            y, x = prop.centroid
            cell_id = int(lbl - 1)
            ax.text(x, y - 15, str(cell_id), color='white', fontsize=9,
                    fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.5,
                              edgecolor='none', pad=1))

    if pixel_size_um is not None and ScaleBar is not None:
        sb = ScaleBar(pixel_size_um, units='um',
                      length_fraction=0.15, location='lower right',
                      box_color='k', color='w',
                      font_properties={'size': 12})
        ax.add_artist(sb)

    ax.set_title('Field of View - All Channels', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight')
    plt.close(fig)


def render_spots_per_cell_boxplot(df_all_cells, channels_spots, save_path):
    """Box/whisker + swarm plot of spots per cell, one box per image, per channel.

    White-face boxplot with red median line, black whiskers/caps, and a
    swarm overlay of individual cell counts. Prints mean +/- SEM per image
    to stdout for quick inspection. Styled after publication-quality figures
    (Arial font, thick spines, whis=[5, 95]).
    """
    if df_all_cells.empty:
        return

    sns.set_style('ticks')
    image_names = sorted(df_all_cells['image_name'].unique())
    n_images = len(image_names)
    fig_w = max(4.0, min(3.0 * n_images, 14.0))
    fig, axes = plt.subplots(1, len(channels_spots),
                             figsize=(fig_w * len(channels_spots), 6),
                             facecolor='white', squeeze=False)
    axes = axes[0]

    for i, ch in enumerate(channels_spots):
        col = f'total_spots_ch_{ch}'
        if col not in df_all_cells.columns:
            continue

        ax = axes[i]
        ax.set_facecolor('white')

        df_plot = df_all_cells[['image_name', col]].rename(
            columns={col: 'spots', 'image_name': 'Image'})

        sns.boxplot(
            x='Image', y='spots', data=df_plot,
            order=image_names, ax=ax,
            showfliers=False, width=0.35, whis=[5, 95],
            boxprops={'facecolor': 'white', 'edgecolor': 'black'},
            medianprops={'color': 'red', 'linewidth': 2},
            whiskerprops={'color': 'black', 'linewidth': 1.5},
            capprops={'color': 'black', 'linewidth': 1.5},
            linewidth=1.5,
        )
        sns.swarmplot(
            x='Image', y='spots', data=df_plot,
            order=image_names, ax=ax,
            color='black', size=4,
        )

        ax.set_title(f'Channel {ch} - spots per cell',
                     fontsize=14, fontname='Arial', color='black')
        ax.set_xlabel('Image', fontsize=14, fontname='Arial', color='black')
        ax.set_ylabel('Spots per cell', fontsize=14, fontname='Arial', color='black')
        ax.tick_params(axis='x', rotation=45, labelsize=12, colors='black',
                       width=2, length=6)
        ax.tick_params(axis='y', labelsize=12, colors='black', width=2, length=6)
        for lbl in ax.get_xticklabels():
            lbl.set_fontname('Arial')
            lbl.set_ha('right')
        for lbl in ax.get_yticklabels():
            lbl.set_fontname('Arial')
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('black')

        # Print mean +/- SEM per image to stdout
        print(f'\n  Channel {ch} spots per cell:')
        for name in image_names:
            vals = df_plot.loc[df_plot['Image'] == name, 'spots'].dropna().values
            if len(vals) == 0:
                continue
            mean = vals.mean()
            sem = vals.std() / (len(vals) ** 0.5) if len(vals) > 1 else 0
            print(f'    {name}: {mean:.1f} +/- {sem:.2f}  (n={len(vals)})')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def render_batch_summary_plot(df_all_cells, channels_spots, save_path):
    """Box/whisker + swarm plot of total mRNA per cell across images.

    Uses mrna_in_nuc + mrna_in_cyto as the y-axis value (total mRNA per cell,
    including cluster-size contributions). Delegates to the same publication-
    quality styling as render_spots_per_cell_boxplot.
    """
    if df_all_cells.empty:
        return

    sns.set_style('ticks')
    image_names = sorted(df_all_cells['image_name'].unique())
    n_images = len(image_names)
    fig_w = max(4.0, min(3.0 * n_images, 14.0))
    fig, axes = plt.subplots(1, len(channels_spots),
                             figsize=(fig_w * len(channels_spots), 6),
                             facecolor='white', squeeze=False)
    axes = axes[0]

    for i, ch in enumerate(channels_spots):
        col_nuc = f'mrna_in_nuc_ch_{ch}'
        col_cyto = f'mrna_in_cyto_ch_{ch}'
        if col_nuc not in df_all_cells.columns:
            continue

        ax = axes[i]
        ax.set_facecolor('white')

        df_plot = df_all_cells[['image_name']].copy()
        df_plot['mRNA'] = df_all_cells[col_nuc] + df_all_cells[col_cyto]
        df_plot = df_plot.rename(columns={'image_name': 'Image'})

        sns.boxplot(
            x='Image', y='mRNA', data=df_plot,
            order=image_names, ax=ax,
            showfliers=False, width=0.35, whis=[5, 95],
            boxprops={'facecolor': 'white', 'edgecolor': 'black'},
            medianprops={'color': 'red', 'linewidth': 2},
            whiskerprops={'color': 'black', 'linewidth': 1.5},
            capprops={'color': 'black', 'linewidth': 1.5},
            linewidth=1.5,
        )
        sns.swarmplot(
            x='Image', y='mRNA', data=df_plot,
            order=image_names, ax=ax,
            color='black', size=4,
        )

        ax.set_title(f'Channel {ch} - mRNA per cell',
                     fontsize=14, fontname='Arial', color='black')
        ax.set_xlabel('Image', fontsize=14, fontname='Arial', color='black')
        ax.set_ylabel('mRNA per cell', fontsize=14, fontname='Arial', color='black')
        ax.tick_params(axis='x', rotation=45, labelsize=12, colors='black',
                       width=2, length=6)
        ax.tick_params(axis='y', labelsize=12, colors='black', width=2, length=6)
        for lbl in ax.get_xticklabels():
            lbl.set_fontname('Arial')
            lbl.set_ha('right')
        for lbl in ax.get_yticklabels():
            lbl.set_fontname('Arial')
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('black')

        print(f'\n  Channel {ch} mRNA per cell:')
        for name in image_names:
            vals = df_plot.loc[df_plot['Image'] == name, 'mRNA'].dropna().values
            if len(vals) == 0:
                continue
            mean = vals.mean()
            sem = vals.std() / (len(vals) ** 0.5) if len(vals) > 1 else 0
            print(f'    {name}: {mean:.1f} +/- {sem:.2f}  (n={len(vals)})')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Metadata
# =============================================================================

def write_metadata(txt_path, json_path, params, processed, rejected,
                    thresholds_per_image):
    """Write human-readable txt metadata and machine-readable json metadata."""
    meta = {
        'computer_user_name': computer_user_name,
        'timestamp': str(pd.Timestamp.now().round('s')),
        'processed_images': [str(n) for n in processed],
        'rejected_images': [{'name': str(r['name']), 'error': r['error']}
                            for r in rejected],
        'thresholds_used_per_image': thresholds_per_image,
        'parameters': {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in params.items()},
    }
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    with open(txt_path, 'w') as f:
        f.write(f"MicroLive multi-cell spot detection - metadata\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"User      : {meta['computer_user_name']}\n")
        f.write(f"Timestamp : {meta['timestamp']}\n\n")
        f.write("Parameters\n----------\n")
        max_key = max(len(k) for k in meta['parameters'])
        for k, v in meta['parameters'].items():
            f.write(f"{k.ljust(max_key)} : {v}\n")
        f.write(f"\nProcessed images ({len(processed)})\n{'-' * 18}\n")
        for i, name in enumerate(processed, 1):
            thr = thresholds_per_image.get(str(name), 'n/a')
            f.write(f"{i:>3}. {name}   (thresholds: {thr})\n")
        if rejected:
            f.write(f"\nRejected images ({len(rejected)})\n{'-' * 17}\n")
            for r in rejected:
                f.write(f"  - {r['name']} : {r['error']}\n")


# =============================================================================
# PDF report
# =============================================================================

def build_pdf_report(pdf_path, per_image_png_paths, per_image_labels,
                      summary_png_path,
                      per_image_extra_pages=None,
                      per_image_cell_dirs=None,
                      overview_pages=None):
    """Concatenate per-image overlays, galleries, and per-cell crops into a PDF.

    Args:
        pdf_path: output PDF file path.
        per_image_png_paths: list of paths to per-image spot overlay PNGs.
        per_image_labels: list of labels (image names) for each image.
        summary_png_path: batch summary PNG (first page).
        per_image_extra_pages: optional list of lists. Each inner list
            contains (label, path) tuples for extra pages per image
            (e.g. gallery, FOV overview).
        per_image_cell_dirs: optional list of paths to per-cell crop
            directories. Each cell PNG in the directory gets its own page.
        overview_pages: optional list of (label, path) tuples for batch-level
            summary pages added at the start of the PDF (before per-image
            sections).
    """
    if FPDF is None:
        warnings.warn(
            "fpdf not installed; skipping PDF report. Install with: pip install fpdf2"
        )
        return

    def _add_image_page(pdf, title, png_path, landscape=False):
        """Add a single image to a new PDF page."""
        if not Path(png_path).exists():
            return
        if landscape:
            pdf.add_page('L')
        else:
            pdf.add_page()
        safe = str(title).encode('latin-1', errors='replace').decode('latin-1')
        pdf.cell(0, 10, safe, 0, 1, 'L')
        with Image.open(png_path) as img:
            aspect = img.size[1] / img.size[0]
        page_w = 267 if landscape else 180
        page_h_limit = 170 if landscape else 250
        w_mm = page_w
        h_mm = int(aspect * w_mm)
        if h_mm > page_h_limit:
            h_mm = page_h_limit
            w_mm = int(h_mm / aspect)
        pdf.image(str(png_path), x=15, y=25, w=w_mm, h=h_mm)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Arial', size=11)

    # Batch-level overview pages (e.g., spots-per-cell whisker plot)
    if overview_pages is not None:
        for ov_label, ov_path in overview_pages:
            # Overview plots tend to be wide (many images on the x-axis);
            # render landscape so boxes stay readable.
            _add_image_page(pdf, f'Summary -- {ov_label}',
                            ov_path, landscape=True)

    # Per-image sections
    n_images = len(per_image_png_paths)
    for idx in range(n_images):
        label = per_image_labels[idx]

        # 1. Spots overlay
        _add_image_page(pdf, f'{label} -- spot overlay',
                        per_image_png_paths[idx])

        # 2. Extra pages (gallery, FOV overview)
        if per_image_extra_pages is not None and idx < len(per_image_extra_pages):
            for extra_label, extra_path in per_image_extra_pages[idx]:
                is_wide = 'gallery' in str(extra_label).lower()
                _add_image_page(pdf, f'{label} -- {extra_label}',
                                extra_path, landscape=is_wide)

        # 3. Per-cell crops
        if per_image_cell_dirs is not None and idx < len(per_image_cell_dirs):
            cell_dir = Path(per_image_cell_dirs[idx])
            if cell_dir.exists():
                cell_pngs = sorted(cell_dir.glob('cell_*.png'))
                for cell_png in cell_pngs:
                    cell_name = cell_png.stem.replace('_', ' ').title()
                    _add_image_page(pdf, f'{label} -- {cell_name}',
                                    cell_png, landscape=True)

    pdf.output(str(pdf_path))


# =============================================================================
# Public entry point
# =============================================================================

def pipeline_multicell_spot_detection(
    data_path,
    channels_spots,
    results_folder_path=None,
    input_axis_order=None,
    pixel_size_xy_nm=DEFAULT_PIXEL_SIZE_XY_NM,
    voxel_size_z_nm=DEFAULT_VOXEL_SIZE_Z_NM,
    time_interval_sec=DEFAULT_TIME_INTERVAL_SEC,
    channel_cytosol=None,
    channel_nucleus=None,
    channel_names=None,
    diameter_cytosol=120,
    diameter_nucleus=60,
    model_cyto_segmentation='cyto3',
    model_nuc_segmentation='nuclei',
    pretrained_model_cyto=None,
    pretrained_model_nuc=None,
    remove_cells_touching_border=True,
    min_cell_area_px=500,
    yx_spot_size_in_px=DEFAULT_YX_SPOT_SIZE_PX,
    z_spot_size_in_px=DEFAULT_Z_SPOT_SIZE_PX,
    cluster_radius_nm=DEFAULT_CLUSTER_RADIUS_NM,
    maximum_spots_cluster=None,
    threshold_for_spot_detection=None,
    use_maximum_projection=False,
    use_log_filter_for_spot_detection=True,
    max_spots_for_threshold=DEFAULT_MAX_SPOTS_FOR_THRESHOLD,
    threshold_sensitivity=1.0,
    threshold_method='elbow',
    save_masks=True,
    save_per_cell_crops=True,
    crop_padding_px=10,
    crop_mode='max_proj',
    visualization_z_slice=None,
    save_pdf_report=True,
    list_images_to_process=None,
    continue_on_error=True,
    show_plot=False,
):
    """Run multi-cell spot detection and quantification on one or more images.

    Args:
        data_path: Path to a .tif/.tiff, .lif, .czi, or a directory containing
            any mix of these formats.
        channels_spots: List[int] of channel indices to detect spots in.
        results_folder_path: Output folder. Defaults to sibling folder of input
            named ``results_multicell_<input_stem>``.
        input_axis_order: Axis order for TIFF inputs (e.g. ['Z','Y','X','C']).
            Ignored for LIF and CZI (axis order is read from file metadata).
        pixel_size_xy_nm: XY pixel size in nm. Overrides LIF/CZI metadata when
            set explicitly; for TIFF this is authoritative. Defaults to 160.
        voxel_size_z_nm: Z voxel size in nm. Defaults to 500.
        time_interval_sec: Time between frames (saved in metadata). Defaults to 1.0.
        channel_cytosol: Channel index for cytosol segmentation, or None.
        channel_nucleus: Channel index for nucleus segmentation, or None.
        channel_names: Optional list[str] of channel labels for plots.
        diameter_cytosol: Expected cytosol diameter in pixels. Defaults to 120.
        diameter_nucleus: Expected nucleus diameter in pixels. Defaults to 60.
        model_cyto_segmentation: Cellpose model name. Defaults to 'cyto3'.
        model_nuc_segmentation: Cellpose nucleus model. Defaults to 'nuclei'.
        pretrained_model_cyto: Path to custom Cellpose model. Defaults to None.
        pretrained_model_nuc: Path to custom nucleus model. Defaults to None.
        remove_cells_touching_border: Drop cells touching image edge. Defaults to True.
        min_cell_area_px: Drop cells smaller than this area. Defaults to 500.
        yx_spot_size_in_px: Expected XY spot diameter (pixels). Defaults to 5.
        z_spot_size_in_px: Expected Z spot diameter (pixels). Defaults to 3.
        cluster_radius_nm: DBSCAN cluster radius in nm. Defaults to 500.
        maximum_spots_cluster: If set, drop clusters with more spots than this.
            Defaults to None (keep all).
        threshold_for_spot_detection: Detection threshold. Accepts:
            - None: auto-detect per channel
            - scalar (int/float): broadcast to all spot channels
            - list: one entry per channel in ``channels_spots`` (None entries auto-detect)
        use_maximum_projection: Use 2D Z-max-projection for detection. Defaults to False.
        use_log_filter_for_spot_detection: Apply LoG filter. Defaults to True.
        max_spots_for_threshold: Cap used during auto-threshold. Defaults to 100000.
        threshold_sensitivity: Multiplier applied to auto-detected thresholds.
            Values < 1.0 lower the threshold (more spots); > 1.0 raises it
            (fewer spots). Only affects auto-detected channels; user-supplied
            numeric thresholds are never scaled. Defaults to 1.0.
        threshold_method: Algorithm for auto-threshold computation. Options:
            - ``'elbow'`` (default): derivative-valley method. Conservative;
              works well for live-cell data with sparse spots.
            - ``'kneedle'``: Kneedle algorithm. More sensitive; finds the
              true knee of the spots-vs-threshold curve. Recommended for
              FISH data with dense spot populations.
        save_masks: Save segmentation masks as TIFFs. Defaults to True.
        save_per_cell_crops: Save per-cell cropped visualizations. Defaults to True.
        crop_padding_px: Padding (pixels) around each cell's bbox. Defaults to 10.
        crop_mode: 'max_proj' or 'z_slice'. Defaults to 'max_proj'.
        visualization_z_slice: Z-slice for spot overlays. None = max projection.
        save_pdf_report: Concatenate per-image overlays into a PDF. Defaults to True.
        list_images_to_process: Names or indices to process. None = all.
        continue_on_error: Skip failing images and keep going. Defaults to True.
        show_plot: Display plots interactively (notebook use). Defaults to False.

    Returns:
        tuple:
            - df_all_spots (pd.DataFrame): long-format spot table across all images.
            - df_all_cells (pd.DataFrame): per-cell summary across all images.
            - run_summary (dict): contains 'results_folder', 'processed',
              'rejected', 'n_images', 'n_cells', 'n_spots', 'elapsed_sec'.
    """
    t0 = time.time()

    # --- Normalize inputs ---
    data_path = Path(data_path)
    channels_spots = list(channels_spots)
    if len(channels_spots) == 0:
        raise ValueError("channels_spots must be a non-empty list of channel indices.")

    thresholds = normalize_thresholds(threshold_for_spot_detection,
                                       len(channels_spots))

    if results_folder_path is None:
        stem = data_path.stem if data_path.is_file() else data_path.name
        results_folder_path = data_path.parent / f"results_multicell_{stem}"
    results_folder_path = Path(results_folder_path)
    results_folder_path.mkdir(parents=True, exist_ok=True)

    # --- Load images ---
    list_images, list_names, pixel_xy_nm, voxel_z_nm, lif_channel_names = load_images(
        data_path, input_axis_order, pixel_size_xy_nm, voxel_size_z_nm
    )
    if channel_names is None:
        channel_names = lif_channel_names

    # Pipeline-supplied values override LIF metadata only when user explicitly
    # passed something different from the defaults
    if pixel_size_xy_nm != DEFAULT_PIXEL_SIZE_XY_NM:
        pixel_xy_nm = pixel_size_xy_nm
    if voxel_size_z_nm != DEFAULT_VOXEL_SIZE_Z_NM:
        voxel_z_nm = voxel_size_z_nm
    list_voxels = [float(voxel_z_nm), float(pixel_xy_nm)]

    # --- Filter image list if requested ---
    if list_images_to_process is not None:
        keep = []
        for i, name in enumerate(list_names):
            if i in list_images_to_process or name in list_images_to_process:
                keep.append(i)
        list_images = [list_images[i] for i in keep]
        list_names = [list_names[i] for i in keep]

    # --- Validate channel indices ---
    if list_images:
        n_channels = list_images[0].shape[-1]
        for ch in channels_spots:
            if not 0 <= ch < n_channels:
                raise ValueError(
                    f"channels_spots contains {ch}, but image has {n_channels} channels."
                )
        for label, ch in [('channel_cytosol', channel_cytosol),
                          ('channel_nucleus', channel_nucleus)]:
            if ch is not None and not 0 <= ch < n_channels:
                raise ValueError(
                    f"{label}={ch} is out of range for image with {n_channels} channels."
                )

    # --- Iterate ---
    all_spots_df = []
    all_cells_df = []
    overlay_paths = []
    overlay_labels = []
    extra_pages_per_image = []
    cell_dirs_per_image = []
    processed = []
    rejected = []
    thresholds_per_image = {}

    for idx, (image_TZYXC, name) in enumerate(zip(list_images, list_names)):
        image_dir = results_folder_path / name
        image_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx + 1}/{len(list_images)}] Processing {name} "
              f"(shape={image_TZYXC.shape})")

        try:
            # --- Segment ---
            mask_cyto, mask_nuc, mask_cno = segment_cells(
                image_TZYXC[0], channel_cytosol, channel_nucleus,
                diameter_cytosol, diameter_nucleus,
                model_cyto_segmentation, model_nuc_segmentation,
                pretrained_model_cyto, pretrained_model_nuc,
                remove_cells_touching_border, min_cell_area_px,
            )
            n_cells = int(np.max(mask_cyto)) if mask_cyto is not None else 0
            print(f"    segmentation: {n_cells} cells")

            # --- Visualize inputs and segmentation ---
            render_raw_channels(image_TZYXC, channel_names,
                                 image_dir / 'raw_channels.png')
            render_segmentation(image_TZYXC, mask_cyto, mask_nuc, mask_cno,
                                 channel_cytosol, channel_nucleus,
                                 image_dir / 'segmentation.png')

            if save_masks and mask_cyto is not None:
                tifffile.imwrite(str(image_dir / 'mask_cytosol.tif'),
                                 mask_cyto.astype(np.uint16))
            if save_masks and mask_nuc is not None:
                tifffile.imwrite(str(image_dir / 'mask_nucleus.tif'),
                                 mask_nuc.astype(np.uint16))

            if n_cells == 0:
                rejected.append({'name': name,
                                 'error': 'Segmentation produced 0 cells.'})
                continue

            # --- Resolve thresholds for this image ---
            # Any ``None`` entries in ``thresholds`` are computed per image via
            # the elbow picker; numeric entries are used as-is. Resolving
            # here (not inside ``detect_and_quantify``) keeps the concrete
            # value available for logging and metadata.
            resolved_thresholds = resolve_thresholds(
                image_TZYXC, channels_spots, thresholds,
                yx_spot_size_in_px, z_spot_size_in_px, list_voxels,
                max_spots_for_threshold,
                save_plot_prefix=(image_dir / 'threshold_elbow')
                if any(t is None for t in thresholds) else None,
                threshold_sensitivity=threshold_sensitivity,
                threshold_method=threshold_method,
            )
            thresholds_per_image[str(name)] = [float(t) for t in resolved_thresholds]
            print(f"    thresholds: "
                  f"{dict(zip(channels_spots, resolved_thresholds))}")

            # --- Detect + quantify ---
            df_spots, df_cells, _thr_echo = detect_and_quantify(
                image_TZYXC, mask_cyto, mask_nuc, mask_cno,
                channels_spots, channel_cytosol, channel_nucleus,
                list_voxels, yx_spot_size_in_px, z_spot_size_in_px,
                cluster_radius_nm, maximum_spots_cluster,
                resolved_thresholds, use_maximum_projection,
                use_log_filter_for_spot_detection,
                image_counter=idx,
            )
            print(f"    detected {len(df_spots)} spots "
                  f"across {len(df_cells)} cells")

            if not df_spots.empty:
                df_spots.insert(0, 'image_name', name)
                df_spots.to_csv(image_dir / 'spots.csv', index=False)
                all_spots_df.append(df_spots)
            if not df_cells.empty:
                df_cells.insert(0, 'image_name', name)
                df_cells.to_csv(image_dir / 'cells.csv', index=False)
                all_cells_df.append(df_cells)

            # --- Render overlays and per-cell crops ---
            overlay_path = image_dir / 'spots_overlay.png'
            render_spots_overlay(image_TZYXC, df_spots, mask_cyto, mask_nuc,
                                  channels_spots, overlay_path,
                                  visualization_z_slice=visualization_z_slice)
            overlay_paths.append(overlay_path)
            overlay_labels.append(name)

            image_extra_pages = []
            image_cell_dir = None
            if save_per_cell_crops:
                pixel_size_um = pixel_xy_nm / 1000.0
                render_per_cell_crops(
                    image_TZYXC, df_spots, mask_cyto, mask_nuc,
                    channels_spots, image_dir / 'per_cell',
                    crop_padding_px=crop_padding_px, crop_mode=crop_mode,
                    pixel_size_um=pixel_size_um,
                )
                image_cell_dir = image_dir / 'per_cell'

                gallery_path = image_dir / 'all_cells_gallery.png'
                render_all_cells_gallery(
                    image_TZYXC, df_spots, mask_cyto, mask_nuc,
                    channels_spots, gallery_path,
                    crop_padding_px=crop_padding_px, crop_mode=crop_mode,
                    pixel_size_um=pixel_size_um,
                )
                image_extra_pages.append(('All cells gallery', gallery_path))

                fov_path = image_dir / 'fov_overview.png'
                render_fov_overview(
                    image_TZYXC, mask_cyto, mask_nuc,
                    channel_nucleus, channels_spots,
                    fov_path, pixel_size_um=pixel_size_um,
                )
                image_extra_pages.append(('FOV overview', fov_path))

            extra_pages_per_image.append(image_extra_pages)
            cell_dirs_per_image.append(image_cell_dir)

            processed.append(name)

        except Exception as exc:
            tb = traceback.format_exc()
            print(f"    ERROR: {exc}")
            rejected.append({'name': name, 'error': f"{exc}\n{tb}"})
            if not continue_on_error:
                raise

    # --- Aggregate + summary plot ---
    valid_spots = [df for df in all_spots_df if not df.empty]
    valid_cells = [df for df in all_cells_df if not df.empty]
    df_all_spots = (pd.concat(valid_spots, ignore_index=True)
                    if valid_spots else pd.DataFrame())
    df_all_cells = (pd.concat(valid_cells, ignore_index=True)
                    if valid_cells else pd.DataFrame())

    if not df_all_spots.empty:
        df_all_spots.to_csv(results_folder_path / 'summary_spots.csv', index=False)
    if not df_all_cells.empty:
        df_all_cells.to_csv(results_folder_path / 'summary_cells.csv', index=False)

    summary_png = results_folder_path / 'summary_spot_counts.png'
    render_batch_summary_plot(df_all_cells, channels_spots, summary_png)

    boxplot_png = results_folder_path / 'summary_spots_per_cell_boxplot.png'
    render_spots_per_cell_boxplot(df_all_cells, channels_spots, boxplot_png)

    # --- Metadata ---
    params = dict(
        data_path=str(data_path),
        results_folder_path=str(results_folder_path),
        input_axis_order=input_axis_order,
        pixel_size_xy_nm=pixel_xy_nm,
        voxel_size_z_nm=voxel_z_nm,
        time_interval_sec=time_interval_sec,
        channels_spots=channels_spots,
        channel_cytosol=channel_cytosol,
        channel_nucleus=channel_nucleus,
        channel_names=list(channel_names) if channel_names is not None else None,
        diameter_cytosol=diameter_cytosol,
        diameter_nucleus=diameter_nucleus,
        model_cyto_segmentation=model_cyto_segmentation,
        model_nuc_segmentation=model_nuc_segmentation,
        pretrained_model_cyto=str(pretrained_model_cyto) if pretrained_model_cyto else None,
        pretrained_model_nuc=str(pretrained_model_nuc) if pretrained_model_nuc else None,
        remove_cells_touching_border=remove_cells_touching_border,
        min_cell_area_px=min_cell_area_px,
        yx_spot_size_in_px=yx_spot_size_in_px,
        z_spot_size_in_px=z_spot_size_in_px,
        cluster_radius_nm=cluster_radius_nm,
        maximum_spots_cluster=maximum_spots_cluster,
        threshold_for_spot_detection=threshold_for_spot_detection,
        use_maximum_projection=use_maximum_projection,
        use_log_filter_for_spot_detection=use_log_filter_for_spot_detection,
        max_spots_for_threshold=max_spots_for_threshold,
        save_masks=save_masks,
        save_per_cell_crops=save_per_cell_crops,
        crop_padding_px=crop_padding_px,
        crop_mode=crop_mode,
        visualization_z_slice=visualization_z_slice,
    )
    write_metadata(
        results_folder_path / 'metadata.txt',
        results_folder_path / 'metadata.json',
        params, processed, rejected, thresholds_per_image,
    )

    # --- PDF report ---
    if save_pdf_report and overlay_paths:
        overview_pages = []
        if boxplot_png.exists():
            overview_pages.append(('Spots per cell (box-whisker)', boxplot_png))
        build_pdf_report(
            results_folder_path / 'summary_report.pdf',
            overlay_paths, overlay_labels, summary_png,
            per_image_extra_pages=extra_pages_per_image,
            per_image_cell_dirs=cell_dirs_per_image,
            overview_pages=overview_pages or None,
        )

    # Real spots have ``spot_id >= 0``; the engine injects rows with
    # ``spot_id = -1`` as placeholders for cells that contain no spots so the
    # cell stays in the table. Count only real spots.
    if not df_all_spots.empty and 'spot_id' in df_all_spots.columns:
        n_spots = int((df_all_spots['spot_id'] >= 0).sum())
    else:
        n_spots = 0

    elapsed = time.time() - t0
    run_summary = {
        'results_folder': str(results_folder_path),
        'processed': processed,
        'rejected': rejected,
        'n_images': len(processed),
        'n_cells': int(len(df_all_cells)),
        'n_spots': n_spots,
        'elapsed_sec': round(elapsed, 2),
    }
    print(f"\nDone in {elapsed:.1f}s. "
          f"{len(processed)} images processed, {len(rejected)} skipped. "
          f"Results: {results_folder_path}")
    return df_all_spots, df_all_cells, run_summary
