"""
FRAP Utilities Module
=====================

This module provides publication-quality plotting functions and data
utilities for FRAP (Fluorescence Recovery After Photobleaching) analysis.

Plotting Functions:
-------------------
- plot_images_frap_all_channels_representative: Plot representative frames for all channels
- plot_kymograph: Generate kymograph from ROI coordinates
- plot_merged_image: Create merged RGB image with FRAP line overlay
- plot_cell_images_individual: Generate separate images for each channel + merged
- plot_combined_cell_and_kymograph: Combined publication figure with cells and kymographs
- save_video_as_avi: Export multi-channel AVI video
- compose_pngs: Combine two PNG images side-by-side
- plot_FRAP_trajectories: Plot FRAP recovery trajectories with statistics
- plot_mean_trajectories_all: Mean trajectory plot with SEM/STD shading
- plot_box_swarm_final_values: Box+swarm plot for final recovery values
- plot_box_swarm_fit_results: Box+swarm plot for fit parameters (t_half, etc.)

Data Utility Functions:
-----------------------
- load_frap_datasets: Scan result folders and build a combined DataFrame
- fit_all_cells: Batch exponential fitting across all cells

Usage:
------
    from frap_utilities import (
        plot_images_frap_all_channels_representative,
        plot_kymograph,
        plot_merged_image,
        plot_cell_images_individual,
        plot_combined_cell_and_kymograph,
        save_video_as_avi,
        compose_pngs,
        plot_FRAP_trajectories,
        load_frap_datasets,
        fit_all_cells,
    )


Author: Rhiannon Sears, Luis Aguilera
Date: 2024-2025
"""

# Standard library
import io
from itertools import combinations
from pathlib import Path

# Third-party
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from PIL import Image
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter
from skimage import exposure
import tifffile

# Local (optional)
try:
    from microlive.imports import green_colormap, magenta_colormap
except ImportError:
    green_colormap = 'Greens'
    magenta_colormap = 'RdPu'


def _get_colormap(cmap):
    """
    Convert colormap name or object to a matplotlib colormap.
    
    Parameters
    ----------
    cmap : str or Colormap
        Colormap name (e.g., 'gray') or colormap object
        
    Returns
    -------
    Colormap object that can be called with values
    """
    if isinstance(cmap, str):
        return plt.cm.get_cmap(cmap)
    return cmap


# =============================================================================
# TIFF EXPORT
# =============================================================================

def save_image_as_tiff(
    image_TZXYC,
    output_path,
    pixel_xy_um=0.2,
    voxel_z_um=1.0,
):
    """
    Save a 5D image array as an ImageJ-compatible TIFF file.
    
    Converts from TZXYC format to TZCYX format for ImageJ compatibility.
    
    Parameters
    ----------
    image_TZXYC : np.ndarray
        5D image array with shape (T, Z, X, Y, C)
    output_path : Path or str
        Output file path (should end with .tif or .tiff)
    pixel_xy_um : float
        Pixel size in microns for X and Y dimensions (default: 0.2)
    voxel_z_um : float
        Voxel size in microns for Z dimension (default: 1.0)
        
    Returns
    -------
    Path
        Path to the saved file
        
    Example
    -------
    >>> save_image_as_tiff(
    ...     image_TZXYC, 
    ...     results_folder / 'my_image.tif',
    ...     pixel_xy_um=0.2,
    ...     voxel_z_um=1.0
    ... )
    """
    output_path = Path(output_path)
    
    # Convert from TZXYC to TZCYX for ImageJ compatibility
    # T=0, Z=1, X=2, Y=3, C=4 -> T=0, Z=1, C=4, Y=3, X=2
    image_TZCYX = np.transpose(image_TZXYC, (0, 1, 4, 3, 2))
    
    tifffile.imwrite(
        output_path,
        image_TZCYX,
        imagej=True,
        ome=False,
        resolution=(pixel_xy_um, pixel_xy_um),
        resolutionunit='MICROMETER',
        metadata={
            'axes': 'TZCYX',
            'PhysicalSizeX': pixel_xy_um,
            'PhysicalSizeY': pixel_xy_um,
            'PhysicalSizeZ': voxel_z_um,
            'PhysicalSizeXUnit': 'um',
            'PhysicalSizeYUnit': 'um',
            'PhysicalSizeZUnit': 'um'
        }
    )
    
    return output_path


# =============================================================================
# REPRESENTATIVE IMAGE PLOTTING
# =============================================================================

def plot_images_frap_all_channels_representative(
    image_TZXYC,
    list_selected_frames=[0, 11, 40, 100, 139],
    cmap_list=None,
    selected_color_channel=None,
    coordinates_roi=None,
    radius_roi_size_px=10,
    plot_name='temp.png',
    list_axis_limits=None,
    y_label_list=None,
    list_selected_frame_values_real_time=None,  # Kept for backwards compatibility
    x_title_list=None,
    masks_TXY=None,
    min_size_image=150,
    show_circle=True,
    pixel_xy_um=0.2,
    scalebar_size=8,
    results_folder=None,
):
    """
    Plot representative frames for all color channels.
    
    Optionally crops around a binary mask (masks_TXY) to a fixed window of size min_size_image.
    
    Parameters
    ----------
    image_TZXYC : np.ndarray
        5D image array (T, Z, Y, X, C)
    list_selected_frames : list
        Frame indices to display
    cmap_list : list, optional
        Colormaps for each channel
    selected_color_channel : int, optional
        If provided, only display this channel
    coordinates_roi : np.ndarray, optional
        ROI coordinates (T, 2) for circle overlay
    radius_roi_size_px : int
        Radius of ROI circle in pixels
    plot_name : str
        Output filename prefix
    list_axis_limits : list, optional
        [xmin, xmax, ymin, ymax] for axis limits
    y_label_list : list, optional
        Labels for each channel (y-axis)
    x_title_list : list, optional
        Labels for each time point (column titles)
    masks_TXY : np.ndarray, optional
        Masks for auto-cropping around cells
    min_size_image : int
        Size of crop window when using masks
    show_circle : bool
        Whether to show ROI circle
    pixel_xy_um : float
        Pixel size in microns for scalebar
    scalebar_size : int
        Font size for scalebar
    results_folder : Path, optional
        Output directory
        
    Returns
    -------
    Path to saved figure
    """
    if results_folder is None:
        results_folder = Path('.').resolve()
    
    if selected_color_channel is not None:
        image_TZXYC = image_TZXYC[..., selected_color_channel]
        image_TZXYC = np.expand_dims(image_TZXYC, axis=-1)

    number_color_channels = image_TZXYC.shape[-1]
    num_frames = len(list_selected_frames)
    
    # Handle mask-based cropping
    if masks_TXY is not None:
        mask_xy = masks_TXY.max(axis=0)
        ys, xs = np.where(mask_xy > 0)
        if ys.size == 0:
            raise ValueError("masks_TXY provided but contains no positive pixels")
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        half = min_size_image // 2
        y_start = center_y - half
        y_end = y_start + min_size_image
        x_start = center_x - half
        x_end = x_start + min_size_image
        
        # Pad if needed
        pad_top = max(0, -y_start)
        pad_bottom = max(0, y_end - image_TZXYC.shape[2])
        pad_left = max(0, -x_start)
        pad_right = max(0, x_end - image_TZXYC.shape[3])
        if any((pad_top, pad_bottom, pad_left, pad_right)):
            image_TZXYC = np.pad(
                image_TZXYC,
                ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant', constant_values=0
            )
            y_start += pad_top
            y_end += pad_top
            x_start += pad_left
            x_end += pad_left
    else:
        y_start, y_end, x_start, x_end = None, None, None, None
    
    if cmap_list is None:
        cmap_list = ['gray'] * number_color_channels
    if y_label_list is None:
        y_label_list = [f'Ch {ch}' for ch in range(number_color_channels)]
    
    assert len(cmap_list) == number_color_channels
    assert len(y_label_list) == number_color_channels
    
    fig_width = num_frames * 2
    fig_height = number_color_channels * 2
    fig, ax = plt.subplots(
        number_color_channels, num_frames,
        figsize=(fig_width, fig_height),
        gridspec_kw={'wspace': 0.02, 'hspace': 0.02}
    )
    # Bug fix: Handle all edge cases for subplot array dimensions
    ax = np.atleast_2d(ax)
    if number_color_channels == 1:
        ax = ax.reshape(1, -1)
    elif num_frames == 1:
        ax = ax.reshape(-1, 1)
    
    for ch in range(number_color_channels):
        cmap = cmap_list[ch]
        ylabel = y_label_list[ch]
        for i, frame in enumerate(list_selected_frames):
            current_ax = ax[ch, i]
            if masks_TXY is not None:
                sub = image_TZXYC[frame, 0, y_start:y_end, x_start:x_end, ch]
            else:
                sub = image_TZXYC[frame, 0, :, :, ch]
            
            current_ax.imshow(sub, vmax=np.percentile(sub, 99.9), cmap=cmap)
            
            if x_title_list is not None and ch == 0:
                current_ax.set_title(x_title_list[i], fontsize=12, fontname='Arial')
            if i == 0:
                current_ax.set_ylabel(ylabel, fontsize=10, fontname='Arial')
            
            if show_circle and coordinates_roi is not None:
                x, y = coordinates_roi[frame]
                if masks_TXY is not None:
                    x -= x_start
                    y -= y_start
                circ = plt.Circle((x, y), radius_roi_size_px,
                                  edgecolor='lightyellow', facecolor='none', linewidth=2)
                current_ax.add_artist(circ)
            
            # Add scalebar
            if scalebar_size > 0:
                scalebar = ScaleBar(
                    dx=pixel_xy_um, units='um', length_fraction=0.3,
                    location='lower right', box_color='black', color='white',
                    font_properties={'size': scalebar_size}
                )
                current_ax.add_artist(scalebar)
            current_ax.set_xticks([])
            current_ax.set_yticks([])
            current_ax.grid(False)
    
    if masks_TXY is None and list_axis_limits is not None:
        xmin, xmax, ymin, ymax = list_axis_limits
        for a in ax.flat:
            a.set_xlim(xmin, xmax)
            a.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    out_png = results_folder.joinpath('time_courses_' + plot_name + '.png')
    plt.savefig(out_png, dpi=900, bbox_inches='tight', pad_inches=0.1)
    # save as svg
    out_svg = results_folder.joinpath('time_courses_' + plot_name + '.svg')
    plt.savefig(out_svg, dpi=900, bbox_inches='tight', pad_inches=0.1)

    plt.show()
    
    return out_png


# =============================================================================
# KYMOGRAPH GENERATION
# =============================================================================

def plot_kymograph(
    image_TZXYC,
    coordinates_roi,
    list_selected_frames,
    x_title_list,
    length_kymograph_line=50,
    cmap_list=None,
    plot_vertical_lines=False,
    plot_name='temp',
    results_folder=None,
):
    """
    Generate and plot kymograph from ROI coordinates.
    
    Parameters
    ----------
    image_TZXYC : np.ndarray
        5D image array (T, Z, Y, X, C)
    coordinates_roi : np.ndarray
        ROI coordinates (T, 2) with x, y positions
    list_selected_frames : list
        Frame indices to mark with vertical lines
    x_title_list : list
        Labels for the time points
    length_kymograph_line : int
        Length of kymograph line profile in pixels
    cmap_list : list, optional
        Colormaps for each channel
    plot_vertical_lines : bool
        Whether to draw vertical lines at selected frames
    plot_name : str
        Output filename prefix
    results_folder : Path, optional
        Output directory
        
    Returns
    -------
    Path to saved kymograph figure
    """
    if results_folder is None:
        results_folder = Path('.').resolve()
        
    y = length_kymograph_line
    x = image_TZXYC.shape[0]
    height, width = image_TZXYC.shape[2:4]
    number_of_channels = image_TZXYC.shape[-1]
    
    # Bug fix: Add default colormap list if None
    if cmap_list is None:
        cmap_list = [green_colormap, magenta_colormap][:number_of_channels]
    
    kymographs = []
    min_kymo_height = y  # Track minimum height across all channels
    
    for selected_color_channel in range(number_of_channels):
        kymograph = np.zeros((y, x))
        for i in range(x):
            coord_x_time_point = int(np.round(coordinates_roi[i, 0]))
            coord_y_time_point = int(np.round(coordinates_roi[i, 1]))
            coord_x_time_point = np.clip(coord_x_time_point, 0, width - 1)
            coord_y_time_point = np.clip(coord_y_time_point, 0, height - 1)
            start_x = max(coord_x_time_point - y // 2, 0)
            end_x = min(coord_x_time_point + y // 2, width)
            range_x = np.arange(start_x, end_x, dtype=int)
            current_y = len(range_x)
            line_profile = image_TZXYC[i, 0, coord_y_time_point, range_x, selected_color_channel]
            kymograph[:current_y, i] = line_profile
            min_kymo_height = min(min_kymo_height, current_y)
        kymographs.append(kymograph)
    
    # Bug fix: Truncate all kymographs to the same height
    kymographs = [k[:min_kymo_height, :] for k in kymographs]
    
    fig, axes = plt.subplots(nrows=number_of_channels, ncols=1, figsize=(9, 5), sharex=True)
    if number_of_channels == 1:
        axes = [axes]
    plt.subplots_adjust(hspace=0, wspace=0)
    
    for ax, kymograph, cmap, channel in zip(axes, kymographs, cmap_list, range(number_of_channels)):
        ax.imshow(kymograph, aspect='auto', cmap=cmap, vmax=np.percentile(kymograph, 99.5))
        ax.grid(False)
        ax.set_yticks([])
        if plot_vertical_lines:
            for frame in list_selected_frames:
                ax.axvline(x=frame, color='w', linestyle='--', linewidth=4)
        if ax != axes[-1]:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x', labelsize=12)
            ax.set_xticks(list_selected_frames)
            ax.set_xticklabels(x_title_list)
    
    plt.tight_layout()
    out_path = results_folder.joinpath('kymograph_' + plot_name + '.png')
    plt.savefig(out_path, dpi=600)
    # save as svg
    plt.savefig(out_path.with_suffix('.svg'), dpi=600)
    plt.show()
    
    return out_path


# =============================================================================
# MERGED IMAGE WITH FRAP LINE
# =============================================================================

def plot_merged_image(
    image_TZXYC,
    coordinates_roi,
    length_kymograph_line,
    cmap_list_imagej,
    list_axis_limits=None,
    channel_order=[0, 1, 2],
    normalize_each_color_channel=False,
    plot_name='temp',
    masks_TXY=None,
    min_size_image=150,
    gamma=0.6,
    clip_limit=0.001,
    line_color_FRAP='yellow',
    scalebar_size=0,
    pixel_xy_um=0.2,
    frame=10,
    results_folder=None,
):
    """
    Create a merged RGB image with FRAP line overlay.
    
    Auto-crops around masks_TXY into a fixed window, then applies gamma
    correction and CLAHE to brighten the cell while keeping background dark.
    
    Parameters
    ----------
    image_TZXYC : np.ndarray
        5D image array (T, Z, Y, X, C)
    coordinates_roi : np.ndarray
        ROI coordinates (T, 2) with x, y positions
    length_kymograph_line : int
        Length of the FRAP line in pixels
    cmap_list_imagej : list
        Colormaps for each channel (ImageJ-style)
    list_axis_limits : list, optional
        [xmin, xmax, ymin, ymax] for axis limits
    channel_order : list
        Order for RGB channels in output
    normalize_each_color_channel : bool
        Whether to normalize each channel individually
    plot_name : str
        Output filename prefix
    masks_TXY : np.ndarray, optional
        Masks for auto-cropping
    min_size_image : int
        Size of crop window
    gamma : float
        Gamma correction value
    clip_limit : float
        CLAHE clip limit
    line_color_FRAP : str
        Color for the FRAP line
    scalebar_size : int
        Font size for scalebar (0 to disable)
    pixel_xy_um : float
        Pixel size in microns
    frame : int
        Frame index to display
    results_folder : Path, optional
        Output directory
        
    Returns
    -------
    Path to saved figure
    """
    if results_folder is None:
        results_folder = Path('.').resolve()
    
    line_len = length_kymograph_line
    H0, W0 = image_TZXYC.shape[2], image_TZXYC.shape[3]
    
    if masks_TXY is not None:
        mask_xy = masks_TXY.max(axis=0)
        ys, xs = np.where(mask_xy > 0)
        if ys.size == 0:
            raise ValueError("masks_TXY has no foreground")
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        half = min_size_image // 2
        y0, y1 = cy - half, cy - half + min_size_image
        x0, x1 = cx - half, cx - half + min_size_image
        
        pad_t = max(0, -y0)
        pad_b = max(0, y1 - H0)
        pad_l = max(0, -x0)
        pad_r = max(0, x1 - W0)
        if any((pad_t, pad_b, pad_l, pad_r)):
            image_TZXYC = np.pad(
                image_TZXYC,
                ((0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                mode='constant', constant_values=0
            )
            masks_TXY = np.pad(
                masks_TXY,
                ((0, 0), (pad_t, pad_b), (pad_l, pad_r)),
                mode='constant', constant_values=0
            )
            y0 += pad_t
            y1 += pad_t
            x0 += pad_l
            x1 += pad_l
        
        # Bug fix: Get updated dimensions after padding
        H_padded, W_padded = image_TZXYC.shape[2], image_TZXYC.shape[3]
        
        # Optional shift for better framing
        shift_for_top = 15
        y0 = max(0, y0 - shift_for_top)
        y1 = max(0, y1 - shift_for_top)  # Bug fix: Also shift y1
        
        # Bug fix: Ensure crop doesn't exceed padded dimensions
        y1 = min(H_padded, y1)
        x1 = min(W_padded, x1)
        
        img_crop = image_TZXYC[:, :, y0:y1, x0:x1, :]
        # Bug fix: Use actual crop dimensions, not assumed min_size_image
        H, W = img_crop.shape[2], img_crop.shape[3]
    else:
        img_crop = image_TZXYC
        H, W = H0, W0
        y0 = x0 = 0
    
    # Calculate line position
    cx = int(round(coordinates_roi[frame, 0])) - x0
    cy = int(round(coordinates_roi[frame, 1])) - y0
    cx = np.clip(cx, 0, W - 1)
    cy = np.clip(cy, 0, H - 1)
    sx, ex = max(cx - line_len // 2, 0), min(cx + line_len // 2, W - 1)
    range_x = np.arange(sx, ex + 1)
    
    # Create merged image
    img = img_crop[frame, 0, :, :, :].astype(np.float32)
    C = img.shape[-1]
    if len(cmap_list_imagej) < C:
        raise ValueError("Not enough colormaps")
    
    merged = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(C):
        ch = img[..., i]
        p1, p99 = np.percentile(ch, 0.1), np.percentile(ch, 99.9)
        ch = np.clip(ch, p1, p99)
        ch = (ch - p1) / (p99 - p1 + 1e-8)
        if normalize_each_color_channel:
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        ch = gaussian_filter(ch, sigma=0.3)
        ch = ch ** gamma
        col = cmap_list_imagej[i](ch)[..., :3]
        merged += col
    
    merged /= float(C)
    merged = np.clip(merged, 0, 1)
    merged = exposure.equalize_adapthist(merged, clip_limit=clip_limit)
    
    if channel_order is not None:
        merged = merged[..., channel_order]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(merged, interpolation='bicubic')
    ax.plot(range_x, [cy] * len(range_x), color=line_color_FRAP, linewidth=6, alpha=0.9)
    
    if masks_TXY is None and list_axis_limits is not None:
        xmin, xmax, ymin, ymax = list_axis_limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    
    if scalebar_size > 0:
        sb = ScaleBar(
            dx=pixel_xy_um, units='um', length_fraction=0.3,
            location='lower right', box_color='black', color='white',
            font_properties={'size': scalebar_size, 'weight': 'bold'}
        )
        ax.add_artist(sb)
    
    ax.axis('off')
    plt.tight_layout()
    out_png = results_folder.joinpath(f"{plot_name}.png")
    plt.savefig(out_png, dpi=1500, bbox_inches='tight', pad_inches=0.05,
                facecolor='white', edgecolor='none', format='png')
    plt.show()
    
    return out_png


# =============================================================================
# INDIVIDUAL CELL IMAGES (PER CHANNEL + MERGE)
# =============================================================================

def plot_cell_images_individual(
    image_TZXYC,
    coordinates_roi=None,
    length_kymograph_line=50,
    cmap_list=None,
    y_label_list=None,
    masks_TXY=None,
    min_size_image=230,
    plot_name='cell_image',
    pixel_xy_um=0.2,
    scalebar_size=8,
    line_color_FRAP='yellow',
    frame_for_images=10,
    gamma=0.6,
    clip_limit=0.001,
    dpi=600,
    fig_size=5,
    show_frap_line=True,
    results_folder=None,
):
    """
    Generate and save individual cell images for each channel and merge.
    
    Creates separate PNG and SVG files for each color channel plus a merged
    image. Uses the same cropping and styling logic as plot_combined_cell_and_kymograph
    to ensure visual consistency.
    
    Parameters
    ----------
    image_TZXYC : np.ndarray
        5D image array (T, Z, Y, X, C)
    coordinates_roi : np.ndarray, optional
        ROI coordinates (T, 2) with x, y positions. Required if show_frap_line=True.
    length_kymograph_line : int
        Length of the FRAP line in pixels
    cmap_list : list, optional
        Colormap list for each channel
    y_label_list : list, optional
        Channel labels (used in filenames)
    masks_TXY : np.ndarray, optional
        Masks for cropping around the cell
    min_size_image : int
        Minimum crop size around cell
    plot_name : str
        Output filename prefix
    pixel_xy_um : float
        Pixel size in microns
    scalebar_size : int
        Scale bar font size (0 to disable)
    line_color_FRAP : str
        Color for the FRAP line on merged image
    frame_for_images : int
        Frame index to use for the images
    gamma : float
        Gamma correction for merged image
    clip_limit : float
        CLAHE clip limit for merged image
    dpi : int
        Output DPI
    fig_size : float
        Figure size in inches (square images)
    show_frap_line : bool
        Whether to show FRAP line on merged image
    results_folder : Path, optional
        Output directory
        
    Returns
    -------
    dict
        Dictionary with keys 'channels' (list of paths) and 'merged' (path)
        for the saved images.
    """
    if results_folder is None:
        results_folder = Path('.').resolve()
    else:
        results_folder = Path(results_folder)
        results_folder.mkdir(parents=True, exist_ok=True)

    number_of_channels = image_TZXYC.shape[-1]
    height, width = image_TZXYC.shape[2:4]
    frame = frame_for_images

    if cmap_list is None:
        cmap_list = [green_colormap, magenta_colormap][:number_of_channels]
    if y_label_list is None:
        y_label_list = [f'Ch{ch}' for ch in range(number_of_channels)]

    # Crop handling (same logic as plot_combined_cell_and_kymograph)
    use_crop = False
    y_start, y_end, x_start, x_end = 0, height, 0, width

    if masks_TXY is not None:
        mask_xy = masks_TXY.max(axis=0)
        ys, xs = np.where(mask_xy > 0)
        if ys.size > 0:
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2
            half = min_size_image // 2
            y_start = center_y - half
            y_end = y_start + min_size_image
            x_start = center_x - half
            x_end = x_start + min_size_image

            pad_top = max(0, -y_start)
            pad_bottom = max(0, y_end - image_TZXYC.shape[2])
            pad_left = max(0, -x_start)
            pad_right = max(0, x_end - image_TZXYC.shape[3])
            if any((pad_top, pad_bottom, pad_left, pad_right)):
                image_TZXYC = np.pad(
                    image_TZXYC,
                    ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant', constant_values=0
                )
                y_start += pad_top
                y_end += pad_top
                x_start += pad_left
                x_end += pad_left
            use_crop = True

    # Update dimensions after potential padding
    height, width = image_TZXYC.shape[2:4]

    if use_crop:
        img_crop = image_TZXYC[:, :, y_start:y_end, x_start:x_end, :]
        H, W = min_size_image, min_size_image
        roi_x_offset = x_start
        roi_y_offset = y_start
    else:
        img_crop = image_TZXYC
        H, W = height, width
        roi_x_offset = 0
        roi_y_offset = 0

    # Calculate FRAP line position for merged image
    if coordinates_roi is not None and show_frap_line:
        cx = int(np.round(coordinates_roi[frame, 0])) - roi_x_offset
        cy = int(np.round(coordinates_roi[frame, 1])) - roi_y_offset
        cx = np.clip(cx, 0, W - 1)
        cy = np.clip(cy, 0, H - 1)
        line_len = length_kymograph_line
        sx = max(cx - line_len // 2, 0)
        ex = min(cx + line_len // 2, W - 1)
        range_x_line = np.arange(sx, ex + 1)
    else:
        range_x_line = None
        cy = None

    output_paths = {'channels': [], 'merged': None}

    # Save individual channel images
    for ch in range(number_of_channels):
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        sub = img_crop[frame, 0, :, :, ch]
        ax.imshow(sub, vmax=np.percentile(sub, 99.9), cmap=cmap_list[ch])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        # Add scalebar to last channel only (or all if desired)
        if scalebar_size > 0 and ch == number_of_channels - 1:
            scalebar = ScaleBar(
                dx=pixel_xy_um, units='um', fixed_value=10, fixed_units='um',
                location='lower right', box_color='black', color='white',
                font_properties={'size': scalebar_size}
            )
            ax.add_artist(scalebar)
        
        plt.tight_layout(pad=0)
        
        # Create safe filename from channel label
        safe_label = y_label_list[ch].replace(' ', '_').replace('-', '_')
        out_png = results_folder / f'{plot_name}_ch{ch}_{safe_label}.png'
        out_svg = results_folder / f'{plot_name}_ch{ch}_{safe_label}.svg'
        
        plt.savefig(out_png, dpi=dpi, bbox_inches='tight', pad_inches=0.02, facecolor='white')
        plt.savefig(out_svg, dpi=dpi, bbox_inches='tight', pad_inches=0.02, facecolor='white')
        plt.close(fig)
        
        output_paths['channels'].append(out_png)

    # Create merged RGB image
    img = img_crop[frame, 0, :, :, :].astype(np.float32)
    C = img.shape[-1]
    merged = np.zeros((H, W, 3), dtype=np.float32)

    for i in range(C):
        ch_data = img[..., i]
        p1, p99 = np.percentile(ch_data, 0.1), np.percentile(ch_data, 99.9)
        ch_data = np.clip(ch_data, p1, p99)
        ch_data = (ch_data - p1) / (p99 - p1 + 1e-8)
        ch_data = gaussian_filter(ch_data, sigma=0.3)
        ch_data = ch_data ** gamma
        col = cmap_list[i](ch_data)[..., :3]
        merged += col
    merged /= float(C)
    merged = np.clip(merged, 0, 1)
    merged = exposure.equalize_adapthist(merged, clip_limit=clip_limit)

    # Plot and save merged image
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(merged, interpolation='bicubic')
    
    # Add FRAP line if requested
    if range_x_line is not None and cy is not None:
        ax.plot(range_x_line, [cy] * len(range_x_line), 
                color=line_color_FRAP, linewidth=3, alpha=0.9)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    if scalebar_size > 0:
        scalebar = ScaleBar(
            dx=pixel_xy_um, units='um', fixed_value=10, fixed_units='um',
            location='lower right', box_color='black', color='white',
            font_properties={'size': scalebar_size}
        )
        ax.add_artist(scalebar)
    
    plt.tight_layout(pad=0)
    
    out_png_merged = results_folder / f'{plot_name}_merged.png'
    out_svg_merged = results_folder / f'{plot_name}_merged.svg'
    
    plt.savefig(out_png_merged, dpi=dpi, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.savefig(out_svg_merged, dpi=dpi, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.show()
    plt.close(fig)
    
    output_paths['merged'] = out_png_merged
    
    print(f"Saved {number_of_channels} channel images + merged image to: {results_folder}")
    
    return output_paths


# =============================================================================
# COMBINED PUBLICATION FIGURE
# =============================================================================

def plot_combined_cell_and_kymograph(
    image_TZXYC,
    coordinates_roi,
    list_selected_frames,
    x_title_list,
    length_kymograph_line=50,
    cmap_list=None,
    y_label_list=None,
    masks_TXY=None,
    min_size_image=230,
    plot_name='combined_figure',
    pixel_xy_um=0.2,
    scalebar_size=8,
    line_color_FRAP='yellow',
    frame_for_merge=10,
    gamma=0.6,
    clip_limit=0.001,
    dpi=600,
    fig_width=12,
    fig_height=6,
    kymograph_ratio=0.4,
    results_folder=None,
):
    """
    Create a combined publication figure with cell images and kymographs.
    
    Layout:
    - Top row: Cell images for each channel + merged image
    - Bottom rows: Kymographs for each channel
    
    Parameters
    ----------
    image_TZXYC : np.ndarray
        5D image array (T, Z, Y, X, C)
    coordinates_roi : np.ndarray
        ROI coordinates (T, 2) with x, y positions
    list_selected_frames : list
        Frame indices to mark on kymograph
    x_title_list : list
        Labels for the time points
    length_kymograph_line : int
        Length of kymograph line profile in pixels
    cmap_list : list, optional
        Colormap list for each channel
    y_label_list : list, optional
        Channel labels
    masks_TXY : np.ndarray, optional
        Masks for cropping around the cell
    min_size_image : int
        Minimum crop size around cell
    plot_name : str
        Output filename prefix
    pixel_xy_um : float
        Pixel size in microns
    scalebar_size : int
        Scale bar font size
    line_color_FRAP : str
        Color for the FRAP line on merged image
    frame_for_merge : int
        Frame index to use for merged image
    gamma : float
        Gamma correction for merged image
    clip_limit : float
        CLAHE clip limit for merged image
    dpi : int
        Output DPI
    fig_width : float
        Total figure width in inches (default: 12)
    fig_height : float
        Total figure height in inches (default: 6)
    kymograph_ratio : float
        Kymograph height as fraction of cell image height (default: 0.4)
        Values < 1 make kymographs shorter than cell images
        Values > 1 make kymographs taller than cell images
    results_folder : Path, optional
        Output directory
        
    Returns
    -------
    Path to saved figure
    """
    if results_folder is None:
        results_folder = Path('.').resolve()

    number_of_channels = image_TZXYC.shape[-1]
    num_time_points = image_TZXYC.shape[0]
    height, width = image_TZXYC.shape[2:4]

    if cmap_list is None:
        cmap_list = [green_colormap, magenta_colormap][:number_of_channels]
    if y_label_list is None:
        y_label_list = [f'Ch {ch}' for ch in range(number_of_channels)]

    # Crop handling
    use_crop = False
    y_start, y_end, x_start, x_end = 0, height, 0, width

    if masks_TXY is not None:
        mask_xy = masks_TXY.max(axis=0)
        ys, xs = np.where(mask_xy > 0)
        if ys.size > 0:
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2
            half = min_size_image // 2
            y_start = center_y - half
            y_end = y_start + min_size_image
            x_start = center_x - half
            x_end = x_start + min_size_image

            pad_top = max(0, -y_start)
            pad_bottom = max(0, y_end - image_TZXYC.shape[2])
            pad_left = max(0, -x_start)
            pad_right = max(0, x_end - image_TZXYC.shape[3])
            if any((pad_top, pad_bottom, pad_left, pad_right)):
                image_TZXYC = np.pad(
                    image_TZXYC,
                    ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant', constant_values=0
                )
                y_start += pad_top
                y_end += pad_top
                x_start += pad_left
                x_end += pad_left
            use_crop = True
    
    # Bug fix: Update width/height after potential padding
    height, width = image_TZXYC.shape[2:4]

    # Build kymographs
    y_kymo = length_kymograph_line
    x_kymo = num_time_points
    kymographs = []
    min_kymo_height = y_kymo  # Track minimum height

    for ch in range(number_of_channels):
        kymograph = np.zeros((y_kymo, x_kymo))
        for i in range(x_kymo):
            coord_x = int(np.round(coordinates_roi[i, 0]))
            coord_y = int(np.round(coordinates_roi[i, 1]))
            coord_x = np.clip(coord_x, 0, width - 1)
            coord_y = np.clip(coord_y, 0, height - 1)
            start_x = max(coord_x - y_kymo // 2, 0)
            end_x = min(coord_x + y_kymo // 2, width)
            range_x = np.arange(start_x, end_x, dtype=int)
            current_y = len(range_x)
            line_profile = image_TZXYC[i, 0, coord_y, range_x, ch]
            kymograph[:current_y, i] = line_profile
            min_kymo_height = min(min_kymo_height, current_y)
        kymographs.append(kymograph)
    
    # Bug fix: Truncate all kymographs to same height
    kymographs = [k[:min_kymo_height, :] for k in kymographs]

    # Create merged image
    frame = frame_for_merge
    H0, W0 = image_TZXYC.shape[2], image_TZXYC.shape[3]

    if use_crop:
        img_crop = image_TZXYC[:, :, y_start:y_end, x_start:x_end, :]
        H, W = min_size_image, min_size_image
        roi_x_offset = x_start
        roi_y_offset = y_start
    else:
        img_crop = image_TZXYC
        H, W = H0, W0
        roi_x_offset = 0
        roi_y_offset = 0

    # Calculate FRAP line position
    cx = int(np.round(coordinates_roi[frame, 0])) - roi_x_offset
    cy = int(np.round(coordinates_roi[frame, 1])) - roi_y_offset
    cx = np.clip(cx, 0, W - 1)
    cy = np.clip(cy, 0, H - 1)
    line_len = length_kymograph_line
    sx = max(cx - line_len // 2, 0)
    ex = min(cx + line_len // 2, W - 1)
    range_x_line = np.arange(sx, ex + 1)

    # Create merged RGB
    img = img_crop[frame, 0, :, :, :].astype(np.float32)
    C = img.shape[-1]
    merged = np.zeros((H, W, 3), dtype=np.float32)

    for i in range(C):
        ch = img[..., i]
        p1, p99 = np.percentile(ch, 0.1), np.percentile(ch, 99.9)
        ch = np.clip(ch, p1, p99)
        ch = (ch - p1) / (p99 - p1 + 1e-8)
        ch = gaussian_filter(ch, sigma=0.3)
        ch = ch ** gamma
        col = cmap_list[i](ch)[..., :3]
        merged += col
    merged /= float(C)
    merged = np.clip(merged, 0, 1)
    merged = exposure.equalize_adapthist(merged, clip_limit=clip_limit)

    # Create figure layout
    num_cols = number_of_channels + 1
    num_rows = 1 + number_of_channels
    
    # Build height ratios: cell images = 1.0, kymographs = kymograph_ratio
    height_ratios = [1.0] + [kymograph_ratio] * number_of_channels

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    gs = fig.add_gridspec(
        num_rows, num_cols,
        height_ratios=height_ratios,
        width_ratios=[1] * num_cols,
        hspace=0.08, wspace=0.005,
        left=0.05, right=0.95, top=0.92, bottom=0.08
    )

    # Top row: cell images
    cell_axes = []
    for ch in range(number_of_channels):
        ax = fig.add_subplot(gs[0, ch])
        if use_crop:
            sub = img_crop[frame, 0, :, :, ch]
        else:
            sub = image_TZXYC[frame, 0, :, :, ch]
        ax.imshow(sub, vmax=np.percentile(sub, 99.9), cmap=cmap_list[ch])
        ax.set_title(y_label_list[ch], fontsize=10, fontname='Arial',)
        ax.set_xticks([])
        ax.set_yticks([])
        cell_axes.append(ax)

    # Merged image
    ax_merge = fig.add_subplot(gs[0, number_of_channels])
    ax_merge.imshow(merged, interpolation='bicubic')
    ax_merge.plot(range_x_line, [cy] * len(range_x_line), color=line_color_FRAP, linewidth=3, alpha=0.9)
    ax_merge.set_title('Merge', fontsize=10, fontname='Arial',)
    ax_merge.set_xticks([])
    ax_merge.set_yticks([])

    if scalebar_size > 0:
        scalebar_merge = ScaleBar(
            dx=pixel_xy_um, units='um', fixed_value=10, fixed_units='um',
            location='lower right', box_color='black', color='white',
            font_properties={'size': scalebar_size}
        )
        ax_merge.add_artist(scalebar_merge)
    cell_axes.append(ax_merge)

    # Kymographs
    fig.canvas.draw()
    first_ax_bbox = cell_axes[0].get_position()
    last_ax_bbox = cell_axes[-1].get_position()
    kymo_left = first_ax_bbox.x0
    kymo_right = last_ax_bbox.x1
    kymo_width = kymo_right - kymo_left

    total_height_ratio = 1.0 + kymograph_ratio * number_of_channels
    kymo_height_frac = kymograph_ratio / total_height_ratio
    cell_bottom = first_ax_bbox.y0
    row_gap = 0.02

    for ch in range(number_of_channels):
        kymo_bottom = cell_bottom - (ch + 1) * (kymo_height_frac * 0.8 + row_gap)
        kymo_height = kymo_height_frac * 0.75
        ax_kymo = fig.add_axes([kymo_left, kymo_bottom, kymo_width, kymo_height])
        kymograph = kymographs[ch]
        ax_kymo.imshow(kymograph, aspect='auto', cmap=cmap_list[ch],
                       vmax=np.percentile(kymograph, 99.5))
        for frame_idx in list_selected_frames:
            ax_kymo.axvline(x=frame_idx, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax_kymo.set_yticks([])
        ax_kymo.grid(False)
        if ch == number_of_channels - 1:
            ax_kymo.set_xticks(list_selected_frames)
            ax_kymo.set_xticklabels(x_title_list, fontsize=12, fontname='Arial', color='black')
            ax_kymo.tick_params(axis='x', labelsize=12, colors='black')
        else:
            ax_kymo.set_xticks([])
            ax_kymo.set_xticklabels([])
        for spine in ax_kymo.spines.values():
            # Bug fix: Use helper to handle string colormaps
            cmap_obj = _get_colormap(cmap_list[ch])
            spine.set_edgecolor(cmap_obj(0.8)[:3])
            spine.set_linewidth(2)

    out_path = Path(results_folder).joinpath(f'combined_{plot_name}.png')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    # save as svg
    out_path_svg = Path(results_folder).joinpath(f'combined_{plot_name}.svg')
    plt.savefig(out_path_svg, dpi=dpi, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.show()

    return out_path


# =============================================================================
# VIDEO EXPORT
# =============================================================================

def save_video_as_avi(
    image_TZXYC,
    avi_name,
    frame_values,
    list_axis_limits=None,
    cmap_list=None,
    y_label_list=None,
    fps=5,
    pixel_xy_um=0.2,
    scalebar_size=0,
    masks_TXY=None,
    min_size_image=250,
    percentile=99.9,
    dpi=200,
):
    """
    Save a multi-channel AVI video with consistent cropping.
    
    Crop behavior:
    - If masks_TXY is provided: crop is a fixed min_size_image × min_size_image window
      centered on the union of the masks across T.
    - Else if list_axis_limits is provided: use axis limits [xmin, xmax, ymin, ymax].
    - Else: show the full field of view.
    
    Parameters
    ----------
    image_TZXYC : np.ndarray
        5D image array (T, Z, Y, X, C)
    avi_name : str or Path
        Output filename
    frame_values : array-like
        Time values for each frame (for annotation)
    list_axis_limits : list, optional
        [xmin, xmax, ymin, ymax] in pixels
    cmap_list : list, optional
        Colormap for each channel
    y_label_list : list, optional
        Labels for each channel
    fps : int
        Frames per second
    pixel_xy_um : float
        Pixel size in microns
    scalebar_size : int
        Font size for scalebar (0 to disable)
    masks_TXY : np.ndarray, optional
        Masks for auto-cropping
    min_size_image : int
        Size of crop window when using masks
    percentile : float
        Percentile for brightness normalization
    dpi : int
        DPI for rendering frames
        
    Returns
    -------
    Path to saved AVI file
    """
    assert image_TZXYC.ndim == 5, f"Expected (T,Z,Y,X,C), got {image_TZXYC.shape}"
    T, Z, Y, X, C = image_TZXYC.shape
    assert Z >= 1, "Z dimension must be >= 1"
    assert len(frame_values) == T, f"len(frame_values)={len(frame_values)} != T={T}"

    if cmap_list is None:
        cmap_list = ["gray"] * C
    else:
        assert len(cmap_list) == C, "len(cmap_list) must equal number of channels"

    if y_label_list is None:
        y_label_list = [f"Ch {i}" for i in range(C)]
    else:
        assert len(y_label_list) == C, "len(y_label_list) must equal number of channels"

    # Compute crop region
    use_mask_crop = masks_TXY is not None
    if use_mask_crop:
        mask_xy = masks_TXY.max(axis=0)
        ys, xs = np.where(mask_xy > 0)
        if ys.size == 0:
            use_mask_crop = False
        else:
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2
            half = min_size_image // 2
            y_start = cy - half
            y_end = y_start + min_size_image
            x_start = cx - half
            x_end = x_start + min_size_image

            pad_top = max(0, -y_start)
            pad_bottom = max(0, y_end - Y)
            pad_left = max(0, -x_start)
            pad_right = max(0, x_end - X)
            if any((pad_top, pad_bottom, pad_left, pad_right)):
                image_TZXYC = np.pad(
                    image_TZXYC,
                    ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="constant", constant_values=0
                )
                T, Z, Y, X, C = image_TZXYC.shape
                y_start += pad_top
                y_end += pad_top
                x_start += pad_left
                x_end += pad_left
            xmin, xmax, ymin, ymax = x_start, x_end, y_start, y_end

    if not use_mask_crop:
        if list_axis_limits is None:
            xmin, xmax, ymin, ymax = 0, X, 0, Y
        else:
            xmin, xmax, ymin, ymax = list_axis_limits

    # Fixed brightness per channel
    vmax = []
    for ch in range(C):
        if use_mask_crop:
            tile = image_TZXYC[:, 0, y_start:y_end, x_start:x_end, ch]
        elif list_axis_limits is not None:
            tile = image_TZXYC[:, 0, ymin:ymax, xmin:xmax, ch]
        else:
            tile = image_TZXYC[:, 0, :, :, ch]
        vmax.append(float(np.percentile(tile, percentile)) if tile.size else 1.0)

    # Render frames
    frame_images = []
    for t in range(T):
        fig, axes = plt.subplots(1, C, figsize=(4 * C, 4))
        if C == 1:
            axes = [axes]

        for ch in range(C):
            if use_mask_crop:
                img = image_TZXYC[t, 0, y_start:y_end, x_start:x_end, ch]
            else:
                img = image_TZXYC[t, 0, :, :, ch]

            ax = axes[ch]
            ax.imshow(img, cmap=cmap_list[ch], vmax=vmax[ch], aspect="equal")
            if not use_mask_crop and list_axis_limits is not None:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            ax.axis("off")

        # Time label
        ax0 = axes[0]
        if use_mask_crop:
            text_x, text_y = 5, img.shape[0] - 5
        else:
            text_x, text_y = xmin + 5, ymax - 5
        ax0.text(text_x, text_y, f"{int(round(frame_values[t]))} s",
                 color="white", fontsize=14, ha="left", va="bottom",
                 bbox=dict(facecolor="black", alpha=0.7, pad=3))

        # Channel labels
        for ch in range(C):
            if y_label_list[ch]:
                if use_mask_crop:
                    lx = img.shape[1] // 2
                    ly = 10
                else:
                    lx = (xmin + xmax) // 2
                    ly = ymin + 10
                axes[ch].text(lx, ly, y_label_list[ch],
                              color="white", fontsize=12, ha="center", va="top",
                              bbox=dict(facecolor="black", alpha=0.7, pad=2))

        # Scalebar
        if scalebar_size > 0:
            sb = ScaleBar(dx=pixel_xy_um, units="um", length_fraction=0.25,
                          location="lower right", box_color="black", color="white",
                          font_properties={'size': scalebar_size})
            axes[-1].add_artist(sb)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, facecolor="black", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        frame = np.array(Image.open(buf))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_images.append(frame)

    if not frame_images:
        raise RuntimeError("No frames were generated.")

    # Ensure constant frame size
    h0, w0 = frame_images[0].shape[:2]
    for i in range(1, len(frame_images)):
        h, w = frame_images[i].shape[:2]
        if (h, w) != (h0, w0):
            frame_images[i] = cv2.resize(frame_images[i], (w0, h0), interpolation=cv2.INTER_AREA)

    # Write AVI
    avi_path = Path(avi_name).with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(avi_path), fourcc, float(fps), (w0, h0))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open the output file.")

    for fr in frame_images:
        writer.write(fr)
    writer.release()

    return avi_path


# =============================================================================
# IMAGE COMPOSITION
# =============================================================================

def compose_pngs(png_path1, png_path2, output_png, spacing=5, target_height=None, bg_color=(255, 255, 255)):
    """
    Combine two PNG images side-by-side.
    
    Parameters
    ----------
    png_path1 : Path or str
        Path to first image
    png_path2 : Path or str
        Path to second image
    output_png : Path or str
        Output path for combined image
    spacing : int
        Pixel spacing between images
    target_height : int, optional
        Target height (uses max height if not specified)
    bg_color : tuple
        Background color (R, G, B)
        
    Returns
    -------
    PIL.Image
        Combined image
    """
    img1 = Image.open(png_path1).convert("RGB")
    img2 = Image.open(png_path2).convert("RGB")
    
    if target_height is None:
        target_height = max(img1.height, img2.height)
    target_height = max(target_height, 800)  # Minimum 800px height
    
    def resize_to_height(img, height):
        ratio = height / img.height
        new_w = int(img.width * ratio)
        # Bug fix: Handle Pillow version compatibility for resampling constants
        try:
            lanczos = Image.Resampling.LANCZOS
            bicubic = Image.Resampling.BICUBIC
        except AttributeError:
            # Older Pillow versions
            lanczos = Image.LANCZOS
            bicubic = Image.BICUBIC
        if ratio < 1:
            return img.resize((new_w, height), lanczos)
        else:
            return img.resize((new_w, height), bicubic)
    
    img1_resized = resize_to_height(img1, target_height)
    img2_resized = resize_to_height(img2, target_height)
    
    total_width = img1_resized.width + img2_resized.width + spacing
    canvas = Image.new("RGB", (total_width, target_height), bg_color)
    canvas.paste(img1_resized, (0, 0))
    canvas.paste(img2_resized, (img1_resized.width + spacing, 0))
    
    output_png = Path(output_png)
    canvas.save(output_png, quality=95, optimize=True)
    
    output_high_res = output_png.parent / (output_png.stem + "_high_res.png")
    canvas.save(output_high_res, quality=100, optimize=False)
    
    return canvas


# =============================================================================
# TRAJECTORY PLOTTING
# =============================================================================

def plot_FRAP_trajectories(
    df_list,
    selected_dataset,
    apply_min_max_normalization=True,
    display_cell_count=True,
    selected_field='mean_roi_frap',  # Fixed: was 'mean_roi_frap_normalized'
    results_folder=None,
):
    """
    Plot FRAP recovery trajectories for a specified dataset.
    
    Parameters
    ----------
    df_list : DataFrame or list of DataFrames
        Data containing FRAP trajectories. Required columns:
        'frame', 'mean_roi_frap', 'image_name', 'dataset_type', 'subfolder_id'
    selected_dataset : str
        Dataset type to filter (e.g., 'utag', 'suntag', 'alfatag')
    apply_min_max_normalization : bool
        Whether to apply min-max normalization per cell
    display_cell_count : bool
        Whether to display cell count on plot
    selected_field : str
        Column name for intensity values
    results_folder : Path, optional
        Output directory
        
    Returns
    -------
    int
        Total number of cells plotted
    """
    if results_folder is None:
        results_folder = Path('.').resolve()
    
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelweight'] = 'normal'
    
    if isinstance(df_list, pd.DataFrame):
        df_list = [df_list]
    
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
    ax.set_facecolor('white')
    
    all_data = []
    total_number_cells = 0
    
    for df in df_list:
        df_sel = df[df['dataset_type'] == selected_dataset]
        unique_cells = df_sel[['subfolder_id', 'image_name']].drop_duplicates()
        
        for _, row in unique_cells.iterrows():
            subfolder_id = row['subfolder_id']
            cell = row['image_name']
            cell_data = df_sel[(df_sel['subfolder_id'] == subfolder_id) & (df_sel['image_name'] == cell)].copy()
            
            if apply_min_max_normalization:
                min_val = cell_data[selected_field].min()
                max_val = cell_data[selected_field].max()
                if max_val > min_val:
                    cell_data[selected_field] = (cell_data[selected_field] - min_val) / (max_val - min_val)
                else:
                    cell_data[selected_field] = 0.0

            ax.plot(cell_data['frame'], cell_data[selected_field], '-', color='dimgray', linewidth=0.5, alpha=0.2)
            all_data.append(cell_data[['frame', selected_field]])
            total_number_cells += 1

    if all_data:
        all_data_df = pd.concat(all_data, ignore_index=True)
        mean_trajectory = all_data_df.groupby('frame')[selected_field].mean().reset_index()
        ax.plot(mean_trajectory['frame'], mean_trajectory[selected_field], '-', color='green', linewidth=3, label='Mean Trajectory')

    ax.set_xlabel("Time (sec)", fontdict={'family': 'Arial', 'size': 20, 'color': 'black'})
    ax.set_ylabel("Normalized Intensity", fontdict={'family': 'Arial', 'size': 20, 'color': 'black'})
    ax.tick_params(axis='both', which='major', labelsize=16, colors='black')

    if display_cell_count:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        text_x = xlims[1] - 0.05 * (xlims[1] - xlims[0])
        text_y = ylims[0] + 0.05 * (ylims[1] - ylims[0])
        ax.text(text_x, text_y, f"n = {total_number_cells}",
                ha='right', va='bottom', fontsize=20, fontname="Arial", color='black')

    for spine in ax.spines.values():
        spine.set_color('black')

    ax.grid(False)
    plt.tight_layout()

    file_name = f"{selected_dataset}_FRAP_trajectories.png"
    file_path = results_folder.joinpath(file_name)
    plt.savefig(file_path, dpi=900, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    return total_number_cells


# =============================================================================
# MEAN TRAJECTORY PLOTTING (WITH SEM)
# =============================================================================

def plot_mean_trajectories_all(
    df,
    selected_datasets,
    selected_field='mean_roi_frap',
    apply_quality_check=True,
    drop_threshold=0.2,
    apply_min_max_normalization=True,
    fig_size=(6, 4),
    color_map=['darkgreen', 'black'],
    use_sem=True,
    results_folder=None,
):
    """
    Plot normalized FRAP recovery curves for each dataset with SEM shading.
    
    Parameters
    ----------
    df : DataFrame
        Data containing FRAP trajectories
    selected_datasets : list
        List of dataset types to plot
    selected_field : str
        Column name for intensity values
    apply_quality_check : bool
        Filter cells by initial drop magnitude
    drop_threshold : float
        Minimum drop required to include cell
    apply_min_max_normalization : bool
        Whether to normalize each cell to [0, 1]
    fig_size : tuple
        Figure size
    color_map : list
        Colors for each dataset
    use_sem : bool
        Use SEM (True) or STD (False) for error shading
    results_folder : Path, optional
        Output directory
        
    Returns
    -------
    matplotlib Axes object
    """
    if results_folder is None:
        results_folder = Path('.').resolve()

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_facecolor('white')

    for index, ds in enumerate(selected_datasets):
        df_ds = df[df['dataset_type'] == ds]
        if df_ds.empty:
            continue

        # Collect individual cell curves
        curves = []
        for (sub, img), group in df_ds.groupby(['subfolder_id', 'image_name']):
            series = group[['frame', selected_field]].copy()

            # Quality filter
            if apply_quality_check:
                init = series[selected_field].iloc[0]
                drop = init - series.loc[series['frame'] <= 20, selected_field].min()
                if drop <= drop_threshold:
                    continue

            # Min-max normalize
            if apply_min_max_normalization:
                mn, mx = series[selected_field].min(), series[selected_field].max()
                if mx > mn:
                    series[selected_field] = (series[selected_field] - mn) / (mx - mn)
                else:
                    series[selected_field] = 0

            curves.append(series.set_index('frame'))

        if not curves:
            continue

        all_cells = pd.concat(curves, axis=1)
        means = all_cells.mean(axis=1)
        errs = all_cells.sem(axis=1) if use_sem else all_cells.std(axis=1)

        c = color_map[index % len(color_map)]
        ax.plot(means.index, means.values, color=c, lw=2, label=ds)
        ax.fill_between(means.index, means - errs, means + errs, color=c, alpha=0.2)

    # Labels
    ax.set_xlabel("Time (sec)", fontsize=18, fontname='Arial', color='black')
    ax.set_ylabel("Normalized Intensity", fontsize=18, fontname='Arial', color='black')
    ax.tick_params(axis='both', which='major', labelsize=14, colors='black')

    # Spines
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)

    # Legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
              frameon=True, framealpha=1, edgecolor='black', facecolor='white')

    fig.tight_layout()
    fig.subplots_adjust(right=0.75)

    out_png = results_folder / "mean_FRAP_trajectories_all.png"
    plt.savefig(out_png, dpi=900, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    return ax


# =============================================================================
# BOX/SWARM PLOT FOR FINAL VALUES
# =============================================================================

def plot_box_swarm_final_values(
    df,
    selected_field,
    figsize=(6, 4),
    xlabel="Dataset Type",
    ylabel="Final Normalized Intensity",
    title="",
    y_min=None,
    y_max=None,
    swarm_color="black",
    tick_size=16,
    show_stats=False,
    order_categories=['sf', 'uv'],
    stats_offset_multiplier=0.15,
    results_folder=None,
):
    """
    Create boxplot with swarmplot overlay for final FRAP values.
    
    Final value is defined as the value at the maximum frame for each cell.
    
    Parameters
    ----------
    df : DataFrame
        Data with columns: 'dataset_type', 'cell_id', 'frame', and selected_field
    selected_field : str
        Column name to plot
    figsize : tuple
        Figure size
    xlabel, ylabel, title : str
        Axis labels and title
    y_min, y_max : float, optional
        Y-axis limits
    swarm_color : str
        Color for swarmplot points
    tick_size : int
        Font size for ticks
    show_stats : bool
        Show Mann-Whitney U test results
    order_categories : list
        Order of categories on x-axis
    stats_offset_multiplier : float
        Spacing for statistical annotations
    results_folder : Path, optional
        Output directory
        
    Returns
    -------
    matplotlib Axes object
    """
    if results_folder is None:
        results_folder = Path('.').resolve()
        
    sns.set_style("ticks")

    # Extract final time point for each cell
    final_df = df.loc[df.groupby('cell_id')['frame'].idxmax()].copy()

    # Print statistics
    print("Mean and Std of final values by dataset type:")
    for dataset in order_categories:
        data_subset = final_df[final_df['dataset_type'] == dataset][selected_field]
        mean_val = data_subset.mean()
        std_val = data_subset.std()
        count_val = data_subset.count()
        sem_val = data_subset.sem()
        print(f"  {dataset}: Mean = {mean_val:.4f}, Std = {std_val:.4f}, n = {count_val} (SEM = {sem_val:.4f})")

    # Create figure
    plt.figure(figsize=figsize, facecolor='white')
    ax = sns.boxplot(
        x="dataset_type", y=selected_field, data=final_df,
        order=order_categories, showfliers=False,
        boxprops={'facecolor': 'white', 'edgecolor': 'black'},
        medianprops={'color': 'red'},
        whiskerprops={'color': 'black'},
        capprops={'color': 'black'}
    )
    ax.set_facecolor('white')

    # Overlay swarmplot
    sns.swarmplot(
        x="dataset_type", y=selected_field, data=final_df,
        order=order_categories, color=swarm_color
    )

    plt.xlabel('')
    plt.ylabel(ylabel, fontsize=tick_size + 4, fontname="Arial", color='black')

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    if show_stats:
        _add_statistical_annotations(ax, final_df, selected_field, order_categories,
                                     stats_offset_multiplier, tick_size, y_max)

    ax.tick_params(axis='x', labelsize=tick_size + 4, colors='black')
    ax.tick_params(axis='y', labelsize=tick_size, colors='black')
    plt.xticks(fontname="Arial")
    plt.yticks(fontname="Arial")
    ax.set_xticklabels(order_categories, fontsize=tick_size + 2, fontname="Arial", color='black')

    plt.tight_layout()

    file_name = f"box_swarm_{selected_field}.png"
    file_path = results_folder.joinpath(file_name)
    plt.savefig(file_path, dpi=900, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    return ax


# =============================================================================
# BOX/SWARM PLOT FOR FIT RESULTS
# =============================================================================

def plot_box_swarm_fit_results(
    df,
    selected_field,
    figsize=(6, 4),
    xlabel="Dataset Type",
    ylabel="Half-life (s)",
    title="",
    y_min=None,
    y_max=None,
    swarm_color="black",
    tick_size=16,
    show_stats=False,
    order_categories=['sf', 'uv'],
    stats_offset_multiplier=0.15,
    results_folder=None,
):
    """
    Create boxplot with swarmplot overlay for FRAP fit results.
    
    Parameters
    ----------
    df : DataFrame
        Data with columns: 'dataset_type' and selected_field
    selected_field : str
        Column name to plot (e.g., 't_half_single')
    figsize : tuple
        Figure size
    xlabel, ylabel, title : str
        Axis labels and title
    y_min, y_max : float, optional
        Y-axis limits
    swarm_color : str
        Color for swarmplot points
    tick_size : int
        Font size for ticks
    show_stats : bool
        Show Mann-Whitney U test results
    order_categories : list
        Order of categories on x-axis
    stats_offset_multiplier : float
        Spacing for statistical annotations
    results_folder : Path, optional
        Output directory
        
    Returns
    -------
    matplotlib Axes object
    """
    if results_folder is None:
        results_folder = Path('.').resolve()
        
    sns.set_style("ticks")

    # Filter out NaN values
    df_clean = df.dropna(subset=[selected_field]).copy()

    # Create figure
    plt.figure(figsize=figsize, facecolor='white')
    ax = sns.boxplot(
        x="dataset_type", y=selected_field, data=df_clean,
        order=order_categories, showfliers=False,
        boxprops={'facecolor': 'white', 'edgecolor': 'black'},
        medianprops={'color': 'red'},
        whiskerprops={'color': 'black'},
        capprops={'color': 'black'}
    )
    ax.set_facecolor('white')

    # Overlay swarmplot
    sns.swarmplot(
        x="dataset_type", y=selected_field, data=df_clean,
        order=order_categories, color=swarm_color, size=4
    )

    # Print statistics
    print("Mean, Std, and SEM of fit results by dataset type:")
    for dataset in order_categories:
        data_subset = df_clean[df_clean['dataset_type'] == dataset][selected_field]
        mean_val = data_subset.mean()
        std_val = data_subset.std()
        count_val = data_subset.count()
        sem_val = data_subset.sem()
        print(f"  {dataset}: Mean = {mean_val:.4f}, Std = {std_val:.4f}, n = {count_val} (SEM = {sem_val:.4f})")

    plt.xlabel('')
    plt.ylabel(ylabel, fontsize=tick_size + 4, fontname="Arial", color='black')
    if title:
        plt.title(title, fontsize=tick_size + 4, fontname="Arial", color='black')

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    if show_stats and len(order_categories) >= 2:
        _add_statistical_annotations(ax, df_clean, selected_field, order_categories,
                                     stats_offset_multiplier, tick_size, y_max)

    ax.tick_params(axis='x', labelsize=tick_size + 4, colors='black')
    ax.tick_params(axis='y', labelsize=tick_size, colors='black')
    plt.xticks(fontname="Arial")
    plt.yticks(fontname="Arial")
    ax.set_xticklabels(order_categories, fontsize=tick_size + 2, fontname="Arial", color='black')

    plt.tight_layout()

    file_name = f"box_swarm_{selected_field}.png"
    file_path = results_folder.joinpath(file_name)
    plt.savefig(file_path, dpi=900, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    return ax


def _add_statistical_annotations(ax, df, selected_field, order_categories, 
                                  stats_offset_multiplier, tick_size, y_max=None):
    """
    Add Mann-Whitney U test annotations to a plot.
    
    Internal helper function for box/swarm plots.
    """
    # Build all pairwise combinations
    comps = [((i, j), j - i) for i, j in combinations(range(len(order_categories)), 2)]  
    
    if not comps:
        return
    
    # Calculate spacing
    max_data_height = df[selected_field].max()
    data_range = df[selected_field].max() - df[selected_field].min()
    offset = stats_offset_multiplier * data_range
    
    max_comparison_level = max([level for ((i, j), level) in comps])
    total_stats_height = offset * (max_comparison_level + 2)
    
    # Adjust y-axis if needed
    current_ylim = ax.get_ylim()
    if y_max is None:
        new_y_max = max_data_height + total_stats_height
        ax.set_ylim(current_ylim[0], new_y_max)

    # Add annotations
    for ((i, j), level) in comps:
        g1 = df.loc[df['dataset_type'] == order_categories[i], selected_field]
        g2 = df.loc[df['dataset_type'] == order_categories[j], selected_field]
        
        if len(g1) > 0 and len(g2) > 0:
            stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            
            # Significance stars
            if p < 1e-4:
                sig = '****'
            elif p < 1e-3:
                sig = '***'
            elif p < 1e-2:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            
            if sig == 'ns':
                continue
                
            # Draw bracket
            y1, y2 = g1.max(), g2.max()
            y_line = max(y1, y2) + offset * (level + 1)
            h = offset * 0.3
            
            ax.plot([i, i, j, j], [y_line, y_line + h, y_line + h, y_line], lw=1.5, c='k')
            ax.text((i + j) * 0.5, y_line + h, sig, ha='center', va='bottom',
                    fontsize=tick_size, fontname="Arial", color='k')
            
            print(f"Comparison {order_categories[i]} vs {order_categories[j]}:")
            print(f"  {order_categories[i]}: n = {len(g1)}")
            print(f"  {order_categories[j]}: n = {len(g2)}")
            print(f"  p-value: {p:.6f} ({sig})")
            print()


# =============================================================================
# DATA UTILITY FUNCTIONS
# =============================================================================

def load_frap_datasets(
    results_main_folder,
    subfolder_strings,
    list_datasets,
    selected_field='mean_roi_frap',
    apply_quality_check=True,
    drop_threshold=0.4,
    apply_min_max_normalization=True,
):
    """
    Scan FRAP result subfolders and build a combined DataFrame.

    Iterates over *results_main_folder*, matches each subfolder to a
    dataset type via *subfolder_strings*, reads the first qualifying CSV,
    optionally applies a quality filter (minimum bleach‐drop) and per‐cell
    min–max normalization.

    Parameters
    ----------
    results_main_folder : Path
        Root directory containing one subfolder per experiment.
    subfolder_strings : dict
        Mapping ``{dataset_type: [match_string_1, match_string_2]}``.
        A subfolder is assigned to ``dataset_type`` when its stem
        contains either match string.
    list_datasets : list of str
        Dataset types to iterate over (keys into *subfolder_strings*).
    selected_field : str
        Column name to analyse (default ``'mean_roi_frap'``).
    apply_quality_check : bool
        If ``True``, discard cells whose initial‐to‐minimum intensity
        drop within the first 20 s is ≤ *drop_threshold*.
    drop_threshold : float
        Minimum bleach‐drop to keep a cell (used when
        *apply_quality_check* is ``True``).
    apply_min_max_normalization : bool
        If ``True``, rescale each cell's *selected_field* to [0, 1].

    Returns
    -------
    combined_df : pd.DataFrame
        Columns include *selected_field*, ``'frame'``,
        ``'image_name'``, ``'dataset_type'``, ``'subfolder_id'``,
        and ``'cell_id'``.
    total_number_cells : int
        Number of unique cells in the combined DataFrame.
    """
    results_main_folder = Path(results_main_folder)
    subfolders = [f for f in results_main_folder.iterdir() if f.is_dir()]

    list_df_FRAP = []
    total_number_cells = 0

    for dataset_type in list_datasets:
        for subfolder in subfolders:
            match_a, match_b = subfolder_strings[dataset_type]
            if match_a not in subfolder.stem and match_b not in subfolder.stem:
                continue

            csv_files = [
                f for f in subfolder.glob("*.csv")
                if "no_roi_detected" not in f.name
                and "df_FRAP_fit" not in f.name
            ]
            if not csv_files:
                continue

            selected_df = pd.read_csv(csv_files[0])
            df = selected_df[['frame', selected_field, 'image_name']].copy()
            df['dataset_type'] = dataset_type
            df['subfolder_id'] = subfolder.stem

            # Unique cell IDs
            unique_cells = df['image_name'].unique()
            cell_id_map = {
                cell: f"{subfolder.stem}_{i}"
                for i, cell in enumerate(unique_cells, start=1)
            }
            df['cell_id'] = df['image_name'].map(cell_id_map)

            # Quality check
            if apply_quality_check:
                keep = []
                for cell in df['image_name'].unique():
                    cell_data = df[df['image_name'] == cell]
                    subset = cell_data[cell_data['frame'] <= 20]
                    if subset.empty:
                        continue
                    drop = subset[selected_field].iloc[0] - subset[selected_field].min()
                    if drop > drop_threshold:
                        keep.append(cell_data)
                    else:
                        print(
                            f"Cell {cell} in {subfolder.stem} dropped "
                            f"due to insufficient drop ({drop:.2f})"
                        )
                df = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()

            # Min–max normalization
            if apply_min_max_normalization and not df.empty:
                normalized = []
                for cell in df['image_name'].unique():
                    cell_data = df[df['image_name'] == cell].copy()
                    mn = cell_data[selected_field].min()
                    mx = cell_data[selected_field].max()
                    if mx > mn:
                        cell_data[selected_field] = (
                            cell_data[selected_field] - mn
                        ) / (mx - mn)
                    else:
                        cell_data[selected_field] = 0.0
                    normalized.append(cell_data)
                df = pd.concat(normalized, ignore_index=True)

            if not df.empty:
                list_df_FRAP.append(df)
                total_number_cells += df['cell_id'].nunique()

    combined_df = pd.concat(list_df_FRAP, ignore_index=True)
    print("Combined DataFrame shape:", combined_df.shape)
    print("Total number of processed cells:", total_number_cells)
    return combined_df, total_number_cells


def fit_all_cells(
    combined_df,
    datasets_to_process,
    frap_time,
    fit_function,
    selected_field='mean_roi_frap',
):
    """
    Batch‐fit FRAP recovery curves for every cell in *combined_df*.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Output of :func:`load_frap_datasets`.
    datasets_to_process : list of str
        Dataset types to fit (e.g. ``['sf', 'uv']``).
    frap_time : float
        Time of the bleach event (seconds).
    fit_function : callable
        The fitting function to call per cell.  Must accept
        ``(time=, intensity=, frap_time=, suptitle=, save_plot=,
        plot_name=)`` and return
        ``(t_half_single, t_half_double_1st, t_half_double_2nd,
        r2_single, r2_double)``.
    selected_field : str
        Column in *combined_df* to use as the intensity signal.

    Returns
    -------
    df_all_fit_results : pd.DataFrame
        One row per cell with fit parameters and metadata.
    """
    all_fit_results = []

    for dataset_type in datasets_to_process:
        print(f"\n{'=' * 50}")
        print(f"Processing dataset: {dataset_type}")
        print(f"{'=' * 50}")

        df_sel = combined_df[combined_df['dataset_type'] == dataset_type]
        unique_cells = df_sel[
            ['subfolder_id', 'image_name', 'cell_id']
        ].drop_duplicates()
        print(f"Found {len(unique_cells)} unique cells in '{dataset_type}'")

        for _, row in unique_cells.iterrows():
            subfolder_id = row['subfolder_id']
            image_name = row['image_name']
            cell_id = row['cell_id']

            df_image = df_sel[
                (df_sel['subfolder_id'] == subfolder_id)
                & (df_sel['image_name'] == image_name)
            ]
            print(
                f"Processing cell: {cell_id} "
                f"(subfolder: {subfolder_id}, image: {image_name}) "
                f"with {len(df_image)} frames"
            )

            try:
                (
                    t_half_single,
                    t_half_double_1st,
                    t_half_double_2nd,
                    r2_single,
                    r2_double,
                ) = fit_function(
                    time=df_image['frame'],
                    intensity=df_image[selected_field],
                    frap_time=frap_time,
                    suptitle=None,
                    save_plot=False,
                    plot_name=None,
                )
                print(
                    f"  Single exp t½: {t_half_single:.2f}, "
                    f"R²: {r2_single:.3f}"
                )
                print(
                    f"  Double exp t½1: {t_half_double_1st:.2f}, "
                    f"t½2: {t_half_double_2nd:.2f}, R²: {r2_double:.3f}"
                )
            except Exception as e:
                print(f"  Error fitting cell {cell_id}: {e}")
                t_half_single = t_half_double_1st = t_half_double_2nd = np.nan
                r2_single = r2_double = np.nan

            print("-" * 50)

            all_fit_results.append({
                'dataset_type': dataset_type,
                'subfolder_id': subfolder_id,
                'image_name': image_name,
                'cell_id': cell_id,
                't_half_single': t_half_single,
                'r_squared_single': r2_single,
                't_half_double_1st_process': t_half_double_1st,
                't_half_double_2nd_process': t_half_double_2nd,
                'r_squared_double': r2_double,
            })

    df_all_fit_results = pd.DataFrame(all_fit_results)
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total cells processed: {len(df_all_fit_results)}")
    print("\nCells per dataset:")
    print(df_all_fit_results['dataset_type'].value_counts())
    return df_all_fit_results
