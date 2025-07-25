"""
micro.py: is a library designed to process live-cell microscope images and perform single-molecule measurements. 
Author: Luis Aguilera
"""

# =============================================================================
# IMPORTS AND GLOBAL CONFIGURATION
# =============================================================================

import sys
import os
import logging
import re
import cv2
import json
import warnings
import pandas as pd
import numpy as np
import tifffile
from pathlib import Path
from PIL import Image
#import multiprocessing
import xml.etree.ElementTree as ET
from joblib import Parallel, delayed, cpu_count
NUMBER_OF_CORES = cpu_count()
# importing paths
gui_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(gui_dir, ".."))
sys.path.insert(0, os.path.join(repo_root, "src"))
src_dir = next((parent / 'src' for parent in Path().absolute().parents if (parent / 'src').is_dir()), None)
sys.path.append(str(src_dir))

from imports import *
# PyQt5 imports
from PyQt5.QtCore import (
    Qt,
    QThread,
    QTimer,
    QtMsgType,
    pyqtSignal,
    pyqtSlot,
    qInstallMessageHandler,
)
from PyQt5.QtGui import (
    QFont,
    QIcon,
    QImage,
    QPixmap,
    QPalette,
    QColor,
    QGuiApplication,
)
from PyQt5.QtWidgets import (
    QAbstractItemView, 
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QLineEdit,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy, 
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QInputDialog,
    QTextEdit,
)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import patches
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from functools import partial
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, label, center_of_mass
from trackpy.linking.utils import SubnetOversizeException


# =============================================================================
# UI DIALOGS, WIDGET, PLOTTING CLASSES
# =============================================================================

# Warnings and logging configuration
def configure_logging_and_styles():
    """
    Set up warnings filters, VisPy logging level, Qt message handler,
    and a logging filter to suppress specific stylesheet parse warnings.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
    try:
        from vispy import logging as vispy_logging
        vispy_logging.set_level('error')
        logging.getLogger('vispy').setLevel(logging.ERROR)
    except ImportError:
        pass
    def qt_message_handler(msg_type, context, message):
        msg = str(message)
        if "parse stylesheet" not in msg.lower():
            sys.__stderr__.write(msg + "\n")
    qInstallMessageHandler(qt_message_handler)
    class StyleParseFilter(logging.Filter):
        def filter(self, record):
            return "Could not parse stylesheet" not in record.getMessage()
    filter_instance = StyleParseFilter()
    logging.getLogger().addFilter(filter_instance)
    logging.getLogger('vispy').addFilter(filter_instance)


class Plots:
    def __init__(self, gui):
        self.gui = gui
    def plot_matrix_pair_crops(self, mean_crop, crop_size=11, plot_name=None, save_plots=False, plot_title=None,
                            max_crops_to_display=None, flag_vector=None, selected_channels=(0, 1), number_columns=20,
                            spacer_size=2, figure=None, show_text_ds=False, crop_spacing=5, flag_color="red"):
        """
        Plot pairs of image crops from different channels side by side in a grid layout.
        
        Creates a visualization where each crop shows two selected channels concatenated 
        horizontally with a spacer between them. Crops are arranged in a grid format
        and can be flagged with colored borders.
        
        Parameters
        ----------
        mean_crop : numpy.ndarray
            3D array of shape (height, width, channels) containing the crop data.
            Height should be divisible by crop_size to determine number of particles.
        crop_size : int, default=11
            Size of each individual crop in pixels (assumes square crops).
        plot_name : str, optional
            Name for the plot (not currently used in implementation).
        save_plots : bool, default=False
            Whether to save the plots (not currently used in implementation).
        plot_title : str, optional
            Title for the plot (not currently used in implementation).
        max_crops_to_display : int, optional
            Maximum number of crops to display. If None, displays all available crops.
        flag_vector : array-like, optional
            Boolean array indicating which crops to flag with colored borders.
            Must have same length as number of crops.
        selected_channels : tuple, default=(0, 1)
            Tuple of two channel indices to display side by side.
        number_columns : int, default=20
            Number of columns in the grid layout.
        spacer_size : int, default=2
            Width of the white spacer between the two channels in each crop pair.
        figure : matplotlib.figure.Figure, optional
            Existing figure to use for plotting. If None, creates a new Figure.
        show_text_ds : bool, default=False
            Whether to show text (not currently used in implementation).
        crop_spacing : int, default=5
            Spacing between crops in the grid layout.
        flag_color : str, default="red"
            Color for flagging crops (currently hardcoded to red in implementation).
        
        Returns
        -------
        None
            Modifies the provided figure or creates a new one with the crop visualization.
        
        Notes
        -----
        - Each channel is individually normalized to 0-255 range for display
        - Flagged crops get a red border on the top 2 rows of pixels
        - The function assumes the input mean_crop has particles stacked vertically
        - Images are resized using LANCZOS interpolation for better quality
        """
        def resize_image_to_target(image, target_size):
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize(target_size, Image.LANCZOS)
            return np.array(image_pil)

        number_color_channels = mean_crop.shape[-1]
        num_particles = mean_crop.shape[0] // crop_size
        if max_crops_to_display is None:
            max_crops_to_display = num_particles
        num_crops = min(num_particles, max_crops_to_display)
        num_rows = int(np.ceil(num_crops / number_columns))
        single_crop_width = crop_size * 2 + spacer_size
        single_crop_height = crop_size
        total_crop_width = single_crop_width + crop_spacing * 2
        total_crop_height = single_crop_height + crop_spacing * 2
        canvas_width = number_columns * total_crop_width
        canvas_height = num_rows * total_crop_height
        background_color = 0
        big_image = np.full((canvas_height, canvas_width, 3), background_color, dtype=np.uint8)
        idx = 0
        for row in range(num_rows):
            for col in range(number_columns):
                if idx < num_crops:
                    crop_img = mean_crop[idx * crop_size: (idx + 1) * crop_size, :, :]
                    combined_img_list = []
                    for ch in selected_channels:
                        if ch < number_color_channels:
                            channel_img = crop_img[:, :, ch]
                            ch_min = np.nanmin(channel_img)
                            ch_max = np.nanmax(channel_img)
                            ch_range = ch_max - ch_min
                            if ch_range > 0:
                                norm_channel_img = ((channel_img - ch_min) / ch_range * 255).astype(np.uint8)
                            else:
                                norm_channel_img = np.zeros_like(channel_img, dtype=np.uint8)
                            combined_img_list.append(norm_channel_img)
                        else:
                            combined_img_list.append(np.zeros_like(crop_img[:, :, 0], dtype=np.uint8))
                    spacer_value = 255
                    spacer_shape = (crop_size, spacer_size)
                    spacer = np.full(spacer_shape, spacer_value, dtype=np.uint8)
                    combined_img = np.concatenate([combined_img_list[0], spacer, combined_img_list[1]], axis=1)
                    target_size = (single_crop_width, single_crop_height)
                    combined_img = resize_image_to_target(combined_img, target_size)
                    combined_img_rgb = np.stack([combined_img, combined_img, combined_img], axis=-1)
                    if flag_vector is not None and flag_vector[idx]:
                        combined_img_rgb[0:2, :, 0] = 255
                        combined_img_rgb[0:2, :, 1] = 0
                        combined_img_rgb[0:2, :, 2] = 0
                    start_y = row * total_crop_height + crop_spacing
                    end_y = start_y + single_crop_height
                    start_x = col * total_crop_width + crop_spacing
                    end_x = start_x + single_crop_width
                    big_image[start_y:end_y, start_x:end_x, :] = combined_img_rgb
                idx += 1
        if figure is None:
            fig = Figure()
        else:
            fig = figure
            fig.clear()
        ax = fig.add_subplot(111)
        ax.imshow(big_image)
        ax.axis('off')
        fig.tight_layout()


    def plot_autocorrelation(self, mean_correlation, error_correlation, lags,
                            time_interval_between_frames_in_seconds=1, channel_label=0,
                            index_max_lag_for_fit=None, start_lag=0, line_color='blue',
                            plot_title=None, fit_type='linear', de_correlation_threshold=0.05,
                            normalize_plot_with_g0=False, axes=None, max_lag_index=None,
                            y_min_percentile=None, y_max_percentile=None):
        
        def single_exponential_decay(tau, A, tau_c, C):
            return A * np.exp(-tau / tau_c) + C
        if axes is None:
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
        else:
            ax = axes
        if normalize_plot_with_g0:
            normalized_correlation = mean_correlation / mean_correlation[start_lag]
        else:
            normalized_correlation = mean_correlation
        ax.plot(lags[start_lag:], normalized_correlation[start_lag:], 'o-', color=line_color, linewidth=2, label='Mean', alpha=0.5)
        ax.fill_between(lags[start_lag:],
                        normalized_correlation[start_lag:] - error_correlation[start_lag:],
                        normalized_correlation[start_lag:] + error_correlation[start_lag:],
                        color=line_color, alpha=0.1)
        if fit_type == 'linear':
            decorrelation_successful = False
            if index_max_lag_for_fit is None:
                index_max_lag_for_fit = normalized_correlation.shape[0]
            else:
                index_max_lag_for_fit = int(index_max_lag_for_fit)
            de_correlation_threshold_value = None
            try:
                decorrelation_successful = True
                de_correlation_threshold_value = normalized_correlation[index_max_lag_for_fit + start_lag]
                print(f"Decorrelation threshold value: {de_correlation_threshold_value}")
            except Exception:
                print('Could not find the decorrelation point automatically. Please provide the index_max_lag_for_fit')
                # Fall back to the last correlation point
                index_max_lag_for_fit = normalized_correlation.shape[0]
                de_correlation_threshold_value = normalized_correlation[index_max_lag_for_fit - 1]
                print(f"Falling back to last point: {de_correlation_threshold_value}")
                decorrelation_successful = False

            if decorrelation_successful:
                autocorrelations = normalized_correlation[start_lag:]
                selected_lags = lags[start_lag + 1:start_lag + index_max_lag_for_fit]
                selected_autocorrelations = autocorrelations[1:index_max_lag_for_fit]
                try:
                    slope, intercept, _, _, _ = linregress(selected_lags, selected_autocorrelations)
                    total_lags = np.arange(-1, index_max_lag_for_fit + 1) * time_interval_between_frames_in_seconds
                    line = slope * total_lags + intercept
                    dwell_time = (-intercept / slope)
                    dt = time_interval_between_frames_in_seconds
                    proj_lags = np.arange(start_lag, dwell_time + dt, dt)
                    proj_vals = slope * proj_lags + intercept
                    mask = proj_vals >= 0
                    proj_lags = proj_lags[mask]
                    proj_vals = proj_vals[mask]
                    ax.plot(proj_lags, proj_vals, 'r-', label='Linear Fit')
                    max_value = autocorrelations[0] * 0.8
                    text_str = f"Dwell Time: {dwell_time:.1f}"
                    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
                    ax.text(total_lags[-1] / 2, max_value, s=text_str, color='black', bbox=props, fontsize=10)
                except:
                    pass
            ax.axhline(y=de_correlation_threshold_value, color='r', linestyle='--', linewidth=1, label='Decor. Threshold')
            if plot_title is None:
                plot_title = f'Linear Fit (Signal {channel_label})'
            ax.set_title(plot_title, fontsize=10)
        elif fit_type == 'exponential':
            if index_max_lag_for_fit is not None:
                G_tau = normalized_correlation[start_lag:index_max_lag_for_fit]
                taus = lags[start_lag:index_max_lag_for_fit]
            else:
                G_tau = normalized_correlation[start_lag:]
                taus = lags[start_lag:]
            G_tau = np.nan_to_num(G_tau)
            tail_length = max(1, len(G_tau) // 10)
            C_guess = np.mean(G_tau[-tail_length:])
            G0 = G_tau[0]
            A_guess = G0 - C_guess
            target_value = C_guess + A_guess / np.e
            idx_tau_c = np.argmin(np.abs(G_tau - target_value))
            if idx_tau_c == 0:
                tau_c_guess = 0.5 * taus[-1]  # fallback
            else:
                tau_c_guess = taus[idx_tau_c]
            initial_guess = [A_guess, tau_c_guess, C_guess]
            params, _ = curve_fit(single_exponential_decay, taus, G_tau, p0=initial_guess)
            A_fitted, tau_c_fitted, C_fitted = params
            G_fitted = single_exponential_decay(taus, *params)
            G0_fitted = single_exponential_decay(0, A_fitted, tau_c_fitted, C_fitted)
            print("Fitted G(0):", G0_fitted)
            threshold_value = de_correlation_threshold
            try:
                dw_index = np.where(G_fitted < threshold_value)[0][0]
                dwell_time = taus[dw_index]
                ax.plot(taus, G_fitted, color='r', linestyle='-',
                        label=f'Fit: tau_c={tau_c_fitted:.1f}, Decorr={dwell_time:.1f}')
                ax.plot(dwell_time, G_fitted[dw_index], 'ro', markersize=10)
                ax.axhline(y=G_fitted[dw_index], color='r', linestyle='--', linewidth=1)
                if plot_title is None:
                    plot_title = f'Exponential Fit (Signal {channel_label})'
                ax.set_title(plot_title, fontsize=10)
            except IndexError:
                print("Could not find a time where G(τ) falls below threshold.")
                ax.axhline(y=threshold_value, color='r', linestyle='--', linewidth=1)
        ax.set_xlabel(r"$\tau$(au)")
        if normalize_plot_with_g0:
            ax.set_ylabel(r"$G(\tau)/G(0)$")
        else:
            ax.set_ylabel(r"$G(\tau)$")
        ax.grid(True)
        if max_lag_index is not None:
            max_lag_index = int(max_lag_index)
            if max_lag_index >= len(lags):
                max_lag_index = len(lags) - 1
                print('Warning: max_lag_index is out of range. Setting it to the last index')
            ax.set_xlim(lags[start_lag]-5, lags[max_lag_index])
        computed_y_min = np.nanpercentile(normalized_correlation[start_lag:], y_min_percentile)
        computed_y_max = np.nanpercentile(normalized_correlation[start_lag:], y_max_percentile)
        if not (np.isfinite(computed_y_min) and np.isfinite(computed_y_max)):
            ax.relim()            
            ax.autoscale_view()   
        else:
            ax.set_ylim(computed_y_min, computed_y_max)
        if axes is None:
            fig.tight_layout()


    def plot_crosscorrelation(self, mean_correlation, error_correlation, lags,
                            line_color='blue', plot_title=None,
                            normalize_plot_with_g0=True, axes=None,
                            max_lag_index=None, y_min_percentile=None, y_max_percentile=None):
        if axes is None:
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
        else:
            ax = axes
        start_lag = np.where(lags == 0)[0][0]
        if max_lag_index is not None:
            max_lag_index = int(max_lag_index)
            left_idx = max(0, start_lag - max_lag_index)
            right_idx = min(len(lags) - 1, start_lag + max_lag_index)
        else:
            left_idx = 0
            right_idx = len(lags) - 1
        lags_slice = lags[left_idx:right_idx + 1]
        mean_corr_slice = mean_correlation[left_idx:right_idx + 1]
        error_corr_slice = error_correlation[left_idx:right_idx + 1]
        if normalize_plot_with_g0:
            new_zero_index = start_lag - left_idx
            zero_val = mean_corr_slice[new_zero_index]
            if zero_val != 0:
                mean_corr_slice = mean_corr_slice / zero_val
                error_corr_slice = error_corr_slice / zero_val
        ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax.plot(lags_slice, mean_corr_slice, 'o-', color=line_color, linewidth=2, alpha=0.5, label='Mean')
        ax.fill_between(lags_slice,
                        mean_corr_slice - error_corr_slice,
                        mean_corr_slice + error_corr_slice,
                        color=line_color, alpha=0.1)
        number_points_to_smooth = 5
        mean_corr_smoothed = np.convolve(mean_corr_slice,
                                        np.ones(min(number_points_to_smooth, len(mean_corr_slice))) / min(number_points_to_smooth, len(mean_corr_slice)),
                                        mode='same')
        ax.plot(lags_slice, mean_corr_smoothed, color=line_color, label='Smoothed', alpha=0.5)
        if plot_title is None:
            plot_title = 'Cross-correlation'
        ax.set_title(plot_title, fontsize=10)
        max_idx_local = np.nanargmax(mean_corr_smoothed)
        max_lag = lags_slice[max_idx_local]
        max_value = mean_corr_smoothed[max_idx_local]
        ax.axvline(x=max_lag, color='r', linestyle='--', linewidth=2)
        text = r'$\tau_{max}$ = ' + f'{max_lag:.2f} au'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        xlim = np.nanpercentile(mean_corr_slice, y_min_percentile)
        ylim = np.nanpercentile(mean_corr_slice, y_max_percentile)
        ax.set_ylim(xlim, ylim)
        # Safely retrieve axis limits for positioning the τₘₐₓ label
        x_limits = ax.get_xlim()
        if isinstance(x_limits, (tuple, list)) and len(x_limits) >= 2:
            delta_x = x_limits[1] - x_limits[0]
        else:
            delta_x = max_lag
        x_position = max_lag + 0.05 * delta_x
        y_limits = ax.get_ylim()
        if isinstance(y_limits, (tuple, list)) and len(y_limits) >= 2:
            delta_y = y_limits[1] - y_limits[0]
        else:
            delta_y = max_value
        y_position = max_value - 0.1 * delta_y
        # Clamp text inside the visible plot region
        x_position = min(
            max(x_position, x_limits[0] + 0.05 * delta_x),
            x_limits[1] - 0.05 * delta_x
        )
        y_position = min(
            max(y_position, y_limits[0] + 0.05 * delta_y),
            y_limits[1] - 0.05 * delta_y
        )
        ax.text(x_position, y_position, s=text, color='black', bbox=props, fontsize=10)
        ax.set_xlabel(r"$\tau$(au)")
        if normalize_plot_with_g0:
            ax.set_ylabel(r"$G(\tau)/G(0)$")
        else:
            ax.set_ylabel(r"$G(\tau)$")
        ax.grid(False)
        if axes is None:
            fig.tight_layout()
        return max_lag

class Metadata:
    def __init__(
        self,
        correct_baseline,
        data_folder_path,
        list_images,
        list_names,
        voxel_yx_nm,
        voxel_z_nm,
        channel_names,
        number_color_channels,
        list_time_intervals,
        time_interval_value,
        bit_depth,
        image_stack,
        segmentation_mode,
        selected_image_index,
        channels_spots,
        channels_cytosol,
        channels_nucleus,
        min_length_trajectory,
        yx_spot_size_in_px,
        z_spot_size_in_px,
        cluster_radius_nm,
        maximum_spots_cluster,
        separate_clusters_and_spots,
        maximum_range_search_pixels,
        memory,
        de_correlation_threshold,
        max_spots_for_threshold,
        threshold_spot_detection,
        user_selected_threshold,
        image_source_combo,
        use_fixed_size_for_intensity_calculation,
        correlation_fit_type,
        index_max_lag_for_fit,
        photobleaching_calculated,
        min_percentage_data_in_trajectory,
        use_maximum_projection,
        photobleaching_mode,
        #photobleaching_model,
        photobleaching_radius,
        #photobleaching_number_removed_initial_points,
        file_path,
        use_ml_checkbox,
        ml_threshold_input,
        link_using_3d_coordinates,
        colocalization_method,
        colocalization_threshold_value,
        multi_tau
    ):
        # --- Correlation Tab ---
        self.correct_baseline = correct_baseline
        self.de_correlation_threshold = de_correlation_threshold
        self.correlation_fit_type = correlation_fit_type
        self.min_percentage_data_in_trajectory = min_percentage_data_in_trajectory
        self.index_max_lag_for_fit = index_max_lag_for_fit

        # --- General / Image Loading ---
        self.data_folder_path = data_folder_path
        self.list_images = list_images
        self.list_names = list_names
        self.voxel_yx_nm = voxel_yx_nm
        self.voxel_z_nm = voxel_z_nm
        self.channel_names = channel_names
        self.number_color_channels = number_color_channels
        self.list_time_intervals = list_time_intervals
        self.time_interval_value = time_interval_value
        self.bit_depth = bit_depth
        self.image_stack = image_stack
        self.selected_image_index = selected_image_index
        self.use_maximum_projection = use_maximum_projection
        self.segmentation_mode = segmentation_mode

        # --- Tracking Tab ---
        self.channels_spots = channels_spots
        self.channels_cytosol = channels_cytosol
        self.channels_nucleus = channels_nucleus
        self.min_length_trajectory = min_length_trajectory
        self.yx_spot_size_in_px = yx_spot_size_in_px
        self.z_spot_size_in_px = z_spot_size_in_px
        self.cluster_radius_nm = cluster_radius_nm
        self.maximum_spots_cluster = maximum_spots_cluster
        self.separate_clusters_and_spots = separate_clusters_and_spots
        self.maximum_range_search_pixels = maximum_range_search_pixels
        self.memory = memory
        self.max_spots_for_threshold = max_spots_for_threshold
        self.threshold_spot_detection = threshold_spot_detection
        self.user_selected_threshold = user_selected_threshold
        self.image_source_combo = image_source_combo
        self.use_fixed_size_for_intensity_calculation = use_fixed_size_for_intensity_calculation
        self.link_using_3d_coordinates = link_using_3d_coordinates
        self.multi_tau = multi_tau

        # --- Photobleaching Tab ---
        self.photobleaching_calculated = photobleaching_calculated
        #self.photobleaching_model = photobleaching_model
        self.photobleaching_mode = photobleaching_mode
        self.photobleaching_radius = photobleaching_radius
        #self.photobleaching_number_removed_initial_points = photobleaching_number_removed_initial_points

        # --- File Path for metadata ---
        self.file_path = file_path

        # --- Machine Learning ---
        self.use_ml_checkbox = use_ml_checkbox
        self.ml_threshold_input = ml_threshold_input

        # --- Colocalization Parameters ---
        self.colocalization_method = colocalization_method
        self.colocalization_threshold_value = colocalization_threshold_value

    def write_metadata(self):
        """
        Write out the metadata parameters, grouped by GUI tab or functionality,
        to a text file specified by `self.file_path`.
        """
        number_spaces_pound_sign = 60  
        try:
            with open(self.file_path, 'w') as fd:
                fd.write('#' * number_spaces_pound_sign)
                fd.write('\nAUTHOR INFORMATION')
                try:
                    fd.write('\n    Author: ' + getpass.getuser())
                    fd.write('\n    Hostname: ' + socket.gethostname() + '\n')
                except:
                    pass
                fd.write('\n    Created: ' + datetime.datetime.today().strftime('%d %b %Y'))
                fd.write('\n    Time: ' + str(datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute))
                fd.write('\n    Operating System: ' + sys.platform + '\n')
                fd.write('#' * number_spaces_pound_sign)
                # General Parameters
                fd.write('\nGENERAL INFORMATION')
                fd.write('\n    data_folder_path: ' + str(self.data_folder_path))
                fd.write('\n    list_images length: ' + str(len(self.list_images) if self.list_images else 0))
                fd.write('\n    list_names: ' + str(self.list_names))
                fd.write('\n    list_time_intervals: ' + str(self.list_time_intervals) + '\n')
                fd.write('#' * number_spaces_pound_sign)
                fd.write('\nSELECTED IMAGE')
                fd.write('\n    selected_image_name: ' + str(self.list_names[self.selected_image_index]))
                fd.write('\n    time_interval_value: ' + str(self.time_interval_value))
                fd.write('\n    voxel_yx_nm: ' + str(self.voxel_yx_nm))
                fd.write('\n    voxel_z_nm: ' + str(self.voxel_z_nm))
                fd.write('\n    channel_names: ' + str(self.channel_names))
                fd.write('\n    number_color_channels: ' + str(self.number_color_channels))
                fd.write('\n    bit_depth: ' + str(self.bit_depth))
                fd.write('\n    selected_image_index: ' + str(self.selected_image_index))
                if self.image_stack is not None:
                    fd.write('\n    image_stack shape: ' + str(self.image_stack.shape))
                else:
                    fd.write('\n    image_stack shape: None')
                fd.write('\n')
                fd.write('#' * number_spaces_pound_sign)
                fd.write('\nSEGMENTATION MODE')
                fd.write('\n    segmentation_mode: ' + str(self.segmentation_mode))
                fd.write('\n')
                # Photobleaching Parameters
                fd.write('#' * number_spaces_pound_sign)
                fd.write('\nPHOTOBLEACHING')
                fd.write('\n    photobleaching_calculated: ' + str(self.photobleaching_calculated))
                #fd.write('\n    photobleaching_model: ' + str(self.photobleaching_model))
                fd.write('\n    photobleaching_mode: ' + str(self.photobleaching_mode))
                fd.write('\n    photobleaching_radius: ' + str(self.photobleaching_radius))
                #fd.write('\n    photobleaching_number_removed_initial_points: ' + str(self.photobleaching_number_removed_initial_points))
                fd.write('\n')
                # Tracking Parameters
                fd.write('#' * number_spaces_pound_sign)
                fd.write('\nTRACKING PARAMETERS')
                fd.write('\n    channels_spots: ' + str(self.channels_spots))
                fd.write('\n    channels_cytosol: ' + str(self.channels_cytosol))
                fd.write('\n    channels_nucleus: ' + str(self.channels_nucleus))
                fd.write('\n    min_length_trajectory: ' + str(self.min_length_trajectory))
                fd.write('\n    yx_spot_size_in_px: ' + str(self.yx_spot_size_in_px))
                fd.write('\n    z_spot_size_in_px: ' + str(self.z_spot_size_in_px))
                fd.write('\n    cluster_radius_nm: ' + str(self.cluster_radius_nm))
                fd.write('\n    maximum_spots_cluster: ' + str(self.maximum_spots_cluster))
                fd.write('\n    separate_clusters_and_spots: ' + str(self.separate_clusters_and_spots))
                fd.write('\n    maximum_range_search_pixels: ' + str(self.maximum_range_search_pixels))
                fd.write('\n    memory: ' + str(self.memory))
                fd.write('\n    max_spots_for_threshold: ' + str(self.max_spots_for_threshold))
                fd.write('\n    threshold_spot_detection: ' + str(self.threshold_spot_detection))
                fd.write('\n    user_selected_threshold: ' + str(self.user_selected_threshold))
                fd.write('\n    use_fixed_size_for_intensity_calculation: ' + str(self.use_fixed_size_for_intensity_calculation))
                fd.write('\n    link_using_3d_coordinates: ' + str(self.link_using_3d_coordinates))
                fd.write('\n    multi_tau: ' + str(self.multi_tau))
                fd.write('\n    -------------------')
                if self.use_maximum_projection:
                    fd.write('\n    Using maximum projection (2D image) for tracking. Trackpy is used.')
                else:
                    fd.write('\n    Using 3D image for tracking. Big-FISH and Trackpy are used. ')
                using_corrected_image = False if self.image_source_combo == "Original Image" else True
                fd.write('\n    Using photobleaching corrected image for tracking: ' + str(using_corrected_image))
                fd.write('\n')
                # Correlation Parameters
                fd.write('#' * number_spaces_pound_sign)
                fd.write('\nCORRELATION PARAMETERS')
                fd.write('\n    correlation_fit_type: ' + str(self.correlation_fit_type))
                fd.write('\n    correct_baseline: ' + str(self.correct_baseline))
                fd.write('\n    de_correlation_threshold: ' + str(self.de_correlation_threshold))
                fd.write('\n    min_percentage_data_in_trajectory: ' + str(self.min_percentage_data_in_trajectory))
                fd.write('\n    index_max_lag_for_fit: ' + str(self.index_max_lag_for_fit))
                fd.write('\n')
                # Colocalization / ML PARAMETERS
                fd.write('#' * number_spaces_pound_sign)
                fd.write('\nCOLOCALIZATION / ML PARAMETERS')
                # Write new colocalization info:
                fd.write('\n    colocalization method: ' + str(self.colocalization_method))
                fd.write('\n    colocalization threshold value: ' + str(self.colocalization_threshold_value))
                fd.write('\n')
                # Reproducibility / Environment
                fd.write('#' * number_spaces_pound_sign)
                fd.write('\nREPRODUCIBILITY / ENVIRONMENT')
                fd.write('\n    Python version: ' + str(sys.version))
                fd.write('\n' + ('#' * number_spaces_pound_sign) + '\n')

        except Exception as e:
            print(f"Error writing metadata: {e}")


# =============================================================================
# =============================================================================
# MAIN APPLICATION WINDOW CLASS
# =============================================================================
# =============================================================================

class GUI(QMainWindow): 
    """
    Micro is a comprehensive GUI application for microscopy image analysis.
    A PyQt5 QMainWindow‐based application for interactive analysis of multi-dimensional microscopy image data.
    Organized into multiple tabs—Display, Segmentation, Photobleaching, Tracking, Distributions, Time Courses,
    Correlation, Colocalization (automated and manual), Tracking Visualization, Crops, and Export. 
    This GUI provides end-to-end workflows for loading, visualizing, processing, analyzing, and exporting microscopy datasets.
    Key Features:
        • Image I/O & Metadata
            – Load LIF or TIFF stacks, read embedded metadata, and prompt for missing fields (voxel sizes, time intervals).
            – Maintain a tree view of loaded files and allow closing and clearing of data.
        • Display & Visualization
            – Multi-channel Z-slice and time navigation, with per-channel contrast, smoothing, and custom colormaps.
            – Channel merging (up to 3 channels), background removal overlays, dark/light theme toggle.
            – Export static images (PNG, OME-TIFF) and time-lapse videos (MP4, GIF) with optional scalebar.
        • Segmentation
            – Manual polygon drawing or watershed segmentation with adjustable threshold factor.
            – Cellpose integration for cytosol/nucleus segmentation.
            – Display segmentation overlay and export binary masks (TIFF).
        • Photobleaching Correction
            – Region selection (inside/outside cell or circular), radius and time-point exclusion controls.
            – Fit intensity decay with exponential, double-exponential, or linear models (with/without baseline).
            – Visualize raw vs. corrected curves and export plots.
        • Particle Tracking
            – Spot detection (single frame or all frames) with percentile-based thresholding, size, clustering parameters.
            – Trajectory linking with maximum search range and memory settings; optional random-spot controls.
            – Plot trajectories, cluster sizes, particle IDs, timestamp and background overlays.
            – Export tracking data (CSV), static images, and videos.
        • Statistical Analyses
            – Distributions tab: histogram of spot intensities, sizes, PSF parameters, SNR, cluster sizes.
            – Time Courses tab: per-channel time-series of particle metrics with interactive percentile filtering.
            – Correlation tab: compute and visualize auto- and cross-correlations with linear or exponential fits.
        • Colocalization
            – Automated intensity‐based or ML‐based colocalization across channels.
            – Manual verification grid with flagging, mosaic export, and CSV output.
        • Cropping & Export
            – Define crops for focused analysis and export composite crop grids.
            – Batch export of images, masks, metadata, user comments, and data tables into structured result folders.
    """
    
    def __init__(self, icon_path):
        super().__init__()
        configure_logging_and_styles()
        self.setWindowTitle("Micro")
        self.setWindowIcon(QIcon(str(icon_path)))
        self.loaded_lif_files = {}
        self.correct_baseline = False
        self.data_folder_path = None
        self.list_images = None
        self.list_names = None
        self.voxel_yx_nm = None
        self.voxel_z_nm = None
        self.channel_names = None
        self.number_color_channels = None
        self.list_time_intervals = None
        self.bit_depth = None
        self.image_stack = None
        self.time_interval_value = None
        self.manual_segmentation_mask = None
        self.manual_current_image_name = None
        self.selected_image_index = 0
        self.current_channel = 0
        self.current_frame = 0
        self.channels_spots = 0
        self.channels_cytosol = [0]
        self.channels_nucleus = [None]
        self.min_length_trajectory = 20
        self.yx_spot_size_in_px = 5
        self.z_spot_size_in_px = 2
        self.cluster_radius_nm = 500
        self.maximum_spots_cluster = None
        self.separate_clusters_and_spots = False
        self.maximum_range_search_pixels = 7
        self.memory = 1
        self.de_correlation_threshold = 0.01
        self.max_spots_for_threshold = 3000
        self.index_max_lag_for_fit = None
        self.threshold_spot_detection = 0
        self.user_selected_threshold = 0.0
        self.image_source_combo_value = "Original Image"
        self.segmentation_mode = "None"
        self.use_fixed_size_for_intensity_calculation = True
        self.display_max_percentile = 99.95
        self.display_min_percentile = 0.1
        self.tracking_min_percentile = 99.95  # self.display_min_percentile
        self.tracking_max_percentile = 0.05   # self.display_max_percentile
        self.display_sigma = 0.7
        self.low_display_sigma = 0.15
        self.correlation_fit_type = 'linear'
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.playing = False
        self.photobleaching_calculated = False
        self.df_tracking = pd.DataFrame()
        self.has_tracked = False
        self.df_random_spots = pd.DataFrame()
        self.min_percentage_data_in_trajectory = 0.3
        self.use_maximum_projection = True
        self.photobleaching_mode = 'inside_cell'
        #self.photobleaching_model = 'exponential'
        self.photobleaching_radius = 20
        #self.photobleaching_number_removed_initial_points = None
        self.corrected_image = None
        self.colocalization_results = None
        self.link_using_3d_coordinates = True
        self.correlation_min_percentile = 0.0
        self.correlation_max_percentile = 100.0
        self.remove_outliers = True
        self.merged_mode = False
        self.ax_zoom = None  # initialize to None
        self.rect_zoom = None
        self.zoom_layout = QVBoxLayout()
        self.channelDisplayParams = {}
        self.random_mode_enabled = False
        self.segmentation_mask = None
        self.total_frames = 0
        self.tracking_remove_background_checkbox = False
        self.tracking_vis_merged = False
        self.plots = Plots(self)
        self.use_multi = False
        mi.Banner().print_banner()
        self.initUI()

# =============================================================================
# =============================================================================
# STARTING THE GUI 
# =============================================================================
# =============================================================================
    def initUI(self):
        """
        Initialize the main user interface of the application.
        This method performs the following steps:
        1. Creates and sets the central widget on the main window.
        2. Configures a vertical box layout for the central widget.
        3. Adds a QTabWidget with the following tabs:
            - Display
            - Segmentation
            - Photobleaching
            - Tracking
            - Distribution
            - Time Courses
            - Correlation
            - Colocalization
            - Colocalization Manual
            - Tracking Visualization
            - Crops
            - Export
        4. Connects the tab widget's currentChanged signal to the on_tab_change handler.
        5. Calls dedicated setup methods to populate each tab with its UI components.
        6. Applies the current theme based on the theme toggle state.
        7. Triggers an initial tab change to ensure the first tab is properly initialized.
        Args:
             self: Instance of the main window class.
        Returns:
             None
        """
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        self.display_tab = QWidget()
        self.tabs.addTab(self.display_tab, "Import")
        self.segmentation_tab = QWidget()
        self.tabs.addTab(self.segmentation_tab, "Segmentation")
        self.photobleaching_tab = QWidget()
        self.tabs.addTab(self.photobleaching_tab, "Photobleaching")
        self.tracking_tab = QWidget()
        self.tabs.addTab(self.tracking_tab, "Tracking")
        self.distribution_tab = QWidget()
        self.tabs.addTab(self.distribution_tab, "Distribution")
        self.time_course_tab = QWidget()
        self.tabs.addTab(self.time_course_tab, "Time Course")
        self.correlation_tab = QWidget()
        self.tabs.addTab(self.correlation_tab, "Correlation")
        self.colocalization_tab = QWidget()
        self.tabs.addTab(self.colocalization_tab, "Colocalization")
        self.colocalization_manual_tab = QWidget()
        self.tabs.addTab(self.colocalization_manual_tab, "Colocalization Manual")
        self.tracking_visualization_tab = QWidget()
        self.tabs.addTab(self.tracking_visualization_tab, "Tracking Visualization")
        self.crops_tab = QWidget()
        self.tabs.addTab(self.crops_tab, "Crops")
        self.export_tab = QWidget()
        self.tabs.addTab(self.export_tab, "Export")
        self.tabs.currentChanged.connect(self.on_tab_change)
        self.setup_display_tab()
        self.setup_segmentation_tab()
        self.setup_photobleaching_tab()
        self.setup_tracking_tab()
        self.setup_tracking_visualization_tab()
        self.setup_distributions_tab()
        self.setup_time_course_tab()
        self.setup_correlation_tab()
        self.setup_colocalization_tab()
        self.setup_colocalization_manual_tab()
        self.setup_crops_tab()
        self.setup_export_tab()
        self.applyTheme(self.themeToggle.isChecked())
        self.on_tab_change(0)


    def open_dimension_mapping_dialog(self, file_shape):
        """
        Open a modal dialog to map the dimensions of a loaded image file to standard 
        microscopy dimensions [T, Z, Y, X, C]. Returns a list of length 5 where each 
        element is either an integer (file axis index) or None (singleton dimension).
        """
        # Create the dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Map Image Dimensions")
        # Standard dimension labels and file dimensions list
        standard_labels = ["T", "Z", "Y", "X", "C"]
        file_dims = list(enumerate(file_shape))  # e.g. [(0, size0), (1, size1), ...]
        mapping = [None] * 5  # will store the mapping result
        # Set up the form layout
        form_layout = QFormLayout(dialog)
        dimensions_label = QLabel(f"Dimensions: {file_shape}", dialog)
        form_layout.addRow(dimensions_label)
        # Create combo boxes for each standard dimension
        combos = []
        for label in standard_labels:
            combo = QComboBox(dialog)
            combo.addItem("Singleton", None)  # option for a singleton dimension
            for idx, size in file_dims:
                combo.addItem(f"Dimension {idx} (size: {size})", idx)
            form_layout.addRow(f"{label}:", combo)
            combos.append(combo)
        # OK/Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        form_layout.addRow(button_box)
        # Define validation function for the OK button
        def validate_and_accept():
            selected_indices = []
            current_mapping = []
            # Gather selections from each combo box
            for combo in combos:
                val = combo.currentData() 
                current_mapping.append(val)
                if val is not None:
                    selected_indices.append(val)
            # Check for duplicate selections among file dimensions
            if len(selected_indices) != len(set(selected_indices)):
                QMessageBox.warning(dialog, "Mapping Error", 
                                     "Each file dimension can be assigned only once.")
                # Do not close the dialog, allow user to adjust selections
            else:
                # Valid mapping: copy to `mapping` and accept the dialog
                mapping[:] = current_mapping  # preserve the results
                dialog.accept()
        # Connect signals for OK and Cancel
        button_box.accepted.connect(validate_and_accept)
        button_box.rejected.connect(dialog.reject)
        # Execute the dialog modally and return the result if accepted
        if dialog.exec_() == QDialog.Accepted:
            return mapping
        else:
            return None  

    def create_channel_visualization_controls(self, channel_index, initial_params):
        """Create a QWidget with sliders and labels for adjusting a single channel's visualization parameters."""
        # Container widget and layout for the controls
        controls_widget = QWidget(self)
        layout = QFormLayout(controls_widget)
        params = initial_params.copy()  # copy initial params so we can modify locally
        # Min Percentile slider + label
        minSlider = QSlider(Qt.Horizontal)
        minSlider.setMinimum(0); minSlider.setMaximum(80)
        minSlider.setValue(int(params['min_percentile']))
        minLabel = QLabel(f"{params['min_percentile']:.2f}%")
        minRow = QHBoxLayout(); minRow.addWidget(minSlider); minRow.addWidget(minLabel)
        layout.addRow("Min Percentile:", minRow)
        # Max Percentile slider + label
        scale_factor = 100  # to allow two-decimal precision (e.g. 99.95%)
        maxSlider = QSlider(Qt.Horizontal)
        maxSlider.setMinimum(90 * scale_factor); maxSlider.setMaximum(100 * scale_factor)
        maxSlider.setValue(int(params['max_percentile'] * scale_factor))
        maxLabel = QLabel(f"{params['max_percentile']:.2f}%")
        maxRow = QHBoxLayout(); maxRow.addWidget(maxSlider); maxRow.addWidget(maxLabel)
        layout.addRow("Max Percentile:", maxRow)
        # High Sigma slider + label
        sigmaSlider = QSlider(Qt.Horizontal)
        sigmaSlider.setMinimum(0); sigmaSlider.setMaximum(50)   # 0.0–5.0 range (step 0.1)
        sigmaSlider.setValue(int(params['sigma'] * 10))
        sigmaLabel = QLabel(f"{params['sigma']:.2f}")
        sigmaRow = QHBoxLayout(); sigmaRow.addWidget(sigmaSlider); sigmaRow.addWidget(sigmaLabel)
        layout.addRow("High Sigma:", sigmaRow)
        # Low Sigma slider + label
        lowSigmaSlider = QSlider(Qt.Horizontal)
        lowSigmaSlider.setMinimum(0); lowSigmaSlider.setMaximum(50)  # 0.0–5.0 range
        lowSigmaSlider.setValue(int(params['low_sigma'] * 10))
        lowSigmaLabel = QLabel(f"{params['low_sigma']:.2f}")
        lowSigmaRow = QHBoxLayout(); lowSigmaRow.addWidget(lowSigmaSlider); lowSigmaRow.addWidget(lowSigmaLabel)
        layout.addRow("Low Sigma:", lowSigmaRow)
        # Connect slider value changes to update params and call the main handler
        def _update_min(val):
            params['min_percentile'] = float(val)
            minLabel.setText(f"{val:.2f}%")
            self.onChannelParamsChanged(channel_index, params)
        def _update_max(val):
            actual = float(val) / scale_factor
            params['max_percentile'] = actual
            maxLabel.setText(f"{actual:.2f}%")
            self.onChannelParamsChanged(channel_index, params)
        def _update_sigma(val):
            actual = float(val) / 10.0
            params['sigma'] = actual
            sigmaLabel.setText(f"{actual:.2f}")
            self.onChannelParamsChanged(channel_index, params)
        def _update_low_sigma(val):
            actual = float(val) / 10.0
            params['low_sigma'] = actual
            lowSigmaLabel.setText(f"{actual:.2f}")
            self.onChannelParamsChanged(channel_index, params)
        minSlider.valueChanged.connect(_update_min)
        maxSlider.valueChanged.connect(_update_max)
        sigmaSlider.valueChanged.connect(_update_sigma)
        lowSigmaSlider.valueChanged.connect(_update_low_sigma)
        return controls_widget

    def applyTheme(self, useDarkTheme: bool):
        """
        Slot to switch between Dark and Light theme styles.
        """
        if useDarkTheme:
            # Dark theme stylesheet
            dark_style = """
            QWidget { background-color: #2b2b2b; color: #e0e0e0; }
            QLabel { color: #e0e0e0; }

            /* Buttons: contrast on dark background */
            QPushButton {
                background-color: #c0c0c0;    /* light gray button */
                color: #000000;               /* black text */
                border: 1px solid #d0d0d0;    /* light gray border */
                border-radius: 2px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #a0a0a0;
            }
            QPushButton:checked {
                background-color: #007acc;
                color: #ffffff;
                border: 1px solid #d0d0d0;
            }

            /* Inputs */
            QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QComboBox {
                background-color: #3a3a3a;
                color: #e0e0e0;
                border: 1px solid #5a5a5a;
                border-radius: 4px;
            }

            /* Panels */
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: #e0e0e0;
            }

            /* Sliders */
            QSlider::groove:horizontal {
                height: 6px;
                background: #333333;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #e0e0e0;      /* bright handle for dark theme */
                border: 1px solid #ffffff; /* white border */
                width: 12px;
                margin: -4px 0;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #777777;
                border-radius: 3px;
            }

            /* List selection */
            QListWidget::item:selected, QListView::item:selected {
                background: #888888;
                color: #e0e0e0;
            }

            /* Tables */
            QTableWidget {
                background-color: #2b2b2b;
                alternate-background-color: #3a3a3a;
                gridline-color: #555555;
            }
            QTableWidget::item:selected {
                background: #007acc;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #3a3a3a;
                color: #e0e0e0;
                padding: 4px;
                border: none;
            }

            /* Spin Boxes */
            QAbstractSpinBox {
                qproperty-buttonSymbols: QAbstractSpinBox.UpDownArrows;
                background-color: #3a3a3a;
                color: #e0e0e0;
                border: 1px solid #5a5a5a;
                border-radius: 4px;
                padding-right: 18px;
            }
            QAbstractSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                background-color: transparent;
                border: none;
            }
            QAbstractSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                background-color: transparent;
                border: none;
            }
            QAbstractSpinBox::up-arrow {
                width: 8px; height: 8px;
                color: #e0e0e0;
            }
            QAbstractSpinBox::down-arrow {
                width: 8px; height: 8px;
                color: #e0e0e0;
            }
            """
            QApplication.instance().setStyleSheet(dark_style)
        else:
            # Light theme stylesheet
            light_style = """
            QWidget { background-color: #f0f0f0; color: #2b2b2b; }
            QLabel { color: #2b2b2b; }

            /* Buttons: contrast on light background */
            QPushButton {
                background-color: #404040;    /* dark gray button */
                color: #ffffff;               /* white text */
                border: 1px solid #404040;    /* dark gray border */
                border-radius: 2px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #303030;
            }
            QPushButton:checked {
                background-color: #007acc;
                color: #ffffff;
                border: 1px solid #404040;
            }

            /* Inputs */
            QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QComboBox {
                background-color: #ffffff;
                color: #2b2b2b;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }

            /* Panels */
            QGroupBox {
                font-weight: bold;
                border: 1px solid #aaaaaa;
                border-radius: 5px;
                margin-top: 10px;
                padding: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: #2b2b2b;
            }

            /* Sliders */
            QSlider::groove:horizontal {
                height: 6px;
                background: #bbbbbb;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #333333;      /* dark handle for light theme */
                border: 1px solid #000000; /* black border */
                width: 12px;
                margin: -4px 0;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #777777;
                border-radius: 3px;
            }

            /* List selection */
            QListWidget::item:selected, QListView::item:selected {
                background: #666666;
                color: #2b2b2b;
            }

            /* Tables */
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f0f0f0;
                gridline-color: #cccccc;
            }
            QTableWidget::item:selected {
                background: #007acc;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                color: #2b2b2b;
                padding: 4px;
                border: none;
            }

            /* Spin Boxes */
            QAbstractSpinBox {
                qproperty-buttonSymbols: QAbstractSpinBox.UpDownArrows;
                background-color: #ffffff;
                color: #2b2b2b;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding-right: 18px;
            }
            QAbstractSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                background-color: transparent;
                border: none;
            }
            QAbstractSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                background-color: transparent;
                border: none;
            }
            QAbstractSpinBox::up-arrow {
                width: 8px; height: 8px;
                color: #2b2b2b;
            }
            QAbstractSpinBox::down-arrow {
                width: 8px; height: 8px;
                color: #2b2b2b;
            }
            """
            QApplication.instance().setStyleSheet(light_style)

        # Re-apply toggle switch styling
        toggle_style = f"""
        QCheckBox#themeToggle {{
            color: {'#e0e0e0' if useDarkTheme else '#2b2b2b'};
        }}
        QCheckBox#themeToggle::indicator {{
            width: 40px; height: 20px;
            border-radius: 10px;
            background-color: #bbb;
        }}
        QCheckBox#themeToggle::indicator:checked {{
            background-color: #007acc;
        }}
        QCheckBox#themeToggle::indicator:unchecked {{
            background-color: #bbb;
        }}
        """
        self.themeToggle.setStyleSheet(toggle_style)
        # Enforce uniform spacing & margins on all tabs
        for tab in (
            self.display_tab, self.segmentation_tab, self.photobleaching_tab,
            self.tracking_tab, self.distribution_tab, self.time_course_tab,
            self.correlation_tab, self.colocalization_tab, self.crops_tab,
            self.export_tab
        ):
            layout = tab.layout()
            if layout:
                layout.setContentsMargins(8, 8, 8, 8)
                layout.setSpacing(8)

    def ask_for_metadata_from_user(self, missing_fields):
        """
        Prompt the user to enter missing metadata fields for TIFF images.
        missing_fields: list of strings naming each missing field.
        """
        for field in missing_fields:
            if "voxel size X" in field:
                # Ensure default is a float
                default_x = float(self.voxel_yx_nm) if isinstance(self.voxel_yx_nm, (int, float)) else 100.0
                val, ok = QInputDialog.getDouble(
                    self,
                    "Missing Metadata",
                    "Enter voxel size X (nm):",
                    default_x,
                    0.1,
                    1e6,
                    3
                )
                if ok:
                    self.voxel_yx_nm = val
                    self.voxel_size_x_nm = val
                    self.voxel_size_y_nm = val
            elif "voxel size Y" in field:
                default_y = float(self.voxel_size_y_nm) if isinstance(self.voxel_size_y_nm, (int, float)) else (float(self.voxel_yx_nm) if isinstance(self.voxel_yx_nm, (int, float)) else 100.0)
                val, ok = QInputDialog.getDouble(
                    self,
                    "Missing Metadata",
                    "Enter voxel size Y (nm):",
                    default_y,
                    0.1,
                    1e6,
                    3
                )
                if ok:
                    self.voxel_size_y_nm = val
            elif "voxel size Z" in field:
                default_z = float(self.voxel_z_nm) if isinstance(self.voxel_z_nm, (int, float)) else 100.0
                val, ok = QInputDialog.getDouble(
                    self,
                    "Missing Metadata",
                    "Enter voxel size Z (nm):",
                    default_z,
                    0.1,
                    1e6,
                    3
                )
                if ok:
                    self.voxel_z_nm = val
                    self.voxel_size_z_nm = val
            elif "time increment" in field or "TimeIncrement" in field:
                default_t = float(self.time_interval_value) if isinstance(self.time_interval_value, (int, float)) else 1.0
                val, ok = QInputDialog.getDouble(
                    self,
                    "Missing Metadata",
                    "Enter time increment (s):",
                    default_t,
                    0.01,
                    1e6,
                    3
                )
                if ok:
                    self.time_interval_value = val

    def onChannelParamsChanged(self, channel, params):
        self.channelDisplayParams[channel] = params
        if self.merged_mode:
            self.merge_color_channels()
        elif channel == self.current_channel:
            self.plot_image()
            if hasattr(self, 'min_percentile_slider_tracking'):
                self.update_tracking_sliders()
        self.plot_segmentation()
        self.plot_tracking()

    
    def create_channel_buttons(self):
        for btn in self.channel_buttons_display:
            btn.setParent(None)
        self.channel_buttons_display = []
        for idx, channel_name in enumerate(self.channel_names):
            button = QPushButton(f"Channel {idx}", self)
            button.clicked.connect(partial(self.update_channel, idx))
            self.channel_buttons_layout_display.addWidget(button)
            self.channel_buttons_display.append(button)
        for btn in self.channel_buttons_tracking:
            btn.setParent(None)
        self.channel_buttons_tracking = []
        for idx, channel_name in enumerate(self.channel_names):
            button = QPushButton(f"Channel {idx}", self)
            button.clicked.connect(partial(self.update_channel, idx))
            self.channel_buttons_layout_tracking.addWidget(button)
            self.channel_buttons_tracking.append(button)
        for btn in getattr(self, 'channel_buttons_tracking_vis', []):
            btn.setParent(None)
        self.channel_buttons_tracking_vis = []
        for idx, channel_name in enumerate(self.channel_names):
            btn = QPushButton(f"Channel {idx}", self)
            btn.clicked.connect(partial(self.select_tracking_vis_channel, idx))
            self.channel_buttons_layout_tracking_vis.addWidget(btn)
            self.channel_buttons_tracking_vis.append(btn)
        if hasattr(self, 'channel_buttons_crops'):
            for btn in self.channel_buttons_crops:
                btn.setParent(None)
        self.channel_buttons_crops = []
        for idx, channel_name in enumerate(self.channel_names):
            button = QPushButton(f"Channel {idx}", self)
            button.clicked.connect(partial(self.update_channel_crops, idx))
            self.channel_buttons_layout_crops.addWidget(button)
            self.channel_buttons_crops.append(button)



# =============================================================================
# =============================================================================
# DISPLAY TAB
# =============================================================================
# =============================================================================

    def convert_to_standard_format(self, image_stack):
        """
        Convert the loaded image_stack to standard 5D format [T, Z, Y, X, C].
        If image does not have 5 dimensions, prompt user to map file dimensions to standard and indicate missing dimensions.
        """
        if image_stack.ndim == 5:
            return image_stack
        mapping = self.open_dimension_mapping_dialog(image_stack.shape)
        if mapping is None:
            # User cancelled; return original array
            return image_stack
        used_axes = [m for m in mapping if m is not None]
        # Validate mapping indices within bounds
        if any(m < 0 or m >= image_stack.ndim for m in used_axes):
            QMessageBox.critical(self, "Error", f"Mapping indices {used_axes} are not valid for an image with {image_stack.ndim} dimensions.")
            return image_stack
        if used_axes:
            try:
                # Rearrange image so used axes appear in selected order
                transposed = np.transpose(image_stack, used_axes)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error transposing image: {e}")
                return image_stack
        else:
            transposed = image_stack
        used_shape = list(transposed.shape)
        new_shape = []
        for m in mapping:
            if m is None:
                new_shape.append(1)
            else:
                if not used_shape:
                    QMessageBox.critical(self, "Error", "Insufficient dimensions after transposition.")
                    return image_stack
                new_shape.append(used_shape.pop(0))
        try:
            final_array = np.reshape(transposed, new_shape)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reshaping image to standard format: {e}")
            return image_stack
        return final_array

    def open_image(self):
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Image Files",
            "",
            "Image Files (*.lif *.tif *.ome.tif);;All Files (*)",
            options=options
        )
        if not file_paths:
            return
        for path in file_paths:
            if path in self.loaded_lif_files:
                continue
            if path.lower().endswith('.lif'):
                # Load LIF file
                reader = mi.ReadLif(path=path, show_metadata=False, save_tif=False, save_png=False, format='TZYXC', lazy=True)
                _, names, yx_um, z_um, channels, nch, intervals, bd, list_laser_lines, list_intensities, list_wave_ranges = reader.read()
                self.loaded_lif_files[path] = (reader, names, yx_um, z_um, channels, nch, intervals, bd, list_laser_lines, list_intensities, list_wave_ranges)
                parent = QTreeWidgetItem(self.image_tree)
                parent.setText(0, Path(path).name)
                parent.setData(0, Qt.UserRole, {'file': path})
                for idx, nm in enumerate(names):
                    child = QTreeWidgetItem(parent)
                    child.setText(0, nm)
                    child.setData(0, Qt.UserRole, {'file': path, 'index': idx})
            elif path.lower().endswith(('.tif', '.ome.tif')):
                # Single-image TIFF: flag it to not show children
                parent = QTreeWidgetItem(self.image_tree)
                parent.setText(0, Path(path).name)
                parent.setData(0, Qt.UserRole, {'file': path, 'tif': True})
        self.image_tree.expandAll()
        if file_paths:
            first_path = file_paths[0]
            first_item = self.image_tree.topLevelItem(0)
            self.image_tree.setCurrentItem(first_item)
            # if first_path.lower().endswith('.lif'):
            #     self.load_lif_image(first_path, 0)
            # else:
            #     self.load_tif_image(first_path)

    def on_tree_item_clicked(self, item, column):
        info = item.data(0, Qt.UserRole) or {}
        if info.get('tif'):
            # Load as single-scene TIFF
            if getattr(self, 'data_folder_path', None) == info['file']:
                return
            self.load_tif_image(info['file'])
        elif info.get('index') is not None:
            # Load .lif scene by index
            self.load_lif_image(info['file'], info['index'])
        else:
            # Toggle folder expansion
            item.setExpanded(not item.isExpanded())
        self.plot_image()
        self.plot_tracking()
        self.reset_tracking_visualization_tab()

    def load_tif_image(self, file_path):
        """
        Load a single-image TIFF (or OME-TIFF) file as a single scene,
        set up metadata, reset the GUI, and display the first frame.
        """
        raw = tifffile.imread(file_path)
        voxel_x_nm = voxel_y_nm = voxel_z_nm = None
        dt_seconds = None
        detected_channel_names = None
        with tifffile.TiffFile(file_path) as tif:
            page0 = tif.pages[0]
            desc = page0.tags.get('ImageDescription')
            try:
                axes_str = tif.series[0].axes 
            except Exception:
                axes_str = None
            if desc is not None:
                desc_text = desc.value
                desc_stripped = desc_text.strip()
                if desc_stripped.startswith('{'):
                    # JSON metadata
                    try:
                        md = json.loads(desc_text)
                        if md.get("PhysicalSizeX") is not None:
                            voxel_x_nm = float(md["PhysicalSizeX"]) * 1000.0
                        if md.get("PhysicalSizeY") is not None:
                            voxel_y_nm = float(md["PhysicalSizeY"]) * 1000.0
                        if md.get("PhysicalSizeZ") is not None:
                            voxel_z_nm = float(md["PhysicalSizeZ"]) * 1000.0
                        if md.get("TimeIncrement") is not None:
                            dt_seconds = float(md["TimeIncrement"])
                        ch_dict = md.get("Channel", {})
                        if isinstance(ch_dict, dict):
                            detected_channel_names = ch_dict.get("Name")
                    except Exception:
                        print(f"Error parsing JSON ImageDescription metadata: {desc_text}")
                else:
                    # OME-XML metadata
                    try:
                        root = ET.fromstring(desc_text)
                        ns = {'ome': root.tag.split('}')[0].strip('{')}
                        pixels = root.find('.//ome:Pixels', ns)
                        if pixels is not None:
                            attrib = pixels.attrib
                            if 'PhysicalSizeX' in attrib:
                                voxel_x_nm = float(attrib['PhysicalSizeX']) * 1000.0
                            if 'PhysicalSizeY' in attrib:
                                voxel_y_nm = float(attrib['PhysicalSizeY']) * 1000.0
                            if 'PhysicalSizeZ' in attrib:
                                voxel_z_nm = float(attrib['PhysicalSizeZ']) * 1000.0
                            if 'TimeIncrement' in attrib:
                                dt_seconds = float(attrib['TimeIncrement'])
                            channel_elems = pixels.findall('ome:Channel', ns)
                            detected_channel_names = [ch.attrib.get('Name') for ch in channel_elems if 'Name' in ch.attrib]
                    except ET.ParseError:
                        print("Error parsing OME-XML ImageDescription metadata")
            else:
                print("No ImageDescription found in TIFF metadata.")
            if voxel_x_nm is None:
                x_res = page0.tags.get('XResolution')
                if x_res:
                    num, den = x_res.value
                    voxel_x_nm = float(num) / float(den) * 1000.0
            if voxel_z_nm is None:
                z_res = page0.tags.get('ZResolution')
                if z_res:
                    num, den = z_res.value
                    voxel_z_nm = float(num) / float(den) * 1000.0
        # If essential metadata is missing, prompt user (as per original logic)
        missing = []
        if voxel_x_nm is None:
            missing.append("voxel size X (nm)")
        if voxel_z_nm is None:
            missing.append("voxel size Z (nm)")
        if dt_seconds is None:
            missing.append("time increment (s)")
        if missing:
            voxel_x_nm = None
            voxel_z_nm = None
            dt_seconds = None
            missing = ["voxel size X (nm)", "voxel size Z (nm)", "time increment (s)"]
            self.ask_for_metadata_from_user(missing)
        # Set voxel sizes and time interval if available
        if voxel_x_nm is not None:
            self.voxel_yx_nm = voxel_x_nm
            self.voxel_size_x_nm = voxel_x_nm
            self.voxel_size_y_nm = voxel_x_nm
        if voxel_z_nm is not None:
            self.voxel_z_nm = voxel_z_nm
            self.voxel_size_z_nm = voxel_z_nm
        self.time_interval_value = dt_seconds if dt_seconds is not None else self.time_interval_value 

        # Determine the data axes order and reshape to standard [T, Z, Y, X, C] if needed
        if axes_str is not None:
            current_axes = list(axes_str)
            #print(f"Detected axes: {current_axes}"  )
            data = raw
            # Add singleton dimensions for missing axes
            for ax in ["T", "Z", "Y", "X", "C"]:
                if ax not in current_axes:
                    data = np.expand_dims(data, axis=-1)
                    current_axes.append(ax)
            # Reorder dimensions to [T, Z, Y, X, C]
            # perform a permutation based on the current axes if they are not in the standard order
            if current_axes != ["T", "Z", "Y", "X", "C"]:
                target_axes = ["T", "Z", "Y", "X", "C"]
                perm = [current_axes.index(ax) for ax in target_axes]
                raw = np.transpose(data, perm)
            else:
                # Already in standard order
                raw = data
        # Convert raw image data to standard internal format
        self.image_stack = self.convert_to_standard_format(raw)
        # Update dimensions and channel count
        dims = self.image_stack.shape
        T = dims[0]
        C = dims[4] if len(dims) == 5 else dims[-1]
        self.total_frames = T
        self.max_lag = T - 1
        if hasattr(self, 'max_lag_input'):
            self.max_lag_input.setMaximum(self.max_lag - 1)
            self.max_lag_input.setValue(self.max_lag - 1)
        self.number_color_channels = C
        if detected_channel_names and len(detected_channel_names) == self.number_color_channels:
            self.channel_names = detected_channel_names
        else:
            self.channel_names = [f"Channel {i}" for i in range(C)]
        # Populate various UI elements with image info
        p = Path(file_path)
        self.data_folder_path = p
        self.selected_image_name = p.stem
        self.list_names = [self.selected_image_name]
        self.list_time_intervals = [self.time_interval_value]
        if getattr(self, 'bit_depth', None) is None:
            dt = self.image_stack.dtype
            self.bit_depth = int(np.iinfo(dt).bits) if np.issubdtype(dt, np.integer) else 16
        self.file_label.setText(p.name)
        self.frames_label.setText(str(T))
        _, Z, Y, X, _ = self.image_stack.shape
        self.z_scales_label.setText(str(Z))
        # Configure the Z-slider range and default position (max -> max projection if Z>1)
        self.z_slider_display.setMinimum(0)
        if Z > 1:
            self.z_slider_display.setMaximum(Z)      # extra top value for max projection
        else:
            self.z_slider_display.setMaximum(0)      # single-plane image
        self.z_slider_display.setValue(Z if Z > 1 else 0)
        self.y_pixels_label.setText(str(Y))
        self.x_pixels_label.setText(str(X))
        self.channels_label.setText(str(C))
        self.voxel_yx_size_label.setText(f"{self.voxel_yx_nm:.0f} nm" if self.voxel_yx_nm is not None else "N/A")
        self.voxel_z_nm_label.setText(f"{self.voxel_z_nm:.0f} nm" if self.voxel_z_nm is not None else "N/A")
        self.bit_depth_label.setText(str(self.bit_depth))
        self.time_interval_label.setText(f"{self.time_interval_value} s" if self.time_interval_value is not None else "N/A")
        # Reset and clear various tabs and controls for new data
        self.display_min_percentile = 1.0
        self.display_max_percentile = 99.5
        self.channelDisplayParams.clear()
        self.reset_display_tab()
        self.reset_segmentation_tab()
        self.reset_photobleaching_tab()
        self.reset_tracking_tab() 
        if hasattr(self, 'min_percentile_slider_tracking'):
            self.update_tracking_sliders()
        self.reset_distribution_tab()
        self.reset_time_course_tab()
        self.reset_correlation_tab()
        self.reset_colocalization_tab()
        self.reset_crops_tab()  
        self.reset_export_comment()
        # Reset manual tracking/colocalization stats
        self.manual_scroll_area.setWidget(QWidget())
        self.manual_checkboxes = []
        self.manual_mean_crop = None
        self.df_manual_colocalization = pd.DataFrame()
        self.manual_stats_label.setText("Total Spots: 0 | Colocalized: 0 | 0.00%")
        self.has_tracked = False
        # Initialize current frame and channel
        self.current_frame = 0
        self.current_channel = 0
        self.time_slider_display.setMaximum(T - 1)
        self.time_slider_display.setValue(0)
        self.time_slider_tracking.setMaximum(T - 1)
        self.time_slider_tracking.setValue(0)
        self.time_slider_tracking_vis.setMaximum(T - 1)
        self.time_slider_tracking_vis.setValue(0)
        self.segmentation_time_slider.setMaximum(T - 1)
        # Create channel buttons/controls for various tabs
        self.create_channel_buttons()                   # Main display tab channel buttons
        self.create_segmentation_channel_buttons()      # Segmentation tab channel buttons/selection
        self.create_correlation_channel_checkboxes()    # Correlation tab channel checkboxes
        self.populate_colocalization_channels()         # Colocalization tab channel selections
        for btn in getattr(self, 'channel_buttons_crops', []):
            btn.setParent(None)
        self.channel_buttons_crops = []
        for idx in range(C):
            button = QPushButton(f"Channel {idx}", self)
            button.clicked.connect(partial(self.update_channel_crops, idx))
            self.channel_buttons_layout_crops.addWidget(button)
            self.channel_buttons_crops.append(button)
        # Clear and repopulate the channel visualization controls
        self.channelControlsTabs.clear()
        for ch in range(C):
            init = self.channelDisplayParams.get(ch, {
                'min_percentile': self.display_min_percentile,
                'max_percentile': self.display_max_percentile,
                'sigma': self.display_sigma,
                'low_sigma': self.low_display_sigma
            })
            widget = self.create_channel_visualization_controls(ch, init)
            self.channelControlsTabs.addTab(widget, f"Ch {ch}")
        # Update intensity channel combo boxes in other UI elements
        self.intensity_channel_combo.clear()
        for ch in range(self.number_color_channels):
            self.intensity_channel_combo.addItem(str(ch), ch)
        self.intensity_channel_combo.setCurrentIndex(0)
        self.time_course_channel_combo.clear()
        for ch in range(self.number_color_channels):
            self.time_course_channel_combo.addItem(str(ch), ch)
        self.time_course_channel_combo.setCurrentIndex(0)
        if hasattr(self, 'min_percentile_spinbox_tracking'):
            self.update_tracking_sliders()
        if self.playing:
            self.play_pause() 
        self.current_frame = 0
        # Finalize by plotting the first frame and initializing tracking overlay if needed
        self.plot_image()
        self.plot_tracking()


    def load_lif_image(self, file_path, image_index):
        reader, names, yx_um, z_um, channels, nch, intervals, bd, list_laser_lines, list_intensities, list_wave_ranges = self.loaded_lif_files[file_path]
        self.lif_reader = reader
        self.list_names = names
        self.voxel_yx_nm = yx_um * 1000
        self.voxel_z_nm  = z_um * 1000
        self.channel_names = channels
        self.number_color_channels = nch
        self.list_time_intervals = intervals
        self.time_interval_value = self.list_time_intervals[image_index]
        self.bit_depth = bd
        raw5d = reader.read_scene(image_index)
        self.image_stack = self.convert_to_standard_format(raw5d)
        self.data_folder_path = Path(file_path)
        self.selected_image_name = self.list_names[image_index]
        self.file_label.setText(self.data_folder_path.name)
        self.frames_label.setText(str(self.image_stack.shape[0]))
        _, Z, Y, X, _ = self.image_stack.shape
        self.z_scales_label.setText(str(Z))
        # Configure the Z-slider range and default position (max -> max projection)
        self.z_slider_display.setMinimum(0)
        if Z > 1:
            self.z_slider_display.setMaximum(Z) 
        else:
            self.z_slider_display.setMaximum(0) 
        self.z_slider_display.setValue(Z if Z > 1 else 0)
        self.y_pixels_label.setText(str(Y))
        self.x_pixels_label.setText(str(X))
        self.channels_label.setText(str(self.number_color_channels))
        self.voxel_yx_size_label.setText(f"{self.voxel_yx_nm:.0f} nm" if self.voxel_yx_nm is not None else "N/A")
        self.voxel_z_nm_label.setText(f"{self.voxel_z_nm:.0f} nm" if self.voxel_z_nm is not None else "N/A")
        self.bit_depth_label.setText(str(self.bit_depth))
        self.time_interval_label.setText(f"{self.time_interval_value} s" if self.time_interval_value is not None else "N/A")
        self.laser_lines_label.setText(str(list_laser_lines[image_index]))
        self.intensities_label.setText(str(list_intensities[image_index]))
        self.wave_ranges_label.setText(str(list_wave_ranges[image_index]))
        self.selected_image_index = image_index
        self.manual_current_image_name = None
        self.reset_segmentation_tab()
        self.reset_photobleaching_tab()
        self.reset_tracking_tab()
        if hasattr(self, 'min_percentile_slider_tracking'):
            self.update_tracking_sliders()
        self.reset_distribution_tab()
        self.reset_time_course_tab()
        self.reset_correlation_tab()
        self.reset_colocalization_tab()
        self.reset_crops_tab()
        self.reset_export_comment()
        self.manual_scroll_area.setWidget(QWidget())
        self.manual_checkboxes = []
        self.manual_mean_crop = None
        self.df_manual_colocalization = pd.DataFrame()
        self.manual_stats_label.setText("Total Spots: 0 | Colocalized: 0 | 0.00%")
        self.has_tracked = False
        self.photobleaching_calculated = False
        self.segmentation_mask = None
        self.df_tracking = pd.DataFrame()
        self.detected_spots_frame = None
        self.corrected_image = None
        self.colocalization_results = None
        self.correlation_results = []
        self.current_total_plots = None
        self.manual_current_image_name = None
        T = self.image_stack.shape[0]
        self.current_frame = 0
        self.current_channel = 0
        self.total_frames = T
        self.max_lag = self.image_stack.shape[0] - 1
        if hasattr(self, 'max_lag_input'):
            self.max_lag_input.setMaximum(self.max_lag - 1)
            self.max_lag_input.setValue(self.max_lag - 1)
        self.time_slider_display.setMaximum(T - 1)
        self.time_slider_display.setValue(0)
        self.time_slider_tracking.setMaximum(T - 1)
        self.time_slider_tracking.setValue(0)
        self.time_slider_tracking_vis.setMaximum(T - 1)
        self.time_slider_tracking_vis.setValue(0)
        self.segmentation_time_slider.setMaximum(T - 1)
        self.create_channel_buttons()
        self.create_correlation_channel_checkboxes()
        self.populate_colocalization_channels()
        self.channelControlsTabs.clear()
        self.create_segmentation_channel_buttons()
        self.display_min_percentile = 1.0
        self.display_max_percentile = 99.95
        self.channelDisplayParams.clear()
        num_ch = self.number_color_channels or 1
        for ch in range(num_ch):
            init_params = self.channelDisplayParams.get(ch, {
                'min_percentile': self.display_min_percentile,
                'max_percentile': self.display_max_percentile,
                'sigma': self.display_sigma,
                'low_sigma': self.low_display_sigma
            })
            widget = self.create_channel_visualization_controls(ch, init_params)
            self.channelControlsTabs.addTab(widget, f"Ch {ch}")

        self.intensity_channel_combo.clear()
        for ch in range(self.number_color_channels):
            self.intensity_channel_combo.addItem(str(ch), ch)
        self.intensity_channel_combo.setCurrentIndex(0)
        self.time_course_channel_combo.clear()
        for ch in range(self.number_color_channels):
            self.time_course_channel_combo.addItem(str(ch), ch)
        self.time_course_channel_combo.setCurrentIndex(0)
        if hasattr(self, 'min_percentile_spinbox_tracking'):
            self.update_tracking_sliders()
        if self.playing:
            self.play_pause() 
        self.current_frame = 0
        
        self.plot_image()
        self.plot_tracking()

    def play_pause(self):
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_button_display.setText("Play")
            self.play_button_tracking.setText("Play")
            self.play_button_tracking_vis.setText("Play")
        else:
            interval = 16 if sys.platform.startswith('win') else 100
            self.timer.start(interval)
            self.playing = True
            self.play_button_display.setText("Pause")
            self.play_button_tracking.setText("Pause")
            self.play_button_tracking_vis.setText("Pause")

    def update_channel(self, channel):
        self.current_channel = channel
        self.merged_mode = False
        if hasattr(self, 'channelControlsTabs'):
            self.channelControlsTabs.blockSignals(True)   
            self.channelControlsTabs.setCurrentIndex(channel) 
            self.channelControlsTabs.blockSignals(False)
        self.plot_image()
        self.plot_tracking()
        self.update_threshold_histogram()
        self.detected_spots_frame = None
        self.populate_colocalization_channels()

    def update_channel_crops(self, channel):
        self.current_channel = channel
        self.display_crops_plot()

    def update_frame(self, value):
        self.current_frame = value
        if self.time_slider_display.value() != value:
            self.time_slider_display.blockSignals(True)
            self.time_slider_display.setValue(value)
            self.time_slider_display.blockSignals(False)
        if self.time_slider_tracking.value() != value:
            self.time_slider_tracking.blockSignals(True)
            self.time_slider_tracking.setValue(value)
            self.time_slider_tracking.blockSignals(False)
        if hasattr(self, 'time_slider_tracking_vis') and self.time_slider_tracking_vis.value() != value:
            self.time_slider_tracking_vis.blockSignals(True)
            self.time_slider_tracking_vis.setValue(value)
            self.time_slider_tracking_vis.blockSignals(False)
        self.detected_spots_frame = None
        self.plot_image()
        self.plot_tracking()
        self.update_threshold_histogram()
        if hasattr(self, 'ax_tracking_vis'):
            self.display_tracking_visualization()

    def plot_image(self):
        self.figure_display.clear()
        self.ax_display = self.figure_display.add_subplot(111)
        self.ax_display.set_facecolor('black')
        self.ax_display.axis('off')
        if self.image_stack is not None:
            # Determine Z dimension size
            _, Z, _, _, _ = self.image_stack.shape  # shape is [T, Z, Y, X, C]
            z_val = self.z_slider_display.value() if hasattr(self, 'z_slider_display') else Z
            if self.merged_mode:
                if z_val == Z:
                    merged_img = self.compute_merged_image()
                    if merged_img is not None:
                        img_to_show = merged_img
                        if self.display_remove_background_checkbox.isChecked() and self.segmentation_mask is not None:
                            mask = (self.segmentation_mask > 0).astype(float)
                            img_to_show = img_to_show * mask[..., None] 
                        self.ax_display.imshow(img_to_show, vmin=0, vmax=1)
                    else:
                        self.ax_display.text(0.5, 0.5, 'Merged image not available.',
                                            horizontalalignment='center', verticalalignment='center',
                                            fontsize=12, color='white', transform=self.ax_display.transAxes)
                else:
                    plane_idx = int(z_val)
                    frame_image = self.image_stack[self.current_frame] 
                    plane_img = frame_image[plane_idx]                 
                    Y, X, channels = plane_img.shape if plane_img.ndim == 3 else (*plane_img.shape, 1)
                    if channels < 2:
                        img_to_show = plane_img.astype(float)
                    else:
                        num_channels_to_merge = min(channels, 3)
                        combined_image = np.zeros((Y, X, 3), dtype=np.float32)
                        def green_cmap(x):   return np.dstack((np.zeros_like(x), x, np.zeros_like(x)))
                        def magenta_cmap(x): return np.dstack((x, np.zeros_like(x), x))
                        def yellow_cmap(x):  return np.dstack((x, x, np.zeros_like(x)))
                        cmap_funcs = [green_cmap, magenta_cmap, yellow_cmap][:num_channels_to_merge]
                        for ch in range(num_channels_to_merge):
                            channel_img = plane_img[:, :, ch]
                            params = self.channelDisplayParams.get(ch, {
                                'min_percentile': self.display_min_percentile,
                                'max_percentile': self.display_max_percentile,
                                'sigma': self.display_sigma,
                                'low_sigma': self.low_display_sigma
                            })
                            min_val = np.percentile(channel_img, params['min_percentile'])
                            max_val = np.percentile(channel_img, params['max_percentile'])
                            norm = (np.clip(channel_img, min_val, max_val) - min_val) / (max_val - min_val + 1e-8)
                            if params['low_sigma'] > 0:
                                norm = gaussian_filter(norm, sigma=params['low_sigma'])
                            if params['sigma'] > 0:
                                norm = gaussian_filter(norm, sigma=params['sigma'])
                            combined_image += cmap_funcs[ch](norm)
                        img_to_show = np.clip(combined_image, 0, 1)
                    if self.display_remove_background_checkbox.isChecked() and self.segmentation_mask is not None:
                        mask = (self.segmentation_mask > 0).astype(float)
                        img_to_show = img_to_show * (mask[..., None] if img_to_show.ndim == 3 else mask)
                    self.ax_display.imshow(img_to_show, vmin=0, vmax=1)
            else:
                image_channel = self.image_stack[self.current_frame, :, :, :, self.current_channel]  # shape: (Z, Y, X)
                if z_val == Z:
                    data_img = np.max(image_channel, axis=0)
                else:
                    plane_idx = int(z_val)
                    data_img = image_channel[plane_idx]
                params = self.channelDisplayParams.get(self.current_channel, {
                    'min_percentile': self.display_min_percentile,
                    'max_percentile': self.display_max_percentile,
                    'sigma': self.display_sigma,
                    'low_sigma': self.low_display_sigma
                })
                rescaled = mi.Utilities().convert_to_int8(data_img, rescale=True,
                                                        min_percentile=params['min_percentile'],
                                                        max_percentile=params['max_percentile'])
                if params['low_sigma'] > 0:
                    rescaled = gaussian_filter(rescaled, sigma=params['low_sigma'])
                if params['sigma'] > 0:
                    rescaled = gaussian_filter(rescaled, sigma=params['sigma'])
                normalized = rescaled.astype(float) / 255.0
                normalized = normalized[..., 0]  
                img_to_show = normalized
                if self.display_remove_background_checkbox.isChecked() and self.segmentation_mask is not None:
                    mask = (self.segmentation_mask > 0).astype(float)
                    img_to_show = img_to_show * mask
                cmap_imagej = cmap_list_imagej[self.current_channel]
                self.ax_display.imshow(img_to_show, cmap=cmap_imagej, vmin=0, vmax=1)
            if self.display_time_text_checkbox.isChecked():
                current_time = self.current_frame * (float(self.time_interval_value) if self.time_interval_value else 1)
                time_str = f"{current_time:.2f} s"
                self.ax_display.text(0.05, 0.95, time_str, transform=self.ax_display.transAxes,
                                    verticalalignment='top', color='white', fontsize=12,
                                    bbox=dict(facecolor='black', alpha=0.5, pad=2))
            if hasattr(self, 'voxel_yx_nm') and self.voxel_yx_nm is not None:
                microns_per_pixel = self.voxel_yx_nm / 1000.0
                scalebar = ScaleBar(microns_per_pixel, units='um', length_fraction=0.2,
                                    location='lower right', box_color='black', color='white', font_properties={'size': 10})
                self.ax_display.add_artist(scalebar)
            self.figure_display.tight_layout()
        self.canvas_display.draw_idle()

    def update_z(self, value):
        """Handle Z-slider value change: update displayed image to selected z-plane or max projection."""
        # No need to sync other sliders; just refresh the display
        self.current_frame = 0  # Reset to first frame for new Z selection
        self.plot_image()

    def close_selected_file(self):
        """
        Remove the currently selected file (LIF or TIFF) from the tree and free its memory. If it was showing, clear the display.
        """
        item = self.image_tree.currentItem()
        if not item:
            return
        # If a child was selected, get its parent
        if item.parent():
            item = item.parent()
        info = item.data(0, Qt.UserRole) or {}
        file_path = info.get('file')
        if not file_path:
            return
        # Remove from loaded files dict
        self.loaded_lif_files.pop(file_path, None)
        # Remove from tree view
        idx = self.image_tree.indexOfTopLevelItem(item)
        if idx >= 0:
            self.image_tree.takeTopLevelItem(idx)
        # If this file was currently displayed, clear the display area
        if hasattr(self, 'data_folder_path') and str(self.data_folder_path) == file_path:
            # Clear image stack and canvas
            self.image_stack = None
            self.figure_display.clear()
            self.canvas_display.draw()
            # Clear info labels
            for lbl in (self.file_label, self.frames_label, self.z_scales_label, self.y_pixels_label, self.x_pixels_label,
                        self.channels_label, self.voxel_yx_size_label, self.voxel_z_nm_label, self.bit_depth_label, self.time_interval_label):
                lbl.setText("")
            # Clear controls
            self.channelControlsTabs.clear()
            self.time_slider_display.setEnabled(False)
            self.play_button_display.setEnabled(False)

    def on_tree_current_item_changed(self, current, previous):
        """
        Load the image whenever the selection moves via keyboard arrow keys.
        """
        if current:
            # Use the same loader as clicking
            self.on_tree_item_clicked(current, 0)
            self.reset_display_tab()
            self.plot_image()
            # Reset segmentation tab
            self.reset_segmentation_tab()
            self.plot_segmentation()
            self.reset_tracking_visualization_tab()
            # reset the current frame and channel to 0 
            self.current_frame = 0
            self.current_channel = 0

    def on_channel_tab_changed(self, index):
        if not self.merged_mode:
            self.current_channel = index
            self.plot_image()
        else:
            self.merge_color_channels()
        if hasattr(self, 'min_percentile_slider_tracking'):
            self.update_tracking_sliders()

    def compute_merged_image(self):
        if self.image_stack is None:
            return None
        # Get current frame’s multi-channel image
        if self.image_stack.ndim == 5:
            # [T, Z, Y, X, C]
            current_frame_image = self.image_stack[self.current_frame]  # shape: [Z, Y, X, C]
            max_proj = np.max(current_frame_image, axis=0)              # shape: [Y, X, C]
        elif self.image_stack.ndim == 4:
            max_proj = self.image_stack  # Already [Y, X, C]
        else:
            return None
        image_size_y, image_size_x, channels = max_proj.shape
        if channels < 2:
            return None  # Nothing to merge if only one channel
        num_channels_to_merge = min(channels, 3)
        # Define custom colormaps for each channel.
        def green_colormap(x):
            return np.dstack((np.zeros_like(x), x, np.zeros_like(x)))
        def magenta_colormap(x):
            return np.dstack((x, np.zeros_like(x), x))
        def yellow_colormap(x):
            return np.dstack((x, x, np.zeros_like(x)))
        cmap_list = ([green_colormap, magenta_colormap] if num_channels_to_merge == 2
                     else [green_colormap, magenta_colormap, yellow_colormap])
        combined_image = np.zeros((image_size_y, image_size_x, 3), dtype=np.float32)
        # For each channel to merge, apply channel-specific display parameters
        for i in range(num_channels_to_merge):
            channel_img = max_proj[:, :, i]
            # Get per-channel parameters or default to global
            params = self.channelDisplayParams.get(i, {
                'min_percentile': self.display_min_percentile,
                'max_percentile': self.display_max_percentile,
                'sigma': self.display_sigma,
                'low_sigma': self.low_display_sigma
            })
            min_val = np.percentile(channel_img, params['min_percentile'])
            max_val = np.percentile(channel_img, params['max_percentile'])
            norm_channel = (np.clip(channel_img, min_val, max_val) - min_val) / (max_val - min_val + 1e-8)
            # Optionally, apply Gaussian smoothing before merging
            if params['low_sigma'] > 0:
                norm_channel = gaussian_filter(norm_channel, sigma=params['low_sigma'])
            if params['sigma'] > 0:
                norm_channel = gaussian_filter(norm_channel, sigma=params['sigma'])
            colored_channel = cmap_list[i](norm_channel)
            combined_image += colored_channel
        merged_img = np.clip(combined_image, 0, 1)
        return merged_img

    def colorize_single_channel(self, gray_img, channel_index):
        """
        Given a single-channel (grayscale) image (uint8), return a 3-channel image
        where intensity is mapped to a specific color based on channel_index.
        Examples:
        - channel 0: green (0, intensity, 0)
        - channel 1: magenta (intensity, 0, intensity)
        - channel 2: yellow (intensity, intensity, 0)
        For other channels, uses a standard grayscale to BGR conversion.
        """
        if channel_index == 0:
            # Green: R=0, B=0, G=intensity
            color_img = np.dstack((np.zeros_like(gray_img), gray_img, np.zeros_like(gray_img)))
        elif channel_index == 1:
            # Magenta: G=0, R=B=intensity
            color_img = np.dstack((gray_img, np.zeros_like(gray_img), gray_img))
        elif channel_index == 2:
            # Yellow: B=0, R=G=intensity
            color_img = np.dstack((gray_img, gray_img, np.zeros_like(gray_img)))
        else:
            # Other channels: use OpenCV conversion (all channels equal)
            color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        return color_img

    def merge_color_channels(self):
        if self.image_stack is None:
            QMessageBox.information(self, "Merge Error", "No image loaded to merge channels.")
            return
        merged_img = self.compute_merged_image()
        if merged_img is None:
            QMessageBox.information(self, "Merge Error", "Not enough channels to merge or unsupported image format.")
            return
        self.merged_mode = True
        self.figure_display.clear()
        self.ax_display = self.figure_display.add_subplot(111)
        # Apply background removal if requested
        img_to_show = merged_img
        if self.display_remove_background_checkbox.isChecked() and self.segmentation_mask is not None:
            mask = (self.segmentation_mask > 0).astype(float)
            # expand mask to match RGB channels
            img_to_show = img_to_show * mask[..., None]
        self.ax_display.imshow(img_to_show, vmin=0, vmax=1)
        self.ax_display.axis('off')
        self.figure_display.tight_layout()
        self.canvas_display.draw()


    def control_panel_image_properties(self, parent_layout):
        self.channelControlsTabs = QTabWidget()
        self.channelControlsTabs.setStyleSheet("""
        QTabBar::tab {
            background: #353535;
            padding: 5px;
            color: #e0e0e0;              /* light text for dark background */
            border: 1px solid #555555;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: -1px;
        }
        QTabBar::tab:selected {
            background: #008fd5;
            color: #ffffff;              /* white text on blue */
            border: 1px solid #007acc;
            border-bottom: none;
        }
        QTabBar::tab:!selected {
            margin-top: 2px;
        }
        QTabBar::tab:hover {
            background: #505050;
        }
        """)
        # If the image isn’t loaded yet, add one tab with default values.
        self.channelControlsTabs.currentChanged.connect(self.on_channel_tab_changed)
        num_channels = 1
        if self.number_color_channels is not None and self.number_color_channels > 0:
            num_channels = self.number_color_channels
        for ch in range(num_channels):
            # Use per-channel parameters if already set, otherwise use global defaults
            init_params = self.channelDisplayParams.get(ch, {
                'min_percentile': self.display_min_percentile,
                'max_percentile': self.display_max_percentile,
                'sigma': self.display_sigma,
                'low_sigma': self.low_display_sigma
            })
            widget = self.create_channel_visualization_controls(ch, init_params)
            self.channelControlsTabs.addTab(widget, f"Ch {ch}")
        parent_layout.addWidget(self.channelControlsTabs)

    def setup_display_tab(self):
        """
        Initialize and configure the “Display” tab.

        This method builds a two‐column interface in `self.display_tab`. The left column (larger)
        hosts:
            - A Dark/Light theme toggle switch with custom styling, connected to `applyTheme`.
            - An “Open File” button to load image data.
            - A Matplotlib canvas showing the current image, backed by `self.figure_display` and `self.ax_display`.
            - A vertical Z‐slice slider (`self.z_slider_display`) for selecting image planes.
            - Channel‐management controls, including per‐channel buttons and a “Merge Channels” action.
            - Time navigation controls: a horizontal frame slider (`self.time_slider_display`) and a Play/Pause button.

        The right column (narrower) contains:
            - A QTreeWidget (`self.image_tree`) for selecting among loaded files.
            - A “Close File” button to remove the selected image.
            - Supplementary visualization controls inserted via `control_panel_image_properties`.
            - An image information panel (scrollable) displaying metadata such as filename, frames,
                dimensions, bit depth, voxel sizes, channels, and acquisition parameters.
            - Export buttons for saving the displayed image or video.
            - Optional checkboxes to toggle time stamp and background removal overlays.
        """

        display_main_layout = QHBoxLayout(self.display_tab)
        # Left side: vertical layout
        display_left_layout = QVBoxLayout()
        # Add Dark/Light theme toggle switch
        self.themeToggle = QCheckBox("Dark Theme")
        self.themeToggle.setObjectName("themeToggle")
        self.themeToggle.setChecked(True)
        self.themeToggle.setStyleSheet("""
            QCheckBox#themeToggle::indicator {
                width: 40px; height: 20px;
                border-radius: 10px;
                background-color: #bbb;
                position: relative;
            }
            QCheckBox#themeToggle::indicator:checked {
                background-color: #007acc;
            }
            QCheckBox#themeToggle::indicator:unchecked {
                background-color: #bbb;
            }
            QCheckBox#themeToggle::indicator::before {
                content: "";
                position: absolute;
                top: 1px; left: 2px;
                width: 18px; height: 18px;
                border-radius: 9px;
                background-color: #ffffff;
            }
            QCheckBox#themeToggle::indicator:checked::before {
                left: 20px;
            }
        """)
        display_left_layout.addWidget(self.themeToggle)
        self.themeToggle.toggled.connect(self.applyTheme)
        display_main_layout.addLayout(display_left_layout, 3)

        # Open File button
        self.open_button = QPushButton("Open File", self)
        self.open_button.clicked.connect(self.open_image)
        self.open_button.setFlat(True)
        display_left_layout.addWidget(self.open_button)
        # Display figure
        self.figure_display, self.ax_display = plt.subplots(figsize=(8, 8))
        self.figure_display.patch.set_facecolor('black')
        self.canvas_display = FigureCanvas(self.figure_display)
        # Create a horizontal layout to hold the canvas and the Z slider
        canvas_slider_layout = QHBoxLayout()
        canvas_slider_layout.addWidget(self.canvas_display)
        # Initialize the vertical Z-plane slider (always present, minimal width)
        self.z_slider_display = QSlider(Qt.Vertical, self)
        self.z_slider_display.setMinimum(0)
        self.z_slider_display.setTickPosition(QSlider.NoTicks)      # minimal ticks (numbers only if shown)
        self.z_slider_display.setFixedWidth(20)                     # keep slider narrow
        self.z_slider_display.valueChanged.connect(self.update_z)   # live update on value change
        canvas_slider_layout.addWidget(self.z_slider_display)
        display_left_layout.addLayout(canvas_slider_layout)
        # Channel buttons layout
        self.channel_buttons_display = []
        self.channel_buttons_layout_display = QHBoxLayout()
        display_left_layout.addLayout(self.channel_buttons_layout_display)
        self.merge_color_channels_button = QPushButton("Merge Channels", self)
        self.merge_color_channels_button.clicked.connect(self.merge_color_channels)
        self.channel_buttons_layout_display.addWidget(self.merge_color_channels_button)
        # Controls: slider + play
        controls_layout = QHBoxLayout()
        self.time_slider_display = QSlider(Qt.Horizontal, self)
        self.time_slider_display.setMinimum(0)
        self.time_slider_display.setMaximum(100)
        self.time_slider_display.setTickPosition(QSlider.TicksBelow)
        self.time_slider_display.setTickInterval(10)
        self.time_slider_display.valueChanged.connect(self.update_frame)
        controls_layout.addWidget(self.time_slider_display)
        self.play_button_display = QPushButton("Play", self)
        self.play_button_display.clicked.connect(self.play_pause)
        controls_layout.addWidget(self.play_button_display)
        display_left_layout.addLayout(controls_layout)
        # Right side
        display_right_layout = QVBoxLayout()
        display_main_layout.addLayout(display_right_layout, 1)
        # Image selection tree
        display_right_layout.addWidget(QLabel("Select Image"))
        self.image_tree = QTreeWidget()
        self.image_tree.setMinimumWidth(200)
        self.image_tree.setMinimumHeight(200)
        self.image_tree.setHeaderHidden(True)
        self.image_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.image_tree.itemClicked.connect(self.on_tree_item_clicked)
        self.image_tree.currentItemChanged.connect(self.on_tree_current_item_changed)
        display_right_layout.addWidget(self.image_tree, 3)
        # Close file button
        self.close_file_button = QPushButton("Close File", self)
        self.close_file_button.clicked.connect(self.close_selected_file)
        display_right_layout.addWidget(self.close_file_button)
        # Visualization controls
        self.control_panel_image_properties(display_right_layout)
        # Group box for image info
        image_info_group = QGroupBox("Image Information")
        image_info_layout = QFormLayout()
        image_info_group.setLayout(image_info_layout)
        # Populate rows
        self.file_label = QLabel("")
        image_info_layout.addRow(QLabel("File Name:"), self.file_label)
        self.frames_label = QLabel("")
        image_info_layout.addRow(QLabel("Frames:"), self.frames_label)
        self.z_scales_label = QLabel("")
        image_info_layout.addRow(QLabel("Z-Slices:"), self.z_scales_label)
        self.y_pixels_label = QLabel("")
        image_info_layout.addRow(QLabel("Y Pixels:"), self.y_pixels_label)
        self.x_pixels_label = QLabel("")
        image_info_layout.addRow(QLabel("X Pixels:"), self.x_pixels_label)
        self.channels_label = QLabel("")
        image_info_layout.addRow(QLabel("Channels:"), self.channels_label)
        self.voxel_yx_size_label = QLabel("")
        image_info_layout.addRow(QLabel("Pixel Size (nm):"), self.voxel_yx_size_label)
        self.voxel_z_nm_label = QLabel("")
        image_info_layout.addRow(QLabel("Voxel Z (nm):"), self.voxel_z_nm_label)
        self.bit_depth_label = QLabel("")
        image_info_layout.addRow(QLabel("Bit Depth:"), self.bit_depth_label)
        self.time_interval_label = QLabel("")
        image_info_layout.addRow(QLabel("Time Interval (s):"), self.time_interval_label)
        self.laser_lines_label = QLabel("")
        image_info_layout.addRow(QLabel("Laser Lines:"), self.laser_lines_label)
        self.intensities_label = QLabel("")
        image_info_layout.addRow(QLabel("Intensities:"), self.intensities_label)
        self.wave_ranges_label = QLabel("")
        image_info_layout.addRow(QLabel("Spectral Ranges:"), self.wave_ranges_label)
        # Wrap in scroll area
        scroll_info = QScrollArea()
        scroll_info.setWidgetResizable(True)
        scroll_info.setWidget(image_info_group)
        scroll_info.setMaximumHeight(200)  # adjust as needed
        display_right_layout.addWidget(scroll_info)
        # Export buttons
        self.export_displayed_image_button = QPushButton("Export Image", self)
        self.export_displayed_image_button.clicked.connect(self.export_displayed_image_as_png)
        self.export_video_button = QPushButton("Export Video", self)
        self.export_video_button.clicked.connect(self.export_displayed_video)
        export_buttons_layout = QHBoxLayout()
        export_buttons_layout.addWidget(self.export_displayed_image_button)
        export_buttons_layout.addWidget(self.export_video_button)
        display_right_layout.addLayout(export_buttons_layout)
        # Time & background checkboxes
        options_layout = QHBoxLayout()
        self.display_time_text_checkbox = QCheckBox("Time")
        self.display_time_text_checkbox.setChecked(False)
        self.display_remove_background_checkbox = QCheckBox("Background")
        self.display_remove_background_checkbox.setChecked(False)
        options_layout.addWidget(self.display_time_text_checkbox)
        options_layout.addWidget(self.display_remove_background_checkbox)
        display_right_layout.addLayout(options_layout)
        display_right_layout.addStretch()

# =============================================================================
# =============================================================================
# SEGMENTATION TAB
# =============================================================================
# =============================================================================

    def manual_segmentation(self):
        """
        Enter manual segmentation mode:
        - Display the current frame (or max‐proj) with filtering and clipping
        - Clear any old manual mask
        - Reset selected points
        - Connect a single click handler
        """
        if self.image_stack is None:
            print("No image loaded")
            return
        ch = self.segmentation_current_channel
        if self.use_max_proj_for_segmentation and self.segmentation_maxproj is not None:
            img = self.segmentation_maxproj[..., ch]
        else:
            fr = self.segmentation_current_frame
            image_channel = self.image_stack[fr, :, :, :, ch]
            img = np.max(image_channel, axis=0)
        # smooth and clip for display
        img_filtered = gaussian_filter(img, sigma=2)
        lo, hi = np.percentile(img_filtered, [0.5, 99.0])
        img_clipped = np.clip(img_filtered, lo, hi)
        # redraw segmentation canvas
        self.figure_segmentation.clear()
        self.ax_segmentation = self.figure_segmentation.add_subplot(111)
        self.ax_segmentation.imshow(img_clipped, cmap='Spectral')
        self.ax_segmentation.axis('off')
        self.figure_segmentation.tight_layout()
        self.canvas_segmentation.draw()
        # clear any previous manual mask
        if hasattr(self, 'manual_segmentation_mask'):
            del self.manual_segmentation_mask
        # enter manual mode
        self.selected_points = []
        self.segmentation_mode = "manual"
        # connect click handler exactly once
        self.cid = self.canvas_segmentation.mpl_connect(
            'button_press_event',
            self.on_click_segmentation)
    def on_click_segmentation(self, event):
        if event.inaxes != self.ax_segmentation:
            return
        if event.xdata is not None and event.ydata is not None:
            self.selected_points.append([int(event.xdata), int(event.ydata)])
            ch = self.segmentation_current_channel
            if self.use_max_proj_for_segmentation:
                max_proj = np.max(self.image_stack, axis=(0, 1))[..., ch]
            else:
                fr = self.segmentation_current_frame
                image_channel = self.image_stack[fr, :, :, :, ch]
                max_proj = np.max(image_channel, axis=0)
            max_proj = gaussian_filter(max_proj, sigma=2)
            max_proj = np.clip(max_proj,
                            np.percentile(max_proj, 0.5),
                            np.percentile(max_proj, 99.))
            self.ax_segmentation.clear()
            self.ax_segmentation.imshow(max_proj, cmap='Spectral')
            self.ax_segmentation.axis('off')
            if len(self.selected_points) > 1:
                polygon = np.array(self.selected_points)
                self.ax_segmentation.plot(polygon[:, 0], polygon[:, 1], 'k-', lw=2)
            self.ax_segmentation.plot(
                [p[0] for p in self.selected_points],
                [p[1] for p in self.selected_points],
                'bo', markersize=6,
            )
            self.canvas_segmentation.draw()


    def finish_segmentation(self):
        """
        Terminate manual segmentation by disconnecting the click callback.
        """
        if hasattr(self, 'selected_points') and self.selected_points:
            fr = self.segmentation_current_frame
            ch = self.segmentation_current_channel
            image_channel = self.image_stack[fr, :, :, :, ch]
            max_proj = np.max(image_channel, axis=0)
            max_proj = gaussian_filter(max_proj, sigma=1)
            max_proj = np.clip(max_proj, np.percentile(max_proj, 0.01), np.percentile(max_proj, 99.95))
            mask = np.zeros(max_proj.shape[:2], dtype=np.uint8)
            polygon = np.array([self.selected_points], dtype=np.int32)
            cv2.fillPoly(mask, polygon, 255)
            self.segmentation_mask = np.array(mask, dtype=np.uint8)
            self.ax_segmentation.clear()
            cmap_imagej = cmap_list_imagej[ch]
            self.ax_segmentation.imshow(max_proj, cmap=cmap_imagej)
            self.ax_segmentation.contour(self.segmentation_mask, levels=[0.5], colors='white', linewidths=1)
            self.ax_segmentation.axis('off')
            self.canvas_segmentation.draw()
            self.photobleaching_calculated = False
        else:
            print("No points selected")
        if hasattr(self, 'cid'):
            try:
                self.canvas_segmentation.mpl_disconnect(self.cid)
            except Exception:
                pass
            del self.cid
        self.selected_points = []

    def next_frame(self):
        if getattr(self, 'total_frames', 0) == 0:
            return
        self.current_frame = (self.current_frame + 1) % self.total_frames
        for slider in (self.time_slider_display, self.time_slider_tracking, getattr(self, 'time_slider_tracking_vis', None)):
            if slider is not None:
                slider.blockSignals(True)
                slider.setValue(self.current_frame)
                slider.blockSignals(False)
        self.plot_image()
        current_tab = self.tabs.currentIndex()
        if current_tab == self.tabs.indexOf(self.tracking_tab):
            self.plot_tracking()
        elif (current_tab == self.tabs.indexOf(self.tracking_visualization_tab)
            and getattr(self, 'has_tracked', False)
            and not self.df_tracking.empty):
            self.display_tracking_visualization()

    def run_cellpose_segmentation(self):
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return
        self.segmentation_mode = "cellpose"
        try:
            cytosol_channel = int(self.cellpose_cytosol_channel_input.text()) if self.cellpose_cytosol_channel_input.text() else None
            nucleus_channel = int(self.cellpose_nucleus_channel_input.text()) if self.cellpose_nucleus_channel_input.text() else None
        except:
            QMessageBox.warning(self, "Invalid Channel Input", "Please enter valid integer channels.")
            return
        if cytosol_channel is None and nucleus_channel is None:
            QMessageBox.warning(self, "No Channels Selected", "Please specify at least a cytosol or nucleus channel.")
            return
        diameter_cytosol = self.cellpose_cytosol_diameter_input.value()
        diameter_nucleus = self.cellpose_nucleus_diameter_input.value()
        segmentation_selection_metric = self.cellpose_selection_metric_input.currentText()
        tested_image = self.image_stack[self.current_frame,:,:,:,:] if self.image_stack.ndim == 5 else self.image_stack
        channels_cytosol = [cytosol_channel] if cytosol_channel is not None else None
        channels_nucleus = [nucleus_channel] if nucleus_channel is not None else None
        try:
            masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = mi.CellSegmentation(
                tested_image,
                channels_cytosol=channels_cytosol,
                channels_nucleus=channels_nucleus,
                diameter_cytosol=diameter_cytosol,
                diameter_nucleus=diameter_nucleus,
                optimization_segmentation_method=None,
                remove_fragmented_cells=False,
                show_plot=False,
                image_name=None,
                NUMBER_OF_CORES=1,
                selection_metric=segmentation_selection_metric
            ).calculate_masks()
            if masks_complete_cells is not None:
                self.segmentation_mask = (masks_complete_cells > 0).astype(np.uint8)
                self.plot_segmentation()
                QMessageBox.information(self, "Cellpose Segmentation", "Cellpose segmentation completed successfully.")
            else:
                QMessageBox.warning(self, "No Masks Found", "Cellpose segmentation returned no masks.")
        except Exception as e:
            QMessageBox.critical(self, "Cellpose Segmentation Failed", str(e))

    def create_segmentation_channel_buttons(self):
        for btn in self.segmentation_channel_buttons:
            btn.setParent(None)
        self.segmentation_channel_buttons = []
        for idx, channel_name in enumerate(self.channel_names):
            btn = QPushButton(f"Channel {idx}", self)
            btn.clicked.connect(partial(self.update_segmentation_channel, idx))
            self.segmentation_channel_buttons_layout.addWidget(btn)
            self.segmentation_channel_buttons.append(btn)

    def update_segmentation_channel(self, channel_index):
        # Clear old mask when changing channel
        self.segmentation_mask = None
        self.segmentation_current_channel = channel_index
        self.plot_segmentation()

    def update_segmentation_frame(self, value):
        # Clear old mask when changing frame
        self.segmentation_mask = None
        self.segmentation_current_frame = value
        self.plot_segmentation()

    def run_watershed_segmentation(self):
        if self.image_stack is not None:
            ch = self.segmentation_current_channel
            if self.use_max_proj_for_segmentation and self.segmentation_maxproj is not None:
                image_to_segment = self.segmentation_maxproj[..., ch]
            else:
                fr = self.segmentation_current_frame
                image_channel = self.image_stack[fr, :, :, :, ch]
                image_to_segment = np.max(image_channel, axis=0)
            # Use default parameter values since GUI inputs are commented out
            footprint_size = 5
            threshold_method = 'li'
            markers_method = 'local'
            separation_size = 5
            threshold_factor = getattr(self, 'watershed_threshold_factor', 1.0)
            watershed_segmentation = mi.CellSegmentationWatershed(
                image=image_to_segment,
                footprint_size=footprint_size,
                threshold_method=threshold_method,
                markers_method=markers_method,
                separation_size=separation_size,
                threshold_factor=threshold_factor
            )
            segmentation_mask = watershed_segmentation.apply_watershed()
            self.segmentation_mask = segmentation_mask
            self.plot_segmentation()
            self.segmentation_mode = "watershed"
        else:
            print("No image loaded")

    def update_watershed_threshold_factor(self, value):
        # Convert slider value (int) to float factor (value/100)
        self.watershed_threshold_factor = value / 100.0
        if self.image_stack is not None:
            self.run_watershed_segmentation()

    def update_segmentation_source(self, state):
        if state == Qt.Checked:
            self.compute_max_proj_segmentation()
            self.use_max_proj_for_segmentation = True
            self.segmentation_time_slider.setEnabled(False)
            self.max_proj_status_label.setText("Max projection is ON")
        else:
            self.use_max_proj_for_segmentation = False
            self.segmentation_time_slider.setEnabled(True)
            self.max_proj_status_label.setText("Max projection is OFF")
            self.plot_segmentation()

    def compute_max_proj_segmentation(self):
        if self.image_stack is None:
            return
        self.segmentation_maxproj = np.max(self.image_stack, axis=(0, 1))
        self.plot_segmentation()

    def plot_segmentation(self):
        self.figure_segmentation.clear()
        self.ax_segmentation = self.figure_segmentation.add_subplot(111)
        self.ax_segmentation.set_facecolor('black')
        if self.image_stack is not None:
            ch = self.segmentation_current_channel
            # Choose image to display (max projection vs current frame)
            if self.use_max_proj_for_segmentation and self.segmentation_maxproj is not None:
                image_to_display = self.segmentation_maxproj[..., ch]
            else:
                image_channel = self.image_stack[self.segmentation_current_frame, :, :, :, ch]
                image_to_display = np.max(image_channel, axis=0)
            # Get display parameters for channel (fallback to global defaults)
            params = self.channelDisplayParams.get(ch, {
                'min_percentile': self.display_min_percentile,
                'max_percentile': self.display_max_percentile,
                'sigma': self.display_sigma,
                'low_sigma': self.low_display_sigma
            })
            # Convert using per-channel percentiles
            rescaled_image = mi.Utilities().convert_to_int8(
                image_to_display,
                rescale=True,
                min_percentile=params['min_percentile'],
                max_percentile=params['max_percentile']
            )
            if params['low_sigma'] > 0:
                rescaled_image = gaussian_filter(rescaled_image, sigma=params['low_sigma'])
            if params['sigma'] > 0:
                rescaled_image = gaussian_filter(rescaled_image, sigma=params['sigma'])
            rescaled_image = mi.Utilities().convert_to_int8(rescaled_image, rescale=False)
            normalized_image = rescaled_image.astype(np.float32) / 255.0
            cmap_used = cmap_list_imagej[ch]
            self.ax_segmentation.imshow(normalized_image[..., 0], cmap=cmap_used, vmin=0, vmax=1)
            if self.segmentation_mask is not None:
                self.ax_segmentation.contour(self.segmentation_mask, levels=[0.5], colors='white', linewidths=1)
        else:
            self.ax_segmentation.text(
                0.5, 0.5, 'No image loaded.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='white', transform=self.ax_segmentation.transAxes
            )
        self.ax_segmentation.axis('off')
        self.figure_segmentation.tight_layout()
        self.canvas_segmentation.draw()

    def setup_segmentation_tab(self):
        """
        Set up the segmentation tab UI.
        Initializes internal state for segmentation and assembles a two-panel interface
        with controls and display components.
        Left Panel:
            • Matplotlib figure & canvas for segmentation preview
            • Frame navigation slider
            • Channel selection buttons container
            • Navigation toolbar
            • Export buttons for segmentation image and mask
        Right Panel:
            • Maximum projection toggle checkbox and status label
            • Manual segmentation ("Manual Segmentation" / "Finish Segmentation") buttons
            • Watershed segmentation threshold slider and "Run Watershed Segmentation" button
        Attributes Created on self:
            segmentation_current_frame (int)
            segmentation_current_channel (int)
            use_max_proj_for_segmentation (bool)
            segmentation_maxproj (Optional[np.ndarray])
            figure_segmentation (matplotlib.figure.Figure)
            ax_segmentation (matplotlib.axes.Axes)
            canvas_segmentation (FigureCanvas)
            segmentation_time_slider (QSlider)
            segmentation_channel_buttons_layout (QHBoxLayout)
            toolbar_segmentation (NavigationToolbar)
            export_segmentation_image_button (QPushButton)
            export_mask_button (QPushButton)
            use_max_proj_checkbox (QCheckBox)
            max_proj_status_label (QLabel)
            segmentation_button (QPushButton)
            finish_segmentation_button (QPushButton)
            watershed_threshold_slider (QSlider)
            run_watershed_button (QPushButton)
        Connected Signals:
            update_segmentation_frame
            export_segmentation_image
            export_mask_as_tiff
            update_segmentation_source
            manual_segmentation
            finish_segmentation
            update_watershed_threshold_factor
            run_watershed_segmentation
        """

        self.segmentation_current_frame = 0
        self.segmentation_current_channel = 0
        self.use_max_proj_for_segmentation = False
        self.segmentation_maxproj = None
        # Create main horizontal layout for segmentation tab
        main_layout = QHBoxLayout(self.segmentation_tab)
        # LEFT PANEL: Segmentation Figure & Controls
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)
        # Create segmentation figure and canvas
        self.figure_segmentation, self.ax_segmentation = plt.subplots()
        self.figure_segmentation.patch.set_facecolor('black')
        self.canvas_segmentation = FigureCanvas(self.figure_segmentation)
        left_layout.addWidget(self.canvas_segmentation)
        # Create lower controls on left panel: channel buttons, time slider, toolbar, etc.
        left_controls_layout = QVBoxLayout()
        # Top row: channel buttons + time slider
        top_controls_layout = QHBoxLayout()
        self.segmentation_channel_buttons = []
        self.segmentation_channel_buttons_layout = QHBoxLayout()
        top_controls_layout.addLayout(self.segmentation_channel_buttons_layout)
        self.segmentation_time_slider = QSlider(Qt.Horizontal)
        self.segmentation_time_slider.setMinimum(0)
        self.segmentation_time_slider.setTickPosition(QSlider.TicksBelow)
        self.segmentation_time_slider.setTickInterval(10)
        self.segmentation_time_slider.valueChanged.connect(self.update_segmentation_frame)
        top_controls_layout.addWidget(self.segmentation_time_slider)
        left_controls_layout.addLayout(top_controls_layout)
        # Bottom row: Navigation toolbar + export buttons (Segmentation Image and Mask)
        toolbar_export_layout = QHBoxLayout()
        self.toolbar_segmentation = NavigationToolbar(self.canvas_segmentation, self)
        toolbar_export_layout.addWidget(self.toolbar_segmentation)
        # Export Segmentation Image button
        self.export_segmentation_image_button = QPushButton("Export Image", self)
        self.export_segmentation_image_button.clicked.connect(self.export_segmentation_image)
        toolbar_export_layout.addWidget(self.export_segmentation_image_button)
        # Export Mask as TIFF button (added next to segmentation export)
        self.export_mask_button = QPushButton("Export Mask", self)
        self.export_mask_button.clicked.connect(self.export_mask_as_tiff)
        toolbar_export_layout.addWidget(self.export_mask_button)
        left_controls_layout.addLayout(toolbar_export_layout)
        left_layout.addLayout(left_controls_layout)
        # RIGHT PANEL: Segmentation Methods & Source Toggle
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=1)
        # Maximum Projection Group
        max_proj_group = QGroupBox("Maximum Projection")
        max_proj_layout = QVBoxLayout()
        self.use_max_proj_checkbox = QCheckBox("Use Max Projection for Segmentation")
        self.use_max_proj_checkbox.stateChanged.connect(self.update_segmentation_source)
        max_proj_layout.addWidget(self.use_max_proj_checkbox)
        self.max_proj_status_label = QLabel("Max projection is OFF")
        self.max_proj_status_label.setStyleSheet("color: limegreen")
        max_proj_layout.addWidget(self.max_proj_status_label)
        max_proj_group.setLayout(max_proj_layout)
        right_layout.addWidget(max_proj_group)
        # Manual Segmentation Group
        manual_group = QGroupBox("Manual Segmentation")
        manual_layout = QVBoxLayout(manual_group)
        button_layout = QHBoxLayout()
        self.segmentation_button = QPushButton("Manual Segmentation", self)
        self.segmentation_button.clicked.connect(self.manual_segmentation)
        button_layout.addWidget(self.segmentation_button)
        self.finish_segmentation_button = QPushButton("Finish Segmentation", self)
        self.finish_segmentation_button.clicked.connect(self.finish_segmentation)
        button_layout.addWidget(self.finish_segmentation_button)
        manual_layout.addLayout(button_layout)
        right_layout.addWidget(manual_group)
        # Watershed Segmentation Group
        watershed_group = QGroupBox("Watershed Segmentation")
        watershed_layout = QFormLayout(watershed_group)
        watershed_layout.setContentsMargins(10, 10, 10, 10)
        watershed_layout.setSpacing(10)
        # Slider values from 10 (0.10) to 150 (2.00) with default 100 (1.0)
        self.watershed_threshold_slider = QSlider(Qt.Horizontal)
        self.watershed_threshold_slider.setMinimum(10)
        self.watershed_threshold_slider.setMaximum(200)
        self.watershed_threshold_slider.setValue(100)
        self.watershed_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.watershed_threshold_slider.setTickInterval(10)
        self.watershed_threshold_slider.valueChanged.connect(self.update_watershed_threshold_factor)
        watershed_layout.addRow(QLabel("Threshold Factor:"), self.watershed_threshold_slider)
        # Run Watershed button
        self.run_watershed_button = QPushButton("Run Watershed Segmentation", self)
        self.run_watershed_button.clicked.connect(self.run_watershed_segmentation)
        watershed_layout.addRow(self.run_watershed_button)
        right_layout.addWidget(watershed_group)
        right_layout.addStretch()
        self.plot_segmentation()
    
# =============================================================================
# =============================================================================
# PHOTOBLEACHING TAB
# =============================================================================
# =============================================================================

    def compute_photobleaching(self):
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return
        if self.segmentation_mask is None:
            QMessageBox.warning(self, "No Segmentation Mask", "Please perform segmentation first.")
            return
        mode = self.mode_combo.currentText().lower()
        self.photobleaching_mode = mode
        radius = self.radius_spinbox.value()
        
        if self.segmentation_mask is None:
            if mode != 'entire_image': 
                QMessageBox.warning(self, "No Segmentation Mask", 
                                    "Please perform segmentation first.")
                return
            else:
                mask_GUI = None 
        else:
            mask_GUI = self.segmentation_mask.copy().astype(int)
            mask_GUI = np.where(mask_GUI > 0, 1, 0)
            mask_GUI.setflags(write=1)

        photobleaching_obj = mi.Photobleaching(
            image_TZYXC=self.image_stack,
            mask_YX=mask_GUI,
            show_plot=False,
            mode=mode,
            radius=radius,
            time_interval_seconds=self.time_interval_value
        )
        self.corrected_image, self.photobleaching_data = photobleaching_obj.apply_photobleaching_correction()
        self.photobleaching_calculated = True
        self.plot_photobleaching()

    def plot_photobleaching(self):
        self.figure_photobleaching.clear()
        if not self.photobleaching_calculated:
            ax = self.figure_photobleaching.add_subplot(111)
            ax.set_facecolor('black')
            ax.axis('off')
            ax.text(
                0.5, 0.5, 'No photobleaching correction applied.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='white', transform=ax.transAxes
            )
            self.canvas_photobleaching.draw()
            return
        num_channels = self.image_stack.shape[-1]
        fig = self.figure_photobleaching
        axs = fig.subplots(num_channels, 2)  
        if num_channels == 1:
            axs = np.array([axs])
        fig.patch.set_facecolor('black')
        decay_rates = self.photobleaching_data['decay_rates']
        time_array = self.photobleaching_data['time_array']
        mean_intensities = self.photobleaching_data['mean_intensities']
        err_intensities = self.photobleaching_data['err_intensities']
        mean_intensities_corrected = self.photobleaching_data['mean_intensities_corrected']
        err_intensities_corrected = self.photobleaching_data['err_intensities_corrected']
        params = np.array(decay_rates)

        if len(params) != 2 * num_channels:
            QMessageBox.warning(self, "Fit Error",
                f"Expected {2 * num_channels} parameters for exponential fit, got {len(params)}")
            return

        for ch in range(num_channels):
            data = mean_intensities[0:, ch]
            t = time_array[0:]

            if len(data) == 0 or np.max(data) == 0:
                axs[ch, 0].text(0.5, 0.5, "No data", ha='center', va='center', color='white', transform=axs[ch,0].transAxes)
                axs[ch, 1].text(0.5, 0.5, "No data", ha='center', va='center', color='white', transform=axs[ch,1].transAxes)
                continue
            
            # Style axes
            for ax_obj in axs[ch, :]:
                ax_obj.set_facecolor('black')
                ax_obj.tick_params(colors='white', which='both')
                for spine in ax_obj.spines.values():
                    spine.set_color('white')
                ax_obj.xaxis.label.set_color('white')
                ax_obj.yaxis.label.set_color('white')
                ax_obj.title.set_color('white')
                ax_obj.grid(True, which='both', color='gray', linestyle='--', linewidth=0.1)
            
            # Get fitted parameters for this channel: [k_fit, I0_fit]
            k_fit = params[2*ch]
            I0_fit = params[2*ch + 1]
            
            # Left subplot: exponential fit
            fitted_curve = I0_fit * np.exp(-k_fit * t)
            
            axs[ch, 0].plot(t, data, 'o', label='Raw Data', color='cyan', lw=2)
            axs[ch, 0].plot(t, fitted_curve, '-', label=f'I₀={I0_fit:.0f}, k={k_fit:.4f}', color='white', lw=2)
            axs[ch, 0].set_title(f'Channel {ch}: Exponential Fit', fontsize=10)
            axs[ch, 0].set_xlabel('Time (s)')
            axs[ch, 0].set_ylabel('Intensity')
            axs[ch, 0].legend(loc='upper right', bbox_to_anchor=(1, 1))
            
            # Right subplot: original vs corrected
            axs[ch, 1].plot(time_array, mean_intensities[:, ch], label='Original', color='cyan', lw=2)
            axs[ch, 1].fill_between(time_array, 
                                mean_intensities[:, ch] - err_intensities[:, ch], 
                                mean_intensities[:, ch] + err_intensities[:, ch], 
                                alpha=0.2, color='cyan')
            axs[ch, 1].plot(time_array, mean_intensities_corrected[:, ch], label='Corrected', color='orangered', lw=2)
            axs[ch, 1].fill_between(time_array, 
                                mean_intensities_corrected[:, ch] - err_intensities_corrected[:, ch], 
                                mean_intensities_corrected[:, ch] + err_intensities_corrected[:, ch], 
                                alpha=0.2, color='orangered')
            axs[ch, 1].set_title(f'Channel {ch} Correction', fontsize=10)
            axs[ch, 1].set_xlabel('Time (s)')
            axs[ch, 1].set_ylabel('Intensity')
            axs[ch, 1].legend(loc='upper right', bbox_to_anchor=(1, 1))

        fig.tight_layout()
        self.canvas_photobleaching.draw()


    def setup_photobleaching_tab(self):
        """
        Initialize and configure the Photobleaching tab UI.
        This method builds the layout and widgets required for performing
        and visualizing photobleaching analysis. It performs the following steps:
        1. Creates a vertical layout for the photobleaching tab.
        2. Constructs a horizontal controls panel containing:
            - A "Mode" combo box with options: "inside_cell", "outside_cell", "use_circular_region".
            - A "Radius" spin box (1–200, default 30).
            - A "Remove Time Points" spin box (0–200, default 0).
            - A "Model Type" combo box with options: "exponential", "linear", "double_exponential".
            - A "Run Photobleaching" button that triggers self.compute_photobleaching.
        3. Adds a Matplotlib Figure and FigureCanvas for plotting the photobleaching curve.
        4. Adds a navigation toolbar and an "Export Photobleaching Image" button,
            which triggers self._export_photobleaching_image.
        5. Stores all interactive widgets as instance attributes for later access.
        Returns
        -------
        None
        """
        photobleaching_layout = QVBoxLayout(self.photobleaching_tab)
        # Controls at the top
        controls_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["inside_cell", "outside_cell", "use_circular_region", "entire_image"])
        controls_layout.addWidget(mode_label)
        controls_layout.addWidget(self.mode_combo)
        radius_label = QLabel("Radius:")
        self.radius_spinbox = QSpinBox()
        self.radius_spinbox.setMinimum(1)
        self.radius_spinbox.setMaximum(200)
        self.radius_spinbox.setValue(30)
        controls_layout.addWidget(radius_label)
        controls_layout.addWidget(self.radius_spinbox)
        # Photobleaching run button
        self.run_photobleaching_button = QPushButton("Run Photobleaching", self)
        self.run_photobleaching_button.clicked.connect(self.compute_photobleaching)
        controls_layout.addWidget(self.run_photobleaching_button)
        # Add controls layout on top
        photobleaching_layout.addLayout(controls_layout)
        # Main figure for photobleaching
        self.figure_photobleaching = Figure()
        self.canvas_photobleaching = FigureCanvas(self.figure_photobleaching)
        photobleaching_layout.addWidget(self.canvas_photobleaching)
        # Horizontal layout for toolbar + export
        toolbar_and_export_layout = QHBoxLayout()
        # Navigation toolbar
        self.toolbar_photobleaching = NavigationToolbar(self.canvas_photobleaching, self)
        toolbar_and_export_layout.addWidget(self.toolbar_photobleaching)
        # Spacer
        toolbar_and_export_layout.addStretch()
        # Export button
        self.export_photobleaching_button = QPushButton("Export Photobleaching Image", self)
        self.export_photobleaching_button.clicked.connect(self._export_photobleaching_image)
        toolbar_and_export_layout.addWidget(self.export_photobleaching_button)
        photobleaching_layout.addLayout(toolbar_and_export_layout)

# =============================================================================
# =============================================================================
# TRACKING TAB
# =============================================================================
# =============================================================================

    def scale_spots(self):
        """
        Determine the scale for displaying spots based on the platform.
        This method sets the SCALE_SPOTS class variable to different values
        depending on whether the code is running on Windows, macOS, or Linux.
        """
        if sys.platform.startswith('win'):
            SCALE_SPOTS = 6
        elif sys.platform.startswith('darwin'):
            SCALE_SPOTS = 1
        elif sys.platform.startswith('linux'):
            SCALE_SPOTS = 1
        else:
            SCALE_SPOTS = 1
        return SCALE_SPOTS
    
    def track_particles(self, corrected_image, mask, parameters, use_maximum_projection):
        """
        Run particle tracking on `corrected_image` + `mask` with the given
        parameters dict and voxel size. Pops up a warning on subnet‐oversize
        and returns an empty list in that case.
        Returns: list of trajectory DataFrames.
        """
        channels_spots      = parameters['channels_spots']
        channels_cytosol    = parameters['channels_cytosol']
        channels_nucleus    = parameters['channels_nucleus']
        min_length_trajectory                = parameters['min_length_trajectory']
        threshold_for_spot_detection         = parameters['threshold_for_spot_detection']
        yx_spot_size_in_px                   = parameters['yx_spot_size_in_px']
        z_spot_size_in_px                    = parameters['z_spot_size_in_px']
        cluster_radius_nm                    = parameters['cluster_radius_nm']
        maximum_spots_cluster                = parameters['maximum_spots_cluster']
        separate_clusters_and_spots          = parameters['separate_clusters_and_spots']
        maximum_range_search_pixels          = parameters['maximum_range_search_pixels']
        use_fixed_size_for_intensity_calculation = parameters['use_fixed_size_for_intensity_calculation']
        link_using_3d_coordinates            = parameters['link_using_3d_coordinates']
        memory           = parameters['memory']
        list_voxels      = parameters['list_voxels']

        try:
            df_list, _ = mi.ParticleTracking(
                image=corrected_image,
                channels_spots=channels_spots,
                masks=mask,
                list_voxels=list_voxels,
                memory=memory,
                channels_cytosol=channels_cytosol,
                channels_nucleus=channels_nucleus,
                min_length_trajectory=min_length_trajectory,
                threshold_for_spot_detection=threshold_for_spot_detection,
                yx_spot_size_in_px=yx_spot_size_in_px,
                z_spot_size_in_px=z_spot_size_in_px,
                cluster_radius_nm=cluster_radius_nm,
                maximum_spots_cluster=maximum_spots_cluster,
                separate_clusters_and_spots=separate_clusters_and_spots,
                maximum_range_search_pixels=maximum_range_search_pixels,
                use_maximum_projection=use_maximum_projection,
                use_fixed_size_for_intensity_calculation=use_fixed_size_for_intensity_calculation,
                link_using_3d_coordinates=link_using_3d_coordinates,
                step_size_in_sec=float(self.time_interval_value),
            ).run()
        except SubnetOversizeException as e:
            QMessageBox.warning(
                None,
                "Tracking Warning",
                f"Tracking not possible due to oversize subnet:\n\n{e}\n\n"
                "Please select fewer particles or adjust the tracking parameters."
            )
            return []

        return df_list
    

    def get_current_image_source(self):
        return self.corrected_image if self.image_source_combo.currentText() == "Photobleaching Corrected" and self.corrected_image is not None else self.image_stack

    def show_tracking_error(self, error_message):
        QMessageBox.warning(self, "Tracking Error", error_message)

    def on_tracking_max_percentile_changed(self, val):
        self.tracking_max_percentile = float(val)
        self.plot_tracking()
    
    def update_threshold_histogram(self):
        if self.image_stack is None:
            self.ax_threshold_hist.clear()
            self.ax_threshold_hist.set_facecolor('black')
            self.ax_threshold_hist.axis('off')
            self.canvas_threshold_hist.draw_idle()
            return
        image_to_use = self.get_current_image_source()
        image_channel = image_to_use[self.current_frame, :, :, :, self.current_channel]
        mask_GUI = np.where(self.segmentation_mask > 0, 1, 0) if self.segmentation_mask is not None else np.ones(image_channel.shape[1:], dtype=image_channel.dtype)
        # Compute maximum projection (across Z)
        max_proj = np.max(image_channel, axis=0) * mask_GUI
        intensity_values = max_proj.flatten()
        if len(intensity_values) == 0:
            return
        lower_limit = 0
        upper_limit = np.percentile(intensity_values, 99.5)
        self.ax_threshold_hist.clear()
        unique_vals = np.unique(intensity_values)
        desired_bins = 60
        bins_to_use = desired_bins if unique_vals.size >= desired_bins else unique_vals.size
        self.ax_threshold_hist.hist(
            intensity_values,
            bins=bins_to_use,
            range=(lower_limit, upper_limit),
            color='aliceblue',
            edgecolor='black'
        )
        self.ax_threshold_hist.set_xlim(lower_limit, upper_limit)
        self.ax_threshold_hist.set_yticks([])
        self.ax_threshold_hist.grid(False)
        self.ax_threshold_hist.tick_params(axis='both', which='major', labelsize=6)
        slider_min = int(lower_limit)
        slider_max = int(upper_limit * 1.1)
        self.threshold_slider.setMinimum(slider_min)
        self.threshold_slider.setMaximum(slider_max)
        if not hasattr(self, 'user_selected_threshold') or self.user_selected_threshold is None:
            self.threshold_slider.setValue(slider_min)
        else:
            self.ax_threshold_hist.axvline(self.user_selected_threshold, color='orangered', linestyle='-', lw=3)
        self.canvas_threshold_hist.draw()

    def update_threshold_value(self, value):
        if self.image_stack is None:
            self.ax_threshold_hist.clear()
            self.ax_threshold_hist.set_facecolor('black')
            self.ax_threshold_hist.axis('off')
            self.canvas_threshold_hist.draw_idle()
            return
        self.user_selected_threshold = value
        self.threshold_spot_detection = float(value)
        self.ax_threshold_hist.clear()
        image_to_use = self.get_current_image_source()
        image_channel = image_to_use[self.current_frame, :, :, :, self.current_channel]
        mask_GUI = np.where(self.segmentation_mask > 0, 1, 0) if self.segmentation_mask is not None else np.ones(image_channel.shape[1:], dtype=image_channel.dtype)
        max_proj = np.max(image_channel, axis=0) * mask_GUI
        intensity_values = max_proj.flatten()
        intensity_values = intensity_values[intensity_values > 0]
        if len(intensity_values) == 0:
            return
        unique_vals = np.unique(intensity_values)
        desired_bins = 60
        bins_to_use = desired_bins if unique_vals.size >= desired_bins else unique_vals.size
        lower_limit = 0
        upper_limit = np.percentile(intensity_values, 99.5)
        self.ax_threshold_hist.hist(
            intensity_values,
            bins=bins_to_use,
            range=(lower_limit, upper_limit),
            color='aliceblue',
            edgecolor='black'
        )
        self.ax_threshold_hist.set_xlim(lower_limit, upper_limit)
        self.ax_threshold_hist.set_yticks([])
        self.ax_threshold_hist.grid(False)
        self.ax_threshold_hist.tick_params(axis='both', which='major', labelsize=6)
        self.ax_threshold_hist.axvline(self.user_selected_threshold, color='orangered', linestyle='-', lw=3)
        self.canvas_threshold_hist.draw()
        self.detect_spots_in_current_frame()

    def on_image_source_changed(self):
        self.image_source_combo_value = self.image_source_combo.currentText()
        self.plot_tracking()

    def update_threshold_spot_detection(self, value):
        self.threshold_spot_detection = value

    def update_min_length_trajectory(self, value):
        self.min_length_trajectory = value

    def update_yx_spot_size(self, value):
        if value % 2 == 0:
            value += 1
        self.yx_spot_size_in_px = value

    def update_z_spot_size(self, value):
        self.z_spot_size_in_px = value

    def update_cluster_radius(self, value):
        self.cluster_radius_nm = value

    def update_max_spots_cluster(self, value):
        self.maximum_spots_cluster = value if value != 0 else None

    def update_use_maximum_projection(self, state):
        self.use_maximum_projection = (state == Qt.Checked)
        if hasattr(self, 'tracking_max_proj_status_label'):
            self.tracking_max_proj_status_label.setText("2D Projection is ON" if self.use_maximum_projection else "2D Projection is OFF")

    def update_max_range_search_pixels(self, value):
        self.maximum_range_search_pixels = value

    def update_memory(self, value):
        self.memory = value

    def update_tracking_sliders(self):
        """
        Sync the Tracking-tab intensity controls to the current channel's display parameters.
        """
        params = self.channelDisplayParams.get(self.current_channel, {
            'min_percentile': self.display_min_percentile,
            'max_percentile': self.display_max_percentile
        })
        # Update tracking sliders/spinboxes without triggering signals
        if hasattr(self, 'min_percentile_slider_tracking'):
            self.min_percentile_slider_tracking.blockSignals(True)
            self.max_percentile_slider_tracking.blockSignals(True)
            self.min_percentile_slider_tracking.setValue(int(params['min_percentile']))
            self.max_percentile_slider_tracking.setValue(int(params['max_percentile']))
            self.min_percentile_slider_tracking.blockSignals(False)
            self.max_percentile_slider_tracking.blockSignals(False)
        if hasattr(self, 'min_percentile_spinbox_tracking'):
            self.min_percentile_spinbox_tracking.blockSignals(True)
            self.max_percentile_spinbox_tracking.blockSignals(True)
            self.min_percentile_spinbox_tracking.setValue(params['min_percentile'])
            self.max_percentile_spinbox_tracking.setValue(params['max_percentile'])
            self.min_percentile_spinbox_tracking.blockSignals(False)
            self.max_percentile_spinbox_tracking.blockSignals(False)
        # Store the updated values for tracking
        self.tracking_min_percentile = params['min_percentile']
        self.tracking_max_percentile = params['max_percentile']

    def generate_random_spots(self, state):
        self.random_mode_enabled = (state == Qt.Checked)
        num_points = self.random_points_input.value()
        if self.random_mode_enabled:
            print(f"Random spots generation enabled with {num_points} spots.")
        else:
            print("Random spots generation disabled.")

    def detect_spots_all_frames(self):
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return
        # Show progress dialog
        progress = QProgressDialog("Performing spot detection ...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Spot Detection")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()
        # Determine DPI-based width for progress bar
        screen = QGuiApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch()
        pixels = int(2 * dpi)  # 2 inches
        progress.setStyleSheet(f"QProgressBar {{ min-width: {pixels}px; min-height: 20px; }}")
        # Choose image source
        image_to_use = self.get_current_image_source()
        # Compute threshold (user-selected or 99th percentile)
        threshold_value = self.user_selected_threshold if getattr(self, 'user_selected_threshold', None) is not None else np.percentile(image_to_use[:, :, :, :, self.current_channel].ravel(), 99)
        # Prepare mask
        mask = (self.segmentation_mask > 0).astype(int) if self.segmentation_mask is not None else np.ones(self.image_stack.shape[2:4], dtype=int)
        self.tracking_channel = self.current_channel
        # Run spot detection (no linking)
        list_dataframes_trajectories, _ = mi.ParticleTracking(
            image=image_to_use,
            channels_spots=[self.current_channel],
            masks=mask,
            list_voxels=[self.voxel_z_nm, self.voxel_yx_nm],
            memory=self.memory,
            channels_cytosol=self.channels_cytosol,
            channels_nucleus=self.channels_nucleus,
            min_length_trajectory=self.min_length_trajectory,
            threshold_for_spot_detection=threshold_value,
            yx_spot_size_in_px=self.yx_spot_size_in_px,
            z_spot_size_in_px=self.z_spot_size_in_px,
            cluster_radius_nm=self.cluster_radius_nm,
            maximum_spots_cluster=self.maximum_spots_cluster,
            separate_clusters_and_spots=self.separate_clusters_and_spots,
            maximum_range_search_pixels=self.maximum_range_search_pixels,
            use_maximum_projection=self.use_maximum_projection,
            use_fixed_size_for_intensity_calculation=self.use_fixed_size_for_intensity_calculation,
            link_particles=False,
            step_size_in_sec=float(self.time_interval_value),
        ).run()
        progress.close()
        # Store tracking results
        if list_dataframes_trajectories:
            self.df_tracking = pd.concat(list_dataframes_trajectories, ignore_index=True)
        else:
            self.df_tracking = pd.DataFrame()
            QMessageBox.information(self, "No Spots Detected", "No spots were detected in any frame.")
        # Optional random-mode run
        if getattr(self, 'random_mode_enabled', True):
            random_tracking = mi.ParticleTracking(
                image=image_to_use,
                channels_spots=[self.current_channel],
                masks=mask,
                list_voxels=[self.voxel_z_nm, self.voxel_yx_nm],
                memory=self.memory,
                channels_cytosol=self.channels_cytosol,
                channels_nucleus=self.channels_nucleus,
                min_length_trajectory=self.min_length_trajectory,
                threshold_for_spot_detection=threshold_value,
                yx_spot_size_in_px=self.yx_spot_size_in_px,
                z_spot_size_in_px=self.z_spot_size_in_px,
                cluster_radius_nm=self.cluster_radius_nm,
                maximum_spots_cluster=self.maximum_spots_cluster,
                separate_clusters_and_spots=self.separate_clusters_and_spots,
                maximum_range_search_pixels=self.maximum_range_search_pixels,
                use_maximum_projection=self.use_maximum_projection,
                use_fixed_size_for_intensity_calculation=self.use_fixed_size_for_intensity_calculation,
                link_particles=False,
                generate_random_particles=True,
                number_of_random_particles_trajectories=self.random_points_input.value(),
                step_size_in_sec=float(self.time_interval_value),
            )
            rand_list, _ = random_tracking.run()
            self.df_random_spots = rand_list[0] if rand_list else pd.DataFrame()
        # Refresh relevant UI after detection
        self.plot_tracking()
        self.populate_colocalization_channels()
        self.manual_current_image_name = None

    def select_tracking_vis_channel(self, channel_idx):
        """Handle channel button click in Tracking Visualization tab (single-channel mode)."""
        self.tracking_vis_merged = False
        nch = getattr(self, 'number_color_channels', 1) or 1
        self.tracking_vis_channels = [False] * nch
        if 0 <= channel_idx < len(self.tracking_vis_channels):
            self.tracking_vis_channels[channel_idx] = True
        self.display_tracking_visualization(selected_channelIndex=channel_idx)

    def merge_tracking_visualization(self):
        """Handle Merge Channels button in Tracking Visualization tab."""
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        self.tracking_vis_merged = True
        self.display_tracking_visualization()
    
    def on_particle_selected(self, current, previous):
        """Respond when a tracked particle is selected from the list."""
        if current is None:
            return
        particle_id = current.data(Qt.UserRole)
        if particle_id is None:
            return
        self.selected_particle_id = int(particle_id)
        if getattr(self, 'playing', False):
            self.play_pause()
        if not self.df_tracking.empty:
            frames = self.df_tracking[self.df_tracking['particle'] == self.selected_particle_id]['frame']
            if not frames.empty:
                first_frame = int(frames.min())
                self.update_frame(first_frame)
                return
        self.display_tracking_visualization()

    def on_tracking_merge_toggled(self, checked):
        self.tracking_vis_merged = checked
        self.display_tracking_visualization()

    def on_tracking_channel_selected(self, channel_index, checked):
        if not checked:
            return
        self.tracking_vis_merged = False
        self.current_channel = channel_index
        self.display_tracking_visualization()

    def on_intensity_changed(self, value):
        self.display_tracking_visualization()


    
    def format_time(self, seconds):
        """Convert time in seconds to 'M min S s' or 'S s' string format."""
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes} min {remaining_seconds} s" if minutes > 0 else f"{remaining_seconds} s"

    def plot_tracking(self):
        self.figure_tracking.clear()
        self.ax_tracking = self.figure_tracking.add_subplot(111)
        self.ax_tracking.set_facecolor('black')
        self.ax_tracking.axis('off')
        SCALE_SPOTS = self.scale_spots()
        image_to_use = self.get_current_image_source()
        if image_to_use is None:
            self.canvas_tracking.draw_idle()
            return
        ch = self.current_channel
        params = {
            'min_percentile': self.display_min_percentile,
            'max_percentile': self.display_max_percentile,
            'sigma': self.display_sigma,
            'low_sigma': self.low_display_sigma
        }
        image_channel = image_to_use[self.current_frame, :, :, :, ch]
        max_proj = np.max(image_channel, axis=0)
        if self.tracking_remove_background_checkbox.isChecked():
            mask = np.where(self.segmentation_mask > 0, 1, 0) if self.segmentation_mask is not None else np.ones(self.image_stack.shape[2:4], dtype=int)
            max_proj = max_proj * mask
        min_p = self.min_percentile_spinbox_tracking.value() if hasattr(self, 'min_percentile_spinbox_tracking') else self.tracking_min_percentile
        max_p = self.max_percentile_spinbox_tracking.value() if hasattr(self, 'max_percentile_spinbox_tracking') else 99.95
        rescaled_image = mi.Utilities().convert_to_int8(
            max_proj,
            rescale=True,
            min_percentile=min_p,
            max_percentile=max_p
        )
        if params['low_sigma'] > 0:
            rescaled_image = gaussian_filter(rescaled_image, sigma=params['low_sigma'])
        if params['sigma'] > 0:
            rescaled_image = gaussian_filter(rescaled_image, sigma=params['sigma'])
        rescaled_image = mi.Utilities().convert_to_int8(rescaled_image, rescale=False)
        normalized_image = rescaled_image.astype(np.float32) / 255.0
        normalized_image = normalized_image[..., 0]
        cmap_imagej = cmap_list_imagej[ch]
        self.ax_tracking.imshow(normalized_image, cmap=cmap_imagej, vmin=0, vmax=1)
        dpi = self.figure_tracking.get_dpi()
        marker_scale = dpi / 100.0
        df_frame = self.df_tracking[self.df_tracking['frame'] == self.current_frame] if not self.df_tracking.empty else (self.detected_spots_frame if hasattr(self, 'detected_spots_frame') and self.detected_spots_frame is not None and not self.detected_spots_frame.empty and self.detected_spots_frame['frame'].iloc[0] == self.current_frame else pd.DataFrame())
        if not df_frame.empty:
            edge_color = "w"
            single_spots = df_frame[df_frame['cluster_size'] <= 1]
            cluster_spots = df_frame[df_frame['cluster_size'] > 1]
            legend_handles = []
            legend_labels = []
            if not single_spots.empty:
                self.ax_tracking.scatter(
                    single_spots['x'], single_spots['y'],
                    s=self.yx_spot_size_in_px * 6 * marker_scale * SCALE_SPOTS,
                    marker='o', linewidth=1,
                    edgecolors=edge_color, facecolors='none'
                )
                count_spots = single_spots.shape[0]
                spot_legend = self.ax_tracking.scatter([], [],
                                                       s=self.yx_spot_size_in_px * 5 * marker_scale,
                                                       marker='o', linewidth=1,
                                                       edgecolors=edge_color, facecolors='none')
                legend_handles.append(spot_legend)
                legend_labels.append(f"Spots: {count_spots}")
            else:
                self.ax_tracking.scatter(
                    [], [],
                    s=self.yx_spot_size_in_px * 6 * marker_scale * SCALE_SPOTS,
                    marker='o', linewidth=1,
                    edgecolors=edge_color, facecolors='none'
                )
                legend_labels.append(f"Spots: 0")
                legend_handles.append(self.ax_tracking.scatter([], [],
                                                               s=self.yx_spot_size_in_px * 5 * marker_scale,
                                                               marker='o', linewidth=1,
                                                               edgecolors=edge_color, facecolors='none'))
            if not cluster_spots.empty:
                self.ax_tracking.scatter(
                    cluster_spots['x'], cluster_spots['y'],
                    s=self.yx_spot_size_in_px * 6 * marker_scale * SCALE_SPOTS,
                    marker='s', linewidth=1,
                    edgecolors=edge_color, facecolors='none'
                )
                count_clusters = cluster_spots.shape[0]
                cluster_legend = self.ax_tracking.scatter([], [],
                                                          s=self.yx_spot_size_in_px * 5 * marker_scale * SCALE_SPOTS,
                                                          marker='s', linewidth=1,
                                                          edgecolors=edge_color, facecolors='none')
                legend_handles.append(cluster_legend)
                legend_labels.append(f"Clusters: {count_clusters}")
            if self.show_cluster_size_checkbox.isChecked():
                for _, row in df_frame.iterrows():
                    self.ax_tracking.text(row['x']+8, row['y'],
                                           f"{int(row['cluster_size'])}",
                                           color='white', fontsize=8,
                                           ha='center', va='center')
            if self.show_particle_id_checkbox.isChecked() and 'particle' in df_frame.columns:
                for _, row in df_frame.iterrows():
                    self.ax_tracking.text(row['x'], row['y'] - 8,
                                           f"{int(row['particle'])}",
                                           color='white', fontsize=6,
                                           ha='center', va='center')
            if self.show_trajectories_checkbox.isChecked() and not self.df_tracking.empty and 'particle' in self.df_tracking.columns:
                df_up_to_current = self.df_tracking[self.df_tracking['frame'] <= self.current_frame]
                for particle_id, grp in df_up_to_current.groupby('particle'):
                    if grp.shape[0] > 1:
                        grp = grp.sort_values('frame')
                        self.ax_tracking.plot(grp['x'], grp['y'], '-', linewidth=1, color='white', alpha=0.5)
            if legend_handles:
                legend = self.ax_tracking.legend(legend_handles, legend_labels,
                                                 loc='upper right', bbox_to_anchor=(1, 1))
                for text in legend.get_texts():
                    text.set_color("w")
        if self.segmentation_mask is not None:
            self.ax_tracking.contour(self.segmentation_mask, levels=[0.5], colors='white', linewidths=1)
        if self.tracking_time_text_checkbox.isChecked():
            current_time = self.current_frame * (float(self.time_interval_value) if self.time_interval_value else 1)
            time_str = f"{int(current_time)} s" if current_time <= 300 else self.format_time(current_time)
            self.ax_tracking.text(0.05, 0.99, time_str,
                                   transform=self.ax_tracking.transAxes,
                                   verticalalignment='top',
                                   color='white',
                                   fontsize=12,
                                   bbox=dict(facecolor='black', alpha=0.5, pad=2))
        self.ax_tracking.axis('off')
        # show scale bar if voxel size is available
        if hasattr(self, 'voxel_yx_nm') and self.voxel_yx_nm is not None:
                font_props = {'size': 10}
                if getattr(self, 'voxel_yx_nm', None) is not None:
                    microns_per_pixel = self.voxel_yx_nm / 1000.0
                    scalebar = ScaleBar(
                        microns_per_pixel, units='um', length_fraction=0.2,
                        location='lower right', box_color='black', color='white',
                        font_properties=font_props
                    )
                    self.ax_tracking.add_artist(scalebar)
        self.figure_tracking.tight_layout()
        self.canvas_tracking.draw_idle()


    def detect_spots_in_current_frame(self):
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return
        image_to_use = self.get_current_image_source()
        image_channel = np.expand_dims(image_to_use[self.current_frame, :, :, :, self.current_channel], axis=3)
        if self.voxel_z_nm == 0:
            self.voxel_z_nm = 0.1 
        list_voxels = [self.voxel_z_nm, self.voxel_yx_nm]
        threshold = self.user_selected_threshold if hasattr(self, 'user_selected_threshold') and self.user_selected_threshold is not None else np.percentile(image_channel, 99)
        mask = (self.segmentation_mask > 0).astype(int) if self.segmentation_mask is not None else np.ones(self.image_stack.shape[2:4], dtype=int)
        spots = self.detect_spots(image_channel, threshold, list_voxels, mask)
        if spots is not None and not spots.empty:
            spots['frame'] = self.current_frame
            self.detected_spots_frame = spots
            self.df_tracking = spots.copy()
        else:
            self.detected_spots_frame = None
            self.df_tracking = pd.DataFrame()
        self.plot_tracking()

    def detect_spots(self, image, threshold, list_voxels, mask):
        z_sp_sz = self.z_spot_size_in_px if self.z_spot_size_in_px is not None else 1
        yx_sp_sz = self.yx_spot_size_in_px if self.yx_spot_size_in_px is not None else 5
        dataframe = mi.SpotDetection(
                image,
                channels_spots=0,
                channels_cytosol=self.channels_cytosol,
                channels_nucleus=self.channels_nucleus,
                masks_complete_cells=mask,
                list_voxels=list_voxels,
                yx_spot_size_in_px=yx_sp_sz,
                z_spot_size_in_px=z_sp_sz,
                cluster_radius_nm=self.cluster_radius_nm,
                show_plot=False,
                save_files=False,
                threshold_for_spot_detection=threshold,
                use_maximum_projection=self.use_maximum_projection,
                calculate_intensity=False,
            ).get_dataframe()[0]
        return dataframe


    def perform_particle_tracking(self):
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return
        if not hasattr(self, 'user_selected_threshold') or self.user_selected_threshold <= 0:
            QMessageBox.warning(self, "Tracking Aborted", "Threshold is zero; please adjust the threshold slider before running tracking.")
            return
        self.df_tracking = pd.DataFrame()
        self.detected_spots_frame = None
        self.plot_tracking()
        mask = (self.segmentation_mask > 0).astype(int) if self.segmentation_mask is not None else np.ones(self.image_stack.shape[2:4], dtype=int)
        image_to_use = self.get_current_image_source()
        if self.use_maximum_projection:
            image_to_use = np.max(image_to_use, axis=1, keepdims=True)
        if self.voxel_z_nm == 0:
            self.voxel_z_nm = 0.1 
        list_voxels = [self.voxel_z_nm, self.voxel_yx_nm]
        channels_spots = [self.current_channel]
        starting_threshold = self.user_selected_threshold if hasattr(self, 'user_selected_threshold') and self.user_selected_threshold is not None else mi.Utilities().calculate_threshold_for_spot_detection(
            image_to_use,
            [self.z_spot_size_in_px, self.yx_spot_size_in_px],
            list_voxels,
            [self.current_channel],
            max_spots_for_threshold=self.max_spots_for_threshold,
            show_plot=False,
            plot_name=None
        )
        progress = QProgressDialog("Performing particle tracking ...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Tracking in Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        screen = QGuiApplication.primaryScreen()
        progress.show()
        QApplication.processEvents()
        self.tracking_button.setText("Tracking in progress...")
        self.tracking_button.setEnabled(False)
        parameters = {
            'channels_spots': channels_spots,
            'channels_cytosol': self.channels_cytosol,
            'channels_nucleus': self.channels_nucleus,
            'min_length_trajectory': self.min_length_trajectory,
            'threshold_for_spot_detection': starting_threshold,
            'yx_spot_size_in_px': self.yx_spot_size_in_px,
            'z_spot_size_in_px': self.z_spot_size_in_px,
            'cluster_radius_nm': self.cluster_radius_nm,
            'maximum_spots_cluster': self.maximum_spots_cluster,
            'separate_clusters_and_spots': self.separate_clusters_and_spots,
            'maximum_range_search_pixels': self.maximum_range_search_pixels,
            'memory': self.memory,
            'list_voxels': list_voxels,
            'use_fixed_size_for_intensity_calculation': self.use_fixed_size_for_intensity_calculation,
            'link_using_3d_coordinates': self.link_using_3d_coordinates,
        }
        try:
            results = self.track_particles(image_to_use, mask, parameters, self.use_maximum_projection)
            self.on_tracking_finished_with_progress(results, progress)
            #return
        except Exception as e:
            QMessageBox.critical(self, "Tracking Error", f"An error occurred while starting tracking:\n{str(e)}")
            self.tracking_button.setText(" Tracking")
            self.tracking_button.setEnabled(True)
            progress.close()
        if hasattr(self, 'random_mode_enabled') and self.random_mode_enabled:
            random_tracking = mi.ParticleTracking(
                image=image_to_use,
                channels_spots=[self.current_channel],
                masks=mask,
                list_voxels=list_voxels,
                memory=self.memory,
                channels_cytosol=self.channels_cytosol,
                channels_nucleus=self.channels_nucleus,
                min_length_trajectory=self.min_length_trajectory,
                threshold_for_spot_detection=starting_threshold,
                yx_spot_size_in_px=self.yx_spot_size_in_px,
                z_spot_size_in_px=self.z_spot_size_in_px,
                cluster_radius_nm=self.cluster_radius_nm,
                maximum_spots_cluster=self.maximum_spots_cluster,
                separate_clusters_and_spots=self.separate_clusters_and_spots,
                maximum_range_search_pixels=self.maximum_range_search_pixels,
                use_maximum_projection=self.use_maximum_projection,
                use_fixed_size_for_intensity_calculation=self.use_fixed_size_for_intensity_calculation,
                link_particles=True,
                generate_random_particles=True,
                number_of_random_particles_trajectories=self.random_points_input.value(),
                step_size_in_sec=self(self.time_interval_value),
            )
            random_df_list, _ = random_tracking.run()
            self.df_random_spots = random_df_list[0] if random_df_list else pd.DataFrame()

    def on_tracking_finished_with_progress(self, list_dataframes_trajectories, progress_dialog):
        self.on_tracking_finished(list_dataframes_trajectories)
        progress_dialog.close()

    def on_tracking_finished(self, list_dataframes_trajectories):
        try:
            if list_dataframes_trajectories and any(not df.empty for df in list_dataframes_trajectories):
                df_tracking = pd.concat(list_dataframes_trajectories, ignore_index=True)
                if 'particle' not in df_tracking.columns or df_tracking['particle'].nunique() == 0:
                    raise ValueError("No particles detected or 'particle' column missing.")
                self.df_tracking = df_tracking.reset_index(drop=True)
                self.has_tracked = True
            else:
                raise ValueError("No particles detected.")
            self.correlation_results = []
            self.current_total_plots = None
            self.detected_spots_frame = None
            self.plot_intensity_time_course()
            self.display_correlation_plot()
            self.channels_spots = [self.current_channel]
            self.populate_colocalization_channels()
            self.manual_current_image_name = None
            self.manual_scroll_area.setWidget(QWidget())
            self.manual_stats_label.setText("Total Spots: 0 | Colocalized: 0 | 0.00%")
            self.MIN_FRAMES_MSD = 20
            self.MIN_PARTICLES_MSD = 10

            if hasattr(self, 'compute_colocalization'):
                self.compute_colocalization()
            self.plot_tracking()
            if hasattr(self, 'channel_checkboxes') and self.channel_checkboxes:
                for idx, cb in enumerate(self.channel_checkboxes):
                    cb.setChecked(idx == 0)
            if (not self.df_tracking.empty) and self.has_tracked: 
                traj_counts = self.df_tracking.groupby('particle')['frame'].nunique()
                if ('particle' in self.df_tracking.columns
                    and traj_counts.min() >= self.MIN_FRAMES_MSD
                    and traj_counts.size >= self.MIN_PARTICLES_MSD):
                    pm = mi.ParticleMotion(
                        self.df_tracking,
                        microns_per_pixel=self.voxel_yx_nm / 1000.0,    # convert nm to microns
                        step_size_in_sec=float(self.time_interval_value),      # time interval between frames (seconds)
                        show_plot=False, 
                        remove_drift=False
                    )
                    D_um2_s, D_px2_s, _, _, _, _ = pm.calculate_msd()
                    self.msd_label.setText(f"Mean Square Displacement: {D_um2_s:.4f} μm²/s " + f" | {D_px2_s:.4f} px²/s")
                else:
                    self.msd_label.setText("Mean Square Displacement: Not enough data")
                    print("Not enough data for MSD calculation: "
                          f"min frames {self.MIN_FRAMES_MSD}, min particles {self.MIN_PARTICLES_MSD}")
            else:
                self.msd_label.setText("Mean Square Displacement: Not enough data")
                print("No tracking data available for MSD calculation.")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Tracking Failed",
                f"Tracking failed or no particles were detected:\n{str(e)}"
            )
            self.df_tracking = pd.DataFrame()
            self.detected_spots_frame = None
            self.plot_tracking()
        finally:
            self.tracking_button.setText(" Tracking")
            self.tracking_button.setEnabled(True)


    def setup_tracking_tab(self):
        """
        Set up the “Tracking” tab of the application GUI.

        This method builds a two-panel layout for particle tracking:
        - Left panel:
            • Matplotlib FigureCanvas for live tracking display (black background).  
            • Intensity percentile controls (min 0–50%, max 90–100%) with spinboxes that update the display.  
            • Channel selection buttons (dynamically generated).  
            • Time slider with play/pause button for frame navigation.  
            • Export buttons for tracking DataFrame, static image, and video.  
            • Display options checkboxes for trajectories, cluster size, particle IDs, timestamp, and background removal.  
        - Right panel (scrollable):
            • “Tracking Parameters” header.  
            • 2D projection toggle with status label.  
            • Source selection combo (Original vs. Photobleaching Corrected) with styled text.  
            • Threshold histogram canvas and slider for interactive thresholding.  
            • Spot detection & tracking action buttons: “Single Frame,” “All Frames,” and “Tracking.” 
                - "Single Frame" processes the current frame only.
                - "All Frames" processes all frames in the stack but does not link trajectories.
                - "Tracking" links trajectories across frames. 
            • Spot detection parameters form:
                – Minimum trajectory length  
                – YX and Z spot size  
                – Cluster radius (nm)  
                – Maximum cluster size  
            • Linking parameters form:
                – Maximum search range (px)  
                – Memory frames  
            • Random‐spot control group with checkbox and spinbox to generate control spots.  

        All widgets are linked to their respective signal handlers to update internal state and refresh the plot.
        """

        self.tracking_min_percentile = self.display_min_percentile
        self.tracking_max_percentile = self.display_max_percentile
        tracking_main_layout = QHBoxLayout(self.tracking_tab)
        # Left side: image display, time slider, play button, export buttons, etc.
        tracking_left_layout = QVBoxLayout()
        tracking_main_layout.addLayout(tracking_left_layout)
        # Right side: scroll area for tracking parameters
        tracking_right_layout = QVBoxLayout()
        tracking_main_layout.addLayout(tracking_right_layout)
        # Left side: Tracking Figure and Canvas
        self.figure_tracking, self.ax_tracking = plt.subplots(figsize=(8, 8))
        self.figure_tracking.patch.set_facecolor('black')
        self.canvas_tracking = FigureCanvas(self.figure_tracking)
        tracking_left_layout.addWidget(self.canvas_tracking)
        # Intensity percentile controls (spinboxes) for Tracking tab
        spin_layout = QHBoxLayout()
        # Min percentile spinbox (0–50%)
        self.min_percentile_spinbox_tracking = QDoubleSpinBox(self)
        self.min_percentile_spinbox_tracking.setRange(0.0, 50.0)
        self.min_percentile_spinbox_tracking.setSingleStep(0.1)
        self.min_percentile_spinbox_tracking.setSuffix("%")
        self.min_percentile_spinbox_tracking.setValue(self.tracking_min_percentile)
        self.min_percentile_spinbox_tracking.valueChanged.connect(
            lambda v: (setattr(self, 'tracking_min_percentile', float(v)), self.plot_tracking())
        )
        spin_layout.addWidget(QLabel("Min Int", self))
        spin_layout.addWidget(self.min_percentile_spinbox_tracking)
        # Max percentile spinbox (90–100%)
        self.max_percentile_spinbox_tracking = QDoubleSpinBox(self)
        self.max_percentile_spinbox_tracking.setRange(90.0, 100.0)
        self.max_percentile_spinbox_tracking.setSingleStep(0.05)
        self.max_percentile_spinbox_tracking.setSuffix("%")
        self.max_percentile_spinbox_tracking.setValue(self.tracking_max_percentile)
        self.max_percentile_spinbox_tracking.valueChanged.connect(
            lambda v: (setattr(self, 'tracking_max_percentile', float(v)), self.plot_tracking())
        )
        spin_layout.addWidget(QLabel("Max Int", self))
        spin_layout.addWidget(self.max_percentile_spinbox_tracking)
        tracking_left_layout.addLayout(spin_layout)
        # Channel buttons horizontally
        self.channel_buttons_tracking = []
        self.channel_buttons_layout_tracking = QHBoxLayout()
        tracking_left_layout.addLayout(self.channel_buttons_layout_tracking)
        # Time slider + play button
        controls_layout = QHBoxLayout()
        tracking_left_layout.addLayout(controls_layout)
        self.time_slider_tracking = QSlider(self)
        self.time_slider_tracking.setOrientation(Qt.Horizontal)
        self.time_slider_tracking.setMinimum(0)
        self.time_slider_tracking.setMaximum(100)
        self.time_slider_tracking.setTickPosition(QSlider.TicksBelow)
        self.time_slider_tracking.setTickInterval(10)
        self.time_slider_tracking.valueChanged.connect(self.update_frame)
        controls_layout.addWidget(self.time_slider_tracking)
        self.play_button_tracking = QPushButton("Play", self)
        self.play_button_tracking.clicked.connect(self.play_pause)
        controls_layout.addWidget(self.play_button_tracking)
        # Export buttons
        export_buttons_layout = QHBoxLayout()
        tracking_left_layout.addLayout(export_buttons_layout)
        self.export_data_button = QPushButton("Export DataFrame", self)
        self.export_data_button.clicked.connect(self.export_tracking_data)
        export_buttons_layout.addWidget(self.export_data_button)
        self.export_tracking_image_button = QPushButton("Export Image", self)
        self.export_tracking_image_button.clicked.connect(self.export_tracking_image)
        export_buttons_layout.addWidget(self.export_tracking_image_button)
        # After adding export tracking data and export tracking image buttons:
        self.export_tracking_video_button = QPushButton("Export Video", self)
        self.export_tracking_video_button.clicked.connect(self.export_tracking_video)
        export_buttons_layout.addWidget(self.export_tracking_video_button)
        # Left-panel checkbox layout
        checkbox_layout = QHBoxLayout()
        self.show_trajectories_checkbox = QCheckBox("Trajectories")
        self.show_trajectories_checkbox.setChecked(False)
        checkbox_layout.addWidget(self.show_trajectories_checkbox)
        # Add cluster size QCheckbox
        self.show_cluster_size_checkbox = QCheckBox("Cluster Size")
        self.show_cluster_size_checkbox.setChecked(False)
        checkbox_layout.addWidget(self.show_cluster_size_checkbox)
        # Add particle ID QCheckbox
        self.show_particle_id_checkbox = QCheckBox("Particle ID")
        self.show_particle_id_checkbox.setChecked(False)
        checkbox_layout.addWidget(self.show_particle_id_checkbox)
        # Add "Display Time Stamp" checkbox (moved from right panel)
        self.tracking_time_text_checkbox = QCheckBox("Time Stamp")
        self.tracking_time_text_checkbox.setChecked(False)
        checkbox_layout.addWidget(self.tracking_time_text_checkbox)
        # Add "Remove Background" checkbox (moved from right panel)
        self.tracking_remove_background_checkbox = QCheckBox("Remove Background")
        self.tracking_remove_background_checkbox.setChecked(False)
        checkbox_layout.addWidget(self.tracking_remove_background_checkbox)
        tracking_left_layout.addLayout(checkbox_layout)
        # RIGHT PANEL: Scroll Area for Parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        right_container = QWidget()
        scroll.setWidget(right_container)
        tracking_right_main_layout = QVBoxLayout(right_container)
        tracking_right_layout.addWidget(scroll)
        # Title
        parameters_label = QLabel("Tracking Parameters")
        tracking_right_main_layout.addWidget(parameters_label)
        max_proj_tracking_group = QGroupBox("2D Projection")
        max_proj_tracking_layout = QVBoxLayout()
        self.use_2d_projection_checkbox = QCheckBox("Use 2D Projection for Tracking")
        # Initialize checkbox using flag
        self.use_2d_projection_checkbox.setChecked(self.use_maximum_projection)
        self.use_2d_projection_checkbox.stateChanged.connect(self.update_use_maximum_projection)
        max_proj_tracking_layout.addWidget(self.use_2d_projection_checkbox)
        status_text = "2D Projection is ON" if self.use_maximum_projection else "2D Projection is OFF"
        self.tracking_max_proj_status_label = QLabel(status_text)
        self.tracking_max_proj_status_label.setStyleSheet("color: limegreen")
        max_proj_tracking_layout.addWidget(self.tracking_max_proj_status_label)
        max_proj_tracking_group.setLayout(max_proj_tracking_layout)
        tracking_right_main_layout.addWidget(max_proj_tracking_group)
        # Group 1: Source & Threshold
        source_threshold_group = QGroupBox("Source (Select Raw Image or Photobleaching Corrected)")
        source_threshold_layout = QVBoxLayout(source_threshold_group)
        tracking_right_main_layout.addWidget(source_threshold_group)
        # Image Source
        source_threshold_layout.addWidget(QLabel("Image Source:"))
        self.image_source_combo = QComboBox()
        self.image_source_combo.addItems(["Original Image", "Photobleaching Corrected"])
        # Set text to orangered and bold for selected item
        self.image_source_combo.setStyleSheet("color: orangered")
        self.image_source_combo.setCurrentIndex(0)
        self.image_source_combo.currentIndexChanged.connect(self.on_image_source_changed)
        source_threshold_layout.addWidget(self.image_source_combo)
        # Threshold Selection & Histogram
        threshold_group = QGroupBox("Threshold Selection")
        threshold_layout = QVBoxLayout(threshold_group)
        source_threshold_layout.addWidget(threshold_group)
        self.figure_threshold_hist, self.ax_threshold_hist = plt.subplots(figsize=(6, 1))
        self.canvas_threshold_hist = FigureCanvas(self.figure_threshold_hist)
        self.canvas_threshold_hist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_threshold_hist.setMaximumHeight(300)
        self.canvas_threshold_hist.setMinimumHeight(200)
        # Initialize threshold histogram as blank black panel
        self.figure_threshold_hist.clear()
        self.ax_threshold_hist = self.figure_threshold_hist.add_subplot(111)
        self.ax_threshold_hist.set_facecolor('black')
        self.ax_threshold_hist.axis('off')
        self.canvas_threshold_hist.draw()
        threshold_layout.addWidget(self.canvas_threshold_hist)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(10000)
        self.threshold_slider.setValue(0)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self.update_threshold_value)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addStretch()
        # Create a new group box for Spot Detection and Tracking
        spot_det_track_group = QGroupBox("Spot Detection and Tracking")
        spot_det_track_layout = QHBoxLayout(spot_det_track_group)
        # Button for detecting spots in current frame, renamed "Frame"
        self.detect_spots_button = QPushButton("Single Frame", self)
        self.detect_spots_button.clicked.connect(self.detect_spots_in_current_frame)
        spot_det_track_layout.addWidget(self.detect_spots_button)
        # Button for detecting spots in all frames, renamed "All Frames"
        self.detect_all_spots_button = QPushButton("Detection", self)
        self.detect_all_spots_button.clicked.connect(self.detect_spots_all_frames)
        spot_det_track_layout.addWidget(self.detect_all_spots_button)
        # Button for performing particle tracking, renamed "Tracking"
        self.tracking_button = QPushButton("Tracking", self)
        self.tracking_button.clicked.connect(self.perform_particle_tracking)
        spot_det_track_layout.addWidget(self.tracking_button)
        source_threshold_layout.addWidget(spot_det_track_group)
        # Group 2: Spot Detection Parameters
        spot_detection_group = QGroupBox("Spot Detection Parameters")
        spot_layout = QFormLayout(spot_detection_group)
        tracking_right_main_layout.addWidget(spot_detection_group)
        # Min length
        self.min_length_input = QSpinBox()
        self.min_length_input.setMinimum(1)
        self.min_length_input.setMaximum(200)
        self.min_length_input.setValue(self.min_length_trajectory)
        self.min_length_input.valueChanged.connect(self.update_min_length_trajectory)
        spot_layout.addRow("Min Length Trajectory:", self.min_length_input)
        # YX Spot Size
        self.spot_size_input = QSpinBox()
        self.spot_size_input.setMinimum(3)
        self.spot_size_input.setValue(self.yx_spot_size_in_px)
        self.spot_size_input.valueChanged.connect(self.update_yx_spot_size)
        spot_layout.addRow("YX Spot Size (px):", self.spot_size_input)
        # Z Spot Size
        self.spot_size_z_input = QSpinBox()
        self.spot_size_z_input.setMinimum(1)
        self.spot_size_z_input.setValue(self.z_spot_size_in_px)
        self.spot_size_z_input.valueChanged.connect(self.update_z_spot_size)
        spot_layout.addRow("Z Spot Size:", self.spot_size_z_input)
        # Cluster radius
        self.cluster_radius_input = QSpinBox()
        self.cluster_radius_input.setMinimum(100)
        self.cluster_radius_input.setMaximum(2000)
        self.cluster_radius_input.setValue(self.cluster_radius_nm)
        self.cluster_radius_input.valueChanged.connect(self.update_cluster_radius)
        spot_layout.addRow("Cluster radius (nm):", self.cluster_radius_input)
        # Max cluster size
        self.max_spots_cluster_input = QSpinBox()
        self.max_spots_cluster_input.setMinimum(0)
        self.max_spots_cluster_input.setMaximum(1000)
        self.max_spots_cluster_input.setValue(self.maximum_spots_cluster if self.maximum_spots_cluster is not None else 0)
        self.max_spots_cluster_input.valueChanged.connect(self.update_max_spots_cluster)
        spot_layout.addRow("Max Cluster Size (0 for None):", self.max_spots_cluster_input)
        # Group 3: Linking Parameters
        linking_group = QGroupBox("Linking Parameters")
        linking_layout = QFormLayout(linking_group)
        tracking_right_main_layout.addWidget(linking_group)
        # Max range
        self.max_range_search_input = QSpinBox()
        self.max_range_search_input.setMinimum(1)
        self.max_range_search_input.setValue(self.maximum_range_search_pixels)
        self.max_range_search_input.valueChanged.connect(self.update_max_range_search_pixels)
        linking_layout.addRow("Max Range Search (px):", self.max_range_search_input)
        # Memory
        self.memory_input = QSpinBox()
        self.memory_input.setMinimum(0)
        self.memory_input.setMaximum(5)
        self.memory_input.setValue(self.memory)
        self.memory_input.valueChanged.connect(self.update_memory)
        linking_layout.addRow("Memory:", self.memory_input)
        # Control: Random Point Generation
        random_points_group = QGroupBox("Control Spots: Random Locations")
        random_points_layout = QFormLayout(random_points_group)
        tracking_right_main_layout.addWidget(random_points_group)
        # Create spin box for random points
        self.random_points_input = QSpinBox()
        self.random_points_input.setMinimum(1)
        self.random_points_input.setMaximum(100)
        self.random_points_input.setValue(20)
        # Create checkbox to enable random spot generation
        generate_random_points_checkbox = QCheckBox("Generate Random Spots")
        generate_random_points_checkbox.setChecked(False)
        generate_random_points_checkbox.stateChanged.connect(self.generate_random_spots)
        # Create horizontal layout for checkbox and spin box
        hbox = QHBoxLayout()
        hbox.addWidget(generate_random_points_checkbox)
        hbox.addWidget(self.random_points_input)        
        # Add horizontal layout as a row in form layout (label empty since group title is descriptive)
        random_points_layout.addRow("", hbox)
        tracking_right_main_layout.addStretch()
        # Create a horizontal layout for the MSD display at the bottom of the right panel.
        self.msd_layout = QHBoxLayout()
        self.msd_label = QLabel("Mean Square Displacement: Not Calculated")
        self.msd_label.setStyleSheet("color: white; font-weight: bold;")
        self.msd_layout.addWidget(self.msd_label)
        # Add this MSD layout to the right panel layout
        tracking_right_main_layout.addLayout(self.msd_layout)



# =============================================================================
# =============================================================================
# DISTRIBUTION TAB
# =============================================================================
# =============================================================================

    def plot_intensity_histogram(self):
        if self.df_tracking.empty:
            QMessageBox.warning(self, "No Data", "No tracking data available.")
            return
        selected_field = self.intensity_field_combo.currentText()
        selected_channel = self.intensity_channel_combo.currentData()  # channel index
        min_percentile = self.intensity_min_percentile_spin.value()
        max_percentile = self.intensity_max_percentile_spin.value()
        # Determine field name
        field_name = "cluster_size" if selected_field == "cluster_size" else f'{selected_field}_ch_{selected_channel}'
        if field_name not in self.df_tracking.columns:
            ax = self.figure_distribution.add_subplot(111)
            ax.set_facecolor('black')
            ax.axis('off')
            ax.text(0.5, 0.5, f"No data for {field_name}.", horizontalalignment='center', verticalalignment='center', fontsize=12, color='white', transform=ax.transAxes)
            self.canvas_distribution.draw()
            return
        data = self.df_tracking[field_name].dropna().values
        if len(data) == 0:
            ax = self.figure_distribution.add_subplot(111)
            ax.set_facecolor('black')
            ax.axis('off')
            ax.text(0.5, 0.5, f"No data points found for {field_name}.", horizontalalignment='center', verticalalignment='center', fontsize=12, color='white', transform=ax.transAxes)
            self.canvas_distribution.draw()
            return
        mean_val = np.mean(data)
        median_val = np.median(data)
        lower_limit = np.nanpercentile(data, min_percentile)
        upper_limit = np.nanpercentile(data, max_percentile)
        data_for_hist = data[(data >= lower_limit) & (data <= upper_limit)]
        color = 'cyan'
        self.figure_distribution.clear()
        ax = self.figure_distribution.add_subplot(111)
        ax.set_facecolor('black')
        ax.hist(
            data_for_hist,
            bins=60,
            histtype='stepfilled',
            alpha=0.8,
            color=color,
            edgecolor='black',
            linewidth=1,
            label=f"{field_name}"
        )
        ax.set_xlabel(selected_field, color='white')
        ax.set_ylabel('Count', color='white')
        ax.tick_params(colors='white', which='both')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
        text_str = f"Mean={mean_val:.2f}"
        text_str += f"\nMedian={median_val:.2f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', color='black', bbox=props, fontsize=10)
        self.figure_distribution.tight_layout()
        self.canvas_distribution.draw()

    def setup_distributions_tab(self):
        """
        Initialize and configure the “Distributions” tab in the GUI.
        This method builds a two‐panel layout for exploring and exporting histograms of spot‐based metrics.
        Left Panel (Data Visualization & Export):
            - Create a Matplotlib figure and axes for plotting intensity histograms.
            - Embed the figure in a Qt FigureCanvas with a NavigationToolbar.
            - Add an “Export Intensity Image” button to trigger self.export_intensity_image(),
              allowing users to save the current histogram plot.
        Right Panel (Controls):
            1. Selection Group:
                • QComboBox for choosing the data field to plot:
                  ['spot_int', 'spot_size', 'psf_amplitude', 'psf_sigma',
                   'total_spot_int', 'cluster_size', 'snr']
                • QComboBox for selecting the data channel.
            2. Histogram Percentiles Group:
                • Min Percentile (QDoubleSpinBox): range 0.0–50.0%, default 0.0%, step 0.5%.
                • Max Percentile (QDoubleSpinBox): range 50.0–100.0%, default 99.5%, step 0.5%.
            3. Plot Button:
                • “Plot Histogram” QPushButton connected to self.plot_intensity_histogram().
        Layout Details:
            - Use QHBoxLayout to arrange left and right panels (3:1 stretch).
            - Nest QVBoxLayout and QFormLayout within group boxes for structured alignment.
            - Add stretch at the bottom of the right panel to keep controls grouped at the top.
        """

        intensity_layout = QHBoxLayout(self.distribution_tab)
        # Left side: Matplotlib Figure (and bottom export layout)
        left_layout = QVBoxLayout()
        self.figure_distribution, self.ax_intensity = plt.subplots()
        self.canvas_distribution = FigureCanvas(self.figure_distribution)
        self.toolbar_intensity = NavigationToolbar(self.canvas_distribution, self)
        left_layout.addWidget(self.canvas_distribution)
        bottom_export_layout = QHBoxLayout()
        bottom_export_layout.addWidget(self.toolbar_intensity)
        # Create "Export Intensity Image" button
        self.export_intensity_button = QPushButton("Export Distribution Image", self)
        self.export_intensity_button.clicked.connect(self.export_intensity_image)
        bottom_export_layout.addWidget(self.export_intensity_button)
        left_layout.addLayout(bottom_export_layout)
        intensity_layout.addLayout(left_layout, 3)
        # Right side: Controls
        right_layout = QVBoxLayout()
        field_channel_group = QGroupBox("Selection")
        field_channel_layout = QFormLayout(field_channel_group)
        self.intensity_field_combo = QComboBox()
        self.intensity_field_combo.addItems(["spot_int", "spot_size", "psf_amplitude", "psf_sigma", "total_spot_int", "cluster_size", "snr"])
        field_channel_layout.addRow(QLabel("Field:"), self.intensity_field_combo)
        self.intensity_channel_combo = QComboBox()
        field_channel_layout.addRow(QLabel("Channel:"), self.intensity_channel_combo)
        right_layout.addWidget(field_channel_group)
        # Percentile controls
        percentile_group = QGroupBox("Histogram Percentiles")
        percentile_layout = QFormLayout(percentile_group)
        self.intensity_min_percentile_spin = QDoubleSpinBox()
        self.intensity_min_percentile_spin.setRange(0.0, 50)
        self.intensity_min_percentile_spin.setValue(0.0)
        self.intensity_min_percentile_spin.setDecimals(1)
        self.intensity_min_percentile_spin.setSingleStep(0.5)
        self.intensity_min_percentile_spin.setSuffix('%')
        self.intensity_max_percentile_spin = QDoubleSpinBox()
        self.intensity_max_percentile_spin.setRange(50.0, 100.0)
        self.intensity_max_percentile_spin.setValue(99.5)
        self.intensity_max_percentile_spin.setDecimals(1)
        self.intensity_max_percentile_spin.setSingleStep(0.5)
        self.intensity_max_percentile_spin.setSuffix('%')
        percentile_layout.addRow(QLabel("Min Percentile:"), self.intensity_min_percentile_spin)
        percentile_layout.addRow(QLabel("Max Percentile:"), self.intensity_max_percentile_spin)
        right_layout.addWidget(percentile_group)
        # Plot button
        self.plot_intensity_button = QPushButton("Plot Histogram")
        self.plot_intensity_button.clicked.connect(self.plot_intensity_histogram)
        right_layout.addWidget(self.plot_intensity_button)
        right_layout.addStretch()
        intensity_layout.addLayout(right_layout, 1)
# =============================================================================
# =============================================================================
# TIME COURSE TAB
# =============================================================================
# =============================================================================
    def on_data_type_changed(self, new_data_type: str):
        """
        Enable the 'Show Individual Traces' checkbox for all data types
        except 'particles'.
        """
        if new_data_type == "particles":
            self.show_traces_checkbox.setChecked(False)
            self.show_traces_checkbox.setEnabled(False)
        else:
            self.show_traces_checkbox.setEnabled(True)

    # def setup_time_course_tab(self):
    #     """
    #     Initialize and configure the "Time Course" tab in the GUI.
    #     This method sets up the layout and widgets for displaying time‐course plots
    #     of selected imaging data. It constructs:
    #       - A vertical main layout (`time_course_layout`) attached to `self.time_course_tab`.
    #       - A top row of controls (`controls_layout`) including:
    #         • Channel selection combo box (`time_course_channel_combo`).
    #         • Data type combo box (`data_type_combo`) with options:
    #           ["particles", "spot_int", "spot_size", "psf_amplitude",
    #            "psf_sigma", "total_spot_int", "snr"].
    #         • Minimum and maximum percentile spin boxes (`min_percentile_spinbox`,
    #           `max_percentile_spinbox`) for range filtering.
    #         • A "Plot Time Course" button (`plot_time_course_button`) that
    #           triggers `self.plot_intensity_time_course`.
    #       - A Matplotlib figure and axes (`figure_time_course`, `ax_time_course`)
    #         with a black background, embedded in a `FigureCanvas`.
    #       - A bottom horizontal layout (`bottom_layout`) containing:
    #         • A Matplotlib navigation toolbar (`toolbar_time_course`).
    #         • A stretch to align controls.
    #         • An "Export Time Courses Image" button
    #           (`export_time_course_button`) that triggers
    #           `self.export_time_course_image`.
    #     It also applies consistent black/white styling to the axes, ticks,
    #     spines, labels, and grid lines, and calls `tight_layout` for proper spacing.
    #     Returns
    #     -------
    #     None
    #     """
    #     time_course_layout = QVBoxLayout(self.time_course_tab)
    #     # Top row of controls
    #     controls_layout = QHBoxLayout()
    #     time_course_layout.addLayout(controls_layout)
    #     # Channel selection
    #     channel_label = QLabel("Select Channel:")
    #     self.time_course_channel_combo = QComboBox()
    #     controls_layout.addWidget(channel_label)
    #     controls_layout.addWidget(self.time_course_channel_combo)
    #     # Data type selection
    #     data_type_label = QLabel("Data Type:")
    #     self.data_type_combo = QComboBox()
    #     self.data_type_combo.addItems(["particles", "spot_int", "spot_size", "psf_amplitude", "psf_sigma", "total_spot_int", "snr"])
    #     controls_layout.addWidget(data_type_label)
    #     controls_layout.addWidget(self.data_type_combo)
    #     # Percentile controls
    #     min_percentile_label = QLabel("Min Percentile:")
    #     self.min_percentile_spinbox = QDoubleSpinBox()
    #     self.min_percentile_spinbox.setRange(0.0, 50.0)
    #     self.min_percentile_spinbox.setValue(5.0)
    #     self.min_percentile_spinbox.setSuffix("%")
    #     controls_layout.addWidget(min_percentile_label)
    #     controls_layout.addWidget(self.min_percentile_spinbox)
    #     max_percentile_label = QLabel("Max Percentile:")
    #     self.max_percentile_spinbox = QDoubleSpinBox()
    #     self.max_percentile_spinbox.setRange(50.0, 100.0)
    #     self.max_percentile_spinbox.setValue(95.0)
    #     self.max_percentile_spinbox.setSuffix("%")
    #     controls_layout.addWidget(max_percentile_label)
    #     controls_layout.addWidget(self.max_percentile_spinbox)
    #     # Plot button
    #     self.plot_time_course_button = QPushButton("Plot Time Course", self)
    #     self.plot_time_course_button.clicked.connect(self.plot_intensity_time_course)
    #     controls_layout.addWidget(self.plot_time_course_button)
    #     # Main figure for time courses
    #     self.figure_time_course, self.ax_time_course = plt.subplots(figsize=(8, 10))
    #     self.figure_time_course.patch.set_facecolor('black')
    #     self.canvas_time_course = FigureCanvas(self.figure_time_course)
    #     time_course_layout.addWidget(self.canvas_time_course)
    #     # Navigation toolbar + export button at bottom
    #     bottom_layout = QHBoxLayout()
    #     self.toolbar_time_course = NavigationToolbar(self.canvas_time_course, self)
    #     bottom_layout.addWidget(self.toolbar_time_course)
    #     # Spacer to push the button to right
    #     bottom_layout.addStretch()
    #     # Export button
    #     self.export_time_course_button = QPushButton("Export Time Courses Image", self)
    #     self.export_time_course_button.clicked.connect(self.export_time_course_image)
    #     bottom_layout.addWidget(self.export_time_course_button)
    #     time_course_layout.addLayout(bottom_layout)
    #     # Set up axis appearance
    #     self.ax_time_course.set_facecolor('black')
    #     self.ax_time_course.tick_params(colors='white', which='both')
    #     self.ax_time_course.spines['bottom'].set_color('white')
    #     self.ax_time_course.spines['top'].set_color('white')
    #     self.ax_time_course.spines['left'].set_color('white')
    #     self.ax_time_course.spines['right'].set_color('white')
    #     self.ax_time_course.xaxis.label.set_color('white')
    #     self.ax_time_course.yaxis.label.set_color('white')
    #     self.ax_time_course.title.set_color('white')
    #     self.ax_time_course.grid(True, which='both', color='gray', linestyle='--', linewidth=0.1)
    #     self.figure_time_course.tight_layout()

    def setup_time_course_tab(self):
        """
        Initialize and configure the "Time Course" tab in the GUI.
        ...
        """
        time_course_layout = QVBoxLayout(self.time_course_tab)

        # Top row of controls
        controls_layout = QHBoxLayout()
        time_course_layout.addLayout(controls_layout)

        # Channel selection
        channel_label = QLabel("Select Channel:")
        self.time_course_channel_combo = QComboBox()
        controls_layout.addWidget(channel_label)
        controls_layout.addWidget(self.time_course_channel_combo)

        # Data type selection
        data_type_label = QLabel("Data Type:")
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems([
            "particles", "spot_int", "spot_size", "psf_amplitude",
            "psf_sigma", "total_spot_int", "snr"
        ])
        controls_layout.addWidget(data_type_label)
        controls_layout.addWidget(self.data_type_combo)

        # Percentile controls
        min_percentile_label = QLabel("Min Percentile:")
        self.min_percentile_spinbox = QDoubleSpinBox()
        self.min_percentile_spinbox.setRange(0.0, 50.0)
        self.min_percentile_spinbox.setValue(5.0)
        self.min_percentile_spinbox.setSuffix("%")
        controls_layout.addWidget(min_percentile_label)
        controls_layout.addWidget(self.min_percentile_spinbox)

        max_percentile_label = QLabel("Max Percentile:")
        self.max_percentile_spinbox = QDoubleSpinBox()
        self.max_percentile_spinbox.setRange(50.0, 100.0)
        self.max_percentile_spinbox.setValue(95.0)
        self.max_percentile_spinbox.setSuffix("%")
        controls_layout.addWidget(max_percentile_label)
        controls_layout.addWidget(self.max_percentile_spinbox)

        # **New: Show Individual Traces checkbox**
        self.show_traces_checkbox = QCheckBox("Show Individual Traces")
        self.show_traces_checkbox.setChecked(False)
        controls_layout.addWidget(self.show_traces_checkbox)

        # Plot button
        self.plot_time_course_button = QPushButton("Plot Time Course", self)
        self.plot_time_course_button.clicked.connect(self.plot_intensity_time_course)
        controls_layout.addWidget(self.plot_time_course_button)

        # Connect data_type changes to enable/disable the checkbox
        self.data_type_combo.currentTextChanged.connect(self.on_data_type_changed)
        # Initialize checkbox enabled state
        self.on_data_type_changed(self.data_type_combo.currentText())

        # Main figure for time courses
        self.figure_time_course, self.ax_time_course = plt.subplots(figsize=(8, 10))
        self.figure_time_course.patch.set_facecolor('black')
        self.canvas_time_course = FigureCanvas(self.figure_time_course)
        time_course_layout.addWidget(self.canvas_time_course)

        # Navigation toolbar + export button at bottom
        bottom_layout = QHBoxLayout()
        self.toolbar_time_course = NavigationToolbar(self.canvas_time_course, self)
        bottom_layout.addWidget(self.toolbar_time_course)
        bottom_layout.addStretch()
        self.export_time_course_button = QPushButton("Export Time Courses Image", self)
        self.export_time_course_button.clicked.connect(self.export_time_course_image)
        bottom_layout.addWidget(self.export_time_course_button)
        time_course_layout.addLayout(bottom_layout)

        # Style the axes for dark theme
        self.ax_time_course.set_facecolor('black')
        self.ax_time_course.tick_params(colors='white', which='both')
        for spine in self.ax_time_course.spines.values():
            spine.set_color('white')
        self.ax_time_course.xaxis.label.set_color('white')
        self.ax_time_course.yaxis.label.set_color('white')
        self.ax_time_course.title.set_color('white')
        self.ax_time_course.grid(True, which='both', color='gray', linestyle='--', linewidth=0.1)
        self.figure_time_course.tight_layout()

# =============================================================================
# =============================================================================
# CORRELATION TAB
# =============================================================================
# =============================================================================

    def update_fit_type(self):
        if self.linear_radio.isChecked():
            self.correlation_fit_type = 'linear'
        elif self.exponential_radio.isChecked():
            self.correlation_fit_type = 'exponential'
        if not self.df_tracking.empty:
            self.compute_correlations()

    def on_correlation_percentile_changed(self):
        self.correlation_min_percentile = self.correlation_min_percentile_input.value()
        self.correlation_max_percentile = self.correlation_max_percentile_input.value()
        if self.correlation_min_percentile >= self.correlation_max_percentile:
            return
        self.display_correlation_plot()

    def update_snr_threshold_for_acf(self, value):
        self.snr_threshold_for_acf_value = value

    def update_correct_baseline(self, state):
        self.correct_baseline = (state == Qt.Checked)

    def update_remove_outliers(self, state):
        self.remove_outliers = (state == Qt.Checked)

    def update_field_name(self, text):
        # Used in compute_correlations
        self.selected_field_name_for_correlation = text

    def update_min_percentage_data_in_trajectory(self, value):
        self.min_percentage_data_in_trajectory = value

    def update_de_correlation_threshold(self, value):
        self.de_correlation_threshold = value

    def update_max_lag(self, value):
        self.max_lag = value
        self.display_correlation_plot()
    
    def update_multi_tau(self, state):
        self.multiTauCheck.setChecked(state)
        self.use_multi = state
        self.display_correlation_plot()

    def create_correlation_channel_checkboxes(self):
        for cb in self.channel_checkboxes:
            self.channel_selection_layout.removeWidget(cb)
            cb.setParent(None)
        self.channel_checkboxes = []
        for idx, channel_name in enumerate(self.channel_names):
            checkbox = QCheckBox(f"Channel {idx}")
            if idx == 0:
                checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.on_channel_selection_changed)
            self.channel_selection_layout.addWidget(checkbox)
            self.channel_checkboxes.append(checkbox)

    @pyqtSlot()
    def on_channel_selection_changed(self):
        self.correlation_results = []
        self.current_total_plots = None
        self.display_correlation_plot()
        self.figure_correlation.clear()
        self.canvas_correlation.draw()
        self.ax_correlation = self.figure_correlation.add_subplot(111)
        self.ax_correlation.set_facecolor('black')
        self.ax_correlation.axis('off')
        self.ax_correlation.text(0.5, 0.5, 'Press "Compute Correlations" to perform calculations.',
                                 horizontalalignment='center', verticalalignment='center',
                                 fontsize=12, color='white', transform=self.ax_correlation.transAxes)
        self.canvas_correlation.draw()


    def compute_correlations(self):
        # 1) sanity checks
        if not getattr(self, 'has_tracked', False):
            QMessageBox.warning(self, "Correlation Unavailable",
                                "You must run particle tracking before computing correlations.")
            return
        if self.df_tracking.empty:
            return
        correlation_type = ('autocorrelation'
                            if self.auto_corr_radio.isChecked()
                            else 'crosscorrelation')
        selected_channels = [
            idx for idx, cb in enumerate(self.channel_checkboxes)
            if cb.isChecked()
        ]
        if correlation_type == 'crosscorrelation' and len(selected_channels) != 2:
            QMessageBox.warning(self, "Invalid Channel Selection",
                                "Please select exactly two channels for crosscorrelation.")
            return
        if correlation_type == 'autocorrelation' and not selected_channels:
            QMessageBox.warning(self, "No Channels Selected",
                                "Please select at least one channel for autocorrelation.")
            return
        field_base = getattr(self, 'selected_field_name_for_correlation', 'spot_int')
        intensity_arrays = {}
        for ch in selected_channels:
            col = f"{field_base}_ch_{ch}"
            if col not in self.df_tracking.columns:
                continue
            arr = mi.Utilities().df_trajectories_to_array(
                dataframe=self.df_tracking,
                selected_field=col,
                fill_value=np.nan,
                total_frames=self.total_frames
            )
            try:
                arr = mi.Utilities().shift_trajectories(
                    arr,
                    min_percentage_data_in_trajectory=self.min_percentage_data_in_trajectory
                )
            except ValueError as e:
                QMessageBox.warning(self, "Correlation Error", str(e))
                return
            intensity_arrays[ch] = arr
        threshold = getattr(self, 'snr_threshold_for_acf_value', 0)
        if threshold > 0:
            for ch, arr_int in list(intensity_arrays.items()):
                col = f'snr_ch_{ch}'
                if col not in self.df_tracking.columns:
                    continue
                arr_snr = mi.Utilities().df_trajectories_to_array(
                    dataframe=self.df_tracking,
                    selected_field=col,
                    fill_value=np.nan,
                    total_frames=self.total_frames
                )
                try:
                    arr_snr = mi.Utilities().shift_trajectories(
                        arr_snr,
                        min_percentage_data_in_trajectory=self.min_percentage_data_in_trajectory
                    )
                except ValueError as e:
                    QMessageBox.warning(self, "Correlation Error", str(e))
                    return
                # compute mean SNR per trajectory, then filter
                #mean_snr = np.nanmean(arr_snr, axis=1)
                #valid_idx = np.where(mean_snr >= threshold)[0]
                #intensity_arrays[ch] = arr_int[valid_idx]
                mean_snr = np.nanmean(arr_snr, axis=1)
                valid_idx = np.where(mean_snr >= threshold)[0]
                valid_idx = np.array(valid_idx, dtype=int)
                if valid_idx.size > 0:
                    arr_len = arr_int.shape[0]
                    invalid = valid_idx[(valid_idx < 0) | (valid_idx >= arr_len)]
                    if invalid.size > 0:
                        bad = int(invalid[0])
                        raise IndexError(
                            f"Index {bad} out of bounds for intensity array "
                            f"of length {arr_len} (channel {ch}). "
                            "Please adjust your SNR threshold or data filtering."
                        )
                intensity_arrays[ch] = arr_int[valid_idx]

        step_size_in_sec = (float(self.list_time_intervals[self.selected_image_index])
                            if getattr(self, 'list_time_intervals', None) else 1.0)
        normalize_g0 = False # self.normalize_g0_checkbox.isChecked()
        start_lag = self.start_lag_input.value()
        index_max = self.index_max_lag_for_fit_input.value()
        use_multi = self.multiTauCheck.isChecked()
        self.correlation_fit_type = 'linear' if self.linear_radio.isChecked() else 'exponential'
        self.correct_baseline = self.correct_baseline_checkbox.isChecked()
        self.remove_outliers = self.remove_outliers_checkbox.isChecked()
        self.index_max_lag_for_fit = index_max
        self.correlation_results = []
        if correlation_type == 'autocorrelation':
            for ch, data in intensity_arrays.items():
                corr = mi.Correlation(
                    primary_data=data,
                    nan_handling='ignore',
                    time_interval_between_frames_in_seconds=step_size_in_sec,
                    start_lag=start_lag,
                    show_plot=False,
                    return_full=False,
                    use_linear_projection_for_lag_0=True,
                    fit_type=self.correlation_fit_type,
                    de_correlation_threshold=self.de_correlation_threshold,
                    correct_baseline=self.correct_baseline,
                    remove_outliers=self.remove_outliers,
                    multi_tau=use_multi,
                )
                mean_corr, std_corr, lags, _, _ = corr.run()
                if index_max >= len(lags):
                    QMessageBox.warning(
                        self, "Max-Lag Adjusted",
                        f"Requested lag {index_max} exceeds available {len(lags)-1} "
                        f"for {'multi-tau' if use_multi else 'linear'} mode.\n"
                        f"Using {len(lags)-1} instead.")
                    index_max = len(lags) - 1
                    self.index_max_lag_for_fit_input.setValue(index_max)
                self.correlation_results.append({
                    'type': 'autocorrelation',
                    'channel': ch,
                    'intensity_array': data,
                    'mean_corr': mean_corr,
                    'std_corr': std_corr,
                    'lags': lags,
                    'step_size_in_sec': step_size_in_sec,
                    'normalize_plot_with_g0': normalize_g0,
                    'index_max_lag_for_fit': index_max,
                    'start_lag': start_lag,
                    'multi_tau': use_multi,
                })

        else:  # crosscorrelation
            ch1, ch2 = selected_channels
            d1 = intensity_arrays.get(ch1)
            d2 = intensity_arrays.get(ch2)
            if d1 is None or d2 is None:
                return
            corr = mi.Correlation(
                primary_data=d1,
                secondary_data=d2,
                nan_handling='ignore',
                time_interval_between_frames_in_seconds=step_size_in_sec,
                show_plot=False,
                return_full=True,
                de_correlation_threshold=self.de_correlation_threshold,
                correct_baseline=self.correct_baseline,
                fit_type=self.correlation_fit_type,
                remove_outliers=self.remove_outliers,
            )
            mean_corr, std_corr, lags, _, _ = corr.run()
            self.correlation_results.append({
                'type': 'crosscorrelation',
                'channel1': ch1,
                'channel2': ch2,
                'intensity_array1': d1,
                'intensity_array2': d2,
                'mean_corr': mean_corr,
                'std_corr': std_corr,
                'lags': lags,
                'step_size_in_sec': step_size_in_sec,
                'normalize_plot_with_g0': normalize_g0,
                'index_max_lag_for_fit': index_max,
                'start_lag': start_lag,
                'multi_tau': use_multi,
            })
        self.display_correlation_plot()


    def setup_correlation_tab(self):
        correlation_layout = QHBoxLayout(self.correlation_tab)
        # Left side: main controls and figure
        left_layout = QVBoxLayout()
        correlation_layout.addLayout(left_layout, stretch=4)
        # Right side: new panel
        right_layout = QVBoxLayout()
        correlation_layout.addLayout(right_layout, stretch=1)
        # Top controls layout (correlation type, select channels, fit type)
        controls_layout = QHBoxLayout()
        left_layout.addLayout(controls_layout)
        # Correlation Type
        correlation_type_group = QGroupBox("Correlation Type")
        correlation_type_layout = QHBoxLayout()
        correlation_type_group.setLayout(correlation_type_layout)
        self.auto_corr_radio = QRadioButton("Auto")
        self.cross_corr_radio = QRadioButton("Cross")
        self.auto_corr_radio.setChecked(True)
        correlation_type_layout.addWidget(self.auto_corr_radio)
        correlation_type_layout.addWidget(self.cross_corr_radio)
        controls_layout.addWidget(correlation_type_group)
        # Channel selection
        channel_selection_group = QGroupBox("Select Channels")
        self.channel_selection_layout = QHBoxLayout()
        channel_selection_group.setLayout(self.channel_selection_layout)
        self.channel_checkboxes = []
        controls_layout.addWidget(channel_selection_group)
        # Fit Type Selection
        correlation_fit_group = QGroupBox("Fit Type")
        correlation_fit_layout = QHBoxLayout()
        correlation_fit_group.setLayout(correlation_fit_layout)
        self.linear_radio = QRadioButton("Linear")
        self.exponential_radio = QRadioButton("Exponential")
        self.linear_radio.setChecked(True)
        correlation_fit_layout.addWidget(self.linear_radio)
        correlation_fit_layout.addWidget(self.exponential_radio)
        self.linear_radio.toggled.connect(self.update_fit_type)
        self.exponential_radio.toggled.connect(self.update_fit_type)
        controls_layout.addWidget(correlation_fit_group)
        # Figure for correlation
        self.figure_correlation = Figure(figsize=(20, 20))
        self.canvas_correlation = FigureCanvas(self.figure_correlation)
        self.canvas_correlation.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.canvas_correlation)
        # Navigation Toolbar + Export button
        correlation_toolbar_layout = QHBoxLayout()
        self.toolbar_correlation = NavigationToolbar(self.canvas_correlation, self)
        correlation_toolbar_layout.addWidget(self.toolbar_correlation)
        export_correlation_image_button = QPushButton("Export Correlation Image", self)
        export_correlation_image_button.clicked.connect(self.export_correlation_image)
        correlation_toolbar_layout.addWidget(export_correlation_image_button)
        left_layout.addLayout(correlation_toolbar_layout)
        # Right panel group for correlation settings
        right_panel_group = QGroupBox("Correlation Settings")
        right_panel_layout = QFormLayout()
        right_panel_group.setLayout(right_panel_layout)
        right_layout.addWidget(right_panel_group)
        # Field selection
        field_label = QLabel("Field:")
        self.field_name_combo = QComboBox()
        self.field_name_combo.addItems(["spot_int", "psf_amplitude", "total_spot_int", "snr"])
        self.field_name_combo.currentTextChanged.connect(self.update_field_name)
        right_panel_layout.addRow(field_label, self.field_name_combo)
        # Max % Empty Data
        max_percentage_label = QLabel("Min % Data:")
        self.max_percentage_spin = QDoubleSpinBox()
        self.max_percentage_spin.setDecimals(3)
        self.max_percentage_spin.setMinimum(0.0)
        self.max_percentage_spin.setMaximum(1.0)
        self.max_percentage_spin.setSingleStep(0.01)
        self.max_percentage_spin.setValue(self.min_percentage_data_in_trajectory)
        self.max_percentage_spin.valueChanged.connect(self.update_min_percentage_data_in_trajectory)
        right_panel_layout.addRow(max_percentage_label, self.max_percentage_spin)
        # Threshold
        threshold_label = QLabel("Decorrelation Threshold:")
        self.de_correlation_threshold_input = QDoubleSpinBox()
        self.de_correlation_threshold_input.setDecimals(3)
        self.de_correlation_threshold_input.setMinimum(0.0)
        self.de_correlation_threshold_input.setMaximum(1.0)
        self.de_correlation_threshold_input.setSingleStep(0.01)
        self.de_correlation_threshold_input.setValue(self.de_correlation_threshold)
        self.de_correlation_threshold_input.valueChanged.connect(self.update_de_correlation_threshold)
        right_panel_layout.addRow(threshold_label, self.de_correlation_threshold_input)
        # Max Lag
        max_lag_label = QLabel("Index Max Lag for Plot:")
        self.max_lag_input = QSpinBox()
        self.max_lag_input.setMinimum(1)
        if hasattr(self, 'max_lag') and self.max_lag is not None:
            self.max_lag_input.setMaximum(self.max_lag - 1)
            self.max_lag_input.setValue(self.max_lag - 1)
        else:
            self.max_lag_input.setMaximum(1)
            self.max_lag_input.setValue(1)
        self.max_lag_input.valueChanged.connect(self.update_max_lag)
        right_panel_layout.addRow(max_lag_label, self.max_lag_input)
        # Index Max Lag for Fit
        self.index_max_lag_for_fit_input = QSpinBox()
        self.index_max_lag_for_fit_input.setMinimum(1)
        self.index_max_lag_for_fit_input.setValue(1000)
        if hasattr(self, 'max_lag') and self.max_lag is not None:
            self.index_max_lag_for_fit_input.setMaximum(self.max_lag - 1)
        else:
            self.index_max_lag_for_fit_input.setMaximum(1000)
        right_panel_layout.addRow(QLabel("Index Max Lag for Fit:"), self.index_max_lag_for_fit_input)
        # Start Lag
        self.start_lag_input = QSpinBox()
        self.start_lag_input.setMinimum(0)
        self.start_lag_input.setValue(1)
        right_panel_layout.addRow(QLabel("Start Lag:"), self.start_lag_input)
        # Min and max percentile for correlation
        self.correlation_min_percentile_input = QDoubleSpinBox()
        self.correlation_min_percentile_input.setDecimals(1)
        self.correlation_min_percentile_input.setMinimum(0)
        self.correlation_min_percentile_input.setMaximum(50.0)
        self.correlation_min_percentile_input.setSingleStep(0.5)
        self.correlation_min_percentile_input.setValue(0.0)  # default
        self.correlation_min_percentile_input.valueChanged.connect(self.on_correlation_percentile_changed)
        right_panel_layout.addRow(QLabel("Min Percentile:"), self.correlation_min_percentile_input)
        self.correlation_max_percentile_input = QDoubleSpinBox()
        self.correlation_max_percentile_input.setDecimals(2)
        self.correlation_max_percentile_input.setMinimum(90.0)
        self.correlation_max_percentile_input.setMaximum(100.0)
        self.correlation_max_percentile_input.setSingleStep(0.1)
        self.correlation_max_percentile_input.setValue(100)  # default
        self.correlation_max_percentile_input.valueChanged.connect(self.on_correlation_percentile_changed)
        right_panel_layout.addRow(QLabel("Max Percentile:"), self.correlation_max_percentile_input)
        # SNR Threshold for ACF
        self.snr_threshold_for_acf = QDoubleSpinBox()
        self.snr_threshold_for_acf.setRange(0.0, 5.0)
        self.snr_threshold_for_acf.setValue(0.1)
        self.snr_threshold_for_acf.setSingleStep(0.1)
        self.snr_threshold_for_acf.valueChanged.connect(self.update_snr_threshold_for_acf)
        right_panel_layout.addRow(QLabel("SNR Threshold for ACF:"), self.snr_threshold_for_acf)
        self.snr_threshold_for_acf_value = self.snr_threshold_for_acf.value()
        # Normalize with G(0) checkbox
        # self.normalize_g0_checkbox = QCheckBox("")
        # self.normalize_g0_checkbox.setChecked(False)
        # right_panel_layout.addRow(QLabel("Normalize:"), self.normalize_g0_checkbox)
        # Correct Baseline checkbox
        self.correct_baseline_checkbox = QCheckBox("")
        self.correct_baseline_checkbox.setChecked(True)
        self.correct_baseline_checkbox.stateChanged.connect(self.update_correct_baseline)
        right_panel_layout.addRow(QLabel("Baseline correction:"), self.correct_baseline_checkbox)
        # Remove outliers from correlation plot checkbox
        self.remove_outliers_checkbox = QCheckBox("")
        self.remove_outliers_checkbox.setChecked(True)
        self.remove_outliers_checkbox.stateChanged.connect(self.update_remove_outliers)
        right_panel_layout.addRow(QLabel("Remove outliers:"), self.remove_outliers_checkbox)
        # Multi-Tau checkbox
        self.multiTauCheck = QCheckBox("")
        self.multiTauCheck.setChecked(False)  # default unchecked (linear correlation)
        self.multiTauCheck.stateChanged.connect(self.update_multi_tau)
        right_panel_layout.addRow(QLabel("Multi-Tau:"), self.multiTauCheck)
        # Compute Correlations Button
        self.compute_correlations_button = QPushButton("Run")
        self.compute_correlations_button.clicked.connect(self.compute_correlations)
        right_layout.addWidget(self.compute_correlations_button)
        right_layout.addStretch()

# =============================================================================
# =============================================================================
# COLOCALIZATION AND COLOCALIZATION MANUAL TABS
# =============================================================================
# =============================================================================

    def update_manual_stats_label(self):
        """Update the manual colocalization stats label based on checked spots."""
        if not hasattr(self, 'manual_checkboxes'):
            return
        total = len(self.manual_checkboxes)
        marked = sum(1 for chk in self.manual_checkboxes if chk.isChecked())
        percent = (marked / total * 100.0) if total > 0 else 0.0
        self.manual_stats_label.setText(f"Total Spots: {total} | Colocalized: {marked} | {percent:.2f}%")

    def populate_manual_checkboxes(self):
        """Populate manual colocalization checkboxes based on last results (ML or Intensity)."""
        if not self.colocalization_results:
            return  # Only proceed if colocalization has been computed
        flag_vector = self.colocalization_results.get('flag_vector')
        if flag_vector is None:
            return
        # Set each checkbox according to the corresponding flag (True/False)
        for checkbox, flag in zip(self.manual_checkboxes, flag_vector):
            checkbox.setChecked(bool(flag))
        self.update_manual_stats_label()

    def cleanup_manual_colocalization(self):
        """Cleanup manual colocalization checkboxes."""
        if not hasattr(self, 'manual_checkboxes'):
            return
        for checkbox in self.manual_checkboxes:
            checkbox.setChecked(False)
        self.update_manual_stats_label()

    def update_colocalization_method(self):
        """Enable the ML threshold input if ML is selected; otherwise, enable the SNR threshold input."""
        if self.method_ml_radio.isChecked():
            self.ml_threshold_input.setEnabled(True)
            self.snr_threshold_input.setEnabled(False)
        else:
            self.ml_threshold_input.setEnabled(False)
            self.snr_threshold_input.setEnabled(True)

    def populate_colocalization_channels(self):
        """Populate the colocalization channel combo boxes.
        The reference channel is automatically set to the channel used in spot detection.
        """
        self.channel_combo_box_1.clear()
        self.channel_combo_box_2.clear()
        if not self.channel_names:
            return
        for idx, name in enumerate(self.channel_names):
            label = f"Ch {idx}"
            self.channel_combo_box_1.addItem(label, idx)
            self.channel_combo_box_2.addItem(label, idx)
        ref_index = self.tracking_channel if hasattr(self, 'tracking_channel') and self.tracking_channel is not None else (self.current_channel if self.current_channel is not None else 0)
        self.channel_combo_box_1.setCurrentIndex(ref_index)
        if len(self.channel_names) > 1:
            other_index = 1 if ref_index == 0 else 0
            self.channel_combo_box_2.setCurrentIndex(other_index)
        else:
            self.channel_combo_box_2.setCurrentIndex(0)
        self.compute_colocalization_button.setEnabled(len(self.channel_names) >= 2)

    def compute_colocalization(self):
        """Perform colocalization analysis and display results."""
        if not getattr(self, 'has_tracked', False) and self.df_tracking.empty:
            QMessageBox.warning(self, "Colocalization Error", "Please run 'All frames' detection and complete tracking before colocalization.")
            return
        ch1 = self.channel_combo_box_1.currentIndex()
        ch2 = self.channel_combo_box_2.currentIndex()
        if ch1 == ch2:
            QMessageBox.warning(self, "Invalid Selection", "Select two different channels.")
            return
        image = self.corrected_image if self.corrected_image is not None else self.image_stack
        if image is None:
            QMessageBox.warning(self, "No Image Data", "Please load and process an image first.")
            return
        if self.use_maximum_projection:
            num_z = image.shape[1]
            max_proj = np.max(image, axis=1, keepdims=True)
            image = np.repeat(max_proj, num_z, axis=1)
        crop_size = int(self.yx_spot_size_in_px) + 5
        if crop_size % 2 == 0:
            crop_size += 1
        croparray, mean_crop, _, crop_size = mi.CropArray(
            image=image,
            df_crops=self.df_tracking,
            crop_size=crop_size,
            remove_outliers=False,
            max_percentile=99.95,
            selected_time_point=None,
            normalize_each_particle=False
        ).run()
        if self.method_ml_radio.isChecked():
            threshold = self.ml_threshold_input.value()
            method_used = "ML"
            crops_norm = mi.Utilities().normalize_crop_return_list(
                array_crops_YXC=mean_crop,
                crop_size=crop_size,
                selected_color_channel=ch2,
                normalize_to_255=True
            )
            flag_vector = ML.predict_crops(model_ML, crops_norm, threshold=threshold)
        else:
            threshold = self.snr_threshold_input.value()
            method_used = "Intensity"
            num_crops = mean_crop.shape[0] // crop_size
            flag_vector = np.array([mi.Utilities().is_spot_in_crop(
                i, crop_size=crop_size, selected_color_channel=ch2,
                array_crops_YXC=mean_crop,
                show_plot=False,
                snr_threshold=threshold)
                for i in range(num_crops)])
        colocal_perc = 0 if len(flag_vector) == 0 else (np.sum(flag_vector) / len(flag_vector)) * 100
        self.colocalization_percentage_label.setText(f"Colocalization Percentage: {colocal_perc:.2f}%")
        self.colocalization_results = {
            'mean_crop_filtered': mean_crop,
            'crop_size': crop_size,
            'flag_vector': flag_vector,
            'ch1_index': ch1,
            'ch2_index': ch2,
            'num_spots_reference': len(flag_vector),
            'num_spots_colocalize': np.sum(flag_vector),
            'colocalization_percentage': colocal_perc,
            'threshold_value': threshold,
            'method': method_used
        }
        self.display_colocalization_results(mean_crop, crop_size, flag_vector, ch1, ch2)
        self.extract_colocalization_data(save_df=False)

    def display_colocalization_results(self, mean_crop, crop_size, flag_vector, ch1, ch2):
        """Display the colocalization result using provided crop data."""
        self.figure_colocalization.clear()
        title = f"Colocalization: {self.colocalization_results['colocalization_percentage']:.2f}%"
        self.plots.plot_matrix_pair_crops(
            mean_crop=mean_crop,
            crop_size=crop_size,
            flag_vector=flag_vector,
            selected_channels=(ch1, ch2),
            figure=self.figure_colocalization,
            crop_spacing=5,
            number_columns=self.columns_spinbox.value(),
            plot_title=title
        )
        try:
            self.plot_image()          
            self.plot_segmentation() 
        except Exception:
            pass
        self.canvas_colocalization.draw()

    def display_colocalization_manual(self):
        """Populate the Manual Colocalization tab with spot crops + checkboxes + separators."""
        scale_factor = getattr(self, "coloc_thumbnail_scale", 4)
        current_name = getattr(self, "selected_image_name", None)
        previous_name = getattr(self, "manual_current_image_name", None)
        if previous_name == current_name:
            self.update_manual_stats_label()
            return
        if not hasattr(self, "manual_current_image_name"):
            self.manual_current_image_name = None
        if not getattr(self, 'has_tracked', False) and self.df_tracking.empty:
            QMessageBox.warning(self, "No Data", "Please perform particle tracking first.")
            return
        image = self.corrected_image if self.corrected_image is not None else self.image_stack
        if image is None:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        if getattr(self, 'use_maximum_projection', False):
            num_z = image.shape[1]
            max_proj = np.max(image, axis=1, keepdims=True)
            image = np.repeat(max_proj, num_z, axis=1)
        crop_size = int(self.yx_spot_size_in_px) + 5
        if crop_size % 2 == 0:
            crop_size += 1
        _, mean_crop, _, crop_size = mi.CropArray(
            image=image,
            df_crops=self.df_tracking,
            crop_size=crop_size,
            remove_outliers=False,
            max_percentile=99.95
        ).run()
        if mean_crop is None or mean_crop.size == 0:
            QMessageBox.information(self, "No Spots", "No detected spots to display.")
            return
        num_spots = mean_crop.shape[0] // crop_size
        self.manual_scroll_area.takeWidget()
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(3)
        container_layout.setContentsMargins(0, 0, 0, 0)
        self.manual_checkboxes = []
        for i in range(num_spots):
            spot_layout = QHBoxLayout()
            spot_layout.setSpacing(1)
            spot_layout.setContentsMargins(0, 0, 0, 0)
            crop_block = mean_crop[i*crop_size:(i+1)*crop_size, :, :]
            for ch in range(image.shape[-1]):
                channel_crop = crop_block[:, :, ch]
                cmin, cmax = np.nanmin(channel_crop), np.nanmax(channel_crop)
                norm = ((channel_crop - cmin) / (cmax - cmin) * 255).astype(np.uint8) if cmax > cmin else np.zeros_like(channel_crop, np.uint8)
                h, w = norm.shape
                qimg = QImage(norm.data, w, h, w, QImage.Format_Grayscale8).copy()
                pix = QPixmap.fromImage(qimg)
                pix = pix.scaled(w*scale_factor, h*scale_factor, Qt.IgnoreAspectRatio, Qt.FastTransformation)
                lbl = QLabel()
                lbl.setPixmap(pix)
                spot_layout.addWidget(lbl)
            chk = QCheckBox(f"Spot {i+1}")
            chk.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            spot_layout.addWidget(chk)
            self.manual_checkboxes.append(chk)
            chk.toggled.connect(self.update_manual_stats_label)
            container_layout.addLayout(spot_layout)
            if i < num_spots - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.HLine)
                sep.setFrameShadow(QFrame.Sunken)
                container_layout.addWidget(sep)
        self.manual_mean_crop = mean_crop
        self.manual_crop_size = crop_size
        self.manual_scroll_area.setWidget(container)
        self.manual_stats_label.setText(f"Total Spots: {num_spots} | Colocalized: 0 | 0.00%")
        self.manual_current_image_name = self.selected_image_name
        try:
            self.plot_image()
            self.plot_segmentation()
        except Exception:
            pass

    def extract_colocalization_data(self, save_df=True):
        if not self.colocalization_results:
            print("No colocalization results!")
            QMessageBox.warning(self, "No Data", "No colocalization data available.")
            return
        ch1 = self.colocalization_results.get('ch1_index', 0)
        ch2 = self.colocalization_results.get('ch2_index', 0)
        ref_spots = self.colocalization_results.get('num_spots_reference', 0)
        col_spots = self.colocalization_results.get('num_spots_colocalize', 0)
        perc = self.colocalization_results.get('colocalization_percentage', 0.0)
        default_filename = self.get_default_export_filename(prefix="colocalization", extension="csv")
        base_name = (self.file_label.text() if hasattr(self, 'file_label') else 'tracking_data').split('.')[0]
        image_name = self.selected_image_name if hasattr(self, 'selected_image_name') else ''
        df = pd.DataFrame({
            "file name": [base_name],
            "image name": [image_name],
            "reference channel": [ch1],
            "colocalize channel": [ch2],
            "number of spots reference": [ref_spots],
            "number of spots colocalize": [col_spots],
            "colocalization percentage": [perc],
            "threshold value": [self.colocalization_results.get("threshold_value")],
            "method": [self.colocalization_results.get("method")]
        })
        self.df_colocalization = df
        if save_df:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Colocalization Data",
                default_filename,
                "CSV Files (*.csv);;All Files (*)",
                options=options
            )
            if file_path:
                if not file_path.lower().endswith('.csv'):
                    file_path += '.csv'
                if os.path.exists(file_path):
                    reply = QMessageBox.question(
                        self,
                        "Overwrite File?",
                        f"'{file_path}' exists. Overwrite?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        return
                try:
                    df.to_csv(file_path, index=False)
                    QMessageBox.information(self, "Success", f"Data exported to:\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Export Failed", f"Error: {str(e)}")

    def reset_colocalization_tab(self):
        self.figure_colocalization.clear()
        ax = self.figure_colocalization.add_subplot(111)
        ax.set_facecolor('black')
        ax.axis('off')
        ax.text(0.5, 0.5, 'No colocalization data available.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='white', transform=ax.transAxes)
        self.canvas_colocalization.draw()
        self.colocalization_results = None
        self.colocalization_percentage_label.setText("")
    
    def extract_manual_colocalization_data(self, save_df=True):
        if not hasattr(self, 'manual_checkboxes') or len(self.manual_checkboxes) == 0:
            print("No manual colocalization data!")
            QMessageBox.warning(self, "No Data", "No manual colocalization selections available.")
            return
        # Summarize results
        ch1 = self.channel_combo_box_1.currentIndex() if hasattr(self, 'channel_combo_box_1') else 0
        ch2 = self.channel_combo_box_2.currentIndex() if hasattr(self, 'channel_combo_box_2') else 1
        total = len(self.manual_checkboxes)
        colocalized = sum(1 for chk in self.manual_checkboxes if chk.isChecked())
        percent = (colocalized / total * 100.0) if total > 0 else 0.0
        # Prepare DataFrame (one summary row)
        base_name = (self.file_label.text() if hasattr(self, 'file_label') else 'tracking_data').split('.')[0]
        image_name = self.selected_image_name if hasattr(self, 'selected_image_name') else ''
        df = pd.DataFrame([{
            "file name": base_name,
            "image name": image_name,
            "reference channel": ch1,
            "colocalize channel": ch2,
            "number of spots reference": total,
            "number of spots colocalize": colocalized,
            "colocalization percentage": percent,
            "threshold value": None,
            "method": "Manual"
        }])
        self.df_manual_colocalization = df
        if save_df:
            default_fname = self.get_default_export_filename(prefix="colocalization_manual", extension="csv")
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Manual Colocalization Data",
                                                      default_fname, "CSV Files (*.csv);;All Files (*)")
            if file_path:
                if not file_path.lower().endswith('.csv'):
                    file_path += '.csv'
                if os.path.exists(file_path):
                    reply = QMessageBox.question(self, "Overwrite File?",
                                     f"'{file_path}' exists. Overwrite?", 
                                     QMessageBox.Yes | QMessageBox.No, 
                                     QMessageBox.No)
                    if reply != QMessageBox.Yes:
                        return
                try:
                    df.to_csv(file_path, index=False)
                    QMessageBox.information(self, "Success", f"Data exported to:\n{file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Export Failed", f"Error: {e}")

    def display_colocalization_plot(self):
        if hasattr(self, 'cid_zoom_coloc'):
            try:
                self.canvas_colocalization.mpl_disconnect(self.cid_zoom_coloc)
            except Exception:
                pass
            self.cid_zoom_coloc = None
        for ax in self.figure_colocalization.axes[1:]:
            try:
                ax.remove()
            except Exception:
                pass
        self.ax_inset = None
        self.figure_colocalization.clear()
        if self.colocalization_results:
            self.display_colocalization_results(
                self.colocalization_results['mean_crop_filtered'],
                self.colocalization_results['crop_size'],
                self.colocalization_results['flag_vector'],
                self.colocalization_results['ch1_index'],
                self.colocalization_results['ch2_index']
            )
        else:
            ax = self.figure_colocalization.add_subplot(111)
            ax.set_facecolor('black')
            ax.axis('off')
            ax.text(0.5, 0.5, 'Press "Compute Colocalization" to calculate.',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='white', transform=ax.transAxes)
        self.canvas_colocalization.draw()
        self.cid_zoom_coloc = self.canvas_colocalization.mpl_connect('motion_notify_event', self.on_colocalization_hover)

    def on_colocalization_hover(self, event):
        # If no axes or no xdata/ydata, do nothing
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if hasattr(self, 'ax_inset') and event.inaxes == self.ax_inset:
            return
        if not self.figure_colocalization.axes:
            return
        ax_main = self.figure_colocalization.axes[0]
        if not ax_main.images:
            return
        x_main, y_main = event.xdata, event.ydata
        im = ax_main.images[0].get_array()
        zoom_fraction = 0.05
        height, width, _ = im.shape if im.ndim == 3 else im.shape
        region_w = int(width * zoom_fraction)
        region_h = int(height * zoom_fraction)
        left = int(np.clip(x_main - region_w/2, 0, width - region_w))
        bottom = int(np.clip(y_main - region_h/2, 0, height - region_h))
        region = im[bottom:bottom+region_h, left:left+region_w, :] if im.ndim == 3 else im[bottom:bottom+region_h, left:left+region_w]
        zoom_scale = 1.5
        zoom_w = int(region_w * zoom_scale)
        zoom_h = int(region_h * zoom_scale)
        region_zoomed = cv2.resize(region, (zoom_w, zoom_h), interpolation=cv2.INTER_NEAREST)
        if self.ax_inset is None or self.ax_inset.figure is not self.figure_colocalization:
            self.ax_inset = inset_axes(ax_main, width="25%", height="25%", loc='upper right', borderpad=1)
            self.ax_inset.set_xticks([])
            self.ax_inset.set_yticks([])
        else:
            self.ax_inset.cla()
        if region_zoomed.ndim == 3:
            self.ax_inset.imshow(region_zoomed, aspect='auto')
        else:
            self.ax_inset.imshow(region_zoomed, cmap='gray', aspect='auto')
        self.ax_inset.set_xticks([])
        self.ax_inset.set_yticks([])
        if hasattr(self, 'rect_zoom') and self.rect_zoom is not None:
            try:
                self.rect_zoom.remove()
            except Exception:
                pass
        self.rect_zoom = patches.Rectangle(
            (left, bottom),
            region_w,
            region_h,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax_main.add_patch(self.rect_zoom)
        self.canvas_colocalization.draw_idle()

    def setup_colocalization_tab(self):
        layout = QVBoxLayout(self.colocalization_tab)
        top_layout = QHBoxLayout()
        channelGroup = QGroupBox("Select Channels")
        chLayout = QHBoxLayout(channelGroup)
        self.channel_combo_box_1 = QComboBox()
        self.channel_combo_box_2 = QComboBox()
        chLayout.addWidget(QLabel("Reference:"))
        chLayout.addWidget(self.channel_combo_box_1)
        chLayout.addWidget(QLabel("Colocalize:"))
        chLayout.addWidget(self.channel_combo_box_2)
        top_layout.addWidget(channelGroup)
        methodGroup = QGroupBox("Colocalization Method")
        methodLayout = QHBoxLayout(methodGroup)
        self.method_ml_radio = QRadioButton("ML")
        self.method_intensity_radio = QRadioButton("Intensity")
        self.method_ml_radio.setChecked(True)
        methodLayout.addWidget(self.method_ml_radio)
        methodLayout.addWidget(self.method_intensity_radio)
        top_layout.addWidget(methodGroup)
        threshOptionsLayout = QHBoxLayout()
        mlGroup = QGroupBox("ML Options")
        mlLayout = QHBoxLayout(mlGroup)
        mlLayout.addWidget(QLabel("ML Threshold:"))
        self.ml_threshold_input = QDoubleSpinBox()
        self.ml_threshold_input.setDecimals(2)
        self.ml_threshold_input.setRange(0.5, 1.0)
        self.ml_threshold_input.setSingleStep(0.05)
        self.ml_threshold_input.setValue(0.50)
        mlLayout.addWidget(self.ml_threshold_input)
        threshOptionsLayout.addWidget(mlGroup)
        intensityGroup = QGroupBox("Intensity Options")
        intensityLayout = QHBoxLayout(intensityGroup)
        intensityLayout.addWidget(QLabel("Threshold:"))
        self.snr_threshold_input = QDoubleSpinBox()
        self.snr_threshold_input.setDecimals(2)
        self.snr_threshold_input.setRange(0.0, 10.0)
        self.snr_threshold_input.setSingleStep(0.1)
        self.snr_threshold_input.setValue(3.0)
        intensityLayout.addWidget(self.snr_threshold_input)
        threshOptionsLayout.addWidget(intensityGroup)
        top_layout.addLayout(threshOptionsLayout)
        columnsGroup = QGroupBox("Crop Columns")
        columnsLayout = QHBoxLayout(columnsGroup)
        columnsLayout.addWidget(QLabel("Columns:"))
        self.columns_spinbox = QSpinBox()
        self.columns_spinbox.setRange(1, 100)
        self.columns_spinbox.setValue(50)
        columnsLayout.addWidget(self.columns_spinbox)
        top_layout.addWidget(columnsGroup)
        actionsGroup = QGroupBox("Actions")
        actionsLayout = QHBoxLayout(actionsGroup)
        self.compute_colocalization_button = QPushButton("Run")
        self.compute_colocalization_button.clicked.connect(self.compute_colocalization)
        actionsLayout.addWidget(self.compute_colocalization_button)
        self.export_colocalization_data_button = QPushButton("Export Data")
        self.export_colocalization_data_button.clicked.connect(lambda: self.extract_colocalization_data(True))
        actionsLayout.addWidget(self.export_colocalization_data_button)
        top_layout.addWidget(actionsGroup)
        top_layout.addStretch()
        layout.addLayout(top_layout, 1)
        self.colocalization_percentage_label = QLabel("")
        self.colocalization_percentage_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.colocalization_percentage_label)
        self.figure_colocalization = Figure()
        self.canvas_colocalization = FigureCanvas(self.figure_colocalization)
        layout.addWidget(self.canvas_colocalization, 8)
        bottom = QHBoxLayout()
        self.toolbar_colocalization = NavigationToolbar(self.canvas_colocalization, self)
        bottom.addWidget(self.toolbar_colocalization)
        self.export_colocalization_image_button = QPushButton("Export Image")
        self.export_colocalization_image_button.clicked.connect(self.export_colocalization_image)
        bottom.addWidget(self.export_colocalization_image_button)
        layout.addLayout(bottom)
        self.populate_colocalization_channels()
        self.method_ml_radio.toggled.connect(self.update_colocalization_method)
        self.update_colocalization_method()
        self.cid_zoom_coloc = self.canvas_colocalization.mpl_connect(
            'motion_notify_event',
            self.on_colocalization_hover
        )


    def setup_colocalization_manual_tab(self):
        manual_layout = QVBoxLayout()
        manual_layout.setContentsMargins(100, 0, 50, 0)
        self.colocalization_manual_tab.setLayout(manual_layout)
        top_bar = QHBoxLayout()
        self.manual_stats_label = QLabel("Total Spots: 0 | Colocalized: 0 | 0.00%")
        top_bar.addWidget(self.manual_stats_label)
        top_bar.addStretch()
        self.populate_manual_coloc_button = QPushButton("Populate")
        self.populate_manual_coloc_button.clicked.connect(self.populate_manual_checkboxes)
        top_bar.addWidget(self.populate_manual_coloc_button)
        self.cleanup_manual_coloc_button = QPushButton("Cleanup")
        self.cleanup_manual_coloc_button.clicked.connect(self.cleanup_manual_colocalization)
        top_bar.addWidget(self.cleanup_manual_coloc_button)
        self.export_manual_coloc_button = QPushButton("Export Data")
        self.export_manual_coloc_button.clicked.connect(lambda: self.extract_manual_colocalization_data(save_df=True))
        top_bar.addWidget(self.export_manual_coloc_button)
        manual_layout.addLayout(top_bar)
        # Scroll area for spot listings (fixed max width)
        self.manual_scroll_area = QScrollArea()
        self.manual_scroll_area.setMaximumWidth(350)
        self.coloc_thumbnail_scale = 4  # thumbnails at 4× size
        self.manual_scroll_area.setWidgetResizable(True)
        self.manual_scroll_area.setContentsMargins(0, 0, 0, 0)
        placeholder = QWidget()
        self.manual_scroll_area.setWidget(placeholder)
        # Center scroll area with horizontal stretches
        hcenter = QHBoxLayout()
        hcenter.addStretch()
        hcenter.addWidget(self.manual_scroll_area)
        hcenter.addStretch()
        manual_layout.addLayout(hcenter)

# =============================================================================
# =============================================================================
# TRACKING VISUALIZATION TAB
# =============================================================================
# =============================================================================

    def display_tracking_visualization(self, selected_channelIndex=None, spot_coord=None):
        """Display the full image with the selected channel (or merged), marking the tracked spot."""
        if not getattr(self, 'has_tracked', False) or self.df_tracking.empty:
            if hasattr(self, 'play_tracking_vis_timer') and self.play_tracking_vis_timer.isActive():
                self.play_tracking_vis_timer.stop()
            if hasattr(self, 'play_tracking_vis_button'):
                self.play_tracking_vis_button.setChecked(False)
            # Clear display without warnings
            self.reset_tracking_visualization_tab()
            return
        if selected_channelIndex is None:
            if not getattr(self, 'tracking_vis_merged', False):
                tvc = getattr(self, 'tracking_vis_channels', None) or []
                try:
                    selected_channelIndex = tvc.index(True)
                except ValueError:
                    selected_channelIndex = self.current_channel
            else:
                selected_channelIndex = self.current_channel
        if spot_coord is None:
            item = self.tracked_particles_list.currentItem()
            found_spot = False
            if item:
                pid = item.data(Qt.UserRole)
                dfm = self.df_tracking[(self.df_tracking['particle'] == pid) & (self.df_tracking['frame'] == self.current_frame)]
                if not dfm.empty:
                    spot_coord = (int(dfm.iloc[0]['y']), int(dfm.iloc[0]['x']))
                    found_spot = True
                else:
                    spot_coord = (0, 0)
            else:
                spot_coord = (0, 0)
        else:
            found_spot = True
        fig = self.figure_tracking_vis
        fig.clear()
        frame_idx = int(self.current_frame)
        img_src = self.get_current_image_source()
        proj = np.max(img_src[frame_idx], axis=0) if img_src.ndim == 5 else (img_src[frame_idx] if img_src.ndim == 4 else img_src)
        # Apply background removal if requested (use segmentation mask)
        frame_img = proj[np.newaxis, ...] if proj.ndim == 2 else proj.transpose(2, 0, 1)
        C, H, W = frame_img.shape
        norm_ch = []
        for ci in range(C):
            plane = frame_img[ci].astype(float)
            # Get channel-specific display parameters or default to global values
            params = self.channelDisplayParams.get(ci, {
                'min_percentile': self.display_min_percentile,
                'max_percentile': self.display_max_percentile,
                'sigma': self.display_sigma,
                'low_sigma': self.low_display_sigma
            })
            lo_val = np.percentile(plane, params['min_percentile'])
            hi_val = np.percentile(plane, params['max_percentile'])
            if hi_val > lo_val:
                plane = np.clip(plane, lo_val, hi_val)
                plane = (plane - lo_val) / (hi_val - lo_val)
            else:
                plane.fill(0)
            # Apply Gaussian smoothing as in plot_image
            if params['low_sigma'] > 0:
                plane = gaussian_filter(plane, sigma=params['low_sigma'])
            if params['sigma'] > 0:
                plane = gaussian_filter(plane, sigma=params['sigma'])
            norm_ch.append(plane)
        norm_stack = np.stack(norm_ch, axis=0)
        # Clamp selected_channelIndex to valid range
        num_channels = norm_stack.shape[0]
        if selected_channelIndex is None or selected_channelIndex >= num_channels:
            selected_channelIndex = min(self.current_channel, num_channels - 1)
        crop_sz = 15
        row, col = spot_coord
        x0 = max(0, min(col - crop_sz // 2, W - crop_sz))
        y0 = max(0, min(row - crop_sz // 2, H - crop_sz))
        x1, y1 = x0 + crop_sz, y0 + crop_sz
        if getattr(self, 'tracking_vis_merged', False):
            main_img = self.compute_merged_image()
            main_cmap = None
        else:
            main_img = norm_stack[selected_channelIndex]
            main_cmap = cmap_list_imagej[selected_channelIndex]
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], hspace=0.1, wspace=0.1)
        ax_main = fig.add_subplot(gs[0, 0])
        gs2 = gs[0, 1].subgridspec(C, 1, hspace=0.1)
        axes_zoom = [fig.add_subplot(gs2[i, 0]) for i in range(C)]        
        # remove background if requested
        if hasattr(self, 'checkbox_remove_bg') and self.checkbox_remove_bg.isChecked():
            if getattr(self, 'segmentation_mask', None) is not None:
                mask_2d = (self.segmentation_mask > 0)
                # If main_img is single‐channel (2D) and mask matches:
                if self.segmentation_mask.shape == main_img.shape:
                    main_img = main_img * mask_2d
                # If main_img is merged RGB (3D) and mask matches height/width:
                elif main_img.ndim == 3 and self.segmentation_mask.shape == main_img.shape[:2]:
                    main_img = main_img * mask_2d[..., None]
        if main_cmap:
            ax_main.imshow(main_img, cmap=main_cmap, interpolation='nearest', vmin=0, vmax=1)
        else:
            ax_main.imshow(main_img, interpolation='nearest')
         # Add scalebar if requested
        if hasattr(self, 'checkbox_scalebar') and self.checkbox_scalebar.isChecked():
            font_props = {'size': 10}
            if getattr(self, 'voxel_yx_nm', None) is not None:
                microns_per_pixel = self.voxel_yx_nm / 1000.0
                scalebar = ScaleBar(
                    microns_per_pixel, units='um', length_fraction=0.2,
                    location='lower right', box_color='black', color='white',
                    font_properties=font_props
                )
                ax_main.add_artist(scalebar)
        # Add timestamp if requested (format in seconds or minutes)
        if hasattr(self, 'checkbox_show_timestamp') and self.checkbox_show_timestamp.isChecked():
            if getattr(self, 'time_interval_value', None) is not None:
                time_val = float(self.current_frame) * float(self.time_interval_value)
                if time_val < 60:
                    ts = f"{time_val:.2f} s"
                else:
                    ts = f"{(time_val / 60):.2f} min"
                ax_main.text(
                    5, 5,
                    ts,
                    color='white',
                    fontsize=12,
                    backgroundcolor='black',
                    va='top',
                    ha='left'
                )
        if found_spot:
            rect = patches.Rectangle((x0, y0), crop_sz, crop_sz, edgecolor='white', facecolor='none', linewidth=2)
            ax_main.add_patch(rect)
        ax_main.axis('off')
        for ci, ax in enumerate(axes_zoom):
            if found_spot:
                crop = norm_stack[ci, y0:y1, x0:x1]
            else:
                crop = np.zeros((crop_sz, crop_sz))
            ax.imshow(crop, cmap=cmap_list_imagej[ci], interpolation='nearest', vmin=0, vmax=1)
            ax.axis('off')
        fig.tight_layout()
        self.canvas_tracking_vis.draw_idle()


    def reset_tracking_visualization_tab(self):
        """Clear the Tracking Visualization tab when the image changes."""
        if hasattr(self, 'play_tracking_vis_timer') and self.play_tracking_vis_timer.isActive():
            self.play_tracking_vis_timer.stop()
        if hasattr(self, 'play_tracking_vis_button'):
            self.play_tracking_vis_button.setChecked(False)
        if hasattr(self, 'tracked_particles_list'):
            self.tracked_particles_list.clear()
        self.has_tracked = False
        self.tracking_vis_merged = False
        self.figure_tracking_vis.clear()
        self.ax_tracking_vis = self.figure_tracking_vis.add_subplot(111)
        self.ax_tracking_vis.set_facecolor('black')
        self.ax_tracking_vis.axis('off')
        self.ax_tracking_vis.text(
            0.5, 0.5, 'No tracking visualization available.',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12, color='white',
            transform=self.ax_tracking_vis.transAxes
        )
        # reset the checkboxes
        if hasattr(self, 'checkbox_remove_bg'):
            self.checkbox_remove_bg.setChecked(False)
        if hasattr(self, 'checkbox_scalebar'):
            self.checkbox_scalebar.setChecked(False)
        if hasattr(self, 'checkbox_show_timestamp'):
            self.checkbox_show_timestamp.setChecked(False)
        self.canvas_tracking_vis.draw_idle()

    
    def setup_tracking_visualization_tab(self):
        """Create and configure the 'Tracking Visualization' tab layout."""
        tracking_vis_layout = QHBoxLayout(self.tracking_visualization_tab)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        tracking_vis_layout.addLayout(left_layout)
        tracking_vis_layout.addLayout(right_layout)
        # Left side: Video display and controls
        self.figure_tracking_vis, self.ax_tracking_vis = plt.subplots(figsize=(8, 8))
        self.figure_tracking_vis.patch.set_facecolor('black')
        self.canvas_tracking_vis = FigureCanvas(self.figure_tracking_vis)
        left_layout.addWidget(self.canvas_tracking_vis)
        self.canvas_tracking_vis.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Percentile spinboxes for intensity scaling
        spin_layout = QHBoxLayout()
        self.min_percentile_spinbox_tracking_vis = QDoubleSpinBox(self)
        self.min_percentile_spinbox_tracking_vis.setRange(0.0, 50.0)
        self.min_percentile_spinbox_tracking_vis.setSingleStep(0.1)
        self.min_percentile_spinbox_tracking_vis.setSuffix("%")
        self.min_percentile_spinbox_tracking_vis.setValue(1.0)
        self.min_percentile_spinbox_tracking_vis.valueChanged.connect(lambda v: self.display_tracking_visualization())
        spin_layout.addWidget(QLabel("Min Int", self))
        spin_layout.addWidget(self.min_percentile_spinbox_tracking_vis)
        self.max_percentile_spinbox_tracking_vis = QDoubleSpinBox(self)
        self.max_percentile_spinbox_tracking_vis.setRange(90.0, 100.0)
        self.max_percentile_spinbox_tracking_vis.setSingleStep(0.05)
        self.max_percentile_spinbox_tracking_vis.setSuffix("%")
        self.max_percentile_spinbox_tracking_vis.setValue(99.9)
        self.max_percentile_spinbox_tracking_vis.valueChanged.connect(lambda v: self.display_tracking_visualization())
        spin_layout.addWidget(QLabel("Max Int", self))
        spin_layout.addWidget(self.max_percentile_spinbox_tracking_vis)
        left_layout.addLayout(spin_layout)
        # Channel selection buttons + Merge toggle
        self.channel_buttons_tracking_vis = []
        self.channel_buttons_layout_tracking_vis = QHBoxLayout()
        left_layout.addLayout(self.channel_buttons_layout_tracking_vis)
        self.merge_tracking_vis_button = QPushButton("Merge Channels", self)
        self.merge_tracking_vis_button.clicked.connect(self.merge_tracking_visualization)
        self.channel_buttons_layout_tracking_vis.addWidget(self.merge_tracking_vis_button)
        # Time slider and Play button
        controls_layout = QHBoxLayout()
        left_layout.addLayout(controls_layout)
        self.time_slider_tracking_vis = QSlider(Qt.Horizontal)
        self.time_slider_tracking_vis.setMinimum(0)
        self.time_slider_tracking_vis.setMaximum(100)
        self.time_slider_tracking_vis.setTickPosition(QSlider.TicksBelow)
        self.time_slider_tracking_vis.setTickInterval(10)
        self.time_slider_tracking_vis.valueChanged.connect(self.update_frame)
        controls_layout.addWidget(self.time_slider_tracking_vis)
        self.play_button_tracking_vis = QPushButton("Play", self)
        self.play_button_tracking_vis.clicked.connect(self.play_pause)
        controls_layout.addWidget(self.play_button_tracking_vis)
        # Export buttons (Image & Video)
        export_buttons_layout = QHBoxLayout()
        left_layout.addLayout(export_buttons_layout)
        self.export_tracking_vis_image_button = QPushButton("Export Image", self)
        self.export_tracking_vis_image_button.clicked.connect(self.export_tracking_visualization_image)
        export_buttons_layout.addWidget(self.export_tracking_vis_image_button)
        self.export_tracking_vis_video_button = QPushButton("Export Video", self)
        self.export_tracking_vis_video_button.clicked.connect(self.export_tracking_visualization_video)
        export_buttons_layout.addWidget(self.export_tracking_vis_video_button)
        # Right side: Tracked particles list
        right_layout.addWidget(QLabel("Tracked Particles:"))
        self.tracked_particles_list = QListWidget()
        self.tracked_particles_list.setFixedWidth(100)
        self.tracked_particles_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tracked_particles_list.currentItemChanged.connect(self.on_particle_selected)
        right_layout.addWidget(self.tracked_particles_list)
        self.checkbox_remove_bg = QCheckBox("Remove Background")    
        self.checkbox_remove_bg.setChecked(False)
        # Add checkbox for background removal
        right_layout.addWidget(self.checkbox_remove_bg)                    
        self.checkbox_scalebar = QCheckBox("Show Scalebar")     
        self.checkbox_scalebar.setChecked(False)    
        # Add checkbox for scalebar
        right_layout.addWidget(self.checkbox_scalebar)     
        self.checkbox_show_timestamp = QCheckBox("Show Time Stamp")
        self.checkbox_show_timestamp.setChecked(False) 
        right_layout.addWidget(self.checkbox_show_timestamp)
        # Connect checkboxes to update visualization    
        self.checkbox_remove_bg.stateChanged.connect(self.display_tracking_visualization)
        self.checkbox_scalebar.stateChanged.connect(self.display_tracking_visualization)   
        self.checkbox_show_timestamp.stateChanged.connect(self.display_tracking_visualization)           
        right_layout.addStretch()

# =============================================================================
# =============================================================================
# CROPS TAB
# =============================================================================
# =============================================================================

    def run_crops_analysis(self):
        """
        Called by the 'Analyze Crops' button.
        Manually trigger analysis and then call display_crops_plot() to update figure.
        """
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image", "No image loaded. Please open an image first.")
            return
        if not getattr(self, 'has_tracked', False):
            QMessageBox.warning(self, "Crops Unavailable", "You must run particle tracking before plotting crops.")
            return
        self.display_crops_plot()

    def setup_crops_tab(self):
        crops_main_layout = QVBoxLayout(self.crops_tab)
        top_controls_layout = QHBoxLayout()
        select_channel_label = QLabel("Select Channel")
        top_controls_layout.addWidget(select_channel_label)
        self.channel_buttons_crops = []
        self.channel_buttons_layout_crops = QHBoxLayout()
        top_controls_layout.addLayout(self.channel_buttons_layout_crops)
        self.analyze_crops_button = QPushButton("Plot Crops")
        self.analyze_crops_button.clicked.connect(self.run_crops_analysis)
        top_controls_layout.addWidget(self.analyze_crops_button)
        top_controls_layout.addStretch()
        crops_main_layout.addLayout(top_controls_layout)
        self.figure_crops = Figure()
        self.canvas_crops = FigureCanvas(self.figure_crops)
        crops_main_layout.addWidget(self.canvas_crops)
        bottom_layout = QHBoxLayout()
        self.toolbar_crops = NavigationToolbar(self.canvas_crops, self)
        bottom_layout.addWidget(self.toolbar_crops)
        self.export_crops_button = QPushButton("Export Crops Image", self)
        self.export_crops_button.clicked.connect(self.export_crops_image)
        bottom_layout.addWidget(self.export_crops_button)
        crops_main_layout.addLayout(bottom_layout)

# =============================================================================
# =============================================================================
# Export TAB
# =============================================================================
# =============================================================================

    def get_default_export_filename(self, prefix=None, extension=None):
        # Base file name comes from file_label if available
        base_file_name = self.file_label.text() if hasattr(self, 'file_label') else 'tracking_data'
        base_file_name = base_file_name.split('.')[0]
        # Selected image name if available
        selected_image_name = self.selected_image_name if hasattr(self, 'selected_image_name') else ''
        # Sanitize strings
        safe_base_file_name = re.sub(r'[^\w\-_\. ]', '_', base_file_name)
        safe_image_name = re.sub(r'[^\w\-_\. ]', '_', selected_image_name)
        # Build name components
        name_components = []
        if prefix:
            name_components.append(prefix)
        name_components.append(safe_base_file_name)
        name_components.append(safe_image_name)
        final_name = '_'.join([comp for comp in name_components if comp])
        # Append extension if provided
        if extension:
            final_name += f".{extension}"
        return final_name

    def on_comments_combo_changed(self, index):
        """
        Update the user comments text edit based on the selected option from the combo box.
        If a preset is chosen, fill the text and disable editing.
        If "Custom" is selected, enable the text edit for user input.
        """
        preset = self.comments_combo.currentText()
        if preset == "Custom":
            self.user_comment_textedit.setEnabled(True)
            self.user_comment_textedit.clear()
            self.user_comment_textedit.setPlaceholderText("Enter your custom comments here...")
        elif preset == "Select a predefined comment":
            self.user_comment_textedit.setEnabled(True)
            self.user_comment_textedit.clear()
        else:
            self.user_comment_textedit.setText(preset)
            self.user_comment_textedit.setEnabled(False)

    def export_selected_items(self):
        options = QFileDialog.Options()
        parent_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Parent Folder for Exports",
            "",
            options=options
        )
        if not parent_folder:
            return
        default_subfolder_name = self.get_default_export_filename(prefix="", extension=None)
        results_folder = Path(parent_folder) / f"results_{default_subfolder_name}"
        results_folder.mkdir(parents=True, exist_ok=True)
        row_count = self.export_table.rowCount()
        for row in range(row_count):
            label_item = self.export_table.item(row, 0)
            if label_item is None:
                continue
            label_text = label_item.text()
            checkbox_widget = self.export_table.cellWidget(row, 1)
            if not checkbox_widget or not isinstance(checkbox_widget, QCheckBox):
                continue
            if checkbox_widget.isChecked():
                # The user wants to export this item
                if label_text == "Export Entire Image as OME-TIF":
                    default_filename = self.get_default_export_filename(prefix=None, extension="ome.tif")
                    self._export_ome_tif(results_folder)

                elif label_text == "Export Displayed Image":
                    default_filename = self.get_default_export_filename(prefix="display", extension="png")
                    out_path = results_folder / default_filename
                    self._export_displayed_image(out_path)

                elif label_text == "Export Segmentation Image":
                    default_filename = self.get_default_export_filename(prefix="segmentation", extension="png")
                    out_path = results_folder / default_filename
                    self._export_segmentation_image(out_path)

                elif label_text == "Export Mask as TIF":
                    default_filename = self.get_default_export_filename(prefix="mask", extension="tif")
                    out_path = results_folder / default_filename
                    self._export_mask_as_tiff(out_path)

                elif label_text == "Export Photobleaching Image":
                    default_filename = self.get_default_export_filename(prefix="photobleaching", extension="png")
                    out_path = results_folder / default_filename
                    self._export_photobleaching_image(out_path)

                elif label_text == "Export Tracking Data":
                    default_filename = self.get_default_export_filename(prefix="tracking", extension="csv")
                    out_path = results_folder / default_filename
                    self._export_tracking_data(out_path)

                elif label_text == "Export Tracking Image":
                    default_filename = self.get_default_export_filename(prefix="tracking_image", extension="png")
                    out_path = results_folder / default_filename
                    self._export_tracking_image(out_path)

                elif label_text == "Export Distributions Image":
                    default_filename = self.get_default_export_filename(prefix="distribution", extension="png")
                    out_path = results_folder / default_filename
                    self._export_intensity_image(out_path)

                elif label_text == "Export Time Course Image":
                    default_filename = self.get_default_export_filename(prefix="time_course", extension="png")
                    out_path = results_folder / default_filename
                    self._export_time_course_image(out_path)

                elif label_text == "Export Correlation Image":
                    default_filename = self.get_default_export_filename(prefix="correlation_image", extension="png")
                    out_path = results_folder / default_filename
                    self._export_correlation_image(out_path)

                elif label_text == "Export Colocalization Image":
                    default_filename = self.get_default_export_filename(prefix="colocalization", extension="png")
                    out_path = results_folder / default_filename
                    self._export_colocalization_image(out_path)

                elif label_text == "Export Colocalization Data":
                    default_filename = self.get_default_export_filename(prefix="colocalization_data", extension="csv")
                    out_path = results_folder / default_filename
                    self._export_colocalization_data_to_csv(out_path)

                elif label_text == "Export Manual Colocalization Image":
                    default_filename = self.get_default_export_filename(prefix="colocalization_manual", extension="png")
                    out_path = results_folder / default_filename
                    self._export_manual_colocalization_image(out_path)

                elif label_text == "Export Manual Colocalization Data":
                    default_filename = self.get_default_export_filename(prefix="colocalization_manual_data", extension="csv")
                    out_path = results_folder / default_filename
                    self._export_manual_colocalization_data_to_csv(out_path)

                elif label_text == "Export Crops Image":
                    default_filename = self.get_default_export_filename(prefix="crops", extension="png")
                    out_path = results_folder / default_filename
                    self._export_crops_image(out_path)

                elif label_text == "Export Metadata File":
                    default_filename = self.get_default_export_filename(prefix="Metadata", extension="txt")
                    out_path = results_folder / default_filename
                    self._export_metadata(file_path=out_path)

                elif label_text == "Export User Comments":
                    default_filename = self.get_default_export_filename(prefix="user_comments", extension="txt")
                    out_path = results_folder / default_filename
                    self._export_user_comments(out_path)

                elif label_text == "Export Random Spots Data":
                    if hasattr(self, 'df_random_spots') and not self.df_random_spots.empty:
                        default_filename = self.get_default_export_filename(prefix="random_location_spots", extension="csv")
                        out_path = results_folder / default_filename
                        try:
                            self.df_random_spots.to_csv(out_path, index=False)
                        except Exception as e:
                            print(f"Error exporting random spots data: {e}")
                    else:
                        print("No random spots data to export.")

        QMessageBox.information(
            self,
            "Export Complete",
            f"Selected items have been exported to:\n{str(results_folder)}"
        )

    def _export_user_comments(self, file_path):
        """
        Write the user comments (from self.user_comment_textedit) into a .txt file.
        """
        comments = self.user_comment_textedit.toPlainText().strip()
        if not comments:
            comments = "No user comments.\n"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(comments)
            print(f"User comments exported to: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export user comments:\n{str(e)}")

    def select_all_exports(self):
        """Check all checkboxes in the Export table."""
        for unique_key, chk in self.export_items_map.items():
            chk.setChecked(True)

    def deselect_all_exports(self):
        """Uncheck all checkboxes in the Export table."""
        for unique_key, chk in self.export_items_map.items():
            chk.setChecked(False)

    def _export_ome_tif(self, out_folder: Path):
        """
        Export the entire image stack as OME-TIFF into out_folder.
        """
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image", "No image to export.")
            return
        # Choose a filename
        default_filename = self.get_default_export_filename(prefix=None, extension=None)
        filename = f"{default_filename}.ome.tif"
        out_path = out_folder / filename
        temp_image = np.moveaxis(self.image_stack, 4, 1)  # move last axis to second place => (T, C, Z, Y, X)
        shape = temp_image.shape  # e.g. (T, C, Z, Y, X)
        bit_depth = 16 if self.bit_depth is None else self.bit_depth
        order = 'TCZYX'
        imagej = False
        time_interval = 1.0
        if hasattr(self, 'time_interval_value') and self.time_interval_value is not None:
            time_interval = float(self.time_interval_value)
        # Convert nm to µm if needed
        physical_size_x = float(self.voxel_yx_nm) / 1000.0 if self.voxel_yx_nm else 1.0
        physical_size_z = float(self.voxel_z_nm) / 1000.0 if self.voxel_z_nm else 1.0
        channel_metadata = {'Name': self.channel_names} if self.channel_names else {}
        # Save using tifffile
        try:
            tifffile.imwrite(
                out_path,
                temp_image.astype(np.uint16),
                shape=shape,
                dtype='uint16',
                imagej=imagej,
                metadata={
                    'axes': order,
                    'PhysicalSizeX': physical_size_x,
                    'PhysicalSizeZ': physical_size_z,
                    'TimeIncrement': time_interval,
                    'TimeIncrementUnit': 's',
                    'SignificantBits': bit_depth,
                    'Channel': channel_metadata
                }
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error writing OME-TIFF:\n{str(e)}")

    def _export_displayed_image(self, file_path):
        """Export the displayed image to a specified file path (without a dialog)."""
        if self.image_stack is None:
            return
        try:
            self.figure_display.savefig(file_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Failed to export displayed image: {e}")

    def _export_segmentation_image(self, file_path):
        try:
            self.figure_segmentation.savefig(file_path, dpi=300)
        except Exception as e:
            print(f"Failed to export segmentation image: {e}")

    def _export_mask_as_tiff(self, file_path):
        if self.segmentation_mask is None:
            return
        mask_to_save = (self.segmentation_mask > 0).astype(np.uint8) * 255
        try:
            tifffile.imwrite(str(file_path), mask_to_save, photometric='minisblack')
        except Exception as e:
            print(f"Failed to export mask: {e}")

    def _export_photobleaching_image(self, file_path):
        try:
            self.figure_photobleaching.savefig(file_path, dpi=300)
        except Exception as e:
            print(f"Failed to export photobleaching image: {e}")

    def _export_tracking_data(self, file_path):
        if self.df_tracking.empty:
            return
        try:
            self.df_tracking.to_csv(file_path, index=False)
        except Exception as e:
            print(f"Failed to export tracking data: {e}")

    def _export_colocalization_data_to_csv(self, out_folder: Path):
        if not hasattr(self, 'df_colocalization') or self.df_colocalization.empty:
            return
        try:
            self.df_colocalization.to_csv(out_folder, index=False)
        except Exception as e:
            print(f"Failed to export colocalization data: {e}")

    def _export_manual_colocalization_image(self, file_path):
        if not hasattr(self, 'manual_checkboxes') or len(self.manual_checkboxes) == 0:
            return  # No manual selections to export
        try:
            # Prepare flag vector from checkboxes
            total = len(self.manual_checkboxes)
            flags = [chk.isChecked() for chk in self.manual_checkboxes]
            percent_marked = (sum(flags) / total * 100.0) if total > 0 else 0.0
            # Determine channels to include (use same channels as selected in UI)
            ch1 = self.channel_combo_box_1.currentIndex() if hasattr(self, 'channel_combo_box_1') else 0
            ch2 = self.channel_combo_box_2.currentIndex() if hasattr(self, 'channel_combo_box_2') else 1
            selected_channels = (ch1, ch2)
            # Create a figure for the manual colocalization mosaic
            fig = Figure()
            title_text = f"Manual Colocalization: {percent_marked:.2f}%"
            # Use the utility to plot all crops, marking selected spots in light blue
            self.plots.plot_matrix_pair_crops(mean_crop=self.manual_mean_crop,
                                    crop_size=self.manual_crop_size,
                                    flag_vector=flags,
                                    selected_channels=selected_channels,
                                    number_columns=self.columns_spinbox.value() if hasattr(self, 'columns_spinbox') else 20,
                                    crop_spacing=5, figure=fig, plot_title=title_text, flag_color="lightblue")
            # Save the figure as a PNG
            fig.savefig(file_path, dpi=300)
        except Exception as e:
            print(f"Failed to export manual colocalization image: {e}")

    def _export_manual_colocalization_data_to_csv(self, out_path: Path):
        if not hasattr(self, 'df_manual_colocalization') or self.df_manual_colocalization.empty:
            return
        try:
            self.df_manual_colocalization.to_csv(out_path, index=False)
        except Exception as e:
            print(f"Failed to export manual colocalization data: {e}")

    def _export_tracking_image(self, file_path):
        try:
            self.figure_tracking.savefig(file_path, dpi=300)
        except Exception as e:
            print(f"Failed to export tracking image: {e}")

    def _export_intensity_image(self, file_path):
        try:
            self.figure_distribution.savefig(file_path, dpi=300)
        except Exception as e:
            print(f"Failed to export intensity image: {e}")

    def _export_time_course_image(self, file_path):
        try:
            #self.figure_time_course.savefig(file_path, dpi=300)
            for ax in self.figure_time_course.axes:
                ax.title.set_fontsize(18)
                ax.xaxis.label.set_size(18)
                ax.yaxis.label.set_size(18)
                ax.tick_params(axis='both', labelsize=16)
            self.figure_time_course.tight_layout()
            self.figure_time_course.savefig(file_path, dpi=300)
        except Exception as e:
            print(f"Failed to export time courses image: {e}")

    def _export_correlation_image(self, file_path):
        try:
            for ax in self.figure_correlation.axes:
                ax.title.set_fontsize(18)
                ax.xaxis.label.set_size(18)
                ax.yaxis.label.set_size(18)
                ax.tick_params(axis='both', labelsize=16)
            self.figure_correlation.tight_layout()
            self.figure_correlation.savefig(file_path, dpi=300)
            #self.figure_correlation.savefig(file_path, dpi=300)
        except Exception as e:
            print(f"Failed to export correlation image: {e}")

    def _export_colocalization_image(self, file_path):
        try:
            self.figure_colocalization.savefig(file_path, dpi=300)
        except Exception as e:
            print(f"Failed to export colocalization image: {e}")

    def _export_crops_image(self, file_path):
        try:
            self.figure_crops.savefig(file_path, dpi=600)
        except Exception as e:
            print(f"Failed to export crops image: {e}")

    def _export_metadata(self, file_path):
        meta = Metadata(
            correct_baseline=self.correct_baseline,
            data_folder_path=self.data_folder_path,
            list_images=self.list_images,
            list_names=self.list_names,
            voxel_yx_nm=self.voxel_yx_nm,
            voxel_z_nm=self.voxel_z_nm,
            channel_names=self.channel_names,
            number_color_channels=self.number_color_channels,
            list_time_intervals=self.list_time_intervals,
            time_interval_value=self.time_interval_value,
            bit_depth=self.bit_depth,
            image_stack=self.image_stack,
            segmentation_mode=self.segmentation_mode,
            selected_image_index=self.selected_image_index,
            channels_spots=self.channels_spots,
            channels_cytosol=self.channels_cytosol,
            channels_nucleus=self.channels_nucleus,
            min_length_trajectory=self.min_length_trajectory,
            yx_spot_size_in_px=self.yx_spot_size_in_px,
            z_spot_size_in_px=self.z_spot_size_in_px,
            cluster_radius_nm=self.cluster_radius_nm,
            maximum_spots_cluster=self.maximum_spots_cluster,
            separate_clusters_and_spots=self.separate_clusters_and_spots,
            maximum_range_search_pixels=self.maximum_range_search_pixels,
            memory=self.memory,
            de_correlation_threshold=self.de_correlation_threshold,
            max_spots_for_threshold=self.max_spots_for_threshold,
            threshold_spot_detection=self.threshold_spot_detection,
            user_selected_threshold=self.user_selected_threshold,
            image_source_combo=self.image_source_combo_value,
            use_fixed_size_for_intensity_calculation=self.use_fixed_size_for_intensity_calculation,
            correlation_fit_type=self.correlation_fit_type,
            index_max_lag_for_fit=self.index_max_lag_for_fit,
            photobleaching_calculated=self.photobleaching_calculated,
            min_percentage_data_in_trajectory=self.min_percentage_data_in_trajectory,
            use_maximum_projection=self.use_maximum_projection,
            photobleaching_mode=self.photobleaching_mode,
            #photobleaching_model=self.photobleaching_model,
            photobleaching_radius=self.photobleaching_radius,
            #photobleaching_number_removed_initial_points=self.photobleaching_number_removed_initial_points,
            file_path=file_path,
            use_ml_checkbox=self.method_ml_radio.isChecked(),
            ml_threshold_input=self.ml_threshold_input,
            link_using_3d_coordinates=self.link_using_3d_coordinates,
            colocalization_method="ML" if self.method_ml_radio.isChecked() else "Intensity",
            colocalization_threshold_value=self.ml_threshold_input.value() if self.method_ml_radio.isChecked() else self.snr_threshold_input.value(),
            multi_tau=self.use_multi
        )
        try:
            meta.write_metadata()
        except Exception as e:
            print(f"Failed to export metadata file: {e}")

    def export_metadata(self):
        if self.data_folder_path is None:
            QMessageBox.warning(self, "No Folder Selected", "Please load or select an image/folder first.")
            return
        default_filename = self.get_default_export_filename(prefix="Metadata", extension="txt")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Metadata File",
            default_filename,
            "Text Files (*.txt);;All Files (*)",
            options=options
        )
        if not file_path:
            return
        meta = Metadata(
            correct_baseline=self.correct_baseline,
            data_folder_path=self.data_folder_path,
            list_images=self.list_images,
            list_names=self.list_names,
            voxel_yx_nm=self.voxel_yx_nm,
            voxel_z_nm=self.voxel_z_nm,
            channel_names=self.channel_names,
            number_color_channels=self.number_color_channels,
            list_time_intervals=self.list_time_intervals,
            time_interval_value=self.time_interval_value,
            bit_depth=self.bit_depth,
            image_stack=self.image_stack,
            segmentation_mode=self.segmentation_mode,
            selected_image_index=self.selected_image_index,
            channels_spots=self.channels_spots,
            channels_cytosol=self.channels_cytosol,
            channels_nucleus=self.channels_nucleus,
            min_length_trajectory=self.min_length_trajectory,
            yx_spot_size_in_px=self.yx_spot_size_in_px,
            z_spot_size_in_px=self.z_spot_size_in_px,
            cluster_radius_nm=self.cluster_radius_nm,
            maximum_spots_cluster=self.maximum_spots_cluster,
            separate_clusters_and_spots=self.separate_clusters_and_spots,
            maximum_range_search_pixels=self.maximum_range_search_pixels,
            memory=self.memory,
            de_correlation_threshold=self.de_correlation_threshold,
            max_spots_for_threshold=self.max_spots_for_threshold,
            threshold_spot_detection=self.threshold_spot_detection,
            user_selected_threshold=self.user_selected_threshold,
            image_source_combo=self.image_source_combo_value,
            use_fixed_size_for_intensity_calculation=self.use_fixed_size_for_intensity_calculation,
            correlation_fit_type=self.correlation_fit_type,
            index_max_lag_for_fit=self.index_max_lag_for_fit,
            photobleaching_calculated=self.photobleaching_calculated,
            min_percentage_data_in_trajectory=self.min_percentage_data_in_trajectory,
            use_maximum_projection=self.use_maximum_projection,
            photobleaching_mode=self.photobleaching_mode,
            #photobleaching_model=self.photobleaching_model,
            photobleaching_radius=self.photobleaching_radius,
            #photobleaching_number_removed_initial_points=self.photobleaching_number_removed_initial_points,
            file_path=file_path,
            use_ml_checkbox=self.use_ml_checkbox,
            ml_threshold_input=self.ml_threshold_input,
            link_using_3d_coordinates=self.link_using_3d_coordinates,
            multi_tau=self.use_multi
        )
        meta.write_metadata()
        QMessageBox.information(self, "Export Success", f"Metadata saved to:\n{file_path}")


    def export_displayed_image_as_png(self):
        """Export the currently displayed image in high quality (300 dpi PNG)."""
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image", "No image to export. Please load an image first.")
            return
        default_filename = self.get_default_export_filename(prefix="display", extension="png")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Displayed Image",
            default_filename,
            "PNG Files (*.png);;All Files (*)",
            options=options
        )
        if file_path:
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
        else:
            return
        try:
            self.figure_display.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Export Successful", f"Image saved as:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"An error occurred:\n{str(e)}")


    def export_tracking_video(self):
        """
        Export the tracking visualization as a video (MP4 or GIF), including any colormaps,
        overlays, and a scalebar (if voxel size is set).
        """
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image", "No image to export. Please load an image first.")
            return
        default_filename = self.get_default_export_filename(prefix="tracking_video", extension="mp4")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Tracking Video",
            default_filename,
            "MP4 Video (*.mp4);;GIF (*.gif)",
            options=options
        )
        if not file_path:
            return
        frames = []
        total_frames = self.image_stack.shape[0]
        for i in range(total_frames):
            self.current_frame = i
            self.plot_tracking()
            if hasattr(self, 'voxel_yx_nm') and self.voxel_yx_nm is not None:
                microns_per_pixel = self.voxel_yx_nm / 1000.0
                font_props = {'size': 10}
                scalebar = ScaleBar(
                    microns_per_pixel, units='um', length_fraction=0.2,
                    location='lower right', box_color='black', color='white',
                    font_properties=font_props
                )
                self.ax_tracking.add_artist(scalebar)
            self.canvas_tracking.draw()
            qimg = self.canvas_tracking.grab().toImage()
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            arr = np.array(ptr).reshape((qimg.height(), qimg.width(), 4))
            frame_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            frames.append(frame_img)
            self.ax_tracking.cla()

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext == ".gif":
            imageio.mimsave(file_path, frames, duration=0.1)
        elif ext == ".mp4":
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(file_path, fourcc, 10, (width, height), True)
            for frame in frames:
                out.write(frame)
            out.release()
        else:
            QMessageBox.warning(self, "Export Error", "Unsupported file extension. Please choose .gif or .mp4")
            return
        QMessageBox.information(self, "Export Video", f"Tracking video exported successfully to:\n{file_path}")


    def export_displayed_video(self):
        """
        Export the currently displayed image (in the Display tab) as a video (MP4 or GIF),
        preserving colormaps, overlays, timestamp, and including a scalebar if voxel size is set.
        """
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image", "No image to export. Please load an image first.")
            return

        default_filename = self.get_default_export_filename(prefix="video", extension="mp4")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Displayed Video",
            default_filename,
            "MP4 Video (*.mp4);;GIF (*.gif)",
            options=options
        )
        if not file_path:
            return
        # Parse ignore-frames list from the line edit
        #ignore_text = self.ignore_frames_line_edit.text()
        #ignore_frames = []
        # if ignore_text:
        #     try:
        #         ignore_frames = [int(x.strip()) for x in ignore_text.split(",") if x.strip().isdigit()]
        #     except Exception:
        #         QMessageBox.warning(self, "Input Error", "Invalid ignore frames format. Please provide comma-separated integers.")
        #         return
        frames = []
        total_frames = self.image_stack.shape[0]
        for i in range(total_frames):
            # if i in ignore_frames:
            #     continue
            # Update the current frame and let plot_image() redraw everything (colormaps, segmentation overlay, etc.)
            self.current_frame = i
            self.plot_image()
            # Add scalebar if voxel size is provided
            if hasattr(self, 'voxel_yx_nm') and self.voxel_yx_nm is not None:
                microns_per_pixel = self.voxel_yx_nm / 1000.0
                font_props = {'size': 10}
                scalebar = ScaleBar(
                    microns_per_pixel, units='um', length_fraction=0.2,
                    location='lower right', box_color='black', color='white',
                    font_properties=font_props
                )
                self.ax_display.add_artist(scalebar)
            # Render the figure and grab as an image
            self.canvas_display.draw()
            qimg = self.canvas_display.grab().toImage()
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            arr = np.array(ptr).reshape((qimg.height(), qimg.width(), 4))
            frame_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            frames.append(frame_img)
            # Clear the axis for the next frame
            self.ax_display.cla()
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext == ".gif":
            imageio.mimsave(file_path, frames, duration=0.1)
        elif ext == ".mp4":
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            isColor = True if (frames[0].ndim == 3 and frames[0].shape[2] == 3) else False
            out = cv2.VideoWriter(file_path, fourcc, 10, (width, height), isColor=isColor)
            for frame in frames:
                if not isColor and frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                out.write(frame)
            out.release()
        else:
            QMessageBox.warning(self, "Export Error", "Unsupported file extension. Please choose .gif or .mp4")
            return
        QMessageBox.information(self, "Export Video", f"Video exported successfully to:\n{file_path}")

    def export_time_course_image(self):
        """Export the currently displayed time courses figure as PNG."""
        options = QFileDialog.Options()
        default_name = self.get_default_export_filename(prefix='time_course', extension='png')
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Time Courses Image",
            default_name,
            "PNG Files (*.png);;All Files (*)",
            options=options
        )
        if not file_path:
            return
        try:
            self.figure_time_course.savefig(file_path, dpi=300)
            QMessageBox.information(self, "Success", f"Time courses image exported successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"An error occurred while exporting:\n{str(e)}")


    def export_tracking_image(self):
        """Export the currently displayed tracking image as a PNG."""
        options = QFileDialog.Options()
        default_name = self.get_default_export_filename(prefix='tracking_image', extension='png')
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Tracking Image",
            default_name,
            "PNG Files (*.png);;All Files (*)",
            options=options
        )
        if not file_path:
            return
        try:
            self.figure_tracking.savefig(file_path, dpi=300)
            QMessageBox.information(self, "Success", f"Tracking image exported successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"An error occurred while exporting:\n{str(e)}")


    def export_tracking_data(self):
        if self.df_tracking.empty:
            QMessageBox.warning(self, "No Data", "No tracking data available to export.")
            return
        default_filename = self.get_default_export_filename(prefix="tracking", extension="csv")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Tracking Data",
            default_filename,
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if file_path:
            if not file_path.lower().endswith('.csv'):
                file_path += '.csv'
            if os.path.exists(file_path):
                reply = QMessageBox.question(
                    self,
                    "Overwrite File?",
                    f"The file '{file_path}' already exists. Do you want to overwrite it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
            try:
                self.df_tracking.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Tracking data exported successfully to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"An error occurred while exporting:\n{str(e)}")


    def export_segmentation_image(self):
        """
        Export the segmentation figure to a PNG file, using the default naming format.
        """
        default_filename = self.get_default_export_filename(prefix="segmentation", extension="png")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Segmentation Image",
            default_filename,
            "PNG Files (*.png);;All Files (*)",
            options=options
        )
        if file_path:
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
            self.figure_segmentation.savefig(file_path, dpi=300)
            QMessageBox.information(self, "Success", f"Segmentation image exported successfully to:\n{file_path}")

    def export_mask_as_tiff(self):
        # Check if mask is available
        if self.segmentation_mask is None:
            QMessageBox.warning(self, "No Mask", "No segmentation mask available to export.")
            return
        default_filename = self.get_default_export_filename(prefix="mask", extension="tif")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mask as TIFF",
            default_filename,
            "TIFF Files (*.tif);;All Files (*)",
            options=options
        )
        if file_path:
            mask_to_save = (self.segmentation_mask > 0).astype(np.uint8)
            mask_to_save = mask_to_save * 255
            try:
                tifffile.imwrite(file_path, mask_to_save, photometric='minisblack')
                QMessageBox.information(self, "Success", f"Mask exported successfully to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"An error occurred while exporting:\n{str(e)}")

    def export_intensity_image(self):
        """
        Export the current Intensity tab figure as a high-resolution PNG.
        """
        default_filename = self.get_default_export_filename(prefix="distribution", extension="png")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Distribution Plot",
            default_filename,
            "PNG Files (*.png);;All Files (*)",
            options=options
        )
        if not file_path:
            return
        try:
            self.figure_distribution.savefig(file_path, dpi=300)
            QMessageBox.information(self, "Export Success", f"Histogram saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error: {str(e)}")
    
    def export_correlation_image(self):
        """Export the currently displayed correlation figure as a PNG."""
        options = QFileDialog.Options()
        default_name = self.get_default_export_filename(prefix='correlation_image', extension='png')
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Correlation Image",
            default_name,
            "PNG Files (*.png);;All Files (*)",
            options=options
        )
        if not file_path:
            return
        try:
            self.figure_correlation.savefig(file_path, dpi=300)
            QMessageBox.information(self, "Success", f"Correlation image exported successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"An error occurred while exporting:\n{str(e)}")
    

    def export_colocalization_image(self):
        """Export the current colocalization figure as a PNG image."""
        if not self.colocalization_results:
            QMessageBox.warning(self, "No Data", "No colocalization image available.")
            return
        default_filename = self.get_default_export_filename(prefix="colocalization", extension="png")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Colocalization Image",
            default_filename,
            "PNG Files (*.png);;All Files (*)",
            options=options
        )
        if file_path:
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
            if os.path.exists(file_path):
                reply = QMessageBox.question(
                    self,
                    "Overwrite File?",
                    f"'{file_path}' exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
            try:
                self.figure_colocalization.savefig(file_path, dpi=300)
                QMessageBox.information(self, "Success", f"Colocalization image exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Error: {str(e)}")

    def export_tracking_visualization_image(self):
        """Export the currently shown tracking visualization frame as a PNG."""
        if self.df_tracking.empty:
            QMessageBox.warning(self, "No Data", "No tracking data available to export.")
            return
        default_filename = self.get_default_export_filename(prefix="tracking_visualization", extension="png")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Tracking Visualization Image", default_filename,
            "PNG Files (*.png);;All Files (*)", options=options
        )
        if not file_path:
            return
        if not file_path.lower().endswith('.png'):
            file_path += '.png'
        if os.path.exists(file_path):
            reply = QMessageBox.question(
                self, "Overwrite File?",
                f"The file '{file_path}' already exists. Do you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        try:
            self.canvas_tracking_vis.draw()
            self.figure_tracking_vis.savefig(file_path, dpi=300)
            QMessageBox.information(self, "Success", f"Image saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"An error occurred while exporting image:\n{e}")

    def export_tracking_visualization_video(self):
        """Export the tracking visualization as a video (MP4 or GIF)."""
        if self.df_tracking.empty:
            QMessageBox.warning(self, "No Data", "No tracking data available to export.")
            return
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        default_filename = self.get_default_export_filename(prefix="tracking_visualization_video", extension="mp4")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Tracking Visualization Video", default_filename,
            "MP4 Video (*.mp4);;GIF (*.gif)", options=options
        )
        if not file_path:
            return
        total_frames = int(self.image_stack.shape[0])
        frames = []
        for i in range(total_frames):
            self.current_frame = i
            self.display_tracking_visualization()
            self.canvas_tracking_vis.draw()
            qimg = self.canvas_tracking_vis.grab().toImage()
            ptr = qimg.bits()
            ptr.setsize(qimg.byteCount())
            arr = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            frames.append(frame_bgr)
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".gif":
                imageio.mimsave(file_path, frames, duration=0.1)
            elif ext == ".mp4":
                height, width, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(file_path, fourcc, 10, (width, height))
                for frame in frames:
                    out.write(frame)
                out.release()
            else:
                QMessageBox.warning(self, "Export Error", "Unsupported file extension. Please choose .mp4 or .gif")
                return
            QMessageBox.information(self, "Export Video", f"Tracking video exported successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"An error occurred while exporting video:\n{e}")

    

    def export_crops_image(self):
        """Export the currently displayed crops figure as PNG."""
        options = QFileDialog.Options()
        default_name = self.get_default_export_filename(prefix='crops', extension='png')
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Crops Image",
            default_name,
            "PNG Files (*.png);;All Files (*)",
            options=options
        )
        if not file_path:
            return
        try:
            self.figure_crops.savefig(file_path, dpi=600)
            QMessageBox.information(self, "Success", f"Crops image exported successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"An error occurred while exporting:\n{str(e)}")


    def setup_export_tab(self):
        """
        Set up the export tab interface with user controls for data export.
        This method creates and configures the export tab layout, which includes:
        - Instructions for the user
        - A predefined comments combo box with common microscopy analysis comments
        - A text edit widget for custom user comments
        - A table widget listing all available export items with checkboxes
        - Control buttons for selecting/deselecting all items and exporting
        The export items include various image formats (OME-TIF, segmentation, tracking),
        data files (tracking data, colocalization data, metadata), and analysis results.
        Each export item can be individually selected or deselected using checkboxes.
        Sets up the following UI components:
        - self.comments_combo: QComboBox for predefined comments
        - self.user_comment_textedit: QTextEdit for custom comments
        - self.export_table: QTableWidget displaying export options
        - self.export_items_map: Dictionary mapping export keys to checkboxes
        - Control buttons for select all, deselect all, and export actions
        The layout uses vertical arrangement with proper margins and stretch spacing.
        """
        
        layout = QVBoxLayout(self.export_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        # Instructions label
        instructions_label = QLabel(
            "Select which items you'd like to export.\n"
            "Use the 'Export Selected Items' button below to export them into a new folder."
        )
        layout.addWidget(instructions_label)
        # --- Predefined Comments Combo Box ---
        # Create a combo box for predefined user comments
        self.comments_combo = QComboBox()
        self.comments_combo.addItem("Select a predefined comment")
        self.comments_combo.addItem("Few or no spots were detected.")
        self.comments_combo.addItem("Aggregates in cell.")
        self.comments_combo.addItem("Cell died during acquisition.")
        self.comments_combo.addItem("Cell divided during acquisition.")
        self.comments_combo.addItem("The cell goes out of focus.")
        self.comments_combo.addItem("Error during microscope acquisition.")
        self.comments_combo.addItem("Error during tracking. Spots not linked correctly.")
        self.comments_combo.addItem("Custom")
        self.comments_combo.currentIndexChanged.connect(self.on_comments_combo_changed)
        layout.addWidget(self.comments_combo)
        # --- User Comments TextEdit ---
        comment_label = QLabel("User Comments:")
        layout.addWidget(comment_label)
        self.user_comment_textedit = QTextEdit()
        self.user_comment_textedit.setPlaceholderText("Enter any notes or comments here...")
        layout.addWidget(self.user_comment_textedit)
        # --- Existing Export Items Table ---
        self.export_table = QTableWidget()
        self.export_table.setColumnCount(2)
        self.export_table.setHorizontalHeaderLabels(["Item", "Export?"])
        self.export_table.horizontalHeader().setStretchLastSection(True)
        self.export_table.setAlternatingRowColors(True)
        export_items = [
            ("Export Entire Image as OME-TIF", "ome_tif"),
            ("Export Displayed Image", "display"),
            ("Export Segmentation Image", "segmentation_img"),
            ("Export Mask as TIF", "segmentation_mask"),
            ("Export Photobleaching Image", "photobleaching"),
            ("Export Tracking Data", "tracking_data"),
            ("Export Tracking Image", "tracking_image"),
            ("Export Distributions Image", "distribution"),
            ("Export Time Course Image", "time_course"),
            ("Export Correlation Image", "correlation"),
            ("Export Colocalization Image", "colocalization"),
            ("Export Colocalization Data", "colocalization_data"),
            ("Export Manual Colocalization Image", "colocalization_manual"),
            ("Export Manual Colocalization Data", "colocalization_manual_data"),
            ("Export Crops Image", "crops"),
            ("Export Metadata File", "metadata"),
            ("Export User Comments", "user_comments"),
            ("Export Random Spots Data", "random_location_spots"),
        ]
        self.export_items_map = {}
        self.export_table.setRowCount(len(export_items))
        for row_idx, (label_text, unique_key) in enumerate(export_items):
            item_label = QTableWidgetItem(label_text)
            item_label.setFlags(item_label.flags() & ~Qt.ItemIsEditable)
            self.export_table.setItem(row_idx, 0, item_label)
            chk = QCheckBox()
            chk.setChecked(True)
            self.export_table.setCellWidget(row_idx, 1, chk)
            self.export_items_map[unique_key] = chk
        self.export_table.resizeColumnsToContents()
        self.export_table.verticalHeader().setDefaultSectionSize(28)
        layout.addWidget(self.export_table)

        # --- Bottom Buttons Layout ---
        buttons_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_exports)
        buttons_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all_exports)
        buttons_layout.addWidget(deselect_all_btn)

        export_selected_btn = QPushButton("Export Selected Items")
        export_selected_btn.clicked.connect(self.export_selected_items)
        buttons_layout.addWidget(export_selected_btn)

        layout.addLayout(buttons_layout)
        layout.addStretch()

# =============================================================================
# =============================================================================
# RESET TABS
# =============================================================================
# =============================================================================

    def reset_export_comment(self):
        """
        Reset the Export tab’s comment fields to their default state.
        """
        self.comments_combo.setCurrentIndex(0)
        self.user_comment_textedit.setEnabled(True)
        self.user_comment_textedit.clear()
        self.user_comment_textedit.setPlaceholderText("Enter any notes or comments here...")

    def reset_display_tab(self):
        self.figure_display.clear()
        self.ax_display = self.figure_display.add_subplot(111)
        self.ax_display.set_facecolor('black')
        self.ax_display.axis('off')
        self.ax_display.text(
            0.5, 0.5, 'No image loaded.',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12, color='white',
            transform=self.ax_display.transAxes
        )
        self.canvas_display.draw()
        self.time_slider_display.setValue(0)
        self.play_button_display.setText("Play")
        self.playing = False

    def reset_segmentation_tab(self):
        self.figure_segmentation.clear()
        self.use_max_proj_for_segmentation = False
        self.ax_segmentation = self.figure_segmentation.add_subplot(111)
        self.ax_segmentation.set_facecolor('black')
        self.ax_segmentation.axis('off')
        self.ax_segmentation.text(
            0.5, 0.5, 'No segmentation performed.',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12, color='white',
            transform=self.ax_segmentation.transAxes
        )
        self.canvas_segmentation.draw()
        self.segmentation_mask = None
        self.selected_points = []

    def reset_photobleaching_tab(self):
        self.figure_photobleaching.clear()
        self.ax_photobleaching = self.figure_photobleaching.add_subplot(111)
        self.ax_photobleaching.set_facecolor('black')
        self.ax_photobleaching.axis('off')
        self.ax_photobleaching.text(
            0.5, 0.5, 'No photobleaching correction applied.',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12, color='white',
            transform=self.ax_photobleaching.transAxes
        )
        self.canvas_photobleaching.draw()
        self.photobleaching_calculated = False
        self.corrected_image = None

    def reset_tracking_tab(self):
        self.df_tracking = pd.DataFrame()
        self.detected_spots_frame = None
        self.figure_tracking.clear()
        self.ax_tracking = self.figure_tracking.add_subplot(111)
        self.ax_tracking.patch.set_facecolor('black')
        self.ax_tracking.axis('off')
        self.ax_tracking.text(
            0.5, 0.5, 'No tracking data available.',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12, color='white',
            transform=self.ax_tracking.transAxes
        )
        self.canvas_tracking.draw()

    def reset_distribution_tab(self):
        self.figure_distribution.clear()
        self.ax_intensity = self.figure_distribution.add_subplot(111)
        self.ax_intensity.set_facecolor('black')
        self.ax_intensity.axis('off')
        self.ax_intensity.text(
            0.5, 0.5, 'No intensity data available.',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12, color='white',
            transform=self.ax_intensity.transAxes
        )
        self.canvas_distribution.draw()

    def reset_time_course_tab(self):
        self.ax_time_course.clear()
        self.ax_time_course.set_facecolor('black')
        self.ax_time_course.set_title('Intensity of Spots', fontsize=10, color='white')
        self.ax_time_course.set_xlabel('Time (s)', color='white')
        self.ax_time_course.set_ylabel('Intensity (au)', color='white')
        self.ax_time_course.text(
            0.5, 0.5, 'No data available.',
            horizontalalignment='center', verticalalignment='center',
            fontsize=12, color='white',
            transform=self.ax_time_course.transAxes
        )
        self.canvas_time_course.draw()

    def reset_correlation_tab(self):
        self.figure_correlation.clear()
        self.ax_correlation = self.figure_correlation.add_subplot(111)
        self.ax_correlation.set_facecolor('black')
        self.ax_correlation.axis('off')
        self.ax_correlation.text(
            0.5, 0.5, 'No correlation data available.',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12, color='white',
            transform=self.ax_correlation.transAxes
        )
        self.canvas_correlation.draw()
        self.correlation_results = []
        self.current_total_plots = None
        for checkbox in self.channel_checkboxes:
            checkbox.setChecked(False)

    def reset_crops_tab(self):
        self.figure_crops.clear()
        self.ax_crops = self.figure_crops.add_subplot(111)
        self.ax_crops.set_facecolor('black')
        self.ax_crops.axis('off')
        self.ax_crops.text(
            0.5, 0.5, 'No crops data available.',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12, color='white',
            transform=self.ax_crops.transAxes
        )
        self.canvas_crops.draw()

    

# =============================================================================
# =============================================================================
# MISC TABS
# =============================================================================
# =============================================================================

    def plot_distribution(self):
        if self.df_tracking.empty:
            self.figure_distribution.clear()
            ax = self.figure_distribution.add_subplot(111)
            ax.set_facecolor('black')
            ax.axis('off')
            ax.text(
                0.5, 0.5, 'No intensity data available.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='white', transform=ax.transAxes
            )
            self.canvas_distribution.draw()
            return
        selected_field = self.intensity_field_combo.currentText()
        selected_channel = self.intensity_channel_combo.currentData()
        min_percentile = self.intensity_min_percentile_spin.value()
        max_percentile = self.intensity_max_percentile_spin.value()
        field_name = "cluster_size" if selected_field == "cluster_size" else f'{selected_field}_ch_{selected_channel}'
        if field_name not in self.df_tracking.columns:
            self.figure_distribution.clear()
            ax = self.figure_distribution.add_subplot(111)
            ax.set_facecolor('black')
            ax.axis('off')
            ax.text(
                0.5, 0.5, f"No data for {field_name}.",
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='white', transform=ax.transAxes
            )
            self.canvas_distribution.draw()
            return
        data = self.df_tracking[field_name].dropna().values
        if len(data) == 0:
            self.figure_distribution.clear()
            ax = self.figure_distribution.add_subplot(111)
            ax.set_facecolor('black')
            ax.axis('off')
            ax.text(
                0.5, 0.5, f"No data points found for {field_name}.",
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='white', transform=ax.transAxes
            )
            self.canvas_distribution.draw()
            return
        median_val = np.nanmedian(data)
        lower_limit = np.nanpercentile(data, min_percentile)
        upper_limit = np.nanpercentile(data, max_percentile)
        data_for_hist = data[(data >= lower_limit) & (data <= upper_limit)]
        self.figure_distribution.clear()
        ax = self.figure_distribution.add_subplot(111)
        ax.set_facecolor('black')
        color = 'cyan'
        ax.hist(data_for_hist, bins=60, alpha=0.8, color=color)
        ax.set_xlabel(selected_field, color='white')
        ax.set_ylabel('Count', color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.1)
        ax.set_title(f"{selected_field} vs Time (Channel {selected_channel})", fontsize=10, color='white')
        text_str = f"Median={median_val:.2f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', color='black', bbox=props, fontsize=10)
        self.figure_distribution.tight_layout()
        self.canvas_distribution.draw()

    
    def display_crops_plot(self):
        # clear & bump DPI for crispness
        self.figure_crops.clear()
        self.figure_crops.set_dpi(300)

        # early exits
        if self.df_tracking.empty:
            return
        if self.corrected_image is None and self.image_stack is None:
            return

        # ==== FIXED LINE: choose the image without using `or` on arrays ====
        if self.corrected_image is not None:
            image_to_use = self.corrected_image
        else:
            image_to_use = self.image_stack

        # compute crop size
        crop_size = int(self.yx_spot_size_in_px) + 5
        if crop_size % 2 == 0:
            crop_size += 1

        # optional max‐projection
        if self.use_maximum_projection:
            image_to_use = np.max(image_to_use, axis=1, keepdims=True)

        # filter & build croparray
        filtered = mi.Utilities().log_filter(image_to_use, spot_radius_px=1)
        croparray, _, _, crop_size = mi.CropArray(
            image=filtered,
            df_crops=self.df_tracking,
            crop_size=crop_size,
            remove_outliers=True,
            max_percentile=99.95,
            selected_time_point=None,
            normalize_each_particle=True
        ).run()

        # render into a single axis
        ax = self.figure_crops.add_subplot(111)
        mi.Plots().plot_croparray(
            croparray=croparray,
            crop_size=crop_size,
            show_particle_labels=True,
            cmap='binary_r',
            max_percentile=99.5,
            selected_channel=self.current_channel,
            axes=[ax]
        )
        ax.set_title(f'Crops — Channel {self.current_channel}')
        ax.axis('off')

        self.figure_crops.tight_layout()
        self.canvas_crops.draw()


    def display_correlation_plot(self):
        fig = self.figure_correlation
        fig.clear()
        fig.patch.set_facecolor('black')
        for ax in fig.axes:
            fig.delaxes(ax)
        results = getattr(self, 'correlation_results', [])
        if not results:
            ax = fig.add_subplot(111)
            ax.set_facecolor('black')
            ax.axis('off')
            ax.text(
                0.5, 0.5,
                'Press "Compute Correlations" to perform calculations.',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12,
                color='white',
                transform=ax.transAxes
            )
            self.canvas_correlation.draw_idle()
            return

        # If multiple autocorrelation results, plot all on one axes for comparison
        is_multi_auto = (
            len(results) > 1
            and all(r['type'] == 'autocorrelation' for r in results)
        )
        if is_multi_auto:
            ax = fig.add_subplot(111)
            for idx, r in enumerate(results):
                color = list_colors_default[idx % len(list_colors_default)]
                self.plots.plot_autocorrelation(
                    mean_correlation                   = r['mean_corr'],
                    error_correlation                  = r['std_corr'],
                    lags                               = np.array(r['lags']) , #* r['step_size_in_sec']
                    time_interval_between_frames_in_seconds = r['step_size_in_sec'],
                    channel_label                      = r['channel'],
                    axes                               = ax,
                    fit_type                           = self.correlation_fit_type,
                    normalize_plot_with_g0             = r.get('normalize_plot_with_g0', False),
                    line_color                         = color,
                    de_correlation_threshold           = self.de_correlation_threshold,
                    start_lag                          = r.get('start_lag', 0),
                    index_max_lag_for_fit              = r.get('index_max_lag_for_fit'),
                    max_lag_index                      = self.max_lag_input.value(),
                    y_min_percentile                   = self.correlation_min_percentile_input.value(),
                    y_max_percentile                   = self.correlation_max_percentile_input.value(),
                    plot_title                         = None,  # title set globally below
                )
            # Combine all autocorrelation values (normalized if needed) to determine y-limits across all channels
            all_vals = np.hstack([
                (
                    (np.array(r['mean_corr']) / np.array(r['mean_corr'])[r['start_lag']])
                    if r.get('normalize_plot_with_g0', False)
                    else np.array(r['mean_corr'])
                )[r.get('start_lag', 0):]
                for r in results
            ])
            ymin = np.nanpercentile(all_vals, self.correlation_min_percentile_input.value())
            ymax = np.nanpercentile(all_vals, self.correlation_max_percentile_input.value())
            ax.set_ylim(ymin, ymax * 1.1)  # 10% padding on top for clarity
            ax.set_facecolor('black')
            ax.tick_params(colors='white', which='both')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.set_xlabel(r'$\tau$ (s)', color='white')
            ylabel = (r"$G(\tau)/G(0)$"
                    if any(r.get('normalize_plot_with_g0') for r in results)
                    else r"$G(\tau)$")
            ax.set_ylabel(ylabel, color='white')
            ax.set_title('Autocorrelation (all channels)', color='white')
            leg = ax.legend(fontsize=8)
            leg.get_frame().set_facecolor('black')
            leg.get_frame().set_edgecolor('white')
            for txt in leg.get_texts():
                txt.set_color('white')
            ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.1)
            fig.tight_layout()
            self.canvas_correlation.draw_idle()
            return

        # Otherwise, plot each result (auto or cross-correlation) in its own subplot
        axes = fig.subplots(nrows=len(results), ncols=1, squeeze=False)
        for i, r in enumerate(results):
            ax = axes[i][0]
            if r['type'] == 'autocorrelation':
                color = list_colors_default[r['channel'] % len(list_colors_default)]
                self.plots.plot_autocorrelation(
                    mean_correlation                   = r['mean_corr'],
                    error_correlation                  = r['std_corr'],
                    lags                               = r['lags'],
                    time_interval_between_frames_in_seconds = r['step_size_in_sec'],
                    channel_label                      = r['channel'],
                    axes                               = ax,
                    plot_title                         = f'Autocorrelation Channel {r["channel"]}',
                    fit_type                           = self.correlation_fit_type,
                    normalize_plot_with_g0             = r.get('normalize_plot_with_g0', False),
                    line_color                         = color,
                    de_correlation_threshold           = self.de_correlation_threshold,
                    max_lag_index                      = self.max_lag_input.value(),
                    index_max_lag_for_fit              = r.get('index_max_lag_for_fit'),
                    start_lag                          = r.get('start_lag', 0),
                    y_min_percentile                   = self.correlation_min_percentile_input.value(),
                    y_max_percentile                   = self.correlation_max_percentile_input.value(),
                )
            else:  # Cross-correlation case
                self.plots.plot_crosscorrelation(
                    mean_correlation       = r['mean_corr'],
                    error_correlation      = r['std_corr'],
                    lags                   = r['lags'],
                    axes                   = ax,
                    normalize_plot_with_g0 = r.get('normalize_plot_with_g0', False),
                    line_color             = 'cyan',
                    max_lag_index          = self.max_lag_input.value(),
                    y_min_percentile       = self.correlation_min_percentile_input.value(),
                    y_max_percentile       = self.correlation_max_percentile_input.value(),
                )
            # Format each subplot with dark theme and grid
            ax.set_facecolor('black')
            ax.tick_params(colors='white', which='both')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.1)
        fig.tight_layout()
        self.canvas_correlation.draw_idle()


    def plot_intensity_time_course(self):
        channel_index = self.time_course_channel_combo.currentIndex()
        data_type = self.data_type_combo.currentText()
        lower_percentile = self.min_percentile_spinbox.value()
        upper_percentile = self.max_percentile_spinbox.value()
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return
        if self.df_tracking.empty:
            QMessageBox.warning(self, "No Tracking Data", "Please perform particle tracking first.")
            return
        self.ax_time_course.clear()
        time_interval = float(self.list_time_intervals[self.selected_image_index]) \
            if self.list_time_intervals and len(self.list_time_intervals) > self.selected_image_index else 1.0
        total_frames = self.image_stack.shape[0]
        time_points_in_seconds = np.arange(0, total_frames * time_interval, time_interval)
        if data_type == "particles":
            particles_per_frame = self.df_tracking.groupby('frame')['particle'].nunique()
            all_frames = np.arange(total_frames)
            particles_per_frame = particles_per_frame.reindex(all_frames, fill_value=0)
            self.ax_time_course.plot(time_points_in_seconds, particles_per_frame, 'o-', color='orangered', linewidth=2)
            self.ax_time_course.set_title("Number of Particles vs Time", fontsize=10, color='white')
            self.ax_time_course.set_xlabel("Time (s)", color='white')
            self.ax_time_course.set_ylabel('Number of Particles', color='white')
            max_particles = particles_per_frame.max()
            self.ax_time_course.set_ylim([0, max_particles + 1])
        else:
            field_name = f"{data_type}_ch_{channel_index}"
            if field_name not in self.df_tracking.columns:
                QMessageBox.warning(self, "Data Error", f"No {data_type} data for Channel {channel_index}.")
                return
            total_frames = self.image_stack.shape[0]  # total frames in the stack
            intensity_array = mi.Utilities().df_trajectories_to_array(
                dataframe=self.df_tracking,
                selected_field=field_name,
                fill_value=np.nan,
                total_frames=total_frames
            )
            # **Plot individual traces if option is enabled** 
            if self.show_traces_checkbox.isChecked():
                for idx in range(intensity_array.shape[0]):
                    trace = intensity_array[idx, :]
                    if np.all(np.isnan(trace)):
                        continue  # skip if a trace has no data (shouldn't happen if particle exists)
                    self.ax_time_course.plot(time_points_in_seconds, trace, '-', color='gray',
                                            linewidth=1, alpha=0.5, label='_nolegend_')
            # Calculate mean and std dev across all particles at each time point
            mean_time_intensity = np.nanmean(intensity_array, axis=0)
            std_time_intensity  = np.nanstd(intensity_array, axis=0)
            # Replace NaN (if any frame has all NaNs) with 0 to avoid issues in plotting
            mean_time_intensity = np.nan_to_num(mean_time_intensity)
            std_time_intensity  = np.nan_to_num(std_time_intensity)
            # Plot the mean intensity over time (as a line) 
            self.ax_time_course.plot(time_points_in_seconds, mean_time_intensity, 'o-',
                                    color='cyan', linewidth=2, label='Mean', alpha=0.5, zorder=3)
            self.ax_time_course.fill_between(time_points_in_seconds,
                                            mean_time_intensity - std_time_intensity,
                                            mean_time_intensity + std_time_intensity,
                                            color='cyan', alpha=0.3, label='Std Dev', zorder=1)
            self.ax_time_course.set_title(f"{data_type.capitalize()} vs Time (Channel {channel_index})",
                                        fontsize=10, color='white')
            self.ax_time_course.set_xlabel("Time (s)", color='white')
            self.ax_time_course.set_ylabel(f"{data_type.capitalize()} (au)", color='white')
            lower_y = np.nanpercentile(intensity_array, lower_percentile)
            upper_y = np.nanpercentile(intensity_array, upper_percentile)
            y_range = upper_y - lower_y
            self.ax_time_course.set_ylim([lower_y - 0.1 * y_range, upper_y + 0.1 * y_range])
            self.ax_time_course.set_xlim([time_points_in_seconds[0], time_points_in_seconds[-1]])
            self.ax_time_course.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, 1))
        self.ax_time_course.tick_params(axis='x', colors='white')
        self.ax_time_course.tick_params(axis='y', colors='white')
        self.figure_time_course.tight_layout()
        self.canvas_time_course.draw()

# =============================================================================
# =============================================================================
# CHANGING TABS
# =============================================================================
# =============================================================================
    
    
    def on_tab_change(self, index):
        if index == 0:
            self.plot_image()
        elif index == 1:
            self.plot_segmentation()
        elif index == 2:
            self.plot_photobleaching()
        elif index == 3:
            self.plot_tracking()
        elif index == 4:
            self.plot_distribution()
        elif index == 5:
            pass
        elif index == 6:
            self.display_correlation_plot()
        elif index == 7:
            self.display_colocalization_plot()
            if hasattr(self, 'canvas_colocalization'):
                if hasattr(self, 'cid_zoom_coloc'):
                    try:
                        self.canvas_colocalization.mpl_disconnect(self.cid_zoom_coloc)
                    except Exception:
                        pass
                self.cid_zoom_coloc = self.canvas_colocalization.mpl_connect('motion_notify_event', self.on_colocalization_hover)
        elif index == 8:
            self.display_colocalization_manual()
        elif index == 9:
            if not (getattr(self, 'has_tracked', False)) or self.df_tracking.empty:
                QMessageBox.warning(self, "No Data", "Please perform particle tracking first.")
                self.tabs.setCurrentIndex(3)
                return
            self.tracked_particles_list.clear()
            for pid in sorted(self.df_tracking['particle'].unique()):
                count = int((self.df_tracking['particle'] == pid).sum())
                item = QListWidgetItem(f"{pid}:{count}")
                item.setData(Qt.UserRole, pid)
                self.tracked_particles_list.addItem(item)
            if self.tracked_particles_list.count() > 0 and self.tracked_particles_list.currentRow() < 0:
                self.tracked_particles_list.setCurrentRow(0)
            self.display_tracking_visualization()
        elif index == 10:
            pass
        elif index == 11:
            if hasattr(self, 'manual_checkboxes'):
                self.extract_manual_colocalization_data(save_df=False)

# =============================================================================
# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
# =============================================================================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    plt.style.use('dark_background')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    app.setApplicationName("micro")
    app.setApplicationDisplayName("micro")
    app.setWindowIcon(QIcon(str(icon_file)))
    main_window = GUI(icon_path=icon_file)
    main_window.show()
    sys.exit(app.exec_())