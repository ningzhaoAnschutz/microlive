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

# Suppress macOS native warnings
if sys.platform == 'darwin':
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

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
vispy_logging = None
try:
    from vispy import logging as vispy_logging
except ImportError:
    pass


# import multiprocessing.resource_tracker
# def fix_multiprocessing_cleanup():
#     original_stop = multiprocessing.resource_tracker.ResourceTracker._stop
#     def new_stop(self, use_blocking_lock=False):
#         try:
#             original_stop(self, use_blocking_lock=use_blocking_lock)
#         except (ChildProcessError, OSError):
#             pass
#     multiprocessing.resource_tracker.ResourceTracker._stop = new_stop
# if 'multiprocessing' in sys.modules:
#     fix_multiprocessing_cleanup()

# =============================================================================
# UI DIALOGS, WIDGET, PLOTTING CLASSES
# =============================================================================

# Warnings and logging configuration
def configure_logging_and_styles():
    """
    Set up warnings filters, VisPy logging level, Qt message handler,
    and a logging filter to suppress specific stylesheet parse warnings.
    """
    # Setup standard logging
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'micro_gui.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
    if vispy_logging is not None:
        vispy_logging.set_level('error')
    logging.getLogger('vispy').setLevel(logging.ERROR)
    def qt_message_handler(msg_type, context, message):
        msg = str(message)
        if "parse stylesheet" not in msg.lower():
            sys.__stderr__.write(msg + "\n")
            if msg_type == QtMsgType.QtWarningMsg:
                logging.warning(f"Qt Warning: {msg}")
            elif msg_type == QtMsgType.QtCriticalMsg:
                logging.error(f"Qt Critical: {msg}")
            elif msg_type == QtMsgType.QtFatalMsg:
                logging.critical(f"Qt Fatal: {msg}")
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
                    # Dynamically concatenate images with spacers
                    if len(combined_img_list) > 1:
                        combined_parts = []
                        for i, img in enumerate(combined_img_list):
                            combined_parts.append(img)
                            if i < len(combined_img_list) - 1:
                                combined_parts.append(spacer)
                        combined_img = np.concatenate(combined_parts, axis=1)
                    elif len(combined_img_list) == 1:
                        combined_img = combined_img_list[0]
                    else:
                        # Should not happen given logic above, but safe fallback
                        combined_img = np.zeros((crop_size, crop_size), dtype=np.uint8)
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


    def plot_autocorrelation(self, mean_correlation, error_correlation, lags, correlations_array=None,
                            time_interval_between_frames_in_seconds=1, channel_label=0,
                            index_max_lag_for_fit=None, start_lag=0, line_color='blue',
                            plot_title=None, fit_type='linear', de_correlation_threshold=0.05,
                            normalize_plot_with_g0=False, axes=None, max_lag_index=None, plot_individual_trajectories=False,
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
        # plotting individual trajectories.
        if plot_individual_trajectories and correlations_array is not None:    
            for i in range(correlations_array.shape[0]):
                ax.plot(lags[start_lag:], correlations_array[i][start_lag:], '-', color='cyan', linewidth=1, alpha=0.5)
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
                logging.info(f"Decorrelation threshold value: {de_correlation_threshold_value}")
            except Exception as e:
                logging.warning(f"Could not find the decorrelation point automatically: {e}")
                # Fall back to the last correlation point
                index_max_lag_for_fit = normalized_correlation.shape[0]
                de_correlation_threshold_value = normalized_correlation[index_max_lag_for_fit - 1]
                logging.info(f"Falling back to last point: {de_correlation_threshold_value}")
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
                except Exception as e:
                    logging.error(f"Error in linear fit: {e}")
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
            logging.info(f"Fitted G(0): {G0_fitted}")
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
                logging.warning("Could not find a time where G(τ) falls below threshold.")
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
                logging.warning('max_lag_index is out of range. Setting it to the last index')
            if max_lag_index < 20:
                space_before_start = 5
            else:
                space_before_start = 20
            ax.set_xlim(lags[start_lag]-space_before_start, lags[max_lag_index])
        if y_min_percentile is None:
            y_min_percentile = 0.1
        if y_max_percentile is None:
            y_max_percentile = 99.9

        valid_data = normalized_correlation[start_lag:]
        if valid_data.size > 0:
            computed_y_min = np.nanpercentile(valid_data, y_min_percentile)
            computed_y_max = np.nanpercentile(valid_data, y_max_percentile)
            # leave some room for computed_y_max value, use 20% more than the maximum
            computed_y_max += 0.2 * abs(computed_y_max) if computed_y_max != 0 else 0.1
            
            if not (np.isfinite(computed_y_min) and np.isfinite(computed_y_max)):
                ax.relim()            
                ax.autoscale_view()   
            else:
                ax.set_ylim(computed_y_min, computed_y_max)
        else:
            ax.relim()
            ax.autoscale_view()
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
        if y_min_percentile is None:
            y_min_percentile = 0.1
        if y_max_percentile is None:
            y_max_percentile = 99.9

        max_idx_local = 0
        max_lag = 0
        max_value = 0
        
        if mean_corr_smoothed.size > 0:
            try:
                max_idx_local = np.nanargmax(mean_corr_smoothed)
                max_lag = lags_slice[max_idx_local]
                max_value = mean_corr_smoothed[max_idx_local]
            except ValueError:
                pass # Handle empty or all-NaN slice

        ax.axvline(x=max_lag, color='r', linestyle='--', linewidth=2)
        text = r'$\tau_{max}$ = ' + f'{max_lag:.2f} au'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        
        if mean_corr_slice.size > 0:
            xlim = np.nanpercentile(mean_corr_slice, y_min_percentile)
            ylim = np.nanpercentile(mean_corr_slice, y_max_percentile)
            ax.set_ylim(xlim, ylim)
        else:
            ax.autoscale()
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
    def __init__(self, **kwargs):
        # Store all arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def write_metadata(self):
        line_width = 70
        separator = '=' * line_width
        sub_separator = '-' * line_width
        
        try:
            with open(self.file_path, 'w') as fd:
                # Helper functions
                def write_section(title):
                    fd.write(f'\n{separator}\n')
                    fd.write(f'{title.upper()}\n')
                    fd.write(f'{separator}\n')
                
                def write_subsection(title):
                    fd.write(f'\n{sub_separator}\n')
                    fd.write(f'{title}\n')
                    fd.write(f'{sub_separator}\n')
                
                def write_attr(label, attr_name, indent=4):
                    val = getattr(self, attr_name, 'N/A')
                    fd.write(f'{" " * indent}{label:.<40} {val}\n')
                
                def write_value(label, value, indent=4):
                    fd.write(f'{" " * indent}{label:.<40} {value}\n')
                
                # Header
                fd.write(separator + '\n')
                fd.write('MICROLIVE METADATA FILE\n')
                fd.write(separator + '\n\n')
                
                # Author Information
                write_section('Author Information')
                try:
                    write_value('Author', str(getpass.getuser()))
                    write_value('Hostname', str(socket.gethostname()))
                except Exception:
                    pass
                write_value('Created', datetime.datetime.today().strftime('%d %b %Y'))
                write_value('Time', datetime.datetime.now().strftime('%H:%M'))
                write_value('Operating System', sys.platform)
                
                # General Information
                write_section('General Information')
                write_attr('Data Folder Path', 'data_folder_path')
                write_value('Number of Images', len(self.list_images) if self.list_images else 0)
                write_attr('Image Names', 'list_names')
                write_attr('Time Intervals', 'list_time_intervals')
                
                # Selected Image
                write_section('Selected Image')
                if self.list_names and self.selected_image_index < len(self.list_names):
                    write_value('Image Name', str(self.list_names[self.selected_image_index]))
                write_attr('Time Interval (s)', 'time_interval_value')
                write_attr('Voxel Size YX (nm)', 'voxel_yx_nm')
                write_attr('Voxel Size Z (nm)', 'voxel_z_nm')
                write_attr('Channel Names', 'channel_names')
                write_attr('Number of Channels', 'number_color_channels')
                write_attr('Bit Depth', 'bit_depth')
                write_attr('Selected Image Index', 'selected_image_index')
                if self.image_stack is not None:
                    write_value('Image Dimensions (T,Z,Y,X,C)', str(self.image_stack.shape))
                else:
                    write_value('Image Dimensions', 'None')
                
                # Segmentation / Masks
                write_section('Segmentation / Masks')
                
                # Active mask source
                active_source = getattr(self, '_active_mask_source', 'none')
                write_value('Active Mask Source', active_source)
                
                write_subsection('Watershed / Manual Segmentation')
                segmentation_mode = getattr(self, 'segmentation_mode', None)
                has_segmentation_mask = self.segmentation_mask is not None
                write_value('Segmentation Mode', segmentation_mode if segmentation_mode else 'None')
                write_value('Mask Available', 'Yes' if has_segmentation_mask else 'No')
                
                write_subsection('Cellpose Segmentation')
                has_cellpose_cyto = self.cellpose_masks_cyto is not None
                has_cellpose_nuc = self.cellpose_masks_nuc is not None
                
                if has_cellpose_cyto:
                    n_cells_cyto = int(self.cellpose_masks_cyto.max())
                    write_value('Cytosol Segmented', f'Yes ({n_cells_cyto} cells)')
                else:
                    write_value('Cytosol Segmented', 'No')
                
                if has_cellpose_nuc:
                    n_cells_nuc = int(self.cellpose_masks_nuc.max())
                    write_value('Nucleus Segmented', f'Yes ({n_cells_nuc} cells)')
                else:
                    write_value('Nucleus Segmented', 'No')
                
                # Photobleaching
                write_section('Photobleaching')
                write_attr('Correction Applied', 'photobleaching_calculated')
                write_attr('Mode', 'photobleaching_mode')
                write_attr('Radius (px)', 'photobleaching_radius')
                
                # Tracking Parameters
                write_section('Tracking Parameters')
                
                write_subsection('Spot Detection')
                write_attr('Threshold (User Selected)', 'user_selected_threshold')
                write_attr('Threshold (Calculated)', 'threshold_spot_detection')
                write_attr('Max Spots for Threshold Calc', 'max_spots_for_threshold')
                write_attr('YX Spot Size (px)', 'yx_spot_size_in_px')
                write_attr('Z Spot Size (px)', 'z_spot_size_in_px')
                write_attr('Cluster Radius (nm)', 'cluster_radius_nm')
                write_attr('Max Spots per Cluster', 'maximum_spots_cluster')
                write_attr('Separate Clusters and Spots', 'separate_clusters_and_spots')
                
                write_subsection('Trajectory Linking')
                write_attr('Min Trajectory Length', 'min_length_trajectory')
                write_attr('Max Search Range (px)', 'maximum_range_search_pixels')
                write_attr('Memory (frames)', 'memory')
                write_attr('Link Using 3D Coordinates', 'link_using_3d_coordinates')
                
                write_subsection('Channels')
                write_attr('Spot Detection Channel', 'channels_spots')
                write_attr('Cytosol Channel', 'channels_cytosol')
                write_attr('Nucleus Channel', 'channels_nucleus')
                
                write_subsection('Options')
                write_attr('Use Fixed Spot Size for Intensity', 'use_fixed_size_for_intensity_calculation')
                if self.use_maximum_projection:
                    write_value('Projection Mode', '2D Maximum Projection (Trackpy)')
                else:
                    write_value('Projection Mode', '3D (Big-FISH + Trackpy)')
                combo_val = getattr(self, 'image_source_combo', '')
                using_corrected = 'Yes' if 'Corrected' in str(combo_val) else 'No'
                write_value('Using Photobleaching Corrected Image', using_corrected)
                
                # Correlation Parameters
                write_section('Correlation Parameters')
                write_attr('Fit Type', 'correlation_fit_type')
                write_attr('Baseline Correction', 'correct_baseline')
                write_attr('Decorrelation Threshold', 'de_correlation_threshold')
                write_attr('Min Data in Trajectory (%)', 'min_percentage_data_in_trajectory')
                write_attr('Max Lag Index for Fit', 'index_max_lag_for_fit')
                write_attr('Multi-Tau', 'multi_tau')
                
                # Colocalization / ML
                write_section('Colocalization Parameters')
                write_attr('Method', 'colocalization_method')
                write_attr('Threshold Value', 'colocalization_threshold_value')
                write_attr('ML Threshold', 'ml_threshold_input')
                
                # Reproducibility
                write_section('Environment')
                write_value('Python Version', sys.version.split()[0])
                
                # Footer
                fd.write(f'\n{separator}\n')
                fd.write('END OF METADATA\n')
                fd.write(separator + '\n')

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
        self.channels_spots = [0]
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
        self.tracking_min_percentile = 0.05   # self.display_min_percentile
        self.tracking_max_percentile = 99.95  # self.display_max_percentile
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
        self.photobleaching_mode = 'entire_image'
        self.photobleaching_radius = 30
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
        self.random_mode_enabled = True
        self.segmentation_mask = None
        self._active_mask_source = 'segmentation'  # 'segmentation' or 'cellpose'
        self.total_frames = 0
        self.tracking_remove_background_checkbox = False
        self.tracking_vis_merged = False
        self.plots = Plots(self)
        self.use_multi = False
        mi.Banner().print_banner()
        self.initUI()

# =============================================================================
# =============================================================================
# MASK ACCESS PROPERTIES
# =============================================================================
# =============================================================================
    @property
    def active_mask(self):
        """
        Returns the currently active binary mask for background removal.
        Uses last generated mask (from Segmentation or Cellpose tab).
        """
        if self._active_mask_source == 'cellpose':
            if self.cellpose_masks_cyto is not None:
                return (self.cellpose_masks_cyto > 0).astype(np.uint8)
            elif self.cellpose_masks_nuc is not None:
                return (self.cellpose_masks_nuc > 0).astype(np.uint8)
        return self.segmentation_mask

    @property
    def active_labeled_mask(self):
        """
        Returns the labeled mask with cell IDs (for per-cell analysis).
        Returns None if no mask is set.
        """
        if self._active_mask_source == 'cellpose':
            if self.cellpose_masks_cyto is not None:
                return self.cellpose_masks_cyto
            elif self.cellpose_masks_nuc is not None:
                return self.cellpose_masks_nuc
        return self.segmentation_mask

    def _get_tracking_masks(self):
        """
        Prepares masks for tracking based on available segmentation data.
        
        Returns
        -------
        tuple: (masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei)
            - masks_complete_cells: Labeled cytosol masks (or nuclei if no cytosol)
            - masks_nuclei: Labeled nucleus masks (or None)
            - masks_cytosol_no_nuclei: Cytosol with overlapping nucleus regions removed (or None)
        """
        if self._active_mask_source == 'cellpose':
            masks_cyto = self.cellpose_masks_cyto  # labeled [Y,X] or None
            masks_nuc = self.cellpose_masks_nuc    # labeled [Y,X] or None
            
            if masks_cyto is not None and masks_nuc is not None:
                # Both exist: compute cytosol-only (cytosol minus overlapping nucleus)
                # Only remove nucleus pixels that are INSIDE the cytosol
                masks_cytosol_no_nuclei = masks_cyto.copy()
                # Zero out pixels where nucleus exists AND cytosol exists
                overlap_mask = (masks_nuc > 0) & (masks_cyto > 0)
                masks_cytosol_no_nuclei[overlap_mask] = 0
                return masks_cyto, masks_nuc, masks_cytosol_no_nuclei
            elif masks_cyto is not None:
                # Only cytosol - cytosol is the "complete cell", no nucleus to subtract
                return masks_cyto, None, None
            elif masks_nuc is not None:
                # Only nuclei - nuclei serve as the "complete cell"
                return masks_nuc, masks_nuc, None
        
        # Fallback: segmentation mask (binary)
        if self.segmentation_mask is not None:
            return self.segmentation_mask, None, None
        
        # No masks at all
        return None, None, None

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
        self.cellpose_tab = QWidget()
        self.tabs.addTab(self.cellpose_tab, "Cellpose")
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
        self.setup_cellpose_tab()
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
        minSlider.setMinimum(0); minSlider.setMaximum(95)
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
                    1e-6,
                    1e6,
                    6
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

    def set_display_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable the Display tab’s time slider and Play button."""
        if hasattr(self, 'time_slider_display') and self.time_slider_display is not None:
            self.time_slider_display.setEnabled(enabled)
        if hasattr(self, 'play_button_display') and self.play_button_display is not None:
            self.play_button_display.setEnabled(enabled)

    def convert_to_standard_format(self, image_stack):
        """
        Convert the loaded image_stack to standard 5D format [T, Z, Y, X, C].
        If image does not have 5 dimensions, prompt user to map file dimensions to standard and indicate missing dimensions.
        """
        if image_stack.ndim == 5:
            return image_stack
        mapping = self.open_dimension_mapping_dialog(image_stack.shape)
        if mapping is None:
            # User cancelled; return None to indicate cancellation
            return None
        used_axes = [m for m in mapping if m is not None]
        # Validate mapping indices within bounds
        if any(m < 0 or m >= image_stack.ndim for m in used_axes):
            QMessageBox.critical(self, "Error", f"Mapping indices {used_axes} are not valid for an image with {image_stack.ndim} dimensions.")
            return None
        if used_axes:
            try:
                # Rearrange image so used axes appear in selected order
                transposed = np.transpose(image_stack, used_axes)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error transposing image: {e}")
                return None
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
                    return None
                new_shape.append(used_shape.pop(0))
        try:
            final_array = np.reshape(transposed, new_shape)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reshaping image to standard format: {e}")
            return None
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
            if first_path.lower().endswith('.lif'):
                self.load_lif_image(first_path, 0)
            else:
                pass
        self.image_tree.expandAll()

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
        
        # Reset segmentation masks when loading a new image
        if info.get('tif') or info.get('index') is not None:
             self.cellpose_masks_cyto = None
             self.cellpose_masks_nuc = None
             if hasattr(self, 'manual_segmentation_mask'):
                 del self.manual_segmentation_mask
        
        self.plot_image()
        self.plot_tracking()
        self.reset_tracking_visualization_tab()

    def _setup_image_ui(self, T, C):
        """
        Shared setup logic after loading a new image.
        Called by both load_tif_image() and load_lif_image() to set up UI elements.
        
        Args:
            T: Number of time frames
            C: Number of channels
        """
        # Reset all tabs and state for new data
        self.reset_all_state()
        
        # Initialize frame counts
        self.total_frames = T
        self.max_lag = T - 1
        if hasattr(self, 'max_lag_input'):
            self.max_lag_input.setMaximum(self.max_lag - 1)
            self.max_lag_input.setValue(self.max_lag - 1)
        
        # Set time slider maximums for all tabs
        self.time_slider_display.setMaximum(T - 1)
        self.time_slider_display.setValue(0)
        self.time_slider_tracking.setMaximum(T - 1)
        self.time_slider_tracking.setValue(0)
        self.time_slider_tracking_vis.setMaximum(T - 1)
        self.time_slider_tracking_vis.setValue(0)
        self.segmentation_time_slider.setMaximum(T - 1)
        if hasattr(self, 'time_slider_cellpose'):
            self.time_slider_cellpose.setMaximum(T - 1)
            self.time_slider_cellpose.setValue(0)
            self.cellpose_current_frame = 0
        
        # Reset TYX mask spinbox and validate against image timepoints
        if hasattr(self, 'max_timepoints_spinbox'):
            self.max_timepoints_spinbox.setMaximum(T)
            # Set default to min(5, T) for meaningful TYX sampling
            self.max_timepoints_spinbox.setValue(min(5, T))
        if hasattr(self, 'chk_calculate_masks_over_time'):
            self.chk_calculate_masks_over_time.setChecked(False)
        if hasattr(self, 'minimal_frames_spinbox'):
            self.minimal_frames_spinbox.setMaximum(T)
            self.minimal_frames_spinbox.setValue(min(2, T))  # Reset to default, capped at T
        
        # Enable display controls
        self.set_display_controls_enabled(True)
        self.playing = False
        self.play_button_display.setText("Play")
        
        # Create channel buttons for all tabs
        self.create_channel_buttons()
        self.create_cellpose_channel_buttons()
        self.create_segmentation_channel_buttons()
        self.create_correlation_channel_checkboxes()
        self.populate_colocalization_channels()
        
        # Update Cellpose channel spinbox ranges based on actual channels
        max_ch = max(0, C - 1)
        if hasattr(self, 'cellpose_cyto_channel_input'):
            self.cellpose_cyto_channel_input.setMaximum(max_ch)
            # Set to channel 1 if available, otherwise channel 0
            self.cellpose_cyto_channel_input.setValue(min(1, max_ch))
        if hasattr(self, 'cellpose_nuc_channel_input'):
            self.cellpose_nuc_channel_input.setMaximum(max_ch)
            self.cellpose_nuc_channel_input.setValue(0)
        
        # Create crops channel buttons
        for btn in getattr(self, 'channel_buttons_crops', []):
            btn.setParent(None)
        self.channel_buttons_crops = []
        for idx in range(C):
            button = QPushButton(f"Channel {idx}", self)
            button.clicked.connect(partial(self.update_channel_crops, idx))
            self.channel_buttons_layout_crops.addWidget(button)
            self.channel_buttons_crops.append(button)
        
        # Setup channel visualization control tabs
        self.channelControlsTabs.clear()
        for ch in range(C):
            init_params = self.channelDisplayParams.get(ch, {
                'min_percentile': self.display_min_percentile,
                'max_percentile': self.display_max_percentile,
                'sigma': self.display_sigma,
                'low_sigma': self.low_display_sigma
            })
            widget = self.create_channel_visualization_controls(ch, init_params)
            self.channelControlsTabs.addTab(widget, f"Ch {ch}")
        
        # Populate channel combo boxes
        self.intensity_channel_combo.clear()
        for ch in range(self.number_color_channels):
            self.intensity_channel_combo.addItem(str(ch), ch)
        self.intensity_channel_combo.setCurrentIndex(0)
        
        self.time_course_channel_combo.clear()
        for ch in range(self.number_color_channels):
            self.time_course_channel_combo.addItem(str(ch), ch)
        self.time_course_channel_combo.addItem("All")
        self.time_course_channel_combo.setCurrentIndex(0)
        
        # Update tracking sliders if needed
        if hasattr(self, 'min_percentile_spinbox_tracking'):
            self.update_tracking_sliders()
        
        # Stop playback if running
        if self.playing:
            self.play_pause()
        
        # Plot first frame
        self.plot_image()
        self.plot_tracking()

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
        if self.image_stack is None:
            return
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
        # Setup UI for the new image
        self._setup_image_ui(T, C)


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
        # Setup UI for the new image
        T = self.image_stack.shape[0]
        C = self.number_color_channels
        self._setup_image_ui(T, C)

    def play_pause(self):
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_button_display.setText("Play")
            self.play_button_tracking.setText("Play")
            self.play_button_tracking_vis.setText("Play")
            if hasattr(self, 'play_button_cellpose'):
                self.play_button_cellpose.setText("Play")
        else:
            interval = 16 if sys.platform.startswith('win') else 100
            self.timer.start(interval)
            self.playing = True
            self.play_button_display.setText("Pause")
            self.play_button_tracking.setText("Pause")
            self.play_button_tracking_vis.setText("Pause")
            if hasattr(self, 'play_button_cellpose'):
                self.play_button_cellpose.setText("Pause")

    def update_channel(self, channel):
        self.current_channel = channel
        self._sync_tracking_channel()
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
        # Sync Cellpose time slider and update TYX masks
        if hasattr(self, 'time_slider_cellpose'):
            if self.time_slider_cellpose.value() != value:
                self.time_slider_cellpose.blockSignals(True)
                self.time_slider_cellpose.setValue(value)
                self.time_slider_cellpose.blockSignals(False)
            # Always update Cellpose frame to sync TYX masks
            self.update_cellpose_frame(value)
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
                        if self.display_remove_background_checkbox.isChecked() and self.active_mask is not None:
                            mask = (self.active_mask > 0).astype(float)
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
                    if self.display_remove_background_checkbox.isChecked() and self.active_mask is not None:
                        mask = (self.active_mask > 0).astype(float)
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
                if self.display_remove_background_checkbox.isChecked() and self.active_mask is not None:
                    mask = (self.active_mask > 0).astype(float)
                    img_to_show = img_to_show * mask
                cmap_imagej = cmap_list_imagej[self.current_channel % len(cmap_list_imagej)]
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
       
        if hasattr(self, 'data_folder_path') and str(self.data_folder_path) == file_path:
            # Clear core data specific to closing a file
            self.image_stack = None
            self.data_folder_path = None
            self.colocalization_results = None
            self.current_total_plots = None
            
            # Use unified reset for all tabs and state
            self.reset_all_state()
            
            # Clear info labels (close-specific: show empty state)
            for lbl in (self.file_label, self.frames_label, self.z_scales_label, 
                    self.y_pixels_label, self.x_pixels_label, self.channels_label, 
                    self.voxel_yx_size_label, self.voxel_z_nm_label, 
                    self.bit_depth_label, self.time_interval_label):
                if hasattr(self, lbl.objectName()) if hasattr(lbl, 'objectName') else True:
                    lbl.setText("")
            
            # Clear additional info labels if they exist
            if hasattr(self, 'laser_lines_label'):
                self.laser_lines_label.setText("")
            if hasattr(self, 'intensities_label'):
                self.intensities_label.setText("")
            if hasattr(self, 'wave_ranges_label'):
                self.wave_ranges_label.setText("")
            
            # Clear channel controls (close-specific: remove channel UI)
            if hasattr(self, 'channelControlsTabs'):
                self.channelControlsTabs.clear()
            
            # Clear channel buttons (close-specific: remove buttons)
            for btn_list in [getattr(self, 'channel_buttons_display', []),
                            getattr(self, 'channel_buttons_tracking', []),
                            getattr(self, 'channel_buttons_tracking_vis', []),
                            getattr(self, 'channel_buttons_crops', []),
                            getattr(self, 'segmentation_channel_buttons', [])]:
                for btn in btn_list:
                    if btn:
                        btn.setParent(None)
            
            # Reset button lists
            self.channel_buttons_display = []
            self.channel_buttons_tracking = []
            if hasattr(self, 'channel_buttons_tracking_vis'):
                self.channel_buttons_tracking_vis = []
            if hasattr(self, 'channel_buttons_crops'):
                self.channel_buttons_crops = []
            if hasattr(self, 'segmentation_channel_buttons'):
                self.segmentation_channel_buttons = []
            
            # Clear channel checkboxes for correlation
            if hasattr(self, 'channel_checkboxes'):
                for cb in self.channel_checkboxes:
                    if cb:
                        cb.setParent(None)
                self.channel_checkboxes = []
            
            # Clear combo boxes
            if hasattr(self, 'intensity_channel_combo'):
                self.intensity_channel_combo.clear()
            if hasattr(self, 'time_course_channel_combo'):
                self.time_course_channel_combo.clear()
            if hasattr(self, 'channel_combo_box_1'):
                self.channel_combo_box_1.clear()
            if hasattr(self, 'channel_combo_box_2'):
                self.channel_combo_box_2.clear()
            
            # Disable controls (close-specific: no image loaded)
            if hasattr(self, 'time_slider_display'):
                self.time_slider_display.setEnabled(False)
                self.time_slider_display.setValue(0)
            if hasattr(self, 'play_button_display'):
                self.play_button_display.setEnabled(False)
            if hasattr(self, 'time_slider_tracking'):
                self.time_slider_tracking.setValue(0)
            if hasattr(self, 'time_slider_tracking_vis'):
                self.time_slider_tracking_vis.setValue(0)
            
            # Stop any playing timers
            if hasattr(self, 'playing') and self.playing:
                self.play_pause()
            
            # Reset current indices
            self.current_frame = 0
            self.current_channel = 0

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
        if self.display_remove_background_checkbox.isChecked() and self.active_mask is not None:
            mask = (self.active_mask > 0).astype(float)
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
            QCheckBox#themeToggle {
                spacing: 5px;
            }
            QCheckBox#themeToggle::indicator {
                width: 40px; height: 20px;
                border-radius: 10px;
                background-color: #bbb;
            }
            QCheckBox#themeToggle::indicator:checked {
                background-color: #007acc;
            }
            QCheckBox#themeToggle::indicator:unchecked {
                background-color: #bbb;
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
            self._active_mask_source = 'segmentation'
            self.ax_segmentation.clear()
            cmap_imagej = cmap_list_imagej[ch % len(cmap_list_imagej)]
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
        self.cellpose_current_frame = self.current_frame  # Sync Cellpose frame
        for slider in (self.time_slider_display, self.time_slider_tracking, 
                       getattr(self, 'time_slider_tracking_vis', None),
                       getattr(self, 'time_slider_cellpose', None)):
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
        elif current_tab == self.tabs.indexOf(self.cellpose_tab):
            # Sync TYX masks if active before plotting
            if getattr(self, 'use_tyx_masks', False):
                if getattr(self, 'cellpose_masks_cyto_tyx', None) is not None:
                    self.cellpose_masks_cyto = self.cellpose_masks_cyto_tyx[self.current_frame]
                if getattr(self, 'cellpose_masks_nuc_tyx', None) is not None:
                    self.cellpose_masks_nuc = self.cellpose_masks_nuc_tyx[self.current_frame]
            self.plot_cellpose_results()

    def setup_cellpose_tab(self):
        """
        Setup the Cellpose tab with a left panel for image display and a right panel for controls.
        """
        layout = QHBoxLayout(self.cellpose_tab)

        # --- Left Panel: Image Display ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Figure Canvas
        self.figure_cellpose = Figure()
        self.canvas_cellpose = FigureCanvas(self.figure_cellpose)
        self.ax_cellpose = self.figure_cellpose.add_subplot(111)
        #self.ax_cellpose.axis('off')
        # grid off
        self.ax_cellpose.grid(False)
        left_layout.addWidget(self.canvas_cellpose)

        # Navigation Controls (Time Slider & Channel Buttons)
        nav_layout = QVBoxLayout()
        
        # Time Slider
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time:"))
        self.time_slider_cellpose = QSlider(Qt.Horizontal)
        self.time_slider_cellpose.valueChanged.connect(self.update_cellpose_frame)
        time_layout.addWidget(self.time_slider_cellpose)
        
        # Play Button
        self.play_button_cellpose = QPushButton("Play", self)
        self.play_button_cellpose.clicked.connect(self.play_pause)
        time_layout.addWidget(self.play_button_cellpose)
        
        nav_layout.addLayout(time_layout)

        # Channel Buttons
        self.cellpose_channel_buttons_layout = QHBoxLayout()
        nav_layout.addLayout(self.cellpose_channel_buttons_layout)
        self.cellpose_channel_buttons = []

        left_layout.addLayout(nav_layout)
        layout.addWidget(left_panel, stretch=2)

        # --- Right Panel: Controls ---
        right_panel = QScrollArea()
        right_panel.setWidgetResizable(True)
        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        
        # Cytosol Segmentation Group
        cyto_group = QGroupBox("Cytosol Segmentation")
        cyto_layout = QFormLayout()
        
        self.cellpose_cyto_model_input = QComboBox()
        self.cellpose_cyto_model_input.addItems(['cyto3', 'cyto2', 'cyto', ])
        self.cellpose_cyto_model_input.setCurrentText('cyto3')
        cyto_layout.addRow("Model:", self.cellpose_cyto_model_input)

        self.cellpose_cyto_channel_input = QSpinBox()
        self.cellpose_cyto_channel_input.setRange(0, 10)
        self.cellpose_cyto_channel_input.setValue(0) # Default to channel 0 (updated dynamically)
        cyto_layout.addRow("Channel:", self.cellpose_cyto_channel_input)
        
        self.cellpose_cyto_diameter_input = QDoubleSpinBox()
        self.cellpose_cyto_diameter_input.setRange(0, 1000)
        self.cellpose_cyto_diameter_input.setValue(150)
        cyto_layout.addRow("Diameter (px):", self.cellpose_cyto_diameter_input)

        self.cellpose_cyto_flow_input = QDoubleSpinBox()
        self.cellpose_cyto_flow_input.setRange(0, 1)
        self.cellpose_cyto_flow_input.setSingleStep(0.1)
        self.cellpose_cyto_flow_input.setValue(0.4)
        cyto_layout.addRow("Flow Threshold:", self.cellpose_cyto_flow_input)
        
        self.chk_optimize_cyto = QCheckBox("Optimize Parameters")
        self.chk_optimize_cyto.setChecked(False)
        cyto_layout.addRow(self.chk_optimize_cyto)
        
        self.btn_run_cyto = QPushButton("Segment Cytosol")
        self.btn_run_cyto.clicked.connect(self.run_cellpose_cyto)
        cyto_layout.addRow(self.btn_run_cyto)
        
        cyto_group.setLayout(cyto_layout)
        right_layout.addWidget(cyto_group)

        # Nucleus Segmentation Group
        nuc_group = QGroupBox("Nucleus Segmentation")
        nuc_layout = QFormLayout()
        
        self.cellpose_nuc_model_input = QComboBox()
        self.cellpose_nuc_model_input.addItems(['nuclei', 'cyto3', 'cyto2', 'cyto'])
        self.cellpose_nuc_model_input.setCurrentText('nuclei')
        nuc_layout.addRow("Model:", self.cellpose_nuc_model_input)

        self.cellpose_nuc_channel_input = QSpinBox()
        self.cellpose_nuc_channel_input.setRange(0, 10)
        self.cellpose_nuc_channel_input.setValue(0) # Default to channel 0
        nuc_layout.addRow("Channel:", self.cellpose_nuc_channel_input)
        
        self.cellpose_nuc_diameter_input = QDoubleSpinBox()
        self.cellpose_nuc_diameter_input.setRange(0, 1000)
        self.cellpose_nuc_diameter_input.setValue(60)
        nuc_layout.addRow("Diameter (px):", self.cellpose_nuc_diameter_input)

        self.cellpose_nuc_flow_input = QDoubleSpinBox()
        self.cellpose_nuc_flow_input.setRange(0, 1)
        self.cellpose_nuc_flow_input.setSingleStep(0.1)
        self.cellpose_nuc_flow_input.setValue(0.4)
        nuc_layout.addRow("Flow Threshold:", self.cellpose_nuc_flow_input)
        
        self.chk_optimize_nuc = QCheckBox("Optimize Parameters")
        self.chk_optimize_nuc.setChecked(False)
        nuc_layout.addRow(self.chk_optimize_nuc)
        
        self.btn_run_nuc = QPushButton("Segment Nucleus")
        self.btn_run_nuc.clicked.connect(self.run_cellpose_nuc)
        nuc_layout.addRow(self.btn_run_nuc)
        
        nuc_group.setLayout(nuc_layout)
        right_layout.addWidget(nuc_group)

        # Time-Varying Masks Group (TYX)
        tyx_group = QGroupBox("Time-Varying Masks (TYX)")
        tyx_layout = QFormLayout()
        
        self.chk_calculate_masks_over_time = QCheckBox("Calculate Masks Over Time")
        self.chk_calculate_masks_over_time.setChecked(False)
        self.chk_calculate_masks_over_time.setToolTip(
            "When enabled, Cellpose will calculate masks at multiple timepoints "
            "and track cell IDs across time using IoU-based linking."
        )
        tyx_layout.addRow(self.chk_calculate_masks_over_time)
        
        self.max_timepoints_spinbox = QSpinBox()
        self.max_timepoints_spinbox.setRange(1, 1000)
        self.max_timepoints_spinbox.setValue(5)  # Default 5 timepoints for meaningful TYX
        self.max_timepoints_spinbox.setToolTip(
            "Maximum number of timepoints to sample for mask calculation. "
            "Intermediate frames use nearest sampled mask."
        )
        tyx_layout.addRow("Max Timepoints:", self.max_timepoints_spinbox)
        
        self.linking_memory_spinbox = QSpinBox()
        self.linking_memory_spinbox.setRange(0, 100)
        self.linking_memory_spinbox.setValue(1)
        self.linking_memory_spinbox.setToolTip(
            "Number of frames a cell can disappear before being assigned a new ID. "
            "Helps track cells that temporarily leave the field of view."
        )
        tyx_layout.addRow("Linking Memory:", self.linking_memory_spinbox)
        
        self.minimal_frames_spinbox = QSpinBox()
        self.minimal_frames_spinbox.setRange(1, 1000)
        self.minimal_frames_spinbox.setValue(2)  # Default: cells must exist for at least 2 frames
        self.minimal_frames_spinbox.setToolTip(
            "Minimum number of frames a cell must exist to be kept. "
            "Cells appearing for fewer frames are removed as artifacts."
        )
        tyx_layout.addRow("Minimal Frames:", self.minimal_frames_spinbox)
        
        tyx_group.setLayout(tyx_layout)
        right_layout.addWidget(tyx_group)

        # Improve Segmentation Group
        improve_group = QGroupBox("Improve Segmentation")
        improve_layout = QFormLayout()

        self.chk_remove_border_cells = QCheckBox("Remove cells touching border")
        self.chk_remove_border_cells.setChecked(False)
        self.chk_remove_border_cells.stateChanged.connect(self.on_remove_border_cells_changed)
        improve_layout.addRow(self.chk_remove_border_cells)

        self.chk_remove_unpaired_cells = QCheckBox("Remove unpaired cells")
        self.chk_remove_unpaired_cells.setChecked(False)
        self.chk_remove_unpaired_cells.stateChanged.connect(self.on_remove_unpaired_cells_changed)
        improve_layout.addRow(self.chk_remove_unpaired_cells)

        improve_group.setLayout(improve_layout)
        right_layout.addWidget(improve_group)

        # Clear Button
        self.btn_clear_cellpose = QPushButton("Clear Masks & IDs")
        self.btn_clear_cellpose.clicked.connect(self.clear_cellpose_masks)
        right_layout.addWidget(self.btn_clear_cellpose)

        right_layout.addStretch()
        right_panel.setWidget(right_content)
        layout.addWidget(right_panel, stretch=1)

        # Initialize state variables for Cellpose tab
        self.cellpose_masks_cyto = None
        self.cellpose_masks_nuc = None
        self.cellpose_current_channel = 0
        self.cellpose_current_frame = 0
        # TYX mask state variables
        self.cellpose_masks_cyto_tyx = None  # [T, Y, X] labeled masks
        self.cellpose_masks_nuc_tyx = None   # [T, Y, X] labeled masks
        self.use_tyx_masks = False           # Flag: are TYX masks active?

    def create_cellpose_channel_buttons(self):
        # Clear existing buttons
        for btn in self.cellpose_channel_buttons:
            btn.setParent(None)
        self.cellpose_channel_buttons = []
        
        # Create new buttons based on loaded channels
        if self.channel_names:
            for idx, name in enumerate(self.channel_names):
                btn = QPushButton(f"Channel {idx}")
                btn.clicked.connect(partial(self.update_cellpose_channel, idx))
                self.cellpose_channel_buttons_layout.addWidget(btn)
                self.cellpose_channel_buttons.append(btn)

    def update_cellpose_frame(self, value):
        self.cellpose_current_frame = value
        # Sync YX masks from TYX when TYX masks are active
        if getattr(self, 'use_tyx_masks', False):
            if getattr(self, 'cellpose_masks_cyto_tyx', None) is not None:
                self.cellpose_masks_cyto = self.cellpose_masks_cyto_tyx[value]
            if getattr(self, 'cellpose_masks_nuc_tyx', None) is not None:
                self.cellpose_masks_nuc = self.cellpose_masks_nuc_tyx[value]
        self.plot_cellpose_results()

    def update_cellpose_channel(self, channel_index):
        self.cellpose_current_channel = channel_index
        self.plot_cellpose_results()

    def run_cellpose_cyto(self):
        if self.image_stack is None:
            return
        
        try:
            # Get parameters
            channel = self.cellpose_cyto_channel_input.value()
            diameter = int(self.cellpose_cyto_diameter_input.value())
            model_name = self.cellpose_cyto_model_input.currentText()
            
            # Check if TYX masks are requested
            if (self.chk_calculate_masks_over_time.isChecked() and 
                self.image_stack.ndim == 5 and 
                self.image_stack.shape[0] > 1):
                
                max_tp = min(self.max_timepoints_spinbox.value(), self.image_stack.shape[0])
                linking_memory = self.linking_memory_spinbox.value()
                
                # Create progress dialog
                progress = QProgressDialog("Calculating TYX cytosol masks...", "Cancel", 0, max_tp, self)
                progress.setWindowTitle("Cellpose Segmentation")
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.show()
                QApplication.processEvents()
                
                # Progress callback for CellposeTimeSeries
                def progress_callback(msg):
                    # Extract frame number from message like "Calculating cytosol masks: frame 3/10 (2/5)"
                    if progress.wasCanceled():
                        return  # User cancelled
                    try:
                        parts = msg.split("(")[1].split(")")[0].split("/")
                        current = int(parts[0])
                        progress.setValue(current)
                        progress.setLabelText(msg)
                    except:
                        progress.setLabelText(msg)
                    QApplication.processEvents()
                
                tyx_generator = mi.CellposeTimeSeries(
                    image=self.image_stack,
                    channels_cytosol=channel,
                    channels_nucleus=None,
                    diameter_cytosol=diameter,
                    diameter_nucleus=60,
                    max_timepoints=max_tp,
                    linking_memory=linking_memory,
                    model_type_cyto=model_name,
                    progress_callback=progress_callback
                )
                
                masks_cyto_tyx, _ = tyx_generator.calculate_tyx_masks()
                progress.close()
                
                if masks_cyto_tyx is not None:
                    # Filter short-lived masks (artifacts) and reindex IDs
                    min_frames = self.minimal_frames_spinbox.value()
                    masks_cyto_tyx = mi.CellposeTimeSeries.filter_short_lived_masks(masks_cyto_tyx, min_frames)
                    
                    self.cellpose_masks_cyto_tyx = masks_cyto_tyx
                    # Also set the current frame's YX mask for compatibility
                    self.cellpose_masks_cyto = masks_cyto_tyx[self.cellpose_current_frame]
                    self.use_tyx_masks = True
                else:
                    self.use_tyx_masks = False
                    
                self.statusBar().showMessage(f"TYX cytosol masks calculated: {max_tp} timepoints")
            else:
                # Standard YX mask (existing behavior)
                if self.image_stack.ndim == 5:
                    img = self.image_stack[self.cellpose_current_frame, :, :, :, :]
                else:
                    img = self.image_stack
                    
                segmenter = mi.CellSegmentation(
                    img,
                    channels_cytosol=[channel],
                    channels_nucleus=None,
                    diameter_cytosol=diameter,
                    selection_metric='max_cells_and_area' if self.chk_optimize_cyto.isChecked() else None,
                    show_plot=False,
                    model_cyto_segmentation=model_name
                )
                
                masks_cyto, _, _ = segmenter.calculate_masks()
                
                self.cellpose_masks_cyto = masks_cyto
                self.cellpose_masks_cyto_tyx = None
                self.use_tyx_masks = False
            
            self._active_mask_source = 'cellpose'
            # Clear watershed mask since we're using Cellpose now
            self.segmentation_mask = None
            self.synchronize_and_plot_cellpose()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Cytosol segmentation failed: {str(e)}")

    def run_cellpose_nuc(self):
        if self.image_stack is None:
            return
            
        try:
            # Get parameters
            channel = self.cellpose_nuc_channel_input.value()
            diameter = int(self.cellpose_nuc_diameter_input.value())
            model_name = self.cellpose_nuc_model_input.currentText()
            
            # Check if TYX masks are requested
            if (self.chk_calculate_masks_over_time.isChecked() and 
                self.image_stack.ndim == 5 and 
                self.image_stack.shape[0] > 1):
                
                max_tp = min(self.max_timepoints_spinbox.value(), self.image_stack.shape[0])
                linking_memory = self.linking_memory_spinbox.value()
                
                # Create progress dialog
                progress = QProgressDialog("Calculating TYX nucleus masks...", "Cancel", 0, max_tp, self)
                progress.setWindowTitle("Cellpose Segmentation")
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.show()
                QApplication.processEvents()
                
                # Progress callback for CellposeTimeSeries
                def progress_callback(msg):
                    if progress.wasCanceled():
                        return
                    try:
                        parts = msg.split("(")[1].split(")")[0].split("/")
                        current = int(parts[0])
                        progress.setValue(current)
                        progress.setLabelText(msg)
                    except:
                        progress.setLabelText(msg)
                    QApplication.processEvents()
                
                tyx_generator = mi.CellposeTimeSeries(
                    image=self.image_stack,
                    channels_cytosol=None,
                    channels_nucleus=channel,
                    diameter_cytosol=150,
                    diameter_nucleus=diameter,
                    max_timepoints=max_tp,
                    linking_memory=linking_memory,
                    model_type_nuc=model_name,
                    progress_callback=progress_callback
                )
                
                _, masks_nuc_tyx = tyx_generator.calculate_tyx_masks()
                progress.close()
                
                if masks_nuc_tyx is not None:
                    # Filter short-lived masks (artifacts) and reindex IDs
                    min_frames = self.minimal_frames_spinbox.value()
                    masks_nuc_tyx = mi.CellposeTimeSeries.filter_short_lived_masks(masks_nuc_tyx, min_frames)
                    
                    self.cellpose_masks_nuc_tyx = masks_nuc_tyx
                    # Also set the current frame's YX mask for compatibility
                    self.cellpose_masks_nuc = masks_nuc_tyx[self.cellpose_current_frame]
                    self.use_tyx_masks = True
                else:
                    self.use_tyx_masks = False
                    
                self.statusBar().showMessage(f"TYX nucleus masks calculated: {max_tp} timepoints")
            else:
                # Standard YX mask (existing behavior)
                if self.image_stack.ndim == 5:
                    img = self.image_stack[self.cellpose_current_frame, :, :, :, :]
                else:
                    img = self.image_stack
                    
                segmenter = mi.CellSegmentation(
                    img,
                    channels_cytosol=None,
                    channels_nucleus=[channel],
                    diameter_nucleus=diameter,
                    selection_metric='max_cells_and_area' if self.chk_optimize_nuc.isChecked() else None,
                    show_plot=False,
                    model_nuc_segmentation=model_name
                )
                
                _, masks_nuc, _ = segmenter.calculate_masks()
                
                self.cellpose_masks_nuc = masks_nuc
                self.cellpose_masks_nuc_tyx = None
                self.use_tyx_masks = False
            
            self._active_mask_source = 'cellpose'
            # Clear watershed mask since we're using Cellpose now
            self.segmentation_mask = None
            self.synchronize_and_plot_cellpose()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Nucleus segmentation failed: {str(e)}")

    def synchronize_and_plot_cellpose(self):
        # Synchronize if both exist
        if self.cellpose_masks_cyto is not None and self.cellpose_masks_nuc is not None:
             self.cellpose_masks_cyto, self.cellpose_masks_nuc = mi.CellSegmentation.synchronize_masks(
                 self.cellpose_masks_cyto, self.cellpose_masks_nuc
             )
        
        self.plot_cellpose_results()
        # Reset dependent tabs since masks changed
        self.reset_photobleaching_tab()
        self.reset_tracking_tab()

    def clear_cellpose_masks(self):
        self.cellpose_masks_cyto = None
        self.cellpose_masks_nuc = None
        # Clear TYX masks too
        self.cellpose_masks_cyto_tyx = None
        self.cellpose_masks_nuc_tyx = None
        self.use_tyx_masks = False
        self.plot_cellpose_results()

    def on_remove_border_cells_changed(self, state):
        """Handle checkbox state change for removing border-touching cells.
        
        For TYX masks: If a cell touches the border in ANY frame, it is removed
        from ALL frames to ensure consistent tracking.
        """
        if state == Qt.Checked:
            # Collect border-touching labels across ALL timepoints
            border_labels = set()
            
            # Check TYX masks first (if they exist)
            if getattr(self, 'use_tyx_masks', False):
                # Scan all timepoints in TYX arrays for border-touching cells
                if self.cellpose_masks_cyto_tyx is not None:
                    for t in range(self.cellpose_masks_cyto_tyx.shape[0]):
                        border_labels.update(self.get_border_touching_labels(self.cellpose_masks_cyto_tyx[t]))
                if self.cellpose_masks_nuc_tyx is not None:
                    for t in range(self.cellpose_masks_nuc_tyx.shape[0]):
                        border_labels.update(self.get_border_touching_labels(self.cellpose_masks_nuc_tyx[t]))
                
                # Remove labels from ALL timepoints in TYX arrays
                if self.cellpose_masks_cyto_tyx is not None and border_labels:
                    self.cellpose_masks_cyto_tyx = self._remove_labels_from_tyx(self.cellpose_masks_cyto_tyx, border_labels)
                    # Update current frame YX mask
                    self.cellpose_masks_cyto = self.cellpose_masks_cyto_tyx[self.cellpose_current_frame]
                if self.cellpose_masks_nuc_tyx is not None and border_labels:
                    self.cellpose_masks_nuc_tyx = self._remove_labels_from_tyx(self.cellpose_masks_nuc_tyx, border_labels)
                    # Update current frame YX mask
                    self.cellpose_masks_nuc = self.cellpose_masks_nuc_tyx[self.cellpose_current_frame]
            else:
                # Standard YX mask handling (non-TYX mode)
                if self.cellpose_masks_cyto is not None:
                    border_labels.update(self.get_border_touching_labels(self.cellpose_masks_cyto))
                if self.cellpose_masks_nuc is not None:
                    border_labels.update(self.get_border_touching_labels(self.cellpose_masks_nuc))
                
                # Remove those labels from BOTH masks
                if self.cellpose_masks_cyto is not None:
                    self.cellpose_masks_cyto = self.remove_labels_and_reindex(self.cellpose_masks_cyto, border_labels)
                if self.cellpose_masks_nuc is not None:
                    self.cellpose_masks_nuc = self.remove_labels_and_reindex(self.cellpose_masks_nuc, border_labels)
        self.plot_cellpose_results()
    
    def _remove_labels_from_tyx(self, masks_tyx, labels_to_remove):
        """Remove specified labels from TYX mask array and reindex IDs."""
        if masks_tyx is None or not labels_to_remove:
            return masks_tyx
        
        # Find all remaining IDs
        all_ids = set(np.unique(masks_tyx))
        all_ids.discard(0)
        remaining_ids = all_ids - labels_to_remove
        
        # Create new TYX array with reindexed IDs
        new_masks = np.zeros_like(masks_tyx)
        new_id = 1
        for old_id in sorted(remaining_ids):
            new_masks[masks_tyx == old_id] = new_id
            new_id += 1
        
        return new_masks

    def on_remove_unpaired_cells_changed(self, state):
        """Handle checkbox state change for removing unpaired cells."""
        if state == Qt.Checked:
            # Only works if both cytosol and nucleus have been segmented
            if self.cellpose_masks_cyto is None or self.cellpose_masks_nuc is None:
                QMessageBox.warning(self, "Warning", 
                    "Please segment both cytosol and nucleus first.")
                self.chk_remove_unpaired_cells.blockSignals(True)
                self.chk_remove_unpaired_cells.setChecked(False)
                self.chk_remove_unpaired_cells.blockSignals(False)
                return
            
            # Find IDs present in both masks
            cyto_ids = set(np.unique(self.cellpose_masks_cyto))
            nuc_ids = set(np.unique(self.cellpose_masks_nuc))
            cyto_ids.discard(0)
            nuc_ids.discard(0)
            
            # Paired IDs are those present in both masks
            paired_ids = cyto_ids & nuc_ids
            
            # Remove unpaired cytosols (IDs only in cyto, not in nuc)
            unpaired_cyto_ids = cyto_ids - paired_ids
            if unpaired_cyto_ids:
                self.cellpose_masks_cyto = self.remove_labels_and_reindex(
                    self.cellpose_masks_cyto, unpaired_cyto_ids)
            
            # Remove unpaired nuclei (IDs only in nuc, not in cyto)
            unpaired_nuc_ids = nuc_ids - paired_ids
            if unpaired_nuc_ids:
                self.cellpose_masks_nuc = self.remove_labels_and_reindex(
                    self.cellpose_masks_nuc, unpaired_nuc_ids)
            
            # Re-synchronize to ensure IDs match after reindexing
            if self.cellpose_masks_cyto is not None and self.cellpose_masks_nuc is not None:
                self.cellpose_masks_cyto, self.cellpose_masks_nuc = mi.CellSegmentation.synchronize_masks(
                    self.cellpose_masks_cyto, self.cellpose_masks_nuc
                )
        
        self.plot_cellpose_results()

    def get_border_touching_labels(self, masks):
        """Get set of labels touching image border."""
        if masks is None or np.max(masks) == 0:
            return set()
        
        border_labels = set()
        border_labels.update(np.unique(masks[0, :]))    # Top
        border_labels.update(np.unique(masks[-1, :]))   # Bottom
        border_labels.update(np.unique(masks[:, 0]))    # Left
        border_labels.update(np.unique(masks[:, -1]))   # Right
        border_labels.discard(0)  # Remove background
        return border_labels

    def remove_labels_and_reindex(self, masks, labels_to_remove):
        """Remove specified labels from masks and reindex remaining."""
        if masks is None or np.max(masks) == 0:
            return masks
        
        result = masks.copy()
        for label in labels_to_remove:
            result[result == label] = 0
        
        return self.reindex_masks(result)

    def reindex_masks(self, masks):
        """Reindex mask labels to be continuous starting from 1."""
        unique_labels = np.unique(masks)
        unique_labels = unique_labels[unique_labels > 0]
        new_masks = np.zeros_like(masks)
        for new_id, old_id in enumerate(unique_labels, start=1):
            new_masks[masks == old_id] = new_id
        return new_masks

    def plot_cellpose_results(self):
        if self.image_stack is None:
            return
            
        self.ax_cellpose.clear()
        
        # Get current image slice
        ch = self.cellpose_current_channel
        if self.image_stack.ndim == 5:
            # [T, Z, Y, X, C] -> Max projection over Z for display
            img_slice = self.image_stack[self.cellpose_current_frame, :, :, :, ch]
            if img_slice.ndim == 3: # ZYX
                 img_slice = np.max(img_slice, axis=0)
        else:
            # Fallback
            img_slice = np.zeros((512, 512))

        # Get display parameters for channel (match other tabs)
        params = self.channelDisplayParams.get(ch, {
            'min_percentile': self.display_min_percentile,
            'max_percentile': self.display_max_percentile,
            'sigma': self.display_sigma,
            'low_sigma': self.low_display_sigma
        })
        
        # Normalize using percentiles (like other tabs)
        rescaled = mi.Utilities().convert_to_int8(
            img_slice,
            rescale=True,
            min_percentile=params['min_percentile'],
            max_percentile=params['max_percentile']
        )
        if params['low_sigma'] > 0:
            rescaled = gaussian_filter(rescaled, sigma=params['low_sigma'])
        if params['sigma'] > 0:
            rescaled = gaussian_filter(rescaled, sigma=params['sigma'])
        normalized = rescaled.astype(float) / 255.0
        if normalized.ndim == 3:
            normalized = normalized[..., 0]
        
        # Use the same colormap as other tabs
        cmap_used = cmap_list_imagej[ch % len(cmap_list_imagej)]
        self.ax_cellpose.imshow(normalized, cmap=cmap_used, vmin=0, vmax=1)
        
        # Overlay Cytosol Masks
        if self.cellpose_masks_cyto is not None:
            # Draw contours
            for label in np.unique(self.cellpose_masks_cyto):
                if label == 0: continue
                mask = self.cellpose_masks_cyto == label
                self.ax_cellpose.contour(mask, levels=[0.5], colors='yellow', linewidths=1)
                
                # Add label ID
                y, x = center_of_mass(mask)
                self.ax_cellpose.text(x, y, str(label), color='yellow', fontsize=8, ha='center', va='center')

        # Overlay Nucleus Masks
        if self.cellpose_masks_nuc is not None:
            # Draw contours
            for label in np.unique(self.cellpose_masks_nuc):
                if label == 0: continue
                mask = self.cellpose_masks_nuc == label
                self.ax_cellpose.contour(mask, levels=[0.5], colors='cyan', linewidths=1)
                
                # Add label ID (if not already added by cyto, or if we want to show it explicitly)
                # If synchronized, IDs should match. 
                if self.cellpose_masks_cyto is None:
                     y, x = center_of_mass(mask)
                     self.ax_cellpose.text(x, y, str(label), color='cyan', fontsize=8, ha='center', va='center')

        #self.ax_cellpose.axis('off')
        # grid off
        self.ax_cellpose.grid(False)
        self.canvas_cellpose.draw()


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
            self._active_mask_source = 'segmentation'
            # Clear Cellpose masks since we're using watershed now
            self.cellpose_masks_cyto = None
            self.cellpose_masks_nuc = None
            self.plot_segmentation()
            self.segmentation_mode = "watershed"
            # Reset dependent tabs since masks changed
            self.reset_cellpose_tab()
            self.reset_photobleaching_tab()
            self.reset_tracking_tab()
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
            cmap_used = cmap_list_imagej[ch % len(cmap_list_imagej)]
            self.ax_segmentation.imshow(normalized_image[..., 0], cmap=cmap_used, vmin=0, vmax=1)
            
            # Draw contours for segmentation mask
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
        
        # Check if we have any mask (segmentation or Cellpose)
        has_segmentation_mask = self.segmentation_mask is not None
        has_cellpose_mask = (self.cellpose_masks_cyto is not None or 
                             self.cellpose_masks_nuc is not None)
        
        mode = self.mode_combo.currentText().lower()
        
        # If no masks at all and mode is not entire_image, show warning
        if not has_segmentation_mask and not has_cellpose_mask:
            if mode != 'entire_image':
                QMessageBox.warning(self, "No Segmentation Mask", 
                                    "Please perform segmentation first, or use 'entire_image' mode.")
                return
        
        # If Cellpose masks exist but no segmentation mask, use entire_image mode
        # (Cellpose masks are labeled, not suitable for photobleaching mask input)
        if has_cellpose_mask and not has_segmentation_mask:
            mode = 'entire_image'
            # Inform user that we're using entire_image mode
            QMessageBox.information(self, "Using Entire Image", 
                                    "Cellpose masks detected. Photobleaching will be calculated using the entire image.")
        
        self.photobleaching_mode = mode
        radius = self.radius_spinbox.value()
        
        if self.segmentation_mask is None:
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

    def _sync_tracking_channel(self):
        self.channels_spots = [self.current_channel]

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
    
    def track_particles(self, corrected_image, masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei, parameters, use_maximum_projection):
        """
        Run particle tracking on `corrected_image` with the given masks and parameters.
        Pops up a warning on subnet-oversize and returns an empty list in that case.
        
        Parameters
        ----------
        corrected_image : ndarray
            Image to track particles in.
        masks_complete_cells : ndarray
            Labeled mask for complete cells (or binary segmentation mask).
        masks_nuclei : ndarray or None
            Labeled mask for nuclei.
        masks_cytosol_no_nuclei : ndarray or None
            Labeled mask for cytosol regions (with nucleus removed).
        parameters : dict
            Tracking parameters.
        use_maximum_projection : bool
            Whether to use maximum projection.
            
        Returns
        -------
        list of DataFrames: Trajectory data per channel.
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
                masks=masks_complete_cells,
                masks_nuclei=masks_nuclei,
                masks_cytosol_no_nuclei=masks_cytosol_no_nuclei,
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
        mask_GUI = (self.active_mask > 0).astype(int) if self.active_mask is not None else np.ones(image_channel.shape[1:], dtype=image_channel.dtype)
        # Compute maximum projection (across Z)
        max_proj = np.max(image_channel, axis=0) * mask_GUI
        intensity_values = max_proj.flatten()
        # Filter out zeros (background/masked pixels)
        intensity_values = intensity_values[intensity_values > 0]
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
        mask_GUI = (self.active_mask > 0).astype(int) if self.active_mask is not None else np.ones(image_channel.shape[1:], dtype=image_channel.dtype)
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

    def update_use_fixed_size_intensity(self, state):
        self.use_fixed_size_for_intensity_calculation = (state == Qt.Checked)

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
        # Prepare masks for tracking (supports both Cellpose and Segmentation)
        masks_complete, masks_nuc, masks_cyto_no_nuc = self._get_tracking_masks()
        if masks_complete is None:
            masks_complete = np.ones(self.image_stack.shape[2:4], dtype=int)
        self.tracking_channel = self.current_channel
        self._sync_tracking_channel()
        # Run spot detection (no linking)
        list_dataframes_trajectories, _ = mi.ParticleTracking(
            image=image_to_use,
            channels_spots=[self.current_channel],
            masks=masks_complete,
            masks_nuclei=masks_nuc,
            masks_cytosol_no_nuclei=masks_cyto_no_nuc,
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
                masks=masks_complete,
                masks_nuclei=masks_nuc,
                masks_cytosol_no_nuclei=masks_cyto_no_nuc,
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
            mask = (self.active_mask > 0).astype(int) if self.active_mask is not None else np.ones(self.image_stack.shape[2:4], dtype=int)
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
        cmap_imagej = cmap_list_imagej[ch % len(cmap_list_imagej)]
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
        # Draw mask contours and IDs if checkbox is checked
        if self.tracking_show_masks_checkbox.isChecked():
            masks_to_draw = []
            
            # Check if Cellpose is the active source - show both mask types
            if self._active_mask_source == 'cellpose':
                if self.cellpose_masks_cyto is not None:
                    masks_to_draw.append(('cyto', self.cellpose_masks_cyto, 'cyan'))
                if self.cellpose_masks_nuc is not None:
                    masks_to_draw.append(('nuc', self.cellpose_masks_nuc, 'magenta'))
            elif self.segmentation_mask is not None:
                # Segmentation mask (binary)
                masks_to_draw.append(('seg', self.segmentation_mask, 'cyan'))
            
            for mask_type, labeled_mask, color in masks_to_draw:
                if labeled_mask is not None:
                    # Draw contours for each labeled region
                    unique_labels = np.unique(labeled_mask)
                    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
                    for label_id in unique_labels:
                        single_mask = (labeled_mask == label_id).astype(np.uint8)
                        self.ax_tracking.contour(single_mask, levels=[0.5], colors=color, linewidths=0.8, alpha=0.7)
                        # Find centroid for label text
                        coords = np.argwhere(single_mask > 0)
                        if len(coords) > 0:
                            cy, cx = coords.mean(axis=0)
                            self.ax_tracking.text(cx, cy, str(int(label_id)),
                                                  color=color, fontsize=6, ha='center', va='center',
                                                  fontweight='bold', alpha=0.9)
        elif self.segmentation_mask is not None:
            # Fallback: show binary segmentation mask contour if checkbox is off
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

    def detect_spots(self, image, threshold, list_voxels, masks_complete_cells, masks_nuclei=None, masks_cytosol_no_nuclei=None):
        z_sp_sz = self.z_spot_size_in_px if self.z_spot_size_in_px is not None else 1
        yx_sp_sz = self.yx_spot_size_in_px if self.yx_spot_size_in_px is not None else 5
        dataframe = mi.SpotDetection(
                image,
                channels_spots=0,
                channels_cytosol=self.channels_cytosol,
                channels_nucleus=self.channels_nucleus,
                masks_complete_cells=masks_complete_cells,
                masks_nuclei=masks_nuclei,
                masks_cytosol_no_nuclei=masks_cytosol_no_nuclei,
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
        # Get masks for tracking (supports both Cellpose and Segmentation)
        masks_complete, masks_nuc, masks_cyto_no_nuc = self._get_tracking_masks()
        if masks_complete is None:
            masks_complete = np.ones(self.image_stack.shape[2:4], dtype=int)
        spots = self.detect_spots(image_channel, threshold, list_voxels, masks_complete, masks_nuc, masks_cyto_no_nuc)
        if spots is not None and not spots.empty:
            spots['frame'] = self.current_frame
            self.detected_spots_frame = spots
            self.df_tracking = spots.copy()
        else:
            self.detected_spots_frame = None
            self.df_tracking = pd.DataFrame()
        self.plot_tracking()

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
        # Get masks for tracking (supports both Cellpose and Segmentation)
        masks_complete, masks_nuc, masks_cyto_no_nuc = self._get_tracking_masks()
        if masks_complete is None:
            masks_complete = np.ones(self.image_stack.shape[2:4], dtype=int)
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
        self._sync_tracking_channel()
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
            results = self.track_particles(image_to_use, masks_complete, masks_nuc, masks_cyto_no_nuc, parameters, self.use_maximum_projection)
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
                masks=masks_complete,
                masks_nuclei=masks_nuc,
                masks_cytosol_no_nuclei=masks_cyto_no_nuc,
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
                step_size_in_sec=float(self.time_interval_value),
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
                    D_um2_s, D_px2_s, _, _, _, _ ,_= pm.calculate_msd()
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
        self.min_percentile_spinbox_tracking.setRange(0.0, 90.0)
        self.min_percentile_spinbox_tracking.setSingleStep(0.1)
        self.min_percentile_spinbox_tracking.setSuffix("%")
        self.min_percentile_spinbox_tracking.setValue(self.tracking_min_percentile)
        self.min_percentile_spinbox_tracking.valueChanged.connect(
            lambda: (setattr(self, 'tracking_min_percentile', self.min_percentile_spinbox_tracking.value()), self.plot_tracking())
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
            lambda: (setattr(self, 'tracking_max_percentile', self.max_percentile_spinbox_tracking.value()), self.plot_tracking())
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
        # Add "Show Masks" checkbox for visualizing mask contours and IDs
        self.tracking_show_masks_checkbox = QCheckBox("Masks")
        self.tracking_show_masks_checkbox.setChecked(False)
        self.tracking_show_masks_checkbox.stateChanged.connect(self.plot_tracking)
        checkbox_layout.addWidget(self.tracking_show_masks_checkbox)
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
        self.min_length_input.setMaximum(1000)
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
        
        # Group 4: Intensity Calculation
        intensity_calc_group = QGroupBox("Intensity Calculation")
        intensity_calc_layout = QVBoxLayout(intensity_calc_group)
        tracking_right_main_layout.addWidget(intensity_calc_group)
        
        self.fixed_size_intensity_checkbox = QCheckBox("Use Fixed Size for Intensity Calculation")
        self.fixed_size_intensity_checkbox.setChecked(self.use_fixed_size_for_intensity_calculation)
        self.fixed_size_intensity_checkbox.stateChanged.connect(self.update_use_fixed_size_intensity)
        intensity_calc_layout.addWidget(self.fixed_size_intensity_checkbox)
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
        generate_random_points_checkbox.setChecked(True)
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
        data_type_label = QLabel("Data:")
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems([
            "particles", "spot_int", "spot_size", "psf_amplitude",
            "psf_sigma", "total_spot_int", "snr"
        ])
        controls_layout.addWidget(data_type_label)
        controls_layout.addWidget(self.data_type_combo)

        # Percentile controls
        min_percentile_label = QLabel("Min_Perc:")
        self.min_percentile_spinbox = QDoubleSpinBox()
        self.min_percentile_spinbox.setRange(0.0, 50.0)
        self.min_percentile_spinbox.setValue(5.0)
        self.min_percentile_spinbox.setSuffix("%")
        controls_layout.addWidget(min_percentile_label)
        controls_layout.addWidget(self.min_percentile_spinbox)

        max_percentile_label = QLabel("Max_Perc:")
        self.max_percentile_spinbox = QDoubleSpinBox()
        self.max_percentile_spinbox.setRange(50.0, 100.0)
        self.max_percentile_spinbox.setValue(95.0)
        self.max_percentile_spinbox.setSuffix("%")
        controls_layout.addWidget(max_percentile_label)
        controls_layout.addWidget(self.max_percentile_spinbox)

        # Show Individual Traces checkbox
        self.show_traces_checkbox = QCheckBox("Individual")
        self.show_traces_checkbox.setChecked(False)
        controls_layout.addWidget(self.show_traces_checkbox)

        # Normalize Data checkbox
        self.normalize_time_course_checkbox = QCheckBox("Normalize")
        self.normalize_time_course_checkbox.setChecked(False)
        controls_layout.addWidget(self.normalize_time_course_checkbox)

        # Show Time in Minutes checkbox
        self.show_time_in_minutes_checkbox = QCheckBox("Minutes")
        self.show_time_in_minutes_checkbox.setChecked(False)
        controls_layout.addWidget(self.show_time_in_minutes_checkbox)

        # Moving Average SpinBox
        ma_label = QLabel("moving_ave:")
        self.moving_average_spinbox = QSpinBox()
        self.moving_average_spinbox.setRange(1, 50)
        self.moving_average_spinbox.setValue(1)
        controls_layout.addWidget(ma_label)
        controls_layout.addWidget(self.moving_average_spinbox)

        # Plot button
        self.plot_time_course_button = QPushButton("Plot", self)
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
        self.export_time_course_button = QPushButton("Export Image", self)
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
        #self.de_correlation_threshold = value
        self.de_correlation_threshold = max(value, 0.0)

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
            new_intensity_arrays = {}
            for ch, arr_int in list(intensity_arrays.items()):
                col = f'snr_ch_{ch}'
                if col not in self.df_tracking.columns:
                    # No SNR column for this channel—keep as-is
                    new_intensity_arrays[ch] = arr_int
                    continue
                # Build intensity & SNR using the SAME particle intersection & order
                arr_int_raw, arr_snr_raw, _ = mi.Utilities().df_fields_to_arrays_aligned(
                    dataframe=self.df_tracking,
                    selected_field_a=f'{field_base}_ch_{ch}',
                    selected_field_b=f'snr_ch_{ch}',
                    total_frames=self.total_frames,
                    require_both_non_nan=True,
                )
                # Now jointly filter/shift/trim with one mask and one cut length
                arr_int_aligned, arr_snr_aligned = mi.Utilities().shift_trajectories(
                    arr_int_raw,
                    arr_snr_raw,
                    min_percentage_data_in_trajectory=self.min_percentage_data_in_trajectory,
                )
                # SNR gating
                mean_snr = np.nanmean(arr_snr_aligned, axis=1)
                valid_idx = np.where(mean_snr >= threshold)[0]
                if valid_idx.size == 0:
                    print(f"After alignment, no valid indices remain for channel {ch}.")
                    continue
                new_intensity_arrays[ch] = arr_int_aligned[valid_idx]
            intensity_arrays = new_intensity_arrays

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
                mean_corr, std_corr, lags, correlations_array, _ = corr.run()
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
                    'correlations_array': correlations_array,
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
            mean_corr, std_corr, lags, correlations_array, _ = corr.run()
            if index_max >= len(lags):
                QMessageBox.warning(
                    self, "Max-Lag Adjusted",
                    f"Requested lag {index_max} exceeds available {len(lags)-1} "
                    f"for {'multi-tau' if use_multi else 'linear'} mode.\n"
                    f"Using {len(lags)-1} instead."
                )
                index_max = len(lags) - 1
                self.index_max_lag_for_fit_input.setValue(index_max)
            self.correlation_results.append({
                'type': 'crosscorrelation',
                'channel1': ch1,
                'channel2': ch2,
                'intensity_array1': d1,
                'intensity_array2': d2,
                'mean_corr': mean_corr,
                'std_corr': std_corr,
                'correlations_array': correlations_array,
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
        self.start_lag_input.setValue(0)
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
        # add a checkbox to use multi-tau
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
        invoked_by_run = (
            hasattr(self, 'compute_colocalization_button')
            and self.sender() is not None
            and self.sender() == self.compute_colocalization_button
        )
        # Require tracking results
        if (not getattr(self, 'has_tracked', False)) and self.df_tracking.empty:
            if invoked_by_run:
                QMessageBox.warning(self, "Colocalization Error",
                                    "Please complete all frames' detection and complete tracking before colocalization.")
            return
        # Require two distinct channels for colocalization
        ch1 = self.channel_combo_box_1.currentIndex()
        ch2 = self.channel_combo_box_2.currentIndex()
        if ch1 == ch2:
            if invoked_by_run:
                QMessageBox.warning(self, "Invalid Selection", "Select two different channels.")
            return
        # Require image data
        image = self.corrected_image if self.corrected_image is not None else self.image_stack
        if image is None:
            if invoked_by_run:
                QMessageBox.warning(self, "No Image Data", "Please load and process an image first.")
            return
        
        if self.use_maximum_projection:
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
            flag_vector, prediction_values_vector = ML.predict_crops(model_ML, crops_norm, threshold=threshold)
        else:
            threshold = self.snr_threshold_input.value()
            method_used = "Intensity"
            num_crops = mean_crop.shape[0] // crop_size
            results_snr = [mi.Utilities().is_spot_in_crop(
                        i, crop_size=crop_size, selected_color_channel=ch2,
                        array_crops_YXC=mean_crop,
                        show_plot=False,
                        snr_threshold=threshold)
                        for i in range(num_crops)]
            flag_vector, prediction_values_vector = zip(*results_snr)
            flag_vector = np.array(flag_vector)
            prediction_values_vector = np.array(prediction_values_vector)
        colocal_perc = 0 if len(flag_vector) == 0 else (np.sum(flag_vector) / len(flag_vector)) * 100
        self.colocalization_percentage_label.setText(f"Colocalization Percentage: {colocal_perc:.2f}%")
        
        # Clear manual colocalization UI so new results can load
        self.manual_scroll_area.setWidget(QWidget())
        self.manual_checkboxes = []
        self.manual_mean_crop = None
        self.manual_stats_label.setText("Total Spots: 0 | Colocalized: 0 | 0.00%")
        self.manual_current_image_name = None

        self.colocalization_results = {
            'mean_crop_filtered': mean_crop,
            'crop_size': crop_size,
            'flag_vector': flag_vector,
            'prediction_values_vector': prediction_values_vector,
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

    def sort_manual_colocalization(self):
        """Sort the manual colocalization results by prediction metric (lowest to highest) and refresh the display."""
        if not hasattr(self, 'manual_checkboxes') or len(self.manual_checkboxes) == 0:
            return  # Only proceed if manual data is loaded

        values = self.colocalization_results.get('prediction_values_vector') if hasattr(self, 'colocalization_results') else None
        if values is None:
            return  # No values to sort by

        sorted_idx = np.argsort(np.array(values))  # sort indices from lowest to highest prediction value

        # Preserve current checked states (if none checked yet, use initial prediction flags)
        current_flags = [chk.isChecked() for chk in self.manual_checkboxes]
        if any(current_flags):
            sorted_flags = [current_flags[i] for i in sorted_idx]
        else:
            pred_flags = self.colocalization_results.get('flag_vector', [])
            sorted_flags = [bool(pred_flags[i]) for i in sorted_idx]

        mean_crop = getattr(self, 'manual_mean_crop', None)
        crop_size = getattr(self, 'manual_crop_size', None)
        if mean_crop is None or crop_size is None:
            return

        num_spots = len(self.manual_checkboxes)
        # Rebuild the scroll area content in sorted order
        self.manual_scroll_area.takeWidget()
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(3)
        container_layout.setContentsMargins(0, 0, 0, 0)
        new_checkboxes = []
        scale_factor = getattr(self, 'coloc_thumbnail_scale', 4)

        for rank, orig_idx in enumerate(sorted_idx.tolist()):
            spot_layout = QHBoxLayout()
            spot_layout.setSpacing(1)
            spot_layout.setContentsMargins(0, 0, 0, 0)
            crop_block = mean_crop[orig_idx * crop_size : (orig_idx + 1) * crop_size, :, :]
            channels = crop_block.shape[-1]
            for ch in range(channels):
                channel_crop = crop_block[:, :, ch]
                cmin, cmax = np.nanmin(channel_crop), np.nanmax(channel_crop)
                if cmax > cmin:
                    norm = ((channel_crop - cmin) / (cmax - cmin) * 255).astype(np.uint8)
                else:
                    norm = np.zeros_like(channel_crop, np.uint8)
                h, w = norm.shape
                qimg = QImage(norm.data, w, h, w, QImage.Format_Grayscale8).copy()
                pix = QPixmap.fromImage(qimg)
                pix = pix.scaled(w * scale_factor, h * scale_factor, Qt.IgnoreAspectRatio, Qt.FastTransformation)
                lbl = QLabel()
                lbl.setPixmap(pix)
                spot_layout.addWidget(lbl)
            chk = QCheckBox(f"Spot {rank+1}")
            chk.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            chk.setChecked(bool(sorted_flags[rank]))
            chk.toggled.connect(self.update_manual_stats_label)
            spot_layout.addWidget(chk)
            new_checkboxes.append(chk)
            container_layout.addLayout(spot_layout)
            if rank < num_spots - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.HLine)
                sep.setFrameShadow(QFrame.Sunken)
                container_layout.addWidget(sep)

        self.manual_checkboxes = new_checkboxes
        self.manual_scroll_area.setWidget(container)
        self.update_manual_stats_label()


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
        
        self.sort_manual_coloc_button = QPushButton("Sort")
        self.sort_manual_coloc_button.clicked.connect(self.sort_manual_colocalization)
        top_bar.addWidget(self.sort_manual_coloc_button)
        
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
        if img_src is None:
            return  # No image loaded yet
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
            main_cmap = cmap_list_imagej[selected_channelIndex % len(cmap_list_imagej)]
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
            ax.imshow(crop, cmap=cmap_list_imagej[ci % len(cmap_list_imagej)], interpolation='nearest', vmin=0, vmax=1)
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
        if hasattr(self, 'time_slider_tracking_vis'):
            self.time_slider_tracking_vis.setValue(0)
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
        self.min_percentile_spinbox_tracking_vis.setRange(0.0, 95.0)
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

                elif label_text == "Export Cellpose Masks":
                    self._export_cellpose_masks(results_folder)

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

    def _export_cellpose_masks(self, results_folder):
        """Export Cellpose masks to the results folder (for batch export)."""
        try:
            if self.cellpose_masks_cyto is not None:
                cyto_filename = self.get_default_export_filename(prefix="cellpose_cytosol", extension="tif")
                cyto_path = results_folder / cyto_filename
                mask_cyto = self.cellpose_masks_cyto.astype(np.uint8)
                tifffile.imwrite(str(cyto_path), mask_cyto, photometric='minisblack')
            
            if self.cellpose_masks_nuc is not None:
                nuc_filename = self.get_default_export_filename(prefix="cellpose_nucleus", extension="tif")
                nuc_path = results_folder / nuc_filename
                mask_nuc = self.cellpose_masks_nuc.astype(np.uint8)
                tifffile.imwrite(str(nuc_path), mask_nuc, photometric='minisblack')
        except Exception as e:
            print(f"Failed to export Cellpose masks: {e}")


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
        # Photobleaching: Read from widgets if available, else fallback to attribute
        pb_mode = self.mode_combo.currentText() if hasattr(self, 'mode_combo') else self.photobleaching_mode
        pb_radius = self.radius_spinbox.value() if hasattr(self, 'radius_spinbox') else self.photobleaching_radius

        # Tracking Parameters: Read directly from spinboxes
        min_len = self.min_length_input.value() if hasattr(self, 'min_length_input') else self.min_length_trajectory
        yx_spot = self.spot_size_input.value() if hasattr(self, 'spot_size_input') else self.yx_spot_size_in_px
        z_spot = self.spot_size_z_input.value() if hasattr(self, 'spot_size_z_input') else self.z_spot_size_in_px
        clust_rad = self.cluster_radius_input.value() if hasattr(self, 'cluster_radius_input') else self.cluster_radius_nm
        max_spots_clust = self.max_spots_cluster_input.value() if hasattr(self, 'max_spots_cluster_input') else self.maximum_spots_cluster
        max_range = self.max_range_search_input.value() if hasattr(self, 'max_range_search_input') else self.maximum_range_search_pixels
        mem = self.memory_input.value() if hasattr(self, 'memory_input') else self.memory
        
        # Thresholds
        thresh_spot = self.threshold_slider.value() if hasattr(self, 'threshold_slider') else self.threshold_spot_detection
        
        # Correlation Settings
        # Read radio buttons directly to ensure accuracy
        if hasattr(self, 'linear_radio') and self.linear_radio.isChecked():
            corr_fit = 'linear'
        else:
            corr_fit = 'exponential'
            
        idx_max_lag = self.index_max_lag_for_fit_input.value() if hasattr(self, 'index_max_lag_for_fit_input') else self.index_max_lag_for_fit
        decorr_thresh = self.de_correlation_threshold_input.value() if hasattr(self, 'de_correlation_threshold_input') else self.de_correlation_threshold
        min_perc_data = self.max_percentage_spin.value() if hasattr(self, 'max_percentage_spin') else self.min_percentage_data_in_trajectory

        # Colocalization / ML
        ml_thresh = self.ml_threshold_input.value() if hasattr(self, 'ml_threshold_input') else 0.5
        snr_thresh = self.snr_threshold_input.value() if hasattr(self, 'snr_threshold_input') else 3.0
        
        is_ml = self.method_ml_radio.isChecked() if hasattr(self, 'method_ml_radio') else False
        coloc_thresh = ml_thresh if is_ml else snr_thresh
        coloc_method = "ML" if is_ml else "Intensity"
        
        img_source = self.image_source_combo.currentText() if hasattr(self, 'image_source_combo') else self.image_source_combo_value

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
            
            # Updated Tracking Params (Values)
            min_length_trajectory=min_len,
            yx_spot_size_in_px=yx_spot,
            z_spot_size_in_px=z_spot,
            cluster_radius_nm=clust_rad,
            maximum_spots_cluster=max_spots_clust,
            separate_clusters_and_spots=self.separate_clusters_and_spots,
            maximum_range_search_pixels=max_range,
            memory=mem,
            
            # Updated Thresholds (Values)
            de_correlation_threshold=decorr_thresh,
            max_spots_for_threshold=self.max_spots_for_threshold,
            threshold_spot_detection=thresh_spot,
            user_selected_threshold=thresh_spot,
            
            image_source_combo=img_source,
            use_fixed_size_for_intensity_calculation=self.use_fixed_size_for_intensity_calculation,
            
            # Updated Correlation Params (Values)
            correlation_fit_type=corr_fit,
            index_max_lag_for_fit=idx_max_lag,
            min_percentage_data_in_trajectory=min_perc_data,
            
            photobleaching_calculated=self.photobleaching_calculated,
            use_maximum_projection=self.use_maximum_projection,
            
            # Updated Photobleaching Params (Values)
            photobleaching_mode=pb_mode,
            photobleaching_radius=pb_radius,
            
            file_path=file_path,
            
            # Updated ML Params (Values)
            use_ml_checkbox=is_ml,
            ml_threshold_input=ml_thresh,
            
            link_using_3d_coordinates=self.link_using_3d_coordinates,
            colocalization_method=coloc_method,
            colocalization_threshold_value=coloc_thresh,
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
            
        # Re-use the logic from _export_metadata to avoid code duplication
        self._export_metadata(file_path)
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

        frames = []
        total_frames = self.image_stack.shape[0]
        for i in range(total_frames):
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

    def export_cellpose_masks_as_tiff(self):
        """
        Export Cellpose masks (cytosol and nucleus) as separate labeled TIFF files.
        Supports both 2D YX masks and 3D TYX time-varying masks.
        """
        # Check for TYX masks first, then YX masks
        has_cyto_tyx = self.cellpose_masks_cyto_tyx is not None
        has_nuc_tyx = self.cellpose_masks_nuc_tyx is not None
        has_cyto = self.cellpose_masks_cyto is not None
        has_nuc = self.cellpose_masks_nuc is not None
        
        if not has_cyto and not has_nuc and not has_cyto_tyx and not has_nuc_tyx:
            QMessageBox.warning(self, "No Cellpose Masks", 
                                "No Cellpose masks available to export.\nRun Cellpose segmentation first.")
            return
        
        # Ask user for base filename
        default_filename = self.get_default_export_filename(prefix="cellpose_masks", extension="tif")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Cellpose Masks (base filename)",
            default_filename,
            "TIFF Files (*.tif);;All Files (*)",
            options=options
        )
        
        if not file_path:
            return
        
        # Remove extension to create base path
        base_path = file_path.rsplit('.', 1)[0] if '.' in file_path else file_path
        
        exported_files = []
        try:
            # Export TYX cytosol mask if available (priority over YX)
            if has_cyto_tyx:
                cyto_path = f"{base_path}_cytosol_TYX.tif"
                # Save as uint16 for TYX (supports larger cell counts over time)
                mask_cyto = np.asarray(self.cellpose_masks_cyto_tyx).astype(np.uint16)
                tifffile.imwrite(cyto_path, mask_cyto, photometric='minisblack', 
                                metadata={'axes': 'TYX'})
                exported_files.append(f"Cytosol TYX: {cyto_path} (shape: {mask_cyto.shape})")
            elif has_cyto:
                cyto_path = f"{base_path}_cytosol.tif"
                mask_cyto = self.cellpose_masks_cyto.astype(np.uint8)
                tifffile.imwrite(cyto_path, mask_cyto, photometric='minisblack')
                exported_files.append(f"Cytosol: {cyto_path}")
            
            # Export TYX nucleus mask if available (priority over YX)
            if has_nuc_tyx:
                nuc_path = f"{base_path}_nucleus_TYX.tif"
                mask_nuc = np.asarray(self.cellpose_masks_nuc_tyx).astype(np.uint16)
                tifffile.imwrite(nuc_path, mask_nuc, photometric='minisblack',
                                metadata={'axes': 'TYX'})
                exported_files.append(f"Nucleus TYX: {nuc_path} (shape: {mask_nuc.shape})")
            elif has_nuc:
                nuc_path = f"{base_path}_nucleus.tif"
                mask_nuc = self.cellpose_masks_nuc.astype(np.uint8)
                tifffile.imwrite(nuc_path, mask_nuc, photometric='minisblack')
                exported_files.append(f"Nucleus: {nuc_path}")
            
            # Show success message
            n_cyto = int(self.cellpose_masks_cyto.max()) if has_cyto else 0
            n_nuc = int(self.cellpose_masks_nuc.max()) if has_nuc else 0
            
            msg = f"Cellpose masks exported successfully!\n\n"
            if has_cyto_tyx or has_nuc_tyx:
                msg += f"Format: 3D Time-Varying [T, Y, X]\n"
            else:
                msg += f"Format: 2D [Y, X]\n"
            msg += f"Max cells per frame: {max(n_cyto, n_nuc)}\n\n"
            msg += "Files:\n" + "\n".join(exported_files)
            QMessageBox.information(self, "Export Success", msg)
            
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
            ("Export Cellpose Masks", "cellpose_masks"),
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
        self.segmentation_current_channel = 0
        self.segmentation_current_frame = 0
        if hasattr(self, 'segmentation_time_slider'):
            self.segmentation_time_slider.setValue(0)

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
        if hasattr(self, 'time_slider_tracking'):
            self.time_slider_tracking.setValue(0)
        if hasattr(self, 'tracking_show_masks_checkbox'):
            self.tracking_show_masks_checkbox.setChecked(False)
        # Reset threshold slider and histogram
        if hasattr(self, 'threshold_slider'):
            self.threshold_slider.setValue(0)
            self.threshold_slider.setMinimum(0)
            self.threshold_slider.setMaximum(10000)
        self.user_selected_threshold = None
        # Clear threshold histogram
        if hasattr(self, 'ax_threshold_hist'):
            self.ax_threshold_hist.clear()
            self.ax_threshold_hist.set_facecolor('black')
            self.ax_threshold_hist.axis('off')
            self.canvas_threshold_hist.draw_idle()

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

    def reset_manual_colocalization(self):
        """Reset manual colocalization state and UI elements."""
        if hasattr(self, 'manual_scroll_area'):
            self.manual_scroll_area.setWidget(QWidget())
        if hasattr(self, 'manual_checkboxes'):
            self.manual_checkboxes = []
        if hasattr(self, 'manual_mean_crop'):
            self.manual_mean_crop = None
        if hasattr(self, 'df_manual_colocalization'):
            self.df_manual_colocalization = pd.DataFrame()
        if hasattr(self, 'manual_stats_label'):
            self.manual_stats_label.setText("Total Spots: 0 | Colocalized: 0 | 0.00%")
        self.manual_current_image_name = None

    def reset_cellpose_tab(self):
        """Reset Cellpose tab state, masks, and UI controls to defaults."""
        # Clear masks (YX)
        self.cellpose_masks_cyto = None
        self.cellpose_masks_nuc = None
        # Clear TYX masks
        self.cellpose_masks_cyto_tyx = None
        self.cellpose_masks_nuc_tyx = None
        self.use_tyx_masks = False
        
        # Reset frame/channel indices
        if hasattr(self, 'cellpose_current_frame'):
            self.cellpose_current_frame = 0
        if hasattr(self, 'cellpose_current_channel'):
            self.cellpose_current_channel = 0
        
        # Reset time slider
        if hasattr(self, 'time_slider_cellpose'):
            self.time_slider_cellpose.setValue(0)
        
        # Reset Cytosol parameters to defaults
        if hasattr(self, 'cellpose_cyto_model_input'):
            self.cellpose_cyto_model_input.setCurrentText('cyto3')
        if hasattr(self, 'cellpose_cyto_channel_input'):
            # Set to channel 1 if available, otherwise channel 0
            max_ch = max(0, (self.number_color_channels or 1) - 1)
            self.cellpose_cyto_channel_input.setMaximum(max_ch)
            default_cyto_ch = min(1, max_ch)
            self.cellpose_cyto_channel_input.setValue(default_cyto_ch)
        if hasattr(self, 'cellpose_cyto_diameter_input'):
            self.cellpose_cyto_diameter_input.setValue(120)
        if hasattr(self, 'cellpose_cyto_flow_input'):
            self.cellpose_cyto_flow_input.setValue(0.4)
        if hasattr(self, 'chk_optimize_cyto'):
            self.chk_optimize_cyto.setChecked(False)
        
        # Reset Nucleus parameters to defaults
        if hasattr(self, 'cellpose_nuc_model_input'):
            self.cellpose_nuc_model_input.setCurrentText('nuclei')
        if hasattr(self, 'cellpose_nuc_channel_input'):
            # Set maximum based on available channels
            max_ch = max(0, (self.number_color_channels or 1) - 1)
            self.cellpose_nuc_channel_input.setMaximum(max_ch)
            self.cellpose_nuc_channel_input.setValue(0)
        if hasattr(self, 'cellpose_nuc_diameter_input'):
            self.cellpose_nuc_diameter_input.setValue(60)
        if hasattr(self, 'cellpose_nuc_flow_input'):
            self.cellpose_nuc_flow_input.setValue(0.4)
        if hasattr(self, 'chk_optimize_nuc'):
            self.chk_optimize_nuc.setChecked(False)
        
        # Reset Improve Segmentation checkboxes
        if hasattr(self, 'chk_remove_border_cells'):
            self.chk_remove_border_cells.setChecked(False)
        if hasattr(self, 'chk_remove_unpaired_cells'):
            self.chk_remove_unpaired_cells.setChecked(False)
        
        # Clear the figure
        if hasattr(self, 'figure_cellpose'):
            self.figure_cellpose.clear()
            self.ax_cellpose = self.figure_cellpose.add_subplot(111)
            self.ax_cellpose.set_facecolor('black')
            self.ax_cellpose.axis('off')
            self.ax_cellpose.text(
                0.5, 0.5, 'No Cellpose segmentation performed.',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12, color='white',
                transform=self.ax_cellpose.transAxes
            )
            if hasattr(self, 'canvas_cellpose'):
                self.canvas_cellpose.draw()

    def reset_all_state(self):
        """
        Unified reset method called when loading a new image.
        Resets all tabs, clears state variables, and prepares the GUI for new data.
        """
        # Reset all tab displays
        self.reset_display_tab()
        self.reset_segmentation_tab()
        self.reset_photobleaching_tab()
        self.reset_tracking_tab()
        self.reset_distribution_tab()
        self.reset_time_course_tab()
        self.reset_correlation_tab()
        self.reset_colocalization_tab()
        self.reset_crops_tab()
        self.reset_tracking_visualization_tab()
        self.reset_export_comment()
        self.reset_manual_colocalization()
        self.reset_cellpose_tab()
        
        # Reset shared state variables
        self.has_tracked = False
        self.photobleaching_calculated = False
        self.detected_spots_frame = None
        self.corrected_image = None
        self.df_tracking = pd.DataFrame()
        self._active_mask_source = 'segmentation'
        
        # Reset display parameters
        self.display_min_percentile = 1.0
        self.display_max_percentile = 99.95
        if hasattr(self, 'channelDisplayParams'):
            self.channelDisplayParams.clear()
        
        # Update tracking sliders if they exist
        if hasattr(self, 'min_percentile_slider_tracking'):
            self.update_tracking_sliders()
        
        # Reset current frame and channel indices
        self.current_frame = 0
        self.current_channel = 0

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
                    correlations_array                  = r['correlations_array'],
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
        channel_text = self.time_course_channel_combo.currentText()
        data_type = self.data_type_combo.currentText()
        lower_percentile = self.min_percentile_spinbox.value()
        upper_percentile = self.max_percentile_spinbox.value()
        normalize = self.normalize_time_course_checkbox.isChecked()
        window_size = self.moving_average_spinbox.value()

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
        
        # Calculate time points
        time_points = np.arange(0, total_frames * time_interval, time_interval)
        
        # Check if minutes are requested
        show_minutes = self.show_time_in_minutes_checkbox.isChecked()
        if show_minutes:
            time_points = time_points / 60.0
            x_label = "Time (min)"
        else:
            x_label = "Time (s)"

        # Helper to apply moving average
        def apply_moving_average(data_array, win_size):
            if win_size <= 1:
                return data_array
            # Use pandas rolling mean for convenience if available, or convolution
            # data_array is 1D
            s = pd.Series(data_array)
            # min_periods=1 ensures we get values from the start
            return s.rolling(window=win_size, min_periods=1).mean().values

        # Helper to get color for channel
        def get_channel_color(ch_idx):
            # 0=Magenta, 1=Green, 2=Yellow
            if ch_idx == 0: return 'magenta'
            if ch_idx == 1: return 'green'
            if ch_idx == 2: return 'yellow'
            # fallback
            return 'cyan'

        # Helper to plot one channel's data
        # Returns (min_val, max_val) for y-axis scaling
        def plot_channel_data(ch_idx, color_override=None):
            field_name = f"{data_type}_ch_{ch_idx}"
            if field_name not in self.df_tracking.columns:
                return None, None

            intensity_array = mi.Utilities().df_trajectories_to_array(
                dataframe=self.df_tracking,
                selected_field=field_name,
                fill_value=np.nan,
                total_frames=total_frames
            )

            # Normalize if requested
            if normalize:
                # Min-Max normalization per trace or global?
                # Usually for time course comparison, global min/max of the mean trace or per-trace?
                # Let's normalize the mean trace to 0-1 for visualization, 
                # OR normalize the entire array to [0,1] based on its global min/max.
                # User request: "Allow the user to normalize this data"
                # Let's normalize the mean curve to 0-1 range for clarity.
                # Actually, standard is usually (val - min) / (max - min).
                pass 

            # Plot individual traces if option is enabled (only if not "All" or maybe yes?)
            # If "All" is selected, individual traces might be too messy.
            # Let's disable individual traces for "All" to avoid clutter, or respect the checkbox.
            # The user didn't specify, but "All" usually implies comparing means.
            # Let's respect the checkbox but maybe with low alpha.
            
            if self.show_traces_checkbox.isChecked() and channel_text != "All":
                for idx in range(intensity_array.shape[0]):
                    trace = intensity_array[idx, :]
                    if np.all(np.isnan(trace)):
                        continue
                    # If normalizing, we'd need to normalize traces too? 
                    # Let's keep raw traces for now unless normalization is simple.
                    self.ax_time_course.plot(time_points, trace, '-', color='gray',
                                            linewidth=1, alpha=0.5, label='_nolegend_')

            # Calculate mean and std dev
            mean_time_intensity = np.nanmean(intensity_array, axis=0)
            std_time_intensity  = np.nanstd(intensity_array, axis=0)
            mean_time_intensity = np.nan_to_num(mean_time_intensity)
            std_time_intensity  = np.nan_to_num(std_time_intensity)

            # Apply Moving Average
            if window_size > 1:
                mean_time_intensity = apply_moving_average(mean_time_intensity, window_size)
                # Also smooth std dev? Or keep it raw? Usually smooth mean is enough.
                # Let's smooth std dev too to match the curve smoothness visually
                std_time_intensity = apply_moving_average(std_time_intensity, window_size)

            if normalize:
                # Normalize mean to 0-1
                min_v = np.min(mean_time_intensity)
                max_v = np.max(mean_time_intensity)
                if max_v > min_v:
                    mean_time_intensity = (mean_time_intensity - min_v) / (max_v - min_v)
                    # Scale std dev proportionally? Or just show mean?
                    # Std dev scaling: std = std / (max - min)
                    std_time_intensity = std_time_intensity / (max_v - min_v)

            color = color_override if color_override else 'cyan'
            label_text = f"Ch {ch_idx}" if channel_text == "All" else "Mean"
            
            self.ax_time_course.plot(time_points, mean_time_intensity, 'o-',
                                    color=color, linewidth=2, label=label_text, alpha=0.8, zorder=3)
            
            # Fill between for std dev (maybe skip for "All" to reduce clutter? or use low alpha)
            self.ax_time_course.fill_between(time_points,
                                            mean_time_intensity - std_time_intensity,
                                            mean_time_intensity + std_time_intensity,
                                            color=color, alpha=0.1, label='_nolegend_', zorder=1)
            
            # Return range for axis scaling
            if normalize:
                return 0.0, 1.0
            else:
                lower_y = np.nanpercentile(intensity_array, lower_percentile)
                upper_y = np.nanpercentile(intensity_array, upper_percentile)
                return lower_y, upper_y

        # --- Plotting Logic ---
        
        if data_type == "particles":
            # Particles is universal.
            particles_per_frame = self.df_tracking.groupby('frame')['particle'].nunique()
            all_frames = np.arange(total_frames)
            particles_per_frame = particles_per_frame.reindex(all_frames, fill_value=0)
            
            y_data = particles_per_frame.values.astype(float)
            
            # Apply Moving Average
            if window_size > 1:
                y_data = apply_moving_average(y_data, window_size)

            if normalize:
                min_v = np.min(y_data)
                max_v = np.max(y_data)
                if max_v > min_v:
                    y_data = (y_data - min_v) / (max_v - min_v)
            
            self.ax_time_course.plot(time_points, y_data, 'o-', color='orangered', linewidth=2, label="Particles")
            self.ax_time_course.set_title("Number of Particles vs Time", fontsize=10, color='white')
            
            if normalize:
                 self.ax_time_course.set_ylim([-0.1, 1.1])
            else:
                 max_particles = particles_per_frame.max()
                 self.ax_time_course.set_ylim([0, max_particles + 1])

        else:
            # Intensity/Size/etc data
            if channel_text == "All":
                # Plot all channels
                y_mins = []
                y_maxs = []
                
                # We need to know which channels exist. 
                # self.number_color_channels should hold this.
                num_ch = getattr(self, 'number_color_channels', 1)
                
                for ch in range(num_ch):
                    c_color = get_channel_color(ch)
                    l_y, u_y = plot_channel_data(ch, color_override=c_color)
                    if l_y is not None:
                        y_mins.append(l_y)
                        y_maxs.append(u_y)
                
                self.ax_time_course.set_title(f"{data_type.capitalize()} vs Time (All Channels)", fontsize=10, color='white')
                
                if y_mins and y_maxs:
                    if normalize:
                        self.ax_time_course.set_ylim([-0.1, 1.1])
                    else:
                        # Find global min/max for axis
                        global_min = min(y_mins)
                        global_max = max(y_maxs)
                        y_range = global_max - global_min
                        self.ax_time_course.set_ylim([global_min - 0.1 * y_range, global_max + 0.1 * y_range])

            else:
                # Single channel
                ch_idx = int(channel_text)
                l_y, u_y = plot_channel_data(ch_idx, color_override='cyan')
                
                self.ax_time_course.set_title(f"{data_type.capitalize()} vs Time (Channel {ch_idx})", fontsize=10, color='white')
                
                if l_y is not None:
                    if normalize:
                        self.ax_time_course.set_ylim([-0.1, 1.1])
                    else:
                        y_range = u_y - l_y
                        self.ax_time_course.set_ylim([l_y - 0.1 * y_range, u_y + 0.1 * y_range])

        self.ax_time_course.set_xlabel(x_label, color='white')
        ylabel = f"{data_type.capitalize()} (Normalized)" if normalize else f"{data_type.capitalize()} (au)"
        if data_type == "particles" and not normalize:
             ylabel = "Number of Particles"
        self.ax_time_course.set_ylabel(ylabel, color='white')
        
        self.ax_time_course.set_xlim([time_points[0], time_points[-1]])
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
            self.plot_cellpose_results()
        elif index == 3:
            self.plot_photobleaching()
        elif index == 4:
            self.plot_tracking()
        elif index == 5:
            self.plot_distribution()
        elif index == 6:
            pass
        elif index == 7:
            self.display_correlation_plot()
        elif index == 8:
            self.display_colocalization_plot()
            if hasattr(self, 'canvas_colocalization'):
                if hasattr(self, 'cid_zoom_coloc'):
                    try:
                        self.canvas_colocalization.mpl_disconnect(self.cid_zoom_coloc)
                    except Exception:
                        pass
                self.cid_zoom_coloc = self.canvas_colocalization.mpl_connect('motion_notify_event', self.on_colocalization_hover)
        elif index == 9:
            self.display_colocalization_manual()
        elif index == 10:
            if not (getattr(self, 'has_tracked', False)) or self.df_tracking.empty:
                QMessageBox.warning(self, "No Data", "Please perform particle tracking first.")
                self.tabs.setCurrentIndex(4)
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
        elif index == 11:
            pass
        elif index == 12:
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