"""
micro.py: is a library designed to process live-cell microscope images and perform single-molecule measurements. 
Author: Luis Aguilera
"""

# =============================================================================
# IMPORTS AND GLOBAL CONFIGURATION
# =============================================================================

import sys
import os
import gc
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

# Suppress macOS native warnings
if sys.platform == 'darwin':
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

# Package-aware imports (pip packaging compatible)
from microlive.imports import *
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
from skimage.segmentation import find_boundaries
from scipy.ndimage import center_of_mass
import matplotlib.colors as mcolors
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
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from functools import partial
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, label, center_of_mass, distance_transform_edt
from scipy.stats import linregress
import trackpy as tp
from trackpy.linking.utils import SubnetOversizeException
vispy_logging = None
try:
    from vispy import logging as vispy_logging
except ImportError:
    pass


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
        
        # Add subplot with no padding to maximize image display area
        ax = fig.add_subplot(111)
        ax.imshow(big_image)
        ax.axis('off')
        
        # Full-bleed layout - remove all padding to fill available space
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.patch.set_facecolor('black')


    def plot_autocorrelation(self, mean_correlation, error_correlation, lags, correlations_array=None,
                            time_interval_between_frames_in_seconds=1, channel_label=0,
                            index_max_lag_for_fit=None, start_lag=0, line_color='blue',
                            plot_title=None, fit_type='linear', de_correlation_threshold=0.05,
                            normalize_plot_with_g0=False, axes=None, max_lag_index=None, plot_individual_trajectories=False,
                            y_min_percentile=None, y_max_percentile=None, verbose=False):
        
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
                logging.debug(f"Decorrelation threshold value: {de_correlation_threshold_value}")
            except Exception as e:
                logging.debug(f"Could not find the decorrelation point automatically: {e}")
                # Fall back to the last correlation point
                index_max_lag_for_fit = normalized_correlation.shape[0]
                de_correlation_threshold_value = normalized_correlation[index_max_lag_for_fit - 1]
                logging.debug(f"Falling back to last point: {de_correlation_threshold_value}")
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
            if verbose:
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
                if verbose:
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
                if verbose:
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
                
                # Registration
                write_subsection('Image Registration')
                registration_used = self.registered_image is not None
                write_value('Registration Applied', 'Yes' if registration_used else 'No')
                if registration_used:
                    write_value('Registration Mode', self.registration_mode)
                    if self.registration_roi:
                        write_value('ROI Bounds (y_min, y_max, x_min, x_max)', str(self.registration_roi))
                
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
                
                # Report Z-slice used for segmentation
                z_used = getattr(self, 'segmentation_z_used_for_mask', -1)
                z_max = getattr(self, 'segmentation_z_max', 0)
                if z_used == -3:
                    z_info = 'Imported Mask'
                elif z_used == -2:
                    z_info = 'Cellpose Z-Optimized'
                elif z_used == -1:
                    z_info = 'Max Z-Projection'
                else:
                    z_info = f'{z_used} (of {z_max})' if z_max > 0 else str(z_used)
                write_value('Segmentation Z-Slice', z_info)
                
                # Check if masks were imported or generated by Cellpose
                masks_imported = getattr(self, 'masks_imported', False)
                has_cellpose_cyto = self.cellpose_masks_cyto is not None
                has_cellpose_nuc = self.cellpose_masks_nuc is not None
                
                if masks_imported and (has_cellpose_cyto or has_cellpose_nuc):
                    write_subsection('Imported Masks')
                    write_value('Source', 'Imported from TIFF files')
                    
                    # Check if TYX masks
                    use_tyx = getattr(self, 'use_tyx_masks', False)
                    write_value('Time-Varying (TYX)', 'Yes' if use_tyx else 'No (static YX)')
                    
                    if has_cellpose_cyto:
                        n_cells_cyto = int(self.cellpose_masks_cyto.max())
                        if use_tyx and self.cellpose_masks_cyto_tyx is not None:
                            write_value('Cytosol Mask', f'Yes ({n_cells_cyto} cells, {self.cellpose_masks_cyto_tyx.shape[0]} frames)')
                        else:
                            write_value('Cytosol Mask', f'Yes ({n_cells_cyto} cells)')
                    else:
                        write_value('Cytosol Mask', 'No')
                    
                    if has_cellpose_nuc:
                        n_cells_nuc = int(self.cellpose_masks_nuc.max())
                        if use_tyx and self.cellpose_masks_nuc_tyx is not None:
                            write_value('Nucleus Mask', f'Yes ({n_cells_nuc} cells, {self.cellpose_masks_nuc_tyx.shape[0]} frames)')
                        else:
                            write_value('Nucleus Mask', f'Yes ({n_cells_nuc} cells)')
                    else:
                        write_value('Nucleus Mask', 'No')
                else:
                    write_subsection('Cellpose Segmentation')
                    
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
                # Add decay rates per channel if available
                if hasattr(self, 'photobleaching_data') and self.photobleaching_data is not None:
                    decay_rates = self.photobleaching_data.get('decay_rates', [])
                    if decay_rates is not None and len(decay_rates) > 0:
                        num_channels = len(decay_rates) // 2
                        for ch in range(num_channels):
                            k_fit = decay_rates[2 * ch]
                            I0_fit = decay_rates[2 * ch + 1]
                            write_value(f'Channel {ch} Decay Rate (k)', f'{k_fit:.6e}')
                            write_value(f'Channel {ch} Initial Intensity (I0)', f'{I0_fit:.2f}')
                
                # Tracking Parameters
                write_section('Tracking Parameters')
                
                # Report tracked channels summary
                tracked_channels = getattr(self, 'tracked_channels', [])
                if tracked_channels:
                    write_value('Tracked Channels', ', '.join([str(ch) for ch in sorted(tracked_channels)]))
                
                # Per-channel tracking parameters
                params_per_channel = getattr(self, 'tracking_parameters_per_channel', {})
                if params_per_channel:
                    write_subsection('Per-Channel Parameters')
                    for ch in sorted(params_per_channel.keys()):
                        params = params_per_channel[ch]
                        # Show if this was detection-only or full tracking
                        is_detection_only = params.get('detection_only', False)
                        analysis_type = 'Detection Only (no linking)' if is_detection_only else 'Full Tracking (with linking)'
                        write_value(f'Channel {ch} Analysis Type', analysis_type)
                        write_value(f'Channel {ch} Threshold', f"{params.get('threshold', 'N/A')}")
                        write_value(f'Channel {ch} YX Spot Size (px)', f"{params.get('yx_spot_size_in_px', 'N/A')}")
                        write_value(f'Channel {ch} Z Spot Size (px)', f"{params.get('z_spot_size_in_px', 'N/A')}")
                        write_value(f'Channel {ch} Cluster Radius (nm)', f"{params.get('cluster_radius_nm', 'N/A')}")
                        write_value(f'Channel {ch} Max Cluster Size', f"{params.get('maximum_spots_cluster', 'N/A')}")
                        write_value(f'Channel {ch} Max Range Search (px)', f"{params.get('maximum_range_search_pixels', 'N/A')}")
                        write_value(f'Channel {ch} Memory', f"{params.get('memory', 'N/A')}")
                        write_value(f'Channel {ch} Min Trajectory Length', f"{params.get('min_length_trajectory', 'N/A')}")
                        mode = '2D Projection' if params.get('use_maximum_projection', False) else '3D Volume'
                        write_value(f'Channel {ch} Tracking Mode', mode)
                        fd.write('\n')  # Blank line between channels
                else:
                    # Fallback to current parameters if no per-channel data
                    write_subsection('Spot Detection')
                    
                    # Report the final threshold used for tracking
                    final_threshold = getattr(self, 'user_selected_threshold', None)
                    if final_threshold is not None and final_threshold > 0:
                        write_value('Final Threshold for Tracking', f'{int(final_threshold)}')
                    else:
                        write_value('Final Threshold for Tracking', 'Not set (use Auto)')
                    
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
                # Report multi-channel tracking if available
                if hasattr(self, 'tracked_channels') and self.tracked_channels:
                    write_value('Tracked Channels', str(self.tracked_channels))
                else:
                    # Fall back to legacy single-channel attribute
                    write_attr('Spot Detection Channel', 'channels_spots')
                write_attr('Cytosol Channel', 'channels_cytosol')
                write_attr('Nucleus Channel', 'channels_nucleus')
                
                write_subsection('Options')
                write_attr('Use Fixed Spot Size for Intensity', 'use_fixed_size_for_intensity_calculation')
                write_attr('Fast Gaussian Fit', 'fast_gaussian_fit')
                if self.use_maximum_projection:
                    write_value('Projection Mode', '2D Maximum Projection (Trackpy)')
                else:
                    write_value('Projection Mode', '3D (Big-FISH + Trackpy)')
                combo_val = getattr(self, 'image_source_combo', '')
                using_corrected = 'Yes' if 'Corrected' in str(combo_val) else 'No'
                write_value('Using Photobleaching Corrected Image', using_corrected)
                
                # MSD Results (from dedicated MSD tab)
                write_subsection('MSD Results')
                msd_ch = getattr(self, 'tracking_msd_channel', None)
                msd_mode = getattr(self, 'tracking_msd_mode', None)
                msd_D = getattr(self, 'tracking_D_um2_s', None)
                msd_D_px = getattr(self, 'tracking_D_px2_s', None)
                if msd_ch is not None:
                    write_value('MSD Calculated for Channel', msd_ch)
                if msd_mode is not None:
                    write_value('MSD Mode', msd_mode)
                if msd_D is not None:
                    write_value('Diffusion Coefficient (µm²/s)', f'{msd_D:.4e}')
                if msd_D_px is not None:
                    write_value('Diffusion Coefficient (px²/s)', f'{msd_D_px:.4e}')
                if msd_ch is None and msd_D is None:
                    write_value('Note', 'MSD not yet calculated in MSD tab')
                
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
                
                write_subsection('Visual (ML/Intensity)')
                write_attr('Method', 'colocalization_method')
                write_attr('Threshold Value', 'colocalization_threshold_value')
                write_attr('ML Threshold', 'ml_threshold_input')
                
                # Distance Colocalization
                write_subsection('Distance-Based')
                dist_results = getattr(self, 'distance_coloc_results', None)
                if dist_results:
                    ch0 = dist_results.get('channel_0', 'N/A')
                    ch1 = dist_results.get('channel_1', 'N/A')
                    threshold_px = dist_results.get('threshold_distance_px', 'N/A')
                    threshold_nm = dist_results.get('threshold_distance_nm', 'N/A')
                    use_3d = dist_results.get('use_3d', False)
                    
                    write_value('Analysis Performed', 'Yes')
                    write_value('Channel 0', ch0)
                    write_value('Channel 1', ch1)
                    write_value('Distance Threshold (px)', threshold_px)
                    write_value('Distance Threshold (nm)', threshold_nm)
                    write_value('3D Distance', 'Yes' if use_3d else 'No')
                    
                    # Add summary statistics if available
                    df_class = dist_results.get('df_classification')
                    if df_class is not None and len(df_class) > 0:
                        total_coloc = int(df_class['num_0_1'].sum())
                        total_ch0_only = int(df_class['num_0_only'].sum())
                        total_ch1_only = int(df_class['num_1_only'].sum())
                        total_unique = total_ch0_only + total_ch1_only + total_coloc
                        pct_coloc = 100 * total_coloc / total_unique if total_unique > 0 else 0
                        
                        write_value('Colocalized Spots', total_coloc)
                        write_value(f'Ch{ch0} Only', total_ch0_only)
                        write_value(f'Ch{ch1} Only', total_ch1_only)
                        write_value('Total Unique Spots', total_unique)
                        write_value('Colocalization %', f'{pct_coloc:.2f}%')
                else:
                    write_value('Analysis Performed', 'No')
                
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
    Correlation, Colocalization (automated and manual), Tracking Visualization, and Export. 
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
        • Export
            – Batch export of images, masks, metadata, user comments, and data tables into structured result folders.
    """
    
    def __init__(self, icon_path):
        super().__init__()
        configure_logging_and_styles()
        self.setWindowTitle("MicroLive")
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
        self.memory = 0
        self.de_correlation_threshold = 0.01
        self.max_spots_for_threshold = 3000
        self.index_max_lag_for_fit = None
        self.threshold_spot_detection = 0
        self.user_selected_threshold = 0.0
        self.auto_threshold_per_channel = {}  # {channel: threshold} - auto-detected thresholds
        self.image_source_combo_value = "Original Image"
        self.segmentation_mode = "None"
        self.use_fixed_size_for_intensity_calculation = True
        self.fast_gaussian_fit = True  # Use fast moment-based PSF estimation by default
        # Registration state
        self.registered_image = None  # [T,Z,Y,X,C] registered image
        self.registration_roi = None  # (y_min, y_max, x_min, x_max)
        self.registration_mode = 'RIGID_BODY'
        self.display_max_percentile = 99.95
        self.display_min_percentile = 0.1
        self.tracking_min_percentile = 0.05   # self.display_min_percentile
        self.tracking_max_percentile = 99.95  # self.display_max_percentile
        self.display_sigma = 0.7
        self.low_display_sigma = 0.15
        self.correlation_fit_type = 'linear'
        # Independent timers and state for each tab's playback
        self.timer_display = QTimer()
        self.timer_display.timeout.connect(self.next_frame_display)
        self.playing_display = False
        
        # Segmentation tab timer (handles Manual, Watershed, and Cellpose sub-tabs)
        self.timer_segmentation = QTimer()
        self.timer_segmentation.timeout.connect(self.next_frame_segmentation)
        self.playing_segmentation = False
        
        self.timer_tracking = QTimer()
        self.timer_tracking.timeout.connect(self.next_frame_tracking)
        self.playing_tracking = False
        
        self.timer_tracking_vis = QTimer()
        self.timer_tracking_vis.timeout.connect(self.next_frame_tracking_vis)
        self.playing_tracking_vis = False
        
        # Legacy compatibility (some code may reference these)
        self.timer = self.timer_display
        self.playing = False
        self.photobleaching_calculated = False
        self.df_tracking = pd.DataFrame()
        self.has_tracked = False
        
        # Multi-channel tracking storage
        self.multi_channel_tracking_data = {}  # Dict: {channel_index: DataFrame}
        self.tracked_channels = []  # List of channel indices that have been tracked
        self.tracking_thresholds = {}  # Dict: {channel_index: threshold_value}
        self.tracking_parameters_per_channel = {}  # Dict: {channel_index: parameters_dict}
        self.primary_tracking_channel = None  # First channel tracked (for default selection)
        
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
        
        # Tracking tab zoom feature - ROI for visualization
        self.tracking_zoom_roi = None  # (x_min, x_max, y_min, y_max) or None for full view
        self.tracking_zoom_selector = None  # RectangleSelector instance
        
        # Import/Display tab zoom feature - ROI for visualization
        self.display_zoom_roi = None  # (x_min, x_max, y_min, y_max) or None for full view
        self.display_zoom_selector = None  # RectangleSelector instance
        
        # Visualization tab zoom feature - ROI for visualization
        self.tracking_vis_zoom_roi = None  # (x_min, x_max, y_min, y_max) or None for full view
        self.tracking_vis_zoom_selector = None  # RectangleSelector instance
        
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
        When TYX mode is active, returns the mask for the current frame.
        """
        if self._active_mask_source == 'cellpose':
            # Check for TYX masks first - return current frame's mask
            if getattr(self, 'use_tyx_masks', False):
                if hasattr(self, 'cellpose_masks_cyto_tyx') and self.cellpose_masks_cyto_tyx is not None:
                    return (self.cellpose_masks_cyto_tyx[self.current_frame] > 0).astype(np.uint8)
                elif hasattr(self, 'cellpose_masks_nuc_tyx') and self.cellpose_masks_nuc_tyx is not None:
                    return (self.cellpose_masks_nuc_tyx[self.current_frame] > 0).astype(np.uint8)
            # Fallback to YX masks
            if self.cellpose_masks_cyto is not None:
                return (self.cellpose_masks_cyto > 0).astype(np.uint8)
            elif self.cellpose_masks_nuc is not None:
                return (self.cellpose_masks_nuc > 0).astype(np.uint8)
        return self.segmentation_mask

    def _format_time_interval(self, value):
        """Format time interval for display, handling small values appropriately.
        
        Args:
            value: Time interval in seconds
            
        Returns:
            Formatted string with appropriate unit (ms or s)
        """
        if value is None:
            return "N/A"
        try:
            val = float(value)
            if val < 0.001:
                # Sub-millisecond: show in microseconds
                return f"{val * 1e6:.2f} µs"
            elif val < 1:
                # Millisecond range: show in ms
                return f"{val * 1000:.2f} ms"
            elif val < 10:
                # 1-10 seconds: show with 2 decimals
                return f"{val:.2f} s"
            else:
                # 10+ seconds: show with 1 decimal
                return f"{val:.1f} s"
        except (TypeError, ValueError):
            return "N/A"

    def _get_tracking_masks(self):
        """
        Prepares masks for tracking based on available segmentation data.
        
        Returns TYX [T,Y,X] masks when TYX mode is active, otherwise YX [Y,X].
        ParticleTracking normalizes to TYX internally for backward compatibility.
        """
        
        if self._active_mask_source == 'cellpose':
            # Check if TYX masks are active
            if getattr(self, 'use_tyx_masks', False):
                masks_cyto = getattr(self, 'cellpose_masks_cyto_tyx', None)
                masks_nuc = getattr(self, 'cellpose_masks_nuc_tyx', None)
                
                if masks_cyto is not None and masks_nuc is not None:
                    # Compute cytosol_no_nuclei as TYX (per-frame overlap removal)
                    masks_cytosol_no_nuclei = masks_cyto.copy()
                    overlap = (masks_nuc > 0) & (masks_cyto > 0)
                    masks_cytosol_no_nuclei[overlap] = 0
                    
                    return masks_cyto, masks_nuc, masks_cytosol_no_nuclei
                elif masks_cyto is not None:
                    return masks_cyto, None, None
                elif masks_nuc is not None:
                    return masks_nuc, masks_nuc, None
            else:
                # Standard YX masks
                masks_cyto = self.cellpose_masks_cyto
                masks_nuc = self.cellpose_masks_nuc
                
                if masks_cyto is not None and masks_nuc is not None:
                    masks_cytosol_no_nuclei = masks_cyto.copy()
                    overlap_mask = (masks_nuc > 0) & (masks_cyto > 0)
                    masks_cytosol_no_nuclei[overlap_mask] = 0
                    return masks_cyto, masks_nuc, masks_cytosol_no_nuclei
                elif masks_cyto is not None:
                    return masks_cyto, None, None
                elif masks_nuc is not None:
                    return masks_nuc, masks_nuc, None
        
        # Fallback: segmentation mask (binary) - always YX
        # Fallback: segmentation mask (binary) - always YX
        if self.segmentation_mask is not None:
            return self.segmentation_mask, None, None
        
        # No masks at all
        return None, None, None

    def _get_validated_voxels(self):
        """
        Returns validated voxel sizes [voxel_z_nm, voxel_yx_nm] with defaults for None/NaN.
        
        Returns
        -------
        list: [voxel_z, voxel_yx] with validated values (defaults: 500, 160)
        """
        voxel_z = self.voxel_z_nm if (self.voxel_z_nm is not None and 
                                       not np.isnan(self.voxel_z_nm) and 
                                       self.voxel_z_nm > 0) else 500
        voxel_yx = self.voxel_yx_nm if (self.voxel_yx_nm is not None and 
                                         not np.isnan(self.voxel_yx_nm) and 
                                         self.voxel_yx_nm > 0) else 160
        return [voxel_z, voxel_yx]

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
        self.registration_tab = QWidget()
        self.tabs.addTab(self.registration_tab, "Registration")
        self.segmentation_tab = QWidget()
        self.tabs.addTab(self.segmentation_tab, "Segmentation")
        # NOTE: Cellpose is now a sub-tab within the Segmentation tab
        self.photobleaching_tab = QWidget()
        self.tabs.addTab(self.photobleaching_tab, "Photobleaching")
        self.tracking_tab = QWidget()
        self.tabs.addTab(self.tracking_tab, "Tracking")
        self.msd_tab = QWidget()
        self.tabs.addTab(self.msd_tab, "MSD")
        self.distribution_tab = QWidget()
        self.tabs.addTab(self.distribution_tab, "Distribution")
        self.time_course_tab = QWidget()
        self.tabs.addTab(self.time_course_tab, "Time Course")
        self.correlation_tab = QWidget()
        self.tabs.addTab(self.correlation_tab, "Correlation")
        self.colocalization_tab = QWidget()
        self.tabs.addTab(self.colocalization_tab, "Colocalization")
        # Note: Coloc Manual is now a sub-tab within Colocalization (Phase 1 consolidation)
        self.tracking_visualization_tab = QWidget()
        self.tabs.addTab(self.tracking_visualization_tab, "Visualization")
        self.export_tab = QWidget()
        self.tabs.addTab(self.export_tab, "Export")
        self.tabs.currentChanged.connect(self.on_tab_change)
        self.setup_display_tab()
        self.setup_registration_tab()
        self.setup_segmentation_tab()  # Includes Cellpose as sub-tab
        self.setup_photobleaching_tab()
        self.setup_tracking_tab()
        self.setup_msd_tab()
        self.setup_tracking_visualization_tab()
        self.setup_distributions_tab()
        self.setup_time_course_tab()
        self.setup_correlation_tab()
        self.setup_colocalization_tab()  # Includes Visual, Distance, and Manual sub-tabs
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

            /* Buttons: contrast on dark background with subtle gradient */
            QPushButton {
                background: qlineargradient(y1:0, y2:1, stop:0 #d0d0d0, stop:1 #b0b0b0);
                color: #000000;               /* black text */
                border: 1px solid #a0a0a0;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: qlineargradient(y1:0, y2:1, stop:0 #e0e0e0, stop:1 #c0c0c0);
            }
            QPushButton:pressed {
                background: qlineargradient(y1:0, y2:1, stop:0 #a0a0a0, stop:1 #909090);
            }
            QPushButton:checked {
                background: qlineargradient(y1:0, y2:1, stop:0 #0090e0, stop:1 #007acc);
                color: #ffffff;
                border: 1px solid #0070a0;
            }
            /* Inputs with focus indicator */
            QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QComboBox {
                background-color: #3a3a3a;
                color: #e0e0e0;
                border: 1px solid #5a5a5a;
                border-radius: 4px;
            }
            QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {
                border: 2px solid #007acc;
            }
            
            /* Tooltips */
            QToolTip {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #007acc;
                border-radius: 4px;
                padding: 6px;
                font-size: 11px;
            }
            
            /* Tab Widget Styling */
            QTabWidget::pane {
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px;
            }
            QTabBar::tab {
                background: #3a3a3a;
                color: #b0b0b0;
                padding: 10px 16px;
                border: 1px solid #555555;
                border-bottom: none;
                border-radius: 4px 4px 0 0;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #2b2b2b;
                color: #ffffff;
                border-bottom: 2px solid #007acc;
            }
            QTabBar::tab:hover:!selected {
                background: #4a4a4a;
                color: #e0e0e0;
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

            /* Buttons: contrast on light background with subtle gradient */
            QPushButton {
                background: qlineargradient(y1:0, y2:1, stop:0 #505050, stop:1 #404040);
                color: #ffffff;               /* white text */
                border: 1px solid #303030;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: qlineargradient(y1:0, y2:1, stop:0 #606060, stop:1 #505050);
            }
            QPushButton:pressed {
                background: qlineargradient(y1:0, y2:1, stop:0 #353535, stop:1 #303030);
            }
            QPushButton:checked {
                background: qlineargradient(y1:0, y2:1, stop:0 #0090e0, stop:1 #007acc);
                color: #ffffff;
                border: 1px solid #0070a0;
            }
            /* Inputs with focus indicator */
            QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QComboBox {
                background-color: #ffffff;
                color: #2b2b2b;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {
                border: 2px solid #007acc;
            }
            
            /* Tooltips */
            QToolTip {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #007acc;
                border-radius: 4px;
                padding: 6px;
                font-size: 11px;
            }
            
            /* Tab Widget Styling */
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 4px;
            }
            QTabBar::tab {
                background: #e8e8e8;
                color: #555555;
                padding: 8px 16px;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-radius: 4px 4px 0 0;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #2b2b2b;
                border-bottom: 2px solid #007acc;
            }
            QTabBar::tab:hover:!selected {
                background: #f5f5f5;
                color: #2b2b2b;
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
            self.correlation_tab, self.colocalization_tab,
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
                default_x = float(self.voxel_yx_nm) if isinstance(self.voxel_yx_nm, (int, float)) and self.voxel_yx_nm is not None else 100.0
                val, ok = QInputDialog.getDouble(
                    self,
                    "Missing Metadata",
                    "Enter voxel size X (nm):",
                    default_x,
                    0.1,
                    1e6,
                    3
                )
                # Set value even if cancelled - use default
                self.voxel_yx_nm = val if ok else default_x
                self.voxel_size_x_nm = self.voxel_yx_nm
                self.voxel_size_y_nm = self.voxel_yx_nm
            elif "voxel size Y" in field:
                default_y = float(self.voxel_size_y_nm) if isinstance(self.voxel_size_y_nm, (int, float)) and self.voxel_size_y_nm is not None else (float(self.voxel_yx_nm) if isinstance(self.voxel_yx_nm, (int, float)) and self.voxel_yx_nm is not None else 100.0)
                val, ok = QInputDialog.getDouble(
                    self,
                    "Missing Metadata",
                    "Enter voxel size Y (nm):",
                    default_y,
                    0.1,
                    1e6,
                    3
                )
                self.voxel_size_y_nm = val if ok else default_y
            elif "voxel size Z" in field:
                default_z = float(self.voxel_z_nm) if isinstance(self.voxel_z_nm, (int, float)) and self.voxel_z_nm is not None else 500.0
                val, ok = QInputDialog.getDouble(
                    self,
                    "Missing Metadata",
                    "Enter voxel size Z (nm):",
                    default_z,
                    0.1,
                    1e6,
                    3
                )
                # Set value even if cancelled - use default
                self.voxel_z_nm = val if ok else default_z
                self.voxel_size_z_nm = self.voxel_z_nm
            elif "time increment" in field or "TimeIncrement" in field:
                default_t = float(self.time_interval_value) if isinstance(self.time_interval_value, (int, float)) and self.time_interval_value is not None else 1.0
                val, ok = QInputDialog.getDouble(
                    self,
                    "Missing Metadata",
                    "Enter time increment (s):",
                    default_t,
                    1e-6,
                    1e6,
                    6
                )
                self.time_interval_value = val if ok else default_t

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
            button = QPushButton(f"Ch {idx}", self)
            button.clicked.connect(partial(self.update_channel, idx))
            self.channel_buttons_layout_display.addWidget(button)
            self.channel_buttons_display.append(button)
        for btn in self.channel_buttons_tracking:
            btn.setParent(None)
        self.channel_buttons_tracking = []
        for idx, channel_name in enumerate(self.channel_names):
            button = QPushButton(f"Ch {idx}", self)
            button.clicked.connect(partial(self.update_channel, idx))
            self.channel_buttons_layout_tracking.addWidget(button)
            self.channel_buttons_tracking.append(button)
        for btn in getattr(self, 'channel_buttons_tracking_vis', []):
            btn.setParent(None)
        self.channel_buttons_tracking_vis = []
        for idx, channel_name in enumerate(self.channel_names):
            btn = QPushButton(f"Ch {idx}", self)
            btn.clicked.connect(partial(self.select_tracking_vis_channel, idx))
            self.channel_buttons_layout_tracking_vis.addWidget(btn)
            self.channel_buttons_tracking_vis.append(btn)
        # Note: Crops tab channel buttons removed - Crops tab has been deprecated
        # Create registration tab channel buttons
        if hasattr(self, 'channel_buttons_reg'):
            for btn in self.channel_buttons_reg:
                btn.setParent(None)
        self.channel_buttons_reg = []
        for idx, channel_name in enumerate(self.channel_names):
            button = QPushButton(f"Ch {idx}", self)
            button.clicked.connect(partial(self.update_registration_channel, idx))
            self.channel_buttons_layout_reg.addWidget(button)
            self.channel_buttons_reg.append(button)


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
        
        # Auto-adjust min_length_trajectory based on movie length
        optimal_min_traj = self._calculate_optimal_min_trajectory(T)
        self.min_length_trajectory = optimal_min_traj
        if hasattr(self, 'min_length_input'):
            self.min_length_input.blockSignals(True)
            self.min_length_input.setValue(optimal_min_traj)
            self.min_length_input.blockSignals(False)
        
        # Set time slider maximums for all tabs
        self.time_slider_display.setMaximum(T - 1)
        self.time_slider_display.setValue(0)
        self.time_slider_tracking.setMaximum(T - 1)
        self.time_slider_tracking.setValue(0)
        self.time_slider_tracking_vis.setMaximum(T - 1)
        self.time_slider_tracking_vis.setValue(0)
        self.segmentation_time_slider.setMaximum(T - 1)
        # NOTE: Cellpose now shares segmentation_time_slider, no separate slider needed
        
        # Setup registration tab time slider
        if hasattr(self, 'time_slider_reg'):
            self.time_slider_reg.setMaximum(T - 1)
            self.time_slider_reg.setValue(0)
        
        # Initialize all frame labels with correct values
        frame_text = f"0/{T - 1}"
        if hasattr(self, 'frame_label_display'):
            self.frame_label_display.setText(frame_text)
        if hasattr(self, 'frame_label_tracking'):
            self.frame_label_tracking.setText(frame_text)
        if hasattr(self, 'frame_label_tracking_vis'):
            self.frame_label_tracking_vis.setText(frame_text)
        if hasattr(self, 'frame_label_segmentation'):
            self.frame_label_segmentation.setText(frame_text)
        if hasattr(self, 'frame_label_reg'):
            self.frame_label_reg.setText(frame_text)
        
        # Reset registration state when loading new image
        self.reset_registration_state()
        # Display image in registration panel
        if hasattr(self, 'plot_registration_panels'):
            self.plot_registration_panels()
        
        # Initialize segmentation Z-slider from image dimensions
        self.reset_segmentation_z_slider()

        # Reset TYX mask spinbox and validate against image timepoints
        # Update Cellpose TYX sliders for new image
        if hasattr(self, '_update_cellpose_sliders_for_image'):
            self._update_cellpose_sliders_for_image(T)
        
        # Enable display controls
        self.set_display_controls_enabled(True)
        self.playing = False
        self.play_button_display.setText("Play")
        
        # Create channel buttons for all tabs
        # NOTE: Cellpose shares the segmentation channel buttons
        self.create_channel_buttons()
        self.create_segmentation_channel_buttons()
        self.create_correlation_channel_checkboxes()
        self.populate_colocalization_channels()
        
        # Note: Cellpose channel spinboxes removed - channel is now determined by left panel selection
        # Note: Crops channel buttons removed - Crops tab has been deprecated
        
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
        self.stop_all_playback()
        
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
                    # Check if it's ImageJ format (starts with "ImageJ=")
                    if desc_stripped.startswith('ImageJ='):
                        # Parse ImageJ metadata
                        # ImageJ doesn't store XY pixel size in description, only spacing (Z)
                        # We'll try to get pixel size from XResolution tag later
                        pass  # ImageJ metadata doesn't contain pixel size in description
                    elif desc_stripped.startswith('<') or desc_stripped.startswith('<?xml'):
                        # OME-XML metadata (starts with < or <?xml)
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
                            # Not valid XML - that's okay, will try XResolution tags
                            pass
            else:
                print("No ImageDescription found in TIFF metadata.")
            # Try to get pixel size from XResolution tag if not found yet
            if voxel_x_nm is None:
                x_res = page0.tags.get('XResolution')
                if x_res:
                    num, den = x_res.value
                    resolution = float(num) / float(den)  # pixels per unit
                    # Check ResolutionUnit: 1=None, 2=inch, 3=centimeter
                    res_unit = page0.tags.get('ResolutionUnit')
                    res_unit_val = res_unit.value if res_unit else 1
                    if resolution > 0:
                        # XResolution is pixels per unit, so pixel size = 1/XResolution
                        pixel_size_in_unit = 1.0 / resolution
                        if res_unit_val == 2:  # inch
                            candidate_nm = pixel_size_in_unit * 25400 * 1000  # inch to nm
                        elif res_unit_val == 3:  # centimeter
                            candidate_nm = pixel_size_in_unit * 10000 * 1000  # cm to nm
                        else:  # None or unknown - assume µm for microscopy images
                            candidate_nm = pixel_size_in_unit * 1000  # µm to nm
                        # Sanity check: microscopy pixel sizes are typically 10-5000 nm
                        # Reject values outside this range as likely corrupted/invalid
                        if 10 <= candidate_nm <= 5000:
                            voxel_x_nm = candidate_nm
            if voxel_z_nm is None:
                z_res = page0.tags.get('ZResolution')
                if z_res:
                    num, den = z_res.value
                    resolution = float(num) / float(den)
                    if resolution > 0:
                        candidate_nm = (1.0 / resolution) * 1000  # assume µm
                        # Sanity check for Z: 50-10000 nm is reasonable
                        if 50 <= candidate_nm <= 10000:
                            voxel_z_nm = candidate_nm
        # If essential metadata is missing, prompt user only for what's missing
        missing = []
        if voxel_x_nm is None:
            missing.append("voxel size X (nm)")
        if voxel_z_nm is None:
            missing.append("voxel size Z (nm)")
        if dt_seconds is None:
            missing.append("time increment (s)")
        if missing:
            # Only prompt for what's actually missing, keep found values
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
            self.channel_names = [f"Ch {i}" for i in range(C)]
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
        # Reset Z-label to default
        if hasattr(self, 'z_label_display'):
            self.z_label_display.setText("Max")
            self.z_label_display.setStyleSheet("color: cyan; font-weight: bold;")
        # Configure Tracking tab Z-slider (same range, defaults to max projection)
        if hasattr(self, 'z_slider_tracking'):
            self.z_slider_tracking.setMinimum(0)
            if Z > 1:
                self.z_slider_tracking.setMaximum(Z)
            else:
                self.z_slider_tracking.setMaximum(0)
            self.z_slider_tracking.setValue(Z if Z > 1 else 0)
        # Reset Tracking Z-label to default
        if hasattr(self, 'z_label_tracking'):
            self.z_label_tracking.setText("Max")
            self.z_label_tracking.setStyleSheet("color: cyan; font-weight: bold;")
        self.y_pixels_label.setText(str(Y))
        self.x_pixels_label.setText(str(X))
        self.channels_label.setText(str(C))
        self.voxel_yx_size_label.setText(f"{self.voxel_yx_nm:.0f} nm" if self.voxel_yx_nm is not None else "N/A")
        self.voxel_z_nm_label.setText(f"{self.voxel_z_nm:.0f} nm" if self.voxel_z_nm is not None else "N/A")
        self.bit_depth_label.setText(str(self.bit_depth))
        self.time_interval_label.setText(self._format_time_interval(self.time_interval_value))
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
        # Reset Z-label to default
        if hasattr(self, 'z_label_display'):
            self.z_label_display.setText("Max")
            self.z_label_display.setStyleSheet("color: cyan; font-weight: bold;")
        # Configure Tracking tab Z-slider (same range, defaults to max projection)
        if hasattr(self, 'z_slider_tracking'):
            self.z_slider_tracking.setMinimum(0)
            if Z > 1:
                self.z_slider_tracking.setMaximum(Z)
            else:
                self.z_slider_tracking.setMaximum(0)
            self.z_slider_tracking.setValue(Z if Z > 1 else 0)
        # Reset Tracking Z-label to default
        if hasattr(self, 'z_label_tracking'):
            self.z_label_tracking.setText("Max")
            self.z_label_tracking.setStyleSheet("color: cyan; font-weight: bold;")
        self.y_pixels_label.setText(str(Y))
        self.x_pixels_label.setText(str(X))
        self.channels_label.setText(str(self.number_color_channels))
        self.voxel_yx_size_label.setText(f"{self.voxel_yx_nm:.0f} nm" if self.voxel_yx_nm is not None else "N/A")
        self.voxel_z_nm_label.setText(f"{self.voxel_z_nm:.0f} nm" if self.voxel_z_nm is not None else "N/A")
        self.bit_depth_label.setText(str(self.bit_depth))
        self.time_interval_label.setText(self._format_time_interval(self.time_interval_value))
        self.laser_lines_label.setText(str(list_laser_lines[image_index]))
        self.intensities_label.setText(str(list_intensities[image_index]))
        self.wave_ranges_label.setText(str(list_wave_ranges[image_index]))
        self.selected_image_index = image_index
        # Setup UI for the new image
        T = self.image_stack.shape[0]
        C = self.number_color_channels
        self._setup_image_ui(T, C)

    def play_pause(self):
        """Legacy compatibility - redirects to display play/pause."""
        self.play_pause_display()
    
    def stop_all_playback(self):
        """Stop playback on all tabs. Called when switching tabs."""
        # Stop display timer
        if hasattr(self, 'timer_display'):
            self.timer_display.stop()
        self.playing_display = False
        if hasattr(self, 'play_button_display'):
            self.play_button_display.setText("Play")
        
        # Stop segmentation timer (handles all segmentation sub-tabs)
        if hasattr(self, 'timer_segmentation'):
            self.timer_segmentation.stop()
        self.playing_segmentation = False
        if hasattr(self, 'play_button_segmentation'):
            self.play_button_segmentation.setText("Play")
        
        # Stop tracking timer
        if hasattr(self, 'timer_tracking'):
            self.timer_tracking.stop()
        self.playing_tracking = False
        if hasattr(self, 'play_button_tracking'):
            self.play_button_tracking.setText("Play")
        
        # Stop tracking vis timer
        if hasattr(self, 'timer_tracking_vis'):
            self.timer_tracking_vis.stop()
        self.playing_tracking_vis = False
        if hasattr(self, 'play_button_tracking_vis'):
            self.play_button_tracking_vis.setText("Play")
        
        # Stop registration timer
        if hasattr(self, 'reg_timer'):
            self.reg_timer.stop()
        self.reg_playing = False
        if hasattr(self, 'play_button_reg'):
            self.play_button_reg.setText("▶")
        
        # Stop distance colocalization timer
        if hasattr(self, 'dist_play_timer'):
            self.dist_play_timer.stop()
        if hasattr(self, 'dist_play_button'):
            self.dist_play_button.setChecked(False)
            self.dist_play_button.setText("▶")
        
        # Legacy compatibility
        self.playing = False
    
    def play_pause_display(self):
        """Toggle play/pause for Import (Display) tab."""
        if self.playing_display:
            self.timer_display.stop()
            self.playing_display = False
            self.play_button_display.setText("Play")
        else:
            interval = 16 if sys.platform.startswith('win') else 100
            self.timer_display.start(interval)
            self.playing_display = True
            self.play_button_display.setText("Pause")
        self.playing = self.playing_display  # Legacy compatibility
    
    def play_pause_segmentation(self):
        """Toggle play/pause for Segmentation tab (handles all sub-tabs).
        
        Does NOT play if maximum temporal projection is enabled, since all
        time points are already shown in the projection.
        """
        # Check if max projection is enabled - if so, don't allow playback
        if getattr(self, 'use_max_proj_for_segmentation', False):
            QMessageBox.information(
                self, 
                "Playback Disabled",
                "Time playback is disabled when Maximum Temporal Projection is active.\n"
                "Disable Maximum Projection to enable frame-by-frame playback."
            )
            return
        
        if self.playing_segmentation:
            self.timer_segmentation.stop()
            self.playing_segmentation = False
            if hasattr(self, 'play_button_segmentation'):
                self.play_button_segmentation.setText("Play")
        else:
            interval = 16 if sys.platform.startswith('win') else 100
            self.timer_segmentation.start(interval)
            self.playing_segmentation = True
            if hasattr(self, 'play_button_segmentation'):
                self.play_button_segmentation.setText("Pause")
    
    def play_pause_tracking(self):
        """Toggle play/pause for Tracking tab."""
        if self.playing_tracking:
            self.timer_tracking.stop()
            self.playing_tracking = False
            self.play_button_tracking.setText("Play")
        else:
            interval = 16 if sys.platform.startswith('win') else 100
            self.timer_tracking.start(interval)
            self.playing_tracking = True
            self.play_button_tracking.setText("Pause")
    
    def play_pause_tracking_vis(self):
        """Toggle play/pause for Tracking Visualization tab."""
        if self.playing_tracking_vis:
            self.timer_tracking_vis.stop()
            self.playing_tracking_vis = False
            self.play_button_tracking_vis.setText("Play")
        else:
            interval = 16 if sys.platform.startswith('win') else 100
            self.timer_tracking_vis.start(interval)
            self.playing_tracking_vis = True
            self.play_button_tracking_vis.setText("Pause")

    def update_channel(self, channel):
        self.current_channel = channel
        self._sync_tracking_channel()
        self.merged_mode = False
        if hasattr(self, 'channelControlsTabs'):
            self.channelControlsTabs.blockSignals(True)   
            self.channelControlsTabs.setCurrentIndex(channel) 
            self.channelControlsTabs.blockSignals(False)
        
        # Clear detection preview BEFORE plotting to avoid showing old channel's spots
        self.detected_spots_frame = None
        
        self.plot_image()
        self.plot_tracking()
        self.update_threshold_histogram()
        self.populate_colocalization_channels()
        
        # Reset or restore threshold for the new channel
        if hasattr(self, 'tracking_thresholds') and channel in self.tracking_thresholds:
            # Restore previously-used threshold for this channel
            self.user_selected_threshold = self.tracking_thresholds[channel]
            if hasattr(self, 'threshold_slider'):
                self.threshold_slider.blockSignals(True)
                self.threshold_slider.setValue(int(self.user_selected_threshold))
                self.threshold_slider.blockSignals(False)
            if hasattr(self, 'threshold_value_label'):
                self.threshold_value_label.setText(f"Value: {int(self.user_selected_threshold)}")
        else:
            # Reset threshold for new channel (auto-detect based on histogram)
            self._reset_threshold_for_new_channel()

    # Note: update_channel_crops removed - Crops tab has been deprecated

    def update_frame(self, value):
        self.current_frame = value
        total_frames = getattr(self, 'total_frames', 1)
        frame_text = f"{value}/{total_frames - 1}"
        
        # Update labels for tabs that share this handler (Import, Tracking, Visualization)
        if hasattr(self, 'frame_label_display'):
            self.frame_label_display.setText(frame_text)
        if hasattr(self, 'frame_label_tracking'):
            self.frame_label_tracking.setText(frame_text)
        if hasattr(self, 'frame_label_tracking_vis'):
            self.frame_label_tracking_vis.setText(frame_text)
        
        # Sync slider values across shared tabs
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
        
        # OPTIMIZATION: Only update the currently visible tab instead of all tabs
        current_tab_index = self.tabs.currentIndex()
        
        # Tab index mapping (Cellpose is now a sub-tab of Segmentation):
        # 0=Import, 1=Registration, 2=Segmentation (includes Cellpose), 3=Photobleaching,
        # 4=Tracking, 5=MSD, 6=Distribution, 7=Time Course, 8=Correlation,
        # 9=Colocalization, 10=Colocalization Manual, 11=Tracking Visualization, 12=Export
        
        if current_tab_index == 0:  # Import tab
            self.plot_image()
        elif current_tab_index == 4:  # Tracking tab (was 5)
            self.plot_tracking()
            self.update_threshold_histogram()
        elif current_tab_index == 11:  # Tracking Visualization tab (was 12)
            if hasattr(self, 'ax_tracking_vis'):
                self.display_tracking_visualization()

    def _on_display_zoom_select(self, eclick, erelease):
        """Handle ROI selection from left-click drag on display/import canvas."""
        if self.image_stack is None:
            return
        
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Handle None values (click outside axes)
        if x1 is None or x2 is None or y1 is None or y2 is None:
            return
        
        # Calculate ROI bounds
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Enforce minimum ROI size (50x50 pixels)
        if (x_max - x_min) < 50 or (y_max - y_min) < 50:
            return
        
        # Clamp to image bounds
        _, _, H, W, _ = self.image_stack.shape
        x_min = max(0, x_min)
        x_max = min(W, x_max)
        y_min = max(0, y_min)
        y_max = min(H, y_max)
        
        # Store ROI
        self.display_zoom_roi = (x_min, x_max, y_min, y_max)
        
        # Update label
        if hasattr(self, 'display_zoom_label'):
            self.display_zoom_label.setText(f"🔍 ROI: X[{int(x_min)}:{int(x_max)}] Y[{int(y_min)}:{int(y_max)}]")
            self.display_zoom_label.setStyleSheet("color: #00d4aa; font-size: 10px; font-weight: bold;")
        
        # Redraw with zoom
        self.plot_image()

    def _on_display_canvas_click(self, event):
        """Handle mouse clicks on display canvas - double-click to reset zoom."""
        if event.dblclick:
            self._reset_display_zoom()

    def _reset_display_zoom(self):
        """Reset display/import tab zoom to show full image."""
        self.display_zoom_roi = None
        
        # Update label
        if hasattr(self, 'display_zoom_label'):
            self.display_zoom_label.setText("🔍 Full View")
            self.display_zoom_label.setStyleSheet("color: #888888; font-size: 10px;")
        
        # Redraw without zoom
        self.plot_image()

    def plot_image(self):
        # Check if ax_display is still valid (in the figure's axes list)
        ax_valid = (hasattr(self, 'ax_display') and 
                   self.ax_display is not None and 
                   self.ax_display in self.figure_display.axes)
        
        if ax_valid:
            # Clear axes content instead of entire figure to preserve RectangleSelector
            self.ax_display.clear()
        else:
            # Need to create new axes
            self.figure_display.clear()
            self.ax_display = self.figure_display.add_subplot(111)
            # Initialize RectangleSelector only when creating new axes
            self.display_zoom_selector = RectangleSelector(
                self.ax_display,
                self._on_display_zoom_select,
                useblit=True,
                button=[1],  # Left mouse button only
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=False,
                props=dict(facecolor='cyan', edgecolor='white', alpha=0.3, linewidth=2)
            )
        
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
                        # Apply brightness scaling (60% default, matches compute_merged_image)
                        brightness = 0.6
                        combined_image = combined_image * brightness
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
                time_str = self._format_time_interval(current_time)
                self.ax_display.text(0.05, 0.95, time_str, transform=self.ax_display.transAxes,
                                    verticalalignment='top', color='white', fontsize=12,
                                    bbox=dict(facecolor='black', alpha=0.5, pad=2))
            if hasattr(self, 'voxel_yx_nm') and self.voxel_yx_nm is not None:
                microns_per_pixel = self.voxel_yx_nm / 1000.0
                scalebar = ScaleBar(microns_per_pixel, units='um', length_fraction=0.2,
                                    location='lower right', box_color='black', color='white', font_properties={'size': 10})
                self.ax_display.add_artist(scalebar)
            
            # Add thin border to show image boundaries
            H, W = self.image_stack.shape[2], self.image_stack.shape[3]
            img_border = patches.Rectangle((0, 0), W-1, H-1, linewidth=0.8, 
                                            edgecolor='#555555', facecolor='none', linestyle='-')
            self.ax_display.add_patch(img_border)
            
            # Use tight_layout for proper spacing
            try:
                self.figure_display.tight_layout()
            except Exception:
                pass  # Ignore layout errors
            
            # Apply zoom AFTER tight_layout to ensure limits are not reset
            if self.display_zoom_roi is not None:
                x_min, x_max, y_min, y_max = self.display_zoom_roi
                self.ax_display.set_xlim(x_min, x_max)
                self.ax_display.set_ylim(y_max, y_min)  # Inverted for image coordinates
                
        self.canvas_display.draw_idle()

    def update_z(self, value):
        """Handle Z-slider value change: update displayed image to selected z-plane or max projection."""
        # Update Z status label
        if hasattr(self, 'z_slider_display') and hasattr(self, 'z_label_display'):
            max_val = self.z_slider_display.maximum()
            if value == max_val:
                self.z_label_display.setText("Max")
                self.z_label_display.setStyleSheet("color: cyan; font-weight: bold;")
            else:
                self.z_label_display.setText(f"Z={value}")
                self.z_label_display.setStyleSheet("color: lime; font-weight: bold;")
        # No need to sync other sliders; just refresh the display
        self.current_frame = 0  # Reset to first frame for new Z selection
        self.plot_image()

    def update_z_tracking(self, value):
        """Handle Z-slider value change for Tracking tab.
        
        When slider is at max (Z), show max projection with all spots.
        When slider is at specific value (0 to Z-1), show that Z-plane
        and only spots from that plane (for 3D tracking).
        """
        # Update Z status label
        if hasattr(self, 'z_slider_tracking') and hasattr(self, 'z_label_tracking'):
            max_val = self.z_slider_tracking.maximum()
            if value == max_val:
                self.z_label_tracking.setText("Max")
                self.z_label_tracking.setStyleSheet("color: cyan; font-weight: bold;")
            else:
                self.z_label_tracking.setText(f"Z={value}")
                self.z_label_tracking.setStyleSheet("color: lime; font-weight: bold;")
        self.plot_tracking()

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
            
            # Use unified reset for all tabs and state (includes registration tab)
            self.reset_all_state()

            # Clear info labels (close-specific: show empty state)
            labels_to_clear = [
                'file_label', 'frames_label', 'z_scales_label',
                'y_pixels_label', 'x_pixels_label', 'channels_label',
                'voxel_yx_size_label', 'voxel_z_nm_label',
                'bit_depth_label', 'time_interval_label',
                'laser_lines_label', 'intensities_label', 'wave_ranges_label'
            ]
            for lbl_name in labels_to_clear:
                if hasattr(self, lbl_name):
                    getattr(self, lbl_name).setText("")
            
            # Clear channel controls (close-specific: remove channel UI)
            if hasattr(self, 'channelControlsTabs'):
                self.channelControlsTabs.clear()
            
            # Clear channel buttons (close-specific: remove buttons)
            for btn_list in [getattr(self, 'channel_buttons_display', []),
                            getattr(self, 'channel_buttons_tracking', []),
                            getattr(self, 'channel_buttons_tracking_vis', []),
                            getattr(self, 'segmentation_channel_buttons', [])]:
                for btn in btn_list:
                    if btn:
                        btn.setParent(None)
            
            # Reset button lists
            self.channel_buttons_display = []
            self.channel_buttons_tracking = []
            if hasattr(self, 'channel_buttons_tracking_vis'):
                self.channel_buttons_tracking_vis = []
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
            self.stop_all_playback()
            
            # Reset current indices
            self.current_frame = 0
            self.current_channel = 0
            
            # Reset merged mode
            self.merged_mode = False

    def close_all_files(self):
        """
        Remove all loaded files (LIF and TIFF) from the tree and free their memory.
        Clears all data and resets the GUI to its initial empty state.
        """
        # Get all top-level items count
        num_files = self.image_tree.topLevelItemCount()
        if num_files == 0:
            return
        
        # Clear all loaded file data
        self.loaded_lif_files.clear()
        
        # Remove all items from tree view
        self.image_tree.clear()
        
        # Clear core data
        self.image_stack = None
        self.data_folder_path = None
        self.colocalization_results = None
        self.current_total_plots = None
        
        # Use unified reset for all tabs and state
        self.reset_all_state()
        
        # Clear info labels
        labels_to_clear = [
            'file_label', 'frames_label', 'z_scales_label',
            'y_pixels_label', 'x_pixels_label', 'channels_label',
            'voxel_yx_size_label', 'voxel_z_nm_label',
            'bit_depth_label', 'time_interval_label',
            'laser_lines_label', 'intensities_label', 'wave_ranges_label'
        ]
        for lbl_name in labels_to_clear:
            if hasattr(self, lbl_name):
                getattr(self, lbl_name).setText("")
        
        # Clear channel controls
        if hasattr(self, 'channelControlsTabs'):
            self.channelControlsTabs.clear()
        
        # Clear channel buttons
        for btn_list in [getattr(self, 'channel_buttons_display', []),
                        getattr(self, 'channel_buttons_tracking', []),
                        getattr(self, 'channel_buttons_tracking_vis', []),
                        getattr(self, 'segmentation_channel_buttons', [])]:
            for btn in btn_list:
                if btn:
                    btn.setParent(None)
        
        # Reset button lists
        self.channel_buttons_display = []
        self.channel_buttons_tracking = []
        if hasattr(self, 'channel_buttons_tracking_vis'):
            self.channel_buttons_tracking_vis = []
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
        
        # Disable controls
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
        self.stop_all_playback()
        
        # Reset current indices
        self.current_frame = 0
        self.current_channel = 0
        
        # Reset merged mode
        self.merged_mode = False

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

    def compute_merged_image(self, use_brightness_slider=False):
        if self.image_stack is None:
            return None
        # Get current frame's multi-channel image
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
        
        # Apply brightness scaling (prevents oversaturation with multiple channels)
        # Import tab uses fixed 60%; Tracking Visualization tab uses slider
        brightness = 0.6  # Default 60% brightness for Import tab
        if use_brightness_slider and hasattr(self, 'merge_brightness_slider'):
            brightness = self.merge_brightness_slider.value() / 100.0
            if hasattr(self, 'merge_brightness_label'):
                self.merge_brightness_label.setText(f"{int(brightness * 100)}%")
        
        if brightness < 1.0:
            combined_image = combined_image * brightness
        
        merged_img = np.clip(combined_image, 0, 1)
        return merged_img

    def merge_color_channels(self):
        if self.image_stack is None:
            # Silently return if no image - this can happen when closing files in merge mode
            return
        merged_img = self.compute_merged_image()
        if merged_img is None:
            QMessageBox.information(self, "Merge Error", "Not enough channels to merge or unsupported image format.")
            return
        self.merged_mode = True
        self.figure_display.clear()
        self.ax_display = self.figure_display.add_subplot(111)
        
        # Recreate RectangleSelector on new axes (fixes zoom after tab switch)
        self.display_zoom_selector = RectangleSelector(
            self.ax_display,
            self._on_display_zoom_select,
            useblit=True,
            button=[1],  # Left mouse button only
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False,
            props=dict(facecolor='cyan', edgecolor='white', alpha=0.3, linewidth=2)
        )
        
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
        
        # Set up zoom feature: RectangleSelector for left-click drag
        self.display_zoom_selector = RectangleSelector(
            self.ax_display,
            self._on_display_zoom_select,
            useblit=True,
            button=[1],  # Left mouse button only
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False,
            props=dict(facecolor='cyan', edgecolor='white', alpha=0.3, linewidth=2)
        )
        
        # Connect double-click to reset zoom
        self.canvas_display.mpl_connect('button_press_event', self._on_display_canvas_click)
        
        # Create a horizontal layout to hold the canvas and the Z slider
        canvas_slider_layout = QHBoxLayout()
        canvas_slider_layout.addWidget(self.canvas_display)
        
        # Z-slider with label (vertical, on the right of canvas) - minimal width
        z_slider_container = QWidget()
        z_slider_container.setFixedWidth(40)
        z_slider_layout_display = QVBoxLayout(z_slider_container)
        z_slider_layout_display.setContentsMargins(2, 0, 2, 0)
        z_slider_layout_display.setSpacing(2)
        
        z_label_top_display = QLabel("Z")
        z_label_top_display.setAlignment(Qt.AlignCenter)
        z_label_top_display.setStyleSheet("color: white; font-weight: bold; font-size: 10px;")
        z_slider_layout_display.addWidget(z_label_top_display)
        
        # Initialize the vertical Z-plane slider
        self.z_slider_display = QSlider(Qt.Vertical, self)
        self.z_slider_display.setMinimum(0)
        self.z_slider_display.setTickPosition(QSlider.NoTicks)
        self.z_slider_display.setInvertedAppearance(True)  # Top = highest Z index (max projection)
        self.z_slider_display.valueChanged.connect(self.update_z)   # live update on value change
        z_slider_layout_display.addWidget(self.z_slider_display, stretch=1)
        
        self.z_label_display = QLabel("Max")
        self.z_label_display.setAlignment(Qt.AlignCenter)
        self.z_label_display.setStyleSheet("color: cyan; font-weight: bold; font-size: 9px;")
        z_slider_layout_display.addWidget(self.z_label_display)
        
        canvas_slider_layout.addWidget(z_slider_container)
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
        
        self.frame_label_display = QLabel("0/0")
        self.frame_label_display.setMinimumWidth(50)
        controls_layout.addWidget(self.frame_label_display)
        
        self.play_button_display = QPushButton("Play", self)
        self.play_button_display.clicked.connect(self.play_pause_display)
        controls_layout.addWidget(self.play_button_display)
        display_left_layout.addLayout(controls_layout)
        
        # Zoom info layout
        zoom_info_layout = QHBoxLayout()
        self.display_zoom_label = QLabel("🔍 Full View")
        self.display_zoom_label.setStyleSheet("color: #888888; font-size: 10px;")
        zoom_info_layout.addWidget(self.display_zoom_label)
        zoom_info_layout.addStretch()
        display_left_layout.addLayout(zoom_info_layout)
        
        # Right side
        display_right_layout = QVBoxLayout()
        display_main_layout.addLayout(display_right_layout, 1)
        # Image selection tree
        display_right_layout.addWidget(QLabel("Select Image"))
        self.image_tree = QTreeWidget()
        self.image_tree.setMinimumWidth(200)
        self.image_tree.setMinimumHeight(120)
        self.image_tree.setMaximumHeight(220)
        self.image_tree.setHeaderHidden(True)
        self.image_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.image_tree.itemClicked.connect(self.on_tree_item_clicked)
        self.image_tree.currentItemChanged.connect(self.on_tree_current_item_changed)
        display_right_layout.addWidget(self.image_tree)
        # Close file buttons (using horizontal layout for both buttons)
        close_buttons_layout = QHBoxLayout()
        self.close_file_button = QPushButton("Close File", self)
        self.close_file_button.clicked.connect(self.close_selected_file)
        close_buttons_layout.addWidget(self.close_file_button)
        self.close_all_files_button = QPushButton("Close All", self)
        self.close_all_files_button.clicked.connect(self.close_all_files)
        close_buttons_layout.addWidget(self.close_all_files_button)
        display_right_layout.addLayout(close_buttons_layout)
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
        scroll_info.setMinimumHeight(280)
        scroll_info.setMaximumHeight(350)
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
            # Apply Z selection
            current_z = getattr(self, 'segmentation_current_z', -1)
            if current_z == -1:
                # Max Z-projection
                max_proj = np.max(image_channel, axis=0)
                self.segmentation_z_used_for_mask = -1
            else:
                # Specific Z-slice
                z_idx = min(current_z, image_channel.shape[0] - 1)
                max_proj = image_channel[z_idx, :, :]
                self.segmentation_z_used_for_mask = z_idx
            max_proj = gaussian_filter(max_proj, sigma=1)
            max_proj = np.clip(max_proj, np.percentile(max_proj, 0.01), np.percentile(max_proj, 99.95))
            # Create labeled mask with cell ID = 1 (not 255 which was incorrect)
            # Use int32 dtype for proper labeled mask compatibility with tracking
            mask = np.zeros(max_proj.shape[:2], dtype=np.int32)
            polygon = np.array([self.selected_points], dtype=np.int32)
            cv2.fillPoly(mask, polygon, 1)  # Fill with cell ID 1, not 255
            self.segmentation_mask = mask
            self._active_mask_source = 'segmentation'
            # Clear Cellpose/imported masks since we're using manual segmentation now
            self.cellpose_masks_cyto = None
            self.cellpose_masks_nuc = None
            self.cellpose_masks_cyto_tyx = None
            self.cellpose_masks_nuc_tyx = None
            self.use_tyx_masks = False
            self.masks_imported = False
            # Reset import status labels
            if hasattr(self, 'label_cyto_mask_status'):
                self.label_cyto_mask_status.setText("No cytosol mask loaded")
                self.label_cyto_mask_status.setStyleSheet("color: gray;")
            if hasattr(self, 'label_nuc_mask_status'):
                self.label_nuc_mask_status.setText("No nucleus mask loaded")
                self.label_nuc_mask_status.setStyleSheet("color: gray;")
            self.ax_segmentation.clear()
            cmap_imagej = cmap_list_imagej[ch % len(cmap_list_imagej)]
            self.ax_segmentation.imshow(max_proj, cmap=cmap_imagej)
            self.ax_segmentation.contour(self.segmentation_mask, levels=[0.5], colors='white', linewidths=1)
            self.ax_segmentation.axis('off')
            self.canvas_segmentation.draw()
            self.photobleaching_calculated = False
            self.segmentation_mode = "manual"
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
        # NOTE: cellpose_current_frame syncing removed - using segmentation_current_frame
        
        # Update frame labels
        total_frames = self.total_frames
        frame_text = f"{self.current_frame}/{total_frames - 1}"
        if hasattr(self, 'frame_label_display'):
            self.frame_label_display.setText(frame_text)
        if hasattr(self, 'frame_label_tracking'):
            self.frame_label_tracking.setText(frame_text)
        if hasattr(self, 'frame_label_tracking_vis'):
            self.frame_label_tracking_vis.setText(frame_text)
        # NOTE: Cellpose now shares the segmentation display, no separate frame label
        
        for slider in (self.time_slider_display, self.time_slider_tracking, 
                       getattr(self, 'time_slider_tracking_vis', None)):
            if slider is not None:
                slider.blockSignals(True)
                slider.setValue(self.current_frame)
                slider.blockSignals(False)
        self.plot_image()
        current_tab = self.tabs.currentIndex()
        if current_tab == self.tabs.indexOf(self.tracking_tab):
            # Sync TYX masks if active before plotting tracking
            if getattr(self, 'use_tyx_masks', False):
                if getattr(self, 'cellpose_masks_cyto_tyx', None) is not None:
                    self.cellpose_masks_cyto = self.cellpose_masks_cyto_tyx[self.current_frame]
                if getattr(self, 'cellpose_masks_nuc_tyx', None) is not None:
                    self.cellpose_masks_nuc = self.cellpose_masks_nuc_tyx[self.current_frame]
            self.plot_tracking()
        elif (current_tab == self.tabs.indexOf(self.tracking_visualization_tab)
            and getattr(self, 'has_tracked', False)
            and not self.df_tracking.empty):
            self.display_tracking_visualization()
        # NOTE: Cellpose is now a sub-tab within Segmentation, handled by its own time slider
    
    def next_frame_display(self):
        """Advance to next frame for Import (Display) tab only."""
        if getattr(self, 'total_frames', 0) == 0:
            return
        self.current_frame = (self.current_frame + 1) % self.total_frames
        # Update frame label
        frame_text = f"{self.current_frame}/{self.total_frames - 1}"
        if hasattr(self, 'frame_label_display'):
            self.frame_label_display.setText(frame_text)
        if hasattr(self, 'time_slider_display'):
            self.time_slider_display.blockSignals(True)
            self.time_slider_display.setValue(self.current_frame)
            self.time_slider_display.blockSignals(False)
        self.plot_image()
    
    def next_frame_segmentation(self):
        """Advance to next frame for Segmentation tab (handles all sub-tabs).
        
        For Cellpose: visualizes TYX masks frame by frame if available.
        For Manual/Watershed: advances through frames for reference viewing.
        """
        if getattr(self, 'total_frames', 0) == 0:
            return
        
        # Advance frame
        self.segmentation_current_frame = (self.segmentation_current_frame + 1) % self.total_frames
        
        # Update frame slider
        if hasattr(self, 'segmentation_time_slider'):
            self.segmentation_time_slider.blockSignals(True)
            self.segmentation_time_slider.setValue(self.segmentation_current_frame)
            self.segmentation_time_slider.blockSignals(False)
        
        # Update frame label
        if hasattr(self, 'frame_label_segmentation'):
            self.frame_label_segmentation.setText(f"{self.segmentation_current_frame}/{self.total_frames - 1}")
        
        # Sync TYX masks if active before plotting
        if getattr(self, 'use_tyx_masks', False):
            if getattr(self, 'cellpose_masks_cyto_tyx', None) is not None:
                self.cellpose_masks_cyto = self.cellpose_masks_cyto_tyx[self.segmentation_current_frame]
            if getattr(self, 'cellpose_masks_nuc_tyx', None) is not None:
                self.cellpose_masks_nuc = self.cellpose_masks_nuc_tyx[self.segmentation_current_frame]
        
        # Plot based on active sub-tab
        if hasattr(self, 'segmentation_method_tabs'):
            current_subtab = self.segmentation_method_tabs.currentIndex()
            if current_subtab == 1 or current_subtab == 3:  # Cellpose or Import sub-tab
                self.plot_cellpose_results()
            else:
                # Manual or Watershed - just update the image display
                self.plot_segmentation()
    
    def next_frame_tracking(self):
        """Advance to next frame for Tracking tab only."""
        if getattr(self, 'total_frames', 0) == 0:
            return
        self.current_frame = (self.current_frame + 1) % self.total_frames
        # Update frame label
        frame_text = f"{self.current_frame}/{self.total_frames - 1}"
        if hasattr(self, 'frame_label_tracking'):
            self.frame_label_tracking.setText(frame_text)
        if hasattr(self, 'time_slider_tracking'):
            self.time_slider_tracking.blockSignals(True)
            self.time_slider_tracking.setValue(self.current_frame)
            self.time_slider_tracking.blockSignals(False)
        # Sync TYX masks if active before plotting tracking
        if getattr(self, 'use_tyx_masks', False):
            if getattr(self, 'cellpose_masks_cyto_tyx', None) is not None:
                self.cellpose_masks_cyto = self.cellpose_masks_cyto_tyx[self.current_frame]
            if getattr(self, 'cellpose_masks_nuc_tyx', None) is not None:
                self.cellpose_masks_nuc = self.cellpose_masks_nuc_tyx[self.current_frame]
        self.plot_tracking()
    
    def next_frame_tracking_vis(self):
        """Advance to next frame for Tracking Visualization tab only."""
        if getattr(self, 'total_frames', 0) == 0:
            return
        self.current_frame = (self.current_frame + 1) % self.total_frames
        # Update frame label
        frame_text = f"{self.current_frame}/{self.total_frames - 1}"
        if hasattr(self, 'frame_label_tracking_vis'):
            self.frame_label_tracking_vis.setText(frame_text)
        if hasattr(self, 'time_slider_tracking_vis'):
            self.time_slider_tracking_vis.blockSignals(True)
            self.time_slider_tracking_vis.setValue(self.current_frame)
            self.time_slider_tracking_vis.blockSignals(False)
        if getattr(self, 'has_tracked', False) and not self.df_tracking.empty:
            self.display_tracking_visualization()


    # NOTE: setup_cellpose_tab() has been removed.
    # Cellpose UI is now set up within setup_segmentation_tab() as a sub-tab.

    def _on_num_masks_slider_changed(self, value):
        """Update the value label and adjust min_frames_slider max range."""
        self.num_masks_value_label.setText(str(value))
        # Ensure min_frames slider doesn't exceed num_masks slider
        if hasattr(self, 'min_frames_slider'):
            self.min_frames_slider.setMaximum(value)
            if self.min_frames_slider.value() > value:
                self.min_frames_slider.setValue(value)
    
    def _on_min_frames_slider_changed(self, value):
        """Update the value label for min_frames slider."""
        self.min_frames_value_label.setText(str(value))
    
    def _on_cyto_size_slider_changed(self, value):
        """
        Apply cytosol mask size adjustment when slider value changes.
        Positive = expand, Negative = shrink, 0 = original.
        Works independently from nucleus mask.
        """
        # Update label
        if hasattr(self, 'cyto_size_label'):
            self.cyto_size_label.setText(str(value))
        
        # Check if we have cytosol mask
        if self.cellpose_masks_cyto is None:
            return
        
        # Store original if not yet stored
        if self._original_cellpose_masks_cyto is None:
            self._original_cellpose_masks_cyto = self.cellpose_masks_cyto.copy()
        
        # For TYX masks
        if getattr(self, 'use_tyx_masks', False):
            if self._original_cellpose_masks_cyto_tyx is None and self.cellpose_masks_cyto_tyx is not None:
                self._original_cellpose_masks_cyto_tyx = self.cellpose_masks_cyto_tyx.copy()
        
        # Apply transformation
        if value == 0:
            # Restore original
            if self._original_cellpose_masks_cyto is not None:
                self.cellpose_masks_cyto = self._original_cellpose_masks_cyto.copy()
            if getattr(self, 'use_tyx_masks', False) and self._original_cellpose_masks_cyto_tyx is not None:
                self.cellpose_masks_cyto_tyx = self._original_cellpose_masks_cyto_tyx.copy()
            msg = "Cytosol mask restored to original size"
        elif value > 0:
            # Expand
            self.cellpose_masks_cyto = self._expand_labeled_mask(
                self._original_cellpose_masks_cyto, value
            )
            if getattr(self, 'use_tyx_masks', False) and self._original_cellpose_masks_cyto_tyx is not None:
                self.cellpose_masks_cyto_tyx = self.cellpose_masks_cyto_tyx.copy()
                for t in range(self._original_cellpose_masks_cyto_tyx.shape[0]):
                    self.cellpose_masks_cyto_tyx[t] = self._expand_labeled_mask(
                        self._original_cellpose_masks_cyto_tyx[t], value
                    )
            msg = f"Cytosol mask expanded by {value}px"
        else:
            # Shrink (value is negative, use abs)
            self.cellpose_masks_cyto = self._shrink_labeled_mask(
                self._original_cellpose_masks_cyto, abs(value)
            )
            if getattr(self, 'use_tyx_masks', False) and self._original_cellpose_masks_cyto_tyx is not None:
                self.cellpose_masks_cyto_tyx = self.cellpose_masks_cyto_tyx.copy()
                for t in range(self._original_cellpose_masks_cyto_tyx.shape[0]):
                    self.cellpose_masks_cyto_tyx[t] = self._shrink_labeled_mask(
                        self._original_cellpose_masks_cyto_tyx[t], abs(value)
                    )
            msg = f"Cytosol mask shrunk by {abs(value)}px"
        
        # Update display
        self.plot_cellpose_results()
        n_cyto = int(self.cellpose_masks_cyto.max()) if self.cellpose_masks_cyto is not None else 0
        self.statusBar().showMessage(f"{msg}. Cells: {n_cyto}")
    
    def _on_nuc_size_slider_changed(self, value):
        """
        Apply nucleus mask size adjustment when slider value changes.
        Positive = expand, Negative = shrink, 0 = original.
        Works independently from cytosol mask.
        """
        # Update label
        if hasattr(self, 'nuc_size_label'):
            self.nuc_size_label.setText(str(value))
        
        # Check if we have nucleus mask
        if self.cellpose_masks_nuc is None:
            return
        
        # Store original if not yet stored
        if self._original_cellpose_masks_nuc is None:
            self._original_cellpose_masks_nuc = self.cellpose_masks_nuc.copy()
        
        # For TYX masks
        if getattr(self, 'use_tyx_masks', False):
            if self._original_cellpose_masks_nuc_tyx is None and self.cellpose_masks_nuc_tyx is not None:
                self._original_cellpose_masks_nuc_tyx = self.cellpose_masks_nuc_tyx.copy()
        
        # Apply transformation
        if value == 0:
            # Restore original
            if self._original_cellpose_masks_nuc is not None:
                self.cellpose_masks_nuc = self._original_cellpose_masks_nuc.copy()
            if getattr(self, 'use_tyx_masks', False) and self._original_cellpose_masks_nuc_tyx is not None:
                self.cellpose_masks_nuc_tyx = self._original_cellpose_masks_nuc_tyx.copy()
            msg = "Nucleus mask restored to original size"
        elif value > 0:
            # Expand
            self.cellpose_masks_nuc = self._expand_labeled_mask(
                self._original_cellpose_masks_nuc, value
            )
            if getattr(self, 'use_tyx_masks', False) and self._original_cellpose_masks_nuc_tyx is not None:
                self.cellpose_masks_nuc_tyx = self.cellpose_masks_nuc_tyx.copy()
                for t in range(self._original_cellpose_masks_nuc_tyx.shape[0]):
                    self.cellpose_masks_nuc_tyx[t] = self._expand_labeled_mask(
                        self._original_cellpose_masks_nuc_tyx[t], value
                    )
            msg = f"Nucleus mask expanded by {value}px"
        else:
            # Shrink (value is negative, use abs)
            self.cellpose_masks_nuc = self._shrink_labeled_mask(
                self._original_cellpose_masks_nuc, abs(value)
            )
            if getattr(self, 'use_tyx_masks', False) and self._original_cellpose_masks_nuc_tyx is not None:
                self.cellpose_masks_nuc_tyx = self.cellpose_masks_nuc_tyx.copy()
                for t in range(self._original_cellpose_masks_nuc_tyx.shape[0]):
                    self.cellpose_masks_nuc_tyx[t] = self._shrink_labeled_mask(
                        self._original_cellpose_masks_nuc_tyx[t], abs(value)
                    )
            msg = f"Nucleus mask shrunk by {abs(value)}px"
        
        # Update display
        self.plot_cellpose_results()
        n_nuc = int(self.cellpose_masks_nuc.max()) if self.cellpose_masks_nuc is not None else 0
        self.statusBar().showMessage(f"{msg}. Cells: {n_nuc}")
    
    def _expand_labeled_mask(self, mask, expansion_pixels):
        """Expand each label in a mask without overlaps (Voronoi-like)."""
        if mask is None:
            return None
        
        expanded = np.zeros_like(mask)
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels != 0]
        
        if len(unique_labels) == 0:
            return mask.copy()
        
        # Compute distance from each cell for all pixels
        all_distances = np.full((len(unique_labels),) + mask.shape, np.inf)
        
        for i, label_id in enumerate(unique_labels):
            cell_mask = (mask == label_id)
            all_distances[i] = distance_transform_edt(~cell_mask)
        
        min_distance_idx = np.argmin(all_distances, axis=0)
        min_distances = np.min(all_distances, axis=0)
        
        for i, label_id in enumerate(unique_labels):
            # Original cell pixels keep their label
            expanded[mask == label_id] = label_id
            # Expanded region: closest to this cell and within expansion distance
            expansion_mask = (
                (min_distance_idx == i) & 
                (min_distances <= expansion_pixels) & 
                (min_distances > 0)
            )
            expanded[expansion_mask] = label_id
        
        return expanded
    
    def _shrink_labeled_mask(self, mask, shrink_pixels):
        """Shrink each label in a mask by eroding from boundaries."""
        if mask is None:
            return None
        
        shrunk = np.zeros_like(mask)
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels != 0]
        
        if len(unique_labels) == 0:
            return mask.copy()
        
        for label_id in unique_labels:
            cell_mask = (mask == label_id)
            # Distance from boundary (positive inside cell)
            dist_inside = distance_transform_edt(cell_mask)
            # Keep only pixels that are more than shrink_pixels from edge
            shrunk[dist_inside > shrink_pixels] = label_id
        
        return shrunk

    

    def _update_cellpose_sliders_for_image(self, total_frames):
        """Update Cellpose TYX sliders when a new image is loaded."""
        if hasattr(self, 'num_masks_slider'):
            # Reset to 1 and update max range
            self.num_masks_slider.blockSignals(True)
            self.num_masks_slider.setMaximum(max(1, total_frames))
            self.num_masks_slider.setValue(1)
            self.num_masks_slider.blockSignals(False)
            self.num_masks_value_label.setText("1")
        
        if hasattr(self, 'min_frames_slider'):
            # Reset to 1 and update max range
            self.min_frames_slider.blockSignals(True)
            self.min_frames_slider.setMaximum(max(1, total_frames))
            self.min_frames_slider.setValue(1)
            self.min_frames_slider.blockSignals(False)
            self.min_frames_value_label.setText("1")

    # NOTE: create_cellpose_channel_buttons, update_cellpose_frame, and update_cellpose_channel
    # have been removed. Cellpose now shares the segmentation display and uses the shared
    # segmentation time slider and channel buttons. Frame/channel changes are handled by
    # update_segmentation_frame and update_segmentation_channel respectively.

    def run_cellpose_cyto(self):
        if self.image_stack is None:
            return
        
        try:
            # Get parameters - channel is determined by left panel selection in Segmentation tab
            channel = self.segmentation_current_channel
            diameter = int(self.cellpose_cyto_diameter_input.value())
            model_name = self.cellpose_cyto_model_input.currentText()
            
            # Check if TYX masks are requested (slider > 1 means TYX mode)
            num_masks = self.num_masks_slider.value()
            if (num_masks > 1 and 
                self.image_stack.ndim == 5 and 
                self.image_stack.shape[0] > 1):
                
                max_tp = min(num_masks, self.image_stack.shape[0])
                linking_memory = 1  # Hardcoded to 1
                
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
                    except Exception:
                        progress.setLabelText(msg)
                    QApplication.processEvents()
                
                image_to_use = self.get_current_image_source()
                
                # Check if optimization is requested
                selection_metric = 'max_cells_and_area' if self.chk_optimize_cyto.isChecked() else None
                
                tyx_generator = mi.CellposeTimeSeries(
                    image=image_to_use,
                    channels_cytosol=channel,
                    channels_nucleus=None,
                    diameter_cytosol=diameter,
                    diameter_nucleus=60,
                    max_timepoints=max_tp,
                    linking_memory=linking_memory,
                    model_type_cyto=model_name,
                    progress_callback=progress_callback,
                    selection_metric_cyto=selection_metric
                )
                
                masks_cyto_tyx, _ = tyx_generator.calculate_tyx_masks()
                progress.close()
                
                if masks_cyto_tyx is not None:
                    # Filter short-lived masks (artifacts) and reindex IDs
                    min_frames = self.min_frames_slider.value()
                    masks_cyto_tyx = mi.CellposeTimeSeries.filter_short_lived_masks(masks_cyto_tyx, min_frames)
                    
                    self.cellpose_masks_cyto_tyx = masks_cyto_tyx
                    # Also set the current frame's YX mask for compatibility
                    self.cellpose_masks_cyto = masks_cyto_tyx[self.segmentation_current_frame]
                    self.use_tyx_masks = True
                else:
                    self.use_tyx_masks = False
                    
                self.statusBar().showMessage(f"TYX cytosol masks calculated: {max_tp} timepoints")
            else:
                # Standard YX mask (existing behavior)
                image_to_use = self.get_current_image_source()
                
                # Get current Z-slice selection (-1 = max projection, else specific Z)
                current_z = getattr(self, 'segmentation_current_z', -1)
                
                if image_to_use.ndim == 5:  # TZYXC format
                    if current_z >= 0 and current_z < image_to_use.shape[1]:
                        # Use specific Z-slice (expand dims to keep 4D: ZYXC -> 1,Y,X,C)
                        img = image_to_use[self.segmentation_current_frame, current_z:current_z+1, :, :, :]
                    else:
                        # Use all Z for max projection (default behavior)
                        img = image_to_use[self.segmentation_current_frame, :, :, :, :]
                else:
                    img = image_to_use
                    
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
            
            # Check if any cells were found
            masks_to_check = self.cellpose_masks_cyto
            if masks_to_check is None or np.max(masks_to_check) == 0:
                QMessageBox.warning(self, "No Cells Found", 
                    "Cellpose did not detect any cells. Try adjusting:\n"
                    "• Cell diameter (larger or smaller)\n"
                    "• Different model type\n"
                    "• Different channel")
            
            self._active_mask_source = 'cellpose'
            # Clear watershed mask since we're using Cellpose now
            self.segmentation_mask = None
            # Record Z-slice used for mask (current_z or -1 for max projection)
            self.segmentation_z_used_for_mask = getattr(self, 'segmentation_current_z', -1)
            
            # Show status message with Z and channel info
            z_info = f"Z={self.segmentation_current_z}" if self.segmentation_current_z >= 0 else "Max Projection"
            n_cells = int(np.max(self.cellpose_masks_cyto)) if self.cellpose_masks_cyto is not None else 0
            self.statusBar().showMessage(f"Cytosol segmented: {n_cells} cells found (Ch{channel}, {z_info})")
            
            # Reset size slider to center (0 = original size)
            if hasattr(self, 'cyto_size_slider'):
                self.cyto_size_slider.blockSignals(True)
                self.cyto_size_slider.setValue(0)
                self.cyto_size_slider.blockSignals(False)
                self.cyto_size_label.setText("0")
            
            self.synchronize_and_plot_cellpose()
            
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Cytosol segmentation failed: {str(e)}")

    def run_cellpose_nuc(self):
        if self.image_stack is None:
            return
            
        try:
            # Get parameters - channel is determined by left panel selection in Segmentation tab
            channel = self.segmentation_current_channel
            diameter = int(self.cellpose_nuc_diameter_input.value())
            model_name = self.cellpose_nuc_model_input.currentText()
            
            # Check if TYX masks are requested (slider > 1 means TYX mode)
            num_masks = self.num_masks_slider.value()
            if (num_masks > 1 and 
                self.image_stack.ndim == 5 and 
                self.image_stack.shape[0] > 1):
                
                max_tp = min(num_masks, self.image_stack.shape[0])
                linking_memory = 1  # Hardcoded to 1
                
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
                    except Exception:
                        progress.setLabelText(msg)
                    QApplication.processEvents()
                
                image_to_use = self.get_current_image_source()
                
                # Check if optimization is requested
                selection_metric = 'max_cells_and_area' if self.chk_optimize_nuc.isChecked() else None
                if image_to_use.shape[0] > 1:
                    diff = np.abs(image_to_use[0].astype(float) - image_to_use[1].astype(float)).sum()
                
                tyx_generator = mi.CellposeTimeSeries(
                    image=image_to_use,
                    channels_cytosol=None,
                    channels_nucleus=channel,
                    diameter_cytosol=150,
                    diameter_nucleus=diameter,
                    max_timepoints=max_tp,
                    linking_memory=linking_memory,
                    model_type_nuc=model_name,
                    progress_callback=progress_callback,
                    selection_metric_nuc=selection_metric
                )
                
                _, masks_nuc_tyx = tyx_generator.calculate_tyx_masks()
                progress.close()
                
                if masks_nuc_tyx is not None:
                    # Filter short-lived masks (artifacts) and reindex IDs
                    min_frames = self.min_frames_slider.value()
                    masks_nuc_tyx = mi.CellposeTimeSeries.filter_short_lived_masks(masks_nuc_tyx, min_frames)
                    
                    self.cellpose_masks_nuc_tyx = masks_nuc_tyx
                    # Also set the current frame's YX mask for compatibility
                    self.cellpose_masks_nuc = masks_nuc_tyx[self.segmentation_current_frame]
                    self.use_tyx_masks = True
                else:
                    self.use_tyx_masks = False
                    
                self.statusBar().showMessage(f"TYX nucleus masks calculated: {max_tp} timepoints")
            else:
                # Standard YX mask (existing behavior)
                image_to_use = self.get_current_image_source()
                
                # Get current Z-slice selection (-1 = max projection, else specific Z)
                current_z = getattr(self, 'segmentation_current_z', -1)
                
                if image_to_use.ndim == 5:  # TZYXC format
                    if current_z >= 0 and current_z < image_to_use.shape[1]:
                        # Use specific Z-slice (expand dims to keep 4D: ZYXC -> 1,Y,X,C)
                        img = image_to_use[self.segmentation_current_frame, current_z:current_z+1, :, :, :]
                    else:
                        # Use all Z for max projection (default behavior)
                        img = image_to_use[self.segmentation_current_frame, :, :, :, :]
                else:
                    img = image_to_use
                    
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
            
            # Check if any nuclei were found
            masks_to_check = self.cellpose_masks_nuc
            if masks_to_check is None or np.max(masks_to_check) == 0:
                QMessageBox.warning(self, "No Nuclei Found", 
                    "Cellpose did not detect any nuclei. Try adjusting:\n"
                    "• Nucleus diameter (larger or smaller)\n"
                    "• Different model type\n"
                    "• Different channel")
            
            self._active_mask_source = 'cellpose'
            # Clear watershed mask since we're using Cellpose now
            self.segmentation_mask = None
            # Record Z-slice used for mask (current_z or -1 for max projection)
            self.segmentation_z_used_for_mask = getattr(self, 'segmentation_current_z', -1)
            
            # Show status message with Z and channel info
            z_info = f"Z={self.segmentation_current_z}" if self.segmentation_current_z >= 0 else "Max Projection"
            n_nuc = int(np.max(self.cellpose_masks_nuc)) if self.cellpose_masks_nuc is not None else 0
            self.statusBar().showMessage(f"Nucleus segmented: {n_nuc} nuclei found (Ch{channel}, {z_info})")
            
            # Reset size slider to center (0 = original size)
            if hasattr(self, 'nuc_size_slider'):
                self.nuc_size_slider.blockSignals(True)
                self.nuc_size_slider.setValue(0)
                self.nuc_size_slider.blockSignals(False)
                self.nuc_size_label.setText("0")
            
            self.synchronize_and_plot_cellpose()
            
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Nucleus segmentation failed: {str(e)}")

    def synchronize_and_plot_cellpose(self):
        # Synchronize if both exist
        if self.cellpose_masks_cyto is not None and self.cellpose_masks_nuc is not None:
             self.cellpose_masks_cyto, self.cellpose_masks_nuc = mi.CellSegmentation.synchronize_masks(
                 self.cellpose_masks_cyto, self.cellpose_masks_nuc
             )
        
        # Synchronize TYX masks if active and both exist
        if (getattr(self, 'use_tyx_masks', False) and 
            getattr(self, 'cellpose_masks_cyto_tyx', None) is not None and 
            getattr(self, 'cellpose_masks_nuc_tyx', None) is not None):
            
            self.cellpose_masks_cyto_tyx, self.cellpose_masks_nuc_tyx = mi.CellSegmentation.synchronize_masks_tyx(
                self.cellpose_masks_cyto_tyx, self.cellpose_masks_nuc_tyx
            )
            
            # Update current 2D frame from synchronized TYX
            if self.current_frame < len(self.cellpose_masks_cyto_tyx):
                self.cellpose_masks_cyto = self.cellpose_masks_cyto_tyx[self.current_frame]
                self.cellpose_masks_nuc = self.cellpose_masks_nuc_tyx[self.current_frame]
        elif (getattr(self, 'use_tyx_masks', False) and 
              getattr(self, 'cellpose_masks_cyto_tyx', None) is None and 
              getattr(self, 'cellpose_masks_nuc_tyx', None) is not None):
            # Update current 2D frame from TYX nucleus masks
            if self.current_frame < len(self.cellpose_masks_nuc_tyx):
                self.cellpose_masks_nuc = self.cellpose_masks_nuc_tyx[self.current_frame]
        elif (getattr(self, 'use_tyx_masks', False) and 
              getattr(self, 'cellpose_masks_cyto_tyx', None) is not None and 
              getattr(self, 'cellpose_masks_nuc_tyx', None) is None):
            # Update current 2D frame from TYX cytosol masks
            if self.current_frame < len(self.cellpose_masks_cyto_tyx):
                self.cellpose_masks_cyto = self.cellpose_masks_cyto_tyx[self.current_frame]

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
        self.masks_imported = False
        # Update import status labels
        if hasattr(self, 'label_cyto_mask_status'):
            self.label_cyto_mask_status.setText("No cytosol mask loaded")
            self.label_cyto_mask_status.setStyleSheet("color: gray;")
        if hasattr(self, 'label_nuc_mask_status'):
            self.label_nuc_mask_status.setText("No nucleus mask loaded")
            self.label_nuc_mask_status.setStyleSheet("color: gray;")
        self.plot_cellpose_results()

    def clear_imported_masks(self):
        """Clear imported masks and reset status labels."""
        self.cellpose_masks_cyto = None
        self.cellpose_masks_nuc = None
        self.cellpose_masks_cyto_tyx = None
        self.cellpose_masks_nuc_tyx = None
        self.use_tyx_masks = False
        self.masks_imported = False
        self._active_mask_source = 'segmentation'
        # Update status labels
        if hasattr(self, 'label_cyto_mask_status'):
            self.label_cyto_mask_status.setText("No cytosol mask loaded")
            self.label_cyto_mask_status.setStyleSheet("color: gray;")
        if hasattr(self, 'label_nuc_mask_status'):
            self.label_nuc_mask_status.setText("No nucleus mask loaded")
            self.label_nuc_mask_status.setStyleSheet("color: gray;")
        self.plot_segmentation()
        self.statusBar().showMessage("Imported masks cleared")

    def import_mask_from_tiff(self, mask_type):
        """Import a mask from a TIFF file.
        
        Args:
            mask_type: 'cytosol' or 'nucleus'
        
        Validates dimensions against current image and normalizes values.
        Supports YX (2D) or TYX (3D) masks.
        """
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        # Get expected dimensions from current image
        T, Z, Y, X, C = self.image_stack.shape
        
        # Open file dialog
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            f"Import {mask_type.capitalize()} Mask",
            "",
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        
        if not filepath:
            return
        
        try:
            # Load the mask
            mask_data = tifffile.imread(filepath)
            
            # Analyze dimensions
            mask_shape = mask_data.shape
            mask_ndim = mask_data.ndim
            
            # Detailed dimension information for user
            dim_info = f"Loaded mask shape: {mask_shape}\n"
            dim_info += f"Expected image dimensions: T={T}, Y={Y}, X={X}\n\n"
            
            # Validate dimensions
            is_valid = False
            is_tyx = False
            
            if mask_ndim == 2:
                # YX mask (single frame)
                mask_Y, mask_X = mask_shape
                if mask_Y == Y and mask_X == X:
                    is_valid = True
                    is_tyx = False
                    dim_info += "✓ Valid 2D (YX) mask - will be applied to all frames.\n"
                else:
                    dim_info += f"✗ Dimension mismatch! Expected YX=({Y}, {X}), got ({mask_Y}, {mask_X})\n"
                    
            elif mask_ndim == 3:
                # TYX mask (time-varying)
                mask_T, mask_Y, mask_X = mask_shape
                if mask_Y == Y and mask_X == X:
                    if mask_T == T:
                        is_valid = True
                        is_tyx = True
                        dim_info += f"✓ Valid 3D (TYX) mask with {mask_T} frames.\n"
                    elif mask_T < T:
                        # Fewer frames than image - ask user
                        reply = QMessageBox.question(
                            self,
                            "Partial TYX Mask",
                            f"Mask has {mask_T} frames but image has {T} frames.\n\n"
                            "Only the first {mask_T} frames will have mask coverage.\n"
                            "Continue anyway?",
                            QMessageBox.Yes | QMessageBox.No
                        )
                        if reply == QMessageBox.Yes:
                            is_valid = True
                            is_tyx = True
                            dim_info += f"⚠ Partial TYX mask: {mask_T}/{T} frames covered.\n"
                        else:
                            return
                    else:
                        # More frames than image
                        reply = QMessageBox.question(
                            self,
                            "Mask Has More Frames",
                            f"Mask has {mask_T} frames but image has only {T} frames.\n\n"
                            f"Only the first {T} frames of the mask will be used.\n"
                            "Continue anyway?",
                            QMessageBox.Yes | QMessageBox.No
                        )
                        if reply == QMessageBox.Yes:
                            mask_data = mask_data[:T]  # Truncate
                            is_valid = True
                            is_tyx = True
                            dim_info += f"⚠ TYX mask truncated to {T} frames.\n"
                        else:
                            return
                else:
                    dim_info += f"✗ Dimension mismatch! Expected YX=({Y}, {X}), got ({mask_Y}, {mask_X})\n"
            else:
                dim_info += f"✗ Unsupported dimension: {mask_ndim}D. Expected 2D (YX) or 3D (TYX).\n"
            
            if not is_valid:
                QMessageBox.critical(
                    self,
                    "Invalid Mask Dimensions",
                    dim_info + "\nPlease ensure the mask matches the image dimensions."
                )
                return
            
            # Analyze and normalize mask values
            unique_values = np.unique(mask_data)
            num_cells = len(unique_values) - (1 if 0 in unique_values else 0)
            min_val, max_val = mask_data.min(), mask_data.max()
            
            # Check if values are appropriate (0 for background, positive integers for cells)
            if min_val < 0:
                QMessageBox.warning(
                    self,
                    "Negative Values",
                    f"Mask contains negative values (min={min_val}).\n"
                    "These will be set to 0 (background)."
                )
                mask_data = np.maximum(mask_data, 0)
            
            # Ensure integer type
            if not np.issubdtype(mask_data.dtype, np.integer):
                mask_data = mask_data.astype(np.int32)
            
            # Show success dialog with details
            value_info = f"Detected {num_cells} unique cell IDs (excluding background).\n"
            value_info += f"Value range: {min_val} to {max_val}"
            
            QMessageBox.information(
                self,
                f"{mask_type.capitalize()} Mask Loaded",
                dim_info + value_info
            )
            
            # Clear other mask sources and set as active
            self.segmentation_mask = None
            self._active_mask_source = 'cellpose'  # Use cellpose path for imported masks
            self.masks_imported = True
            
            # Store the mask
            if mask_type == 'cytosol':
                if is_tyx:
                    self.cellpose_masks_cyto_tyx = mask_data
                    self.cellpose_masks_cyto = mask_data[self.segmentation_current_frame]
                    self.use_tyx_masks = True
                else:
                    self.cellpose_masks_cyto = mask_data
                    self.cellpose_masks_cyto_tyx = None
                    # Only set use_tyx_masks to False if nucleus is not TYX
                    if self.cellpose_masks_nuc_tyx is None:
                        self.use_tyx_masks = False
                
                # Store originals for expansion/shrink
                self._original_cellpose_masks_cyto = self.cellpose_masks_cyto.copy() if self.cellpose_masks_cyto is not None else None
                self._original_cellpose_masks_cyto_tyx = self.cellpose_masks_cyto_tyx.copy() if self.cellpose_masks_cyto_tyx is not None else None
                
                # Update status label
                if is_tyx:
                    self.label_cyto_mask_status.setText(f"✓ Loaded: TYX ({mask_data.shape[0]} frames, {num_cells} cells)")
                else:
                    self.label_cyto_mask_status.setText(f"✓ Loaded: YX ({num_cells} cells)")
                self.label_cyto_mask_status.setStyleSheet("color: limegreen;")
                
            elif mask_type == 'nucleus':
                if is_tyx:
                    self.cellpose_masks_nuc_tyx = mask_data
                    self.cellpose_masks_nuc = mask_data[self.segmentation_current_frame]
                    self.use_tyx_masks = True
                else:
                    self.cellpose_masks_nuc = mask_data
                    self.cellpose_masks_nuc_tyx = None
                    # Only set use_tyx_masks to False if cytosol is not TYX
                    if self.cellpose_masks_cyto_tyx is None:
                        self.use_tyx_masks = False
                
                # Store originals for expansion/shrink
                self._original_cellpose_masks_nuc = self.cellpose_masks_nuc.copy() if self.cellpose_masks_nuc is not None else None
                self._original_cellpose_masks_nuc_tyx = self.cellpose_masks_nuc_tyx.copy() if self.cellpose_masks_nuc_tyx is not None else None
                
                # Update status label
                if is_tyx:
                    self.label_nuc_mask_status.setText(f"✓ Loaded: TYX ({mask_data.shape[0]} frames, {num_cells} cells)")
                else:
                    self.label_nuc_mask_status.setText(f"✓ Loaded: YX ({num_cells} cells)")
                self.label_nuc_mask_status.setStyleSheet("color: limegreen;")
            
            # Plot the results
            self.plot_cellpose_results()
            # Record Z-slice used: -3 indicates imported mask
            self.segmentation_z_used_for_mask = -3  # Special value for imported masks
            self.statusBar().showMessage(f"{mask_type.capitalize()} mask imported successfully")
            
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Import Error", f"Failed to import mask:\n{str(e)}")

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
                    self.cellpose_masks_cyto = self.cellpose_masks_cyto_tyx[self.segmentation_current_frame]
                if self.cellpose_masks_nuc_tyx is not None and border_labels:
                    self.cellpose_masks_nuc_tyx = self._remove_labels_from_tyx(self.cellpose_masks_nuc_tyx, border_labels)
                    # Update current frame YX mask
                    self.cellpose_masks_nuc = self.cellpose_masks_nuc_tyx[self.segmentation_current_frame]
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
        """Handle checkbox state change for removing unpaired cells.
        
        For TYX masks: Finds unpaired IDs across ALL frames and removes them
        from ALL frames to ensure consistent tracking.
        """
        if state == Qt.Checked:
            # Check if we're in TYX mode
            if getattr(self, 'use_tyx_masks', False):
                # TYX mode: handle 3D mask arrays
                if self.cellpose_masks_cyto_tyx is None or self.cellpose_masks_nuc_tyx is None:
                    QMessageBox.warning(self, "Warning", 
                        "Please segment both cytosol and nucleus over time first.")
                    self.chk_remove_unpaired_cells.blockSignals(True)
                    self.chk_remove_unpaired_cells.setChecked(False)
                    self.chk_remove_unpaired_cells.blockSignals(False)
                    return
                
                # Find all IDs present across ALL timepoints
                cyto_ids_all = set(np.unique(self.cellpose_masks_cyto_tyx))
                nuc_ids_all = set(np.unique(self.cellpose_masks_nuc_tyx))
                cyto_ids_all.discard(0)
                nuc_ids_all.discard(0)
                
                # Paired IDs are those present in BOTH mask types (across all time)
                paired_ids = cyto_ids_all & nuc_ids_all
                
                # Remove unpaired cytosols from ALL frames
                unpaired_cyto_ids = cyto_ids_all - paired_ids
                if unpaired_cyto_ids:
                    self.cellpose_masks_cyto_tyx = self._remove_labels_from_tyx(
                        self.cellpose_masks_cyto_tyx, unpaired_cyto_ids)
                
                # Remove unpaired nuclei from ALL frames
                unpaired_nuc_ids = nuc_ids_all - paired_ids
                if unpaired_nuc_ids:
                    self.cellpose_masks_nuc_tyx = self._remove_labels_from_tyx(
                        self.cellpose_masks_nuc_tyx, unpaired_nuc_ids)
                
                # Re-synchronize mask IDs across all timepoints
                # Get final IDs after removal
                final_cyto_ids = set(np.unique(self.cellpose_masks_cyto_tyx))
                final_nuc_ids = set(np.unique(self.cellpose_masks_nuc_tyx))
                final_cyto_ids.discard(0)
                final_nuc_ids.discard(0)
                
                # Synchronize: ensure same IDs in both masks
                common_ids = final_cyto_ids & final_nuc_ids
                ids_to_remove_cyto = final_cyto_ids - common_ids
                ids_to_remove_nuc = final_nuc_ids - common_ids
                
                if ids_to_remove_cyto:
                    self.cellpose_masks_cyto_tyx = self._remove_labels_from_tyx(
                        self.cellpose_masks_cyto_tyx, ids_to_remove_cyto)
                if ids_to_remove_nuc:
                    self.cellpose_masks_nuc_tyx = self._remove_labels_from_tyx(
                        self.cellpose_masks_nuc_tyx, ids_to_remove_nuc)
                
                # Update current frame YX masks
                self.cellpose_masks_cyto = self.cellpose_masks_cyto_tyx[self.segmentation_current_frame]
                self.cellpose_masks_nuc = self.cellpose_masks_nuc_tyx[self.segmentation_current_frame]
                
            else:
                # Standard YX mask handling (non-TYX mode)
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
        """Plot Cellpose segmentation results on the shared segmentation canvas."""
        if self.image_stack is None:
            return
        
        # Use the shared segmentation canvas
        self.ax_segmentation.clear()
        
        # Get current image slice - use registered image if available
        image_to_use = self.get_current_image_source()
        ch = self.segmentation_current_channel  # Use segmentation channel, not separate cellpose channel
        fr = self.segmentation_current_frame    # Use segmentation frame, not separate cellpose frame
        
        if image_to_use.ndim == 5:
            # [T, Z, Y, X, C] -> Use selected Z-slice or max projection
            current_z = getattr(self, 'segmentation_current_z', -1)
            
            if current_z >= 0 and current_z < image_to_use.shape[1]:
                # Use specific Z-slice
                img_slice = image_to_use[fr, current_z, :, :, ch]
            else:
                # Max projection over Z for display (-1 or invalid)
                img_slice = image_to_use[fr, :, :, :, ch]
                if img_slice.ndim == 3:  # ZYX
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
        self.ax_segmentation.imshow(normalized, cmap=cmap_used, vmin=0, vmax=1)
        
        # Overlay Cytosol Masks
        if self.cellpose_masks_cyto is not None:
            # Draw contours
            for label in np.unique(self.cellpose_masks_cyto):
                if label == 0: continue
                mask = self.cellpose_masks_cyto == label
                self.ax_segmentation.contour(mask, levels=[0.5], colors='yellow', linewidths=1)
                
                # Add label ID
                y, x = center_of_mass(mask)
                self.ax_segmentation.text(x, y, str(label), color='yellow', fontsize=8, ha='center', va='center')

        # Overlay Nucleus Masks
        if self.cellpose_masks_nuc is not None:
            # Draw contours
            for label in np.unique(self.cellpose_masks_nuc):
                if label == 0: continue
                mask = self.cellpose_masks_nuc == label
                self.ax_segmentation.contour(mask, levels=[0.5], colors='cyan', linewidths=1)
                
                # Add label ID (if not already added by cyto)
                if self.cellpose_masks_cyto is None:
                    y, x = center_of_mass(mask)
                    self.ax_segmentation.text(x, y, str(label), color='cyan', fontsize=8, ha='center', va='center')

        self.ax_segmentation.grid(False)
        
        # Add axis labels in pixels (helpful for Cellpose diameter estimation)
        height, width = img_slice.shape[:2]
        self.ax_segmentation.set_xlabel('X (pixels)', color='white', fontsize=10)
        self.ax_segmentation.set_ylabel('Y (pixels)', color='white', fontsize=10)
        
        num_x_ticks = 5
        x_tick_positions = np.linspace(0, width - 1, num_x_ticks)
        self.ax_segmentation.set_xticks(x_tick_positions)
        self.ax_segmentation.set_xticklabels([f'{int(pos)}' for pos in x_tick_positions], color='white', fontsize=8)
        
        num_y_ticks = 5
        y_tick_positions = np.linspace(0, height - 1, num_y_ticks)
        self.ax_segmentation.set_yticks(y_tick_positions)
        self.ax_segmentation.set_yticklabels([f'{int(pos)}' for pos in y_tick_positions], color='white', fontsize=8)
        
        self.ax_segmentation.tick_params(axis='both', colors='white', direction='out', length=4)
        self.ax_segmentation.spines['bottom'].set_color('white')
        self.ax_segmentation.spines['left'].set_color('white')
        self.ax_segmentation.spines['top'].set_visible(False)
        self.ax_segmentation.spines['right'].set_visible(False)
        
        # Add subtle grid lines
        self.ax_segmentation.grid(True, linewidth=0.3, alpha=0.3, color='white')
        
        self.canvas_segmentation.draw()


    def create_segmentation_channel_buttons(self):
        for btn in self.segmentation_channel_buttons:
            btn.setParent(None)
        self.segmentation_channel_buttons = []
        for idx, channel_name in enumerate(self.channel_names):
            btn = QPushButton(f"Ch {idx}", self)
            btn.clicked.connect(partial(self.update_segmentation_channel, idx))
            self.segmentation_channel_buttons_layout.addWidget(btn)
            self.segmentation_channel_buttons.append(btn)

    def update_segmentation_channel(self, channel_index):
        # Clear old mask when changing channel
        self.segmentation_mask = None
        self.segmentation_current_channel = channel_index
        
        # Refresh display based on active sub-tab
        if hasattr(self, 'segmentation_method_tabs'):
            current_subtab = self.segmentation_method_tabs.currentIndex()
            if current_subtab == 1 or current_subtab == 3:  # Cellpose or Import sub-tab
                self.plot_cellpose_results()
            else:
                self.plot_segmentation()
        else:
            self.plot_segmentation()

    def update_segmentation_frame(self, value):
        # Clear old manual/watershed mask when changing frame
        self.segmentation_mask = None
        self.segmentation_current_frame = value
        
        # Update frame label
        total_frames = getattr(self, 'total_frames', 1)
        if hasattr(self, 'frame_label_segmentation'):
            self.frame_label_segmentation.setText(f"{value}/{total_frames - 1}")
        
        # Sync TYX Cellpose masks if active
        if getattr(self, 'use_tyx_masks', False):
            if getattr(self, 'cellpose_masks_cyto_tyx', None) is not None:
                self.cellpose_masks_cyto = self.cellpose_masks_cyto_tyx[value]
            if getattr(self, 'cellpose_masks_nuc_tyx', None) is not None:
                self.cellpose_masks_nuc = self.cellpose_masks_nuc_tyx[value]
        
        # Refresh display based on active sub-tab
        if hasattr(self, 'segmentation_method_tabs'):
            current_subtab = self.segmentation_method_tabs.currentIndex()
            if current_subtab == 1 or current_subtab == 3:  # Cellpose or Import sub-tab
                self.plot_cellpose_results()
            else:
                self.plot_segmentation()
        else:
            self.plot_segmentation()

    def run_watershed_segmentation(self):
        if self.image_stack is not None:
            # Use registered image if available
            image_to_use = self.get_current_image_source()
            ch = self.segmentation_current_channel
            if self.use_max_proj_for_segmentation and self.segmentation_maxproj is not None:
                image_to_segment = self.segmentation_maxproj[..., ch]
                # Store Z used for metadata: -1 means max projection
                self.segmentation_z_used_for_mask = -1
            else:
                fr = self.segmentation_current_frame
                image_channel = image_to_use[fr, :, :, :, ch]
                # Apply Z selection
                current_z = getattr(self, 'segmentation_current_z', -1)
                if current_z == -1:
                    # Max Z-projection
                    image_to_segment = np.max(image_channel, axis=0)
                    self.segmentation_z_used_for_mask = -1
                else:
                    # Specific Z-slice
                    z_idx = min(current_z, image_channel.shape[0] - 1)
                    image_to_segment = image_channel[z_idx, :, :]
                    self.segmentation_z_used_for_mask = z_idx
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
            # Clear Cellpose/imported masks since we're using watershed now
            self.cellpose_masks_cyto = None
            self.cellpose_masks_nuc = None
            self.cellpose_masks_cyto_tyx = None
            self.cellpose_masks_nuc_tyx = None
            self.use_tyx_masks = False
            self.masks_imported = False
            # Reset import status labels
            if hasattr(self, 'label_cyto_mask_status'):
                self.label_cyto_mask_status.setText("No cytosol mask loaded")
                self.label_cyto_mask_status.setStyleSheet("color: gray;")
            if hasattr(self, 'label_nuc_mask_status'):
                self.label_nuc_mask_status.setText("No nucleus mask loaded")
                self.label_nuc_mask_status.setStyleSheet("color: gray;")
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
        
        # Update the value label
        if hasattr(self, 'watershed_threshold_label'):
            self.watershed_threshold_label.setText(f"{self.watershed_threshold_factor:.2f}")
        
        # Clear original mask storage since we're generating a new mask
        self._original_watershed_mask = None
        
        # Reset size slider to 0 (no adjustment on new mask)
        if hasattr(self, 'watershed_size_slider'):
            self.watershed_size_slider.blockSignals(True)
            self.watershed_size_slider.setValue(0)
            self.watershed_size_slider.blockSignals(False)
            if hasattr(self, 'watershed_size_label'):
                self.watershed_size_label.setText("0")
        
        if self.image_stack is not None:
            self.run_watershed_segmentation()

    def _on_watershed_size_slider_changed(self, value):
        """
        Apply watershed mask size adjustment when slider value changes.
        Positive = expand, Negative = shrink, 0 = original.
        """
        # Update label
        if hasattr(self, 'watershed_size_label'):
            self.watershed_size_label.setText(str(value))
        
        # Check if we have a watershed/segmentation mask
        if self.segmentation_mask is None:
            return
        
        # Store original if not yet stored
        if self._original_watershed_mask is None:
            self._original_watershed_mask = self.segmentation_mask.copy()
        
        # Apply transformation
        if value == 0:
            # Restore original
            if self._original_watershed_mask is not None:
                self.segmentation_mask = self._original_watershed_mask.copy()
            msg = "Watershed mask restored to original size"
        elif value > 0:
            # Expand
            self.segmentation_mask = self._expand_labeled_mask(
                self._original_watershed_mask, value
            )
            msg = f"Watershed mask expanded by {value}px"
        else:
            # Shrink (value is negative, use abs)
            self.segmentation_mask = self._shrink_labeled_mask(
                self._original_watershed_mask, abs(value)
            )
            msg = f"Watershed mask shrunk by {abs(value)}px"
        
        # Update the active mask source
        self._active_mask_source = 'segmentation'
        
        # Update display
        self.plot_segmentation()
        n_cells = int(self.segmentation_mask.max()) if self.segmentation_mask is not None else 0
        self.statusBar().showMessage(f"{msg}. Cells: {n_cells}")

    def update_segmentation_source(self, state):
        if state == Qt.Checked:
            self.compute_max_proj_segmentation()
            self.use_max_proj_for_segmentation = True
            # Disable time controls since all frames are projected
            self.segmentation_time_slider.setEnabled(False)
            if hasattr(self, 'play_button_segmentation'):
                self.play_button_segmentation.setEnabled(False)
                # Stop any ongoing playback
                if getattr(self, 'playing_segmentation', False):
                    self.timer_segmentation.stop()
                    self.playing_segmentation = False
                    self.play_button_segmentation.setText("Play")
            # Disable Z-slider (temporal max projection includes all Z)
            if hasattr(self, 'segmentation_z_slider'):
                self.segmentation_z_slider.setEnabled(False)
            self.max_proj_status_label.setText("Max projection is ON (T+Z)")
        else:
            self.use_max_proj_for_segmentation = False
            # Re-enable time controls
            self.segmentation_time_slider.setEnabled(True)
            if hasattr(self, 'play_button_segmentation'):
                self.play_button_segmentation.setEnabled(True)
            # Re-enable Z-slider (only if image has multiple Z)
            if hasattr(self, 'segmentation_z_slider'):
                z_dim = self.image_stack.shape[1] if self.image_stack is not None else 1
                self.segmentation_z_slider.setEnabled(z_dim > 1)
            self.max_proj_status_label.setText("Max projection is OFF")
            self.plot_segmentation()

    def compute_max_proj_segmentation(self):
        if self.image_stack is None:
            return
        # Use registered image if available
        image_to_use = self.get_current_image_source()
        self.segmentation_maxproj = np.max(image_to_use, axis=(0, 1))
        self.plot_segmentation()

    def on_segmentation_z_changed(self, value):
        """Handle Z-slider value changes in segmentation tab.
        
        The slider uses inverted appearance so:
        - Top position (value = max) = max Z-projection (default)
        - Bottom position (value = 0) = Z-slice 0
        """
        if not hasattr(self, 'segmentation_z_slider'):
            return
            
        max_val = self.segmentation_z_slider.maximum()
        
        if value == max_val:
            # Top of slider = max Z-projection
            self.segmentation_current_z = -1
            self.segmentation_z_label.setText("Max")
            self.segmentation_z_label.setStyleSheet("color: cyan; font-weight: bold;")
        else:
            # Specific Z-slice (slider value directly maps to Z index)
            self.segmentation_current_z = value
            self.segmentation_z_label.setText(f"Z={value}")
            self.segmentation_z_label.setStyleSheet("color: lime; font-weight: bold;")
        
        # Update display (but don't change the stored mask)
        self.plot_segmentation()

    def reset_segmentation_z_slider(self):
        """Reset Z-slider to default (max projection) and update from image dimensions."""
        if self.image_stack is None:
            self.segmentation_z_max = 0
            if hasattr(self, 'segmentation_z_slider'):
                self.segmentation_z_slider.setMaximum(0)
                self.segmentation_z_slider.setValue(0)
            self.segmentation_current_z = -1
            self.segmentation_z_used_for_mask = -1
            return
        
        # Get Z dimension from image (TZYXC format, Z is axis 1)
        z_dim = self.image_stack.shape[1]
        self.segmentation_z_max = z_dim
        
        if hasattr(self, 'segmentation_z_slider'):
            # For inverted slider: max value = max Z index (max projection at top)
            self.segmentation_z_slider.blockSignals(True)
            self.segmentation_z_slider.setMinimum(0)
            self.segmentation_z_slider.setMaximum(z_dim)  # max = "Max Projection" position
            self.segmentation_z_slider.setValue(z_dim)    # Default to top (max projection)
            self.segmentation_z_slider.blockSignals(False)
            
            # Enable slider only if more than 1 Z-slice (otherwise it can't move)
            self.segmentation_z_slider.setEnabled(z_dim > 1)
        
        self.segmentation_current_z = -1  # Default to max projection
        self.segmentation_z_used_for_mask = -1
        
        if hasattr(self, 'segmentation_z_label'):
            self.segmentation_z_label.setText("Max")
            self.segmentation_z_label.setStyleSheet("color: cyan; font-weight: bold;")


    def plot_segmentation(self):
        self.figure_segmentation.clear()
        self.ax_segmentation = self.figure_segmentation.add_subplot(111)
        self.ax_segmentation.set_facecolor('black')
        if self.image_stack is not None:
            # Use registered image if available
            image_to_use = self.get_current_image_source()
            ch = self.segmentation_current_channel
            # Choose image to display (temporal max projection vs current frame, then Z selection)
            if self.use_max_proj_for_segmentation and self.segmentation_maxproj is not None:
                # Temporal max projection (already max over T and Z)
                image_to_display = self.segmentation_maxproj[..., ch]
            else:
                # Get current time frame's Z-stack for this channel
                image_channel = image_to_use[self.segmentation_current_frame, :, :, :, ch]
                # Apply Z selection
                if getattr(self, 'segmentation_current_z', -1) == -1:
                    # Max Z-projection (default)
                    image_to_display = np.max(image_channel, axis=0)
                else:
                    # Specific Z-slice
                    z_idx = min(self.segmentation_current_z, image_channel.shape[0] - 1)
                    image_to_display = image_channel[z_idx, :, :]
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
            
            # Add axis labels in pixels (helpful for Cellpose diameter estimation)
            height, width = image_to_display.shape[:2]
            self.ax_segmentation.set_xlabel('X (pixels)', color='white', fontsize=10)
            self.ax_segmentation.set_ylabel('Y (pixels)', color='white', fontsize=10)
            
            num_x_ticks = 5
            x_tick_positions = np.linspace(0, width - 1, num_x_ticks)
            self.ax_segmentation.set_xticks(x_tick_positions)
            self.ax_segmentation.set_xticklabels([f'{int(pos)}' for pos in x_tick_positions], color='white', fontsize=8)
            
            num_y_ticks = 5
            y_tick_positions = np.linspace(0, height - 1, num_y_ticks)
            self.ax_segmentation.set_yticks(y_tick_positions)
            self.ax_segmentation.set_yticklabels([f'{int(pos)}' for pos in y_tick_positions], color='white', fontsize=8)
            
            self.ax_segmentation.tick_params(axis='both', colors='white', direction='out', length=4)
            self.ax_segmentation.spines['bottom'].set_color('white')
            self.ax_segmentation.spines['left'].set_color('white')
            self.ax_segmentation.spines['top'].set_visible(False)
            self.ax_segmentation.spines['right'].set_visible(False)
            
            # Add subtle grid lines
            self.ax_segmentation.grid(True, linewidth=0.3, alpha=0.3, color='white')
        else:
            self.ax_segmentation.text(
                0.5, 0.5, 'No image loaded.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='white', transform=self.ax_segmentation.transAxes
            )
            self.ax_segmentation.axis('off')
        self.figure_segmentation.tight_layout()
        self.canvas_segmentation.draw()

    def setup_registration_tab(self):
        """
        Setup the Registration tab with dual-panel layout:
        - Left panel: Original image with square ROI selection
        - Right panel: Registered image result
        - Bottom: Channel buttons, time slider, mode dropdown, registration buttons
        """
        layout = QVBoxLayout(self.registration_tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # --- Workflow Instructions ---
        instructions_frame = QFrame()
        instructions_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a3a;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        instructions_layout = QVBoxLayout(instructions_frame)
        instructions_layout.setContentsMargins(10, 8, 10, 8)
        instructions_layout.setSpacing(4)
        
        workflow_label = QLabel("📋 <b>Workflow:</b>  ① Draw ROI on the left image (click & drag)  →  ② Click 'Perform Registration'")
        workflow_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        instructions_layout.addWidget(workflow_label)
        
        # ROI Status indicator
        roi_status_layout = QHBoxLayout()
        roi_status_layout.setSpacing(10)
        
        self.roi_status_label = QLabel("⬜ ROI: Not Selected")
        self.roi_status_label.setStyleSheet("color: #ff9900; font-size: 11px; font-weight: bold;")
        roi_status_layout.addWidget(self.roi_status_label)
        
        roi_status_layout.addStretch()
        
        # Registration status
        self.registration_status_label = QLabel("")
        self.registration_status_label.setStyleSheet("color: #888888; font-size: 10px;")
        roi_status_layout.addWidget(self.registration_status_label)
        
        instructions_layout.addLayout(roi_status_layout)
        layout.addWidget(instructions_frame)
        
        # --- Top: Dual image panels ---
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(10)
        
        # Left panel: Original Image
        left_panel = QFrame()
        left_panel.setStyleSheet("""
            QFrame {
                border: 2px solid #555;
                border-radius: 4px;
                background-color: #1a1a1a;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(2)
        
        left_title = QLabel("🖼️ Original Image")
        left_title.setStyleSheet("color: white; font-weight: bold; font-size: 11px; border: none; background: transparent;")
        left_layout.addWidget(left_title)
        
        self.figure_reg_original = Figure()
        self.figure_reg_original.patch.set_facecolor('#1a1a1a')
        self.figure_reg_original.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        self.canvas_reg_original = FigureCanvas(self.figure_reg_original)
        self.ax_reg_original = self.figure_reg_original.add_subplot(111)
        self.ax_reg_original.set_facecolor('black')
        self.ax_reg_original.axis('off')
        left_layout.addWidget(self.canvas_reg_original, stretch=1)
        panels_layout.addWidget(left_panel)
        
        # Right panel: Registered Image
        right_panel = QFrame()
        right_panel.setStyleSheet("""
            QFrame {
                border: 2px solid #555;
                border-radius: 4px;
                background-color: #1a1a1a;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(2)
        
        right_title = QLabel("✅ Registered Image")
        right_title.setStyleSheet("color: white; font-weight: bold; font-size: 11px; border: none; background: transparent;")
        right_layout.addWidget(right_title)
        
        self.figure_reg_result = Figure()
        self.figure_reg_result.patch.set_facecolor('#1a1a1a')
        self.figure_reg_result.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        self.canvas_reg_result = FigureCanvas(self.figure_reg_result)
        self.ax_reg_result = self.figure_reg_result.add_subplot(111)
        self.ax_reg_result.set_facecolor('black')
        self.ax_reg_result.axis('off')
        right_layout.addWidget(self.canvas_reg_result, stretch=1)
        panels_layout.addWidget(right_panel)
        
        layout.addLayout(panels_layout, stretch=1)
        
        # --- Controls row: Channel + Time slider ---
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)
        
        # Channel buttons
        channel_layout = QHBoxLayout()
        channel_layout.setSpacing(5)
        channel_layout.addWidget(QLabel("Channel:"))
        self.channel_buttons_reg = []
        self.channel_buttons_layout_reg = QHBoxLayout()
        channel_layout.addLayout(self.channel_buttons_layout_reg)
        controls_layout.addLayout(channel_layout)
        
        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: #555;")
        controls_layout.addWidget(sep1)
        
        # Time slider
        time_layout = QHBoxLayout()
        time_layout.setSpacing(5)
        time_layout.addWidget(QLabel("Time:"))
        self.time_slider_reg = QSlider(Qt.Horizontal)
        self.time_slider_reg.setMinimum(0)
        self.time_slider_reg.setMaximum(0)
        self.time_slider_reg.valueChanged.connect(self.on_registration_time_changed)
        time_layout.addWidget(self.time_slider_reg, stretch=1)
        
        self.frame_label_reg = QLabel("0/0")
        self.frame_label_reg.setMinimumWidth(50)
        time_layout.addWidget(self.frame_label_reg)
        
        self.play_button_reg = QPushButton("▶")
        self.play_button_reg.setFixedWidth(40)
        self.play_button_reg.clicked.connect(self.toggle_playback_registration)
        time_layout.addWidget(self.play_button_reg)
        controls_layout.addLayout(time_layout, stretch=1)
        
        layout.addLayout(controls_layout)
        
        # --- Action row: Mode + Buttons ---
        action_layout = QHBoxLayout()
        action_layout.setSpacing(15)
        
        # Mode dropdown
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(5)
        mode_layout.addWidget(QLabel("Mode:"))
        self.registration_mode_combo = QComboBox()
        self.registration_mode_combo.addItems(['RIGID_BODY', 'TRANSLATION', 'SCALED_ROTATION', 'AFFINE'])
        self.registration_mode_combo.setCurrentText('RIGID_BODY')
        self.registration_mode_combo.setMinimumWidth(140)
        self.registration_mode_combo.currentTextChanged.connect(self.on_registration_mode_changed)
        mode_layout.addWidget(self.registration_mode_combo)
        action_layout.addLayout(mode_layout)
        
        action_layout.addStretch()
        
        # Main action button - PROMINENT
        self.perform_registration_btn = QPushButton("▶  Perform Registration")
        self.perform_registration_btn.setMinimumHeight(45)
        self.perform_registration_btn.setMinimumWidth(220)
        self.perform_registration_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                font-size: 13px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #34c759;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        self.perform_registration_btn.clicked.connect(self.perform_registration)
        action_layout.addWidget(self.perform_registration_btn)
        
        # Remove button - less prominent
        self.remove_registration_btn = QPushButton("✕ Remove")
        self.remove_registration_btn.setMinimumHeight(45)
        self.remove_registration_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                color: #ccc;
                font-size: 11px;
                border-radius: 6px;
                border: none;
                padding: 0 15px;
            }
            QPushButton:hover {
                background-color: #dc3545;
                color: white;
            }
        """)
        self.remove_registration_btn.clicked.connect(self.remove_registration)
        action_layout.addWidget(self.remove_registration_btn)
        
        layout.addLayout(action_layout)
        
        # --- ROI drawing state ---
        self.reg_roi_start = None
        self.reg_roi_rect = None
        self.reg_timer = QTimer()
        self.reg_timer.timeout.connect(self.registration_next_frame)
        self.reg_playing = False
        
        # --- Connect mouse events for ROI drawing ---
        self.canvas_reg_original.mpl_connect('button_press_event', self.on_reg_mouse_press)
        self.canvas_reg_original.mpl_connect('motion_notify_event', self.on_reg_mouse_move)
        self.canvas_reg_original.mpl_connect('button_release_event', self.on_reg_mouse_release)
    
    def on_registration_mode_changed(self, mode):
        """Update registration mode when dropdown changes."""
        self.registration_mode = mode
    
    def on_registration_time_changed(self, value):
        """Update display when time slider changes in registration tab."""
        self.current_frame = value
        # Update Registration label only
        total_frames = getattr(self, 'total_frames', 1)
        if hasattr(self, 'frame_label_reg'):
            self.frame_label_reg.setText(f"{value}/{total_frames - 1}")
        self.plot_registration_panels()
    
    def toggle_playback_registration(self):
        """Toggle play/pause for registration tab."""
        if self.reg_playing:
            self.reg_timer.stop()
            self.play_button_reg.setText("▶")
            self.reg_playing = False
        else:
            self.reg_timer.start(200)
            self.play_button_reg.setText("❚❚")
            self.reg_playing = True
    
    def registration_next_frame(self):
        """Advance to next frame during playback."""
        if self.image_stack is None:
            return
        max_frame = self.image_stack.shape[0] - 1
        next_frame = (self.current_frame + 1) % (max_frame + 1)
        self.time_slider_reg.setValue(next_frame)
    
    def on_reg_mouse_press(self, event):
        """Start drawing square ROI on mouse press."""
        if event.inaxes != self.ax_reg_original or event.xdata is None:
            return
        self.reg_roi_start = (int(event.xdata), int(event.ydata))
        # Remove old preview rectangle (may fail if axes were cleared)
        if self.reg_roi_rect is not None:
            try:
                self.reg_roi_rect.remove()
            except (NotImplementedError, ValueError):
                pass
            self.reg_roi_rect = None
        # Clear stored ROI and refresh to remove old drawn rectangle
        if self.registration_roi is not None:
            self.registration_roi = None
            self.plot_registration_panels()
    
    def on_reg_mouse_move(self, event):
        """Update ROI rectangle preview during drag."""
        if self.reg_roi_start is None or event.inaxes != self.ax_reg_original or event.xdata is None:
            return
        x0, y0 = self.reg_roi_start
        x1, y1 = int(event.xdata), int(event.ydata)
        
        # Update rectangle preview (allow rectangles of any size)
        if self.reg_roi_rect is not None:
            try:
                self.reg_roi_rect.remove()
            except (NotImplementedError, ValueError):
                pass
            self.reg_roi_rect = None
        rect_x = min(x0, x1)
        rect_y = min(y0, y1)
        rect_w = abs(x1 - x0)
        rect_h = abs(y1 - y0)
        self.reg_roi_rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h,
                                               linewidth=2, edgecolor='cyan', facecolor='none')
        self.ax_reg_original.add_patch(self.reg_roi_rect)
        self.canvas_reg_original.draw_idle()
    
    def on_reg_mouse_release(self, event):
        """Finalize ROI rectangle on mouse release."""
        if self.reg_roi_start is None:
            return
        if event.xdata is None or event.inaxes != self.ax_reg_original:
            self.reg_roi_start = None
            return
        
        x0, y0 = self.reg_roi_start
        x1, y1 = int(event.xdata), int(event.ydata)
        
        # Normalize bounds (allow any rectangle)
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        
        # Silently clamp to safe region (10px margin from borders)
        margin = 10
        if self.image_stack is not None:
            H, W = self.image_stack.shape[2], self.image_stack.shape[3]
            x_min = max(margin, x_min)
            y_min = max(margin, y_min)
            x_max = min(W - margin, x_max)
            y_max = min(H - margin, y_max)
        
        area = (x_max - x_min) * (y_max - y_min)
        if area < 20:
            # Silently reject too-small ROI
            self.reg_roi_start = None
            return
        
        # Store ROI as (y_min, y_max, x_min, x_max)
        self.registration_roi = (y_min, y_max, x_min, x_max)
        self.reg_roi_start = None
        
        # Update ROI status label
        if hasattr(self, 'roi_status_label'):
            roi_w = x_max - x_min
            roi_h = y_max - y_min
            self.roi_status_label.setText(f"ROI: {roi_w} x {roi_h} px")
            self.roi_status_label.setStyleSheet("color: #00cc66; font-size: 11px; font-weight: bold;")
        
        # Update display
        self.plot_registration_panels()
    
    def plot_registration_panels(self):
        """Plot both original and registered image panels."""
        if self.image_stack is None:
            return
        
        # Get current channel
        ch = self.current_channel
        cmap = cmap_list_imagej[ch % len(cmap_list_imagej)]
        
        # Get image dimensions
        H, W = self.image_stack.shape[2], self.image_stack.shape[3]
        
        # --- Original panel ---
        self.ax_reg_original.clear()
        self.ax_reg_original.set_facecolor('#1a1a1a')
        
        # Max Z projection of current frame and channel
        img_orig = np.max(self.image_stack[self.current_frame, :, :, :, ch], axis=0)
        # Use configurable percentile values for contrast (consistent with other tabs)
        vmin, vmax = np.percentile(img_orig, [self.display_min_percentile, self.display_max_percentile])
        self.ax_reg_original.imshow(img_orig, cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Draw image border to show boundaries
        img_border = patches.Rectangle((0, 0), W-1, H-1, linewidth=1.5, 
                                        edgecolor='#666666', facecolor='none', linestyle='-')
        self.ax_reg_original.add_patch(img_border)
        
        # Draw ROI rectangle if exists
        if self.registration_roi is not None:
            y_min, y_max, x_min, x_max = self.registration_roi
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                       linewidth=2, edgecolor='cyan', facecolor='none')
            self.ax_reg_original.add_patch(rect)
        
        self.ax_reg_original.axis('off')
        self.figure_reg_original.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        self.canvas_reg_original.draw_idle()
        
        # --- Registered panel ---
        self.ax_reg_result.clear()
        self.ax_reg_result.set_facecolor('#1a1a1a')
        
        if self.registered_image is not None:
            img_reg = np.max(self.registered_image[self.current_frame, :, :, :, ch], axis=0)
            # Use percentile on non-zero pixels only (outside ROI is zeros)
            nonzero_pixels = img_reg[img_reg > 0]
            if len(nonzero_pixels) > 0:
                vmin, vmax = np.percentile(nonzero_pixels, [0.5, 99.5])
            else:
                vmin, vmax = 0, 1
            self.ax_reg_result.imshow(img_reg, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Draw image border
            img_border_reg = patches.Rectangle((0, 0), W-1, H-1, linewidth=1.5, 
                                                edgecolor='#666666', facecolor='none', linestyle='-')
            self.ax_reg_result.add_patch(img_border_reg)
            
            # Draw ROI rectangle on registered image too
            if self.registration_roi is not None:
                y_min, y_max, x_min, x_max = self.registration_roi
                rect_reg = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              linewidth=2, edgecolor='cyan', facecolor='none')
                self.ax_reg_result.add_patch(rect_reg)
        else:
            self.ax_reg_result.text(0.5, 0.5, "Click 'Perform Registration' to start", ha='center', va='center',
                                     color='#888888', fontsize=11, transform=self.ax_reg_result.transAxes)
        
        self.ax_reg_result.axis('off')
        self.figure_reg_result.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        self.canvas_reg_result.draw_idle()
    
    def perform_registration(self):
        """Perform image registration using the selected ROI and mode."""
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        T = self.image_stack.shape[0]
        if T <= 1:
            QMessageBox.warning(self, "Single Timepoint", "Registration not necessary for single timepoint images.")
            return
        
        # If no ROI selected, use full image with padding and show warning
        padding = 10
        if self.registration_roi is None:
            H, W = self.image_stack.shape[2], self.image_stack.shape[3]
            # Use full image minus padding from all edges
            self.registration_roi = (padding, H - padding, padding, W - padding)
            QMessageBox.warning(
                self, 
                "Using Full Image", 
                "No ROI selected. Using full image for registration.\n\n"
                "Tip: Registration may be more accurate if you select a specific "
                "region with distinct features (e.g., a single cell or bright structure)."
            )
        
        # Create progress dialog
        progress = QProgressDialog("Performing Registration...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Registration")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()
        
        def update_progress(message):
            if "Progress:" in message:
                try:
                    pct = int(message.split(":")[1].split("%")[0].strip())
                    progress.setValue(pct)
                except Exception:
                    pass
            progress.setLabelText(message)
            QApplication.processEvents()
        
        try:
            reg = mi.Registration(
                image=self.image_stack,
                roi_bounds=self.registration_roi,
                reference_channel=self.current_channel,
                mode=self.registration_mode,
                padding=10,
                progress_callback=update_progress
            )
            self.registered_image = reg.get_registered_image()
            progress.setValue(100)
            
            # Check if existing masks have different dimensions than registered image
            reg_shape = self.registered_image.shape[2:4]  # [Y, X]
            masks_invalidated = False
            mask_names = []
            
            if self.cellpose_masks_cyto is not None and self.cellpose_masks_cyto.shape != reg_shape:
                masks_invalidated = True
                mask_names.append("Cellpose cytosol")
            if self.cellpose_masks_nuc is not None and self.cellpose_masks_nuc.shape != reg_shape:
                masks_invalidated = True
                mask_names.append("Cellpose nucleus")
            if self.segmentation_mask is not None and self.segmentation_mask.shape != reg_shape:
                masks_invalidated = True
                mask_names.append("Watershed/Manual")
            
            if masks_invalidated:
                # Clear masks that are now invalid
                self.cellpose_masks_cyto = None
                self.cellpose_masks_nuc = None
                self.cellpose_masks_cyto_tyx = None
                self.cellpose_masks_nuc_tyx = None
                self.segmentation_mask = None
                self._active_mask_source = 'none'
                
                # Also clear photobleaching and tracking that depended on them
                self.corrected_image = None
                self.photobleaching_calculated = False
                self.df_tracking = pd.DataFrame()
                self.multi_channel_tracking_data = {}
                self.tracked_channels = []
                
                QMessageBox.warning(
                    self, 
                    "Masks Invalidated",
                    f"Registration changed the image dimensions.\n\n"
                    f"The following masks have been cleared because they no longer match:\n"
                    f"• {', '.join(mask_names)}\n\n"
                    f"Please re-run segmentation on the registered image."
                )
            
            self.plot_registration_panels()
            # Update other tabs that use the registered image
            self.plot_segmentation()
            if hasattr(self, 'plot_cellpose_results'):
                self.plot_cellpose_results()
            self.statusBar().showMessage(f"Registration complete using {self.registration_mode} mode.")
            # Update registration status label
            if hasattr(self, 'registration_status_label'):
                self.registration_status_label.setText(f"✅ Registered ({self.registration_mode})")
                self.registration_status_label.setStyleSheet("color: #00cc66; font-size: 10px; font-weight: bold;")
        except Exception as e:
            QMessageBox.critical(self, "Registration Error", str(e))
        finally:
            progress.close()
    
    def remove_registration(self):
        """Remove registration and reset to original image."""
        self.registered_image = None
        self.registration_roi = None
        
        # Clear photobleaching correction that was based on registered image
        # (corrected_image may have been computed from registered_image)
        if self.corrected_image is not None:
            self.corrected_image = None
            self.photobleaching_calculated = False
            self.photobleaching_data = None
            self.statusBar().showMessage("Registration removed. Photobleaching correction also reset.")
        else:
            self.statusBar().showMessage("Registration removed.")
        
        if hasattr(self, 'reg_roi_rect') and self.reg_roi_rect is not None:
            try:
                self.reg_roi_rect.remove()
            except Exception:
                pass
            self.reg_roi_rect = None
        # Reset status labels
        if hasattr(self, 'roi_status_label'):
            self.roi_status_label.setText("⬜ ROI: Not Selected")
            self.roi_status_label.setStyleSheet("color: #ff9900; font-size: 11px; font-weight: bold;")
        if hasattr(self, 'registration_status_label'):
            self.registration_status_label.setText("")
        self.plot_registration_panels()
    
    def reset_registration_state(self):
        """Reset all registration state. Called on new image load or close."""
        self.registered_image = None
        self.registration_roi = None
        if hasattr(self, 'reg_roi_rect') and self.reg_roi_rect is not None:
            try:
                self.reg_roi_rect.remove()
            except Exception:
                pass
            self.reg_roi_rect = None
    
    def update_registration_channel(self, idx):
        """Update channel for registration tab and redraw."""
        self.current_channel = idx
        self.plot_registration_panels()

    def setup_segmentation_tab(self):
        """
        Set up the segmentation tab UI.
        Initializes internal state for segmentation and assembles a two-panel interface
        with controls and display components. Includes Cellpose as an integrated sub-tab.
        
        Left Panel:
            • Matplotlib figure & canvas for segmentation preview
            • Frame navigation slider
            • Channel selection buttons container
            • Navigation toolbar
            • Export buttons for segmentation image and mask
        
        Right Panel:
            • Sub-tabs for segmentation methods (QTabWidget):
                - "Watershed" tab: Watershed threshold slider and run button
                - "Cellpose" tab: Deep learning segmentation with cytosol/nucleus options
                - "Import" tab: Import pre-computed masks from TIFF files (YX or TYX)
                - "Manual" tab: Manual polygon segmentation buttons
            • Maximum projection toggle checkbox and status label (below sub-tabs)
        
        Import Masks Sub-Tab Features:
            • Import cytosol mask button (supports YX or TYX TIFFs)
            • Import nucleus mask button (supports YX or TYX TIFFs)
            • Dimension validation against current image
            • Value normalization (0=background, 1,2,3...=cell IDs)
            • Status labels showing loaded mask info
        
        Cellpose Sub-Tab Features (shares the segmentation canvas and time controls):
            • Cytosol/Nucleus segmentation controls with model selection
            • Time-varying masks options
            • Improve segmentation options (border cells, expand/shrink sliders)
        
        Attributes Created on self:
            segmentation_current_frame (int)
            segmentation_current_channel (int)
            use_max_proj_for_segmentation (bool)
            segmentation_maxproj (Optional[np.ndarray])
            figure_segmentation (matplotlib.figure.Figure)
            ax_segmentation (matplotlib.axes.Axes)
            canvas_segmentation (FigureCanvas)
            segmentation_time_slider (QSlider)
            frame_label_segmentation (QLabel): Shows current frame / total frames
            play_button_segmentation (QPushButton): Play/Pause time playback
            segmentation_channel_buttons_layout (QHBoxLayout)
            toolbar_segmentation (NavigationToolbar)
            export_segmentation_image_button (QPushButton)
            export_mask_button (QPushButton)
            segmentation_method_tabs (QTabWidget)
            use_max_proj_checkbox (QCheckBox)
            max_proj_status_label (QLabel)
            segmentation_button (QPushButton)
            finish_segmentation_button (QPushButton)
            watershed_threshold_slider (QSlider)
            watershed_threshold_label (QLabel)
            watershed_size_slider (QSlider)
            watershed_size_label (QLabel)
            # Import Masks-specific attributes:
            btn_import_cyto_mask, btn_import_nuc_mask (QPushButton)
            label_cyto_mask_status, label_nuc_mask_status (QLabel)
            btn_clear_imported_masks (QPushButton)
            masks_imported (bool): True if masks were imported vs generated
            # Cellpose-specific attributes (Cellpose shares the segmentation canvas):
            cellpose_cyto_* and cellpose_nuc_* inputs (model, channel, diameter)
            btn_run_cyto, btn_run_nuc (QPushButton)
            num_masks_slider, min_frames_slider, cell_expansion_slider, cell_shrink_slider
            chk_remove_border_cells, chk_remove_unpaired_cells (QCheckBox)
            btn_clear_cellpose (QPushButton)
            cellpose_masks_cyto, cellpose_masks_nuc (Optional[np.ndarray])
            cellpose_masks_cyto_tyx, cellpose_masks_nuc_tyx (Optional[np.ndarray])
        
        Connected Signals:
            update_segmentation_frame
            export_segmentation_image
            export_mask_as_tiff
            update_segmentation_source
            manual_segmentation
            finish_segmentation
            update_watershed_threshold_factor
            run_watershed_segmentation
            _on_segmentation_subtab_changed
            # Cellpose signals:
            update_cellpose_frame, play_pause_cellpose
            run_cellpose_cyto, run_cellpose_nuc, clear_cellpose_masks
            _on_num_masks_slider_changed, _on_min_frames_slider_changed
            _on_expansion_slider_changed, _on_shrink_slider_changed
            on_remove_border_cells_changed, on_remove_unpaired_cells_changed
        """

        self.segmentation_current_frame = 0
        self.segmentation_current_channel = 0
        self.use_max_proj_for_segmentation = False
        self.segmentation_maxproj = None
        # Z-plane state variables
        self.segmentation_current_z = -1  # -1 means "max Z-projection" (default) - for DISPLAY
        self.segmentation_z_max = 0  # Will be set from image dimensions
        self.segmentation_z_used_for_mask = -1  # Z value actually used for segmentation (for metadata)
        
        # Create main horizontal layout for segmentation tab
        main_layout = QHBoxLayout(self.segmentation_tab)
        # LEFT PANEL: Segmentation Figure & Controls
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)
        
        # Create canvas + Z-slider layout (horizontal: canvas on left, Z-slider on right)
        canvas_z_layout = QHBoxLayout()
        
        # Create segmentation figure and canvas
        self.figure_segmentation, self.ax_segmentation = plt.subplots()
        self.figure_segmentation.patch.set_facecolor('black')
        self.canvas_segmentation = FigureCanvas(self.figure_segmentation)
        canvas_z_layout.addWidget(self.canvas_segmentation, stretch=1)
        
        # Z-slider (vertical, on the right of canvas)
        z_slider_layout = QVBoxLayout()
        z_slider_layout.setContentsMargins(5, 0, 5, 0)
        
        z_label_top = QLabel("Z")
        z_label_top.setAlignment(Qt.AlignCenter)
        z_label_top.setStyleSheet("color: white; font-weight: bold;")
        z_slider_layout.addWidget(z_label_top)
        
        self.segmentation_z_slider = QSlider(Qt.Vertical)
        self.segmentation_z_slider.setMinimum(0)
        self.segmentation_z_slider.setMaximum(0)  # Will be set when image loads
        self.segmentation_z_slider.setTickPosition(QSlider.TicksLeft)
        self.segmentation_z_slider.setTickInterval(1)
        self.segmentation_z_slider.setInvertedAppearance(True)  # Top = highest Z index (max projection)
        self.segmentation_z_slider.valueChanged.connect(self.on_segmentation_z_changed)
        z_slider_layout.addWidget(self.segmentation_z_slider, stretch=1)
        
        self.segmentation_z_label = QLabel("Max")
        self.segmentation_z_label.setAlignment(Qt.AlignCenter)
        self.segmentation_z_label.setStyleSheet("color: cyan; font-weight: bold;")
        self.segmentation_z_label.setMinimumWidth(40)
        z_slider_layout.addWidget(self.segmentation_z_label)
        
        canvas_z_layout.addLayout(z_slider_layout)
        left_layout.addLayout(canvas_z_layout, stretch=1)
        # Create lower controls on left panel: channel buttons, time slider, toolbar, etc.
        left_controls_layout = QVBoxLayout()
        # Top row: channel buttons + time slider + frame label + play button
        top_controls_layout = QHBoxLayout()
        self.segmentation_channel_buttons = []
        self.segmentation_channel_buttons_layout = QHBoxLayout()
        top_controls_layout.addLayout(self.segmentation_channel_buttons_layout)
        
        # Time slider
        self.segmentation_time_slider = QSlider(Qt.Horizontal)
        self.segmentation_time_slider.setMinimum(0)
        self.segmentation_time_slider.setTickPosition(QSlider.TicksBelow)
        self.segmentation_time_slider.setTickInterval(10)
        self.segmentation_time_slider.valueChanged.connect(self.update_segmentation_frame)
        top_controls_layout.addWidget(self.segmentation_time_slider)
        
        # Frame label (shows current frame / total frames)
        self.frame_label_segmentation = QLabel("0/0")
        self.frame_label_segmentation.setMinimumWidth(60)
        self.frame_label_segmentation.setAlignment(Qt.AlignCenter)
        top_controls_layout.addWidget(self.frame_label_segmentation)
        
        # Play/Pause button
        self.play_button_segmentation = QPushButton("Play", self)
        self.play_button_segmentation.setMinimumWidth(60)
        self.play_button_segmentation.clicked.connect(self.play_pause_segmentation)
        top_controls_layout.addWidget(self.play_button_segmentation)
        
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
        
        # =====================================================================
        # SEGMENTATION METHOD SUB-TABS
        # =====================================================================
        self.segmentation_method_tabs = QTabWidget()
        
        # --- Manual Segmentation Tab ---
        manual_tab = QWidget()
        manual_tab_layout = QVBoxLayout(manual_tab)
        manual_tab_layout.setContentsMargins(10, 10, 10, 10)
        manual_tab_layout.setSpacing(10)
        
        # Instructions label
        manual_instructions = QLabel(
            "Draw a polygon by clicking points on the image.\n"
            "Click 'Manual Segmentation' to start, then click to add vertices.\n"
            "Click 'Finish Segmentation' to complete the polygon."
        )
        manual_instructions.setWordWrap(True)
        manual_instructions.setStyleSheet("color: gray; font-size: 11px;")
        manual_tab_layout.addWidget(manual_instructions)
        
        # Button row for manual segmentation
        button_layout = QHBoxLayout()
        self.segmentation_button = QPushButton("Manual Segmentation", self)
        self.segmentation_button.clicked.connect(self.manual_segmentation)
        button_layout.addWidget(self.segmentation_button)
        self.finish_segmentation_button = QPushButton("Finish Segmentation", self)
        self.finish_segmentation_button.clicked.connect(self.finish_segmentation)
        button_layout.addWidget(self.finish_segmentation_button)
        manual_tab_layout.addLayout(button_layout)
        manual_tab_layout.addStretch()
        
        # Manual tab will be added last (after Cellpose)
        
        # --- Watershed Segmentation Tab (Cytosol Only) ---
        watershed_tab = QWidget()
        watershed_tab_layout = QVBoxLayout(watershed_tab)
        watershed_tab_layout.setContentsMargins(10, 10, 10, 10)
        watershed_tab_layout.setSpacing(6)
        
        # Instructions label - emphasize cytosol-only and slider-driven
        watershed_instructions = QLabel(
            "🔬 <b>Cytosol Segmentation</b> using watershed algorithm.<br>"
            "Drag the slider below to detect cell boundaries.<br>"
            "<i>Segmentation updates automatically.</i>"
        )
        watershed_instructions.setTextFormat(Qt.RichText)
        watershed_instructions.setWordWrap(True)
        watershed_instructions.setStyleSheet("color: #cccccc; font-size: 10px; padding: 4px; background-color: #2a2a2a; border-radius: 4px;")
        watershed_tab_layout.addWidget(watershed_instructions)
        
        # === MAIN THRESHOLD SLIDER (Prominent, Green, Large) ===
        threshold_group = QGroupBox("Detection Threshold")
        threshold_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                color: #00ff88; 
                font-size: 12px;
                border: 2px solid #00aa55;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        threshold_group_layout = QVBoxLayout(threshold_group)
        threshold_group_layout.setSpacing(6)
        threshold_group_layout.setContentsMargins(10, 15, 10, 10)
        
        # Descriptive hints row
        hints_layout = QHBoxLayout()
        hint_decrease = QLabel("← Decrease area")
        hint_decrease.setStyleSheet("color: #888888; font-size: 10px;")
        hint_increase = QLabel("Increase area →")
        hint_increase.setStyleSheet("color: #888888; font-size: 10px;")
        hints_layout.addWidget(hint_decrease)
        hints_layout.addStretch()
        hints_layout.addWidget(hint_increase)
        threshold_group_layout.addLayout(hints_layout)
        
        # Large slider with green styling (like tracking tab)
        self.watershed_threshold_slider = QSlider(Qt.Horizontal)
        self.watershed_threshold_slider.setMinimum(10)
        self.watershed_threshold_slider.setMaximum(200)
        self.watershed_threshold_slider.setValue(100)
        self.watershed_threshold_slider.setMinimumHeight(30)  # Taller slider
        self.watershed_threshold_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 10px;
                background: #333;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #00ff88;
                border: 2px solid #00aa55;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #44ffaa;
            }
            QSlider::sub-page:horizontal {
                background: linear-gradient(to right, #005533, #00aa55);
                border-radius: 5px;
            }
        """)
        self.watershed_threshold_slider.setToolTip("Drag to adjust detection threshold")
        self.watershed_threshold_slider.valueChanged.connect(self.update_watershed_threshold_factor)
        threshold_group_layout.addWidget(self.watershed_threshold_slider)
        
        # Value label showing current threshold factor (centered below slider)
        value_layout = QHBoxLayout()
        value_layout.addStretch()
        self.watershed_threshold_label = QLabel("1.00")
        self.watershed_threshold_label.setAlignment(Qt.AlignCenter)
        self.watershed_threshold_label.setStyleSheet("color: #00ff88; font-weight: bold; font-size: 14px;")
        value_layout.addWidget(self.watershed_threshold_label)
        value_layout.addStretch()
        threshold_group_layout.addLayout(value_layout)
        
        watershed_tab_layout.addWidget(threshold_group)
        
        # === MASK SIZE ADJUSTMENT (Modest, compact) ===
        size_layout = QHBoxLayout()
        size_layout.setSpacing(8)
        
        size_label = QLabel("Size Adjust:")
        size_label.setStyleSheet("color: #888888; font-size: 10px;")
        size_layout.addWidget(size_label)
        
        self.watershed_size_slider = QSlider(Qt.Horizontal)
        self.watershed_size_slider.setMinimum(-20)
        self.watershed_size_slider.setMaximum(20)
        self.watershed_size_slider.setValue(0)
        self.watershed_size_slider.setMaximumWidth(100)  # Compact slider
        self.watershed_size_slider.setToolTip("Fine-tune mask size: - = shrink, + = expand")
        self.watershed_size_slider.valueChanged.connect(self._on_watershed_size_slider_changed)
        size_layout.addWidget(self.watershed_size_slider)
        
        self.watershed_size_label = QLabel("0")
        self.watershed_size_label.setMinimumWidth(25)
        self.watershed_size_label.setAlignment(Qt.AlignCenter)
        self.watershed_size_label.setStyleSheet("color: #888888; font-size: 10px;")
        size_layout.addWidget(self.watershed_size_label)
        
        size_layout.addStretch()
        watershed_tab_layout.addLayout(size_layout)
        
        watershed_tab_layout.addStretch()
        
        self.segmentation_method_tabs.addTab(watershed_tab, "Watershed")  # Index 0
        
        # --- Cellpose Segmentation Tab ---
        # NOTE: Cellpose shares the main segmentation canvas (figure_segmentation)
        # This tab only contains the controls, not a separate canvas
        cellpose_tab = QWidget()
        cellpose_tab_layout = QVBoxLayout(cellpose_tab)
        cellpose_tab_layout.setContentsMargins(10, 10, 10, 10)
        cellpose_tab_layout.setSpacing(10)
        
        # Create scrollable area for Cellpose controls
        cellpose_scroll = QScrollArea()
        cellpose_scroll.setWidgetResizable(True)
        cellpose_scroll.setFrameShape(QFrame.NoFrame)
        cellpose_content = QWidget()
        cellpose_layout = QVBoxLayout(cellpose_content)
        cellpose_layout.setSpacing(8)
        
        # Instructions
        cellpose_instructions = QLabel(
            "Deep learning-based cell segmentation using Cellpose.\n"
            "Segment cytosol and/or nucleus using pretrained models."
        )
        cellpose_instructions.setWordWrap(True)
        cellpose_instructions.setStyleSheet("color: gray; font-size: 11px;")
        cellpose_layout.addWidget(cellpose_instructions)
        
        # Cytosol Segmentation Group
        cyto_group = QGroupBox("Cytosol Segmentation")
        cyto_layout = QFormLayout()
        
        self.cellpose_cyto_model_input = QComboBox()
        self.cellpose_cyto_model_input.addItems(['cyto3', 'cyto2', 'cyto'])
        self.cellpose_cyto_model_input.setCurrentText('cyto3')
        cyto_layout.addRow("Model:", self.cellpose_cyto_model_input)
        
        self.cellpose_cyto_diameter_input = QDoubleSpinBox()
        self.cellpose_cyto_diameter_input.setRange(0, 1000)
        self.cellpose_cyto_diameter_input.setValue(150)
        cyto_layout.addRow("Diameter (px):", self.cellpose_cyto_diameter_input)
        
        self.chk_optimize_cyto = QCheckBox("Optimize Parameters")
        self.chk_optimize_cyto.setChecked(False)
        cyto_layout.addRow(self.chk_optimize_cyto)
        
        self.btn_run_cyto = QPushButton("Segment Cytosol")
        self.btn_run_cyto.clicked.connect(self.run_cellpose_cyto)
        cyto_layout.addRow(self.btn_run_cyto)
        
        # Cytosol Size Adjustment Slider (-20 to +20, centered at 0)
        cyto_size_layout = QHBoxLayout()
        cyto_size_layout.addWidget(QLabel("Size Adjust:"))
        self.cyto_size_slider = QSlider(Qt.Horizontal)
        self.cyto_size_slider.setMinimum(-20)
        self.cyto_size_slider.setMaximum(20)
        self.cyto_size_slider.setValue(0)
        self.cyto_size_slider.setTickPosition(QSlider.TicksBelow)
        self.cyto_size_slider.setTickInterval(5)
        self.cyto_size_slider.setToolTip(
            "Adjust cytosol mask size: 0 = original, + = expand, - = shrink (px)"
        )
        self.cyto_size_slider.valueChanged.connect(self._on_cyto_size_slider_changed)
        cyto_size_layout.addWidget(self.cyto_size_slider)
        self.cyto_size_label = QLabel("0")
        self.cyto_size_label.setMinimumWidth(30)
        cyto_size_layout.addWidget(self.cyto_size_label)
        cyto_layout.addRow(cyto_size_layout)
        
        cyto_group.setLayout(cyto_layout)
        cellpose_layout.addWidget(cyto_group)
        
        # Nucleus Segmentation Group
        nuc_group = QGroupBox("Nucleus Segmentation")
        nuc_layout = QFormLayout()
        
        self.cellpose_nuc_model_input = QComboBox()
        self.cellpose_nuc_model_input.addItems(['nuclei', 'cyto3', 'cyto2', 'cyto'])
        self.cellpose_nuc_model_input.setCurrentText('nuclei')
        nuc_layout.addRow("Model:", self.cellpose_nuc_model_input)
        
        self.cellpose_nuc_diameter_input = QDoubleSpinBox()
        self.cellpose_nuc_diameter_input.setRange(0, 1000)
        self.cellpose_nuc_diameter_input.setValue(60)
        nuc_layout.addRow("Diameter (px):", self.cellpose_nuc_diameter_input)
        
        self.chk_optimize_nuc = QCheckBox("Optimize Parameters")
        self.chk_optimize_nuc.setChecked(False)
        nuc_layout.addRow(self.chk_optimize_nuc)
        
        self.btn_run_nuc = QPushButton("Segment Nucleus")
        self.btn_run_nuc.clicked.connect(self.run_cellpose_nuc)
        nuc_layout.addRow(self.btn_run_nuc)
        
        # Nucleus Size Adjustment Slider (-20 to +20, centered at 0)
        nuc_size_layout = QHBoxLayout()
        nuc_size_layout.addWidget(QLabel("Size Adjust:"))
        self.nuc_size_slider = QSlider(Qt.Horizontal)
        self.nuc_size_slider.setMinimum(-20)
        self.nuc_size_slider.setMaximum(20)
        self.nuc_size_slider.setValue(0)
        self.nuc_size_slider.setTickPosition(QSlider.TicksBelow)
        self.nuc_size_slider.setTickInterval(5)
        self.nuc_size_slider.setToolTip(
            "Adjust nucleus mask size: 0 = original, + = expand, - = shrink (px)"
        )
        self.nuc_size_slider.valueChanged.connect(self._on_nuc_size_slider_changed)
        nuc_size_layout.addWidget(self.nuc_size_slider)
        self.nuc_size_label = QLabel("0")
        self.nuc_size_label.setMinimumWidth(30)
        nuc_size_layout.addWidget(self.nuc_size_label)
        nuc_layout.addRow(nuc_size_layout)
        
        nuc_group.setLayout(nuc_layout)
        cellpose_layout.addWidget(nuc_group)
        
        # Time-Varying Masks Group (TYX)
        tyx_group = QGroupBox("Time-Varying Masks")
        tyx_main_layout = QVBoxLayout()
        
        # Number of Frames to Calculate Masks slider
        num_frames_layout = QVBoxLayout()
        num_frames_label = QLabel("Number of Frames to Calculate Masks:")
        num_frames_layout.addWidget(num_frames_label)
        
        num_frames_slider_layout = QHBoxLayout()
        self.num_masks_slider = QSlider(Qt.Horizontal)
        self.num_masks_slider.setRange(1, 100)
        self.num_masks_slider.setValue(1)
        self.num_masks_slider.setTickPosition(QSlider.TicksBelow)
        self.num_masks_slider.setTickInterval(10)
        self.num_masks_slider.setToolTip(
            "Number of frames to calculate masks. Default=1 calculates a single mask. "
            "Values >1 enable time-varying mask calculation (TYX mode)."
        )
        self.num_masks_slider.valueChanged.connect(self._on_num_masks_slider_changed)
        num_frames_slider_layout.addWidget(self.num_masks_slider)
        
        self.num_masks_value_label = QLabel("1")
        self.num_masks_value_label.setMinimumWidth(30)
        num_frames_slider_layout.addWidget(self.num_masks_value_label)
        num_frames_layout.addLayout(num_frames_slider_layout)
        tyx_main_layout.addLayout(num_frames_layout)
        
        # Minimal Frames to Detect a Cell slider
        min_frames_layout = QVBoxLayout()
        min_frames_label = QLabel("Minimal Frames to Detect a Cell:")
        min_frames_layout.addWidget(min_frames_label)
        
        min_frames_slider_layout = QHBoxLayout()
        self.min_frames_slider = QSlider(Qt.Horizontal)
        self.min_frames_slider.setRange(1, 100)
        self.min_frames_slider.setValue(1)
        self.min_frames_slider.setTickPosition(QSlider.TicksBelow)
        self.min_frames_slider.setTickInterval(10)
        self.min_frames_slider.setToolTip(
            "Minimum number of frames a cell must exist to be kept. "
            "Cells appearing for fewer frames are removed as artifacts."
        )
        self.min_frames_slider.valueChanged.connect(self._on_min_frames_slider_changed)
        min_frames_slider_layout.addWidget(self.min_frames_slider)
        
        self.min_frames_value_label = QLabel("1")
        self.min_frames_value_label.setMinimumWidth(30)
        min_frames_slider_layout.addWidget(self.min_frames_value_label)
        min_frames_layout.addLayout(min_frames_slider_layout)
        tyx_main_layout.addLayout(min_frames_layout)
        
        tyx_group.setLayout(tyx_main_layout)
        cellpose_layout.addWidget(tyx_group)
        
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
        cellpose_layout.addWidget(improve_group)
        
        # Clear Button
        self.btn_clear_cellpose = QPushButton("Clear Masks & IDs")
        self.btn_clear_cellpose.clicked.connect(self.clear_cellpose_masks)
        cellpose_layout.addWidget(self.btn_clear_cellpose)
        
        cellpose_layout.addStretch()
        cellpose_scroll.setWidget(cellpose_content)
        cellpose_tab_layout.addWidget(cellpose_scroll)
        
        self.segmentation_method_tabs.addTab(cellpose_tab, "Cellpose")  # Index 1
        
        # --- Import Masks Tab ---
        import_tab = QWidget()
        import_tab_layout = QVBoxLayout(import_tab)
        import_tab_layout.setContentsMargins(10, 10, 10, 10)
        import_tab_layout.setSpacing(10)
        
        # Instructions label
        import_instructions = QLabel(
            "Import pre-computed masks from TIFF files.\n"
            "Masks can be 2D (YX) or 3D (TYX) for time-varying analysis.\n"
            "Pixel values should be: 0=background, 1,2,3...=cell IDs."
        )
        import_instructions.setWordWrap(True)
        import_instructions.setStyleSheet("color: gray; font-size: 11px;")
        import_tab_layout.addWidget(import_instructions)
        
        # Cytosol Mask Import Group
        cyto_import_group = QGroupBox("Cytosol Mask")
        cyto_import_layout = QVBoxLayout()
        
        self.btn_import_cyto_mask = QPushButton("Import Cytosol Mask (.tif)")
        self.btn_import_cyto_mask.clicked.connect(lambda: self.import_mask_from_tiff('cytosol'))
        cyto_import_layout.addWidget(self.btn_import_cyto_mask)
        
        self.label_cyto_mask_status = QLabel("No cytosol mask loaded")
        self.label_cyto_mask_status.setStyleSheet("color: gray;")
        cyto_import_layout.addWidget(self.label_cyto_mask_status)
        
        cyto_import_group.setLayout(cyto_import_layout)
        import_tab_layout.addWidget(cyto_import_group)
        
        # Nucleus Mask Import Group
        nuc_import_group = QGroupBox("Nucleus Mask")
        nuc_import_layout = QVBoxLayout()
        
        self.btn_import_nuc_mask = QPushButton("Import Nucleus Mask (.tif)")
        self.btn_import_nuc_mask.clicked.connect(lambda: self.import_mask_from_tiff('nucleus'))
        nuc_import_layout.addWidget(self.btn_import_nuc_mask)
        
        self.label_nuc_mask_status = QLabel("No nucleus mask loaded")
        self.label_nuc_mask_status.setStyleSheet("color: gray;")
        nuc_import_layout.addWidget(self.label_nuc_mask_status)
        
        nuc_import_group.setLayout(nuc_import_layout)
        import_tab_layout.addWidget(nuc_import_group)
        
        # Clear imported masks button
        self.btn_clear_imported_masks = QPushButton("Clear Imported Masks")
        self.btn_clear_imported_masks.clicked.connect(self.clear_imported_masks)
        import_tab_layout.addWidget(self.btn_clear_imported_masks)
        
        import_tab_layout.addStretch()
        
        # Add Manual tab (Index 2) before Import
        self.segmentation_method_tabs.addTab(manual_tab, "Manual")
        
        # Add Import tab last (Index 3)
        self.segmentation_method_tabs.addTab(import_tab, "Import")
        
        # Initialize Cellpose/imported mask state variables
        # NOTE: Uses segmentation_current_frame and segmentation_current_channel (shared)
        self.cellpose_masks_cyto = None
        self.cellpose_masks_nuc = None
        self.cellpose_masks_cyto_tyx = None
        self.cellpose_masks_nuc_tyx = None
        self.use_tyx_masks = False
        self._original_cellpose_masks_cyto = None
        self._original_cellpose_masks_nuc = None
        self._original_cellpose_masks_cyto_tyx = None
        self._original_cellpose_masks_nuc_tyx = None
        self._original_watershed_mask = None  # Store original watershed mask for size adjustment
        self.masks_imported = False  # Track if masks were imported vs generated
        
        # Connect sub-tab change to refresh the display
        self.segmentation_method_tabs.currentChanged.connect(self._on_segmentation_subtab_changed)
        
        # Add the tabs with stretch factor so they expand to fill available space
        right_layout.addWidget(self.segmentation_method_tabs, stretch=1)
        
        # =====================================================================
        # MAXIMUM PROJECTION GROUP (below sub-tabs)
        # =====================================================================
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
        
        # No stretch at the end - let the tabs expand to fill available space
        self.plot_segmentation()
    
    def _on_segmentation_subtab_changed(self, index):
        """Handle switching between segmentation method sub-tabs (Watershed, Cellpose, Manual, Import).
        
        Tab indices:
            0 = Watershed
            1 = Cellpose
            2 = Manual
            3 = Import (uses Cellpose-style display)
        """
        if index == 1 or index == 3:  # Cellpose or Import sub-tab
            # Show Cellpose/imported masks on the shared segmentation canvas
            self.plot_cellpose_results()
        else:
            # For Watershed and Manual, use the standard segmentation display
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
            # Inform user via status bar (non-blocking)
            self.statusBar().showMessage("Cellpose detected: using entire image for photobleaching calculation.", 5000)
        
        self.photobleaching_mode = mode
        radius = self.radius_slider.value()
        
        if self.segmentation_mask is None:
            mask_GUI = None 
        else:
            mask_GUI = self.segmentation_mask.copy().astype(int)
            mask_GUI = np.where(mask_GUI > 0, 1, 0)
            mask_GUI.setflags(write=1)

        # Step 1: Calculate photobleaching parameters from RAW image (avoids registration artifacts)
        raw_photobleaching_obj = mi.Photobleaching(
            image_TZYXC=self.image_stack,  # Always use raw for parameter calculation
            mask_YX=mask_GUI,
            show_plot=False,
            mode=mode,
            radius=radius,
            time_interval_seconds=self.time_interval_value if self.time_interval_value is not None else 1.0
        )
        decay_params = raw_photobleaching_obj.calculate_photobleaching()
        
        # Store raw mean intensities for plotting (not affected by registration artifacts)
        raw_mean_intensities = raw_photobleaching_obj.mean_intensities.copy()
        raw_err_intensities = raw_photobleaching_obj.err_intensities.copy()
        
        # MEMORY OPTIMIZATION: Delete raw_photobleaching_obj to free memory
        # (It holds a reference to the full image array)
        del raw_photobleaching_obj
        gc.collect()

        # Step 2: Determine which image to apply correction to
        image_for_correction = self.registered_image if self.registered_image is not None else self.image_stack

        # Step 3: Apply correction using pre-calculated parameters from raw image
        correction_obj = mi.Photobleaching(
            image_TZYXC=image_for_correction,  # Apply to registered (or raw if no registration)
            mask_YX=mask_GUI,
            show_plot=False,
            mode=mode,
            radius=radius,
            time_interval_seconds=self.time_interval_value if self.time_interval_value is not None else 1.0,
            precalulated_list_decay_rates=decay_params  # Use params from raw image
        )
        self.corrected_image, self.photobleaching_data = correction_obj.apply_photobleaching_correction()
        
        # MEMORY OPTIMIZATION: Delete correction_obj to free memory
        del correction_obj
        gc.collect()
        
        # Override mean_intensities in photobleaching_data with raw image values for accurate plot
        # This shows the actual decay curve (from raw) as "Original" in the plot
        self.photobleaching_data['mean_intensities'] = raw_mean_intensities
        self.photobleaching_data['err_intensities'] = raw_err_intensities
        
        # Also compute corrected intensities from RAW intensities (not registered)
        # This avoids registration edge artifacts in the "Corrected" line
        T, C = raw_mean_intensities.shape
        time_array = self.photobleaching_data['time_array']
        params = decay_params
        
        # Calculate correction factors and apply to raw intensities
        raw_intensities_corrected = np.zeros_like(raw_mean_intensities)
        raw_err_corrected = np.zeros_like(raw_err_intensities)
        
        for ch in range(C):
            k_fit = params[2*ch]
            I0_fit = params[2*ch + 1]
            
            if k_fit > 0:
                # Compute correction factor: I0 / I_fit(t) = exp(k*t)
                correction_factors = np.exp(k_fit * time_array)
                raw_intensities_corrected[:, ch] = raw_mean_intensities[:, ch] * correction_factors
                raw_err_corrected[:, ch] = raw_err_intensities[:, ch] * correction_factors
            else:
                # No correction applied for this channel
                raw_intensities_corrected[:, ch] = raw_mean_intensities[:, ch]
                raw_err_corrected[:, ch] = raw_err_intensities[:, ch]
        
        self.photobleaching_data['mean_intensities_corrected'] = raw_intensities_corrected
        self.photobleaching_data['err_intensities_corrected'] = raw_err_corrected
        
        self.photobleaching_calculated = True
        
        # Auto-select "Photobleaching Corrected" in tracking tab image source
        if hasattr(self, 'image_source_combo'):
            self.image_source_combo.setCurrentIndex(1)  # "Photobleaching Corrected"
        
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
            axs[ch, 0].plot(t, fitted_curve, '-', label=f'I₀={I0_fit:.0f}, k={k_fit:.2e}', color='white', lw=2)
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
        
        # Mode selection
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["inside_cell", "outside_cell", "use_circular_region", "entire_image"])
        self.mode_combo.setCurrentText("entire_image")  # Default to entire_image mode
        self.mode_combo.setMinimumWidth(140)
        controls_layout.addWidget(mode_label)
        controls_layout.addWidget(self.mode_combo)
        
        controls_layout.addSpacing(20)
        
        # Radius slider (replaces spinbox)
        radius_label = QLabel("Radius:")
        controls_layout.addWidget(radius_label)
        
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setMinimum(10)
        self.radius_slider.setMaximum(100)
        self.radius_slider.setValue(30)
        self.radius_slider.setMinimumWidth(120)
        self.radius_slider.setMaximumWidth(200)
        self.radius_slider.setTickPosition(QSlider.TicksBelow)
        self.radius_slider.setTickInterval(10)
        controls_layout.addWidget(self.radius_slider)
        
        self.radius_value_label = QLabel("30")
        self.radius_value_label.setMinimumWidth(30)
        self.radius_value_label.setStyleSheet("font-weight: bold;")
        controls_layout.addWidget(self.radius_value_label)
        
        # Connect slider to update label
        self.radius_slider.valueChanged.connect(
            lambda v: self.radius_value_label.setText(str(v))
        )
        
        controls_layout.addStretch()
        
        # Photobleaching run button - PROMINENT (styled like Registration button)
        self.run_photobleaching_button = QPushButton("Run Photobleaching")
        self.run_photobleaching_button.setMinimumHeight(40)
        self.run_photobleaching_button.setMinimumWidth(180)
        self.run_photobleaching_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                font-size: 12px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #34c759;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
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

        # Inform user about single-frame limitation
        if getattr(self, 'total_frames', 0) == 1:
            self.statusBar().showMessage(
                "Single frame detected: Running detection only (linking requires multiple frames)"
            )

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
                step_size_in_sec=float(self.time_interval_value) if self.time_interval_value is not None else 1.0,
                fast_gaussian_fit=self.fast_gaussian_fit,
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
        """Returns the current image source in priority order: corrected > registered > original.
        
        Priority:
        1. Photobleaching corrected image (highest - this would be correction of registered if that was used)
        2. Registered image (stabilized)
        3. Original raw image
        
        The user selection combo box is for display purposes - the processing pipeline
        always uses the most processed image available for consistency.
        """
        # Return corrected image if available (highest priority)
        # This handles the case where photobleaching is applied to registered image
        if self.corrected_image is not None:
            return self.corrected_image
        # Then check for registered image
        if self.registered_image is not None:
            return self.registered_image
        # Fall back to original
        return self.image_stack

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
            # Block signals to prevent triggering detection when just updating histogram
            self.threshold_slider.blockSignals(True)
            self.threshold_slider.setValue(slider_min)
            self.threshold_slider.blockSignals(False)
        else:
            self.ax_threshold_hist.axvline(self.user_selected_threshold, color='orangered', linestyle='-', lw=3)
        self.canvas_threshold_hist.draw_idle()

    def update_threshold_value(self, value):
        # Update the threshold value label
        if hasattr(self, 'threshold_value_label'):
            self.threshold_value_label.setText(f"Value: {value}")
        
        if self.image_stack is None:
            self.ax_threshold_hist.clear()
            self.ax_threshold_hist.set_facecolor('black')
            self.ax_threshold_hist.axis('off')
            self.canvas_threshold_hist.draw_idle()
            return
        
        # Clear current channel's data when threshold changes
        # This discards previous tracking/detection for fresh analysis
        current_ch = self.current_channel
        if current_ch in self.multi_channel_tracking_data:
            del self.multi_channel_tracking_data[current_ch]
        if current_ch in self.tracked_channels:
            self.tracked_channels.remove(current_ch)
        if current_ch in self.tracking_thresholds:
            del self.tracking_thresholds[current_ch]
        if current_ch in self.tracking_parameters_per_channel:
            del self.tracking_parameters_per_channel[current_ch]
        
        # Update primary channel if needed
        if self.primary_tracking_channel == current_ch:
            self.primary_tracking_channel = self.tracked_channels[0] if self.tracked_channels else None
        
        # Rebuild combined dataframe
        self._rebuild_combined_tracking_dataframe()
        self._update_tracked_channels_list()
        
        # Clear detection preview
        self.detected_spots_frame = None
        
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
        
        # Set figure background
        self.figure_threshold_hist.patch.set_facecolor('#1a1a1a')
        self.ax_threshold_hist.set_facecolor('#1a1a1a')
        
        self.ax_threshold_hist.hist(
            intensity_values,
            bins=bins_to_use,
            range=(lower_limit, upper_limit),
            color='aliceblue',
            edgecolor='#333333'
        )
        self.ax_threshold_hist.set_xlim(lower_limit, upper_limit)
        self.ax_threshold_hist.set_yticks([])
        self.ax_threshold_hist.grid(False)
        
        # Make x-axis ticks visible with white color
        self.ax_threshold_hist.tick_params(axis='x', which='major', labelsize=8, colors='white', length=3)
        self.ax_threshold_hist.spines['bottom'].set_color('white')
        self.ax_threshold_hist.spines['top'].set_visible(False)
        self.ax_threshold_hist.spines['left'].set_visible(False)
        self.ax_threshold_hist.spines['right'].set_visible(False)
        
        self.ax_threshold_hist.axvline(self.user_selected_threshold, color='orangered', linestyle='-', lw=2)
        
        # Tight layout to ensure ticks are visible
        self.figure_threshold_hist.tight_layout(pad=0.3)
        self.canvas_threshold_hist.draw_idle()
        self.plot_tracking()  # Update display to clear old spots
        self.detect_spots_in_current_frame()

    def _reset_threshold_for_new_channel(self):
        """Reset threshold slider to minimum when switching to a new (untracked) channel."""
        if not hasattr(self, 'threshold_slider'):
            return
            
        # Reset user threshold to None to trigger auto-mode
        self.user_selected_threshold = None
        
        # Set slider to minimum value
        slider_min = self.threshold_slider.minimum()
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(slider_min)
        self.threshold_slider.blockSignals(False)
        
        # Update label to indicate auto/unset
        if hasattr(self, 'threshold_value_label'):
            self.threshold_value_label.setText("Value: Auto")
        
        # Redraw histogram without threshold line
        self.update_threshold_histogram()

    def on_auto_threshold_clicked(self):
        """Handle auto-threshold button click - calculate optimal threshold automatically."""
        if self.image_stack is None:
            self.statusBar().showMessage("No image loaded")
            return
        
        channel = self.current_channel
        
        # Get current image source (follows pipeline: corrected > registered > original)
        image_to_use = self.get_current_image_source()
        if image_to_use is None:
            self.statusBar().showMessage("No image available")
            return
        
        # Show progress
        self.statusBar().showMessage("Calculating optimal threshold...")
        QApplication.processEvents()
        
        try:
            # Get current frame's image for this channel
            # Shape: [Z, Y, X]
            image_channel = image_to_use[self.current_frame, :, :, :, channel]
            
            # Determine if using 3D or 2D mode
            use_3d = not self.use_maximum_projection
            
            # Get voxel sizes
            voxel_yx = getattr(self, 'voxel_yx_nm', 130.0)
            voxel_z = getattr(self, 'voxel_z_nm', 300.0)
            
            # Get spot sizes
            yx_spot_size = getattr(self, 'yx_spot_size_in_px', 5)
            z_spot_size = getattr(self, 'z_spot_size_in_px', 2)
            
            # Calculate threshold using AutoThreshold class
            auto_thresh = mi.AutoThreshold(
                image=image_channel,
                voxel_size_yx=voxel_yx,
                voxel_size_z=voxel_z,
                yx_spot_size_in_px=yx_spot_size,
                z_spot_size_in_px=z_spot_size,
                use_3d=use_3d
            )
            threshold = auto_thresh.calculate()
            method_used = auto_thresh.method_used
            
            # Store per-channel
            self.auto_threshold_per_channel[channel] = threshold
            
            # Update user_selected_threshold
            self.user_selected_threshold = threshold
            
            # Update slider range if needed (ensure threshold fits)
            current_max = self.threshold_slider.maximum()
            if threshold > current_max:
                self.threshold_slider.setMaximum(int(threshold * 1.2))
            
            # Update slider (block signals to prevent recursion)
            self.threshold_slider.blockSignals(True)
            self.threshold_slider.setValue(int(threshold))
            self.threshold_slider.blockSignals(False)
            
            # Update value label
            if hasattr(self, 'threshold_value_label'):
                self.threshold_value_label.setText(f"Value: {int(threshold)}")
            
            # Update histogram with threshold line
            self.update_threshold_histogram()
            
            # Auto-run single frame detection
            self.detect_spots_in_current_frame()
            
            # Show result
            self.statusBar().showMessage(
                f"Auto-threshold Ch{channel}: {int(threshold)} (method: {method_used})"
            )
            
        except Exception as e:
            traceback.print_exc()
            self.statusBar().showMessage(f"Auto-threshold failed: {str(e)}")

    def on_image_source_changed(self):
        self.image_source_combo_value = self.image_source_combo.currentText()
        self.plot_tracking()

    def update_min_length_trajectory(self, value):
        self.min_length_trajectory = value
    
    def _calculate_optimal_min_trajectory(self, total_frames):
        """
        Calculate optimal min_length_trajectory based on movie length.
        
        Scaling:
        - Very short movies (≤5 frames): 1-2 (allow minimal trajectories)
        - Short movies (5-20 frames): 2-10
        - Medium movies (20-80 frames): 10-20
        - Long movies (80+ frames): 20-50 (capped)
        
        Args:
            total_frames: Number of frames in the movie
            
        Returns:
            int: Optimal min trajectory length
        """
        if total_frames <= 1:
            return 1  # Single frame: allow detection-only (no linking possible)
        elif total_frames <= 5:
            return max(2, total_frames // 2)  # At least 2, but no more than half
        elif total_frames <= 10:
            return 5
        elif total_frames <= 20:
            # Scale from 5 to 10 for 10-20 frames
            return 5 + int((total_frames - 10) * 5 / 10)
        elif total_frames <= 40:
            # Scale from 10 to 15 for 20-40 frames
            return 10 + int((total_frames - 20) * 5 / 20)
        elif total_frames <= 80:
            # Scale from 15 to 25 for 40-80 frames
            return 15 + int((total_frames - 40) * 10 / 40)
        elif total_frames <= 200:
            # Scale from 25 to 50 for 80-200 frames
            return 25 + int((total_frames - 80) * 25 / 120)
        else:
            # Cap at 50 for very long movies
            return 50

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
        # Update legacy label
        if hasattr(self, 'tracking_max_proj_status_label'):
            self.tracking_max_proj_status_label.setText("2D Projection is ON" if self.use_maximum_projection else "2D Projection is OFF")
        # Update new toggle buttons and status
        self._update_tracking_mode_buttons()
        self._update_tracking_mode_status()

    def update_max_range_search_pixels(self, value):
        self.maximum_range_search_pixels = value

    def update_memory(self, value):
        self.memory = value

    def update_use_fixed_size_intensity(self, state):
        self.use_fixed_size_for_intensity_calculation = (state == Qt.Checked)

    def update_fast_gaussian_fit(self, state):
        self.fast_gaussian_fit = (state == Qt.Checked)

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

    def _set_tracking_mode(self, is_2d):
        """Set the tracking mode (2D projection or 3D volume) and update UI."""
        self.use_maximum_projection = is_2d
        # Sync with legacy checkbox
        if hasattr(self, 'use_2d_projection_checkbox'):
            self.use_2d_projection_checkbox.blockSignals(True)
            self.use_2d_projection_checkbox.setChecked(is_2d)
            self.use_2d_projection_checkbox.blockSignals(False)
        # Update button styles and status
        self._update_tracking_mode_buttons()
        self._update_tracking_mode_status()
        # Update legacy label if it exists
        if hasattr(self, 'tracking_max_proj_status_label'):
            self.tracking_max_proj_status_label.setText("2D Projection is ON" if is_2d else "2D Projection is OFF")
        
        # Reset threshold when switching modes (2D/3D use different algorithms)
        # The threshold calculated for 2D is not valid for 3D and vice versa
        if hasattr(self, 'auto_threshold_per_channel'):
            self.auto_threshold_per_channel.clear()  # Clear all auto-thresholds
        
        # Reset user threshold and slider
        self.user_selected_threshold = None
        if hasattr(self, 'threshold_slider'):
            self.threshold_slider.blockSignals(True)
            self.threshold_slider.setValue(0)
            self.threshold_slider.blockSignals(False)
        if hasattr(self, 'threshold_value_label'):
            self.threshold_value_label.setText("Value: --")
        
        # Clear any detected spots
        self.detected_spots_frame = None
        
        # Update histogram (without threshold line)
        self.update_threshold_histogram()
        
        # Redraw tracking plot (will show no spots)
        self.plot_tracking()

    def _update_tracking_mode_buttons(self):
        """Update the visual state of 2D/3D toggle buttons."""
        is_2d = self.use_maximum_projection
        
        # Check if 3D mode is even possible (need Z > 1)
        z_dim = self.image_stack.shape[1] if self.image_stack is not None else 1
        z_insufficient = (z_dim <= 1)
        
        # If Z=1 and user selected 3D, force back to 2D mode silently
        if z_insufficient and not is_2d:
            self.use_maximum_projection = True
            is_2d = True
        
        # Common styles
        base_style = """
            QPushButton {{
                font-size: 13px;
                font-weight: bold;
                padding: 8px 16px;
                border: 2px solid {border_color};
                {border_radius}
                background-color: {bg_color};
                color: {text_color};
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:disabled {{
                background-color: #1a1a1a;
                color: #444444;
                border-color: #333333;
            }}
        """
        
        # 3D button (left side, rounded left corners)
        if is_2d:
            # 3D is inactive
            style_3d = base_style.format(
                border_color="#555555",
                border_radius="border-top-left-radius: 6px; border-bottom-left-radius: 6px;",
                bg_color="#2a2a2a",
                text_color="#888888",
                hover_color="#3a3a3a"
            )
        else:
            # 3D is active
            style_3d = base_style.format(
                border_color="#9b59b6",
                border_radius="border-top-left-radius: 6px; border-bottom-left-radius: 6px;",
                bg_color="#9b59b6",
                text_color="#ffffff",
                hover_color="#a569bd"
            )
        
        # 2D button (right side, rounded right corners)
        if is_2d:
            # 2D is active
            style_2d = base_style.format(
                border_color="#00d4aa",
                border_radius="border-top-right-radius: 6px; border-bottom-right-radius: 6px;",
                bg_color="#00d4aa",
                text_color="#000000",
                hover_color="#00e5b8"
            )
        else:
            # 2D is inactive
            style_2d = base_style.format(
                border_color="#555555",
                border_radius="border-top-right-radius: 6px; border-bottom-right-radius: 6px;",
                bg_color="#2a2a2a",
                text_color="#888888",
                hover_color="#3a3a3a"
            )
        
        if hasattr(self, 'btn_mode_3d'):
            self.btn_mode_3d.setStyleSheet(style_3d)
            self.btn_mode_3d.setChecked(not is_2d)
            # Disable 3D button if Z=1 (3D tracking impossible)
            self.btn_mode_3d.setEnabled(not z_insufficient)
        if hasattr(self, 'btn_mode_2d'):
            self.btn_mode_2d.setStyleSheet(style_2d)
            self.btn_mode_2d.setChecked(is_2d)

    def _update_tracking_mode_status(self):
        """Update the status indicator showing current tracking mode details."""
        is_2d = self.use_maximum_projection
        
        # Check if 3D mode is even possible (need Z > 1)
        z_dim = self.image_stack.shape[1] if self.image_stack is not None else 1
        z_insufficient = (z_dim <= 1)
        
        if z_insufficient:
            # Single Z-plane: 3D is not available
            status_html = """
                <div style='text-align: center; padding: 6px;'>
                    <span style='color: #f39c12; font-size: 14px; font-weight: bold;'>
                        ⚠ 2D MODE (Single Z-plane)
                    </span><br/>
                    <span style='color: #aaaaaa; font-size: 11px;'>
                        Image has only 1 Z-plane; 3D detection unavailable
                    </span>
                </div>
            """
            border_color = "#f39c12"
        elif is_2d:
            status_html = """
                <div style='text-align: center; padding: 6px;'>
                    <span style='color: #00d4aa; font-size: 14px; font-weight: bold;'>
                        ✓ 2D PROJECTION MODE
                    </span><br/>
                    <span style='color: #aaaaaa; font-size: 11px;'>
                        Fast detection using maximum Z-projection (Trackpy)
                    </span>
                </div>
            """
            border_color = "#00d4aa"
        else:
            status_html = """
                <div style='text-align: center; padding: 6px;'>
                    <span style='color: #9b59b6; font-size: 14px; font-weight: bold;'>
                        ✓ 3D VOLUME MODE
                    </span><br/>
                    <span style='color: #aaaaaa; font-size: 11px;'>
                        Full 3D detection across all Z-planes (Big-FISH)
                    </span>
                </div>
            """
            border_color = "#9b59b6"
        
        if hasattr(self, 'tracking_mode_status'):
            self.tracking_mode_status.setText(status_html)
            self.tracking_mode_status.setStyleSheet(f"""
                QLabel {{
                    background-color: rgba(40, 40, 40, 0.8);
                    border: 2px solid {border_color};
                    border-radius: 6px;
                    padding: 4px;
                }}
            """)

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
            list_voxels=self._get_validated_voxels(),
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
            step_size_in_sec=float(self.time_interval_value) if self.time_interval_value is not None else 1.0,
            fast_gaussian_fit=self.fast_gaussian_fit,
        ).run()
        progress.close()
        # Store detection results - detection now properly stores data for export
        if list_dataframes_trajectories:
            df_detected = pd.concat(list_dataframes_trajectories, ignore_index=True)
            df_detected['spot_type'] = self.current_channel  # Set spot_type to actual channel number
            
            # Check if there's existing tracking data with linking (not detection-only)
            # If so, warn user that detection will clear tracking data
            has_tracking_data = False
            for ch, params in self.tracking_parameters_per_channel.items():
                if ch != self.current_channel and not params.get('detection_only', False):
                    has_tracking_data = True
                    break
            
            if has_tracking_data:
                reply = QMessageBox.question(
                    self, "Mode Mismatch",
                    "You have tracking data (with linking) for other channels.\n\n"
                    "Detection (without linking) cannot be mixed with tracking data.\n\n"
                    "Clear all channel data and start fresh with detection-only?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
                # Clear all existing data
                self._clear_all_tracking_data()
            
            # Store in multi-channel tracking data (replaces any previous for this channel)
            self.multi_channel_tracking_data[self.current_channel] = df_detected.copy()
            
            # Add to tracked channels if not already present
            if self.current_channel not in self.tracked_channels:
                self.tracked_channels.append(self.current_channel)
            
            # Set primary channel if first detection/tracking
            if self.primary_tracking_channel is None:
                self.primary_tracking_channel = self.current_channel
            
            # Store threshold and parameters used for this channel
            self.tracking_thresholds[self.current_channel] = self.user_selected_threshold
            self.tracking_parameters_per_channel[self.current_channel] = {
                'threshold': self.user_selected_threshold,
                'min_length_trajectory': self.min_length_trajectory,
                'yx_spot_size_in_px': self.yx_spot_size_in_px,
                'z_spot_size_in_px': self.z_spot_size_in_px,
                'memory': self.memory,
                'maximum_range_search_pixels': self.maximum_range_search_pixels,
                'use_maximum_projection': self.use_maximum_projection,
                'cluster_radius_nm': self.cluster_radius_nm,
                'maximum_spots_cluster': self.maximum_spots_cluster,
                'detection_only': True,  # Flag to indicate detection without linking
            }
            
            # Rebuild combined df_tracking from all channels
            self._rebuild_combined_tracking_dataframe()
            
            # Update tracked channels list widget
            self._update_tracked_channels_list()
            
            # Also set as preview for visualization
            self.detected_spots_frame = df_detected
        else:
            QMessageBox.information(self, "No Spots Detected", "No spots were detected in any frame.")
        # Optional random-mode run
        if getattr(self, 'random_mode_enabled', True):
            random_tracking = mi.ParticleTracking(
                image=image_to_use,
                channels_spots=[self.current_channel],
                masks=masks_complete,
                masks_nuclei=masks_nuc,
                masks_cytosol_no_nuclei=masks_cyto_no_nuc,
                list_voxels=self._get_validated_voxels(),
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
                step_size_in_sec=float(self.time_interval_value) if self.time_interval_value is not None else 1.0,
                fast_gaussian_fit=self.fast_gaussian_fit,
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
    
    def _select_all_particles(self):
        """Select all particles in the list."""
        if hasattr(self, 'tracked_particles_list'):
            self.tracked_particles_list.selectAll()
    
    def _clear_particle_selection(self):
        """Clear particle selection."""
        if hasattr(self, 'tracked_particles_list'):
            self.tracked_particles_list.clearSelection()
    
    def _update_particle_list_filtered(self):
        """Update particle list based on cell filter."""
        if self.df_tracking.empty:
            return
        
        selected_cell = self.vis_cell_filter_combo.currentData()
        particle_col = 'unique_particle' if 'unique_particle' in self.df_tracking.columns else 'particle'
        
        # Filter by cell_id if specified
        if selected_cell >= 0 and 'cell_id' in self.df_tracking.columns:
            df_filtered = self.df_tracking[self.df_tracking['cell_id'] == selected_cell]
        else:
            df_filtered = self.df_tracking
        
        # Get unique particles
        particles = df_filtered[particle_col].unique()
        
        # Update list
        self.tracked_particles_list.blockSignals(True)
        self.tracked_particles_list.clear()
        for pid in sorted(particles):
            item = QListWidgetItem(str(pid))
            item.setData(Qt.UserRole, pid)
            self.tracked_particles_list.addItem(item)
        self.tracked_particles_list.blockSignals(False)
        
        self.display_tracking_visualization()
    
    def _update_playback_speed(self):
        """Update playback timer interval based on speed selection."""
        if hasattr(self, 'timer_tracking_vis'):
            speed = self.playback_speed_combo.currentData()
            base_interval = 100  # Base interval in ms (10 fps)
            new_interval = max(25, int(base_interval / speed))
            self.timer_tracking_vis.setInterval(new_interval)
    
    def _populate_vis_cell_filter(self):
        """Populate the cell filter dropdown for visualization tab."""
        if not hasattr(self, 'vis_cell_filter_combo'):
            return
        
        self.vis_cell_filter_combo.blockSignals(True)
        self.vis_cell_filter_combo.clear()
        self.vis_cell_filter_combo.addItem("All Cells", -1)
        
        if not self.df_tracking.empty and 'cell_id' in self.df_tracking.columns:
            cell_ids = sorted(self.df_tracking['cell_id'].dropna().unique())
            for cid in cell_ids:
                n_particles = self.df_tracking[self.df_tracking['cell_id'] == cid]['particle'].nunique()
                self.vis_cell_filter_combo.addItem(f"Cell {int(cid)} ({n_particles})", int(cid))
        
        self.vis_cell_filter_combo.blockSignals(False)

    def _draw_trajectories_on_axes(self, ax, current_frame):
        """Draw trajectory paths on the visualization axes.
        
        Args:
            ax: Matplotlib axes to draw on
            current_frame: Current frame index
        """
        if self.df_tracking.empty:
            return
        
        # Get selected particles from the list
        selected_items = self.tracked_particles_list.selectedItems()
        if not selected_items:
            # If no selection, draw all particles
            particle_col = 'unique_particle' if 'unique_particle' in self.df_tracking.columns else 'particle'
            selected_pids = self.df_tracking[particle_col].unique()
        else:
            selected_pids = [item.data(Qt.UserRole) for item in selected_items]
        
        # Get tail length setting
        tail_length = self.trajectory_tail_spinbox.value() if hasattr(self, 'trajectory_tail_spinbox') else 10
        
        # Get color-by setting
        color_by = self.trajectory_color_combo.currentData() if hasattr(self, 'trajectory_color_combo') else 'particle'
        
        # Prepare color mapping
        particle_col = 'unique_particle' if 'unique_particle' in self.df_tracking.columns else 'particle'
        
        if color_by == 'cell_id' and 'cell_id' in self.df_tracking.columns:
            unique_vals = sorted(self.df_tracking['cell_id'].dropna().unique())
            color_key = 'cell_id'
        elif color_by == 'spot_type' and 'spot_type' in self.df_tracking.columns:
            unique_vals = sorted(self.df_tracking['spot_type'].dropna().unique())
            color_key = 'spot_type'
        else:
            unique_vals = sorted(self.df_tracking[particle_col].unique())
            color_key = particle_col
        
        # Color palette - bright colors for dark background
        color_palette = [
            '#00FFFF',  # Cyan
            '#FF00FF',  # Magenta
            '#00FF00',  # Lime
            '#FF8000',  # Orange
            '#FFFF00',  # Yellow
            '#FF0000',  # Red
            '#00BFFF',  # DeepSkyBlue
            '#FF69B4',  # HotPink
            '#7FFF00',  # Chartreuse
            '#FF6347',  # Tomato
            '#FFD700',  # Gold
            '#00FA9A',  # MediumSpringGreen
            '#9400D3',  # DarkViolet
            '#00CED1',  # DarkTurquoise
            '#FF1493',  # DeepPink
        ]
        
        # Create color map
        val_to_color = {}
        for i, val in enumerate(unique_vals):
            val_to_color[val] = color_palette[i % len(color_palette)]
        
        # Draw trajectories
        for pid in selected_pids:
            traj = self.df_tracking[self.df_tracking[particle_col] == pid].sort_values('frame')
            
            if traj.empty:
                continue
            
            # Filter to tail (frames up to current_frame)
            traj_visible = traj[traj['frame'] <= current_frame]
            if tail_length > 0:
                traj_visible = traj_visible.tail(tail_length)
            
            if len(traj_visible) < 2:
                continue
            
            # Get color based on color_by setting
            if color_key in traj_visible.columns:
                color_val = traj_visible.iloc[0][color_key]
            else:
                color_val = pid
            color = val_to_color.get(color_val, '#FFFFFF')
            
            # Draw trajectory line only (no marker at current position - the white crop square is sufficient)
            x_coords = traj_visible['x'].values
            y_coords = traj_visible['y'].values
            ax.plot(x_coords, y_coords, color=color, linewidth=1.5, alpha=0.8)
    
    def _update_trajectory_stats(self):
        """Update the trajectory statistics label with info about selected particle(s)."""
        if not hasattr(self, 'trajectory_stats_label'):
            return
        
        selected_items = self.tracked_particles_list.selectedItems()
        if not selected_items or self.df_tracking.empty:
            self.trajectory_stats_label.setText("Select a trajectory")
            return
        
        particle_col = 'unique_particle' if 'unique_particle' in self.df_tracking.columns else 'particle'
        
        # If single selection, show detailed stats
        if len(selected_items) == 1:
            pid = selected_items[0].data(Qt.UserRole)
            traj = self.df_tracking[self.df_tracking[particle_col] == pid].sort_values('frame')
            
            if traj.empty:
                self.trajectory_stats_label.setText("No data for trajectory")
                return
            
            # Basic stats
            n_frames = len(traj)
            first_frame = int(traj['frame'].min())
            last_frame = int(traj['frame'].max())
            
            # Displacement (start to end)
            x_start, y_start = traj.iloc[0]['x'], traj.iloc[0]['y']
            x_end, y_end = traj.iloc[-1]['x'], traj.iloc[-1]['y']
            displacement_px = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
            
            # Convert to µm if voxel size known
            voxel_yx = getattr(self, 'voxel_yx_nm', None)
            if voxel_yx:
                displacement_um = float(displacement_px) * float(voxel_yx) / 1000
                disp_str = f"{displacement_um:.2f} µm"
            else:
                disp_str = f"{displacement_px:.1f} px"
            
            # Total path length
            dx = np.diff(traj['x'].values)
            dy = np.diff(traj['y'].values)
            path_length_px = np.sum(np.sqrt(dx**2 + dy**2))
            
            if voxel_yx:
                path_um = float(path_length_px) * float(voxel_yx) / 1000
                path_str = f"{path_um:.2f} µm"
            else:
                path_str = f"{path_length_px:.1f} px"
            
            # Mean speed
            time_interval = getattr(self, 'time_interval_value', None)
            if time_interval and n_frames > 1:
                # Convert to float to handle Decimal types from metadata
                time_interval_f = float(time_interval)
                total_time = (n_frames - 1) * time_interval_f
                voxel_yx = getattr(self, 'voxel_yx_nm', None)
                if voxel_yx:
                    voxel_yx_f = float(voxel_yx)
                    speed = (float(path_length_px) * voxel_yx_f / 1000) / total_time
                    speed_str = f"{speed:.3f} µm/s"
                else:
                    speed = float(path_length_px) / total_time
                    speed_str = f"{speed:.2f} px/s"
            else:
                speed_str = "N/A"
            
            # Cell ID if available
            cell_str = ""
            if 'cell_id' in traj.columns:
                cell_id = traj.iloc[0]['cell_id']
                cell_str = f"\nCell ID: {int(cell_id)}"
            
            stats_text = (
                f"Particle: {pid}\n"
                f"Frames: {n_frames} ({first_frame}-{last_frame})\n"
                f"Displacement: {disp_str}\n"
                f"Path Length: {path_str}\n"
                f"Mean Speed: {speed_str}"
                f"{cell_str}"
            )
        else:
            # Multiple selection - show summary
            n_selected = len(selected_items)
            total_frames = 0
            for item in selected_items:
                pid = item.data(Qt.UserRole)
                traj = self.df_tracking[self.df_tracking[particle_col] == pid]
                total_frames += len(traj)
            
            avg_frames = total_frames / n_selected if n_selected > 0 else 0
            stats_text = (
                f"Selected: {n_selected} trajectories\n"
                f"Total points: {total_frames}\n"
                f"Avg length: {avg_frames:.1f} frames"
            )
        
        self.trajectory_stats_label.setText(stats_text)
    
    def _add_trajectory_legend(self, ax):
        """Add a compact color legend for trajectory colors in the upper-left corner."""
        if self.df_tracking.empty:
            return
        
        # Get the color-by setting
        color_by = self.trajectory_color_combo.currentData() if hasattr(self, 'trajectory_color_combo') else 'particle'
        particle_col = 'unique_particle' if 'unique_particle' in self.df_tracking.columns else 'particle'
        
        # Determine unique values and labels
        if color_by == 'cell_id' and 'cell_id' in self.df_tracking.columns:
            unique_vals = sorted(self.df_tracking['cell_id'].dropna().unique())
            label_prefix = "Cell "
        elif color_by == 'spot_type' and 'spot_type' in self.df_tracking.columns:
            unique_vals = sorted(self.df_tracking['spot_type'].dropna().unique())
            label_prefix = "Ch "
        else:
            # Don't show legend for particle ID (too many entries)
            return
        
        if len(unique_vals) < 2 or len(unique_vals) > 10:
            # Skip legend if only 1 category or too many
            return
        
        # Color palette
        color_palette = [
            '#00FFFF', '#FF00FF', '#00FF00', '#FF8000', '#FFFF00',
            '#FF0000', '#00BFFF', '#FF69B4', '#7FFF00', '#FF6347'
        ]
        
        # Build legend handles
        legend_handles = []
        for i, val in enumerate(unique_vals):
            color = color_palette[i % len(color_palette)]
            line = plt.Line2D([0], [0], color=color, linewidth=2, label=f"{label_prefix}{int(val)}")
            legend_handles.append(line)
        
        # Add legend to axes
        leg = ax.legend(
            handles=legend_handles,
            loc='upper left',
            fontsize=7,
            framealpha=0.7,
            facecolor='black',
            edgecolor='white',
            labelcolor='white',
            handlelength=1,
            borderpad=0.3,
            labelspacing=0.2
        )
        leg.set_zorder(100)

    def _on_tracking_zoom_select(self, eclick, erelease):
        """Handle ROI selection from right-click drag on tracking canvas."""
        if self.image_stack is None:
            return
        
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Handle None values (click outside axes)
        if x1 is None or x2 is None or y1 is None or y2 is None:
            return
        
        # Calculate ROI bounds
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Enforce minimum ROI size (50x50 pixels)
        if (x_max - x_min) < 50 or (y_max - y_min) < 50:
            return
        
        # Clamp to image bounds
        _, _, H, W, _ = self.image_stack.shape
        x_min = max(0, x_min)
        x_max = min(W, x_max)
        y_min = max(0, y_min)
        y_max = min(H, y_max)
        
        # Store ROI
        self.tracking_zoom_roi = (x_min, x_max, y_min, y_max)
        
        # Update label
        if hasattr(self, 'tracking_zoom_label'):
            self.tracking_zoom_label.setText(f"🔍 ROI: X[{int(x_min)}:{int(x_max)}] Y[{int(y_min)}:{int(y_max)}]")
            self.tracking_zoom_label.setStyleSheet("color: #00d4aa; font-size: 10px; font-weight: bold;")
        
        # Redraw with zoom
        self.plot_tracking()

    def _on_tracking_canvas_click(self, event):
        """Handle mouse clicks on tracking canvas - double-click to reset zoom."""
        if event.dblclick:
            self._reset_tracking_zoom()

    def _reset_tracking_zoom(self):
        """Reset zoom to show full image."""
        self.tracking_zoom_roi = None
        
        # Update label
        if hasattr(self, 'tracking_zoom_label'):
            self.tracking_zoom_label.setText("🔍 Full View")
            self.tracking_zoom_label.setStyleSheet("color: #888888; font-size: 10px;")
        
        # Redraw without zoom
        self.plot_tracking()

    def plot_tracking(self):
        # Check if ax_tracking is still valid (in the figure's axes list)
        ax_valid = (hasattr(self, 'ax_tracking') and 
                   self.ax_tracking is not None and 
                   self.ax_tracking in self.figure_tracking.axes)
        
        if ax_valid:
            # Clear axes content instead of entire figure to preserve RectangleSelector
            self.ax_tracking.clear()
        else:
            # Need to create new axes
            self.figure_tracking.clear()
            self.ax_tracking = self.figure_tracking.add_subplot(111)
            # Initialize RectangleSelector only when creating new axes
            self.tracking_zoom_selector = RectangleSelector(
                self.ax_tracking,
                self._on_tracking_zoom_select,
                useblit=True,
                button=[1],  # Left mouse button only
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=False,
                props=dict(facecolor='cyan', edgecolor='white', alpha=0.3, linewidth=2)
            )
        
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
        
        # Get Z-slice selection from slider
        _, Z, _, _, _ = image_to_use.shape  # shape is [T, Z, Y, X, C]
        z_val = self.z_slider_tracking.value() if hasattr(self, 'z_slider_tracking') else Z
        
        image_channel = image_to_use[self.current_frame, :, :, :, ch]
        
        # Display based on Z-slider: max projection (z_val == Z) or specific plane
        if z_val >= Z:
            # Max projection (default, slider at top)
            display_image = np.max(image_channel, axis=0)
        else:
            # Specific Z-plane
            display_image = image_channel[z_val]
        
        if self.tracking_remove_background_checkbox.isChecked():
            mask = (self.active_mask > 0).astype(int) if self.active_mask is not None else np.ones(self.image_stack.shape[2:4], dtype=int)
            display_image = display_image * mask
        
        min_p = self.min_percentile_spinbox_tracking.value() if hasattr(self, 'min_percentile_spinbox_tracking') else self.tracking_min_percentile
        max_p = self.max_percentile_spinbox_tracking.value() if hasattr(self, 'max_percentile_spinbox_tracking') else 99.95
        
        # If zoomed, calculate percentiles based on ROI only
        if self.tracking_zoom_roi is not None:
            x_min, x_max, y_min, y_max = self.tracking_zoom_roi
            roi_data = display_image[int(y_min):int(y_max), int(x_min):int(x_max)]
            if roi_data.size > 0:
                # Calculate actual min/max values from ROI for rescaling
                roi_min = np.percentile(roi_data[roi_data > 0], min_p) if np.any(roi_data > 0) else 0
                roi_max = np.percentile(roi_data[roi_data > 0], max_p) if np.any(roi_data > 0) else 1
                # Manual rescaling based on ROI percentiles
                rescaled_image = np.clip((display_image - roi_min) / (roi_max - roi_min + 1e-10) * 255, 0, 255).astype(np.uint8)
                rescaled_image = rescaled_image[..., np.newaxis]  # Add channel dim for consistency
            else:
                rescaled_image = mi.Utilities().convert_to_int8(
                    display_image,
                    rescale=True,
                    min_percentile=min_p,
                    max_percentile=max_p
                )
        else:
            rescaled_image = mi.Utilities().convert_to_int8(
                display_image,
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
        
        # Apply zoom immediately after imshow to prevent other elements from resetting limits
        zoom_scale = 1.0  # Default: no zoom adjustment
        if self.tracking_zoom_roi is not None:
            x_min, x_max, y_min, y_max = self.tracking_zoom_roi
            self.ax_tracking.set_xlim(x_min, x_max)
            self.ax_tracking.set_ylim(y_max, y_min)  # Inverted for image coordinates
            # Calculate zoom factor: ratio of visible area to full image
            # Smaller visible area = more zoomed in = LARGER markers (inverted from before)
            full_width = normalized_image.shape[1]
            visible_width = x_max - x_min
            if full_width > 0 and visible_width > 0:
                zoom_ratio = visible_width / full_width
                # Invert the scale: full view (zoom_ratio=1) gets smaller markers
                # Zoomed in (zoom_ratio=0.1) gets normal/larger markers
                # At full view: zoom_scale ~ 0.4; When zoomed to 10%: zoom_scale ~ 1.0
                zoom_scale = max(0.4, min(1.0, 1.0 - (zoom_ratio ** 0.5) * 0.6))
        else:
            # Full view (no zoom): use smaller markers based on image size
            # For smaller images like 928x624, markers should be proportionally smaller
            full_width = normalized_image.shape[1]
            # Scale down for smaller images: 1024px = 0.4, 2048px = 0.7, 4096px+ = 1.0
            image_scale = min(1.0, max(0.3, (full_width / 2048.0) ** 0.5))
            zoom_scale = image_scale
        
        dpi = self.figure_tracking.get_dpi()
        marker_scale = dpi / 100.0
        # Apply zoom-aware line width (thinner at full view, thicker when zoomed)
        spot_linewidth = max(0.3, 1.0 * zoom_scale)
        
        # Get tracking data for current frame
        # Priority: 1) Multi-channel tracking data for this frame (filtered by channel if needed)
        #           2) Detection preview (detected_spots_frame) for current frame
        df_frame = pd.DataFrame()
        
        # Check if we should show all channels or just current channel
        show_all_channels = (hasattr(self, 'tracking_show_all_channels_checkbox') and 
                            self.tracking_show_all_channels_checkbox.isChecked())
        
        # First, get tracked data for current frame
        if len(self.df_tracking) > 0:
            df_frame = self.df_tracking[self.df_tracking['frame'] == self.current_frame].copy()
            
            # Filter by current channel if "All Ch" is not checked
            if not show_all_channels and 'spot_type' in df_frame.columns and len(df_frame) > 0:
                # Only show spots for the currently selected channel
                # Use explicit int casting to ensure proper comparison
                df_frame = df_frame[df_frame['spot_type'].astype(int) == int(self.current_channel)]
        
        # If we have a detection preview (detected_spots_frame), include it
        # This allows showing detection results while preserving multi-channel tracking
        # IMPORTANT: Skip preview logic when "All Channels" is checked and we have tracking data,
        # since the preview is only for the current channel and would interfere with multi-channel display
        use_preview = (
            hasattr(self, 'detected_spots_frame') and 
            self.detected_spots_frame is not None and 
            len(self.detected_spots_frame) > 0 and
            not (show_all_channels and not df_frame.empty)  # Skip preview if showing all channels with existing data
        )
        
        if use_preview:
            # Handle both single frame and multi-frame detection preview
            if 'frame' in self.detected_spots_frame.columns:
                preview_frame = self.detected_spots_frame[
                    self.detected_spots_frame['frame'] == self.current_frame
                ]
            else:
                # Single frame detection without frame column
                preview_frame = self.detected_spots_frame
            
            # Only include preview if it's for the current channel (filter to get only current ch)
            if 'spot_type' in preview_frame.columns and len(preview_frame) > 0:
                preview_frame = preview_frame[preview_frame['spot_type'] == self.current_channel]
            
            if len(preview_frame) > 0:
                # If df_frame is empty, use preview directly
                if df_frame.empty:
                    df_frame = preview_frame.copy()
                else:
                    # Preview takes precedence for its channel (replaces tracked data for that channel)
                    preview_spot_type = preview_frame['spot_type'].iloc[0] if 'spot_type' in preview_frame.columns else None
                    if preview_spot_type is not None:
                        # Only remove/replace data for the preview's channel, preserving other channels
                        df_frame = df_frame[df_frame['spot_type'] != preview_spot_type]
                        # Add the preview spots
                        df_frame = pd.concat([df_frame, preview_frame], ignore_index=True)
        
        
        # Filter spots by Z-plane if viewing a specific Z-slice (for 3D tracking)
        if z_val < Z and 'z' in df_frame.columns and len(df_frame) > 0:
            # Filter spots to those within ±0.5 of the selected Z-plane
            z_tolerance = 0.5
            df_frame = df_frame[(df_frame['z'] >= z_val - z_tolerance) & 
                               (df_frame['z'] <= z_val + z_tolerance)]
        if not df_frame.empty:
            # Define channel colors for multi-channel display
            # ImageJ-style colors: green, magenta, yellow, red, cyan
            channel_colors = ['#00ff00', '#ff00ff', '#ffff00', '#ff0000', '#00ffff', '#ff8800', '#8800ff']
            
            # Check if we have multi-channel tracking data
            has_multi_channel = (
                'spot_type' in df_frame.columns and 
                len(self.tracked_channels) > 1
            )
            
            legend_handles = []
            legend_labels = []
            
            if has_multi_channel:
                # Plot per-channel with different colors
                for spot_type in sorted(df_frame['spot_type'].unique()):
                    df_ch = df_frame[df_frame['spot_type'] == spot_type]
                    edge_color = channel_colors[int(spot_type) % len(channel_colors)]
                    
                    # Handle missing cluster_size column (treat all as single spots)
                    if 'cluster_size' in df_ch.columns:
                        single_spots = df_ch[df_ch['cluster_size'] <= 1]
                        cluster_spots = df_ch[df_ch['cluster_size'] > 1]
                    else:
                        single_spots = df_ch
                        cluster_spots = pd.DataFrame()
                    
                    if not single_spots.empty:
                        self.ax_tracking.scatter(
                            single_spots['x'], single_spots['y'],
                            s=self.yx_spot_size_in_px * 6 * marker_scale * SCALE_SPOTS * zoom_scale,
                            marker='o', linewidth=spot_linewidth,
                            edgecolors=edge_color, facecolors='none'
                        )
                    
                    if not cluster_spots.empty:
                        self.ax_tracking.scatter(
                            cluster_spots['x'], cluster_spots['y'],
                            s=self.yx_spot_size_in_px * 6 * marker_scale * SCALE_SPOTS * zoom_scale,
                            marker='s', linewidth=spot_linewidth,
                            edgecolors=edge_color, facecolors='none'
                        )
                    
                    # Add legend entry for this channel
                    total_count = len(df_ch)
                    ch_legend = self.ax_tracking.scatter([], [],
                                                         s=self.yx_spot_size_in_px * 5 * marker_scale,
                                                         marker='o', linewidth=spot_linewidth,
                                                         edgecolors=edge_color, facecolors='none')
                    legend_handles.append(ch_legend)
                    legend_labels.append(f"Ch {int(spot_type)}: {total_count}")
            else:
                # Single channel: use white edge color (original behavior)
                edge_color = "w"
                
                # Handle missing cluster_size column (treat all as single spots)
                if 'cluster_size' in df_frame.columns:
                    single_spots = df_frame[df_frame['cluster_size'] <= 1]
                    cluster_spots = df_frame[df_frame['cluster_size'] > 1]
                else:
                    single_spots = df_frame
                    cluster_spots = pd.DataFrame()
                
                if not single_spots.empty:
                    self.ax_tracking.scatter(
                        single_spots['x'], single_spots['y'],
                        s=self.yx_spot_size_in_px * 6 * marker_scale * SCALE_SPOTS * zoom_scale,
                        marker='o', linewidth=spot_linewidth,
                        edgecolors=edge_color, facecolors='none'
                    )
                    count_spots = single_spots.shape[0]
                    spot_legend = self.ax_tracking.scatter([], [],
                                                           s=self.yx_spot_size_in_px * 5 * marker_scale,
                                                           marker='o', linewidth=spot_linewidth,
                                                           edgecolors=edge_color, facecolors='none')
                    legend_handles.append(spot_legend)
                    legend_labels.append(f"Spots: {count_spots}")
                else:
                    self.ax_tracking.scatter(
                        [], [],
                        s=self.yx_spot_size_in_px * 6 * marker_scale * SCALE_SPOTS * zoom_scale,
                        marker='o', linewidth=spot_linewidth,
                        edgecolors=edge_color, facecolors='none'
                    )
                    legend_labels.append(f"Spots: 0")
                    legend_handles.append(self.ax_tracking.scatter([], [],
                                                                   s=self.yx_spot_size_in_px * 5 * marker_scale,
                                                                   marker='o', linewidth=spot_linewidth,
                                                                   edgecolors=edge_color, facecolors='none'))
                if not cluster_spots.empty:
                    self.ax_tracking.scatter(
                        cluster_spots['x'], cluster_spots['y'],
                        s=self.yx_spot_size_in_px * 6 * marker_scale * SCALE_SPOTS * zoom_scale,
                        marker='s', linewidth=spot_linewidth,
                        edgecolors=edge_color, facecolors='none'
                    )
                    count_clusters = cluster_spots.shape[0]
                    cluster_legend = self.ax_tracking.scatter([], [],
                                                              s=self.yx_spot_size_in_px * 5 * marker_scale * SCALE_SPOTS,
                                                              marker='s', linewidth=spot_linewidth,
                                                              edgecolors=edge_color, facecolors='none')
                    legend_handles.append(cluster_legend)
                    legend_labels.append(f"Clusters: {count_clusters}")
            if self.show_cluster_size_checkbox.isChecked() and 'cluster_size' in df_frame.columns:
                # Filter to ROI if zoomed
                df_to_label = df_frame
                if self.tracking_zoom_roi is not None:
                    x_min, x_max, y_min, y_max = self.tracking_zoom_roi
                    df_to_label = df_frame[(df_frame['x'] >= x_min) & (df_frame['x'] <= x_max) &
                                          (df_frame['y'] >= y_min) & (df_frame['y'] <= y_max)]
                for _, row in df_to_label.iterrows():
                    # Use channel-specific color when showing all channels
                    if show_all_channels and 'spot_type' in row.index:
                        label_color = channel_colors[int(row['spot_type']) % len(channel_colors)]
                    else:
                        label_color = 'white'
                    # Position: bottom-right of spot to avoid overlap with particle ID
                    self.ax_tracking.text(row['x'] + 4, row['y'] + 4,
                                           f"{int(row['cluster_size'])}",
                                           color=label_color, fontsize=7,
                                           ha='left', va='top')
            if self.show_particle_id_checkbox.isChecked() and 'particle' in df_frame.columns:
                # Filter to ROI if zoomed
                df_to_label = df_frame
                if self.tracking_zoom_roi is not None:
                    x_min, x_max, y_min, y_max = self.tracking_zoom_roi
                    df_to_label = df_frame[(df_frame['x'] >= x_min) & (df_frame['x'] <= x_max) &
                                          (df_frame['y'] >= y_min) & (df_frame['y'] <= y_max)]
                for _, row in df_to_label.iterrows():
                    # Use channel-specific color when showing all channels
                    if show_all_channels and 'spot_type' in row.index:
                        label_color = channel_colors[int(row['spot_type']) % len(channel_colors)]
                    else:
                        label_color = 'white'
                    # Position: top-left of spot to avoid overlap with cluster size
                    self.ax_tracking.text(row['x'] - 4, row['y'] - 4,
                                           f"{int(row['particle'])}",
                                           color=label_color, fontsize=6,
                                           ha='right', va='bottom')
            if self.show_trajectories_checkbox.isChecked() and not self.df_tracking.empty:
                # Use unique_particle to avoid cross-cell trajectory connections
                particle_col = 'unique_particle' if 'unique_particle' in self.df_tracking.columns else 'particle'
                if particle_col in self.df_tracking.columns:
                    df_up_to_current = self.df_tracking[self.df_tracking['frame'] <= self.current_frame]
                    
                    # Filter by channel if not showing all channels
                    if not show_all_channels and 'spot_type' in df_up_to_current.columns:
                        df_up_to_current = df_up_to_current[df_up_to_current['spot_type'] == self.current_channel]
                    
                    # Filter trajectories to those with at least one point in ROI if zoomed
                    if self.tracking_zoom_roi is not None:
                        x_min, x_max, y_min, y_max = self.tracking_zoom_roi
                        particles_in_roi = df_up_to_current[
                            (df_up_to_current['x'] >= x_min) & (df_up_to_current['x'] <= x_max) &
                            (df_up_to_current['y'] >= y_min) & (df_up_to_current['y'] <= y_max)
                        ][particle_col].unique()
                        df_up_to_current = df_up_to_current[df_up_to_current[particle_col].isin(particles_in_roi)]
                    
                    for particle_id, grp in df_up_to_current.groupby(particle_col):
                        if grp.shape[0] > 1:
                            grp = grp.sort_values('frame')
                            # Use channel-specific color when showing all channels
                            if show_all_channels and 'spot_type' in grp.columns:
                                traj_color = channel_colors[int(grp['spot_type'].iloc[0]) % len(channel_colors)]
                            else:
                                traj_color = 'white'
                            self.ax_tracking.plot(grp['x'], grp['y'], '-', linewidth=1, color=traj_color, alpha=0.7)
            # Only show legend when viewing full image (not zoomed)
            # because counts are for entire image, not just the zoomed region
            if legend_handles and self.tracking_zoom_roi is None:
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
                        # Optimized mask visualization
                    if labeled_mask.max() > 0:
                        # Draw boundaries efficiently using FIND_BOUNDARIES (vectorized)
                        # mode='inner' draws boundaries inside the object
                        boundaries = find_boundaries(labeled_mask, mode='inner', background=0)
                        boundaries = np.ma.masked_where(boundaries == 0, boundaries)
                        
                        # Convert named color to RGB for Colormap
                        color_rgb = mcolors.to_rgb(color)
                        cmap_boundary = mcolors.ListedColormap([color_rgb])
                        
                        self.ax_tracking.imshow(boundaries, cmap=cmap_boundary, alpha=0.7, interpolation='none')

                        # Optimized label placement using CENTER_OF_MASS
                        unique_labels = np.unique(labeled_mask)
                        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
                        
                        # Limit text labels for performance if too many cells
                        if len(unique_labels) > 0 and len(unique_labels) < 500:
                            # Use labeled_mask for geometric center calculation
                            centers = center_of_mass(labeled_mask, labels=labeled_mask, index=unique_labels)
                            # center_of_mass returns a list of tuples [(y, x), ...] when index is an array
                            if not isinstance(centers, list): 
                                centers = [centers]
                                
                            for label_id, center in zip(unique_labels, centers):
                                if center is None or np.isnan(center).any():
                                    continue
                                cy, cx = center
                                self.ax_tracking.text(cx, cy, str(int(label_id)),
                                                    color=color, fontsize=6, ha='center', va='center',
                                                    fontweight='bold', alpha=0.9)
        if self.tracking_time_text_checkbox.isChecked():
            current_time = self.current_frame * (float(self.time_interval_value) if self.time_interval_value else 1)
            time_str = self._format_time_interval(current_time)
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
        
        # Add thin border to show image boundaries
        if self.image_stack is not None:
            H, W = self.image_stack.shape[2], self.image_stack.shape[3]
            img_border = patches.Rectangle((0, 0), W-1, H-1, linewidth=0.8, 
                                            edgecolor='#555555', facecolor='none', linestyle='-')
            self.ax_tracking.add_patch(img_border)
        
        # Use tight_layout for proper spacing
        try:
            self.figure_tracking.tight_layout()
        except Exception:
            pass  # Ignore layout errors
        
        # Apply zoom AFTER tight_layout to ensure limits are not reset
        if self.tracking_zoom_roi is not None:
            x_min, x_max, y_min, y_max = self.tracking_zoom_roi
            self.ax_tracking.set_xlim(x_min, x_max)
            self.ax_tracking.set_ylim(y_max, y_min)  # Inverted for image coordinates
        
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
                image_counter=getattr(self, 'current_frame', 0),
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
        list_voxels = self._get_validated_voxels()
        threshold = self.user_selected_threshold if hasattr(self, 'user_selected_threshold') and self.user_selected_threshold is not None else np.percentile(image_channel, 99)
        
        # Get masks for tracking (supports both Cellpose and Segmentation)
        masks_complete, masks_nuc, masks_cyto_no_nuc = self._get_tracking_masks()
        
        # For single-frame detection, slice TYX masks to current frame's YX mask
        if masks_complete is not None and masks_complete.ndim == 3:
            masks_complete = masks_complete[self.current_frame]
        if masks_nuc is not None and masks_nuc.ndim == 3:
            masks_nuc = masks_nuc[self.current_frame]
        if masks_cyto_no_nuc is not None and masks_cyto_no_nuc.ndim == 3:
            masks_cyto_no_nuc = masks_cyto_no_nuc[self.current_frame]
            
            
        if masks_complete is None:
            masks_complete = np.ones(self.image_stack.shape[2:4], dtype=int)
        spots = self.detect_spots(image_channel, threshold, list_voxels, masks_complete, masks_nuc, masks_cyto_no_nuc)
        if spots is not None and not spots.empty:
            spots['frame'] = self.current_frame
            spots['spot_type'] = self.current_channel  # Set spot_type to actual channel number
            self.detected_spots_frame = spots
            
            # For single frame detection, show as preview without replacing multi-channel tracked data
            # Only replace df_tracking if we don't have any multi-channel tracking data
            if not self.multi_channel_tracking_data:
                self.df_tracking = spots.copy()
            # else: Keep the existing multi-channel tracking data, detection is just a preview
        else:
            self.detected_spots_frame = None
            if not self.multi_channel_tracking_data:
                self.df_tracking = pd.DataFrame()
        self.plot_tracking()


    def perform_particle_tracking(self):
        if self.image_stack is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return
        if not hasattr(self, 'user_selected_threshold') or self.user_selected_threshold is None or self.user_selected_threshold <= 0:
            QMessageBox.warning(self, "Tracking Aborted", "Threshold is zero; please adjust the threshold slider before running tracking.")
            return
        
        # Clear detection preview (not the multi-channel tracking data)
        self.detected_spots_frame = None
        self.reset_msd_tab()
        self.plot_tracking()
        # Get masks for tracking (supports both Cellpose and Segmentation)
        masks_complete, masks_nuc, masks_cyto_no_nuc = self._get_tracking_masks()
        
        if masks_complete is None:
            masks_complete = np.ones(self.image_stack.shape[2:4], dtype=int)
        image_to_use = self.get_current_image_source()
        if self.use_maximum_projection:
            image_to_use = np.max(image_to_use, axis=1, keepdims=True)
        list_voxels = self._get_validated_voxels()
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
                step_size_in_sec=float(self.time_interval_value) if self.time_interval_value is not None else 1.0,
                fast_gaussian_fit=self.fast_gaussian_fit,
            )
            random_df_list, _ = random_tracking.run()
            self.df_random_spots = random_df_list[0] if random_df_list else pd.DataFrame()

    def on_tracking_finished_with_progress(self, list_dataframes_trajectories, progress_dialog):
        self.on_tracking_finished(list_dataframes_trajectories)
        progress_dialog.close()

    def on_tracking_finished(self, list_dataframes_trajectories):
        try:
            tracking_channel = self.current_channel  # Channel that was tracked
            
            if list_dataframes_trajectories and any(not df.empty for df in list_dataframes_trajectories):
                df_tracking = pd.concat(list_dataframes_trajectories, ignore_index=True)
                if 'particle' not in df_tracking.columns or df_tracking['particle'].nunique() == 0:
                    raise ValueError("No particles detected or 'particle' column missing.")
                
                # Ensure spot_type is set to actual channel number
                df_tracking['spot_type'] = tracking_channel
                
                # Check if there's existing detection-only data for other channels
                # If so, warn user since we can't mix tracking with detection-only
                has_detection_only = False
                for ch, params in self.tracking_parameters_per_channel.items():
                    if ch != tracking_channel and params.get('detection_only', False):
                        has_detection_only = True
                        break
                
                if has_detection_only:
                    # Ask user before clearing detection-only data
                    reply = QMessageBox.question(
                        self, "Mode Mismatch",
                        "You have detection data (without linking) for other channels.\n\n"
                        "Tracking (with linking) cannot be mixed with detection-only data.\n\n"
                        "Clear all channel data and start fresh with tracking?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return
                    self._clear_all_tracking_data()
                
                # Store per-channel tracking data
                self.multi_channel_tracking_data[tracking_channel] = df_tracking.copy()
                if tracking_channel not in self.tracked_channels:
                    self.tracked_channels.append(tracking_channel)
                
                # Set primary channel if first tracking
                if self.primary_tracking_channel is None:
                    self.primary_tracking_channel = tracking_channel
                
                # Store threshold and parameters used for this channel
                self.tracking_thresholds[tracking_channel] = self.user_selected_threshold
                self.tracking_parameters_per_channel[tracking_channel] = {
                    'threshold': self.user_selected_threshold,
                    'min_length_trajectory': self.min_length_trajectory,
                    'yx_spot_size_in_px': self.yx_spot_size_in_px,
                    'z_spot_size_in_px': self.z_spot_size_in_px,
                    'memory': self.memory,
                    'maximum_range_search_pixels': self.maximum_range_search_pixels,
                    'use_maximum_projection': self.use_maximum_projection,
                    'cluster_radius_nm': self.cluster_radius_nm,
                    'maximum_spots_cluster': self.maximum_spots_cluster,
                }
                
                # Rebuild combined df_tracking from all channels
                self._rebuild_combined_tracking_dataframe()
                
                # Update tracked channels list widget if available
                self._update_tracked_channels_list()
            else:
                raise ValueError("No particles detected.")
            self.correlation_results = []
            self.current_total_plots = None
            self.detected_spots_frame = None
            self.plot_intensity_time_course()
            self.display_correlation_plot()
            self.channels_spots = [self.current_channel]
            self.populate_colocalization_channels()
            # Reset verification subtabs
            if hasattr(self, 'verify_visual_scroll_area'):
                self.verify_visual_scroll_area.setWidget(QWidget())
            if hasattr(self, 'verify_distance_scroll_area'):
                self.verify_distance_scroll_area.setWidget(QWidget())
            self.MIN_FRAMES_MSD = 20
            self.MIN_PARTICLES_MSD = 10

            if hasattr(self, 'compute_colocalization'):
                self.compute_colocalization()
            self.plot_tracking()
            if hasattr(self, 'channel_checkboxes') and self.channel_checkboxes:
                for idx, cb in enumerate(self.channel_checkboxes):
                    cb.setChecked(idx == 0)
            
            # MSD calculation is now performed only in the dedicated MSD tab
            # Clear any previously stored MSD values from tracking
            self.tracking_D_um2_s = None
            self.tracking_D_px2_s = None
            self.tracking_msd_mode = None

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

    def _rebuild_combined_tracking_dataframe(self):
        """Rebuild df_tracking from all per-channel tracking data.
        
        This method combines tracking data from all tracked channels into a single
        DataFrame, ensuring unique_particle IDs are unique across cells and channels.
        """
        if not self.multi_channel_tracking_data:
            self.df_tracking = pd.DataFrame()
            self.has_tracked = False
            return
        
        all_dfs = []
        for channel, df in self.multi_channel_tracking_data.items():
            if not df.empty:
                # Ensure spot_type is set correctly
                df_copy = df.copy()
                df_copy['spot_type'] = channel
                all_dfs.append(df_copy)
        
        if all_dfs:
            self.df_tracking = pd.concat(all_dfs, ignore_index=True)
            
            # Rebuild unique_particle to include spot_type for disambiguation
            if 'particle' in self.df_tracking.columns:
                if 'cell_id' in self.df_tracking.columns:
                    self.df_tracking['unique_particle'] = (
                        self.df_tracking['cell_id'].astype(str) + '_' + 
                        self.df_tracking['spot_type'].astype(str) + '_' +
                        self.df_tracking['particle'].astype(str)
                    )
                else:
                    self.df_tracking['unique_particle'] = (
                        self.df_tracking['spot_type'].astype(str) + '_' +
                        self.df_tracking['particle'].astype(str)
                    )
            
            self.df_tracking = self.df_tracking.reset_index(drop=True)
            self.has_tracked = True
        else:
            self.df_tracking = pd.DataFrame()
            self.has_tracked = False

    def _update_tracked_channels_list(self):
        """Update the tracked channels list widget in the UI.
        
        This method updates the list widget showing which channels have been tracked,
        along with their threshold and spot count information.
        """
        if not hasattr(self, 'tracked_channels_list'):
            return
        
        self.tracked_channels_list.clear()
        for ch in sorted(self.tracked_channels):
            if ch in self.multi_channel_tracking_data:
                df_ch = self.multi_channel_tracking_data[ch]
                spot_count = len(df_ch)
                threshold = self.tracking_thresholds.get(ch, 'N/A')
                
                # Format threshold to 2 decimal places
                if isinstance(threshold, (int, float)):
                    threshold_str = f"{threshold:.2f}"
                else:
                    threshold_str = str(threshold)
                
                # Check if this is detection-only or full tracking
                params = self.tracking_parameters_per_channel.get(ch, {})
                is_detection_only = params.get('detection_only', False)
                
                if is_detection_only:
                    # Detection-only: show spots only (no trajectories)
                    item_text = f"Ch {ch}: thr={threshold_str}, {spot_count} spots (detect)"
                else:
                    # Full tracking: show spots and trajectory count
                    traj_count = df_ch['particle'].nunique() if 'particle' in df_ch.columns else 0
                    item_text = f"Ch {ch}: thr={threshold_str}, {spot_count} spots, {traj_count} traj"
                
                self.tracked_channels_list.addItem(item_text)

    def clear_channel_tracking(self, channel=None):
        """Clear tracking data for a specific channel or the currently selected channel.
        
        Parameters
        ----------
        channel : int, optional
            The channel index to clear. If None, clears the currently selected
            channel from the tracked_channels_list widget.
        """
        if channel is None:
            if hasattr(self, 'tracked_channels_list'):
                current_item = self.tracked_channels_list.currentRow()
                if current_item >= 0 and current_item < len(self.tracked_channels):
                    channel = sorted(self.tracked_channels)[current_item]
        
        if channel is not None and channel in self.tracked_channels:
            # Remove from all data structures
            self.tracked_channels.remove(channel)
            if channel in self.multi_channel_tracking_data:
                del self.multi_channel_tracking_data[channel]
            if channel in self.tracking_thresholds:
                del self.tracking_thresholds[channel]
            if channel in self.tracking_parameters_per_channel:
                del self.tracking_parameters_per_channel[channel]
            
            # Update primary channel if needed
            if self.primary_tracking_channel == channel:
                self.primary_tracking_channel = self.tracked_channels[0] if self.tracked_channels else None
            
            # Rebuild combined dataframe
            self._rebuild_combined_tracking_dataframe()
            self._update_tracked_channels_list()
            self.plot_tracking()

    def clear_all_tracking(self):
        """Clear all tracking data for all channels."""
        self.multi_channel_tracking_data = {}
        self.tracked_channels = []
        self.tracking_thresholds = {}
        self.auto_threshold_per_channel = {}
        self.tracking_parameters_per_channel = {}
        self.primary_tracking_channel = None
        self.df_tracking = pd.DataFrame()
        self.has_tracked = False
        
        self._update_tracked_channels_list()
        self.plot_tracking()

    def _clear_all_tracking_data(self):
        """Internal method to clear all tracking data (alias for clear_all_tracking)."""
        self.multi_channel_tracking_data = {}
        self.tracked_channels = []
        self.tracking_thresholds = {}
        self.auto_threshold_per_channel = {}
        self.tracking_parameters_per_channel = {}
        self.primary_tracking_channel = None
        self.df_tracking = pd.DataFrame()
        self.has_tracked = False
        self._update_tracked_channels_list()

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
        
        # Set up zoom feature: RectangleSelector for left-click drag
        self.tracking_zoom_selector = RectangleSelector(
            self.ax_tracking,
            self._on_tracking_zoom_select,
            useblit=True,
            button=[1],  # Left mouse button only
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False,
            props=dict(facecolor='cyan', edgecolor='white', alpha=0.3, linewidth=2)
        )
        
        # Connect double-click to reset zoom
        self.canvas_tracking.mpl_connect('button_press_event', self._on_tracking_canvas_click)
        
        # Create horizontal layout to hold canvas + Z slider
        canvas_slider_layout_tracking = QHBoxLayout()
        canvas_slider_layout_tracking.addWidget(self.canvas_tracking)
        
        # Z-slider with label (vertical, on the right of canvas) - minimal width
        z_slider_container_tracking = QWidget()
        z_slider_container_tracking.setFixedWidth(40)
        z_slider_layout_tracking = QVBoxLayout(z_slider_container_tracking)
        z_slider_layout_tracking.setContentsMargins(2, 0, 2, 0)
        z_slider_layout_tracking.setSpacing(2)
        
        z_label_top_tracking = QLabel("Z")
        z_label_top_tracking.setAlignment(Qt.AlignCenter)
        z_label_top_tracking.setStyleSheet("color: white; font-weight: bold; font-size: 10px;")
        z_slider_layout_tracking.addWidget(z_label_top_tracking)
        
        # Initialize vertical Z-plane slider for tracking visualization
        self.z_slider_tracking = QSlider(Qt.Vertical, self)
        self.z_slider_tracking.setMinimum(0)
        self.z_slider_tracking.setMaximum(0)  # Will be set when image loads
        self.z_slider_tracking.setTickPosition(QSlider.NoTicks)
        self.z_slider_tracking.setInvertedAppearance(True)  # Top = highest Z index (max projection)
        self.z_slider_tracking.valueChanged.connect(self.update_z_tracking)
        z_slider_layout_tracking.addWidget(self.z_slider_tracking, stretch=1)
        
        self.z_label_tracking = QLabel("Max")
        self.z_label_tracking.setAlignment(Qt.AlignCenter)
        self.z_label_tracking.setStyleSheet("color: cyan; font-weight: bold; font-size: 9px;")
        z_slider_layout_tracking.addWidget(self.z_label_tracking)
        
        canvas_slider_layout_tracking.addWidget(z_slider_container_tracking)
        tracking_left_layout.addLayout(canvas_slider_layout_tracking)
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
        
        # Zoom ROI status label and instructions
        zoom_info_layout = QHBoxLayout()
        zoom_info_layout.setContentsMargins(0, 2, 0, 2)
        
        self.tracking_zoom_label = QLabel("🔍 Full View")
        self.tracking_zoom_label.setStyleSheet("color: #888888; font-size: 10px;")
        zoom_info_layout.addWidget(self.tracking_zoom_label)
        
        zoom_info_layout.addStretch()
        
        zoom_hint_label = QLabel("Click-drag to zoom, Double-click to reset")
        zoom_hint_label.setStyleSheet("color: #555555; font-size: 9px; font-style: italic;")
        zoom_info_layout.addWidget(zoom_hint_label)
        
        tracking_left_layout.addLayout(zoom_info_layout)
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
        
        self.frame_label_tracking = QLabel("0/0")
        self.frame_label_tracking.setMinimumWidth(50)
        controls_layout.addWidget(self.frame_label_tracking)
        
        self.play_button_tracking = QPushButton("Play", self)
        self.play_button_tracking.clicked.connect(self.play_pause_tracking)
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
        self.show_trajectories_checkbox.stateChanged.connect(self.plot_tracking)
        checkbox_layout.addWidget(self.show_trajectories_checkbox)
        # Add cluster size QCheckbox
        self.show_cluster_size_checkbox = QCheckBox("Cluster Size")
        self.show_cluster_size_checkbox.setChecked(False)
        self.show_cluster_size_checkbox.stateChanged.connect(self.plot_tracking)
        checkbox_layout.addWidget(self.show_cluster_size_checkbox)
        # Add particle ID QCheckbox
        self.show_particle_id_checkbox = QCheckBox("Particle ID")
        self.show_particle_id_checkbox.setChecked(False)
        self.show_particle_id_checkbox.stateChanged.connect(self.plot_tracking)
        checkbox_layout.addWidget(self.show_particle_id_checkbox)
        # Add "Display Time Stamp" checkbox (moved from right panel)
        self.tracking_time_text_checkbox = QCheckBox("Time Stamp")
        self.tracking_time_text_checkbox.setChecked(False)
        self.tracking_time_text_checkbox.stateChanged.connect(self.plot_tracking)
        checkbox_layout.addWidget(self.tracking_time_text_checkbox)
        # Add "Remove Background" checkbox (moved from right panel)
        self.tracking_remove_background_checkbox = QCheckBox("Remove Background")
        self.tracking_remove_background_checkbox.setChecked(False)
        self.tracking_remove_background_checkbox.stateChanged.connect(self.plot_tracking)
        checkbox_layout.addWidget(self.tracking_remove_background_checkbox)
        # Add "Show Masks" checkbox for visualizing mask contours and IDs
        self.tracking_show_masks_checkbox = QCheckBox("Masks")
        self.tracking_show_masks_checkbox.setChecked(True)  # Default to showing masks
        self.tracking_show_masks_checkbox.stateChanged.connect(self.plot_tracking)
        checkbox_layout.addWidget(self.tracking_show_masks_checkbox)
        # Add "All Channels" checkbox to show spots from all tracked channels (vs. current only)
        self.tracking_show_all_channels_checkbox = QCheckBox("All Channels")
        self.tracking_show_all_channels_checkbox.setChecked(False)  # Default: show only current channel spots
        self.tracking_show_all_channels_checkbox.setToolTip("Show spots from all tracked channels (uncheck to see only current channel)")
        self.tracking_show_all_channels_checkbox.stateChanged.connect(self.plot_tracking)
        checkbox_layout.addWidget(self.tracking_show_all_channels_checkbox)
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
        
        # ========== Enhanced 2D/3D Mode Toggle Section ==========
        mode_group = QGroupBox("Tracking Mode")
        mode_main_layout = QVBoxLayout(mode_group)
        mode_main_layout.setSpacing(8)
        
        # Segmented toggle button container
        toggle_container = QWidget()
        toggle_layout = QHBoxLayout(toggle_container)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(0)
        
        # Create 3D button (left side)
        self.btn_mode_3d = QPushButton("📦 3D Volume")
        self.btn_mode_3d.setCheckable(True)
        self.btn_mode_3d.setMinimumHeight(36)
        self.btn_mode_3d.setCursor(Qt.PointingHandCursor)
        
        # Create 2D button (right side)
        self.btn_mode_2d = QPushButton("🔲 2D Projection")
        self.btn_mode_2d.setCheckable(True)
        self.btn_mode_2d.setMinimumHeight(36)
        self.btn_mode_2d.setCursor(Qt.PointingHandCursor)
        
        toggle_layout.addWidget(self.btn_mode_3d)
        toggle_layout.addWidget(self.btn_mode_2d)
        
        # Style for the toggle buttons
        self._update_tracking_mode_buttons()
        
        # Connect buttons (exclusive selection)
        self.btn_mode_2d.clicked.connect(lambda: self._set_tracking_mode(is_2d=True))
        self.btn_mode_3d.clicked.connect(lambda: self._set_tracking_mode(is_2d=False))
        
        mode_main_layout.addWidget(toggle_container)
        
        # Keep legacy checkbox hidden but functional for compatibility
        self.use_2d_projection_checkbox = QCheckBox("Use 2D Projection for Tracking")
        self.use_2d_projection_checkbox.setChecked(self.use_maximum_projection)
        self.use_2d_projection_checkbox.stateChanged.connect(self.update_use_maximum_projection)
        self.use_2d_projection_checkbox.setVisible(False)  # Hidden, new UI handles this
        mode_main_layout.addWidget(self.use_2d_projection_checkbox)
        
        # Hidden legacy label (kept for compatibility)
        self.tracking_max_proj_status_label = QLabel()
        self.tracking_max_proj_status_label.setVisible(False)
        mode_main_layout.addWidget(self.tracking_max_proj_status_label)
        
        # Create hidden status label for compatibility (not displayed)
        self.tracking_mode_status = QLabel()
        self.tracking_mode_status.setVisible(False)
        
        tracking_right_main_layout.addWidget(mode_group)
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
        threshold_layout.setSpacing(4)
        threshold_layout.setContentsMargins(6, 6, 6, 6)
        source_threshold_layout.addWidget(threshold_group)
        
        # Histogram - reduced height for compact UI
        self.figure_threshold_hist, self.ax_threshold_hist = plt.subplots(figsize=(6, 0.8))
        self.canvas_threshold_hist = FigureCanvas(self.figure_threshold_hist)
        self.canvas_threshold_hist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.canvas_threshold_hist.setMaximumHeight(140)
        self.canvas_threshold_hist.setMinimumHeight(100)
        # Initialize threshold histogram as blank black panel
        self.figure_threshold_hist.clear()
        self.ax_threshold_hist = self.figure_threshold_hist.add_subplot(111)
        self.ax_threshold_hist.set_facecolor('black')
        self.ax_threshold_hist.axis('off')
        self.canvas_threshold_hist.draw()
        threshold_layout.addWidget(self.canvas_threshold_hist)
        
        # Instructional label with threshold value
        slider_instruction_container = QWidget()
        slider_instruction_layout = QHBoxLayout(slider_instruction_container)
        slider_instruction_layout.setContentsMargins(0, 2, 0, 0)
        slider_instruction_layout.setSpacing(4)
        
        instruction_label = QLabel("◀ Drag to set threshold ▶")
        instruction_label.setStyleSheet("color: #888888; font-size: 10px;")
        slider_instruction_layout.addWidget(instruction_label)
        
        slider_instruction_layout.addStretch()
        
        # Auto-threshold button (compact, matching slider color)
        self.auto_threshold_btn = QPushButton("Auto")
        self.auto_threshold_btn.setFixedWidth(45)
        self.auto_threshold_btn.setFixedHeight(20)
        self.auto_threshold_btn.setToolTip("Auto-detect optimal threshold")
        self.auto_threshold_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4aa;
                color: #1a1a1a;
                border: none;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
                padding: 2px 6px;
            }
            QPushButton:hover {
                background-color: #00e5bb;
            }
            QPushButton:pressed {
                background-color: #00b894;
            }
        """)
        self.auto_threshold_btn.clicked.connect(self.on_auto_threshold_clicked)
        slider_instruction_layout.addWidget(self.auto_threshold_btn)
        
        self.threshold_value_label = QLabel("Value: --")
        self.threshold_value_label.setStyleSheet("color: #00d4aa; font-size: 11px; font-weight: bold;")
        slider_instruction_layout.addWidget(self.threshold_value_label)
        
        threshold_layout.addWidget(slider_instruction_container)
        
        # Threshold Slider with styled handle
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(10000)
        self.threshold_slider.setValue(0)
        self.threshold_slider.setTickPosition(QSlider.NoTicks)  # Remove ticks for cleaner look
        self.threshold_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 6px;
                background: #2a2a2a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00d4aa;
                border: 2px solid #00b894;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #00e5bb;
                border: 2px solid #00d4aa;
            }
            QSlider::sub-page:horizontal {
                background: #00d4aa;
                border-radius: 3px;
            }
        """)
        self.threshold_slider.valueChanged.connect(self.update_threshold_value)
        threshold_layout.addWidget(self.threshold_slider)
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
        
        # Group 2: Detection & Linking Parameters (combined)
        params_group = QGroupBox("Detection & Linking Parameters")
        params_layout = QFormLayout(params_group)
        tracking_right_main_layout.addWidget(params_group)
        
        # Min length
        self.min_length_input = QSpinBox()
        self.min_length_input.setMinimum(1)
        self.min_length_input.setMaximum(1000)
        self.min_length_input.setValue(self.min_length_trajectory)
        self.min_length_input.valueChanged.connect(self.update_min_length_trajectory)
        params_layout.addRow("Min Length Trajectory:", self.min_length_input)
        
        # YX Spot Size
        self.spot_size_input = QSpinBox()
        self.spot_size_input.setMinimum(3)
        self.spot_size_input.setValue(self.yx_spot_size_in_px)
        self.spot_size_input.valueChanged.connect(self.update_yx_spot_size)
        params_layout.addRow("YX Spot Size (px):", self.spot_size_input)
        
        # Z Spot Size
        self.spot_size_z_input = QSpinBox()
        self.spot_size_z_input.setMinimum(1)
        self.spot_size_z_input.setValue(self.z_spot_size_in_px)
        self.spot_size_z_input.valueChanged.connect(self.update_z_spot_size)
        params_layout.addRow("Z Spot Size:", self.spot_size_z_input)
        
        # Cluster radius
        self.cluster_radius_input = QSpinBox()
        self.cluster_radius_input.setMinimum(100)
        self.cluster_radius_input.setMaximum(2000)
        self.cluster_radius_input.setValue(self.cluster_radius_nm)
        self.cluster_radius_input.valueChanged.connect(self.update_cluster_radius)
        params_layout.addRow("Cluster radius (nm):", self.cluster_radius_input)
        
        # Max cluster size
        self.max_spots_cluster_input = QSpinBox()
        self.max_spots_cluster_input.setMinimum(0)
        self.max_spots_cluster_input.setMaximum(1000)
        self.max_spots_cluster_input.setValue(self.maximum_spots_cluster if self.maximum_spots_cluster is not None else 0)
        self.max_spots_cluster_input.valueChanged.connect(self.update_max_spots_cluster)
        params_layout.addRow("Max Cluster Size (0=None):", self.max_spots_cluster_input)
        
        # Max range search
        self.max_range_search_input = QSpinBox()
        self.max_range_search_input.setMinimum(1)
        self.max_range_search_input.setValue(self.maximum_range_search_pixels)
        self.max_range_search_input.valueChanged.connect(self.update_max_range_search_pixels)
        params_layout.addRow("Max Range Search (px):", self.max_range_search_input)
        
        # Memory
        self.memory_input = QSpinBox()
        self.memory_input.setMinimum(0)
        self.memory_input.setMaximum(5)
        self.memory_input.setValue(self.memory)
        self.memory_input.valueChanged.connect(self.update_memory)
        params_layout.addRow("Memory:", self.memory_input)
        
        # Group 3: Multi-Channel Tracking Results (moved up for better workflow)
        multi_channel_group = QGroupBox("Tracked Channels")
        multi_channel_layout = QVBoxLayout(multi_channel_group)
        multi_channel_layout.setSpacing(4)
        multi_channel_layout.setContentsMargins(6, 6, 6, 6)
        tracking_right_main_layout.addWidget(multi_channel_group)
        
        # Info label
        tracked_info_label = QLabel("Run tracking in each channel to add:")
        tracked_info_label.setStyleSheet("color: #888888; font-size: 10px;")
        multi_channel_layout.addWidget(tracked_info_label)
        
        # List widget to show tracked channels
        self.tracked_channels_list = QListWidget()
        self.tracked_channels_list.setMaximumHeight(100)
        self.tracked_channels_list.setStyleSheet("""
            QListWidget {
                background-color: #1a1a1a;
                color: #00d4aa;
                border: 1px solid #444;
                border-radius: 4px;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 2px 4px;
            }
            QListWidget::item:selected {
                background-color: #333;
            }
        """)
        multi_channel_layout.addWidget(self.tracked_channels_list)
        
        # Buttons for clearing tracking data
        clear_buttons_layout = QHBoxLayout()
        
        self.clear_channel_button = QPushButton("Clear Channel")
        self.clear_channel_button.setStyleSheet("font-size: 10px;")
        self.clear_channel_button.setToolTip("Remove tracking data for the selected channel")
        self.clear_channel_button.clicked.connect(self.clear_channel_tracking)
        clear_buttons_layout.addWidget(self.clear_channel_button)
        
        self.clear_all_tracking_button = QPushButton("Clear All")
        self.clear_all_tracking_button.setStyleSheet("font-size: 10px; color: #ff6b6b;")
        self.clear_all_tracking_button.setToolTip("Remove all tracking data from all channels")
        self.clear_all_tracking_button.clicked.connect(self.clear_all_tracking)
        clear_buttons_layout.addWidget(self.clear_all_tracking_button)
        
        multi_channel_layout.addLayout(clear_buttons_layout)
        
        # Group 5: Intensity Calculation
        intensity_calc_group = QGroupBox("Intensity Calculation")
        intensity_calc_layout = QHBoxLayout(intensity_calc_group)
        tracking_right_main_layout.addWidget(intensity_calc_group)
        
        self.fixed_size_intensity_checkbox = QCheckBox("Use Fixed Size")
        self.fixed_size_intensity_checkbox.setChecked(self.use_fixed_size_for_intensity_calculation)
        self.fixed_size_intensity_checkbox.stateChanged.connect(self.update_use_fixed_size_intensity)
        intensity_calc_layout.addWidget(self.fixed_size_intensity_checkbox)
        
        self.fast_gaussian_fit_checkbox = QCheckBox("Fast Gaussian Fit")
        self.fast_gaussian_fit_checkbox.setChecked(True)  # Default to fast mode
        self.fast_gaussian_fit_checkbox.setToolTip("Use moment-based PSF estimation (faster but less accurate)")
        self.fast_gaussian_fit_checkbox.stateChanged.connect(self.update_fast_gaussian_fit)
        intensity_calc_layout.addWidget(self.fast_gaussian_fit_checkbox)
        
        # Group 6: Control - Random Point Generation
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


# =============================================================================
# =============================================================================
# DISTRIBUTION TAB
# =============================================================================
# =============================================================================

    def plot_intensity_histogram(self):
        if self.df_tracking.empty:
            self.figure_distribution.clear()
            ax = self.figure_distribution.add_subplot(111)
            ax.set_facecolor('black')
            self.figure_distribution.patch.set_facecolor('black')
            ax.axis('off')
            ax.text(0.5, 0.5, 'No tracking data available.\nPlease run tracking first.',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='white', transform=ax.transAxes)
            self.canvas_distribution.draw()
            return
        
        # Filter by tracking channel (spot_type) - single channel required
        df_to_plot = self.df_tracking.copy()
        tracking_ch = None
        if hasattr(self, 'distribution_tracking_channel_combo'):
            tracking_ch = self.distribution_tracking_channel_combo.currentData()
            # Check if placeholder "No tracked channels" is selected
            if tracking_ch == -1:
                self.figure_distribution.clear()
                ax = self.figure_distribution.add_subplot(111)
                ax.set_facecolor('black')
                self.figure_distribution.patch.set_facecolor('black')
                ax.axis('off')
                ax.text(0.5, 0.5, 'No tracked channels available.\nPlease run tracking first.',
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=12, color='white', transform=ax.transAxes)
                self.canvas_distribution.draw()
                return
            # Filter by spot_type
            if 'spot_type' in df_to_plot.columns:
                df_to_plot = df_to_plot[df_to_plot['spot_type'] == tracking_ch]
        
        if df_to_plot.empty:
            self.figure_distribution.clear()
            ax = self.figure_distribution.add_subplot(111)
            ax.set_facecolor('black')
            self.figure_distribution.patch.set_facecolor('black')
            ax.axis('off')
            ax.text(0.5, 0.5, f'No data for selected tracking channel.',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='white', transform=ax.transAxes)
            self.canvas_distribution.draw()
            return
        
        selected_field = self.intensity_field_combo.currentText()
        selected_channel = self.intensity_channel_combo.currentData()  # channel index
        min_percentile = self.intensity_min_percentile_spin.value()
        max_percentile = self.intensity_max_percentile_spin.value()
        # Determine field name
        field_name = "cluster_size" if selected_field == "cluster_size" else f'{selected_field}_ch_{selected_channel}'
        if field_name not in df_to_plot.columns:
            ax = self.figure_distribution.add_subplot(111)
            ax.set_facecolor('black')
            ax.axis('off')
            ax.text(0.5, 0.5, f"No data for {field_name}.", horizontalalignment='center', verticalalignment='center', fontsize=12, color='white', transform=ax.transAxes)
            self.canvas_distribution.draw()
            return
        
        # Get unique cell IDs
        if 'cell_id' in df_to_plot.columns:
            cell_ids = sorted(df_to_plot['cell_id'].dropna().unique())
        else:
            cell_ids = [0]  # Fallback if no cell_id column
        
        # Color palette for cells (bright colors for dark background)
        cell_colors = ['cyan', 'magenta', 'lime', 'orange', 'yellow', 'red', 
                       'deepskyblue', 'hotpink', 'chartreuse', 'coral', 
                       'gold', 'tomato', 'aqua', 'violet', 'springgreen']
        
        self.figure_distribution.clear()
        ax = self.figure_distribution.add_subplot(111)
        ax.set_facecolor('black')
        self.figure_distribution.patch.set_facecolor('black')
        
        # Calculate global percentile limits for consistent binning
        all_data = df_to_plot[field_name].dropna().values
        if len(all_data) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, f"No data points found for {field_name}.", 
                    horizontalalignment='center', verticalalignment='center', 
                    fontsize=12, color='white', transform=ax.transAxes)
            self.canvas_distribution.draw()
            return
        
        lower_limit = np.nanpercentile(all_data, min_percentile)
        upper_limit = np.nanpercentile(all_data, max_percentile)
        
        # Calculate alpha based on number of cells (more cells = more transparency)
        n_cells = len(cell_ids)
        alpha = max(0.3, min(0.7, 1.0 / (n_cells ** 0.5)))  # Adaptive alpha
        
        # Plot histogram for each cell
        stats_text = ""
        for idx, cell_id in enumerate(cell_ids):
            cell_data = df_to_plot[df_to_plot['cell_id'] == cell_id][field_name].dropna().values
            if len(cell_data) == 0:
                continue
            
            # Filter by percentile limits
            cell_data_filtered = cell_data[(cell_data >= lower_limit) & (cell_data <= upper_limit)]
            if len(cell_data_filtered) == 0:
                continue
            
            color = cell_colors[idx % len(cell_colors)]
            
            ax.hist(
                cell_data_filtered,
                bins=40,
                histtype='stepfilled',
                alpha=alpha,
                color=color,
                edgecolor=color,
                linewidth=1,
                label=f"Cell {int(cell_id)} (n={len(cell_data_filtered)})"
            )
            
            # Add stats for each cell
            mean_val = np.mean(cell_data)
            median_val = np.median(cell_data)
            stats_text += f"Cell {int(cell_id)}: µ={mean_val:.1f}, M={median_val:.1f}\n"
        
        # Styling
        ax.set_xlabel(selected_field, color='white', fontsize=11)
        ax.set_ylabel('Count', color='white', fontsize=11)
        ax.tick_params(colors='white', which='both')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.3, alpha=0.5)
        ax.set_title(f"{selected_field} Distribution by Cell", fontsize=12, color='white')
        
        # Legend
        if n_cells <= 10:
            ax.legend(loc='upper right', fontsize=8, facecolor='black', 
                     edgecolor='white', labelcolor='white', framealpha=0.8)
        
        # Stats box
        if stats_text:
            props = dict(boxstyle='round', facecolor='black', edgecolor='white', alpha=0.8)
            ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='left', 
                   color='white', bbox=props, fontsize=8, family='monospace')
        
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
        
        # Tracking Channel group (first/prominent)
        tracking_group = QGroupBox("Tracking Channel")
        tracking_layout = QHBoxLayout()
        self.distribution_tracking_channel_combo = QComboBox()
        # Will be populated on tab switch with tracked channels (no "All" option for multi-cell)
        tracking_layout.addWidget(self.distribution_tracking_channel_combo)
        tracking_group.setLayout(tracking_layout)
        right_layout.addWidget(tracking_group)
        
        field_channel_group = QGroupBox("Selection")
        field_channel_layout = QFormLayout(field_channel_group)
        self.intensity_field_combo = QComboBox()
        self.intensity_field_combo.addItems(["spot_int", "spot_size", "psf_amplitude", "psf_sigma", "total_spot_int", "cluster_size", "snr"])
        field_channel_layout.addRow(QLabel("Field:"), self.intensity_field_combo)
        self.intensity_channel_combo = QComboBox()
        field_channel_layout.addRow(QLabel("Data Channel:"), self.intensity_channel_combo)
        
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

        # Tracking channel filter (first/prominent - single channel only)
        tracking_ch_label = QLabel("Tracking Ch:")
        self.time_course_tracking_channel_combo = QComboBox()
        # Will be populated on tab switch with tracked channels (no "All" option for multi-cell)
        controls_layout.addWidget(tracking_ch_label)
        controls_layout.addWidget(self.time_course_tracking_channel_combo)

        # Channel selection (data channel)
        channel_label = QLabel("Data Channel:")
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
        # Auto-recompute
        self._trigger_correlation_recompute()

    def update_remove_outliers(self, state):
        self.remove_outliers = (state == Qt.Checked)
        # Auto-recompute
        self._trigger_correlation_recompute()

    def update_field_name(self, text):
        # Used in compute_correlations
        self.selected_field_name_for_correlation = text
        # Auto-recompute
        self._trigger_correlation_recompute()

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
        # Auto-recompute
        self._trigger_correlation_recompute()

    def create_correlation_channel_checkboxes(self):
        for cb in self.channel_checkboxes:
            self.channel_selection_layout.removeWidget(cb)
            cb.setParent(None)
        self.channel_checkboxes = []
        for idx, channel_name in enumerate(self.channel_names):
            checkbox = QCheckBox(f"Ch {idx}")
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
        # Update slider ranges based on current data
        self._update_correlation_sliders_for_data()
        
        # 1) sanity checks
        
        # Check for insufficient time points (correlation needs multiple frames)
        if getattr(self, 'total_frames', 0) < 3:
            QMessageBox.warning(self, "Insufficient Data", 
                "Correlation analysis requires at least 3 time points.\n\n"
                f"Current image has {getattr(self, 'total_frames', 0)} frame(s). "
                "Autocorrelation measures temporal patterns which require multiple time points.")
            return
        
        if not getattr(self, 'has_tracked', False):
            QMessageBox.warning(self, "Correlation Unavailable",
                                "You must run particle tracking before computing correlations.")
            return
        if self.df_tracking.empty:
            return
        
        # Filter by tracking channel (spot_type) - single channel required
        df_for_correlation = self.df_tracking.copy()
        tracking_ch = None
        if hasattr(self, 'correlation_tracking_channel_combo'):
            tracking_ch = self.correlation_tracking_channel_combo.currentData()
            # Check if placeholder "No tracked channels" is selected
            if tracking_ch == -1:
                QMessageBox.warning(self, "No Data", "No tracked channels available. Please run tracking first.")
                return
            # Filter by spot_type
            if 'spot_type' in df_for_correlation.columns:
                df_for_correlation = df_for_correlation[df_for_correlation['spot_type'] == tracking_ch]
        
        if df_for_correlation.empty:
            QMessageBox.warning(self, "No Data", "No tracking data for selected channel.")
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
            if col not in df_for_correlation.columns:
                continue
            arr = mi.Utilities().df_trajectories_to_array(
                dataframe=df_for_correlation,
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
                if col not in df_for_correlation.columns:
                    # No SNR column for this channel—keep as-is
                    new_intensity_arrays[ch] = arr_int
                    continue
                # Build intensity & SNR using the SAME particle intersection & order
                arr_int_raw, arr_snr_raw, _ = mi.Utilities().df_fields_to_arrays_aligned(
                    dataframe=df_for_correlation,
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
                    logging.debug(f"After alignment, no valid indices remain for channel {ch}.")
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
        
        # Get unique cell IDs for per-cell correlation
        if 'cell_id' in df_for_correlation.columns:
            cell_ids = sorted(df_for_correlation['cell_id'].dropna().unique())
        else:
            cell_ids = [None]  # No cell separation
        
        if correlation_type == 'autocorrelation':
            for ch, data_all in intensity_arrays.items():
                # Compute per-cell correlations
                for cell_id in cell_ids:
                    # Filter data for this cell
                    if cell_id is not None:
                        cell_df = df_for_correlation[df_for_correlation['cell_id'] == cell_id]
                        col = f"{field_base}_ch_{ch}"
                        if col not in cell_df.columns or cell_df.empty:
                            continue
                        data = mi.Utilities().df_trajectories_to_array(
                            dataframe=cell_df,
                            selected_field=col,
                            fill_value=np.nan,
                            total_frames=self.total_frames
                        )
                        try:
                            data = mi.Utilities().shift_trajectories(
                                data,
                                min_percentage_data_in_trajectory=self.min_percentage_data_in_trajectory
                            )
                        except ValueError:
                            continue
                        if data.shape[0] == 0:
                            continue
                    else:
                        data = data_all
                    
                    try:
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
                    except Exception as e:
                        logging.debug(f"Correlation failed for cell {cell_id}, ch {ch}: {e}")
                        continue
                    
                    if index_max >= len(lags):
                        index_max = len(lags) - 1
                    
                    self.correlation_results.append({
                        'type': 'autocorrelation',
                        'channel': ch,
                        'cell_id': cell_id,
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
                        'n_trajectories': data.shape[0],
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
        """Setup Correlation tab with interactive sliders and reorganized layout."""
        correlation_layout = QHBoxLayout(self.correlation_tab)
        
        # =====================================================================
        # LEFT SIDE: Plot area with sliders
        # =====================================================================
        left_layout = QVBoxLayout()
        correlation_layout.addLayout(left_layout, stretch=4)
        
        # Top controls (correlation type, channels, fit type)
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
        
        controls_layout.addStretch()
        
        # ---------------------------------------------------------------------
        # Plot area with decorrelation slider on the right
        # ---------------------------------------------------------------------
        plot_row = QHBoxLayout()
        
        # Main figure
        self.figure_correlation = Figure(figsize=(20, 20))
        self.canvas_correlation = FigureCanvas(self.figure_correlation)
        self.canvas_correlation.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_row.addWidget(self.canvas_correlation, stretch=1)
        
        # Decorrelation threshold vertical slider (right of plot)
        decorr_container = QWidget()
        decorr_container.setFixedWidth(50)
        decorr_layout = QVBoxLayout(decorr_container)
        decorr_layout.setContentsMargins(2, 5, 2, 5)
        decorr_layout.setSpacing(2)
        
        decorr_top_label = QLabel("Decorr")
        decorr_top_label.setAlignment(Qt.AlignCenter)
        decorr_top_label.setStyleSheet("font-size: 10px;")
        decorr_layout.addWidget(decorr_top_label)
        
        # Slider range: -500 to 1000 maps to -0.500 to 1.000 (0.001 resolution)
        # This allows threshold to go into negative ACF values
        self.decorr_threshold_slider = QSlider(Qt.Vertical)
        self.decorr_threshold_slider.setRange(-500, 1000)  # Maps to -0.500 to 1.000
        self.decorr_threshold_slider.setValue(10)  # Default 0.010
        self.decorr_threshold_slider.setInvertedAppearance(True)  # Higher values at top
        self.decorr_threshold_slider.valueChanged.connect(self._on_decorr_slider_label_update)  # Update label only
        self.decorr_threshold_slider.sliderReleased.connect(self._on_decorr_slider_released)  # Recompute on release
        self.decorr_threshold_slider.setToolTip("Decorrelation threshold for dwell time calculation\n(Can be negative to match ACF minimum)")
        decorr_layout.addWidget(self.decorr_threshold_slider, stretch=1)
        
        self.decorr_value_label = QLabel("0.010")
        self.decorr_value_label.setAlignment(Qt.AlignCenter)
        self.decorr_value_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        decorr_layout.addWidget(self.decorr_value_label)
        
        plot_row.addWidget(decorr_container)
        left_layout.addLayout(plot_row, stretch=1)
        
        # ---------------------------------------------------------------------
        # X-axis (max lag) horizontal slider below plot
        # ---------------------------------------------------------------------
        x_slider_container = QWidget()
        x_slider_layout = QHBoxLayout(x_slider_container)
        x_slider_layout.setContentsMargins(5, 0, 5, 0)
        
        x_slider_layout.addWidget(QLabel("τ max:"))
        
        self.x_max_lag_slider = QSlider(Qt.Horizontal)
        self.x_max_lag_slider.setRange(10, 1000)  # Will be updated when data loads
        self.x_max_lag_slider.setValue(200)
        self.x_max_lag_slider.valueChanged.connect(self._on_x_lag_slider_changed)
        self.x_max_lag_slider.setToolTip("Maximum lag time to display on X-axis")
        x_slider_layout.addWidget(self.x_max_lag_slider, stretch=1)
        
        self.x_lag_label = QLabel("200 frames")
        self.x_lag_label.setMinimumWidth(120)
        x_slider_layout.addWidget(self.x_lag_label)
        
        left_layout.addWidget(x_slider_container)
        
        # ---------------------------------------------------------------------
        # Y-axis percentile sliders (horizontal, for simplicity)
        # ---------------------------------------------------------------------
        y_slider_container = QWidget()
        y_slider_layout = QHBoxLayout(y_slider_container)
        y_slider_layout.setContentsMargins(5, 0, 5, 0)
        
        y_slider_layout.addWidget(QLabel("Y-axis:"))
        
        y_slider_layout.addWidget(QLabel("Min%"))
        self.y_min_percentile_slider = QSlider(Qt.Horizontal)
        self.y_min_percentile_slider.setRange(-5, 50)  # Allow negative for padding below min data
        self.y_min_percentile_slider.setValue(0)
        self.y_min_percentile_slider.valueChanged.connect(self._on_y_percentile_changed)
        self.y_min_percentile_slider.setFixedWidth(80)
        y_slider_layout.addWidget(self.y_min_percentile_slider)
        
        self.y_min_label = QLabel("0%")
        self.y_min_label.setFixedWidth(30)
        y_slider_layout.addWidget(self.y_min_label)
        
        y_slider_layout.addWidget(QLabel("Max%"))
        self.y_max_percentile_slider = QSlider(Qt.Horizontal)
        self.y_max_percentile_slider.setRange(80, 100)  # Finer resolution for top 20%
        self.y_max_percentile_slider.setValue(100)
        self.y_max_percentile_slider.valueChanged.connect(self._on_y_percentile_changed)
        self.y_max_percentile_slider.setFixedWidth(80)
        y_slider_layout.addWidget(self.y_max_percentile_slider)
        
        self.y_max_label = QLabel("100%")
        self.y_max_label.setFixedWidth(35)
        y_slider_layout.addWidget(self.y_max_label)
        
        y_slider_layout.addStretch()
        left_layout.addWidget(y_slider_container)
        
        # ---------------------------------------------------------------------
        # Navigation Toolbar + Export button
        # ---------------------------------------------------------------------
        correlation_toolbar_layout = QHBoxLayout()
        self.toolbar_correlation = NavigationToolbar(self.canvas_correlation, self)
        correlation_toolbar_layout.addWidget(self.toolbar_correlation)
        export_correlation_image_button = QPushButton("Export Correlation Image", self)
        export_correlation_image_button.clicked.connect(self.export_correlation_image)
        correlation_toolbar_layout.addWidget(export_correlation_image_button)
        left_layout.addLayout(correlation_toolbar_layout)
        
        # =====================================================================
        # RIGHT SIDE: Settings panel
        # =====================================================================
        right_layout = QVBoxLayout()
        correlation_layout.addLayout(right_layout, stretch=1)
        
        # Tracking Channel group
        tracking_group = QGroupBox("Tracking Channel")
        tracking_layout = QHBoxLayout()
        self.correlation_tracking_channel_combo = QComboBox()
        tracking_layout.addWidget(self.correlation_tracking_channel_combo)
        tracking_group.setLayout(tracking_layout)
        right_layout.addWidget(tracking_group)
        
        # ---------------------------------------------------------------------
        # Data Selection group
        # ---------------------------------------------------------------------
        data_group = QGroupBox("Data Selection")
        data_layout = QVBoxLayout()
        data_group.setLayout(data_layout)
        
        # Field selection
        field_row = QHBoxLayout()
        field_row.addWidget(QLabel("Field:"))
        self.field_name_combo = QComboBox()
        self.field_name_combo.addItems(["spot_int", "psf_amplitude", "total_spot_int", "snr"])
        self.field_name_combo.currentTextChanged.connect(self.update_field_name)
        field_row.addWidget(self.field_name_combo, stretch=1)
        data_layout.addLayout(field_row)
        
        # Min % Data slider with frame display
        min_pct_row = QHBoxLayout()
        min_pct_row.addWidget(QLabel("Min % Data:"))
        
        self.min_pct_data_slider = QSlider(Qt.Horizontal)
        self.min_pct_data_slider.setRange(1, 50)  # 1% to 50%
        self.min_pct_data_slider.setValue(int(self.min_percentage_data_in_trajectory * 100))
        self.min_pct_data_slider.valueChanged.connect(self._on_min_pct_slider_label_update)  # Update label only
        self.min_pct_data_slider.sliderReleased.connect(self._on_min_pct_slider_released)  # Recompute on release
        self.min_pct_data_slider.setToolTip("Minimum percentage of frames with valid data per trajectory")
        min_pct_row.addWidget(self.min_pct_data_slider, stretch=1)
        data_layout.addLayout(min_pct_row)
        
        # Frame count label
        self.min_pct_label = QLabel(f"{int(self.min_percentage_data_in_trajectory * 100)}% (-- / -- frames)")
        self.min_pct_label.setStyleSheet("font-size: 10px; color: #aaa;")
        data_layout.addWidget(self.min_pct_label)
        
        # Hidden spinbox for backward compatibility
        self.max_percentage_spin = QDoubleSpinBox()
        self.max_percentage_spin.setDecimals(3)
        self.max_percentage_spin.setMinimum(0.0)
        self.max_percentage_spin.setMaximum(1.0)
        self.max_percentage_spin.setValue(self.min_percentage_data_in_trajectory)
        self.max_percentage_spin.setVisible(False)
        self.max_percentage_spin.valueChanged.connect(self.update_min_percentage_data_in_trajectory)
        
        # Start Lag (keep as spinbox for precision)
        start_lag_row = QHBoxLayout()
        start_lag_row.addWidget(QLabel("Start Lag:"))
        self.start_lag_input = QSpinBox()
        self.start_lag_input.setMinimum(0)
        self.start_lag_input.setValue(0)
        self.start_lag_input.setFixedWidth(60)
        start_lag_row.addWidget(self.start_lag_input)
        start_lag_row.addStretch()
        data_layout.addLayout(start_lag_row)
        
        # Fit Lag slider
        fit_lag_row = QHBoxLayout()
        fit_lag_row.addWidget(QLabel("Fit Lag:"))
        
        self.fit_lag_slider = QSlider(Qt.Horizontal)
        self.fit_lag_slider.setRange(10, 500)  # Will be updated when data loads
        self.fit_lag_slider.setValue(99)
        self.fit_lag_slider.valueChanged.connect(self._on_fit_lag_slider_changed)
        self.fit_lag_slider.setToolTip("Maximum lag index for fitting (determines dwell time calculation range)")
        fit_lag_row.addWidget(self.fit_lag_slider, stretch=1)
        data_layout.addLayout(fit_lag_row)
        
        self.fit_lag_label = QLabel("99 frames")
        self.fit_lag_label.setStyleSheet("font-size: 10px; color: #aaa;")
        data_layout.addWidget(self.fit_lag_label)
        
        # Hidden spinbox for backward compatibility
        self.index_max_lag_for_fit_input = QSpinBox()
        self.index_max_lag_for_fit_input.setMinimum(1)
        self.index_max_lag_for_fit_input.setValue(99)
        self.index_max_lag_for_fit_input.setMaximum(1000)
        self.index_max_lag_for_fit_input.setVisible(False)
        
        right_layout.addWidget(data_group)
        
        # Keep hidden spinboxes for backward compatibility
        self.max_lag_input = QSpinBox()
        self.max_lag_input.setMinimum(1)
        self.max_lag_input.setMaximum(10000)
        self.max_lag_input.setValue(200)
        self.max_lag_input.setVisible(False)
        self.max_lag_input.valueChanged.connect(self.update_max_lag)
        
        self.de_correlation_threshold_input = QDoubleSpinBox()
        self.de_correlation_threshold_input.setDecimals(3)
        self.de_correlation_threshold_input.setMinimum(0.0)
        self.de_correlation_threshold_input.setMaximum(1.0)
        self.de_correlation_threshold_input.setValue(self.de_correlation_threshold)
        self.de_correlation_threshold_input.setVisible(False)
        self.de_correlation_threshold_input.valueChanged.connect(self.update_de_correlation_threshold)
        
        # Hidden percentile spinboxes for backward compatibility
        self.correlation_min_percentile_input = QDoubleSpinBox()
        self.correlation_min_percentile_input.setValue(0.0)
        self.correlation_min_percentile_input.setVisible(False)
        self.correlation_min_percentile_input.valueChanged.connect(self.on_correlation_percentile_changed)
        
        self.correlation_max_percentile_input = QDoubleSpinBox()
        self.correlation_max_percentile_input.setValue(100.0)
        self.correlation_max_percentile_input.setVisible(False)
        self.correlation_max_percentile_input.valueChanged.connect(self.on_correlation_percentile_changed)
        
        # ---------------------------------------------------------------------
        # Quality Controls group (collapsible)
        # ---------------------------------------------------------------------
        quality_group = QGroupBox("Quality Controls")
        quality_group.setCheckable(True)
        quality_group.setChecked(True)  # Start expanded
        quality_layout = QVBoxLayout()
        quality_group.setLayout(quality_layout)
        
        # Baseline correction
        self.correct_baseline_checkbox = QCheckBox("Baseline Correction")
        self.correct_baseline_checkbox.setChecked(True)
        self.correct_baseline_checkbox.stateChanged.connect(self.update_correct_baseline)
        quality_layout.addWidget(self.correct_baseline_checkbox)
        
        # Remove outliers
        self.remove_outliers_checkbox = QCheckBox("Remove Outliers")
        self.remove_outliers_checkbox.setChecked(True)
        self.remove_outliers_checkbox.stateChanged.connect(self.update_remove_outliers)
        quality_layout.addWidget(self.remove_outliers_checkbox)
        
        # Multi-Tau
        self.multiTauCheck = QCheckBox("Multi-Tau")
        self.multiTauCheck.setChecked(False)
        self.multiTauCheck.stateChanged.connect(self.update_multi_tau)
        quality_layout.addWidget(self.multiTauCheck)
        
        # Show Individual Traces
        self.show_individual_traces_checkbox = QCheckBox("Show Individual Traces")
        self.show_individual_traces_checkbox.setChecked(False)
        self.show_individual_traces_checkbox.setToolTip("Plot individual trajectory ACFs behind the mean (can slow down rendering)")
        self.show_individual_traces_checkbox.stateChanged.connect(self._on_show_traces_changed)
        quality_layout.addWidget(self.show_individual_traces_checkbox)
        
        # SNR Threshold
        snr_row = QHBoxLayout()
        snr_row.addWidget(QLabel("SNR Threshold:"))
        self.snr_threshold_for_acf = QDoubleSpinBox()
        self.snr_threshold_for_acf.setRange(0.0, 5.0)
        self.snr_threshold_for_acf.setValue(0.1)
        self.snr_threshold_for_acf.setSingleStep(0.1)
        self.snr_threshold_for_acf.valueChanged.connect(self.update_snr_threshold_for_acf)
        snr_row.addWidget(self.snr_threshold_for_acf)
        snr_row.addStretch()
        quality_layout.addLayout(snr_row)
        
        self.snr_threshold_for_acf_value = self.snr_threshold_for_acf.value()
        
        right_layout.addWidget(quality_group)
        
        # ---------------------------------------------------------------------
        # Run Button
        # ---------------------------------------------------------------------
        self.compute_correlations_button = QPushButton("Run")
        self.compute_correlations_button.clicked.connect(self.compute_correlations)
        self.compute_correlations_button.setMinimumHeight(40)
        self.compute_correlations_button.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(self.compute_correlations_button)
        
        right_layout.addStretch()
    
    def _on_show_traces_changed(self, state):
        """Handle Show Individual Traces checkbox change - refresh plot."""
        self.display_correlation_plot()
    
    # =========================================================================
    # Correlation Tab Slider Handlers
    # =========================================================================
    
    def _on_decorr_slider_changed(self, value):
        """Handle decorrelation threshold slider change (legacy, full update)."""
        threshold = value / 1000.0
        self.decorr_value_label.setText(f"{threshold:.3f}")
        self.de_correlation_threshold = threshold
        # Sync with spinbox
        self.de_correlation_threshold_input.blockSignals(True)
        self.de_correlation_threshold_input.setValue(threshold)
        self.de_correlation_threshold_input.blockSignals(False)
        # Update plot line in real-time
        self._update_decorr_line_on_plot(threshold)
        # Auto-recompute to update dwell time calculation
        self._trigger_correlation_recompute()
    
    def _on_decorr_slider_label_update(self, value):
        """Update label and plot line only while dragging (no recompute)."""
        threshold = value / 1000.0
        self.decorr_value_label.setText(f"{threshold:.3f}")
        self.de_correlation_threshold = threshold
        # Sync with spinbox
        self.de_correlation_threshold_input.blockSignals(True)
        self.de_correlation_threshold_input.setValue(threshold)
        self.de_correlation_threshold_input.blockSignals(False)
        # Update plot line in real-time
        self._update_decorr_line_on_plot(threshold)
    
    def _on_decorr_slider_released(self):
        """Recompute correlation when user releases slider."""
        self._trigger_correlation_recompute()
    
    def _update_decorr_line_on_plot(self, threshold):
        """Update the decorrelation threshold line on the plot without re-running."""
        if not hasattr(self, 'figure_correlation') or not self.figure_correlation.axes:
            return
        for ax in self.figure_correlation.axes:
            # Find and update horizontal lines
            for line in ax.get_lines():
                if hasattr(line, '_is_decorr_line') and line._is_decorr_line:
                    line.set_ydata([threshold, threshold])
        self.canvas_correlation.draw_idle()
    
    def _on_x_lag_slider_changed(self, value):
        """Handle X-axis max lag slider change."""
        time_interval = getattr(self, 'time_interval_value', 1.0) or 1.0
        try:
            time_interval = float(time_interval)
        except (TypeError, ValueError):
            time_interval = 1.0
        max_time = value * time_interval
        self.x_lag_label.setText(f"{value} frames ({max_time:.1f} s)")
        # Sync with hidden spinbox
        self.max_lag_input.blockSignals(True)
        self.max_lag_input.setValue(value)
        self.max_lag_input.blockSignals(False)
        # Update plot X-axis
        self._update_x_axis_limit(max_time)
    
    def _update_x_axis_limit(self, max_time):
        """Update X-axis limit on correlation plot."""
        if not hasattr(self, 'figure_correlation') or not self.figure_correlation.axes:
            return
        for ax in self.figure_correlation.axes:
            ax.set_xlim(0, max_time)
        self.canvas_correlation.draw_idle()
    
    def _on_y_percentile_changed(self):
        """Handle Y-axis percentile slider change."""
        min_pct = self.y_min_percentile_slider.value()
        max_pct = self.y_max_percentile_slider.value()
        self.y_min_label.setText(f"{min_pct}%")
        self.y_max_label.setText(f"{max_pct}%")
        # Sync with hidden spinboxes
        self.correlation_min_percentile_input.blockSignals(True)
        self.correlation_min_percentile_input.setValue(float(min_pct))
        self.correlation_min_percentile_input.blockSignals(False)
        self.correlation_max_percentile_input.blockSignals(True)
        self.correlation_max_percentile_input.setValue(float(max_pct))
        self.correlation_max_percentile_input.blockSignals(False)
        # Update plot Y-axis
        self._update_y_axis_limits(min_pct, max_pct)
    
    def _update_y_axis_limits(self, min_pct, max_pct):
        """Update Y-axis limits based on percentile values."""
        if not hasattr(self, 'correlation_results') or not self.correlation_results:
            return
        # Collect all correlation values
        all_values = []
        for result in self.correlation_results:
            if 'mean_corr' in result and result['mean_corr'] is not None:
                all_values.extend(result['mean_corr'])
        if not all_values:
            return
        y_min = np.nanpercentile(all_values, min_pct)
        y_max = np.nanpercentile(all_values, max_pct)
        if y_min >= y_max:
            return
        for ax in self.figure_correlation.axes:
            ax.set_ylim(y_min, y_max)
        self.canvas_correlation.draw_idle()
    
    def _on_min_pct_slider_changed(self, value):
        """Handle Min % Data slider change (legacy, called by both new methods)."""
        pct = value / 100.0
        total_frames = getattr(self, 'total_frames', 0) or 0
        min_frames = int(pct * total_frames) if total_frames > 0 else 0
        self.min_pct_label.setText(f"{value}% ({min_frames} / {total_frames} frames)")
        self.min_percentage_data_in_trajectory = pct
        # Sync with hidden spinbox
        self.max_percentage_spin.blockSignals(True)
        self.max_percentage_spin.setValue(pct)
        self.max_percentage_spin.blockSignals(False)
        # Auto-recompute if we have tracking data
        self._trigger_correlation_recompute()
    
    def _on_min_pct_slider_label_update(self, value):
        """Update label only while dragging (no recompute)."""
        pct = value / 100.0
        total_frames = getattr(self, 'total_frames', 0) or 0
        min_frames = int(pct * total_frames) if total_frames > 0 else 0
        self.min_pct_label.setText(f"{value}% ({min_frames} / {total_frames} frames)")
        self.min_percentage_data_in_trajectory = pct
        # Sync with hidden spinbox
        self.max_percentage_spin.blockSignals(True)
        self.max_percentage_spin.setValue(pct)
        self.max_percentage_spin.blockSignals(False)
    
    def _on_min_pct_slider_released(self):
        """Recompute correlation when user releases slider."""
        self._trigger_correlation_recompute()
    
    def _update_correlation_sliders_for_data(self):
        """Update slider ranges based on loaded data."""
        total_frames = getattr(self, 'total_frames', 0) or 100
        time_interval = getattr(self, 'time_interval_value', 1.0) or 1.0
        try:
            time_interval = float(time_interval)
        except (TypeError, ValueError):
            time_interval = 1.0
        
        # Update X-axis slider range
        if hasattr(self, 'x_max_lag_slider'):
            self.x_max_lag_slider.setRange(10, max(total_frames - 1, 10))
            current_val = min(self.x_max_lag_slider.value(), total_frames - 1)
            self.x_max_lag_slider.setValue(current_val)
            max_time = current_val * time_interval
            self.x_lag_label.setText(f"{current_val} frames ({max_time:.1f} s)")
        
        # Update Min % Data label
        if hasattr(self, 'min_pct_data_slider'):
            pct = self.min_pct_data_slider.value()
            min_frames = int((pct / 100.0) * total_frames)
            self.min_pct_label.setText(f"{pct}% ({min_frames} / {total_frames} frames)")
        
        # Update max lag spinboxes
        if hasattr(self, 'max_lag_input'):
            self.max_lag_input.setMaximum(total_frames - 1)
        if hasattr(self, 'index_max_lag_for_fit_input'):
            self.index_max_lag_for_fit_input.setMaximum(total_frames - 1)
        
        # Update fit lag slider range
        if hasattr(self, 'fit_lag_slider'):
            self.fit_lag_slider.setRange(10, max(total_frames - 1, 10))
            current_val = min(self.fit_lag_slider.value(), total_frames - 1)
            self.fit_lag_slider.setValue(current_val)
            fit_time = current_val * time_interval
            self.fit_lag_label.setText(f"{current_val} frames ({fit_time:.1f} s)")
    
    def _on_fit_lag_slider_changed(self, value):
        """Handle Fit Lag slider change."""
        time_interval = getattr(self, 'time_interval_value', 1.0) or 1.0
        try:
            time_interval = float(time_interval)
        except (TypeError, ValueError):
            time_interval = 1.0
        fit_time = value * time_interval
        self.fit_lag_label.setText(f"{value} frames ({fit_time:.1f} s)")
        # Sync with hidden spinbox
        self.index_max_lag_for_fit_input.blockSignals(True)
        self.index_max_lag_for_fit_input.setValue(value)
        self.index_max_lag_for_fit_input.blockSignals(False)
        self.index_max_lag_for_fit = value
        # Auto-recompute
        self._trigger_correlation_recompute()
    
    def _trigger_correlation_recompute(self):
        """Trigger correlation recomputation with debouncing.
        
        This prevents excessive recomputation when sliders are being dragged.
        Uses a timer to wait for user to stop adjusting before recomputing.
        """
        # Only recompute if we have tracking data and have already run once
        if not getattr(self, 'has_tracked', False):
            return
        if not hasattr(self, 'correlation_results') or not self.correlation_results:
            return  # Only auto-recompute if we've already run once
        
        # Cancel any pending recompute
        if hasattr(self, '_correlation_recompute_timer') and self._correlation_recompute_timer is not None:
            self._correlation_recompute_timer.stop()
        
        # Create timer if it doesn't exist
        if not hasattr(self, '_correlation_recompute_timer') or self._correlation_recompute_timer is None:
            from PyQt5.QtCore import QTimer
            self._correlation_recompute_timer = QTimer()
            self._correlation_recompute_timer.setSingleShot(True)
            self._correlation_recompute_timer.timeout.connect(self._do_correlation_recompute)
        
        # Start timer (300ms debounce)
        self._correlation_recompute_timer.start(300)
    
    def _do_correlation_recompute(self):
        """Actually perform the correlation recomputation."""
        try:
            self.compute_correlations()
        except Exception as e:
            import logging
            logging.debug(f"Auto-recompute failed: {e}")

# =============================================================================
# COLOCALIZATION AND COLOCALIZATION MANUAL TABS
# =============================================================================
# =============================================================================

    # Note: update_manual_stats_label(), populate_manual_checkboxes(), and cleanup_manual_colocalization()
    # have been removed and replaced by dedicated functions in each verification subtab:
    # - _update_verify_visual_stats(), populate_verify_visual(), cleanup_verify_visual()
    # - _update_verify_distance_stats(), populate_verify_distance(), cleanup_verify_distance()

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

    def on_colocalization_tracking_channel_changed(self, index):
        """When tracking channel changes, auto-set Reference channel to match."""
        if not hasattr(self, 'colocalization_tracking_channel_combo'):
            return
        tracking_ch = self.colocalization_tracking_channel_combo.currentData()
        if tracking_ch is None or tracking_ch == -1:
            return
        # Set Reference channel (channel_combo_box_1) to the tracking channel
        if hasattr(self, 'channel_combo_box_1') and self.channel_combo_box_1.count() > tracking_ch:
            self.channel_combo_box_1.setCurrentIndex(tracking_ch)
            # Auto-set Colocalize channel to a different channel if possible
            if hasattr(self, 'channel_combo_box_2') and self.channel_combo_box_2.count() > 1:
                other_index = 1 if tracking_ch == 0 else 0
                self.channel_combo_box_2.setCurrentIndex(other_index)

    def _populate_coloc_cell_selector(self):
        """Populate the cell selector combo box based on tracking data."""
        if not hasattr(self, 'coloc_cell_combo'):
            return
        
        # Remember current selection
        current_data = self.coloc_cell_combo.currentData()
        
        # Clear and repopulate
        self.coloc_cell_combo.clear()
        self.coloc_cell_combo.addItem("All Cells (pooled)", -1)
        self.coloc_cell_combo.addItem("All Cells (per-cell avg)", -2)
        
        # Add individual cells if cell_id column exists
        if (hasattr(self, 'df_tracking') and not self.df_tracking.empty 
            and 'cell_id' in self.df_tracking.columns):
            cell_ids = sorted(self.df_tracking['cell_id'].dropna().unique())
            for cid in cell_ids:
                n_spots = len(self.df_tracking[self.df_tracking['cell_id'] == cid])
                self.coloc_cell_combo.addItem(f"Cell {int(cid)} ({n_spots} spots)", int(cid))
        
        # Restore selection if possible
        if current_data is not None:
            idx = self.coloc_cell_combo.findData(current_data)
            if idx >= 0:
                self.coloc_cell_combo.setCurrentIndex(idx)
        

    def compute_colocalization(self):
        """Perform colocalization analysis with per-cell support."""
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
        
        # Filter by tracking channel (spot_type) - single channel required
        df_for_coloc = self.df_tracking.copy()
        tracking_ch = None
        if hasattr(self, 'colocalization_tracking_channel_combo'):
            tracking_ch = self.colocalization_tracking_channel_combo.currentData()
            # Check if placeholder "No tracked channels" is selected
            if tracking_ch == -1:
                if invoked_by_run:
                    QMessageBox.warning(self, "No Data", "No tracked channels available. Please run tracking first.")
                return
            # Filter by spot_type
            if 'spot_type' in df_for_coloc.columns:
                df_for_coloc = df_for_coloc[df_for_coloc['spot_type'] == tracking_ch]
        
        if df_for_coloc.empty:
            if invoked_by_run:
                QMessageBox.warning(self, "No Data", "No tracking data for selected channel.")
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
        
        # Get analysis parameters
        threshold = self.ml_threshold_input.value() if self.method_ml_radio.isChecked() else self.snr_threshold_input.value()
        method_used = "ML" if self.method_ml_radio.isChecked() else "Intensity"
        
        # Get cell selection mode
        selected_cell = self.coloc_cell_combo.currentData() if hasattr(self, 'coloc_cell_combo') else -1
        
        # Determine cell IDs to analyze
        if 'cell_id' in df_for_coloc.columns:
            all_cell_ids = sorted(df_for_coloc['cell_id'].dropna().unique())
        else:
            all_cell_ids = [None]  # No cell info
        
        # Store per-cell results
        per_cell_results = {}
        
        # Analyze per cell
        for cell_id in all_cell_ids:
            if cell_id is not None:
                cell_df = df_for_coloc[df_for_coloc['cell_id'] == cell_id]
            else:
                cell_df = df_for_coloc
            
            if len(cell_df) < 1:
                continue
            
            # Compute crops for this cell
            try:
                _, mean_crop_cell, _, _ = mi.CropArray(
                    image=image,
                    df_crops=cell_df,
                    crop_size=crop_size,
                    remove_outliers=False,
                    max_percentile=99.95,
                    selected_time_point=None,
                    normalize_each_particle=False
                ).run()
            except Exception:
                continue
            
            # Compute colocalization
            flag_vector_cell, pred_values_cell = self._compute_coloc_flags(
                mean_crop_cell, crop_size, ch2, method_used, threshold
            )
            
            n_spots = len(flag_vector_cell)
            n_coloc = int(np.sum(flag_vector_cell)) if n_spots > 0 else 0
            pct = (n_coloc / n_spots * 100) if n_spots > 0 else 0.0
            
            per_cell_results[cell_id] = {
                'n_spots': n_spots,
                'n_colocalized': n_coloc,
                'percentage': pct,
                'mean_crop': mean_crop_cell,
                'flag_vector': flag_vector_cell,
                'prediction_values': pred_values_cell
            }
        
        # Compute summary statistics
        if len(per_cell_results) > 0:
            percentages = [r['percentage'] for r in per_cell_results.values()]
            total_spots = sum(r['n_spots'] for r in per_cell_results.values())
            total_coloc = sum(r['n_colocalized'] for r in per_cell_results.values())
            
            mean_pct = np.mean(percentages) if percentages else 0.0
            std_pct = np.std(percentages) if len(percentages) > 1 else 0.0
            pooled_pct = (total_coloc / total_spots * 100) if total_spots > 0 else 0.0
        else:
            mean_pct = std_pct = pooled_pct = 0.0
            total_spots = total_coloc = 0
        
        # Update per-cell summary table
        self._update_coloc_percell_table(per_cell_results, mean_pct, std_pct, pooled_pct)
        
        # Determine what to display based on selection
        if selected_cell >= 0 and selected_cell in per_cell_results:
            # Single cell selected
            result = per_cell_results[selected_cell]
            display_crop = result['mean_crop']
            display_flags = result['flag_vector']
            display_pct = result['percentage']
            display_pred = result['prediction_values']
            label_text = f"Cell {selected_cell} Colocalization: {display_pct:.2f}% ({result['n_colocalized']}/{result['n_spots']} spots)"
        elif selected_cell == -2:
            # Per-cell average mode
            display_pct = mean_pct
            label_text = f"Per-Cell Average: {mean_pct:.2f}% ± {std_pct:.2f}% (n={len(per_cell_results)} cells)"
            # Combine all crops for display
            display_crop, display_flags, display_pred = self._combine_percell_crops(per_cell_results, crop_size)
        else:
            # Pooled mode (default)
            display_pct = pooled_pct
            label_text = f"Pooled Colocalization: {pooled_pct:.2f}% ({total_coloc}/{total_spots} spots)"
            display_crop, display_flags, display_pred = self._combine_percell_crops(per_cell_results, crop_size)
        
        self.colocalization_percentage_label.setText(label_text)
        
        # Clear Verify Visual UI so new results can load
        if hasattr(self, 'verify_visual_scroll_area'):
            self.verify_visual_scroll_area.setWidget(QWidget())
        if hasattr(self, 'verify_visual_checkboxes'):
            self.verify_visual_checkboxes = []
        if hasattr(self, 'verify_visual_stats_label'):
            self.verify_visual_stats_label.setText("Run Visual colocalization first, then click Populate")

        self.colocalization_results = {
            'mean_crop_filtered': display_crop,
            'crop_size': crop_size,
            'flag_vector': display_flags,
            'prediction_values_vector': display_pred,
            'ch1_index': ch1,
            'ch2_index': ch2,
            'num_spots_reference': len(display_flags) if display_flags is not None else 0,
            'num_spots_colocalize': int(np.sum(display_flags)) if display_flags is not None else 0,
            'colocalization_percentage': display_pct,
            'threshold_value': threshold,
            'method': method_used,
            'per_cell_results': per_cell_results,
            'pooled_percentage': pooled_pct,
            'mean_percentage': mean_pct,
            'std_percentage': std_pct
        }
        
        if display_crop is not None and display_flags is not None:
            self.display_colocalization_results(display_crop, crop_size, display_flags, ch1, ch2)
        self.extract_colocalization_data(save_df=False)
    
    def _compute_coloc_flags(self, mean_crop, crop_size, ch2, method, threshold):
        """Compute colocalization flags for a set of crops."""
        if mean_crop is None or mean_crop.shape[0] < crop_size:
            return np.array([]), np.array([])
        
        if method == "ML":
            crops_norm = mi.Utilities().normalize_crop_return_list(
                array_crops_YXC=mean_crop,
                crop_size=crop_size,
                selected_color_channel=ch2,
                normalize_to_255=True
            )
            flag_vector, pred_values = ML.predict_crops(model_ML, crops_norm, threshold=threshold)
        else:
            num_crops = mean_crop.shape[0] // crop_size
            if num_crops == 0:
                return np.array([]), np.array([])
            results_snr = [mi.Utilities().is_spot_in_crop(
                        i, crop_size=crop_size, selected_color_channel=ch2,
                        array_crops_YXC=mean_crop,
                        show_plot=False,
                        snr_threshold=threshold)
                        for i in range(num_crops)]
            flag_vector, pred_values = zip(*results_snr)
            flag_vector = np.array(flag_vector)
            pred_values = np.array(pred_values)
        
        return flag_vector, pred_values
    
    def _combine_percell_crops(self, per_cell_results, crop_size):
        """Combine crops from all cells for display."""
        all_crops = []
        all_flags = []
        all_preds = []
        
        for result in per_cell_results.values():
            if result['mean_crop'] is not None and len(result['flag_vector']) > 0:
                all_crops.append(result['mean_crop'])
                all_flags.append(result['flag_vector'])
                all_preds.append(result['prediction_values'])
        
        if not all_crops:
            return None, np.array([]), np.array([])
        
        combined_crop = np.concatenate(all_crops, axis=0)
        combined_flags = np.concatenate(all_flags)
        combined_preds = np.concatenate(all_preds)
        
        return combined_crop, combined_flags, combined_preds
    
    def _update_coloc_percell_table(self, per_cell_results, mean_pct, std_pct, pooled_pct):
        """Update the per-cell summary table display."""
        if not hasattr(self, 'coloc_percell_table'):
            return
        
        if not per_cell_results:
            self.coloc_percell_table.setText("No cells analyzed.")
            return
        
        # Build table text
        lines = ["Cell ID  │ N Spots │ Colocal │  Pct  │ Status"]
        lines.append("─" * 50)
        
        for cell_id, result in sorted(per_cell_results.items()):
            n_spots = result['n_spots']
            n_coloc = result['n_colocalized']
            pct = result['percentage']
            status = "" if n_spots >= 5 else "⚠️ Low N"
            cell_str = str(int(cell_id)) if cell_id is not None else "All"
            lines.append(f"  {cell_str:5}  │  {n_spots:5}  │   {n_coloc:4}  │ {pct:5.1f}% │ {status}")
        
        lines.append("─" * 50)
        total_spots = sum(r['n_spots'] for r in per_cell_results.values())
        total_coloc = sum(r['n_colocalized'] for r in per_cell_results.values())
        lines.append(f" POOLED  │  {total_spots:5}  │   {total_coloc:4}  │ {pooled_pct:5.1f}% │")
        if len(per_cell_results) > 1:
            lines.append(f"   MEAN  │    -    │    -   │ {mean_pct:5.1f}% │ ±{std_pct:.1f}%")
        
        self.coloc_percell_table.setText("\n".join(lines))



    def display_colocalization_results(self, mean_crop, crop_size, flag_vector, ch1, ch2):
        """Display the colocalization result using provided crop data."""
        self.figure_colocalization.clear()
        title = f"Colocalization: {self.colocalization_results['colocalization_percentage']:.2f}%"
        
        # Auto-calculate optimal number of columns based on spot count
        n_spots = len(flag_vector) if flag_vector is not None else 0
        optimal_cols = self._calculate_optimal_coloc_columns(n_spots)
        
        # Update the spinbox to reflect the auto-calculated value
        if hasattr(self, 'columns_spinbox'):
            self.columns_spinbox.blockSignals(True)
            self.columns_spinbox.setValue(optimal_cols)
            self.columns_spinbox.blockSignals(False)
        
        self.plots.plot_matrix_pair_crops(
            mean_crop=mean_crop,
            crop_size=crop_size,
            flag_vector=flag_vector,
            selected_channels=(ch1, ch2),
            figure=self.figure_colocalization,
            crop_spacing=5,
            number_columns=optimal_cols,
            plot_title=title
        )
        try:
            self.plot_image()          
            self.plot_segmentation() 
        except Exception:
            pass
        self.canvas_colocalization.draw()
    
    def _calculate_optimal_coloc_columns(self, n_spots):
        """
        Calculate optimal number of columns for colocalization crop display.
        
        Uses adaptive scaling to create a balanced, wide grid:
        - Small counts (≤100): 8-15 columns
        - Medium counts (100-500): 15-40 columns
        - Large counts (500-2000): 40-80 columns
        - Very large counts (2000-5000): 80-120 columns
        - Massive counts (5000+): 120-200 columns
        
        Args:
            n_spots: Number of spots to display
            
        Returns:
            int: Optimal number of columns
        """
        if n_spots <= 0:
            return 15  # Default
        
        if n_spots <= 100:
            # Small: bounded 8-15
            optimal = max(8, min(15, int(np.sqrt(n_spots) * 1.5)))
        elif n_spots <= 500:
            # Medium: scale from 15 to 40
            optimal = 15 + int((n_spots - 100) * 25 / 400)
        elif n_spots <= 2000:
            # Large: scale from 40 to 80 between 500-2000 spots
            optimal = 40 + int((n_spots - 500) * 40 / 1500)
        elif n_spots <= 5000:
            # Very large: scale from 80 to 120 between 2000-5000 spots
            optimal = 80 + int((n_spots - 2000) * 40 / 3000)
        else:
            # Massive: scale from 120 to 200 for 5000+
            optimal = min(200, 120 + int((n_spots - 5000) * 80 / 10000))
        
        return optimal


    # Note: display_colocalization_manual() removed - replaced by separate 
    # populate_verify_visual() and populate_verify_distance() functions


    def extract_colocalization_data(self, save_df=True):
        if not self.colocalization_results:
            logging.warning("No colocalization results!")
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
        
        # Check if we have per-cell results
        per_cell_results = self.colocalization_results.get('per_cell_results', {})
        
        if per_cell_results and len(per_cell_results) > 0:
            # Create rows for each cell plus summary rows
            rows = []
            for cell_id, result in sorted(per_cell_results.items()):
                cell_str = str(int(cell_id)) if cell_id is not None else "All"
                rows.append({
                    "file name": base_name,
                    "image name": image_name,
                    "cell_id": cell_str,
                    "reference channel": ch1,
                    "colocalize channel": ch2,
                    "number of spots reference": result['n_spots'],
                    "number of spots colocalize": result['n_colocalized'],
                    "colocalization percentage": result['percentage'],
                    "threshold value": self.colocalization_results.get("threshold_value"),
                    "method": self.colocalization_results.get("method")
                })
            
            # Add summary rows
            pooled_pct = self.colocalization_results.get('pooled_percentage', perc)
            mean_pct = self.colocalization_results.get('mean_percentage', perc)
            std_pct = self.colocalization_results.get('std_percentage', 0.0)
            total_spots = sum(r['n_spots'] for r in per_cell_results.values())
            total_coloc = sum(r['n_colocalized'] for r in per_cell_results.values())
            
            rows.append({
                "file name": base_name,
                "image name": image_name,
                "cell_id": "POOLED",
                "reference channel": ch1,
                "colocalize channel": ch2,
                "number of spots reference": total_spots,
                "number of spots colocalize": total_coloc,
                "colocalization percentage": pooled_pct,
                "threshold value": self.colocalization_results.get("threshold_value"),
                "method": self.colocalization_results.get("method")
            })
            
            if len(per_cell_results) > 1:
                rows.append({
                    "file name": base_name,
                    "image name": image_name,
                    "cell_id": "MEAN",
                    "reference channel": ch1,
                    "colocalize channel": ch2,
                    "number of spots reference": "-",
                    "number of spots colocalize": "-",
                    "colocalization percentage": f"{mean_pct:.2f} ± {std_pct:.2f}",
                    "threshold value": self.colocalization_results.get("threshold_value"),
                    "method": self.colocalization_results.get("method")
                })
            
            df = pd.DataFrame(rows)
        else:
            # Original single-row format
            df = pd.DataFrame({
                "file name": [base_name],
                "image name": [image_name],
                "cell_id": ["All"],
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
        """Reset all colocalization sub-tabs: Visual, Distance, and Manual."""
        # === Reset Visual (ML/Intensity) sub-tab ===
        if hasattr(self, 'figure_colocalization'):
            self.figure_colocalization.clear()
            ax = self.figure_colocalization.add_subplot(111)
            ax.set_facecolor('black')
            ax.axis('off')
            ax.text(0.5, 0.5, 'No colocalization data available.',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='white', transform=ax.transAxes)
            self.canvas_colocalization.draw()
        self.colocalization_results = None
        if hasattr(self, 'colocalization_percentage_label'):
            self.colocalization_percentage_label.setText("")
        
        # Reset columns spinbox to default
        if hasattr(self, 'columns_spinbox'):
            self.columns_spinbox.setValue(12)
        
        # Reset Visual channel combos
        if hasattr(self, 'channel_combo_box_1'):
            self.channel_combo_box_1.clear()
        if hasattr(self, 'channel_combo_box_2'):
            self.channel_combo_box_2.clear()
        if hasattr(self, 'colocalization_tracking_channel_combo'):
            self.colocalization_tracking_channel_combo.clear()
        
        # Reset ML/Intensity settings to defaults
        if hasattr(self, 'ml_threshold_input'):
            self.ml_threshold_input.setValue(0.51)
        if hasattr(self, 'snr_threshold_input'):
            self.snr_threshold_input.setValue(3.0)
        if hasattr(self, 'method_ml_radio'):
            self.method_ml_radio.setChecked(True)
        
        # Reset per-cell table
        if hasattr(self, 'coloc_percell_table'):
            self.coloc_percell_table.setText("")
        
        # Reset cell selector to default
        if hasattr(self, 'coloc_cell_combo'):
            self.coloc_cell_combo.clear()
            self.coloc_cell_combo.addItem("All Cells (pooled)", -1)
            self.coloc_cell_combo.addItem("All Cells (per-cell avg)", -2)
        
        # === Reset Distance sub-tab ===
        if hasattr(self, 'distance_coloc_results'):
            self.distance_coloc_results = None
        if hasattr(self, 'dist_results_label'):
            self.dist_results_label.setText("")
        if hasattr(self, 'dist_percell_table'):
            self.dist_percell_table.setText("")
        if hasattr(self, 'dist_cell_combo'):
            self.dist_cell_combo.clear()
            self.dist_cell_combo.addItem("All Cells (pooled)", -1)
            self.dist_cell_combo.addItem("All Cells (per-cell avg)", -2)
        if hasattr(self, 'dist_channel_0_combo'):
            self.dist_channel_0_combo.clear()
        if hasattr(self, 'dist_channel_1_combo'):
            self.dist_channel_1_combo.clear()
        if hasattr(self, 'figure_dist_coloc'):
            self.figure_dist_coloc.clear()
            ax = self.figure_dist_coloc.add_subplot(111)
            ax.set_facecolor('black')
            ax.axis('off')
            ax.text(0.5, 0.5, 'Run distance colocalization to see results.',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='white', transform=ax.transAxes)
            
            # Store axes reference and recreate RectangleSelector
            self.ax_dist_coloc = ax
            self.dist_coloc_zoom_selector = RectangleSelector(
                ax,
                self._on_dist_coloc_zoom_select,
                useblit=True,
                button=[1],
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=False,
                props=dict(facecolor='cyan', edgecolor='white', alpha=0.3, linewidth=2)
            )
            
            self.canvas_dist_coloc.draw()
        if hasattr(self, 'dist_frame_slider'):
            self.dist_frame_slider.setMaximum(0)
            self.dist_frame_slider.setValue(0)
        if hasattr(self, 'dist_frame_label'):
            self.dist_frame_label.setText("Frame: 0/0")
        if hasattr(self, 'dist_play_timer') and self.dist_play_timer.isActive():
            self.dist_play_timer.stop()
        if hasattr(self, 'dist_play_button'):
            self.dist_play_button.setChecked(False)
            self.dist_play_button.setText("▶")
        
        # Reset Z-slider
        if hasattr(self, 'z_slider_dist_coloc'):
            self.z_slider_dist_coloc.setMaximum(0)
            self.z_slider_dist_coloc.setValue(0)
            self.z_slider_dist_coloc.setEnabled(False)
        if hasattr(self, 'z_label_dist_coloc'):
            self.z_label_dist_coloc.setText("Max")
            self.z_label_dist_coloc.setStyleSheet("color: cyan; font-weight: bold;")
        self.dist_coloc_current_z = -1
        
        # Reset zoom ROI
        self.dist_coloc_zoom_roi = None
        if hasattr(self, 'dist_coloc_zoom_label'):
            self.dist_coloc_zoom_label.setText("🔍 Full View")
            self.dist_coloc_zoom_label.setStyleSheet("color: #888888; font-size: 10px;")
        
        # === Reset Manual Verify sub-tab ===
        self.reset_manual_colocalization()
    
    def extract_manual_colocalization_data(self, save_df=True):
        """Extract and optionally save manual colocalization data.
        
        Checks both Verify Visual and Verify Distance subtabs for data.
        """
        # Check which verification subtab has data
        has_visual = hasattr(self, 'verify_visual_checkboxes') and len(self.verify_visual_checkboxes) > 0
        has_distance = hasattr(self, 'verify_distance_checkboxes') and len(self.verify_distance_checkboxes) > 0
        
        if not has_visual and not has_distance:
            print("No manual colocalization data!")
            QMessageBox.warning(self, "No Data", "No manual colocalization selections available.\nUse Verify Visual or Verify Distance tabs first.")
            return
        
        # Prioritize the most recently active/populated subtab
        current_tab = self.coloc_subtabs.currentIndex() if hasattr(self, 'coloc_subtabs') else -1
        
        if current_tab == 1 and has_visual:  # Verify Visual tab
            method_name = "Verify Visual"
            checkboxes = self.verify_visual_checkboxes
            if hasattr(self, 'colocalization_results') and self.colocalization_results:
                ch1 = self.colocalization_results.get('ch1_index', 0)
                ch2 = self.colocalization_results.get('ch2_index', 1)
                threshold_value = self.colocalization_results.get('threshold_value')
                orig_method = self.colocalization_results.get('method', 'ML')
                method_name = f"Verify Visual ({orig_method})"
            else:
                ch1, ch2, threshold_value = 0, 1, None
        elif current_tab == 3 and has_distance:  # Verify Distance tab
            method_name = "Verify Distance"
            checkboxes = self.verify_distance_checkboxes
            if hasattr(self, 'distance_coloc_results') and self.distance_coloc_results:
                results = self.distance_coloc_results
                ch1 = results.get('channel_0', 0)
                ch2 = results.get('channel_1', 1)
                threshold_value = results.get('threshold_distance_px', 2.0)
                use_3d = results.get('use_3d', False)
                method_name = f"Verify Distance {'3D' if use_3d else '2D'}"
            else:
                ch1, ch2, threshold_value = 0, 1, None
        elif has_visual:  # Default to Visual if available
            method_name = "Verify Visual"
            checkboxes = self.verify_visual_checkboxes
            ch1, ch2, threshold_value = 0, 1, None
        else:  # Default to Distance
            method_name = "Verify Distance"
            checkboxes = self.verify_distance_checkboxes
            ch1, ch2, threshold_value = 0, 1, None
        
        # Summarize results
        total = len(checkboxes)
        colocalized = sum(1 for chk in checkboxes if chk.isChecked())
        percent = (colocalized / total * 100.0) if total > 0 else 0.0
        
        # Prepare DataFrame
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
            "threshold value": threshold_value,
            "method": method_name
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
        self.cid_leave_coloc = self.canvas_colocalization.mpl_connect('figure_leave_event', self.on_colocalization_leave)

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

    def on_colocalization_leave(self, event):
        """Hide the zoom inset and rectangle when mouse leaves the colocalization figure."""
        # Remove the inset axes if it exists
        if hasattr(self, 'ax_inset') and self.ax_inset is not None:
            try:
                self.ax_inset.remove()
            except Exception:
                pass
            self.ax_inset = None
        # Remove the red rectangle if it exists
        if hasattr(self, 'rect_zoom') and self.rect_zoom is not None:
            try:
                self.rect_zoom.remove()
            except Exception:
                pass
            self.rect_zoom = None
        # Redraw the canvas
        if hasattr(self, 'canvas_colocalization'):
            self.canvas_colocalization.draw_idle()

    def setup_colocalization_tab(self):
        """Setup colocalization tab with sub-tabs for Visual, Distance, and verification."""
        main_layout = QVBoxLayout(self.colocalization_tab)
        
        # Create sub-tab widget
        self.coloc_subtabs = QTabWidget()
        
        # Tab 0: Visual (ML/Intensity) - Analysis
        self.coloc_visual_widget = QWidget()
        self.setup_coloc_visual_subtab()
        self.coloc_subtabs.addTab(self.coloc_visual_widget, "Visual")
        
        # Tab 1: Verify Visual - Manual verification of Visual results
        self.coloc_verify_visual_widget = QWidget()
        self.setup_coloc_verify_visual_subtab()
        self.coloc_subtabs.addTab(self.coloc_verify_visual_widget, "Verify Visual")
        
        # Tab 2: Distance - Analysis
        self.coloc_distance_widget = QWidget()
        self.setup_coloc_distance_subtab()
        self.coloc_subtabs.addTab(self.coloc_distance_widget, "Distance")
        
        # Tab 3: Verify Distance - Manual verification of Distance results
        self.coloc_verify_distance_widget = QWidget()
        self.setup_coloc_verify_distance_subtab()
        self.coloc_subtabs.addTab(self.coloc_verify_distance_widget, "Verify Distance")
        
        main_layout.addWidget(self.coloc_subtabs)
    
    def setup_coloc_visual_subtab(self):
        """Setup Visual (ML/Intensity) colocalization sub-tab."""
        layout = QVBoxLayout(self.coloc_visual_widget)
        top_layout = QHBoxLayout()
        
        # Tracking channel selector (single channel only)
        trackingChannelGroup = QGroupBox("Tracking Channel")
        trackingChLayout = QHBoxLayout(trackingChannelGroup)
        self.colocalization_tracking_channel_combo = QComboBox()
        # Will be populated on tab switch with tracked channels
        trackingChLayout.addWidget(self.colocalization_tracking_channel_combo)
        top_layout.addWidget(trackingChannelGroup)
        
        # Cell selector for multi-cell analysis
        cellGroup = QGroupBox("Cell Selection")
        cellLayout = QHBoxLayout(cellGroup)
        self.coloc_cell_combo = QComboBox()
        self.coloc_cell_combo.addItem("All Cells (pooled)", -1)
        self.coloc_cell_combo.addItem("All Cells (per-cell avg)", -2)
        cellLayout.addWidget(self.coloc_cell_combo)
        top_layout.addWidget(cellGroup)
        
        channelGroup = QGroupBox("Select Channels")
        chLayout = QHBoxLayout(channelGroup)
        self.channel_combo_box_1 = QComboBox()
        self.channel_combo_box_1.setMinimumWidth(100)  # Ensure channel names are visible
        self.channel_combo_box_2 = QComboBox()
        self.channel_combo_box_2.setMinimumWidth(100)  # Ensure channel names are visible
        chLayout.addWidget(QLabel("Reference:"))
        chLayout.addWidget(self.channel_combo_box_1)
        chLayout.addWidget(QLabel("Colocalize:"))
        chLayout.addWidget(self.channel_combo_box_2)
        top_layout.addWidget(channelGroup)
        
        # Connect tracking channel change to auto-set Reference channel
        self.colocalization_tracking_channel_combo.currentIndexChanged.connect(
            self.on_colocalization_tracking_channel_changed
        )
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
        self.ml_threshold_input.setValue(0.51)  # Default 0.51 for optimal accuracy (97.5%)
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
        self.columns_spinbox.setRange(4, 200)
        self.columns_spinbox.setValue(20)  # Auto-adjusted when running
        self.columns_spinbox.setToolTip("Auto-adjusted based on spot count (larger = wider image)")
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
        self.colocalization_percentage_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00cc66;")
        layout.addWidget(self.colocalization_percentage_label)
        
        # Per-cell summary table (hidden - kept for export functionality)
        self.coloc_percell_group = QGroupBox("Per-Cell Results")
        self.coloc_percell_group.setCheckable(True)
        self.coloc_percell_group.setChecked(False)
        self.coloc_percell_group.setVisible(False)  # Hidden to maximize visualization space
        percell_layout = QVBoxLayout(self.coloc_percell_group)
        self.coloc_percell_table = QLabel("")
        self.coloc_percell_table.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.coloc_percell_table.setWordWrap(True)
        percell_layout.addWidget(self.coloc_percell_table)
        layout.addWidget(self.coloc_percell_group)
        
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
    
    def setup_coloc_distance_subtab(self):
        """Setup Distance-based colocalization sub-tab.
        
        This tab enables colocalization analysis based on Euclidean distance
        between spots detected in two different tracked channels.
        UI designed to match Tracking tab style for consistency.
        """
        layout = QVBoxLayout(self.coloc_distance_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Controls row (compact)
        controls = QHBoxLayout()
        controls.setSpacing(10)
        
        # Channel selectors (tracked channels only)
        ch_group = QGroupBox("Channels")
        ch_layout = QHBoxLayout(ch_group)
        ch_layout.setContentsMargins(5, 5, 5, 5)
        ch_layout.addWidget(QLabel("Reference:"))
        self.dist_channel_0_combo = QComboBox()
        self.dist_channel_0_combo.setMinimumWidth(80)
        ch_layout.addWidget(self.dist_channel_0_combo)
        ch_layout.addWidget(QLabel("Colocalize:"))
        self.dist_channel_1_combo = QComboBox()
        self.dist_channel_1_combo.setMinimumWidth(80)
        ch_layout.addWidget(self.dist_channel_1_combo)
        controls.addWidget(ch_group)
        
        # Cell selector
        cell_group = QGroupBox("Cell")
        cell_layout = QHBoxLayout(cell_group)
        cell_layout.setContentsMargins(5, 5, 5, 5)
        self.dist_cell_combo = QComboBox()
        self.dist_cell_combo.addItem("All Cells (pooled)", -1)
        self.dist_cell_combo.addItem("All Cells (per-cell avg)", -2)
        # Connect to re-run analysis when cell selection changes (only if results exist)
        self.dist_cell_combo.currentIndexChanged.connect(self._on_dist_cell_changed)
        cell_layout.addWidget(self.dist_cell_combo)
        controls.addWidget(cell_group)
        
        # Distance threshold
        thresh_group = QGroupBox("Threshold")
        thresh_layout = QHBoxLayout(thresh_group)
        thresh_layout.setContentsMargins(5, 5, 5, 5)
        self.dist_threshold_spinbox = QDoubleSpinBox()
        self.dist_threshold_spinbox.setRange(0.5, 20.0)
        self.dist_threshold_spinbox.setValue(2.0)
        self.dist_threshold_spinbox.setSingleStep(0.5)
        self.dist_threshold_spinbox.setToolTip("Maximum distance in pixels for spots to be considered colocalized")
        self.dist_threshold_spinbox.valueChanged.connect(self.update_distance_nm_label)
        thresh_layout.addWidget(self.dist_threshold_spinbox)
        thresh_layout.addWidget(QLabel("px"))
        self.dist_nm_label = QLabel("= 0.0 nm")
        thresh_layout.addWidget(self.dist_nm_label)
        controls.addWidget(thresh_group)
        
        # 3D checkbox
        self.dist_use_3d_checkbox = QCheckBox("Use 3D")
        self.dist_use_3d_checkbox.setToolTip("Include Z-axis in distance calculation")
        controls.addWidget(self.dist_use_3d_checkbox)
        
        # Actions (Verify is in separate subtab)
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions_group)
        actions_layout.setContentsMargins(5, 5, 5, 5)
        self.dist_run_button = QPushButton("Run")
        self.dist_run_button.clicked.connect(self.run_distance_colocalization)
        actions_layout.addWidget(self.dist_run_button)
        self.dist_export_data_button = QPushButton("Export Data")
        self.dist_export_data_button.clicked.connect(self.export_distance_colocalization_data)
        actions_layout.addWidget(self.dist_export_data_button)
        self.dist_export_image_button = QPushButton("Export Image")
        self.dist_export_image_button.clicked.connect(self.export_distance_colocalization_image)
        actions_layout.addWidget(self.dist_export_image_button)
        controls.addWidget(actions_group)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Results summary (compact)
        self.dist_results_label = QLabel("")
        self.dist_results_label.setAlignment(Qt.AlignCenter)
        self.dist_results_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #00cc66;")
        layout.addWidget(self.dist_results_label)
        
        # Per-cell results - keep but hidden by default for export functionality
        self.dist_percell_group = QGroupBox("Per-Cell Results")
        self.dist_percell_group.setCheckable(True)
        self.dist_percell_group.setChecked(False)
        self.dist_percell_group.setVisible(False)  # Hidden by default
        percell_layout = QVBoxLayout(self.dist_percell_group)
        self.dist_percell_table = QLabel("")
        self.dist_percell_table.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.dist_percell_table.setWordWrap(True)
        percell_layout.addWidget(self.dist_percell_table)
        layout.addWidget(self.dist_percell_group)
        
        # Hidden radio buttons (keep for functionality but don't show)
        self.dist_view_scatter_radio = QRadioButton("Scatter")
        self.dist_view_scatter_radio.setChecked(False)  # Default to overlay
        self.dist_view_scatter_radio.setVisible(False)
        self.dist_view_overlay_radio = QRadioButton("Overlay")
        self.dist_view_overlay_radio.setChecked(True)  # Default to overlay
        self.dist_view_overlay_radio.setVisible(False)
        self.dist_view_scatter_radio.toggled.connect(self.display_distance_colocalization)
        
        # Matplotlib canvas with Z-slider (matching Tracking tab layout)
        self.figure_dist_coloc = Figure(facecolor='black')
        self.canvas_dist_coloc = FigureCanvas(self.figure_dist_coloc)
        self.canvas_dist_coloc.setStyleSheet("background-color: black;")
        
        # Initialize axes for RectangleSelector
        self.ax_dist_coloc = self.figure_dist_coloc.add_subplot(111)
        self.ax_dist_coloc.set_facecolor('black')
        
        # Initialize zoom ROI state
        self.dist_coloc_zoom_roi = None
        
        # Set up zoom feature: RectangleSelector for left-click drag
        self.dist_coloc_zoom_selector = RectangleSelector(
            self.ax_dist_coloc,
            self._on_dist_coloc_zoom_select,
            useblit=True,
            button=[1],  # Left mouse button only
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False,
            props=dict(facecolor='cyan', edgecolor='white', alpha=0.3, linewidth=2)
        )
        
        # Connect double-click to reset zoom
        self.canvas_dist_coloc.mpl_connect('button_press_event', self._on_dist_coloc_canvas_click)
        
        # Create horizontal layout for canvas + Z-slider
        canvas_slider_layout = QHBoxLayout()
        canvas_slider_layout.addWidget(self.canvas_dist_coloc, 1)  # Stretch
        
        # Z-slider with label (vertical, on the right of canvas) - minimal width
        z_slider_container = QWidget()
        z_slider_container.setFixedWidth(40)
        z_slider_layout = QVBoxLayout(z_slider_container)
        z_slider_layout.setContentsMargins(2, 0, 2, 0)
        z_slider_layout.setSpacing(2)
        
        z_label_top = QLabel("Z")
        z_label_top.setAlignment(Qt.AlignCenter)
        z_label_top.setStyleSheet("color: white; font-weight: bold; font-size: 10px;")
        z_slider_layout.addWidget(z_label_top)
        
        # Initialize vertical Z-plane slider for distance colocalization visualization
        self.z_slider_dist_coloc = QSlider(Qt.Vertical, self)
        self.z_slider_dist_coloc.setMinimum(0)
        self.z_slider_dist_coloc.setMaximum(0)  # Will be set when image loads
        self.z_slider_dist_coloc.setTickPosition(QSlider.NoTicks)
        self.z_slider_dist_coloc.setInvertedAppearance(True)  # Top = highest Z index (max projection)
        self.z_slider_dist_coloc.valueChanged.connect(self.update_z_dist_coloc)
        z_slider_layout.addWidget(self.z_slider_dist_coloc, stretch=1)
        
        self.z_label_dist_coloc = QLabel("Max")
        self.z_label_dist_coloc.setAlignment(Qt.AlignCenter)
        self.z_label_dist_coloc.setStyleSheet("color: cyan; font-weight: bold; font-size: 9px;")
        z_slider_layout.addWidget(self.z_label_dist_coloc)
        
        canvas_slider_layout.addWidget(z_slider_container)
        layout.addLayout(canvas_slider_layout, 1)  # Stretch factor for canvas area
        
        # Zoom ROI status label and instructions
        zoom_info_layout = QHBoxLayout()
        zoom_info_layout.setContentsMargins(0, 2, 0, 2)
        
        self.dist_coloc_zoom_label = QLabel("🔍 Full View")
        self.dist_coloc_zoom_label.setStyleSheet("color: #888888; font-size: 10px;")
        zoom_info_layout.addWidget(self.dist_coloc_zoom_label)
        
        zoom_info_layout.addStretch()
        
        zoom_hint_label = QLabel("Click-drag to zoom, Double-click to reset")
        zoom_hint_label.setStyleSheet("color: #555555; font-size: 9px; font-style: italic;")
        zoom_info_layout.addWidget(zoom_hint_label)
        
        layout.addLayout(zoom_info_layout)
        
        # Bottom controls (time slider like tracking)
        bottom_controls = QHBoxLayout()
        bottom_controls.setSpacing(5)
        
        # Play button
        self.dist_play_button = QPushButton("▶")
        self.dist_play_button.setCheckable(True)
        self.dist_play_button.setMaximumWidth(35)
        self.dist_play_button.clicked.connect(self.toggle_distance_playback)
        bottom_controls.addWidget(self.dist_play_button)
        
        # Time slider
        self.dist_frame_slider = QSlider(Qt.Horizontal)
        self.dist_frame_slider.setMinimum(0)
        self.dist_frame_slider.setMaximum(0)
        self.dist_frame_slider.valueChanged.connect(self.on_distance_frame_changed)
        bottom_controls.addWidget(self.dist_frame_slider, 1)  # Stretch
        
        # Frame label
        self.dist_frame_label = QLabel("0/0")
        self.dist_frame_label.setMinimumWidth(50)
        bottom_controls.addWidget(self.dist_frame_label)
        
        layout.addLayout(bottom_controls)
        
        # Timer for playback
        self.dist_play_timer = QTimer(self)
        self.dist_play_timer.timeout.connect(self.advance_distance_frame)
        
        # Initialize distance nm label
        self.update_distance_nm_label()
        
        # Initialize current Z-plane state (-1 = max projection)
        self.dist_coloc_current_z = -1

    def setup_coloc_verify_visual_subtab(self):
        """Setup Verify Visual sub-tab for manual verification of Visual (ML/Intensity) results."""
        layout = QVBoxLayout(self.coloc_verify_visual_widget)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Info label
        info_label = QLabel("Review and correct Visual (ML/Intensity) colocalization results:")
        info_label.setStyleSheet("font-style: italic; color: #999;")
        layout.addWidget(info_label)
        
        # Top bar with stats and buttons
        top_bar = QHBoxLayout()
        self.verify_visual_stats_label = QLabel("Run Visual colocalization first, then click Populate")
        top_bar.addWidget(self.verify_visual_stats_label)
        top_bar.addStretch()
        
        self.verify_visual_populate_button = QPushButton("Populate")
        self.verify_visual_populate_button.clicked.connect(self.populate_verify_visual)
        top_bar.addWidget(self.verify_visual_populate_button)
        
        self.verify_visual_sort_button = QPushButton("Sort")
        self.verify_visual_sort_button.clicked.connect(self.sort_verify_visual)
        top_bar.addWidget(self.verify_visual_sort_button)
        
        self.verify_visual_cleanup_button = QPushButton("Cleanup")
        self.verify_visual_cleanup_button.clicked.connect(self.cleanup_verify_visual)
        top_bar.addWidget(self.verify_visual_cleanup_button)
        
        self.verify_visual_export_button = QPushButton("Export Data")
        self.verify_visual_export_button.clicked.connect(self.export_verify_visual_data)
        top_bar.addWidget(self.verify_visual_export_button)
        
        layout.addLayout(top_bar)
        
        # Scroll area for spot listings
        self.verify_visual_scroll_area = QScrollArea()
        self.verify_visual_scroll_area.setMaximumWidth(400)
        self.verify_visual_scroll_area.setWidgetResizable(True)
        self.verify_visual_scroll_area.setContentsMargins(0, 0, 0, 0)
        placeholder = QWidget()
        self.verify_visual_scroll_area.setWidget(placeholder)
        
        # Center scroll area
        hcenter = QHBoxLayout()
        hcenter.addStretch()
        hcenter.addWidget(self.verify_visual_scroll_area)
        hcenter.addStretch()
        layout.addLayout(hcenter)
        
        # Initialize checkbox list
        self.verify_visual_checkboxes = []
    
    def setup_coloc_verify_distance_subtab(self):
        """Setup Verify Distance sub-tab for manual verification of Distance-based results."""
        layout = QVBoxLayout(self.coloc_verify_distance_widget)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Info label
        info_label = QLabel("Review and correct Distance-based colocalization results:")
        info_label.setStyleSheet("font-style: italic; color: #999;")
        layout.addWidget(info_label)
        
        # Top bar with stats and buttons
        top_bar = QHBoxLayout()
        self.verify_distance_stats_label = QLabel("Run Distance colocalization first, then click Populate")
        top_bar.addWidget(self.verify_distance_stats_label)
        top_bar.addStretch()
        
        self.verify_distance_populate_button = QPushButton("Populate")
        self.verify_distance_populate_button.clicked.connect(self.populate_verify_distance)
        top_bar.addWidget(self.verify_distance_populate_button)
        
        self.verify_distance_sort_button = QPushButton("Sort")
        self.verify_distance_sort_button.clicked.connect(self.sort_verify_distance)
        top_bar.addWidget(self.verify_distance_sort_button)
        
        self.verify_distance_cleanup_button = QPushButton("Cleanup")
        self.verify_distance_cleanup_button.clicked.connect(self.cleanup_verify_distance)
        top_bar.addWidget(self.verify_distance_cleanup_button)
        
        self.verify_distance_export_button = QPushButton("Export Data")
        self.verify_distance_export_button.clicked.connect(self.export_verify_distance_data)
        top_bar.addWidget(self.verify_distance_export_button)
        
        layout.addLayout(top_bar)
        
        # Scroll area for spot listings
        self.verify_distance_scroll_area = QScrollArea()
        self.verify_distance_scroll_area.setMaximumWidth(400)
        self.verify_distance_scroll_area.setWidgetResizable(True)
        self.verify_distance_scroll_area.setContentsMargins(0, 0, 0, 0)
        placeholder = QWidget()
        self.verify_distance_scroll_area.setWidget(placeholder)
        
        # Center scroll area
        hcenter = QHBoxLayout()
        hcenter.addStretch()
        hcenter.addWidget(self.verify_distance_scroll_area)
        hcenter.addStretch()
        layout.addLayout(hcenter)
        
        # Initialize checkbox list
        self.verify_distance_checkboxes = []
    
    # === Distance Colocalization Zoom Handlers ===
    
    def _on_dist_coloc_zoom_select(self, eclick, erelease):
        """Handle ROI selection from left-click drag on distance colocalization canvas."""
        if self.image_stack is None:
            return
        
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Handle None values (click outside axes)
        if x1 is None or x2 is None or y1 is None or y2 is None:
            return
        
        # Calculate ROI bounds
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Enforce minimum ROI size (50x50 pixels)
        if (x_max - x_min) < 50 or (y_max - y_min) < 50:
            return
        
        # Clamp to image bounds
        _, _, H, W, _ = self.image_stack.shape
        x_min = max(0, x_min)
        x_max = min(W, x_max)
        y_min = max(0, y_min)
        y_max = min(H, y_max)
        
        # Store ROI
        self.dist_coloc_zoom_roi = (x_min, x_max, y_min, y_max)
        
        # Update label
        if hasattr(self, 'dist_coloc_zoom_label'):
            self.dist_coloc_zoom_label.setText(f"🔍 ROI: X[{int(x_min)}:{int(x_max)}] Y[{int(y_min)}:{int(y_max)}]")
            self.dist_coloc_zoom_label.setStyleSheet("color: #00d4aa; font-size: 10px; font-weight: bold;")
        
        # Redraw with zoom
        self.display_distance_colocalization()

    def _on_dist_coloc_canvas_click(self, event):
        """Handle mouse clicks on distance coloc canvas - double-click to reset zoom."""
        if event.dblclick:
            self._reset_dist_coloc_zoom()

    def _reset_dist_coloc_zoom(self):
        """Reset zoom to show full image."""
        self.dist_coloc_zoom_roi = None
        
        # Update label
        if hasattr(self, 'dist_coloc_zoom_label'):
            self.dist_coloc_zoom_label.setText("🔍 Full View")
            self.dist_coloc_zoom_label.setStyleSheet("color: #888888; font-size: 10px;")
        
        # Redraw without zoom
        self.display_distance_colocalization()
    
    def update_z_dist_coloc(self, value):
        """Handle Z-slider value change for Distance Colocalization tab.
        
        When slider is at max (Z), show max projection with all spots.
        When slider is at specific value (0 to Z-1), show that Z-plane
        and only spots from that plane (for 3D tracking).
        """
        # Update current Z state
        if hasattr(self, 'z_slider_dist_coloc'):
            max_val = self.z_slider_dist_coloc.maximum()
            if value == max_val:
                self.dist_coloc_current_z = -1  # Max projection
                if hasattr(self, 'z_label_dist_coloc'):
                    self.z_label_dist_coloc.setText("Max")
                    self.z_label_dist_coloc.setStyleSheet("color: cyan; font-weight: bold;")
            else:
                self.dist_coloc_current_z = value
                if hasattr(self, 'z_label_dist_coloc'):
                    self.z_label_dist_coloc.setText(f"Z={value}")
                    self.z_label_dist_coloc.setStyleSheet("color: lime; font-weight: bold;")
        
        # Redraw with new Z-plane
        self.display_distance_colocalization()
    
    def reset_dist_coloc_z_slider(self):
        """Reset Z-slider to default (max projection) and update from image dimensions."""
        if not hasattr(self, 'z_slider_dist_coloc'):
            return
            
        if self.image_stack is None:
            self.z_slider_dist_coloc.setMaximum(0)
            self.z_slider_dist_coloc.setValue(0)
            self.z_slider_dist_coloc.setEnabled(False)
            self.dist_coloc_current_z = -1
            return
        
        # Get Z dimension from image (TZYXC format, Z is axis 1)
        z_dim = self.image_stack.shape[1]
        
        self.z_slider_dist_coloc.blockSignals(True)
        self.z_slider_dist_coloc.setMinimum(0)
        if z_dim > 1:
            self.z_slider_dist_coloc.setMaximum(z_dim)  # max = "Max Projection" position
        else:
            self.z_slider_dist_coloc.setMaximum(0)
        self.z_slider_dist_coloc.setValue(z_dim if z_dim > 1 else 0)  # Default to max projection
        self.z_slider_dist_coloc.blockSignals(False)
        
        # Enable slider only if more than 1 Z-slice
        self.z_slider_dist_coloc.setEnabled(z_dim > 1)
        
        self.dist_coloc_current_z = -1  # Default to max projection
        
        if hasattr(self, 'z_label_dist_coloc'):
            self.z_label_dist_coloc.setText("Max")
            self.z_label_dist_coloc.setStyleSheet("color: cyan; font-weight: bold;")
    
    # === Distance Colocalization Helper Methods (Placeholder) ===
    
    def update_distance_nm_label(self):
        """Update the nm equivalent label based on current threshold and pixel size."""
        threshold_px = self.dist_threshold_spinbox.value() if hasattr(self, 'dist_threshold_spinbox') else 2.0
        # Use voxel_yx_nm which is set from the Import tab (in nanometers)
        pixel_size_nm = getattr(self, 'voxel_yx_nm', None)
        if pixel_size_nm is None or np.isnan(pixel_size_nm) or pixel_size_nm <= 0:
            pixel_size_nm = 130.0  # Default: 130 nm per pixel
        threshold_nm = threshold_px * pixel_size_nm
        if hasattr(self, 'dist_nm_label'):
            self.dist_nm_label.setText(f"= {threshold_nm:.1f} nm")
    
    def populate_distance_channel_combos(self):
        """Populate distance colocalization channel combos with tracked channels."""
        if not hasattr(self, 'dist_channel_0_combo') or not hasattr(self, 'dist_channel_1_combo'):
            return
        
        self.dist_channel_0_combo.clear()
        self.dist_channel_1_combo.clear()
        
        # Get tracked channels from df_tracking
        if hasattr(self, 'df_tracking') and not self.df_tracking.empty and 'spot_type' in self.df_tracking.columns:
            tracked_channels = sorted(self.df_tracking['spot_type'].unique().tolist())
            for ch in tracked_channels:
                self.dist_channel_0_combo.addItem(f"Ch {ch}", ch)
                self.dist_channel_1_combo.addItem(f"Ch {ch}", ch)
            
            # Default: select different channels if possible
            if len(tracked_channels) >= 2:
                self.dist_channel_0_combo.setCurrentIndex(0)
                self.dist_channel_1_combo.setCurrentIndex(1)
        else:
            self.dist_channel_0_combo.addItem("No tracked channels", -1)
            self.dist_channel_1_combo.addItem("No tracked channels", -1)
        
        # Also populate cell selector
        self._populate_dist_cell_selector()
        
        # Update 3D checkbox availability based on Z planes
        self._update_distance_3d_availability()
        
        # Update nm label
        self.update_distance_nm_label()
    
    def _populate_dist_cell_selector(self):
        """Populate the distance colocalization cell selector."""
        if not hasattr(self, 'dist_cell_combo'):
            return
        
        current_data = self.dist_cell_combo.currentData()
        self.dist_cell_combo.clear()
        self.dist_cell_combo.addItem("All Cells (pooled)", -1)
        self.dist_cell_combo.addItem("All Cells (per-cell avg)", -2)
        
        # Add individual cells if available
        if hasattr(self, 'df_tracking') and not self.df_tracking.empty and 'cell_id' in self.df_tracking.columns:
            cell_ids = sorted(self.df_tracking['cell_id'].unique().tolist())
            for cid in cell_ids:
                self.dist_cell_combo.addItem(f"Cell {cid}", cid)
        
        # Restore previous selection if possible
        if current_data is not None:
            idx = self.dist_cell_combo.findData(current_data)
            if idx >= 0:
                self.dist_cell_combo.setCurrentIndex(idx)
    
    def _update_distance_3d_availability(self):
        """Enable/disable 3D checkbox based on available Z planes."""
        if not hasattr(self, 'dist_use_3d_checkbox'):
            return
        
        # Check if we have multiple Z planes
        has_3d = False
        if hasattr(self, 'df_tracking') and not self.df_tracking.empty and 'z' in self.df_tracking.columns:
            z_values = self.df_tracking['z'].unique()
            has_3d = len(z_values) > 1
        
        self.dist_use_3d_checkbox.setEnabled(has_3d)
        if not has_3d:
            self.dist_use_3d_checkbox.setChecked(False)
            self.dist_use_3d_checkbox.setToolTip("3D requires multiple Z planes (only 1 Z plane detected)")
        else:
            self.dist_use_3d_checkbox.setToolTip("Include Z-axis in distance calculation (uses voxel size for scaling)")
    
    def _on_dist_cell_changed(self, index):
        """Handle cell selection change in Distance colocalization - re-run analysis if results exist."""
        # Only re-run if we already have results (user has run analysis at least once)
        if hasattr(self, 'distance_coloc_results') and self.distance_coloc_results is not None:
            self.run_distance_colocalization()
    
    def run_distance_colocalization(self):
        """Run distance-based colocalization analysis using the ColocalizationDistance class."""
        if not getattr(self, 'has_tracked', False) or self.df_tracking.empty:
            QMessageBox.warning(self, "No Tracking Data", 
                                "Please run tracking on at least 2 channels first.")
            return
        
        # Get tracked channels from df_tracking
        tracked_channels = sorted(self.df_tracking['spot_type'].unique().tolist())
        if len(tracked_channels) < 2:
            QMessageBox.warning(self, "Insufficient Channels",
                                f"Distance colocalization requires ≥2 tracked channels.\n"
                                f"Found: {len(tracked_channels)} channel(s).")
            return
        
        # Get parameters from UI
        ch0 = self.dist_channel_0_combo.currentData()
        ch1 = self.dist_channel_1_combo.currentData()
        
        if ch0 is None or ch1 is None or ch0 == -1 or ch1 == -1:
            QMessageBox.warning(self, "Invalid Channels", "Please select valid channels.")
            return
        
        if ch0 == ch1:
            QMessageBox.warning(self, "Same Channel", "Please select two different channels.")
            return
        
        threshold_px = self.dist_threshold_spinbox.value()
        use_3d = self.dist_use_3d_checkbox.isChecked()
        cell_selection = self.dist_cell_combo.currentData()
        
        # Calculate nm equivalent using validated voxel sizes
        pixel_size_nm = getattr(self, 'voxel_yx_nm', None)
        if pixel_size_nm is None or np.isnan(pixel_size_nm) or pixel_size_nm <= 0:
            pixel_size_nm = 130.0  # Default: 130 nm per pixel
        threshold_nm = threshold_px * pixel_size_nm
        
        # Prepare voxel sizes for 3D
        # To properly handle anisotropic voxels, we need to scale Z coordinates
        # relative to XY. ColocalizationDistance uses: scale[0] = voxel_z / psf_z
        # We want scale[0] = voxel_z / voxel_xy to match the MSD approach.
        # So we set psf_z = voxel_xy and psf_yx = voxel_xy.
        # This makes: scale = [voxel_z/voxel_xy, voxel_xy/voxel_xy, voxel_xy/voxel_xy]
        #                   = [voxel_z/voxel_xy, 1, 1]
        # Z is scaled by ratio of voxel sizes, XY unchanged.
        voxel_z = None
        psf_z = None
        voxel_xy = None
        psf_yx = None
        
        if use_3d:
            # Get Z voxel size in nm
            voxel_z_nm = getattr(self, 'voxel_z_nm', None)
            if voxel_z_nm is None or np.isnan(voxel_z_nm) or voxel_z_nm <= 0:
                voxel_z_nm = 300.0  # Default: 300 nm per z-slice
            voxel_z = voxel_z_nm
            voxel_xy = pixel_size_nm
            
            # For anisotropic scaling, we want Z in "XY-equivalent pixels"
            # Set psf_z = voxel_xy so that scale[0] = voxel_z / voxel_xy
            # This scales Z coordinates by the ratio of voxel sizes
            psf_z = voxel_xy  # NOT voxel_z - this gives us the correct ratio
            psf_yx = voxel_xy  # XY stays at 1:1
        
        # Filter df_tracking based on cell selection if needed
        df = self.df_tracking.copy()
        if cell_selection is not None and cell_selection >= 0:
            df = df[df['cell_id'] == cell_selection]
        
        if df.empty:
            QMessageBox.warning(self, "No Data", "No spots found for the selected cell.")
            return
        
        # Run ColocalizationDistance
        try:
            from microlive.microscopy import ColocalizationDistance
            coloc = ColocalizationDistance(
                df=df,
                list_spot_type_to_compare=[ch0, ch1],
                threshold_distance=threshold_px,
                voxel_size_z=voxel_z,
                psf_z=psf_z,
                voxel_size_yx=voxel_xy,
                psf_yx=psf_yx,
                show_plot=False,
                report_codetected_spots_in_both_channels=False
            )
            
            (df_classification, df_colocalized, df_ch0_only, 
             df_ch1_only, df_ch0_all, df_ch1_all) = coloc.extract_spot_classification_from_df()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Distance colocalization failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        # Store results
        self.distance_coloc_results = {
            'method': 'distance',
            'channel_0': ch0,
            'channel_1': ch1,
            'threshold_distance_px': threshold_px,
            'threshold_distance_nm': threshold_nm,
            'use_3d': use_3d,
            'voxel_z_nm': voxel_z,
            'voxel_xy_nm': voxel_xy,
            'cell_selection': cell_selection,
            'df_classification': df_classification,
            'df_colocalized': df_colocalized,
            'df_ch0_only': df_ch0_only,
            'df_ch1_only': df_ch1_only,
            'df_ch0_all': df_ch0_all,
            'df_ch1_all': df_ch1_all
        }
        
        # Update frame slider for time-lapse
        if hasattr(self, 'image_stack') and self.image_stack is not None:
            num_frames = self.image_stack.shape[0] if self.image_stack.ndim >= 3 else 1
            self.dist_frame_slider.setMaximum(max(0, num_frames - 1))
            self.dist_frame_slider.setValue(self.current_frame)
            self.dist_frame_label.setText(f"Frame: {self.current_frame}/{num_frames - 1}")
        
        # Initialize Z-slider based on image dimensions
        self.reset_dist_coloc_z_slider()
        
        # Reset zoom state
        self._reset_dist_coloc_zoom()
        
        # Update display
        self.display_distance_colocalization()
    
    def export_distance_colocalization_data(self):
        """Export distance colocalization data to CSV."""
        if not hasattr(self, 'distance_coloc_results') or self.distance_coloc_results is None:
            QMessageBox.warning(self, "No Results", 
                                "Please run Distance colocalization first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Distance Colocalization Data", "", "CSV Files (*.csv)")
        
        if not file_path:
            return
        
        try:
            results = self.distance_coloc_results
            df_class = results['df_classification'].copy()
            
            # Add metadata columns
            df_class['method'] = 'distance'
            df_class['threshold_px'] = results['threshold_distance_px']
            df_class['threshold_nm'] = results['threshold_distance_nm']
            df_class['use_3d'] = results['use_3d']
            df_class['channel_0'] = results['channel_0']
            df_class['channel_1'] = results['channel_1']
            
            df_class.to_csv(file_path, index=False)
            QMessageBox.information(self, "Export Complete", 
                                    f"Distance colocalization data exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error exporting data:\n{str(e)}")
    
    def export_distance_colocalization_image(self):
        """Export distance colocalization visualization."""
        if not hasattr(self, 'distance_coloc_results') or self.distance_coloc_results is None:
            QMessageBox.warning(self, "No Results", 
                                "Please run Distance colocalization first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Distance Colocalization Image", "", "PNG Files (*.png);;PDF Files (*.pdf)")
        
        if not file_path:
            return
        
        try:
            self.figure_dist_coloc.savefig(file_path, dpi=150, bbox_inches='tight', 
                                           facecolor='black', edgecolor='none')
            QMessageBox.information(self, "Export Complete", 
                                    f"Image exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error exporting image:\n{str(e)}")
    
    def display_distance_colocalization(self):
        """Display distance colocalization results."""
        if not hasattr(self, 'distance_coloc_results') or self.distance_coloc_results is None:
            # Show placeholder with tracking-like black background
            fig = self.figure_dist_coloc
            fig.clear()
            fig.patch.set_facecolor('black')
            ax = fig.add_subplot(111)
            ax.set_facecolor('black')
            ax.axis('off')
            ax.text(0.5, 0.5, 'Run distance colocalization to see results.',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, color='#666666', transform=ax.transAxes)
            fig.tight_layout(pad=0)
            self.canvas_dist_coloc.draw()
            return
        
        results = self.distance_coloc_results
        df_class = results['df_classification']
        ch0, ch1 = results['channel_0'], results['channel_1']
        
        # Calculate totals
        total_coloc = int(df_class['num_0_1'].sum())
        total_ch0_only = int(df_class['num_0_only'].sum())
        total_ch1_only = int(df_class['num_1_only'].sum())
        total_ch0 = int(df_class['num_0_total'].sum())
        total_ch1 = int(df_class['num_1_total'].sum())
        total_unique = total_ch0_only + total_ch1_only + total_coloc
        pct_coloc = 100 * total_coloc / total_unique if total_unique > 0 else 0
        
        # Update summary label
        self.dist_results_label.setText(
            f"Colocalized: {total_coloc}/{total_unique} ({pct_coloc:.1f}%)  |  "
            f"Ch{ch0} only: {total_ch0_only} ({100*total_ch0_only/total_unique:.1f}%)  |  "
            f"Ch{ch1} only: {total_ch1_only} ({100*total_ch1_only/total_unique:.1f}%)"
        )
        
        # Update per-cell table
        self._update_distance_percell_table(df_class, ch0, ch1)
        
        # Draw visualization
        is_scatter = self.dist_view_scatter_radio.isChecked()
        if is_scatter:
            self._draw_distance_scatter()
        else:
            self._draw_distance_overlay()
    
    def _update_distance_percell_table(self, df_class, ch0, ch1):
        """Update the per-cell results table for distance colocalization."""
        if not hasattr(self, 'dist_percell_table'):
            return
        
        # Build table text
        lines = [f"{'Cell':>6} | {'Ch'+str(ch0)+' Only':>10} | {'Ch'+str(ch1)+' Only':>10} | {'Colocalized':>11} | {'Total':>6} | {'% Coloc':>8}"]
        lines.append("-" * 70)
        
        for _, row in df_class.iterrows():
            cell_id = int(row['cell_id'])
            ch0_only = int(row['num_0_only'])
            ch1_only = int(row['num_1_only'])
            coloc = int(row['num_0_1'])
            total = ch0_only + ch1_only + coloc
            pct = 100 * coloc / total if total > 0 else 0
            lines.append(f"{cell_id:>6} | {ch0_only:>10} | {ch1_only:>10} | {coloc:>11} | {total:>6} | {pct:>7.1f}%")
        
        self.dist_percell_table.setText("\n".join(lines))
    
    def _draw_distance_scatter(self):
        """Draw scatter plot of spot classifications."""
        fig = self.figure_dist_coloc
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1a1a1a')
        
        results = self.distance_coloc_results
        ch0, ch1 = results['channel_0'], results['channel_1']
        
        # Get coordinate dataframes
        df_ch0 = results['df_ch0_only']
        df_ch1 = results['df_ch1_only']
        df_both = results['df_colocalized']
        
        # Get colors for each channel
        color_ch0 = self._get_channel_color_for_scatter(ch0)
        color_ch1 = self._get_channel_color_for_scatter(ch1)
        
        # Plot each category
        if not df_ch0.empty:
            ax.scatter(df_ch0['x'], df_ch0['y'], c=[color_ch0], 
                       s=20, alpha=0.7, label=f'Ch{ch0} only ({len(df_ch0)})', 
                       marker='o', edgecolors='none')
        if not df_ch1.empty:
            ax.scatter(df_ch1['x'], df_ch1['y'], c=[color_ch1],
                       s=20, alpha=0.7, label=f'Ch{ch1} only ({len(df_ch1)})',
                       marker='o', edgecolors='none')
        if not df_both.empty:
            ax.scatter(df_both['x'], df_both['y'], c='white', edgecolors='yellow',
                       s=40, alpha=0.9, label=f'Colocalized ({len(df_both)})',
                       marker='o', linewidths=1)
        
        ax.set_xlabel('X (pixels)', color='white')
        ax.set_ylabel('Y (pixels)', color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', facecolor='#333333', edgecolor='white', 
                  labelcolor='white', fontsize=9)
        ax.invert_yaxis()  # Image coordinates
        ax.set_aspect('equal')
        
        # Add title with threshold info
        threshold_px = results['threshold_distance_px']
        threshold_nm = results['threshold_distance_nm']
        use_3d = results['use_3d']
        dim_str = "3D" if use_3d else "2D"
        ax.set_title(f"Distance Colocalization ({dim_str}): Threshold = {threshold_px:.1f} px ({threshold_nm:.0f} nm)",
                     color='white', fontsize=11)
        
        fig.tight_layout()
        self.canvas_dist_coloc.draw()
    
    def _draw_distance_overlay(self):
        """Draw image overlay with spot markers for the current frame (tracking-like style)."""
        # Early return if no results
        if not hasattr(self, 'distance_coloc_results') or self.distance_coloc_results is None:
            return
        
        fig = self.figure_dist_coloc
        fig.clear()
        fig.patch.set_facecolor('black')
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        
        # Re-initialize RectangleSelector on new axes
        self.ax_dist_coloc = ax
        self.dist_coloc_zoom_selector = RectangleSelector(
            ax,
            self._on_dist_coloc_zoom_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False,
            props=dict(facecolor='cyan', edgecolor='white', alpha=0.3, linewidth=2)
        )
        
        results = self.distance_coloc_results
        ch0, ch1 = results['channel_0'], results['channel_1']
        
        # Get current frame image
        frame_idx = self.dist_frame_slider.value() if hasattr(self, 'dist_frame_slider') else self.current_frame
        img_src = self.get_current_image_source()
        
        if img_src is None:
            ax.text(0.5, 0.5, 'No image available.',
                    ha='center', va='center', fontsize=12, color='white', transform=ax.transAxes)
            self.canvas_dist_coloc.draw()
            return
        
        # Get current Z-plane selection (-1 = max projection)
        current_z = getattr(self, 'dist_coloc_current_z', -1)
        
        # Get frame projection or specific Z-plane
        if img_src.ndim == 5:  # TZYXC format
            if current_z == -1:
                # Max Z projection
                proj = np.max(img_src[frame_idx], axis=0)
            else:
                # Specific Z-plane
                z_idx = min(current_z, img_src.shape[1] - 1)
                proj = img_src[frame_idx, z_idx]
        elif img_src.ndim == 4:  # TYXC
            proj = img_src[frame_idx]
        else:
            proj = img_src
        
        # Normalize and create RGB for display
        if proj.ndim == 2:
            proj = proj[:, :, np.newaxis]
        
        # Create merged RGB view of the two channels
        H, W, C = proj.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        
        # Channel colors
        color_ch0 = self._get_channel_color_for_scatter(ch0)
        color_ch1 = self._get_channel_color_for_scatter(ch1)
        
        # Use percentile-based scaling like Tracking tab (not min-max which always fills full range)
        # Brightness factor: 0.4 = 40% brightness (allows spots/markers to stand out)
        brightness = 0.4
        
        if ch0 < C:
            ch0_data = proj[:, :, ch0].astype(np.float32)
            # Percentile-based normalization (1st-99th percentile)
            p_low, p_high = np.percentile(ch0_data[ch0_data > 0], [1, 99]) if np.any(ch0_data > 0) else (0, 1)
            ch0_norm = np.clip((ch0_data - p_low) / (p_high - p_low + 1e-8), 0, 1)
            for i, c in enumerate(color_ch0[:3]):
                rgb[:, :, i] += ch0_norm * c * brightness
        
        if ch1 < C:
            ch1_data = proj[:, :, ch1].astype(np.float32)
            # Percentile-based normalization
            p_low, p_high = np.percentile(ch1_data[ch1_data > 0], [1, 99]) if np.any(ch1_data > 0) else (0, 1)
            ch1_norm = np.clip((ch1_data - p_low) / (p_high - p_low + 1e-8), 0, 1)
            for i, c in enumerate(color_ch1[:3]):
                rgb[:, :, i] += ch1_norm * c * brightness
        
        rgb = np.clip(rgb, 0, 1)
        
        ax.imshow(rgb)
        
        # Apply zoom if ROI is set
        if self.dist_coloc_zoom_roi is not None:
            x_min, x_max, y_min, y_max = self.dist_coloc_zoom_roi
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)  # Inverted for image coordinates
        
        # Filter spots by current frame using the original tracking dataframe
        # Get all spots for this frame from the tracking dataframe
        df_track = self.df_tracking.copy()
        df_frame = df_track[df_track['frame'] == frame_idx]
        
        # Filter by cell selection if a specific cell is selected
        cell_selection = self.dist_cell_combo.currentData() if hasattr(self, 'dist_cell_combo') else -1
        if cell_selection is not None and cell_selection >= 0 and 'cell_id' in df_frame.columns:
            df_frame = df_frame[df_frame['cell_id'] == cell_selection]
        
        # Filter by Z-plane if not in max projection mode
        if current_z != -1 and 'z' in df_frame.columns:
            # Filter spots to current Z-plane (with some tolerance for subpixel positions)
            z_tolerance = 0.5
            df_frame = df_frame[np.abs(df_frame['z'] - current_z) <= z_tolerance]
        
        # Get colocalized coordinates (from all frames - we'll match by position)
        df_coloc_all = results['df_colocalized']
        
        # Separate by channel and determine colocalization status for this frame
        df_ch0_frame = df_frame[df_frame['spot_type'] == ch0]
        df_ch1_frame = df_frame[df_frame['spot_type'] == ch1]
        
        # For each spot in this frame, check if it was marked as colocalized
        # We do this by checking if the spot's coordinates match any in the colocalized set
        threshold = results['threshold_distance_px']
        
        def is_colocalized(x, y, z, df_coloc):
            """Check if a spot at (x,y,z) is in the colocalized set (within 1px tolerance)."""
            if df_coloc.empty:
                return False
            # Use a small tolerance for matching
            matches = df_coloc[
                (np.abs(df_coloc['x'] - x) < 1.5) & 
                (np.abs(df_coloc['y'] - y) < 1.5) &
                (np.abs(df_coloc['z'] - z) < 1.5)
            ]
            return len(matches) > 0
        
        # Classify spots in this frame
        spots_ch0_only = []  # (x, y) for ch0 only
        spots_ch1_only = []  # (x, y) for ch1 only  
        spots_colocalized = []  # (x, y) for colocalized
        
        for _, row in df_ch0_frame.iterrows():
            x, y, z = row['x'], row['y'], row.get('z', 0)
            if is_colocalized(x, y, z, df_coloc_all):
                spots_colocalized.append((x, y))
            else:
                spots_ch0_only.append((x, y))
        
        for _, row in df_ch1_frame.iterrows():
            x, y, z = row['x'], row['y'], row.get('z', 0)
            if not is_colocalized(x, y, z, df_coloc_all):
                spots_ch1_only.append((x, y))
            # Note: colocalized spots already added from ch0
        
        # Calculate dynamic marker size based on zoom level (V-curve pattern from Tracking tab)
        # This makes markers smaller at full view, larger when zoomed in
        zoom_scale = 1.0
        if self.dist_coloc_zoom_roi is not None:
            x_min, x_max, y_min, y_max = self.dist_coloc_zoom_roi
            full_width = W
            visible_width = x_max - x_min
            if full_width > 0 and visible_width > 0:
                zoom_ratio = visible_width / full_width
                # Invert scale: full view (ratio=1) gets smaller markers, zoomed (ratio=0.1) gets larger
                zoom_scale = max(0.4, min(1.0, 1.0 - (zoom_ratio ** 0.5) * 0.6))
        else:
            # Full view: scale based on image size (smaller images get smaller markers)
            image_scale = min(1.0, max(0.3, (W / 2048.0) ** 0.5))
            zoom_scale = image_scale
        
        # Plot markers - dynamic size like tracking tab (unfilled circles/squares)
        dpi = fig.get_dpi()
        marker_scale = dpi / 100.0
        base_marker_size = 20  # Base size in points^2
        marker_size = base_marker_size * marker_scale * zoom_scale
        linewidth = max(0.3, 1.0 * zoom_scale)  # Thinner at full view, thicker when zoomed
        
        if spots_ch0_only:
            xs, ys = zip(*spots_ch0_only)
            ax.scatter(xs, ys, facecolors='none', edgecolors=color_ch0, 
                       s=marker_size, alpha=0.9, marker='s', linewidths=linewidth)
        if spots_ch1_only:
            xs, ys = zip(*spots_ch1_only)
            ax.scatter(xs, ys, facecolors='none', edgecolors=color_ch1,
                       s=marker_size, alpha=0.9, marker='s', linewidths=linewidth)
        if spots_colocalized:
            xs, ys = zip(*spots_colocalized)
            ax.scatter(xs, ys, facecolors='none', edgecolors='yellow',
                       s=marker_size * 1.2, alpha=1.0, marker='o', linewidths=linewidth * 1.5)
        
        ax.axis('off')
        
        # Add compact legend in upper right corner
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor=color_ch0,
                   markersize=6, label=f'Ch{ch0} only ({len(spots_ch0_only)})', linestyle='None', markeredgewidth=1),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor=color_ch1,
                   markersize=6, label=f'Ch{ch1} only ({len(spots_ch1_only)})', linestyle='None', markeredgewidth=1),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='yellow', markersize=7, label=f'Colocalized ({len(spots_colocalized)})', linestyle='None', markeredgewidth=1.5)
        ]
        ax.legend(handles=legend_elements, loc='upper right', facecolor='#333333', 
                  edgecolor='none', labelcolor='white', fontsize=8)
        
        # Update frame label
        total_frames = getattr(self, 'total_frames', 1)
        self.dist_frame_label.setText(f"{frame_idx}/{total_frames - 1}")
        
        fig.tight_layout(pad=0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Re-apply zoom limits AFTER tight_layout (which can reset them)
        if self.dist_coloc_zoom_roi is not None:
            x_min, x_max, y_min, y_max = self.dist_coloc_zoom_roi
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)  # Inverted for image coordinates
        
        self.canvas_dist_coloc.draw()
    
    def _get_channel_color_for_scatter(self, channel_idx):
        """Get a color for a given channel index for scatter plots."""
        # Use the same channel colors as tracking visualization
        channel_colors = [
            (0.2, 1.0, 0.2),   # Ch0: Green
            (1.0, 0.2, 0.2),   # Ch1: Red
            (0.2, 0.6, 1.0),   # Ch2: Blue
            (1.0, 1.0, 0.2),   # Ch3: Yellow
            (1.0, 0.2, 1.0),   # Ch4: Magenta
            (0.2, 1.0, 1.0),   # Ch5: Cyan
        ]
        if channel_idx < len(channel_colors):
            return channel_colors[channel_idx]
        return (0.7, 0.7, 0.7)  # Default gray
    
    def toggle_distance_playback(self):
        """Toggle play/pause for distance colocalization time-lapse."""
        if self.dist_play_button.isChecked():
            self.dist_play_button.setText("⏸")
            self.dist_play_timer.start(200)  # 5 fps
        else:
            self.dist_play_button.setText("▶")
            self.dist_play_timer.stop()
    
    def advance_distance_frame(self):
        """Advance to next frame during playback."""
        if hasattr(self, 'dist_frame_slider'):
            current = self.dist_frame_slider.value()
            max_val = self.dist_frame_slider.maximum()
            if current < max_val:
                self.dist_frame_slider.setValue(current + 1)
            else:
                self.dist_frame_slider.setValue(0)  # Loop
    
    def on_distance_frame_changed(self, value):
        """Handle frame slider change for distance visualization."""
        if hasattr(self, 'dist_frame_label') and hasattr(self, 'dist_frame_slider'):
            max_val = self.dist_frame_slider.maximum()
            self.dist_frame_label.setText(f"Frame: {value}/{max_val}")
        
        # Re-run analysis for this frame if overlay mode
        if self.dist_view_overlay_radio.isChecked() and hasattr(self, 'distance_coloc_results') and self.distance_coloc_results is not None:
            self._draw_distance_overlay()
    
    # === Verify Visual Subtab Methods ===
    
    def populate_verify_visual(self):
        """Populate the Verify Visual subtab with Visual (ML/Intensity) colocalization results."""
        if not hasattr(self, 'colocalization_results') or not self.colocalization_results:
            QMessageBox.warning(self, "No Results", 
                "Please run Visual (ML/Intensity) colocalization first.")
            return
        
        # Get results
        results = self.colocalization_results
        flag_vector = results.get('flag_vector')
        mean_crop = results.get('mean_crop_filtered')
        crop_size = results.get('crop_size', 15)
        ch1 = results.get('ch1_index', 0)
        ch2 = results.get('ch2_index', 1)
        
        if flag_vector is None or mean_crop is None:
            QMessageBox.warning(self, "No Data", "Visual colocalization results are incomplete.")
            return
        
        # Create spot crops with checkboxes
        self._create_verification_crops(
            scroll_area=self.verify_visual_scroll_area,
            checkboxes_list_attr='verify_visual_checkboxes',
            mean_crop=mean_crop,
            crop_size=crop_size,
            flag_vector=flag_vector,
            stats_label=self.verify_visual_stats_label,
            num_channels=2,
            channels=(ch1, ch2)
        )
        
        # Update stats label
        self._update_verify_visual_stats()
    
    def _update_verify_visual_stats(self):
        """Update the stats label for Verify Visual subtab."""
        if not hasattr(self, 'verify_visual_checkboxes'):
            return
        total = len(self.verify_visual_checkboxes)
        marked = sum(1 for chk in self.verify_visual_checkboxes if chk.isChecked())
        pct = 100 * marked / total if total > 0 else 0
        
        method = "ML" if hasattr(self, 'method_ml_radio') and self.method_ml_radio.isChecked() else "Intensity"
        self.verify_visual_stats_label.setText(
            f"[{method}] Total: {total} | Colocalized: {marked} ({pct:.1f}%)"
        )
    
    def sort_verify_visual(self):
        """Sort Verify Visual results by prediction value (lowest to highest)."""
        if not hasattr(self, 'verify_visual_checkboxes') or len(self.verify_visual_checkboxes) == 0:
            return
        
        values = self.colocalization_results.get('prediction_values_vector') if hasattr(self, 'colocalization_results') else None
        if values is None or len(values) == 0:
            QMessageBox.information(self, "Cannot Sort", "No prediction values available for sorting.")
            return
        
        # Re-populate with sorted order would require rebuilding crops
        # For now, show a message that sorting is based on visual arrangement
        QMessageBox.information(self, "Sort", 
            "Spots are already displayed in their original detection order. "
            "Lower prediction values indicate uncertain colocalization.")
    
    def cleanup_verify_visual(self):
        """Clear all checkboxes in Verify Visual subtab."""
        if not hasattr(self, 'verify_visual_checkboxes'):
            return
        for checkbox in self.verify_visual_checkboxes:
            checkbox.setChecked(False)
        self._update_verify_visual_stats()
    
    def export_verify_visual_data(self):
        """Export Verify Visual results to CSV."""
        if not hasattr(self, 'verify_visual_checkboxes') or len(self.verify_visual_checkboxes) == 0:
            QMessageBox.warning(self, "No Data", "No verification data to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Visual Verification Data", "", "CSV Files (*.csv)")
        
        if not file_path:
            return
        
        try:
            total = len(self.verify_visual_checkboxes)
            flags = [chk.isChecked() for chk in self.verify_visual_checkboxes]
            marked = sum(flags)
            
            results = self.colocalization_results if hasattr(self, 'colocalization_results') else {}
            
            df = pd.DataFrame({
                'spot_index': range(total),
                'colocalized_manual': flags,
                'colocalized_auto': results.get('flag_vector', [False] * total)[:total]
            })
            df['method'] = 'visual'
            df['threshold'] = results.get('threshold_value', 'N/A')
            df['image_name'] = getattr(self, 'selected_image_name', '')
            
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Export Complete", 
                f"Exported {total} spots ({marked} colocalized) to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error exporting data:\n{str(e)}")
    
    # === Verify Distance Subtab Methods ===
    
    def populate_verify_distance(self):
        """Populate the Verify Distance subtab with Distance colocalization results."""
        if not hasattr(self, 'distance_coloc_results') or not self.distance_coloc_results:
            QMessageBox.warning(self, "No Results", 
                "Please run Distance colocalization first.")
            return
        
        # Get results
        results = self.distance_coloc_results
        ch0 = results.get('channel_0', 0)
        ch1 = results.get('channel_1', 1)
        df_coloc = results.get('df_colocalized', pd.DataFrame())
        threshold_px = results.get('threshold_distance_px', 2.0)
        threshold_nm = results.get('threshold_distance_nm', 130.0)
        
        # We need to create crops from tracking data
        if not hasattr(self, 'df_tracking') or self.df_tracking.empty:
            QMessageBox.warning(self, "No Data", "No tracking data available.")
            return
        
        # Get image for cropping
        image = self.corrected_image if self.corrected_image is not None else self.image_stack
        if image is None:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        
        # Filter to reference channel (ch0)
        df_ch0 = self.df_tracking[self.df_tracking['spot_type'] == ch0]
        if df_ch0.empty:
            QMessageBox.warning(self, "No Data", f"No spots found for channel {ch0}.")
            return
        
        # Create crops and determine colocalization status
        crop_size = int(getattr(self, 'yx_spot_size_in_px', 5)) + 5
        if crop_size % 2 == 0:
            crop_size += 1
        
        try:
            _, mean_crop, _, crop_size = mi.CropArray(
                image=image,
                df_crops=df_ch0,
                crop_size=crop_size,
                remove_outliers=False,
                max_percentile=99.95
            ).run()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create crops:\n{str(e)}")
            return
        
        if mean_crop is None or mean_crop.size == 0:
            QMessageBox.information(self, "No Spots", "No detected spots to display.")
            return
        
        num_spots = mean_crop.shape[0] // crop_size
        
        # Create flag vector based on distance colocalization
        # Mark spots as colocalized if their coordinates match
        coloc_coords = set()
        if not df_coloc.empty:
            for _, row in df_coloc.iterrows():
                coord = (round(row.get('x', 0), 1), round(row.get('y', 0), 1))
                coloc_coords.add(coord)
        
        flag_vector = []
        for i, (_, row) in enumerate(df_ch0.drop_duplicates(subset=['particle']).iterrows()):
            if i >= num_spots:
                break
            coord = (round(row.get('x', 0), 1), round(row.get('y', 0), 1))
            flag_vector.append(coord in coloc_coords)
        
        # Pad flag_vector if needed
        while len(flag_vector) < num_spots:
            flag_vector.append(False)
        
        # Store for later use
        self.verify_distance_mean_crop = mean_crop
        self.verify_distance_crop_size = crop_size
        
        # Create spot crops with checkboxes
        self._create_verification_crops(
            scroll_area=self.verify_distance_scroll_area,
            checkboxes_list_attr='verify_distance_checkboxes',
            mean_crop=mean_crop,
            crop_size=crop_size,
            flag_vector=flag_vector,
            stats_label=self.verify_distance_stats_label,
            num_channels=image.shape[-1] if image.ndim == 5 else 1,
            channels=(ch0, ch1)
        )
        
        # Update stats label
        self._update_verify_distance_stats()
    
    def _update_verify_distance_stats(self):
        """Update the stats label for Verify Distance subtab."""
        if not hasattr(self, 'verify_distance_checkboxes'):
            return
        total = len(self.verify_distance_checkboxes)
        marked = sum(1 for chk in self.verify_distance_checkboxes if chk.isChecked())
        pct = 100 * marked / total if total > 0 else 0
        
        if hasattr(self, 'distance_coloc_results') and self.distance_coloc_results:
            threshold_px = self.distance_coloc_results.get('threshold_distance_px', 0)
            threshold_nm = self.distance_coloc_results.get('threshold_distance_nm', 0)
            self.verify_distance_stats_label.setText(
                f"[Distance: {threshold_px:.1f}px / {threshold_nm:.0f}nm] "
                f"Total: {total} | Colocalized: {marked} ({pct:.1f}%)"
            )
        else:
            self.verify_distance_stats_label.setText(
                f"Total: {total} | Colocalized: {marked} ({pct:.1f}%)"
            )
    
    def sort_verify_distance(self):
        """Sort Verify Distance results (by cell ID or coordinate)."""
        QMessageBox.information(self, "Sort", 
            "Distance colocalization spots are displayed in detection order.")
    
    def cleanup_verify_distance(self):
        """Clear all checkboxes in Verify Distance subtab."""
        if not hasattr(self, 'verify_distance_checkboxes'):
            return
        for checkbox in self.verify_distance_checkboxes:
            checkbox.setChecked(False)
        self._update_verify_distance_stats()
    
    def export_verify_distance_data(self):
        """Export Verify Distance results to CSV."""
        if not hasattr(self, 'verify_distance_checkboxes') or len(self.verify_distance_checkboxes) == 0:
            QMessageBox.warning(self, "No Data", "No verification data to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Distance Verification Data", "", "CSV Files (*.csv)")
        
        if not file_path:
            return
        
        try:
            total = len(self.verify_distance_checkboxes)
            flags = [chk.isChecked() for chk in self.verify_distance_checkboxes]
            marked = sum(flags)
            
            results = self.distance_coloc_results if hasattr(self, 'distance_coloc_results') else {}
            
            df = pd.DataFrame({
                'spot_index': range(total),
                'colocalized_manual': flags,
            })
            df['method'] = 'distance'
            df['threshold_px'] = results.get('threshold_distance_px', 'N/A')
            df['threshold_nm'] = results.get('threshold_distance_nm', 'N/A')
            df['use_3d'] = results.get('use_3d', False)
            df['image_name'] = getattr(self, 'selected_image_name', '')
            
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Export Complete", 
                f"Exported {total} spots ({marked} colocalized) to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error exporting data:\n{str(e)}")
    
    # === Shared Verification Helper ===
    
    def _create_verification_crops(self, scroll_area, checkboxes_list_attr, mean_crop, 
                                     crop_size, flag_vector, stats_label, num_channels, channels):
        """Create spot crop thumbnails with checkboxes for verification subtabs."""
        scale_factor = getattr(self, 'coloc_thumbnail_scale', 4)
        num_spots = mean_crop.shape[0] // crop_size
        
        # Clear existing
        scroll_area.takeWidget()
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(3)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        checkboxes = []
        
        for i in range(num_spots):
            spot_layout = QHBoxLayout()
            spot_layout.setSpacing(1)
            spot_layout.setContentsMargins(0, 0, 0, 0)
            
            crop_block = mean_crop[i*crop_size:(i+1)*crop_size, :, :]
            
            # Display channels (up to 2)
            for ch_idx, ch in enumerate(channels[:2]):
                if ch < crop_block.shape[-1]:
                    channel_crop = crop_block[:, :, ch]
                    cmin, cmax = np.nanmin(channel_crop), np.nanmax(channel_crop)
                    if cmax > cmin:
                        norm = ((channel_crop - cmin) / (cmax - cmin) * 255).astype(np.uint8)
                    else:
                        norm = np.zeros_like(channel_crop, np.uint8)
                    h, w = norm.shape
                    qimg = QImage(norm.data, w, h, w, QImage.Format_Grayscale8).copy()
                    pix = QPixmap.fromImage(qimg)
                    pix = pix.scaled(w*scale_factor, h*scale_factor, Qt.IgnoreAspectRatio, Qt.FastTransformation)
                    lbl = QLabel()
                    lbl.setPixmap(pix)
                    spot_layout.addWidget(lbl)
            
            # Checkbox
            chk = QCheckBox(f"Spot {i+1}")
            chk.setChecked(bool(flag_vector[i]) if i < len(flag_vector) else False)
            chk.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            
            # Connect to stats update
            if checkboxes_list_attr == 'verify_visual_checkboxes':
                chk.toggled.connect(self._update_verify_visual_stats)
            else:
                chk.toggled.connect(self._update_verify_distance_stats)
            
            spot_layout.addWidget(chk)
            checkboxes.append(chk)
            container_layout.addLayout(spot_layout)
            
            # Separator
            if i < num_spots - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.HLine)
                sep.setFrameShadow(QFrame.Sunken)
                container_layout.addWidget(sep)
        
        scroll_area.setWidget(container)
        setattr(self, checkboxes_list_attr, checkboxes)

    # Note: sort_manual_colocalization() removed - replaced by sort_verify_visual() 
    # and sort_verify_distance() in the separate verification subtabs


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
                # Use unique_particle if available, otherwise fall back to particle
                particle_col = 'unique_particle' if 'unique_particle' in self.df_tracking.columns else 'particle'
                dfm = self.df_tracking[(self.df_tracking[particle_col] == pid) & (self.df_tracking['frame'] == self.current_frame)]
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
            main_img = self.compute_merged_image(use_brightness_slider=True)
            # Fallback for single-channel images (compute_merged_image returns None)
            if main_img is None:
                main_img = norm_stack[selected_channelIndex]
                main_cmap = cmap_list_imagej[selected_channelIndex % len(cmap_list_imagej)]
            else:
                main_cmap = None
        else:
            main_img = norm_stack[selected_channelIndex]
            main_cmap = cmap_list_imagej[selected_channelIndex % len(cmap_list_imagej)]
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], hspace=0.1, wspace=0.1)
        ax_main = fig.add_subplot(gs[0, 0])
        
        # Store main axes reference and recreate RectangleSelector
        self.ax_tracking_vis_main = ax_main
        self.tracking_vis_zoom_selector = RectangleSelector(
            ax_main,
            self._on_tracking_vis_zoom_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False,
            props=dict(facecolor='cyan', edgecolor='white', alpha=0.3, linewidth=2)
        )
        
        gs2 = gs[0, 1].subgridspec(C, 1, hspace=0.1)
        axes_zoom = [fig.add_subplot(gs2[i, 0]) for i in range(C)]        
        # remove background if requested
        if hasattr(self, 'checkbox_remove_bg') and self.checkbox_remove_bg.isChecked():
            seg_mask = getattr(self, 'segmentation_mask', None)
            if seg_mask is not None:
                mask_2d = (seg_mask > 0)
                # If main_img is single‐channel (2D) and mask matches:
                if seg_mask.shape == main_img.shape:
                    main_img = main_img * mask_2d
                # If main_img is merged RGB (3D) and mask matches height/width:
                elif main_img.ndim == 3 and seg_mask.shape == main_img.shape[:2]:
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
        # Add timestamp if requested (format in seconds, ms, or minutes)
        if hasattr(self, 'checkbox_show_timestamp') and self.checkbox_show_timestamp.isChecked():
            if getattr(self, 'time_interval_value', None) is not None:
                time_val = float(self.current_frame) * float(self.time_interval_value)
                ts = self._format_time_interval(time_val)
                ax_main.text(
                    5, 5,
                    ts,
                    color='white',
                    fontsize=12,
                    backgroundcolor='black',
                    va='top',
                    ha='left'
                )
        
        # Draw trajectories if enabled
        if hasattr(self, 'checkbox_show_trajectories') and self.checkbox_show_trajectories.isChecked():
            self._draw_trajectories_on_axes(ax_main, frame_idx)
            
            # Add small color legend if multiple trajectories are drawn
            self._add_trajectory_legend(ax_main)
        
        # Draw rectangle around selected spot
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
        
        # Add thin white frame border to main image
        for spine in self.ax_tracking_vis_main.spines.values():
            spine.set_visible(True)
            spine.set_color('white')
            spine.set_linewidth(0.5)
        
        # Apply zoom if set - must be after tight_layout
        if self.tracking_vis_zoom_roi is not None:
            x_min, x_max, y_min, y_max = self.tracking_vis_zoom_roi
            self.ax_tracking_vis_main.set_xlim(x_min, x_max)
            self.ax_tracking_vis_main.set_ylim(y_max, y_min)  # Inverted for image coordinates
        
        self.canvas_tracking_vis.draw_idle()
        
        # Update trajectory statistics panel
        self._update_trajectory_stats()

    # === Visualization Tab Zoom Methods ===

    def _on_tracking_vis_zoom_select(self, eclick, erelease):
        """Handle rectangle selection for zoom in Visualization tab."""
        if eclick.xdata is None or erelease.xdata is None:
            return
        
        x_min, x_max = sorted([eclick.xdata, erelease.xdata])
        y_min, y_max = sorted([eclick.ydata, erelease.ydata])
        
        # Require minimum selection size
        if (x_max - x_min) < 10 or (y_max - y_min) < 10:
            return
        
        # Store the ROI
        self.tracking_vis_zoom_roi = (x_min, x_max, y_min, y_max)
        
        # Update label
        if hasattr(self, 'tracking_vis_zoom_label'):
            self.tracking_vis_zoom_label.setText(
                f"🔍 ROI: X[{int(x_min)}:{int(x_max)}] Y[{int(y_min)}:{int(y_max)}]"
            )
            self.tracking_vis_zoom_label.setStyleSheet(
                "color: #00d4aa; font-size: 10px; font-weight: bold;"
            )
        
        # Redraw with zoom
        self.display_tracking_visualization()

    def _on_tracking_vis_canvas_click(self, event):
        """Handle mouse clicks on visualization canvas - double-click to reset zoom."""
        if event.dblclick:
            self._reset_tracking_vis_zoom()

    def _reset_tracking_vis_zoom(self):
        """Reset visualization tab zoom to show full image."""
        self.tracking_vis_zoom_roi = None
        
        # Update label
        if hasattr(self, 'tracking_vis_zoom_label'):
            self.tracking_vis_zoom_label.setText("🔍 Full View")
            self.tracking_vis_zoom_label.setStyleSheet("color: #888888; font-size: 10px;")
        
        # Redraw without zoom
        self.display_tracking_visualization()

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
        
        # Recreate RectangleSelector on new axes
        self.tracking_vis_zoom_selector = RectangleSelector(
            self.ax_tracking_vis,
            self._on_tracking_vis_zoom_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False,
            props=dict(facecolor='cyan', edgecolor='white', alpha=0.3, linewidth=2)
        )
        
        # Reset zoom ROI
        self.tracking_vis_zoom_roi = None
        if hasattr(self, 'tracking_vis_zoom_label'):
            self.tracking_vis_zoom_label.setText("🔍 Full View")
            self.tracking_vis_zoom_label.setStyleSheet("color: #888888; font-size: 10px;")
        
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

    def setup_msd_tab(self):
        """
        Set up the 'MSD' (Mean Squared Displacement) tab for diffusion analysis.
        
        Layout:
        - Left panel: MSD plot canvas with log-log toggle and export buttons
        - Right panel: Parameters (max_fit_points, 3D checkbox) and results display
        """
        msd_main_layout = QHBoxLayout(self.msd_tab)
        
        # Left panel: Plot and controls
        msd_left_layout = QVBoxLayout()
        msd_main_layout.addLayout(msd_left_layout)
        
        # MSD Plot Canvas
        self.figure_msd, self.ax_msd = plt.subplots(figsize=(8, 6))
        self.figure_msd.patch.set_facecolor('black')
        self.canvas_msd = FigureCanvas(self.figure_msd)
        msd_left_layout.addWidget(self.canvas_msd)
        self.toolbar_msd = NavigationToolbar(self.canvas_msd, self.msd_tab)
        msd_left_layout.addWidget(self.toolbar_msd)
        
        # Style the axes for dark theme
        self.ax_msd.set_facecolor('black')
        self.ax_msd.tick_params(colors='white', which='both')
        for spine in self.ax_msd.spines.values():
            spine.set_color('white')
        self.ax_msd.xaxis.label.set_color('white')
        self.ax_msd.yaxis.label.set_color('white')
        self.ax_msd.title.set_color('white')
        self.ax_msd.grid(True, which='both', color='gray', linestyle='--', linewidth=0.1)
        
        # Log-log scale checkbox
        self.msd_loglog_checkbox = QCheckBox("Log-Log Scale")
        self.msd_loglog_checkbox.setChecked(False)
        self.msd_loglog_checkbox.stateChanged.connect(self.plot_msd)
        msd_left_layout.addWidget(self.msd_loglog_checkbox)
        
        # Export buttons
        export_layout = QHBoxLayout()
        self.export_msd_dataframe_button = QPushButton("Export DataFrame")
        self.export_msd_dataframe_button.clicked.connect(self.export_msd_dataframe)
        export_layout.addWidget(self.export_msd_dataframe_button)
        
        self.export_msd_plot_button = QPushButton("Export Plot")
        self.export_msd_plot_button.clicked.connect(self.export_msd_plot)
        export_layout.addWidget(self.export_msd_plot_button)
        msd_left_layout.addLayout(export_layout)
        
        # Right panel: Parameters and results
        msd_right_layout = QVBoxLayout()
        msd_main_layout.addLayout(msd_right_layout)
        
        # Tracking Channel group (first/prominent)
        tracking_group = QGroupBox("Tracking Channel")
        tracking_layout = QHBoxLayout()
        self.msd_tracking_channel_combo = QComboBox()
        # Will be populated on tab switch with tracked channels (no "All" option)
        tracking_layout.addWidget(self.msd_tracking_channel_combo)
        tracking_group.setLayout(tracking_layout)
        msd_right_layout.addWidget(tracking_group)
        
        # Parameters group
        params_group = QGroupBox("MSD Parameters")
        params_layout = QFormLayout()
        
        # Number of points to fit (defaults to half of lag points)
        self.msd_fit_points_spinbox = QSpinBox()
        self.msd_fit_points_spinbox.setRange(2, 1000)
        self.msd_fit_points_spinbox.setValue(20)
        params_layout.addRow("Fit Points:", self.msd_fit_points_spinbox)
        
        # Mode indicator (auto-detected)
        self.msd_mode_label = QLabel("Mode: Auto-detect")
        self.msd_mode_label.setStyleSheet("color: gray; font-style: italic;")
        params_layout.addRow(self.msd_mode_label)
        
        params_group.setLayout(params_layout)
        msd_right_layout.addWidget(params_group)
        
        # Calculate button
        self.calculate_msd_button = QPushButton("Calculate MSD")
        self.calculate_msd_button.clicked.connect(self.calculate_msd_from_gui)
        self.calculate_msd_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        msd_right_layout.addWidget(self.calculate_msd_button)
        
        # Results group
        results_group = QGroupBox("Results")
        results_layout = QFormLayout()
        
        self.msd_diffusion_label = QLabel("D = --")
        self.msd_diffusion_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        results_layout.addRow("Diffusion Coefficient:", self.msd_diffusion_label)
        
        # Diffusion coefficient in px²/s
        self.msd_diffusion_px_label = QLabel("D = --")
        self.msd_diffusion_px_label.setStyleSheet("font-size: 12px;")
        results_layout.addRow("D (px²/s):", self.msd_diffusion_px_label)
        
        self.msd_r_squared_label = QLabel("R² = --")
        results_layout.addRow("R² (Linear Fit):", self.msd_r_squared_label)
        
        self.msd_n_particles_label = QLabel("N = --")
        results_layout.addRow("Particles:", self.msd_n_particles_label)
        
        results_group.setLayout(results_layout)
        msd_right_layout.addWidget(results_group)
        
        msd_right_layout.addStretch()
        
        # Initialize MSD data storage
        self.msd_data = None
        self.msd_per_trajectory = None

    
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
        
        # Set up zoom feature: RectangleSelector for left-click drag
        self.tracking_vis_zoom_selector = RectangleSelector(
            self.ax_tracking_vis,
            self._on_tracking_vis_zoom_select,
            useblit=True,
            button=[1],  # Left mouse button only
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False,
            props=dict(facecolor='cyan', edgecolor='white', alpha=0.3, linewidth=2)
        )
        
        # Connect double-click to reset zoom
        self.canvas_tracking_vis.mpl_connect('button_press_event', self._on_tracking_vis_canvas_click)
        
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
        
        # Merge brightness slider (reduces saturation for multi-channel merge)
        spin_layout.addWidget(QLabel("Merge Brightness", self))
        self.merge_brightness_slider = QSlider(Qt.Horizontal)
        self.merge_brightness_slider.setRange(10, 100)
        self.merge_brightness_slider.setValue(50)  # Default 50% to prevent oversaturation
        self.merge_brightness_slider.setMaximumWidth(80)
        self.merge_brightness_slider.valueChanged.connect(lambda v: self.display_tracking_visualization())
        spin_layout.addWidget(self.merge_brightness_slider)
        self.merge_brightness_label = QLabel("50%")
        self.merge_brightness_label.setMinimumWidth(35)
        spin_layout.addWidget(self.merge_brightness_label)
        
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
        
        self.frame_label_tracking_vis = QLabel("0/0")
        self.frame_label_tracking_vis.setMinimumWidth(50)
        controls_layout.addWidget(self.frame_label_tracking_vis)
        
        self.play_button_tracking_vis = QPushButton("Play", self)
        self.play_button_tracking_vis.clicked.connect(self.play_pause_tracking_vis)
        controls_layout.addWidget(self.play_button_tracking_vis)
        
        # Zoom status label
        self.tracking_vis_zoom_label = QLabel("🔍 Full View")
        self.tracking_vis_zoom_label.setStyleSheet("color: #888888; font-size: 10px;")
        controls_layout.addWidget(self.tracking_vis_zoom_label)
        # Export buttons (Image & Video)
        export_buttons_layout = QHBoxLayout()
        left_layout.addLayout(export_buttons_layout)
        self.export_tracking_vis_image_button = QPushButton("Export Image", self)
        self.export_tracking_vis_image_button.clicked.connect(self.export_tracking_visualization_image)
        export_buttons_layout.addWidget(self.export_tracking_vis_image_button)
        self.export_tracking_vis_video_button = QPushButton("Export Video", self)
        self.export_tracking_vis_video_button.clicked.connect(self.export_tracking_visualization_video)
        export_buttons_layout.addWidget(self.export_tracking_vis_video_button)
        # Right side: Enhanced controls panel
        right_layout.addWidget(QLabel("<b>Trajectory Display</b>"))
        
        # Show trajectories checkbox
        self.checkbox_show_trajectories = QCheckBox("Show Trajectories")
        self.checkbox_show_trajectories.setChecked(True)
        self.checkbox_show_trajectories.stateChanged.connect(self.display_tracking_visualization)
        right_layout.addWidget(self.checkbox_show_trajectories)
        
        # Trajectory tail length
        tail_layout = QHBoxLayout()
        tail_layout.addWidget(QLabel("Tail Length:"))
        self.trajectory_tail_spinbox = QSpinBox()
        self.trajectory_tail_spinbox.setRange(1, 100)
        self.trajectory_tail_spinbox.setValue(10)
        self.trajectory_tail_spinbox.setToolTip("Number of frames to show in trajectory tail (0 = full)")
        self.trajectory_tail_spinbox.valueChanged.connect(self.display_tracking_visualization)
        tail_layout.addWidget(self.trajectory_tail_spinbox)
        right_layout.addLayout(tail_layout)
        
        # Color-by dropdown
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color By:"))
        self.trajectory_color_combo = QComboBox()
        self.trajectory_color_combo.addItem("Particle ID", "particle")
        self.trajectory_color_combo.addItem("Cell ID", "cell_id")
        self.trajectory_color_combo.addItem("Channel", "spot_type")
        self.trajectory_color_combo.currentIndexChanged.connect(self.display_tracking_visualization)
        color_layout.addWidget(self.trajectory_color_combo)
        right_layout.addLayout(color_layout)
        
        # Separator
        right_layout.addWidget(QLabel(""))
        right_layout.addWidget(QLabel("<b>Particle Selection</b>"))
        
        # Filter by cell_id
        cell_filter_layout = QHBoxLayout()
        cell_filter_layout.addWidget(QLabel("Filter Cell:"))
        self.vis_cell_filter_combo = QComboBox()
        self.vis_cell_filter_combo.addItem("All Cells", -1)
        self.vis_cell_filter_combo.currentIndexChanged.connect(self._update_particle_list_filtered)
        cell_filter_layout.addWidget(self.vis_cell_filter_combo)
        right_layout.addLayout(cell_filter_layout)
        
        # Tracked particles list (multi-select)
        self.tracked_particles_list = QListWidget()
        self.tracked_particles_list.setFixedWidth(150)
        self.tracked_particles_list.setFixedHeight(200)
        self.tracked_particles_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tracked_particles_list.itemSelectionChanged.connect(self.display_tracking_visualization)
        right_layout.addWidget(self.tracked_particles_list)
        
        # Select All / Clear buttons
        select_buttons_layout = QHBoxLayout()
        self.select_all_particles_btn = QPushButton("Select All")
        self.select_all_particles_btn.clicked.connect(self._select_all_particles)
        select_buttons_layout.addWidget(self.select_all_particles_btn)
        self.clear_particles_btn = QPushButton("Clear")
        self.clear_particles_btn.clicked.connect(self._clear_particle_selection)
        select_buttons_layout.addWidget(self.clear_particles_btn)
        right_layout.addLayout(select_buttons_layout)
        
        # Separator
        right_layout.addWidget(QLabel(""))
        right_layout.addWidget(QLabel("<b>Display Options</b>"))
        
        # Existing checkboxes
        self.checkbox_remove_bg = QCheckBox("Remove Background")    
        self.checkbox_remove_bg.setChecked(False)
        right_layout.addWidget(self.checkbox_remove_bg)
        
        self.checkbox_scalebar = QCheckBox("Show Scalebar")     
        self.checkbox_scalebar.setChecked(False)    
        right_layout.addWidget(self.checkbox_scalebar)
        
        self.checkbox_show_timestamp = QCheckBox("Show Time Stamp")
        self.checkbox_show_timestamp.setChecked(False) 
        right_layout.addWidget(self.checkbox_show_timestamp)
        
        # Connect checkboxes to update visualization    
        self.checkbox_remove_bg.stateChanged.connect(self.display_tracking_visualization)
        self.checkbox_scalebar.stateChanged.connect(self.display_tracking_visualization)   
        self.checkbox_show_timestamp.stateChanged.connect(self.display_tracking_visualization)
        
        # Separator
        right_layout.addWidget(QLabel(""))
        right_layout.addWidget(QLabel("<b>Playback</b>"))
        
        # Playback speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.playback_speed_combo = QComboBox()
        self.playback_speed_combo.addItem("0.5x", 0.5)
        self.playback_speed_combo.addItem("1x", 1.0)
        self.playback_speed_combo.addItem("2x", 2.0)
        self.playback_speed_combo.addItem("4x", 4.0)
        self.playback_speed_combo.setCurrentIndex(1)  # Default 1x
        self.playback_speed_combo.currentIndexChanged.connect(self._update_playback_speed)
        speed_layout.addWidget(self.playback_speed_combo)
        right_layout.addLayout(speed_layout)
        
        # Separator
        right_layout.addWidget(QLabel(""))
        right_layout.addWidget(QLabel("<b>Trajectory Stats</b>"))
        
        # Trajectory statistics label (updated when selection changes)
        self.trajectory_stats_label = QLabel("Select a trajectory")
        self.trajectory_stats_label.setWordWrap(True)
        self.trajectory_stats_label.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                color: #e0e0e0;
                padding: 5px;
                border-radius: 3px;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        right_layout.addWidget(self.trajectory_stats_label)
        
        right_layout.addStretch()

# =============================================================================
# =============================================================================
# CROPS TAB
# =============================================================================
# =============================================================================

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
        # Only add image name if different from base file name (avoid duplication)
        if safe_image_name and safe_image_name != safe_base_file_name:
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

                elif label_text == "Export MSD Data":
                    if hasattr(self, 'msd_per_trajectory') and self.msd_per_trajectory is not None:
                        default_filename = self.get_default_export_filename(prefix="msd_dataframe", extension="csv")
                        out_path = results_folder / default_filename
                        try:
                            self.msd_per_trajectory.to_csv(out_path, index=False)
                        except Exception as e:
                            print(f"Error exporting MSD data: {e}")
                    else:
                        print("No MSD data to export. Run MSD calculation first.")

                elif label_text == "Export MSD Image":
                    if hasattr(self, 'figure_msd') and self.msd_data is not None:
                        default_filename = self.get_default_export_filename(prefix="msd_plot", extension="png")
                        out_path = results_folder / default_filename
                        try:
                            self.figure_msd.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='black')
                        except Exception as e:
                            print(f"Error exporting MSD image: {e}")
                    else:
                        print("No MSD plot to export. Run MSD calculation first.")

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

                elif label_text == "Export Distance Coloc Image":
                    default_filename = self.get_default_export_filename(prefix="distance_colocalization", extension="png")
                    out_path = results_folder / default_filename
                    self._export_distance_coloc_image(out_path)

                elif label_text == "Export Distance Coloc Data":
                    default_filename = self.get_default_export_filename(prefix="distance_colocalization_data", extension="csv")
                    out_path = results_folder / default_filename
                    self._export_distance_coloc_data_to_csv(out_path)

                elif label_text == "Export Manual Colocalization Image":
                    default_filename = self.get_default_export_filename(prefix="colocalization_manual", extension="png")
                    out_path = results_folder / default_filename
                    self._export_manual_colocalization_image(out_path)

                elif label_text == "Export Manual Colocalization Data":
                    default_filename = self.get_default_export_filename(prefix="colocalization_manual_data", extension="csv")
                    out_path = results_folder / default_filename
                    self._export_manual_colocalization_data_to_csv(out_path)

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
        # Check both verification subtabs for checkbox data
        has_visual = hasattr(self, 'verify_visual_checkboxes') and len(self.verify_visual_checkboxes) > 0
        has_distance = hasattr(self, 'verify_distance_checkboxes') and len(self.verify_distance_checkboxes) > 0
        
        if not has_visual and not has_distance:
            return  # No manual selections to export
        
        try:
            # Use visual checkboxes if available, otherwise distance
            if has_visual:
                checkboxes = self.verify_visual_checkboxes
                mean_crop = getattr(self, 'colocalization_results', {}).get('mean_crop_filtered')
                crop_size = getattr(self, 'colocalization_results', {}).get('crop_size', 15)
            else:
                checkboxes = self.verify_distance_checkboxes
                mean_crop = getattr(self, 'verify_distance_mean_crop', None)
                crop_size = getattr(self, 'verify_distance_crop_size', 15)
            
            if mean_crop is None:
                return
            
            # Prepare flag vector from checkboxes
            total = len(checkboxes)
            flags = [chk.isChecked() for chk in checkboxes]
            percent_marked = (sum(flags) / total * 100.0) if total > 0 else 0.0
            # Determine channels to include (use same channels as selected in UI)
            ch1 = self.channel_combo_box_1.currentIndex() if hasattr(self, 'channel_combo_box_1') else 0
            ch2 = self.channel_combo_box_2.currentIndex() if hasattr(self, 'channel_combo_box_2') else 1
            selected_channels = (ch1, ch2)
            # Create a figure for the manual colocalization mosaic
            fig = Figure()
            title_text = f"Manual Colocalization: {percent_marked:.2f}%"
            # Use the utility to plot all crops, marking selected spots in light blue
            self.plots.plot_matrix_pair_crops(mean_crop=mean_crop,
                                    crop_size=crop_size,
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

    def _export_distance_coloc_image(self, file_path):
        """Export Distance Colocalization image (overlay view) from Export tab."""
        if not hasattr(self, 'distance_coloc_results') or not self.distance_coloc_results:
            return
        try:
            self.figure_dist_coloc.savefig(file_path, dpi=150, bbox_inches='tight', 
                                           facecolor='black', edgecolor='none')
        except Exception as e:
            print(f"Failed to export distance colocalization image: {e}")

    def _export_distance_coloc_data_to_csv(self, out_path: Path):
        """Export Distance Colocalization data as CSV from Export tab."""
        if not hasattr(self, 'distance_coloc_results') or not self.distance_coloc_results:
            return
        try:
            results = self.distance_coloc_results
            df_classification = results.get('df_classification')
            if df_classification is not None and not df_classification.empty:
                df_classification.to_csv(out_path, index=False)
        except Exception as e:
            print(f"Failed to export distance colocalization data: {e}")

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

    # Note: _export_crops_image removed - Crops tab has been deprecated

    


    def _export_metadata(self, file_path):        
        # Photobleaching: Read from widgets if available, else fallback to attribute
        pb_mode = self.mode_combo.currentText() if hasattr(self, 'mode_combo') else self.photobleaching_mode
        pb_radius = self.radius_slider.value() if hasattr(self, 'radius_slider') else self.photobleaching_radius

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
            
            # Registration Parameters
            registered_image=self.registered_image,
            registration_mode=self.registration_mode,
            registration_roi=self.registration_roi,
            
            # Segmentation Masks
            segmentation_mask=self.segmentation_mask,
            cellpose_masks_cyto=self.cellpose_masks_cyto,
            cellpose_masks_nuc=self.cellpose_masks_nuc,
            _active_mask_source=self._active_mask_source,
            segmentation_z_used_for_mask=getattr(self, 'segmentation_z_used_for_mask', -1),
            segmentation_z_max=getattr(self, 'segmentation_z_max', 0),
            
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
            photobleaching_data=getattr(self, 'photobleaching_data', None),
            
            # MSD Results from Tracking
            tracking_D_um2_s=getattr(self, 'tracking_D_um2_s', None),
            tracking_D_px2_s=getattr(self, 'tracking_D_px2_s', None),
            tracking_msd_mode=getattr(self, 'tracking_msd_mode', None),
            tracking_msd_channel=getattr(self, 'tracking_msd_channel', None),
            
            file_path=file_path,
            
            # Updated ML Params (Values)
            use_ml_checkbox=is_ml,
            ml_threshold_input=ml_thresh,
            
            link_using_3d_coordinates=self.link_using_3d_coordinates,
            colocalization_method=coloc_method,
            colocalization_threshold_value=coloc_thresh,
            multi_tau=self.use_multi,
            
            # Multi-Channel Tracking Data
            tracked_channels=getattr(self, 'tracked_channels', []),
            tracking_thresholds=getattr(self, 'tracking_thresholds', {}),
            tracking_parameters_per_channel=getattr(self, 'tracking_parameters_per_channel', {}),
            primary_tracking_channel=getattr(self, 'primary_tracking_channel', None),
            
            # Distance Colocalization Parameters
            distance_coloc_results=getattr(self, 'distance_coloc_results', None)
        )
        try:
            meta.write_metadata()
        except Exception as e:
            print(f"Failed to export metadata file: {e}")

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

    

    # Note: export_crops_image removed - Crops tab has been deprecated


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
            ("Export MSD Data", "msd_data"),
            ("Export MSD Image", "msd_image"),
            ("Export Distributions Image", "distribution"),
            ("Export Time Course Image", "time_course"),
            ("Export Correlation Image", "correlation"),
            ("Export Colocalization Image", "colocalization"),
            ("Export Colocalization Data", "colocalization_data"),
            ("Export Distance Coloc Image", "distance_coloc_image"),
            ("Export Distance Coloc Data", "distance_coloc_data"),
            ("Export Manual Colocalization Image", "colocalization_manual"),
            ("Export Manual Colocalization Data", "colocalization_manual_data"),
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
        
        # Reset zoom state
        self.display_zoom_roi = None
        if hasattr(self, 'display_zoom_label'):
            self.display_zoom_label.setText("🔍 Full View")
            self.display_zoom_label.setStyleSheet("color: #888888; font-size: 10px;")
        
        # Recreate RectangleSelector for new axes
        self.display_zoom_selector = RectangleSelector(
            self.ax_display,
            self._on_display_zoom_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False,
            props=dict(facecolor='cyan', edgecolor='white', alpha=0.3, linewidth=2)
        )
        
        self.canvas_display.draw()
        self.time_slider_display.setValue(0)
        self.play_button_display.setText("Play")
        self.playing = False

    def reset_registration_tab(self):
        """Reset registration tab state and display."""
        # Reset state variables
        self.registered_image = None
        self.registration_roi = None
        if hasattr(self, 'reg_roi_rect') and self.reg_roi_rect is not None:
            try:
                self.reg_roi_rect.remove()
            except Exception:
                pass
            self.reg_roi_rect = None
        self.reg_roi_start = None
        
        # Reset time slider
        if hasattr(self, 'time_slider_reg'):
            self.time_slider_reg.setValue(0)
        
        # Stop playback if running
        if hasattr(self, 'reg_playing') and self.reg_playing:
            if hasattr(self, 'reg_timer'):
                self.reg_timer.stop()
            self.reg_playing = False
            if hasattr(self, 'play_button_reg'):
                self.play_button_reg.setText("▶")
        
        # Clear the original panel
        if hasattr(self, 'figure_reg_original'):
            self.figure_reg_original.clear()
            self.ax_reg_original = self.figure_reg_original.add_subplot(111)
            self.ax_reg_original.set_facecolor('black')
            self.ax_reg_original.axis('off')
            self.ax_reg_original.text(
                0.5, 0.5, 'No image loaded.',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12, color='white',
                transform=self.ax_reg_original.transAxes
            )
            self.figure_reg_original.subplots_adjust(left=0, right=1, top=1, bottom=0)
            if hasattr(self, 'canvas_reg_original'):
                self.canvas_reg_original.draw()
        
        # Clear the result panel
        if hasattr(self, 'figure_reg_result'):
            self.figure_reg_result.clear()
            self.ax_reg_result = self.figure_reg_result.add_subplot(111)
            self.ax_reg_result.set_facecolor('black')
            self.ax_reg_result.axis('off')
            self.ax_reg_result.text(
                0.5, 0.5, 'No registration yet.',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12, color='white',
                transform=self.ax_reg_result.transAxes
            )
            self.figure_reg_result.subplots_adjust(left=0, right=1, top=1, bottom=0)
            if hasattr(self, 'canvas_reg_result'):
                self.canvas_reg_result.draw()
        
        # Clear channel buttons
        if hasattr(self, 'channel_buttons_reg'):
            for btn in self.channel_buttons_reg:
                if btn:
                    btn.setParent(None)
            self.channel_buttons_reg = []

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
        # Reset Z-slider to max projection (default)
        self.reset_segmentation_z_slider()
        
        # Reset watershed controls to defaults
        self._original_watershed_mask = None
        self.watershed_threshold_factor = 1.0
        if hasattr(self, 'watershed_threshold_slider'):
            self.watershed_threshold_slider.blockSignals(True)
            self.watershed_threshold_slider.setValue(100)  # 1.0 factor
            self.watershed_threshold_slider.blockSignals(False)
        if hasattr(self, 'watershed_threshold_label'):
            self.watershed_threshold_label.setText("1.00")
        if hasattr(self, 'watershed_size_slider'):
            self.watershed_size_slider.blockSignals(True)
            self.watershed_size_slider.setValue(0)
            self.watershed_size_slider.blockSignals(False)
        if hasattr(self, 'watershed_size_label'):
            self.watershed_size_label.setText("0")

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
        
        # Reset image source combo to "Original Image" in tracking tab
        if hasattr(self, 'image_source_combo'):
            self.image_source_combo.setCurrentIndex(0)  # "Original Image"

    def reset_tracking_tab(self):
        self.df_tracking = pd.DataFrame()
        self.detected_spots_frame = None
        
        # Clear multi-channel tracking data
        self.multi_channel_tracking_data = {}
        self.tracked_channels = []
        self.tracking_thresholds = {}
        self.auto_threshold_per_channel = {}
        self.tracking_parameters_per_channel = {}
        self.primary_tracking_channel = None
        self.has_tracked = False
        
        # Update tracked channels list widget if available
        if hasattr(self, 'tracked_channels_list'):
            self.tracked_channels_list.clear()
        
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
        
        # Re-initialize RectangleSelector on the new axes
        self.tracking_zoom_selector = RectangleSelector(
            self.ax_tracking,
            self._on_tracking_zoom_select,
            useblit=True,
            button=[1],  # Left mouse button only
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False,
            props=dict(facecolor='cyan', edgecolor='white', alpha=0.3, linewidth=2)
        )
        
        self.canvas_tracking.draw()
        if hasattr(self, 'time_slider_tracking'):
            self.time_slider_tracking.setValue(0)
        if hasattr(self, 'tracking_show_masks_checkbox'):
            self.tracking_show_masks_checkbox.setChecked(True)  # Keep masks visible by default
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
        # Reset threshold value label
        if hasattr(self, 'threshold_value_label'):
            self.threshold_value_label.setText("Value: --")
        # Reset fast gaussian fit checkbox to default (True)
        if hasattr(self, 'fast_gaussian_fit_checkbox'):
            self.fast_gaussian_fit = True
            self.fast_gaussian_fit_checkbox.setChecked(True)
        # Update 2D/3D mode toggle buttons to reflect current state
        if hasattr(self, 'btn_mode_2d') and hasattr(self, 'btn_mode_3d'):
            self._update_tracking_mode_buttons()
            self._update_tracking_mode_status()
        
        # Reset zoom ROI
        self.tracking_zoom_roi = None
        if hasattr(self, 'tracking_zoom_label'):
            self.tracking_zoom_label.setText("🔍 Full View")
            self.tracking_zoom_label.setStyleSheet("color: #888888; font-size: 10px;")

    def reset_msd_tab(self):
        """Reset the MSD tab to its initial state."""
        if hasattr(self, 'figure_msd'):
            self.figure_msd.clear()
            self.ax_msd = self.figure_msd.add_subplot(111)
            self.ax_msd.text(
                0.5, 0.5, 'No MSD data available.\nRun tracking first, then click "Calculate MSD".',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12, color='gray'
            )
            self.ax_msd.axis('off')
            self.canvas_msd.draw_idle()
        
        if hasattr(self, 'msd_diffusion_label'):
            self.msd_diffusion_label.setText("D = --")
        if hasattr(self, 'msd_diffusion_px_label'):
            self.msd_diffusion_px_label.setText("D = --")
        if hasattr(self, 'msd_r_squared_label'):
            self.msd_r_squared_label.setText("R² = --")
        if hasattr(self, 'msd_n_particles_label'):
            self.msd_n_particles_label.setText("N = --")
        
        # Reset mode label to default
        if hasattr(self, 'msd_mode_label'):
            self.msd_mode_label.setText("Mode: Auto-detect")
            self.msd_mode_label.setStyleSheet("color: gray; font-style: italic;")
        
        # Reset fit points spinbox to default
        if hasattr(self, 'msd_fit_points_spinbox'):
            self.msd_fit_points_spinbox.setValue(20)
        
        self.msd_data = None
        self.msd_per_trajectory = None
        
        # Also reset tracking D values for metadata consistency
        self.tracking_D_um2_s = None
        self.tracking_D_px2_s = None
        self.tracking_msd_mode = None
        self.tracking_msd_channel = None


    def calculate_msd_from_gui(self):
        """Calculate MSD using tracked particle data from the Tracking tab."""
        
        # Check for single-frame images (MSD requires at least 2 time points)
        if getattr(self, 'total_frames', 0) < 2:
            QMessageBox.warning(self, "Insufficient Data", 
                "MSD analysis requires at least 2 time points.\n\n"
                "Current image has only 1 frame. MSD measures how particles "
                "move over time, which requires multiple frames.")
            return
        
        # Check if tracking data exists
        if not hasattr(self, 'df_tracking') or self.df_tracking is None or len(self.df_tracking) == 0:
            QMessageBox.warning(self, "No Data", "No tracking data available. Please run tracking first.")
            return
        
        # Filter by tracking channel (spot_type) - single channel required
        df_to_analyze = self.df_tracking.copy()
        tracking_ch = None
        if hasattr(self, 'msd_tracking_channel_combo'):
            tracking_ch = self.msd_tracking_channel_combo.currentData()
            # Check if placeholder "No tracked channels" is selected
            if tracking_ch == -1:
                QMessageBox.warning(self, "No Data", "No tracked channels available. Please run tracking first.")
                return
            # Filter by spot_type
            if 'spot_type' in df_to_analyze.columns:
                df_to_analyze = df_to_analyze[df_to_analyze['spot_type'] == tracking_ch]
        
        if df_to_analyze.empty:
            QMessageBox.warning(self, "No Data", "No tracking data for selected channel.")
            return
        
        try:
            # Auto-detect 2D vs 3D based on tracking mode
            # 2D if: (1) use_maximum_projection is True, OR (2) all Z values are constant
            is_2d_projection = getattr(self, 'use_maximum_projection', False)
            
            # Check if Z values are constant (all same value = 2D tracking)
            z_is_constant = False
            if 'z' in df_to_analyze.columns:
                z_unique = df_to_analyze['z'].nunique()
                z_is_constant = (z_unique <= 1)
            else:
                z_is_constant = True  # No Z column means 2D
            
            # Determine if 3D: only if NOT using 2D projection AND Z values vary
            is_3d = not is_2d_projection and not z_is_constant
            
            # Update the mode label to reflect auto-detected value
            mode_text = "3D (D = slope/6)" if is_3d else "2D (D = slope/4)"
            self.msd_mode_label.setText(f"Mode: {mode_text}")
            self.msd_mode_label.setStyleSheet("color: lime; font-weight: bold;" if is_3d else "color: cyan; font-weight: bold;")
            
            max_fit_points = self.msd_fit_points_spinbox.value()
            
            # Get metadata - convert voxel_yx_nm (nanometers) to microns
            if hasattr(self, 'voxel_yx_nm') and self.voxel_yx_nm is not None:
                # Ensure scalar conversion (voxel_yx_nm might be a numpy array)
                voxel_val = self.voxel_yx_nm
                if hasattr(voxel_val, 'item'):  # numpy array with 1 element
                    voxel_val = voxel_val.item()
                microns_per_pixel = float(voxel_val) / 1000.0  # nm to µm
            else:
                microns_per_pixel = 1.0  # Fallback
                logging.warning("voxel_yx_nm not set, using 1.0 µm/px for MSD")
            
            # Get time interval from metadata
            if hasattr(self, 'time_interval_value') and self.time_interval_value is not None:
                time_val = self.time_interval_value
                if hasattr(time_val, 'item'):  # numpy array with 1 element
                    time_val = time_val.item()
                step_size_in_sec = float(time_val)
            else:
                step_size_in_sec = 1.0  # Fallback
                logging.warning("time_interval_value not set, using 1.0 s for MSD")
            # Get Z voxel size for 3D MSD - convert from nm to microns
            if is_3d and hasattr(self, 'voxel_z_nm') and self.voxel_z_nm is not None:
                z_val = self.voxel_z_nm
                if hasattr(z_val, 'item'):  # numpy array with 1 element
                    z_val = z_val.item()
                microns_per_pixel_z = float(z_val) / 1000.0  # nm to µm
            else:
                microns_per_pixel_z = None  # Will use microns_per_pixel for Z (isotropic assumption)
            
            # Create ParticleMotion instance
            motion = mi.ParticleMotion(
                trackpy_dataframe=df_to_analyze,
                microns_per_pixel=microns_per_pixel,
                step_size_in_sec=step_size_in_sec,
                max_lagtime=None,
                show_plot=False,
                remove_drift=False,
                max_fit_points=max_fit_points,
                is_3d=is_3d,
                microns_per_pixel_z=microns_per_pixel_z
            )
            
            # Calculate MSD
            D_um2_s, D_px2_s, em_um2, em_px2, fit_times, fit_line_msd, trackpy_df = motion.calculate_msd()
            
            # Get the selected tracking channel
            selected_tracking_ch = None
            if hasattr(self, 'msd_tracking_channel_combo'):
                selected_tracking_ch = self.msd_tracking_channel_combo.currentData()
            
            # Store results with channel info
            self.msd_data = {
                'D_um2_s': D_um2_s,
                'D_px2_s': D_px2_s,
                'em_um2': em_um2,
                'em_px2': em_px2,
                'fit_times': fit_times,
                'fit_line_msd': fit_line_msd,
                'trackpy_df': trackpy_df,
                'is_3d': is_3d,
                'tracking_channel': selected_tracking_ch
            }
            
            # Also update tracking values for metadata export
            # This ensures metadata reflects the most recent MSD calculation
            self.tracking_D_um2_s = D_um2_s
            self.tracking_D_px2_s = D_px2_s
            self.tracking_msd_mode = "3D" if is_3d else "2D"
            self.tracking_msd_channel = selected_tracking_ch  # Store which channel MSD was calculated for
            
            # Calculate per-trajectory MSD for export (use filtered df_to_analyze to preserve cell_id)
            self._calculate_per_trajectory_msd(df_to_analyze, microns_per_pixel, step_size_in_sec)
            
            # Calculate R² value

            slope, intercept, r_value, p_value, std_err = linregress(fit_times, em_um2.values[:len(fit_times)])
            
            # Update result labels - use scientific notation for D
            n_particles = trackpy_df['particle'].nunique()
            
            # Check if we have per-cell data for summary
            if hasattr(self, 'msd_per_cell') and self.msd_per_cell:
                # Only count cells with enough particles (same threshold as plot_msd)
                MIN_PARTICLES_PER_CELL = 10
                valid_cells = {cid: data for cid, data in self.msd_per_cell.items() 
                               if data['n_particles'] >= MIN_PARTICLES_PER_CELL}
                n_cells = len(valid_cells)
                all_D_values = [d for cell in valid_cells.values() for d in cell['D_values']]
                # Also count only particles from valid cells
                n_particles_valid = sum(cell['n_particles'] for cell in valid_cells.values())
                if all_D_values:
                    D_mean = np.mean(all_D_values)
                    D_std = np.std(all_D_values)
                    self.msd_diffusion_label.setText(f"D = {D_mean:.2e} ± {D_std:.2e} µm²/s")
                    # Convert to px²/s: D_px2 = D_um2 / (microns_per_pixel)^2
                    D_mean_px = D_mean / (microns_per_pixel ** 2)
                    D_std_px = D_std / (microns_per_pixel ** 2)
                    self.msd_diffusion_px_label.setText(f"D = {D_mean_px:.2e} ± {D_std_px:.2e} px²/s")
                    self.msd_n_particles_label.setText(f"N = {n_particles_valid} (from {n_cells} cells)")
                else:
                    self.msd_diffusion_label.setText(f"D = {D_um2_s:.2e} µm²/s")
                    self.msd_diffusion_px_label.setText(f"D = {D_px2_s:.2e} px²/s")
                    self.msd_n_particles_label.setText(f"N = {n_particles}")
            else:
                self.msd_diffusion_label.setText(f"D = {D_um2_s:.2e} µm²/s")
                self.msd_diffusion_px_label.setText(f"D = {D_px2_s:.2e} px²/s")
                self.msd_n_particles_label.setText(f"N = {n_particles}")
            self.msd_r_squared_label.setText(f"R² = {r_value**2:.4f}")
            
            # Plot results
            self.plot_msd()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"MSD calculation failed:\n{str(e)}")
            traceback.print_exc()

    def _calculate_per_trajectory_msd(self, trackpy_df, microns_per_pixel, step_size_in_sec):
        """Calculate MSD for each individual trajectory for export, organized by cell."""
        
        # Get unique cell IDs
        if 'cell_id' in trackpy_df.columns:
            cell_ids = sorted(trackpy_df['cell_id'].dropna().unique())
        else:
            cell_ids = [0]  # Default to single cell if no cell_id column
        
        msd_dict = {}  # {(cell_id, particle_id): em}
        max_lag = 0
        
        # Per-cell results for plotting
        self.msd_per_cell = {}
        
        for cell_id in cell_ids:
            if 'cell_id' in trackpy_df.columns:
                # Handle potential NaN cell_id values safely
                cell_df = trackpy_df[trackpy_df['cell_id'] == cell_id].copy()
            else:
                cell_df = trackpy_df.copy()
            
            if len(cell_df) == 0:
                continue
            
            particles = cell_df['particle'].unique()
            cell_msd_values = []
            cell_D_values = []
            
            for particle_id in particles:
                traj = cell_df[cell_df['particle'] == particle_id]
                if len(traj) < 2:
                    continue
                try:
                    em = tp.emsd(traj, mpp=float(microns_per_pixel), fps=1.0/float(step_size_in_sec))
                    msd_dict[(cell_id, particle_id)] = em
                    max_lag = max(max_lag, len(em))
                    cell_msd_values.append(em)
                    
                    # Calculate D for this trajectory
                    max_fit = min(self.msd_fit_points_spinbox.value(), len(em))
                    if max_fit >= 2:
                        is_3d = self.msd_data.get('is_3d', False) if hasattr(self, 'msd_data') and self.msd_data else False
                        divisor = 6 if is_3d else 4
                        slope, intercept, r_val, _, _ = linregress(em.index[:max_fit], em.values[:max_fit])
                        # Ensure slope is a scalar (not a Series)
                        slope = float(slope)
                        D = slope / divisor if slope > 0 else 0.0
                        cell_D_values.append(D)
                except Exception as e:
                    logging.debug(f"MSD: Failed for Cell {cell_id}, particle {particle_id}: {e}")
                    continue
            
            # Store per-cell summary
            if cell_D_values:
                self.msd_per_cell[cell_id] = {
                    'D_values': cell_D_values,
                    'D_mean': np.mean(cell_D_values),
                    'D_std': np.std(cell_D_values),
                    'n_particles': len(cell_D_values),
                    'msd_values': cell_msd_values
                }
        
        # Create DataFrame with time as first column and MSD per trajectory_X_cell_Y
        if msd_dict:
            # Get all unique lag times
            all_times = set()
            for em in msd_dict.values():
                all_times.update(em.index.tolist())
            all_times = sorted(all_times)
            
            # Build DataFrame
            df_msd = pd.DataFrame({'time_lag_s': all_times})
            for (cid, pid), em in sorted(msd_dict.items()):
                col_name = f'msd_traj_{pid}_cell_{cid}'
                df_msd[col_name] = df_msd['time_lag_s'].map(lambda t, em=em: em.get(t, np.nan) if t in em.index else np.nan)
            
            self.msd_per_trajectory = df_msd

    def plot_msd(self):
        """Plot MSD vs lag time with per-cell coloring and optional log-log scale."""
        if self.msd_data is None:
            return
        
        em_um2 = self.msd_data['em_um2']
        fit_times = self.msd_data['fit_times']
        fit_line_msd = self.msd_data['fit_line_msd']
        D_um2_s = self.msd_data['D_um2_s']
        is_3d = self.msd_data['is_3d']
        
        self.figure_msd.clear()
        self.ax_msd = self.figure_msd.add_subplot(111)
        
        # Apply dark mode styling
        self.figure_msd.patch.set_facecolor('black')
        self.ax_msd.set_facecolor('black')
        self.ax_msd.tick_params(colors='white', which='both')
        for spine in self.ax_msd.spines.values():
            spine.set_color('white')
        self.ax_msd.xaxis.label.set_color('white')
        self.ax_msd.yaxis.label.set_color('white')
        self.ax_msd.title.set_color('white')
        
        # Per-cell color palette (same as Distribution/Time Course tabs)
        cell_colors = [
            '#00FFFF', '#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3',
            '#F38181', '#AA96DA', '#FCBAD3', '#A8D8EA', '#FF9F43',
            '#6C5CE7', '#00CEC9', '#FD79A8', '#FFEAA7', '#74B9FF'
        ]
        
        # Check if we have per-cell data
        if hasattr(self, 'msd_per_cell') and self.msd_per_cell:
            # Plot per-cell MSD curves (mean with error)
            n_cells = len(self.msd_per_cell)
            stats_lines = []
            MIN_TRAJECTORIES_PER_LAG = 5  # Filter lags with fewer trajectories
            MIN_PARTICLES_PER_CELL = 10  # Minimum particles to display a cell
            
            for i, (cell_id, cell_data) in enumerate(sorted(self.msd_per_cell.items())):
                color = cell_colors[i % len(cell_colors)]
                
                # Skip cells with too few particles
                if cell_data['n_particles'] < MIN_PARTICLES_PER_CELL:
                    logging.debug(f"MSD: Skipping Cell {cell_id} (only {cell_data['n_particles']} particles, need {MIN_PARTICLES_PER_CELL})")
                    continue
                
                # Compute mean MSD across all trajectories for this cell
                msd_values_list = cell_data['msd_values']
                if not msd_values_list:
                    continue
                
                # Collect all lag times and MSD values
                all_lags = set()
                for em in msd_values_list:
                    all_lags.update(em.index.tolist())
                all_lags = sorted(all_lags)
                
                # Calculate mean and std MSD at each lag, filter by min trajectories
                mean_msd = []
                std_msd = []
                valid_lags = []
                for lag in all_lags:
                    values_at_lag = []
                    for em in msd_values_list:
                        if lag in em.index:
                            val = em.get(lag)
                            # Handle case where val could be scalar, array, or Series
                            if val is None:
                                continue
                            # If it's an array/series with multiple elements, take the mean
                            if hasattr(val, '__len__') and not isinstance(val, str):
                                if len(val) == 1:
                                    val = float(val.iloc[0]) if hasattr(val, 'iloc') else float(val[0])
                                else:
                                    val = float(np.nanmean(val))
                            elif hasattr(val, 'item'):
                                val = val.item()
                            else:
                                val = float(val)
                            if not np.isnan(val):
                                values_at_lag.append(val)
                    if len(values_at_lag) >= MIN_TRAJECTORIES_PER_LAG:
                        mean_msd.append(np.mean(values_at_lag))
                        std_msd.append(np.std(values_at_lag))
                        valid_lags.append(lag)
                
                if not valid_lags:
                    continue
                
                mean_msd = np.array(mean_msd)
                std_msd = np.array(std_msd)
                valid_lags = np.array(valid_lags)
                
                # Plot mean MSD with error bars
                self.ax_msd.errorbar(valid_lags, mean_msd, yerr=std_msd, 
                                    fmt='o', color=color, markersize=4, 
                                    linewidth=1.5, alpha=0.8, capsize=2)
                
                # Add per-cell linear fit line
                D_mean = cell_data['D_mean']
                n_particles = cell_data['n_particles']
                divisor = 6 if is_3d else 4  # 3D: D = slope/6, 2D: D = slope/4
                slope = D_mean * divisor  # Reverse the calculation to get slope
                # Use the global fit_times max (based on Fit Points spinbox) for consistent x-range
                fit_max_time = float(fit_times[-1]) if len(fit_times) > 0 else valid_lags.max()
                fit_x = np.linspace(0, fit_max_time, 50)
                fit_y = slope * fit_x  # MSD = slope * t (assuming intercept = 0)
                self.ax_msd.plot(fit_x, fit_y, '--', color=color, linewidth=2, alpha=0.7)
                
                # Add to legend with D value
                cell_label = f"Cell {cell_id}: D={D_mean:.2e} (n={n_particles})"
                self.ax_msd.plot([], [], 'o--', color=color, label=cell_label)
                
                stats_lines.append(f"Cell {cell_id}: D={D_mean:.2e} ± {cell_data['D_std']:.2e} µm²/s")
            
            # Plot overall fit line
            fit_line_times = np.linspace(0.0, float(fit_times[-1]) * 1.2, 50)
            self.ax_msd.plot(fit_line_times, fit_line_msd[:len(fit_line_times)] if len(fit_line_msd) >= 50 else fit_line_msd,
                            '-', linewidth=2, color='white', linestyle='--', label=f'Overall D={D_um2_s:.2e}')
            
            # Add stats text box
            if stats_lines:
                stats_text = '\n'.join(stats_lines[:10])  # Limit to 10 cells
                self.ax_msd.text(0.98, 0.02, stats_text, transform=self.ax_msd.transAxes,
                                fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                                color='white', family='monospace',
                                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='gray'))
        else:
            # Fallback: single combined MSD plot
            self.ax_msd.plot(em_um2.index, em_um2.values, 'o', alpha=0.6, color='cyan', label='MSD data')
            self.ax_msd.plot(fit_times, em_um2.values[:len(fit_times)], 'o', markersize=8, color='orange', label='Fitted region')
            fit_line_times = np.linspace(0.0, float(fit_times[-1]) * 1.2, 50)
            self.ax_msd.plot(fit_line_times, fit_line_msd[:len(fit_line_times)] if len(fit_line_msd) >= 50 else fit_line_msd,
                            '-', linewidth=2, color='lime', label=f'D = {D_um2_s:.2e} µm²/s')
        
        # Labels
        dim_text = "3D" if is_3d else "2D"
        self.ax_msd.set_xlabel('Time lag (s)')
        self.ax_msd.set_ylabel(r'MSD [µm²]')
        self.ax_msd.set_title(f'Mean Squared Displacement ({dim_text})')
        self.ax_msd.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white', fontsize=8)
        self.ax_msd.grid(True, which='both', color='gray', linestyle='--', linewidth=0.3, alpha=0.5)
        
        # Apply log-log scale if checked
        if self.msd_loglog_checkbox.isChecked():
            self.ax_msd.set_xscale('log')
            self.ax_msd.set_yscale('log')
        
        self.figure_msd.tight_layout()
        self.canvas_msd.draw_idle()

    def export_msd_dataframe(self):
        """Export MSD vs lag time for all trajectories as CSV."""
        if self.msd_per_trajectory is None:
            QMessageBox.warning(self, "No Data", "No MSD data to export. Calculate MSD first.")
            return
        
        # Generate filename
        base_name = self.file_label.text().split('.')[0] if hasattr(self, 'file_label') else 'data'
        base_name = re.sub(r'[^\w\-_\. ]', '_', base_name)
        default_name = f"msd_dataframe_{base_name}.csv"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export MSD DataFrame", default_name, "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.msd_per_trajectory.to_csv(file_path, index=False)
            QMessageBox.information(self, "Export Complete", f"MSD DataFrame exported to:\n{file_path}")

    def export_msd_plot(self):
        """Export MSD plot as PNG."""
        if self.msd_data is None:
            QMessageBox.warning(self, "No Data", "No MSD plot to export. Calculate MSD first.")
            return
        
        # Generate filename
        base_name = self.file_label.text().split('.')[0] if hasattr(self, 'file_label') else 'data'
        base_name = re.sub(r'[^\w\-_\. ]', '_', base_name)
        default_name = f"msd_plot_{base_name}.png"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export MSD Plot", default_name, "PNG Files (*.png);;All Files (*)"
        )
        
        if file_path:
            self.figure_msd.savefig(file_path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Export Complete", f"MSD plot exported to:\n{file_path}")

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
        """Reset Correlation tab to default state (called when new image is loaded)."""
        # Clear plot
        self.figure_correlation.clear()
        self.ax_correlation = self.figure_correlation.add_subplot(111)
        self.ax_correlation.set_facecolor('black')
        self.ax_correlation.axis('off')
        self.ax_correlation.text(
            0.5, 0.5, 'Press "Run" to compute correlations.',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12, color='white',
            transform=self.ax_correlation.transAxes
        )
        self.canvas_correlation.draw()
        
        # Clear results
        self.correlation_results = []
        self.current_total_plots = None
        
        # Reset channel checkboxes
        for checkbox in self.channel_checkboxes:
            checkbox.setChecked(False)
        
        # Reset decorrelation threshold slider
        if hasattr(self, 'decorr_threshold_slider'):
            self.decorr_threshold_slider.blockSignals(True)
            self.decorr_threshold_slider.setValue(1)  # 0.01
            self.decorr_threshold_slider.blockSignals(False)
            self.decorr_value_label.setText("0.01")
            self.de_correlation_threshold = 0.01
        
        # Reset X-axis (max lag) slider
        if hasattr(self, 'x_max_lag_slider'):
            self.x_max_lag_slider.blockSignals(True)
            self.x_max_lag_slider.setValue(200)
            self.x_max_lag_slider.blockSignals(False)
            self.x_lag_label.setText("200 frames")
        
        # Reset Y-axis percentile sliders
        if hasattr(self, 'y_min_percentile_slider'):
            self.y_min_percentile_slider.blockSignals(True)
            self.y_min_percentile_slider.setValue(0)
            self.y_min_percentile_slider.blockSignals(False)
            self.y_min_label.setText("0%")
        if hasattr(self, 'y_max_percentile_slider'):
            self.y_max_percentile_slider.blockSignals(True)
            self.y_max_percentile_slider.setValue(100)
            self.y_max_percentile_slider.blockSignals(False)
            self.y_max_label.setText("100%")
        
        # Reset Min % Data slider
        if hasattr(self, 'min_pct_data_slider'):
            self.min_pct_data_slider.blockSignals(True)
            self.min_pct_data_slider.setValue(30)  # 30% default
            self.min_pct_data_slider.blockSignals(False)
            self.min_pct_label.setText("30% (-- / -- frames)")
            self.min_percentage_data_in_trajectory = 0.30
        
        # Reset Start Lag and Fit Lag
        if hasattr(self, 'start_lag_input'):
            self.start_lag_input.setValue(0)
        if hasattr(self, 'fit_lag_slider'):
            self.fit_lag_slider.blockSignals(True)
            self.fit_lag_slider.setValue(99)
            self.fit_lag_slider.blockSignals(False)
            self.fit_lag_label.setText("99 frames")
        if hasattr(self, 'index_max_lag_for_fit_input'):
            self.index_max_lag_for_fit_input.setValue(99)
        
        # Reset Quality Controls
        if hasattr(self, 'correct_baseline_checkbox'):
            self.correct_baseline_checkbox.setChecked(True)
        if hasattr(self, 'remove_outliers_checkbox'):
            self.remove_outliers_checkbox.setChecked(True)
        if hasattr(self, 'multiTauCheck'):
            self.multiTauCheck.setChecked(False)
        if hasattr(self, 'snr_threshold_for_acf'):
            self.snr_threshold_for_acf.setValue(0.1)
            self.snr_threshold_for_acf_value = 0.1
        
        # Reset fit type radio
        if hasattr(self, 'linear_radio'):
            self.linear_radio.setChecked(True)

    # Note: reset_crops_tab removed - Crops tab has been deprecated

    def reset_manual_colocalization(self):
        """Reset manual colocalization verification subtabs."""
        # Reset Verify Visual
        if hasattr(self, 'verify_visual_scroll_area'):
            self.verify_visual_scroll_area.setWidget(QWidget())
        if hasattr(self, 'verify_visual_checkboxes'):
            self.verify_visual_checkboxes = []
        if hasattr(self, 'verify_visual_stats_label'):
            self.verify_visual_stats_label.setText("Run Visual colocalization first, then click Populate")
        
        # Reset Verify Distance
        if hasattr(self, 'verify_distance_scroll_area'):
            self.verify_distance_scroll_area.setWidget(QWidget())
        if hasattr(self, 'verify_distance_checkboxes'):
            self.verify_distance_checkboxes = []
        if hasattr(self, 'verify_distance_stats_label'):
            self.verify_distance_stats_label.setText("Run Distance colocalization first, then click Populate")

    def reset_cellpose_tab(self):
        """Reset Cellpose tab state, masks, and UI controls to defaults."""
        # Clear masks (YX)
        self.cellpose_masks_cyto = None
        self.cellpose_masks_nuc = None
        # Clear TYX masks
        self.cellpose_masks_cyto_tyx = None
        self.cellpose_masks_nuc_tyx = None
        self.use_tyx_masks = False
        
        # Clear original masks (for size sliders)
        self._original_cellpose_masks_cyto = None
        self._original_cellpose_masks_nuc = None
        self._original_cellpose_masks_cyto_tyx = None
        self._original_cellpose_masks_nuc_tyx = None
        
        # Reset cytosol size slider to 0
        if hasattr(self, 'cyto_size_slider'):
            self.cyto_size_slider.blockSignals(True)
            self.cyto_size_slider.setValue(0)
            self.cyto_size_slider.blockSignals(False)
        if hasattr(self, 'cyto_size_label'):
            self.cyto_size_label.setText("0")
        
        # Reset nucleus size slider to 0
        if hasattr(self, 'nuc_size_slider'):
            self.nuc_size_slider.blockSignals(True)
            self.nuc_size_slider.setValue(0)
            self.nuc_size_slider.blockSignals(False)
        if hasattr(self, 'nuc_size_label'):
            self.nuc_size_label.setText("0")
        

        # Reset frame/channel indices
        # NOTE: cellpose_current_frame and cellpose_current_channel are deprecated
        # Cellpose now uses segmentation_current_frame and segmentation_current_channel
        
        # Reset Cytosol parameters to defaults
        if hasattr(self, 'cellpose_cyto_model_input'):
            self.cellpose_cyto_model_input.setCurrentText('cyto3')
        # Note: Cellpose channel spinboxes removed - channel is determined by left panel selection
        if hasattr(self, 'cellpose_cyto_diameter_input'):
            self.cellpose_cyto_diameter_input.setValue(120)
        if hasattr(self, 'chk_optimize_cyto'):
            self.chk_optimize_cyto.setChecked(False)
        
        # Reset Nucleus parameters to defaults
        if hasattr(self, 'cellpose_nuc_model_input'):
            self.cellpose_nuc_model_input.setCurrentText('nuclei')
        # Note: Cellpose channel spinbox removed - channel is determined by left panel selection
        if hasattr(self, 'cellpose_nuc_diameter_input'):
            self.cellpose_nuc_diameter_input.setValue(60)
        if hasattr(self, 'chk_optimize_nuc'):
            self.chk_optimize_nuc.setChecked(False)
        
        # Reset Improve Segmentation checkboxes
        if hasattr(self, 'chk_remove_border_cells'):
            self.chk_remove_border_cells.setChecked(False)
        if hasattr(self, 'chk_remove_unpaired_cells'):
            self.chk_remove_unpaired_cells.setChecked(False)
        
        # Refresh the shared segmentation display
        # (Cellpose now shares the segmentation canvas, so just refresh it)
        if hasattr(self, 'figure_segmentation'):
            self.plot_segmentation()

    def reset_all_state(self):
        """
        Unified reset method called when loading a new image.
        Resets all tabs, clears state variables, and prepares the GUI for new data.
        """
        # Reset all tab displays
        self.reset_display_tab()
        self.reset_registration_tab()
        self.reset_segmentation_tab()
        self.reset_photobleaching_tab()
        self.reset_tracking_tab()
        self.reset_msd_tab()
        self.reset_distribution_tab()
        self.reset_time_course_tab()
        self.reset_correlation_tab()
        self.reset_colocalization_tab()  # Includes Visual, Distance, and Manual sub-tabs
        # Note: Crops tab has been removed - reset_crops_tab() is deprecated
        self.reset_tracking_visualization_tab()
        self.reset_export_comment()
        self.reset_cellpose_tab()
        
        # Reset shared state variables
        self.has_tracked = False
        self.photobleaching_calculated = False
        self.detected_spots_frame = None
        self.corrected_image = None
        self.df_tracking = pd.DataFrame()
        self._active_mask_source = 'segmentation'
        
        # Reset multi-channel tracking data
        self.multi_channel_tracking_data = {}
        self.tracked_channels = []
        self.tracking_thresholds = {}
        self.auto_threshold_per_channel = {}
        self.tracking_parameters_per_channel = {}
        self.primary_tracking_channel = None
        
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
        """Delegate to plot_intensity_histogram for per-cell overlay histograms."""
        # Update tracking channel combo with tracked channels (single channel only)
        if hasattr(self, 'distribution_tracking_channel_combo'):
            self.distribution_tracking_channel_combo.clear()
            
            if self.tracked_channels:
                # Multi-channel tracking: show each tracked channel
                for ch in sorted(self.tracked_channels):
                    self.distribution_tracking_channel_combo.addItem(f"Ch {ch}", ch)
            elif hasattr(self, 'df_tracking') and not self.df_tracking.empty and 'spot_type' in self.df_tracking.columns:
                # Get unique spot_type values from data
                unique_channels = sorted(self.df_tracking['spot_type'].unique())
                for ch in unique_channels:
                    self.distribution_tracking_channel_combo.addItem(f"Ch {ch}", ch)
            else:
                # No tracking data at all
                self.distribution_tracking_channel_combo.addItem("No tracked channels", -1)
        
        self.plot_intensity_histogram()

    # Note: display_crops_plot removed - Crops tab has been deprecated


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
            
            # Check if we should show individual traces
            show_traces = getattr(self, 'show_individual_traces_checkbox', None)
            show_traces = show_traces.isChecked() if show_traces else False
            
            # Plot individual traces FIRST (behind mean) if enabled
            if show_traces:
                for r in results:
                    corr_array = r.get('correlations_array')
                    if corr_array is None or len(corr_array) == 0:
                        continue
                    lags = np.array(r['lags'])
                    step_size = r['step_size_in_sec']
                    normalize = r.get('normalize_plot_with_g0', False)
                    start_lag = r.get('start_lag', 0)
                    color = list_colors_default[r['channel'] % len(list_colors_default)]
                    
                    # Subsample if too many traces (performance guard)
                    n_traces = corr_array.shape[0]
                    max_traces_to_plot = 100
                    if n_traces > max_traces_to_plot:
                        indices = np.linspace(0, n_traces - 1, max_traces_to_plot, dtype=int)
                    else:
                        indices = np.arange(n_traces)
                    
                    for trace_idx in indices:
                        trace = corr_array[trace_idx]
                        if normalize and len(trace) > start_lag and trace[start_lag] != 0:
                            trace = trace / trace[start_lag]
                        # Convert lags to time
                        time_lags = lags * step_size
                        ax.plot(time_lags, trace, color=color, alpha=0.15, linewidth=0.5)
            
            # Now plot the mean curves on top
            for idx, r in enumerate(results):
                # Use actual channel number for color, not loop index
                color = list_colors_default[r['channel'] % len(list_colors_default)]
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
            # Update title to indicate per-cell if applicable
            has_cells = any(r.get('cell_id') is not None for r in results)
            title = 'Autocorrelation (per-cell)' if has_cells else 'Autocorrelation (all channels)'
            ax.set_title(title, color='white')
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
                # Per-cell coloring: use cell_id if available
                cell_id = r.get('cell_id')
                if cell_id is not None:
                    color = list_colors_default[r['channel'] % len(list_colors_default)]
                    title = f'Cell {cell_id} - Ch {r["channel"]} (n={r.get("n_trajectories", "?")})'
                else:
                    color = list_colors_default[r['channel'] % len(list_colors_default)]
                    title = f'Autocorrelation Channel {r["channel"]}'
                
                # Plot individual traces FIRST if enabled
                show_traces = getattr(self, 'show_individual_traces_checkbox', None)
                show_traces = show_traces.isChecked() if show_traces else False
                if show_traces:
                    corr_array = r.get('correlations_array')
                    if corr_array is not None and len(corr_array) > 0:
                        lags = np.array(r['lags'])
                        step_size = r['step_size_in_sec']
                        normalize = r.get('normalize_plot_with_g0', False)
                        start_lag = r.get('start_lag', 0)
                        
                        # Subsample if too many traces
                        n_traces = corr_array.shape[0]
                        max_traces_to_plot = 100
                        if n_traces > max_traces_to_plot:
                            indices = np.linspace(0, n_traces - 1, max_traces_to_plot, dtype=int)
                        else:
                            indices = np.arange(n_traces)
                        
                        for trace_idx in indices:
                            trace = corr_array[trace_idx]
                            if normalize and len(trace) > start_lag and trace[start_lag] != 0:
                                trace = trace / trace[start_lag]
                            time_lags = lags * step_size
                            ax.plot(time_lags, trace, color=color, alpha=0.15, linewidth=0.5)
                
                # Plot mean on top
                self.plots.plot_autocorrelation(
                    mean_correlation                   = r['mean_corr'],
                    error_correlation                  = r['std_corr'],
                    lags                               = r['lags'],
                    time_interval_between_frames_in_seconds = r['step_size_in_sec'],
                    channel_label                      = f"Cell {cell_id}" if cell_id is not None else r['channel'],
                    axes                               = ax,
                    plot_title                         = title,
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
            # Match list_colors_default: 0=Green, 1=Magenta, 2=Yellow, 3=Red
            if ch_idx == 0: return 'green'
            if ch_idx == 1: return 'magenta'
            if ch_idx == 2: return 'yellow'
            if ch_idx == 3: return 'red'
            # fallback for additional channels
            return 'cyan'
        
        # Color palette for cells (bright colors for dark background)
        cell_colors = ['cyan', 'magenta', 'lime', 'orange', 'yellow', 'red', 
                       'deepskyblue', 'hotpink', 'chartreuse', 'coral', 
                       'gold', 'tomato', 'aqua', 'violet', 'springgreen']

        # Helper to plot one channel's data with per-cell coloring
        # Returns (min_val, max_val) for y-axis scaling
        def plot_channel_data(ch_idx, color_override=None):
            field_name = f"{data_type}_ch_{ch_idx}"
            if field_name not in self.df_tracking.columns:
                return None, None

            # Get unique cell IDs
            if 'cell_id' in self.df_tracking.columns:
                cell_ids = sorted(self.df_tracking['cell_id'].dropna().unique())
            else:
                cell_ids = [0]
            
            n_cells = len(cell_ids)
            all_y_mins = []
            all_y_maxs = []
            
            # Calculate alpha based on number of cells
            trace_alpha = max(0.2, min(0.5, 0.8 / (n_cells ** 0.5)))
            mean_alpha = max(0.5, min(0.9, 1.0 / (n_cells ** 0.3)))
            
            for cell_idx, cell_id in enumerate(cell_ids):
                # Filter data for this cell
                cell_df = self.df_tracking[self.df_tracking['cell_id'] == cell_id]
                if cell_df.empty:
                    continue
                
                # Get color for this cell
                cell_color = cell_colors[cell_idx % len(cell_colors)] if n_cells > 1 else (color_override or 'cyan')
                
                intensity_array = mi.Utilities().df_trajectories_to_array(
                    dataframe=cell_df,
                    selected_field=field_name,
                    fill_value=np.nan,
                    total_frames=total_frames
                )
                
                if intensity_array.size == 0 or np.all(np.isnan(intensity_array)):
                    continue

                # Plot individual traces if option is enabled
                if self.show_traces_checkbox.isChecked() and channel_text != "All":
                    for idx in range(intensity_array.shape[0]):
                        trace = intensity_array[idx, :]
                        if np.all(np.isnan(trace)):
                            continue
                        # Apply moving average to traces if requested
                        if window_size > 1:
                            trace = apply_moving_average(trace, window_size)
                        # Normalize trace if requested
                        if normalize:
                            min_t = np.nanmin(trace)
                            max_t = np.nanmax(trace)
                            if max_t > min_t:
                                trace = (trace - min_t) / (max_t - min_t)
                        self.ax_time_course.plot(time_points, trace, '-', color=cell_color,
                                                linewidth=0.8, alpha=trace_alpha, label='_nolegend_')

                # Calculate mean and std dev for this cell
                mean_time_intensity = np.nanmean(intensity_array, axis=0)
                std_time_intensity  = np.nanstd(intensity_array, axis=0)
                mean_time_intensity = np.nan_to_num(mean_time_intensity)
                std_time_intensity  = np.nan_to_num(std_time_intensity)

                # Apply Moving Average
                if window_size > 1:
                    mean_time_intensity = apply_moving_average(mean_time_intensity, window_size)
                    std_time_intensity = apply_moving_average(std_time_intensity, window_size)

                if normalize:
                    min_v = np.min(mean_time_intensity)
                    max_v = np.max(mean_time_intensity)
                    if max_v > min_v:
                        mean_time_intensity = (mean_time_intensity - min_v) / (max_v - min_v)
                        std_time_intensity = std_time_intensity / (max_v - min_v)

                # Label for legend
                if n_cells > 1:
                    label_text = f"Cell {int(cell_id)}"
                    if channel_text == "All":
                        label_text = f"C{int(cell_id)} Ch{ch_idx}"
                else:
                    label_text = f"Ch {ch_idx}" if channel_text == "All" else "Mean"
                
                self.ax_time_course.plot(time_points, mean_time_intensity, 'o-',
                                        color=cell_color, linewidth=2, label=label_text, 
                                        alpha=mean_alpha, zorder=3, markersize=3)
                
                # Fill between for std dev
                fill_alpha = 0.1 if n_cells > 3 else 0.15
                self.ax_time_course.fill_between(time_points,
                                                mean_time_intensity - std_time_intensity,
                                                mean_time_intensity + std_time_intensity,
                                                color=cell_color, alpha=fill_alpha, 
                                                label='_nolegend_', zorder=1)
                
                # Track y-range
                if normalize:
                    all_y_mins.append(0.0)
                    all_y_maxs.append(1.0)
                else:
                    all_y_mins.append(np.nanpercentile(intensity_array, lower_percentile))
                    all_y_maxs.append(np.nanpercentile(intensity_array, upper_percentile))
            
            # Return range for axis scaling
            if all_y_mins and all_y_maxs:
                return min(all_y_mins), max(all_y_maxs)
            else:
                return None, None

        # --- Plotting Logic ---
        
        if data_type == "particles":
            # Particles per cell over time
            # Per-cell color palette (same as MSD and Distribution tabs)
            cell_colors = [
                '#FF6B6B', '#00FFFF', '#4ECDC4', '#FFE66D', '#95E1D3',
                '#F38181', '#AA96DA', '#FCBAD3', '#A8D8EA', '#FF9F43',
                '#6C5CE7', '#00CEC9', '#FD79A8', '#FFEAA7', '#74B9FF'
            ]
            
            # Filter by tracking channel (spot_type) - single channel required
            df_to_plot = self.df_tracking.copy()
            tracking_ch = None
            if hasattr(self, 'time_course_tracking_channel_combo'):
                tracking_ch = self.time_course_tracking_channel_combo.currentData()
                # Check if placeholder "No tracked channels" is selected
                if tracking_ch == -1:
                    self.ax_time_course.text(0.5, 0.5, 'No tracked channels available.\nPlease run tracking first.',
                                             transform=self.ax_time_course.transAxes,
                                             ha='center', va='center', color='white', fontsize=12)
                    self.canvas_time_course.draw_idle()
                    return
                # Filter by spot_type
                if 'spot_type' in df_to_plot.columns:
                    df_to_plot = df_to_plot[df_to_plot['spot_type'] == tracking_ch]
            
            if df_to_plot.empty:
                self.ax_time_course.text(0.5, 0.5, 'No data for selected tracking channel.',
                                         transform=self.ax_time_course.transAxes,
                                         ha='center', va='center', color='white', fontsize=12)
                self.canvas_time_course.draw_idle()
                return
            
            # Get unique cell IDs
            if 'cell_id' in df_to_plot.columns:
                cell_ids = sorted(df_to_plot['cell_id'].dropna().unique())
            else:
                cell_ids = [0]  # Default to single "cell" if no cell_id column
            
            all_frames = np.arange(total_frames)
            all_y_data = []  # For mean calculation
            
            for i, cell_id in enumerate(cell_ids):
                if 'cell_id' in df_to_plot.columns:
                    cell_df = df_to_plot[df_to_plot['cell_id'] == cell_id]
                else:
                    cell_df = df_to_plot
                
                particles_per_frame = cell_df.groupby('frame')['particle'].nunique()
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
                
                all_y_data.append(y_data)
                color = cell_colors[i % len(cell_colors)]
                self.ax_time_course.plot(time_points, y_data, 'o-', color=color, 
                                          linewidth=1.5, markersize=3, alpha=0.8,
                                          label=f"Cell {int(cell_id)}")
            
            # Plot overall mean if multiple cells
            if len(cell_ids) > 1 and all_y_data:
                mean_y = np.mean(all_y_data, axis=0)
                self.ax_time_course.plot(time_points, mean_y, '--', color='white', 
                                          linewidth=2, alpha=0.9, label="Mean")
            
            self.ax_time_course.set_title("Number of Particles vs Time (per Cell)", fontsize=10, color='white')
            self.ax_time_course.legend(loc='upper right', fontsize=8, framealpha=0.7)
            
            if normalize:
                 self.ax_time_course.set_ylim([-0.1, 1.1])
            else:
                 # Find max across all cells
                 max_particles = max(np.max(y) for y in all_y_data) if all_y_data else 1
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
        # Stop any running playback when switching tabs
        self.stop_all_playback()
        
        # Tab index mapping (must match order in initUI):
        # Cellpose is now a sub-tab within Segmentation
        # Coloc Manual is now a sub-tab within Colocalization (Phase 1 consolidation)
        # 0=Import, 1=Registration, 2=Segmentation, 3=Photobleaching, 4=Tracking, 
        # 5=MSD, 6=Distribution, 7=Time Course, 8=Correlation, 9=Colocalization,
        # 10=Tracking Visualization, 11=Export
        if index == 0:  # Import
            self.plot_image()
        elif index == 1:  # Registration
            self.plot_registration_panels()
        elif index == 2:  # Segmentation (includes Cellpose sub-tab)
            # Check which sub-tab is active
            if hasattr(self, 'segmentation_method_tabs'):
                current_subtab = self.segmentation_method_tabs.currentIndex()
                if current_subtab == 1 or current_subtab == 3:  # Cellpose or Import sub-tab
                    self.plot_cellpose_results()
                else:
                    self.plot_segmentation()
            else:
                self.plot_segmentation()
        elif index == 3:  # Photobleaching
            self.plot_photobleaching()
        elif index == 4:  # Tracking
            # Reset MSD results when returning to tracking tab (user may add new channels)
            self.reset_msd_tab()
            self.plot_tracking()
            self.update_threshold_histogram()
        elif index == 5:  # MSD
            # Update tracking channel combo with tracked channels (single channel only)
            if hasattr(self, 'msd_tracking_channel_combo'):
                self.msd_tracking_channel_combo.clear()
                
                if self.tracked_channels:
                    # Multi-channel tracking: show each tracked channel
                    for ch in sorted(self.tracked_channels):
                        self.msd_tracking_channel_combo.addItem(f"Ch {ch}", ch)
                elif hasattr(self, 'df_tracking') and not self.df_tracking.empty and 'spot_type' in self.df_tracking.columns:
                    # Get unique spot_type values from data
                    unique_channels = sorted(self.df_tracking['spot_type'].unique())
                    for ch in unique_channels:
                        self.msd_tracking_channel_combo.addItem(f"Ch {ch}", ch)
                else:
                    # No tracking data at all
                    self.msd_tracking_channel_combo.addItem("No tracked channels", -1)
        elif index == 6:  # Distribution
            self.plot_distribution()
        elif index == 7:  # Time Course
            # Update tracking channel combo with tracked channels (single channel only)
            if hasattr(self, 'time_course_tracking_channel_combo'):
                self.time_course_tracking_channel_combo.clear()
                
                if self.tracked_channels:
                    # Multi-channel tracking: show each tracked channel
                    for ch in sorted(self.tracked_channels):
                        self.time_course_tracking_channel_combo.addItem(f"Ch {ch}", ch)
                elif hasattr(self, 'df_tracking') and not self.df_tracking.empty and 'spot_type' in self.df_tracking.columns:
                    # Get unique spot_type values from data
                    unique_channels = sorted(self.df_tracking['spot_type'].unique())
                    for ch in unique_channels:
                        self.time_course_tracking_channel_combo.addItem(f"Ch {ch}", ch)
                else:
                    # No tracking data at all
                    self.time_course_tracking_channel_combo.addItem("No tracked channels", -1)
        elif index == 8:  # Correlation
            # Update tracking channel combo with tracked channels (single channel only)
            if hasattr(self, 'correlation_tracking_channel_combo'):
                self.correlation_tracking_channel_combo.clear()
                
                if self.tracked_channels:
                    # Multi-channel tracking: show each tracked channel
                    for ch in sorted(self.tracked_channels):
                        self.correlation_tracking_channel_combo.addItem(f"Ch {ch}", ch)
                elif hasattr(self, 'df_tracking') and not self.df_tracking.empty and 'spot_type' in self.df_tracking.columns:
                    # Get unique spot_type values from data
                    unique_channels = sorted(self.df_tracking['spot_type'].unique())
                    for ch in unique_channels:
                        self.correlation_tracking_channel_combo.addItem(f"Ch {ch}", ch)
                else:
                    # No tracking data at all
                    self.correlation_tracking_channel_combo.addItem("No tracked channels", -1)
            self.display_correlation_plot()
        elif index == 9:  # Colocalization (includes Visual, Distance, Manual sub-tabs)
            # Update Visual (ML/Intensity) sub-tab tracking channel combo
            if hasattr(self, 'colocalization_tracking_channel_combo'):
                self.colocalization_tracking_channel_combo.clear()
                
                if self.tracked_channels:
                    for ch in sorted(self.tracked_channels):
                        self.colocalization_tracking_channel_combo.addItem(f"Ch {ch}", ch)
                elif hasattr(self, 'df_tracking') and not self.df_tracking.empty and 'spot_type' in self.df_tracking.columns:
                    unique_channels = sorted(self.df_tracking['spot_type'].unique())
                    for ch in unique_channels:
                        self.colocalization_tracking_channel_combo.addItem(f"Ch {ch}", ch)
                else:
                    self.colocalization_tracking_channel_combo.addItem("No tracked channels", -1)
                
                # Auto-sync Reference channel to match tracking channel
                self.on_colocalization_tracking_channel_changed(0)
            
            # Populate cell selector with available cells (Visual sub-tab)
            self._populate_coloc_cell_selector()
            
            # Populate Distance sub-tab channel combos
            self.populate_distance_channel_combos()
            
            self.display_colocalization_plot()
            if hasattr(self, 'canvas_colocalization'):
                if hasattr(self, 'cid_zoom_coloc'):
                    try:
                        self.canvas_colocalization.mpl_disconnect(self.cid_zoom_coloc)
                    except Exception:
                        pass
                self.cid_zoom_coloc = self.canvas_colocalization.mpl_connect('motion_notify_event', self.on_colocalization_hover)
        elif index == 10:  # Tracking Visualization
            if not (getattr(self, 'has_tracked', False)) or self.df_tracking.empty:
                # Silently reset the visualization tab without warning
                self.reset_tracking_visualization_tab()
                return
            
            # Populate cell filter dropdown
            if hasattr(self, '_populate_vis_cell_filter'):
                self._populate_vis_cell_filter()
            
            self.tracked_particles_list.clear()
            # Use unique_particle if available, otherwise fall back to particle
            particle_col = 'unique_particle' if 'unique_particle' in self.df_tracking.columns else 'particle'
            for pid in sorted(self.df_tracking[particle_col].unique(), key=str):
                count = int((self.df_tracking[particle_col] == pid).sum())
                # Display cell_id and particle number if using unique_particle
                display_text = f"{pid}:{count}"
                item = QListWidgetItem(display_text)
                item.setData(Qt.UserRole, pid)
                self.tracked_particles_list.addItem(item)
            if self.tracked_particles_list.count() > 0 and self.tracked_particles_list.currentRow() < 0:
                self.tracked_particles_list.setCurrentRow(0)
            self.display_tracking_visualization()
        elif index == 11:  # Export
            # Check if any verification subtab has data
            has_verify_data = (hasattr(self, 'verify_visual_checkboxes') and len(self.verify_visual_checkboxes) > 0) or \
                              (hasattr(self, 'verify_distance_checkboxes') and len(self.verify_distance_checkboxes) > 0)
            if has_verify_data:
                self.extract_manual_colocalization_data(save_df=False)

# =============================================================================
# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
# =============================================================================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set modern font based on platform
    if sys.platform == 'win32':
        app.setFont(QFont("Segoe UI", 11))
    elif sys.platform == 'darwin':
        app.setFont(QFont("SF Pro", 11))
    else:
        app.setFont(QFont("Inter", 11))
    
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
    app.setApplicationName("MicroLive")
    app.setApplicationDisplayName("micro")
    app.setWindowIcon(QIcon(str(icon_file)))
    main_window = GUI(icon_path=icon_file)
    main_window.show()
    sys.exit(app.exec_())