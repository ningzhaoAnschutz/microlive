# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Authors: Luis U. Aguilera

'''

# Importing Libraries
import datetime
import gc
import getpass
import glob
import itertools
from itertools import compress
import math
#import multiprocessing
from joblib import Parallel, delayed, cpu_count
import os; from os import listdir; from os.path import isfile, join
import pathlib
from pathlib import Path
import platform
import re
import shutil
import socket
import sys
### Third-party imports
import pandas as pd
import numpy as np
import yaml
import tifffile
import pkg_resources
from fpdf import FPDF
from smb.SMBConnection import SMBConnection
import joypy
import zipfile
import seaborn as sns
import cv2
from IPython import get_ipython
import trackpy as tp
tp.quiet(suppress=True)
from PIL import Image, ImageDraw
import imageio
from functools import wraps,partial
import inspect
import traceback
import random

### Skimage imports
from skimage import img_as_float64, img_as_uint, exposure, filters, morphology, exposure,measure, filters,feature
from skimage.filters import gaussian, threshold_otsu
from skimage.io import imread
from skimage.morphology import erosion, binary_erosion, disk, remove_small_objects, remove_small_holes
from skimage.measure import find_contours, profile_line
#from skimage import measure, filters
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, blob_log

### Scipy imports
from scipy import signal, ndimage
from scipy.ndimage import gaussian_filter, binary_dilation, gaussian_filter1d
from scipy.optimize import curve_fit
#import scipy.stats as stats
from scipy.spatial.distance import cdist
from scipy.stats import linregress, pearsonr
from scipy.signal import find_peaks
from scipy.ndimage import center_of_mass

### Big FISH imports
import bigfish.stack as stack
import bigfish.plot as plot
import bigfish.detection as detection
#import bigfish.multistack as multistack
###  Cellpose imports
import contextlib, io
_f = io.StringIO()
with contextlib.redirect_stdout(_f), contextlib.redirect_stderr(_f):
    from cellpose import models, denoise
### Matplotlib imports
import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.path as mpltPath
from matplotlib.patches import Circle, Rectangle,Polygon
from  matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import cm
from matplotlib.colors import to_rgb
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import TextBox
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as matplotlib_path

from IPython.display import HTML
import ipywidgets as widgets
from IPython.display import display
### Lif file imports
from readlif.reader import LifFile
from bioio import BioImage
import xml.etree.ElementTree as ET
try:
    #import napari
    #from napari_animation import Animation
    import torch
except:
    pass

# Configure settings and warnings
import warnings
from PyQt5.QtWidgets import QMessageBox

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')
mpl.rc('image', cmap='viridis')
plt.style.use('ggplot')
font_props = {'size': 16}


#number_gpus = multiprocessing.cpu_count()
# if number_gpus >1 : # number_gpus
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] =  str(np.random.randint(0,number_gpus,1)[0])        

# Define a custom green colormap from black to green
cdict_green = {
    'red':   ((0.0, 0.0, 0.0),  # No red at any point
              (1.0, 0.0, 0.0)),
    'green': ((0.0, 0.0, 0.0),  # Start with no green
              (1.0, 1.0, 1.0)), # Full green at the end
    'blue':  ((0.0, 0.0, 0.0),  # No blue at any point
              (1.0, 0.0, 0.0))
}
cdict_magenta = {
    'red':   ((0.0, 0.0, 0.0),  # Start with no red
              (1.0, 1.0, 1.0)), # Full red at the end
    'green': ((0.0, 0.0, 0.0),  # No green at any point
              (1.0, 0.0, 0.0)),
    'blue':  ((0.0, 0.0, 0.0),  # Start with no blue
              (1.0, 1.0, 1.0))  # Full blue at the end
}
cdict_red = {
    'red':   ((0.0, 0.0, 0.0),  # Start with no red
              (1.0, 1.0, 1.0)), # Full red at the end
    'green': ((0.0, 0.0, 0.0),  # No green at any point
              (1.0, 0.0, 0.0)),
    'blue':  ((0.0, 0.0, 0.0),  # No blue at any point
              (1.0, 0.0, 0.0))
}
cdict_yellow = {
    'red':   ((0.0, 0.0, 0.0),  # Start with no red
              (1.0, 1.0, 1.0)), # Full red at the end
    'green': ((0.0, 0.0, 0.0),  # Start with no green
              (1.0, 1.0, 1.0)), # Full green at the end
    'blue':  ((0.0, 0.0, 0.0),  # No blue at any point
              (1.0, 0.0, 0.0))
}

yellow_colormap = LinearSegmentedColormap('BlackYellow', cdict_yellow)
red_colormap = LinearSegmentedColormap('BlackRed', cdict_red)
green_colormap = LinearSegmentedColormap('BlackGreen', cdict_green)
magenta_colormap = LinearSegmentedColormap('BlackMagenta', cdict_magenta)

cmap_list_imagej = [green_colormap, magenta_colormap ,yellow_colormap,red_colormap]

# List of colors for plotting in GUI
color_green = (0.0, 1.0, 0.0)  # No Red, Full Green, No Blue
color_magenta = (1.0, 0.0, 1.0)  # Full Red, No Green, Full Blue
color_red = (1.0, 0.0, 0.0)  # Full Red, No Green, No Blue
color_yellow = (1.0, 1.0, 0.0)  # Full Red, Full Green, No Blue

list_colors_default = [ color_green, color_magenta, color_yellow, color_red]


# Define a custom green colormap from black to green
cdict_green = {
    'red':   ((0.0, 0.0, 0.0),  # No red at any point
              (1.0, 0.0, 0.0)),
    'green': ((0.0, 0.0, 0.0),  # Start with no green
              (1.0, 1.0, 1.0)), # Full green at the end
    'blue':  ((0.0, 0.0, 0.0),  # No blue at any point
              (1.0, 0.0, 0.0))
}
cdict_magenta = {
    'red':   ((0.0, 0.0, 0.0),  # Start with no red
              (1.0, 1.0, 1.0)), # Full red at the end
    'green': ((0.0, 0.0, 0.0),  # No green at any point
              (1.0, 0.0, 0.0)),
    'blue':  ((0.0, 0.0, 0.0),  # Start with no blue
              (1.0, 1.0, 1.0))  # Full blue at the end
}
cdict_red = {
    'red':   ((0.0, 0.0, 0.0),  # Start with no red
              (1.0, 1.0, 1.0)), # Full red at the end
    'green': ((0.0, 0.0, 0.0),  # No green at any point
              (1.0, 0.0, 0.0)),
    'blue':  ((0.0, 0.0, 0.0),  # No blue at any point
              (1.0, 0.0, 0.0))
}
cdict_yellow = {
    'red':   ((0.0, 0.0, 0.0),  # Start with no red
              (1.0, 1.0, 1.0)), # Full red at the end
    'green': ((0.0, 0.0, 0.0),  # Start with no green
              (1.0, 1.0, 1.0)), # Full green at the end
    'blue':  ((0.0, 0.0, 0.0),  # No blue at any point
              (1.0, 0.0, 0.0))
}

yellow_colormap = LinearSegmentedColormap('BlackYellow', cdict_yellow)
red_colormap = LinearSegmentedColormap('BlackRed', cdict_red)
green_colormap = LinearSegmentedColormap('BlackGreen', cdict_green)
magenta_colormap = LinearSegmentedColormap('BlackMagenta', cdict_magenta)
cmap_list_imagej = [magenta_colormap, green_colormap,yellow_colormap,red_colormap]



class Banner:
    TEXT = r"""


    


:*************************************************: 
.......-----=====+++++*****##########%%%%%%%°°°°°°° 
.......°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°° 
888b     d8888888888 .d8888b. 8888888b.  .d88888b.  
8888b   d8888  888  d88P  Y88b888   Y88bd88P" "Y88b 
88888b.d88888  888  888    888888    888888     888 
888Y88888P888  888  888       888   d88P888     888 
888 Y888P 888  888  888       8888888P" 888     888 
888  Y8P  888  888  888    888888 T88b  888     888 
888   "   888  888  Y88b  d88P888  T88b Y88b. .d88P 
888       8888888888 "Y8888P" 888   T88b "Y88888P"  
                                    by luisub, 2025 
.......°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°° 
.......-----=====+++++*****##########%%%%%%%°°°°°°° 
:**************************************************: 

               
                            """


    IMAGE = r"""                    
                  @=#@     
                  #%-@     
                  ++*@     
            :+*@*%%@##    
          #%#%#%@@@**#   
         +@@@@@+*@@+*@ 
    =-@%@@%     @@#=*%
    *@-=#@+      #=-% 
      :          *=.%    
      :          *::%    
*****###%%------#:=%   
%=##**##*%#**##*++*    
    +##%%%-+*#**@:    
    ----##@##@%#-     
        %#%@%@@=      
    -@@@@#*=*+*-      
-=..:-::-===:-=*#-    
--=+*########%%%#-    
:****************:      
    """

    def __init__(self, text: str = None, image: str = None,
                 show: bool = True, padding: int = 5):
        """
        :param text:    Multiline ASCII banner text (defaults to Banner.TEXT)
        :param image:   Multiline ASCII art for the icon (defaults to Banner.IMAGE)
        :param show:    Whether to actually print when print_banner() is called
        :param padding: Number of spaces between text and image columns
        """
        self.text_lines  = (text  or self.TEXT ).strip("\n").splitlines()
        self.image_lines = (image or self.IMAGE).strip("\n").splitlines()
        self.show        = show
        self.padding     = " " * padding

    def print_banner(self):
        """Prints the banner (text + image) side by side."""
        if not self.show:
            return

        # figure out the widest text line
        text_width = max(len(line) for line in self.text_lines)

        # total rows needed (whichever is taller)
        total_rows = max(len(self.text_lines), len(self.image_lines))

        for i in range(total_rows):
            t   = self.text_lines[i]  if i < len(self.text_lines)  else ""
            img = self.image_lines[i] if i < len(self.image_lines) else ""
            print(t.ljust(text_width) + self.padding + img)


class Photobleaching:
    def __init__(
        self,
        image_TZYXC,
        mask_YX=None,
        show_plot=True,
        mode='inside_cell',
        precalulated_list_decay_rates=None,
        plot_name=None,
        radius=50,
        time_interval_seconds = None,
    ):
        
        if mode not in ['inside_cell', 'outside_cell', 'use_circular_region', 'entire_image']:
            raise ValueError(
                "mode must be 'inside_cell', 'outside_cell', 'use_circular_region', or 'entire_image'"
            )
        if mode == 'outside_cell' and mask_YX is None:
            raise ValueError("When mode='outside_cell', mask_YX must be provided.")

        # If entire_image mode, ignore any mask and use full image
        if mode == 'entire_image':  
            mask_YX = None      
        
        if time_interval_seconds is not None:
            self.time_interval_seconds = float(time_interval_seconds)
        else:
            self.time_interval_seconds = 1.0
        self.image_TZYXC = image_TZYXC
        self.show_plot = show_plot
        self.mode = mode
        self.precalculated_list_decay_rates = precalulated_list_decay_rates
        self.plot_name = plot_name
        self.radius = radius
        if mask_YX is not None:
            self.mask_YX = (mask_YX > 0)
            self.user_provided_mask = True
        else:
            self.mask_YX = np.ones((image_TZYXC.shape[2], image_TZYXC.shape[3]), dtype=bool)
            self.user_provided_mask = False
    
    def calculate_photobleaching(self):
        """
        Fits a simple exponential decay model: I(t) = I0 * exp(-k * t)
        Uses log-linear fitting exactly like the working reference code.
        """
        def create_circular_mask(h, w, center=None, radius=None):
            if center is None:
                center = (w // 2, h // 2)
            if radius is None:
                radius = min(center[0], center[1], w - center[0], h - center[1])
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            return dist <= radius
        h, w = self.image_TZYXC.shape[2], self.image_TZYXC.shape[3]
        if self.mode == 'use_circular_region':
            if self.user_provided_mask and self.mask_YX.any():
                indices = np.argwhere(self.mask_YX)
                cy, cx = indices.mean(axis=0)
                cy, cx = int(cy), int(cx)
            else:
                cy, cx = h // 2, w // 2
            mask = create_circular_mask(h, w, (cx, cy), self.radius)
        elif self.mode == 'inside_cell':
            mask = self.mask_YX
        elif self.mode == 'outside_cell':
            mask = ~self.mask_YX
        elif self.mode == 'entire_image': 
            mask = np.ones((h, w), dtype=bool)
        self.mask = mask
        T, Z, Y, X, C = self.image_TZYXC.shape
        mean_intensities = np.zeros((T, C), dtype=np.float64)
        err_intensities = np.zeros((T, C), dtype=np.float64)
        for i in range(T):
            for ch in range(C):
                stack_ch = self.image_TZYXC[i, :, :, :, ch]
                max_proj = np.max(stack_ch, axis=0)
                masked_pixels = max_proj[mask]
                masked_pixels = masked_pixels[masked_pixels != 0]
                if masked_pixels.size > 0:
                    mean_intensities[i, ch] = np.mean(masked_pixels)
                    err_intensities[i, ch] = np.std(masked_pixels)
        self.mean_intensities = mean_intensities
        self.err_intensities = err_intensities/ np.sqrt(np.sum(mask))  # Standard error of the mean
        time_array = np.arange(T, dtype=float) * self.time_interval_seconds
        params = []
        for ch in range(C):
            raw_intensities = mean_intensities[:, ch]
            if len(raw_intensities) < 2:
                params.extend([0.0, 0.01])  # [k_fit, I0_fit]
                print(f"Warning: Not enough data for channel {ch}. No correction applied.")
                continue
            intensity_decrease = (raw_intensities[0] - raw_intensities[-1]) / raw_intensities[0]
            if intensity_decrease < 0.05 or np.mean(np.diff(raw_intensities)) >= 0:
                params.extend([0.0, raw_intensities[0]])  # [k_fit=0, I0_fit=initial_intensity]
                print(f"Warning: Photobleaching correction not necessary for channel {ch}. No correction applied.")
                continue
            eps = 1e-9
            log_int = np.log(raw_intensities + eps)
            slope, intercept = np.polyfit(time_array, log_int, 1)
            k_fit = -slope
            I0_fit = np.exp(intercept)
            params.extend([k_fit, I0_fit])
        if self.show_plot or (self.plot_name is not None):
            fig, axes = plt.subplots(1, C, figsize=(5*C, 5))
            if C == 1:
                axes = [axes]
            for ch in range(C):
                ax = axes[ch]
                raw_data = mean_intensities[:, ch]
                k_fit, I0_fit = params[2*ch], params[2*ch+1]
                fitted_curve = I0_fit * np.exp(-k_fit * time_array)
                ax.plot(time_array, raw_data, 'o', color='gray', label='Raw Data')
                ax.plot(time_array, fitted_curve, '-', color='blue', label='Exponential Fit')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Intensity')
                ax.set_title(f"Exponential Fit ch={ch}\nI0={I0_fit:.2f}, k={k_fit:.4f}")
                ax.legend()
            
            plt.tight_layout()
            if self.plot_name is not None:
                plt.savefig(self.plot_name)
            if self.show_plot:
                plt.show()
        return params

    def apply_photobleaching_correction(self):
        """
        Applies photobleaching correction normalized to the initial intensity (t=0).
        The correction factor is I_fit(0) / I_fit(t), ensuring the initial frame is unchanged.
        """
        params = self.precalculated_list_decay_rates or self.calculate_photobleaching()
        T, Z, Y, X, C = self.image_TZYXC.shape
        corrected_image = self.image_TZYXC.astype(np.float32).copy()

        start_idx = 0 # self.number_removed_initial_points or 0
        time_array = np.arange(T, dtype=float) * self.time_interval_seconds

        for ch in range(C):
            # **Threshold check:** compute final intensity for this channel
            if hasattr(self, 'mean_intensities'):
                final_intensity = self.mean_intensities[-1, ch]
            else:
                # Calculate final intensity if not available
                if not hasattr(self, 'mask'):
                    h, w = self.image_TZYXC.shape[2], self.image_TZYXC.shape[3]
                    if self.mode == 'use_circular_region':
                        if self.user_provided_mask and self.mask_YX.any():
                            cy, cx = np.mean(np.argwhere(self.mask_YX), axis=0).astype(int)
                        else:
                            cy, cx = h // 2, w // 2
                        Y, X_grid = np.ogrid[:h, :w]
                        dist = np.sqrt((X_grid - cx) ** 2 + (Y - cy) ** 2)
                        mask = dist <= self.radius
                    elif self.mode == 'inside_cell':
                        mask = self.mask_YX
                    elif self.mode == 'outside_cell':
                        mask = ~self.mask_YX
                    elif self.mode == 'entire_image':
                        mask = np.ones((h, w), dtype=bool)
                    self.mask = mask
                final_frame = self.image_TZYXC[-1, :, :, :, ch]
                max_proj = np.max(final_frame, axis=0)
                masked_pixels = max_proj[self.mask]
                final_intensity = np.mean(masked_pixels) if masked_pixels.size > 0 else 0.0

            # **Skip correction if final intensity is below threshold (100)**
            if final_intensity < 10:
                try:
                    QMessageBox.warning(None, "Photobleaching Correction",
                        f"Photobleaching correction skipped for channel {ch} "
                        f"(final intensity {final_intensity:.2f} < 100).")
                except Exception:
                    print(f"Warning: Photobleaching correction skipped for channel {ch} "
                        f"(final intensity {final_intensity:.2f} < 100).")
                continue            
            if hasattr(self, 'mean_intensities'):
                raw_intensities = self.mean_intensities[:, ch]
            else:
                raw_intensities = np.zeros(T)
                for i in range(T):
                    stack_ch = self.image_TZYXC[i, :, :, :, ch]
                    max_proj = np.max(stack_ch, axis=0)
                    masked_pixels = max_proj[self.mask]
                    masked_pixels = masked_pixels[masked_pixels != 0]
                    if masked_pixels.size > 0:
                        raw_intensities[i] = np.mean(masked_pixels)
            eps = 1e-9
            log_int = np.log(raw_intensities + eps)
            slope, intercept = np.polyfit(time_array, log_int, 1)
            k_fit = -slope
            I0_fit = np.exp(intercept)
            I_fit = I0_fit * np.exp(-k_fit * time_array)
            correction_factors = I0_fit / I_fit  # This gives exp(k_fit * time_array)
            for i in range(T):
                corrected_image[i, ..., ch] *= correction_factors[i]
        mean_intensities_corr = np.zeros((T, C), dtype=float)
        err_intensities_corr  = np.zeros((T, C), dtype=float)
        for ch in range(C):
            for i in range(T):
                max_proj_corr = np.max(corrected_image[i, ..., ch], axis=0)
                masked_pixels_corr = max_proj_corr[self.mask]
                masked_pixels_corr = masked_pixels_corr[masked_pixels_corr != 0]  # exclude zeros
                if masked_pixels_corr.size > 0:
                    mean_intensities_corr[i, ch] = masked_pixels_corr.mean()
                    err_intensities_corr[i, ch] = masked_pixels_corr.std() / np.sqrt(masked_pixels_corr.size)
        if self.show_plot:
            orig = [self.image_TZYXC[i].mean() for i in range(T)]
            corr = [corrected_image[i].mean() for i in range(T)]
            plt.figure(figsize=(5,4))
            plt.plot(orig, 'o-', label='Original', color='gray')
            plt.plot(corr, 'o-', label='Corrected', color='blue')
            plt.xlabel('Frame')
            plt.ylabel('Mean Intensity')
            plt.legend()
            plt.tight_layout()
            plt.show()
        photobleaching_data = {
            'decay_rates': params,
            'time_array': time_array,
            'mean_intensities': self.mean_intensities if hasattr(self, 'mean_intensities') else raw_intensities.reshape(-1,1),
            'err_intensities': self.err_intensities if hasattr(self, 'err_intensities') else np.zeros((T,C)),
            'mean_intensities_corrected': mean_intensities_corr,
            'err_intensities_corrected': err_intensities_corr,
        }
        corrected_uint16 = np.clip(corrected_image, 0, 65535).astype(np.uint16)
        return corrected_uint16, photobleaching_data

class ReadLif:
    """
    Read .lif files and extract images, metadata, and per-scene laser/spectro info.

    Args:
        path (str or Path):      The path to the .lif file.
        show_metadata (bool):    Whether to print metadata to stdout.
        save_tif (bool):         Whether to export each scene as OME-TIFF.
        save_png (bool):         Whether to export each scene as PNG.
        format (str):            Axis order for returned arrays (e.g. 'TZYXC').
        lazy (bool):             If True, pixel data loads on demand.
    """
    def __init__(self, path, show_metadata=True, save_tif=False, save_png=False,
                 format='TZYXC', lazy=False):
        # Path setup
        if isinstance(path, str):
            self.path = Utilities.convert_str_to_path(path)
        else:
            self.path = Path(path)

        self.show_metadata = show_metadata
        self.save_tif      = save_tif
        self.save_png      = save_png
        self.format        = format
        self.lazy          = lazy

        # AICSImage handle (lazy or not)
        #self._aics = AICSImage(str(self.path))
        self._aics = BioImage(str(self.path))

    def read(self):
        """
        Reads all scenes and returns a tuple:
          ( list_images, list_names,
            pixel_XY, pixel_Z,
            channel_names, num_channels,
            list_time_intervals, bit_depth,
            list_laser_lines, list_intensities, list_wave_ranges )
        """
        # attempt to open LIF metadata
        try:
            lif_image = LifFile(self.path)
            read_meta = True
        except:
            read_meta = False
            print("⚠️ Could not read LIF metadata; proceeding without it.")
        bit_depth = 0
        if read_meta:
            try:
                bit_depth = lif_image.get_image(0).bit_depth[0]
            except:
                pass
        # scene list + pixel sizes + channel names
        #images   = self._aics if self.lazy else AICSImage(str(self.path))
        images   = self._aics if self.lazy else BioImage(str(self.path))
        scenes   = list(images.scenes)
        pixel_Z  = abs(images.physical_pixel_sizes.Z or 0)
        pixel_XY = abs(images.physical_pixel_sizes.Y or 0)
        ch_names = list(images.channel_names)
        # prepare outputs
        list_images       = []
        list_names        = []
        list_time_intervals = []
        list_laser_lines  = []
        list_intensities  = []
        list_wave_ranges  = []
        for idx, scene in enumerate(scenes):
            ti = 0
            if read_meta:
                try:
                    ti = lif_image.get_image(idx).settings.get('CycleTime', 0)
                except:
                    pass
            list_time_intervals.append(ti)
            images.set_scene(idx)
            arr = images.get_image_data(self.format)
            list_images.append(arr)
            list_names.append(scene)
            ll, iv, wr = self.get_laser_info(idx)
            list_laser_lines.append(ll)
            list_intensities.append(iv)
            list_wave_ranges.append(wr)
            if self.show_metadata:
                if idx == 0:
                    print(f"Number of images: {len(scenes)}\n" + "-"*40)
                print(f"[{idx}] {scene}")
                print(f"  shape={arr.shape}, dtype={arr.dtype}")
                print(f"  channels = {ch_names}")
                print(f"  pixel Z={pixel_Z:.3f}  XY={pixel_XY:.3f}")
                print(f"  time={ti}s  bitdepth={bit_depth}")
                print(f"  laser lines   = {ll}")
                print(f"  intensities    = {iv}")
                print(f"  spectral ranges= {wr}")
                print("-"*40)
            stem = f"{self.path.stem}_{idx}"
            outdir = self.path.parent / "images_reformatted"
            if self.save_tif:
                outdir.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(
                    outdir / f"{stem}.ome.tif",
                    arr,
                    imagej=False,
                    metadata={
                        'axes': self.format,
                        'PhysicalSizeX': pixel_XY,
                        'PhysicalSizeY': pixel_XY,
                        'PhysicalSizeZ': pixel_Z,
                        'TimeIncrement': ti,
                        'TimeIncrementUnit': 's',
                        'SignificantBits': bit_depth,
                        'Channel': {'Name': ch_names},
                    }
                )
            if self.save_png:
                outdir.mkdir(parents=True, exist_ok=True)
                pngpath = outdir / f"{stem}.png"
                mp = np.moveaxis(np.max(arr, axis=0), 0, -1)
                Plots().plot_images(
                    image=mp,
                    figsize=(8,4),
                    plot_name=pngpath,
                    save_plots=True,
                    show_plot=False,
                    use_maximum_projection=False,
                    use_gaussian_filter=True,
                    cmap='Greys',
                    min_max_percentile=[0.5,99.5]
                )

        return (
            list_images,
            list_names,
            pixel_XY,
            pixel_Z,
            ch_names,
            len(ch_names),
            list_time_intervals,
            bit_depth,
            list_laser_lines,
            list_intensities,
            list_wave_ranges,
        )
  
    def read_scene(self, image_index: int):
        """Return the raw numpy array for a single scene index."""
        if not (0 <= image_index < len(self._aics.scenes)):
            raise IndexError("Scene index out of range")
        self._aics.set_scene(image_index)
        return self._aics.get_image_data(self.format)

    def get_laser_info(self, image_index: int):
        """
        For the given scene index returns a triple:
          ( laser_lines, intensities, wave_ranges )

        * laser_lines  : [405, 488, …] ints (per-scene)
        * intensities  : [0.000, 2.300, …] floats (per-scene)
        * wave_ranges  : [(b,e), …] tuples of ints (global Spectro windows)

        Missing blocks simply yield empty lists.
        """
        # parse OME-XML header
        try:
            lif  = LifFile(self.path)
            root = ET.fromstring(lif.xml_header)
        except:
            return [], [], []
        # 1) descend into the matching <Element> (handles nested names)
        proj  = root.find('Element')
        parts = self._aics.scenes[image_index].split('/')
        elem  = proj
        for p in parts:
            ch = elem.find('Children')
            if ch is None:
                return [], [], []
            elem = next((e for e in ch.findall('Element') if e.get('Name')==p), None)
            if elem is None:
                return [], [], []
        # 2) find this series’ SequentialSettingIndex
        seq_idx = None
        for cp in elem.findall('.//ChannelProperty'):
            if cp.findtext('Key') == 'SequentialSettingIndex':
                try:
                    seq_idx = int(cp.findtext('Value'))
                except:
                    pass
                break
        # 3) global spectral ranges
        wave_ranges = []
        if seq_idx is not None:
            defs = root.findall('.//ATLConfocalSettingDefinition')
            if 0 <= seq_idx < len(defs):
                cg = defs[seq_idx]
                spec = cg.find('Spectro')
                if spec is not None:
                    for mb in spec.findall('MultiBand'):
                        try:
                            b = float(mb.get('TargetWaveLengthBegin',0))
                            e = float(mb.get('TargetWaveLengthEnd',0))
                            wave_ranges.append((int(round(b)), int(round(e))))
                        except:
                            pass
        # 4) per-scene laser intensities
        laser_lines, intensities = [], []
        hw = elem.find('.//Attachment[@Name="HardwareSetting"]')
        if hw is not None:
            seq = hw.find('.//LDM_Block_Sequential_List') or \
                  hw.find('.//LDM_Block_Sequential')
            if seq is not None:
                cl = seq.find('ATLConfocalSettingDefinition')
                if cl is not None:
                    for aotf in cl.findall('.//Aotf'):
                        for lls in aotf.findall('LaserLineSetting'):
                            try:
                                wl  = int(lls.get('LaserLine',0))
                                idv = round(float(lls.get('IntensityDev',0)),3)
                                laser_lines.append(wl)
                                intensities.append(idv)
                            except:
                                pass
        return laser_lines, intensities, wave_ranges

class ConvertFormat:
    """
    A class to convert images between different dimension formats.

    This class allows conversion of images with any order of dimensions to a desired format.
    It can handle adding missing dimensions, swapping axes, and reducing dimensions.

    Parameters
    ----------
    image : np.ndarray
        The input image array.
    original_order : list of str
        A list representing the dimension order of the input image.
        Possible dimensions are 'T' (time), 'Z' (depth), 'Y' (height), 'X' (width), 'C' (channel).
    desired_order : list of str
        A list representing the desired dimension order in the output image.

    Methods
    -------
    convert():
        Converts the image to the desired format and returns the converted image.
    """

    def __init__(self, image: np.ndarray, original_order: list, desired_order: list):
        self.image = image
        self.original_order = original_order
        self.desired_order = desired_order
        self.valid_dims = {'T', 'Z', 'Y', 'X', 'C'}
        # Validate dimension orders
        self._validate_dimensions()

    def _validate_dimensions(self):
        # Check if provided dimensions are valid
        for dim in self.original_order + self.desired_order:
            if dim not in self.valid_dims:
                raise ValueError(f"Invalid dimension '{dim}'. Valid dimensions are {self.valid_dims}.")
        # Check if the length of original_order matches the image dimensions
        if len(self.original_order) != self.image.ndim:
            raise ValueError(f"Length of original_order {len(self.original_order)} does not match "
                             f"number of dimensions in image {self.image.ndim}.")

    def convert(self):
        # Map dimension names to indices in the original image
        orig_dim_indices = {dim: idx for idx, dim in enumerate(self.original_order)}
        # Start with the original image
        img = self.image
        # Handle dimensions that are in the original image but not in desired_order (need to reduce)
        dims_to_reduce = [dim for dim in self.original_order if dim not in self.desired_order]
        for dim in dims_to_reduce:
            axis = orig_dim_indices[dim]
            img = np.max(img, axis=axis, keepdims=False)
            # After reducing, adjust indices of remaining dimensions
            orig_dim_indices = {d: (i - 1 if i > axis else i) for d, i in orig_dim_indices.items() if d != dim}
        # Handle dimensions that are in desired_order but not in the original image (need to add new axis)
        dims_to_add = [dim for dim in self.desired_order if dim not in self.original_order]
        for dim in dims_to_add:
            img = np.expand_dims(img, axis=0)
            # Update dimension indices
            orig_dim_indices[dim] = 0
            orig_dim_indices = {d: (i + 1 if d != dim else i) for d, i in orig_dim_indices.items()}
        # Rearrange axes to match desired_order
        axis_order = [orig_dim_indices[dim] for dim in self.desired_order]
        img = np.transpose(img, axes=axis_order)
        return img
    


class GaussianFilter():
    '''
    This class is intended to apply high and low bandpass filters to the video. The format of the video must be [Z, Y, X, C]. This class uses **difference_of_gaussians** from skimage.filters.

    Parameters

    video : NumPy array
        Array of images with dimensions [Z, Y, X, C].
    sigma : float, optional
        Sigma value for the gaussian filter. The default is 1.
    

    '''
    def __init__(self, video:np.ndarray, sigma:float = 1):
        # Making the values for the filters are odd numbers
        self.video = video
        self.sigma = sigma
        self.NUMBER_OF_CORES = cpu_count()
    def apply_filter(self):
        '''
        This method applies high and low bandpass filters to the video.

        Returns
        
        video_filtered : np.uint16
            Filtered video resulting from the bandpass process. Array with format [T, Y, X, C].
        '''
        video_bp_filtered_float = np.zeros_like(self.video, dtype = np.float64)
        video_filtered = np.zeros_like(self.video, dtype = np.uint16)
        number_time_points, number_channels   = self.video.shape[0], self.video.shape[3]
        for index_channels in range(0, number_channels):
            for index_time in range(0, number_time_points):
                if np.max(self.video[index_time, :, :, index_channels])>0:
                    video_bp_filtered_float[index_time, :, :, index_channels] = gaussian(self.video[index_time, :, :, index_channels], self.sigma)
        # temporal function that converts floats to uint
        def img_uint(image):
            temp_vid = img_as_uint(image)
            return temp_vid
        # returning the image normalized as uint. Notice that difference_of_gaussians converts the image into float.
        for index_channels in range(0, number_channels):
            init_video = Parallel(n_jobs = self.NUMBER_OF_CORES)(delayed(img_uint)(video_bp_filtered_float[i, :, :, index_channels]) for i in range(0, number_time_points))
            video_filtered[:,:,:,index_channels] = np.asarray(init_video)
        return video_filtered # video_bp_filtered_float # img_as_uint(video_bp_filtered_float)


class NASConnection():
    '''
    This class is intended to establish a connection between Network-Attached storage and a remote (or local) computer using `pysmb <https://github.com/miketeo/pysmb>`_ . The class allows the user to connect to NAS, download specific files, and write backfiles to NAS.
    This class doesn't allow the user to delete, modify or overwrite files in NAS. For a complete description of pysmb check the `pysmb documentation <https://pysmb.readthedocs.io/en/latest/>`_ .
    To use this class, you need to:
    
    1) Use the university's network or use the two-factor authentication to connect to the university's VPN.
    2) You need to create a configuration YAML file with the following format:
    
    .. code-block:: bash

        user:
        username: name_of_the_user_in_the_nas_server
        password: user_password_in_the_nas_server 
        remote_address : ip or name for the nas server
        domain: domain for the nas server 
    
    Parameters
    
    path_to_config_file : str, or Pathlib object
        The path in the local computer contains the config file.
    share_name: str
        Name of the share partition to access in NAS. The default is 'share'.
    '''
    def __init__(self,path_to_config_file,share_name = 'share'):
        # Loading credentials
        conf = yaml.safe_load(open(str(path_to_config_file)))
        usr = str(conf['user']['username'])
        pwd = str(conf['user']['password'])
        remote_address = str(conf['user']['remote_address'])
        domain = str(conf['user']['domain'])
        # LOCAL NAME
        try:
            local_name = socket.gethostbyname(socket.gethostname())
        except:
            local_name = socket.gethostname()
        # SERVER NAME
        self.share_name = share_name
        self.server_name, _, _ = socket.gethostbyaddr(remote_address)
        # Defining the connection to NAS
        self.conn = SMBConnection(username=usr, password=pwd, domain=domain, my_name=local_name, remote_name=str(self.server_name), is_direct_tcp=True)
    def connect_to_server(self,timeout=60):
        '''
        This method establishes the connection to the NAS.
        
        Parameters 
        
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        '''
        is_connected = self.conn.connect(str(self.server_name),timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        return self.conn
    
    def read_files(self, remote_folder_path, timeout=60):
        '''
        This method reads all files from a NAS directory
        
        Parameters
        
        remote_folder_path : str, Pathlib obj
            The path in the remote folder to download.
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        '''
        # Connecting to NAS
        is_connected = self.conn.connect(str(self.server_name),timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        # Converting the paths to a Pathlib object
        if type(remote_folder_path)==str:
            remote_folder_path = pathlib.Path(remote_folder_path)
        # Iterate in the folder to download all tif files
        list_files =[]
        list_dir = self.conn.listPath(self.share_name, str(remote_folder_path))
        for file in list_dir:
            list_files.append(file.filename)
        return list_files

    def download_file(self, remote_file_path, local_folder_path, timeout=600):
        '''
        This method download an specific file
        
        Parameters
        
        remote_file_path : str, Pathlib obj
            The path in the remote file to download.
        local_folder_path : str, Pathlib obj
            The path in the local computer where the files will be copied.
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        '''
        # Connecting to NAS
        is_connected = self.conn.connect(str(self.server_name),timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        # Converting the paths to a Pathlib object
        if type(local_folder_path) == str:
            local_folder_path = pathlib.Path(local_folder_path)
        if type(remote_file_path)==str:
            remote_file_path = pathlib.Path(remote_file_path)
        # Making the local directory
        if not (os.path.exists(local_folder_path)) :
            os.makedirs(str(local_folder_path))
        filename = remote_file_path.name
        fileobj = open(remote_file_path.name,'wb')
        self.conn.retrieveFile(self.share_name, str(remote_file_path), fileobj)
        fileobj.close()
        # moving files in the local computer
        shutil.move(pathlib.Path().absolute().joinpath(filename), local_folder_path.joinpath(filename))
        print('Files downloaded to: ' + str(local_folder_path.joinpath(filename)))
        return None
    
    def copy_files(self, remote_folder_path, local_folder_path, timeout=600, file_extension =['.index','.tif']):
        '''
        This method downloads tif files from NAS to a temporal folder in the local computer.
        
        Parameters
        
        remote_folder_path : str, Pathlib obj
            The path in the remote folder to download.
        local_folder_path : str, Pathlib obj
            The path in the local computer where the files will be copied.
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        file_extension : str, optional.
            String representing the file type to download.
        '''
        # Connecting to NAS
        is_connected = self.conn.connect(str(self.server_name),timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        # Converting the paths to a Pathlib object
        if type(local_folder_path) == str:
            local_folder_path = pathlib.Path(local_folder_path)
        if type(remote_folder_path)==str:
            remote_folder_path = pathlib.Path(remote_folder_path)
        # Making the local directory
        if (os.path.exists(local_folder_path))  and  (str(local_folder_path.name)[0:5] ==  'temp_'):
            shutil.rmtree(local_folder_path)
        os.makedirs(str(local_folder_path))
        # Iterate in the folder to download all tif files
        list_dir = self.conn.listPath(self.share_name, str(remote_folder_path))
        for file in list_dir:
            #if (file.filename not in ['.', '..']) and (file_extension in file.filename) :
            if (file.filename not in ['.', '..']) and  any(file.filename.endswith(ext) for ext in file_extension):
                print ('File Downloaded :', file.filename)
                fileobj = open(file.filename,'wb')
                self.conn.retrieveFile(self.share_name, str( pathlib.Path(remote_folder_path).joinpath(file.filename) ),fileobj)
                fileobj.close()
                # moving files in the local computer
                shutil.move(pathlib.Path().absolute().joinpath(file.filename), local_folder_path.joinpath(file.filename))
        print('Files downloaded to: ' + str(local_folder_path))
        return None
    
    def write_files_to_NAS(self, local_file_to_send_to_NAS, remote_folder_path,  timeout=600):
        '''
        This method writes files from a local computer to NAS 
        
        Parameters
        
        local_file_to_send_to_NAS : str, Pathlib obj
            The path in the file to send to the NAS.
        remote_folder_path : str, Pathlib obj
            The path in the remote folder to download.
        timeout : int, optional
            Time in seconds to maintain a connection with the NAS. The default is 60 seconds.
        '''
        # Connecting to NAS
        is_connected = self.conn.connect(str(self.server_name),timeout=timeout)
        if is_connected == True:
            print('Connection established')
        else:
            print('Connection failed')
        # Converting the paths to a Pathlib object
        if type(local_file_to_send_to_NAS) == str:
            local_file_to_send_to_NAS = pathlib.Path(local_file_to_send_to_NAS)
        if type(remote_folder_path)==str:
            remote_folder_path = pathlib.Path(remote_folder_path)
        # checks that the file doesn't exist on NAS. If it exist it will create a new name as follows original_name__1
        # Iterate in the folder to download all tif files
        list_dir = self.conn.listPath(self.share_name, str(remote_folder_path))
        list_all_files_in_NAS = [file.filename for file in list_dir]
        if str(local_file_to_send_to_NAS.name) not in list_all_files_in_NAS:
            with open(str(local_file_to_send_to_NAS), 'rb') as file_obj:
                self.conn.storeFile(self.share_name, str( pathlib.Path(remote_folder_path).joinpath(local_file_to_send_to_NAS.name) ) ,  file_obj )
                print ('The file was uploaded to NAS in location:', str( pathlib.Path(remote_folder_path).joinpath(local_file_to_send_to_NAS.name))  )
        return None


class ReadImages():
    '''
    This class reads all tif images in a given folder and returns each image as a Numpy array inside a list, the names of these files, path, and the number of files.
    
    Parameters
    
    directory: str or PosixPath
        Directory containing the images to read.
    '''    
    def __init__(self, directory:str,number_of_images_to_process=None):
        if type(directory)== pathlib.PosixPath or type(directory)== pathlib.WindowsPath:
            self.directory = directory
        else:
            self.directory = pathlib.Path(directory)
        self.number_of_images_to_process = number_of_images_to_process
    def read(self):
        '''
        Method takes all the images in the folder and merges those with similar names.
        
        Returns
        
        list_images : List of NumPy arrays 
            List of NumPy arrays with format np.uint16 and dimensions [Z, Y, X, C] or [T, Y, X, C]. 
        path_files : List of strings 
            List of strings containing the path to each image.
        list_files_names : List of strings 
            List of strings where each element is the name of the files in the directory.
        number_files : int 
            The number of images in the folder.
        '''
        list_files_names_complete = sorted([f for f in listdir(self.directory) if isfile(join(self.directory, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
        list_files_names_complete.sort(key=lambda f: int(re.sub(r'\D', '', f)))  # sorting the index in numerical order
        path_files_complete = [ str(self.directory.joinpath(f).resolve()) for f in list_files_names_complete ] # creating the complete path for each file
        number_files = len(path_files_complete)
        list_images_complete = [imread(str(f)) for f in path_files_complete]
        
        # This section reads the images to process in the repository.  If the user defines "number_of_images_to_process" as an integer N the code will process a subset N of these images. 
        if (self.number_of_images_to_process is None) or (self.number_of_images_to_process>number_files):
            list_images =list_images_complete
            path_files = path_files_complete
            list_files_names = list_files_names_complete
        else:
            list_images = list_images_complete[0:self.number_of_images_to_process]
            path_files = path_files_complete[0:self.number_of_images_to_process]
            list_files_names = list_files_names_complete[0:self.number_of_images_to_process]
            number_files = self.number_of_images_to_process
        return list_images, path_files, list_files_names, number_files


class MergeChannels():
    '''
    This class takes images as arrays with format [Z, Y, X] and merges them in a NumPy array with format [Z, Y, X, C].
    It recursively merges the channels in a new dimension in the array. The minimum number of channels 2 maximum is 4.
    
    Parameters

    directory: str or PosixPath
        Directory containing the images to merge.
    substring_to_detect_in_file_name: str
        String with the prefix to detect the names of the files. 
    save_figure: bool, optional
        If True, it saves the merged images as tif. The default is False. 
    '''
    def __init__(self, directory:str ,substring_to_detect_in_file_name:str = '.*_C0.tif', save_figure:bool = False ):
        if type(directory)== pathlib.PosixPath or type(directory)== pathlib.WindowsPath:
            self.directory = directory
        else:
            self.directory = pathlib.Path(directory)
        self.substring_to_detect_in_file_name = substring_to_detect_in_file_name
        self.save_figure=save_figure

    def checking_images(self):
        '''
        Method that reads all images in the folder and returns a flag indicating if each channel in the image is separated in an independent file.
        
        Returns
        
        Flag : Bool 
            If True, it indicates that each channel is split into different files. If False, it indicates that the image is contained in a single file.
        '''
        ending_string = re.compile(self.substring_to_detect_in_file_name)  # detecting files ending in _C0.tif
        for _, _, files in os.walk(self.directory):
            for file in files:
                if ending_string.match(file) and file[0]!= '.' : # detecting a match in the end, not consider hidden files starting with '.'
                    prefix = file.rpartition('_')[0]            # stores a string with the first part of the file name before the last underscore character in the file name string.
                    list_files_per_image = sorted ( glob.glob( str(self.directory.joinpath(prefix)) + '*.tif')) # List of files that match the pattern 'file_prefix_C*.tif'
                    if len(list_files_per_image)>1:             # creating merged files if more than one images with the same ending substring are detected.
                        return True
                    else:
                        return False
    
    def merge(self):
        '''
        Method takes all the images in the folder and merges those with similar names.
        
        Returns
        
        list_file_names : List of strings 
            List with strings of names.
        list_merged_images : List of NumPy arrays
            List of NumPy arrays with format np.uint16 and dimensions [Z, Y, X, C].
        number_files : int
            The number of merged images in the folder.
        '''
        list_file_names =[]
        list_merged_images =[]  # list that stores all files belonging to the same image in a sublist
        ending_string = re.compile(self.substring_to_detect_in_file_name)  # detecting files ending in _C0.tif
        save_to_path = self.directory.joinpath('merged')
        for _, _, files in os.walk(self.directory):
            for file in files:
                if ending_string.match(file) and file[0]!= '.': # detecting a match in the end, not consider hidden files starting with '.'
                    prefix = file.rpartition('_')[0]            # stores a string with the first part of the file name before the last underscore character in the file name string.
                    list_files_per_image = sorted ( glob.glob( str(self.directory.joinpath(prefix)) + '*.tif') ) # List of files that match the pattern 'file_prefix_C*.tif'
                    if len(list_files_per_image)>1:             # creating merged files if more than one image with the exact ending substring is detected.
                        try:
                            list_file_names.append(prefix)
                            merged_img = np.concatenate([ imread(list_files_per_image[i])[..., np.newaxis] for i,_ in enumerate(list_files_per_image)],axis=-1).astype('uint16')
                            list_merged_images.append(merged_img) 
                        except:
                            print('A corrupted file was detected during the merge of the following files: \n', list_files_per_image,'\n')
                    if self.save_figure ==1 and len(list_files_per_image)>1:
                        if not os.path.exists(str(save_to_path)):
                            os.makedirs(str(save_to_path))
                        tifffile.imsave(str(save_to_path.joinpath(prefix+'_merged'+'.tif')), merged_img)
        number_files = len(list_file_names)
        return list_file_names, list_merged_images, number_files,save_to_path


class Intensity():
    def __init__(self, original_image, spot_size=5, array_spot_location_z_y_x=None, 
                 use_max_projection=False, optimize_spot_size=False, allow_subpixel_repositioning=False):
        self.original_image = original_image
        if array_spot_location_z_y_x is None:
            self.array_spot_location_z_y_x = np.array([[0, 0, 0]])
        else:
            self.array_spot_location_z_y_x = array_spot_location_z_y_x
        self.number_spots = self.array_spot_location_z_y_x.shape[0]
        self.number_channels = original_image.shape[-1]
        self.PIXELS_AROUND_SPOT = 3
        self.MAXIMUM_SPOT_SIZE = 11
        if isinstance(spot_size, (int, np.integer)):
            self.spot_size = np.full(self.number_spots, spot_size, dtype=int) 
        else:
            self.spot_size = np.array(spot_size, dtype=int) 
        self.use_maximum_projection = use_max_projection
        self.optimize_spot_size = optimize_spot_size
        self.allow_subpixel_repositioning = allow_subpixel_repositioning

    def two_dimensional_gaussian(self, xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
        (x, y) = xy
        g = offset + amplitude * np.exp(
            -(((x - x0) ** 2) / (2 * sigma_x**2) + ((y - y0) ** 2) / (2 * sigma_y**2))
        )
        return g.ravel()

    def fit_2D_gaussian(self, data):
        y_size, x_size = data.shape
        x = np.linspace(0, x_size - 1, x_size)
        y = np.linspace(0, y_size - 1, y_size)
        x, y = np.meshgrid(x, y)

        total_intensity = data.sum() + 1e-9
        x_center = (np.arange(x_size) * data.sum(axis=0)).sum() / total_intensity
        y_center = (np.arange(y_size) * data.sum(axis=1)).sum() / total_intensity
        # make amplitude_guess the 95th percentile of the data
        amplitude_guess = np.percentile(data, 95)
        offset_guess = np.percentile(data, 5)
        sigma_x_guess = np.sqrt(np.sum(((np.arange(x_size) - x_center) ** 2) * data.sum(axis=0)) / total_intensity)
        sigma_y_guess = np.sqrt(np.sum(((np.arange(y_size) - y_center) ** 2) * data.sum(axis=1)) / total_intensity)
        initial_guess = (amplitude_guess, x_center, y_center, sigma_x_guess, sigma_y_guess, offset_guess)
        bounds = (
            [0, 0, 0, 0.1, 0.1, data.min()-abs(data.min())],
            [amplitude_guess*5, x_size-1, y_size-1, max(x_size,y_size), max(x_size,y_size), data.max()+abs(data.max())]
        )
        try:
            popt, _ = curve_fit(
                self.two_dimensional_gaussian,
                (x, y),
                data.ravel(),
                p0=initial_guess,
                bounds=bounds
            )
            amplitude, x_pos, y_pos, sigma_x, sigma_y, offset = popt
            if amplitude <= 0 or sigma_x <= 0 or sigma_y <= 0:
                return None, np.inf
            fitted_data = self.two_dimensional_gaussian((x, y), *popt).reshape(data.shape)
            residuals = data - fitted_data
            ss_res = np.sum(residuals**2)
            return {
                'amplitude': amplitude,
                'x_position': x_pos,
                'y_position': y_pos,
                'sigma_x': abs(sigma_x),
                'sigma_y': abs(sigma_y),
                'offset': offset,
            }, ss_res
        except:
            return None, np.inf

    def optimize_spot_size_method(self, frame_data, x_pos, y_pos):
        best_fit = None
        best_residual = np.inf
        best_spot_size = None
        for spot_size in range(5, 12, 2):
            half = spot_size // 2
            y_min = max(y_pos - half, 0)
            y_max = min(y_pos + half + 1, frame_data.shape[0])
            x_min = max(x_pos - half, 0)
            x_max = min(x_pos + half + 1, frame_data.shape[1])
            spot_data = frame_data[y_min:y_max, x_min:x_max]
            fit_result, ss_res = self.fit_2D_gaussian(spot_data)
            if ss_res < best_residual:
                best_residual = ss_res
                best_fit = fit_result
                best_spot_size = spot_size
        return best_fit, best_spot_size

    def search_best_center(self, frame_data, x_pos, y_pos, spot_size):
        # If allow_subpixel_repositioning is True, search within +/- 2 px in x and y
        # to find the position with best Gaussian fit (lowest residual).
        best_fit = None
        best_residual = np.inf
        best_x = x_pos
        best_y = y_pos

        half = spot_size // 2
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                test_y = y_pos + dy
                test_x = x_pos + dx
                # Extract spot_data for this test position
                y_min = max(test_y - half, 0)
                y_max = min(test_y + half + 1, frame_data.shape[0])
                x_min = max(test_x - half, 0)
                x_max = min(test_x + half + 1, frame_data.shape[1])
                spot_data = frame_data[y_min:y_max, x_min:x_max]
                fit_result, ss_res = self.fit_2D_gaussian(spot_data)
                if ss_res < best_residual:
                    best_residual = ss_res
                    best_fit = fit_result
                    best_x = test_x
                    best_y = test_y

        return best_fit, best_x, best_y

    def calculate_intensity(self):
        def return_crop(image: np.ndarray, x: int, y: int, spot_range):
            return image[y+spot_range[0]:y+(spot_range[-1]+1), x+spot_range[0]:x+(spot_range[-1]+1)].copy()

        def return_donut(image, spot_size):
            tem_img = image.copy().astype(float)
            center = tem_img.shape[0] // 2
            half_size = spot_size // 2
            min_index = center - half_size
            max_index = center + half_size + 1
            tem_img[min_index:max_index, min_index:max_index] = np.nan
            donut_values = tem_img[~np.isnan(tem_img)].astype('uint16')
            return donut_values

        def signal_to_noise_ratio(values_disk, values_donut):
            mean_disk = np.mean(values_disk.astype(float))
            mean_donut = np.mean(values_donut.astype(float))
            std_donut = np.std(values_donut.astype(float))
            SNR = (mean_disk - mean_donut) / std_donut if std_donut > 0 else 0
            return SNR, mean_donut, std_donut

        def disk_donut(values_disk, values_donut, spot_size):
            mean_disk = np.mean(values_disk.astype(float))
            std_disk = np.std(values_disk.astype(float))
            mean_donut = np.mean(values_donut.astype(float))
            spot_intensity = mean_disk - mean_donut
            return spot_intensity, std_disk

        intensities_snr = np.zeros((self.number_spots, self.number_channels ))
        intensities_background_mean = np.zeros((self.number_spots, self.number_channels ))
        intensities_background_std = np.zeros((self.number_spots, self.number_channels ))
        intensities = np.zeros((self.number_spots, self.number_channels ))
        intensities_total = np.zeros((self.number_spots, self.number_channels ))
        intensities_std = np.zeros((self.number_spots, self.number_channels ))
        psfs_amplitude = np.zeros((self.number_spots, self.number_channels ))*np.nan
        psfs_sigma = np.zeros((self.number_spots, self.number_channels ))*np.nan

        for sp in range(self.number_spots):
            z_pos = int(self.array_spot_location_z_y_x[sp,0])
            y_pos = int(self.array_spot_location_z_y_x[sp,1])
            x_pos = int(self.array_spot_location_z_y_x[sp,2])
            if self.use_maximum_projection:
                frame_data = np.max(self.original_image[:,:,:, :], axis=0)
            else:
                frame_data = self.original_image[z_pos, :, :, :]
            for i in range(self.number_channels):
                # Save the original coordinates for this spot for each channel
                orig_x, orig_y = x_pos, y_pos

                if self.optimize_spot_size:
                    best_fit, best_size = self.optimize_spot_size_method(frame_data[:,:,i], orig_x, orig_y)
                    if best_fit is None:
                        best_size = self.spot_size[sp]
                else:
                    best_size = self.spot_size[sp]
                    half = best_size // 2
                    y_min = max(orig_y - half, 0)
                    y_max = min(orig_y + half + 1, frame_data.shape[0])
                    x_min = max(orig_x - half, 0)
                    x_max = min(orig_x + half + 1, frame_data.shape[1])
                    spot_data = frame_data[y_min:y_max, x_min:x_max, i]
                    best_fit, ss_res = self.fit_2D_gaussian(spot_data)

                # Use temporary variables for subpixel repositioning so as not to override the original spot center
                current_x, current_y = orig_x, orig_y
                if self.allow_subpixel_repositioning:
                    improved_fit, improved_x, improved_y = self.search_best_center(frame_data[:,:,i], current_x, current_y, best_size)
                    if improved_fit is not None:
                        best_fit = improved_fit
                        current_x = improved_x
                        current_y = improved_y

                if best_fit is not None:
                    amplitude = best_fit['amplitude']
                    sigma_x = best_fit['sigma_x']
                    sigma_y = best_fit['sigma_y']    
                    psfs_amplitude[sp,i] = amplitude
                    psfs_sigma[sp,i] = (sigma_x + sigma_y)/2
                else:
                    psfs_amplitude[sp,i] = np.nan
                    psfs_sigma[sp,i] = np.nan

                # use current_x and current_y instead of x_pos and y_pos in the cropping operations
                half = best_size // 2
                y_min = max(current_y - half, 0)
                y_max = min(current_y + half + 1, frame_data.shape[0])
                x_min = max(current_x - half, 0)
                x_max = min(current_x + half + 1, frame_data.shape[1])
                values_disk = frame_data[y_min:y_max, x_min:x_max, i]
                crop_size = best_size + self.PIXELS_AROUND_SPOT
                crop_range = np.arange(-(crop_size - 1)/2,(crop_size-1)/2+1,1,dtype=int)
                crop_disk_and_donut = return_crop(frame_data[:,:,i], x_pos, y_pos, spot_range=crop_range)
                values_donut = return_donut(crop_disk_and_donut, spot_size=best_size)
                intensities_snr[sp,i], intensities_background_mean[sp,i], intensities_background_std[sp,i] = signal_to_noise_ratio(values_disk, values_donut)
                # disk_donut calculation
                intensities[sp,i], intensities_std[sp,i] = disk_donut(values_disk, values_donut, spot_size=best_size)
                intensities_total[sp,i] = np.sum(values_disk)

        return (
            np.round(intensities,4), 
            np.round(intensities_std,4), 
            np.round(intensities_snr,4), 
            np.round(intensities_background_mean,4), 
            np.round(intensities_background_std,4),
            np.round(psfs_amplitude,4),
            np.round(psfs_sigma,4),
            np.round(intensities_total,4)
        )


class RemoveExtrema:
    '''
    This class removes extreme values from an image array by clipping intensity values based on specified percentiles.

    Parameters
    ----------
    image : np.ndarray
        Array of images with dimensions [Y, X], [Y, X, C], [Z, Y, X, C], or [T, Z, Y, X, C].
    min_percentile : float, optional
        Lower bound percentile to normalize intensity. Default is 1.
    max_percentile : float, optional
        Upper bound percentile to normalize intensity. Default is 99.
    selected_channels : list of int or None, optional
        List of channel indices to apply the extrema removal. If None, applies to all channels. Default is None.
    '''

    def __init__(self, image: np.ndarray, min_percentile: float = 1, max_percentile: float = 99, selected_channels=None):
        self.image = image
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        if selected_channels is None:
            self.selected_channels = None
        elif isinstance(selected_channels, int):
            self.selected_channels = [selected_channels]
        elif isinstance(selected_channels, list):
            self.selected_channels = selected_channels
        else:
            raise ValueError("selected_channels must be an int, list of ints, or None.")

    def remove_outliers(self):
        '''
        Normalizes the values of an image by clipping extreme values based on specified percentiles.

        Returns
        -------
        normalized_image : np.ndarray
            Normalized image with the same dimensions as the input image.
        '''
        normalized_image = self.image.copy().astype(np.float32)
        num_dims = normalized_image.ndim
        # Handle different image formats
        if num_dims == 2:
            # Format: [Y, X]
            normalized_image = self._process_slice(normalized_image)
        elif num_dims == 3:
            # Possible formats: [Y, X, C] or [Z, Y, X]
            if self.image.shape[2] <= 4:
                # Assume format: [Y, X, C]
                for c in range(self.image.shape[2]):
                    if self.selected_channels is None or c in self.selected_channels:
                        normalized_image[:, :, c] = self._process_slice(normalized_image[:, :, c])
            else:
                # Assume format: [Z, Y, X]
                for z in range(self.image.shape[0]):
                    normalized_image[z, :, :] = self._process_slice(normalized_image[z, :, :])
        elif num_dims == 4:
            # Possible formats: [Z, Y, X, C] or [T, Y, X, C]
            if self.image.shape[3] <= 4:
                # Format: [Z, Y, X, C]
                for c in range(self.image.shape[3]):
                    if self.selected_channels is None or c in self.selected_channels:
                        for z in range(self.image.shape[0]):
                            normalized_image[z, :, :, c] = self._process_slice(normalized_image[z, :, :, c])
            else:
                # Format: [T, Z, Y, X]
                for t in range(self.image.shape[0]):
                    for z in range(self.image.shape[1]):
                        normalized_image[t, z, :, :] = self._process_slice(normalized_image[t, z, :, :])
        elif num_dims == 5:
            # Format: [T, Z, Y, X, C]
            for c in range(self.image.shape[4]):
                if self.selected_channels is None or c in self.selected_channels:
                    for t in range(self.image.shape[0]):
                        for z in range(self.image.shape[1]):
                            normalized_image[t, z, :, :, c] = self._process_slice(normalized_image[t, z, :, :, c])
        else:
            raise ValueError("Unsupported image dimensions.")

        return normalized_image.astype(np.uint16)

    def _process_slice(self, image_slice):
        '''
        Clips the values of a 2D image slice based on the specified percentiles.

        Parameters
        ----------
        image_slice : np.ndarray
            2D array representing an image slice.

        Returns
        -------
        np.ndarray
            The processed image slice.
        '''
        if np.max(image_slice) == 0:
            return image_slice
        min_val = np.percentile(image_slice, self.min_percentile)
        max_val = np.percentile(image_slice, self.max_percentile)
        image_slice = np.clip(image_slice, min_val, max_val)
        return image_slice


class Cellpose():
    '''
    This class is intended to detect cells by image masking using `Cellpose <https://github.com/MouseLand/cellpose>`_ . The class uses optimization to maximize the number of cells or maximize the size of the detected cells.
    For a complete description of Cellpose check the `Cellpose documentation <https://cellpose.readthedocs.io/en/latest/>`_ .
    
    Parameters
    
    image : NumPy array
        Array of images with dimensions [Z, Y, X, C].
    num_iterations : int, optional
        Number of iterations for the optimization process. The default is 5.
    channels : List, optional
        List with the channels in the image. For gray images use [0, 0], for RGB images with intensity for cytosol and nuclei use [0, 1] . The default is [0, 0].
    diameter : float, optional
        Average cell size. The default is 120.
    model_type : str, optional
        Cellpose model type the options are 'cyto' for cytosol or 'nuclei' for the nucleus. The default is 'cyto'.
    selection_method : str, optional
        Option to use the optimization algorithm to maximize the number of cells or maximize the size options are 'max_area' or 'max_cells' or 'max_cells_and_area'. The default is 'max_cells_and_area'.
    NUMBER_OF_CORES : int, optional
        The number of CPU cores to use for parallel computing. The default is 1.
    '''
    def __init__(self, image:np.ndarray, num_iterations:int = 3, channels:list = [0, 0], diameter:float = 120, model_type:str = 'cyto3', selection_method:str = 'cellpose_max_cells_and_area', NUMBER_OF_CORES:int=1,pretrained_model=None,selection_metric='max_cells'):
        def preprocess_image(image):
            # Normalize image
            #image = (image - np.min(image)) / (np.max(image) - np.min(image))
            # Apply Gaussian blur
            image = exposure.rescale_intensity(image)            # Enhance contrast
            image = exposure.equalize_adapthist(image)
            image =  Utilities().convert_to_int8( image, rescale=True, min_percentile=0.05, max_percentile=99.95, padding_zeros=False) #Utilities().convert_to_int8(image)
            return image

        # Apply preprocessing before segmentation
        self.image = preprocess_image(image)
        self.image = image
        self.num_iterations = num_iterations
        self.minimum_flow_threshold = 0.0
        self.maximum_flow_threshold = 0.5
        self.channels = channels
        self.diameter = diameter
        self.model_type = model_type # options are 'cyto' or 'nuclei'
        self.selection_method = selection_method # options are 'max_area' or 'max_cells'
        self.NUMBER_OF_CORES = NUMBER_OF_CORES
        self.default_flow_threshold = 0.4 # default is 0.4
        self.optimization_parameter = np.unique(  np.round(np.linspace(self.minimum_flow_threshold, self.maximum_flow_threshold, self.num_iterations), 2) )
        self.MINIMUM_CELL_AREA = 10000# np.pi*(diameter/4)**2 #1000  # using half of the diameter to calculate area.
        self.BATCH_SIZE = 80
        self.pretrained_model=pretrained_model
        self.minimun_diameter_ratio = 0.5
        self.maximum_diameter_ratio = 2.5
    def calculate_masks(self):
        '''
        This method performs the process of image masking using **Cellpose**.
        
        Returns
        
        selected_masks : NumPy array
            NumPy array with values between 0 and the number of detected cells in the image,
            where an integer larger than zero represents the masked area for each cell, and 0 represents the background in the image.
        '''

        # Next two lines suppressing output from Cellpose
        #gc.collect()
        #torch.cuda.empty_cache()
        # Check if GPU is available
        use_gpu = 0 #models.use_gpu()
        #print('GPU activated' if use_gpu else 'GPU not found')
        # Initialize Cellpose model
        if self.pretrained_model is None:
            if self.model_type == 'cyto3':
                model = denoise.CellposeDenoiseModel(gpu=use_gpu, model_type="cyto3",
                                     restore_type="denoise_cyto3")
                print('Model cyto3 loaded')
            else:
                model = models.Cellpose(gpu=use_gpu, model_type=self.model_type)  # model_type = 'cyto' or 'nuclei'

        else:
            model = models.CellposeModel(gpu=use_gpu, pretrained_model=self.pretrained_model)

        # Define ranges for optimization parameters
        flow_thresholds = np.linspace(self.minimum_flow_threshold, self.maximum_flow_threshold, self.num_iterations)
        diameter_ratios = np.linspace(self.minimun_diameter_ratio, self.maximum_diameter_ratio, self.num_iterations) # Adjust diameter by ±20%
        diameters = self.diameter * diameter_ratios
        diameters = diameters.astype(int)
        # Create grid of parameter combinations
        param_grid = [(flow, dia) for flow in flow_thresholds for dia in diameters]

        # Initialize variables to store the best results
        best_metric = -np.inf
        best_masks = None
        best_params = None

        def run_cellpose(flow_threshold, diameter):
            try:
                masks = model.eval(
                    self.image,
                    batch_size=self.BATCH_SIZE,
                    normalize=True,
                    flow_threshold=flow_threshold,
                    diameter=diameter,
                    min_size=self.MINIMUM_CELL_AREA,
                    channels=self.channels,
                    #progress=None
                )[0]
                # Removing artifacts
                masks = Utilities().remove_artifacts_from_mask_image(masks, minimal_mask_area_size=self.MINIMUM_CELL_AREA)
            except Exception as e:
                print(f"Error during cellpose_max_cells evaluation: {e}")
                masks = np.zeros_like(self.image[:, :, 0])
            if np.max(masks):
                metric = Utilities().metric_max_cells_and_area(masks, mode=self.selection_method)
            else:
                metric = 0
            return metric, masks

        # Function to evaluate masks based on the selection method
        def evaluate_masks(params):
            flow_threshold, diameter = params
            metric, masks = run_cellpose(flow_threshold, diameter) 
            return metric, masks, params
        # If no optimization is desired (selection_method is None)
        if self.selection_method is None:
            selected_masks = model.eval(
                self.image,
                batch_size=self.BATCH_SIZE,
                normalize=True,
                flow_threshold=self.default_flow_threshold,
                diameter=self.diameter,
                min_size=self.MINIMUM_CELL_AREA,
                channels=self.channels,
                #progress=None
            )[0]
        else:
            # Evaluate parameter combinations in parallel
            results = Parallel(n_jobs=self.NUMBER_OF_CORES)(
                delayed(evaluate_masks)(params) for params in param_grid
            )
            # Find the best result
            for metric, masks, params in results:
                #print(f"Parameters: flow_threshold={params[0]}, diameter={params[1]}, metric={metric}")
                if metric > best_metric:
                    best_metric = metric
                    best_masks = masks
                    best_params = params

            if best_masks is not None and np.max(best_masks) > 0:
                #print(f"Best parameters: flow_threshold={best_params[0]}, diameter={best_params[1]}")
                selected_masks = best_masks
            else:
                #print("No masks detected with any parameter combination. Returning empty mask.")
                if len(self.image.shape) >= 3:
                    selected_masks = np.zeros_like(self.image[:, :, 0])
                else:
                    selected_masks = np.zeros_like(self.image)
        selected_masks = Utilities().remove_artifacts_from_mask_image(selected_masks, minimal_mask_area_size=self.MINIMUM_CELL_AREA)

        return selected_masks
    

class CellSegmentationWatershed:
    """
    Enhanced cell segmentation using a gradient-based Watershed algorithm.
    """
    def __init__(self, 
                 image: np.ndarray, 
                 footprint_size = 3,
                 expected_radius=200,
                 threshold_method='li',
                 threshold_factor=1.0,
                 markers_method='local',
                 canny_sigma=2.0,
                 min_object_size=500,
                 separation_size=1):
        """
        Parameters
        ----------
        image : np.ndarray
            Your input image, either [Y, X], [Y, X, C], or [Z, Y, X, C].
        expected_radius : float
            Approximate cell radius in pixels (default=200).
        threshold_method : str
            'otsu' or 'li' threshold for initial foreground detection (default='li').
        threshold_factor : float
            Factor to multiply or divide the base threshold (default=1.0).
        markers_method : str
            Either 'local' (local maxima in distance transform) or 'distance' (label the whole mask).
        canny_sigma : float
            Sigma for Canny edge detection. Edges can help split touching cells. 
            Lower means more edges, higher means fewer edges (default=2).
        min_object_size : int
            The smallest object size to keep (default=500).
        separation_size : int
            The disk size for final erosion or morphological separation. 
            If there's only one cell, we skip or reduce this (default=3).
        """
        self.image = image
        self.expected_radius = expected_radius
        self.threshold_method = threshold_method
        self.threshold_factor = threshold_factor
        self.markers_method = markers_method
        self.canny_sigma = canny_sigma
        self.min_object_size = min_object_size
        self.separation_size = separation_size

    def apply_watershed(self):
        if self.image.ndim == 4:  # [Z, Y, X, C]
            image_2d = np.max(self.image, axis=0)
        elif self.image.ndim == 3 and self.image.shape[2] <= 4:  # [Y, X, C]
            image_2d = self.image
        else:
            image_2d = self.image
        if image_2d.ndim == 3 and image_2d.shape[2] > 1:
            image_gray = np.mean(image_2d, axis=2)
        else:
            image_gray = image_2d.squeeze()
        image_eq = exposure.equalize_adapthist(image_gray)
        sigma_smooth = self.expected_radius / 10.0  # heuristic
        smoothed = filters.gaussian(image_eq, sigma=sigma_smooth)
        if self.threshold_method == 'otsu':
            base_thresh = filters.threshold_otsu(smoothed)
        elif self.threshold_method == 'li':
            base_thresh = filters.threshold_li(smoothed)
        else:
            raise ValueError("Use 'otsu' or 'li' threshold_method.")
        thresh_val = base_thresh / self.threshold_factor
        initial_mask = smoothed > thresh_val
        clean_mask = remove_small_objects(initial_mask, min_size=self.min_object_size)
        clean_mask = remove_small_holes(clean_mask, area_threshold=5*self.min_object_size)
        fg_pixels = np.sum(clean_mask)
        edges = feature.canny(smoothed, sigma=self.canny_sigma)
        edges_dil = morphology.dilation(edges, disk(2))
        circle_area = np.pi * (self.expected_radius**2)
        multi_cell_mask = clean_mask.copy()
        if fg_pixels > 2 * circle_area:
            multi_cell_mask = np.logical_and(multi_cell_mask, ~edges_dil)
        distance = ndimage.distance_transform_edt(multi_cell_mask)
        if self.markers_method == 'local' and fg_pixels > 1.5 * circle_area:
            min_dist = int(self.expected_radius / 4.0)
            coords = feature.peak_local_max(distance, min_distance=min_dist, labels=multi_cell_mask)
            local_max = np.zeros_like(distance, dtype=bool)
            if coords.size:
                local_max[tuple(coords.T)] = True
            markers, _ = ndimage.label(local_max)
        else:
            markers, _ = ndimage.label(multi_cell_mask)
        sobelx = filters.sobel_h(smoothed)
        sobely = filters.sobel_v(smoothed)
        gradient = np.hypot(sobelx, sobely)
        labels = watershed(gradient, markers=markers, mask=multi_cell_mask)
        labels = remove_small_objects(labels, min_size=self.min_object_size)
        labels = labels.astype(np.int32)
        num_objects = labels.max()
        if num_objects == 0:
            print("No objects detected.")
            return np.zeros_like(labels, dtype=int)
        props = measure.regionprops(labels)
        center_y, center_x = (image_gray.shape[0]/2.0, image_gray.shape[1]/2.0)
        best_region = None
        best_metric = -np.inf
        for rp in props:
            area = rp.area
            cy, cx = rp.centroid
            dist_center = np.sqrt((cy - center_y)**2 + (cx - center_x)**2)
            lam = 1.0
            metric = area - lam*dist_center
            if metric > best_metric:
                best_metric = metric
                best_region = rp
        if not best_region:
            print("No best region found somehow.")
            return np.zeros_like(labels, dtype=int)
        best_mask = (labels == best_region.label)
        best_mask = ndimage.binary_fill_holes(best_mask)
        relabeled, count = ndimage.label(best_mask)
        if count > 1:
            comp_areas = ndimage.sum(np.ones_like(relabeled), relabeled, index=np.arange(1, count+1))
            largest_comp = np.argmax(comp_areas) + 1
            best_mask = (relabeled == largest_comp)
        if num_objects > 1:
            best_mask = binary_erosion(best_mask, disk(self.separation_size))

        return best_mask.astype(int)


class CellSegmentationWatershed_standard:
    '''
    This class is intended to detect cells using the Watershed algorithm.
    It segments complete cells and returns the largest segmented object.
    '''
    def __init__(self, image: np.ndarray, footprint_size=5, threshold_method='li', 
                 markers_method='distance', separation_size=5, threshold_factor=1.0):
        self.image = image
        self.footprint_size = footprint_size
        self.threshold_method = threshold_method
        self.markers_method = markers_method
        self.separation_size = separation_size
        self.threshold_factor = threshold_factor  

    def apply_watershed(self):
        # Preprocess the image
        if self.image.ndim == 4:  # [Z, Y, X, C]
            image_proj = np.max(self.image, axis=0)
        elif self.image.ndim == 3 and self.image.shape[2] <= 4:  # [Y, X, C]
            image_proj = self.image
        else:  # [Y, X]
            image_proj = self.image

        # Convert to grayscale if necessary
        if image_proj.ndim == 3 and image_proj.shape[2] > 1:
            image_gray = np.mean(image_proj, axis=2)
        else:
            image_gray = image_proj.squeeze()
        # Normalize and enhance the image
        image_gray = exposure.equalize_adapthist(image_gray)
        # Smooth the image
        smoothed_image = filters.gaussian(image_gray, sigma=4)
        # Apply threshold (compute the base threshold, then adjust by the factor)
        if self.threshold_method == 'otsu':
            base_thresh = filters.threshold_otsu(smoothed_image)
        elif self.threshold_method == 'li':
            base_thresh = filters.threshold_li(smoothed_image)
        else:
            raise ValueError("Unsupported threshold method. Use 'otsu' or 'li'.")
        # Adjust the threshold by the factor from the slider
        #thresh = base_thresh * self.threshold_factor
        thresh = base_thresh / self.threshold_factor
        binary_image = smoothed_image > thresh
        # Remove small objects and holes from binary image before labeling
        binary_image = remove_small_objects(binary_image, min_size=500)
        binary_image = remove_small_holes(binary_image, area_threshold=1000)
        # Compute the distance transform
        distance = ndimage.distance_transform_edt(binary_image)
        # Generate markers for watershed
        if self.markers_method == 'distance':
            markers, _ = ndimage.label(binary_image)
        elif self.markers_method == 'local':
            local_maxi = morphology.local_maxima(distance)
            markers, _ = ndimage.label(local_maxi)
        else:
            raise ValueError("Unsupported markers method. Use 'distance' or 'local'.")
        # Apply watershed algorithm
        labels = watershed(-distance, markers, mask=binary_image)
        # Remove small objects from labels
        min_object_size = 500  # Adjust as needed
        labels = morphology.remove_small_objects(labels, min_size=min_object_size)
        labels = labels.astype(np.int32)
        num_objects = labels.max()
        print(f"Number of segmented objects: {num_objects}")
        if num_objects == 0:
            print("No objects detected.")
            return np.zeros_like(labels, dtype=int)
        # Find the largest object by area
        label_areas = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, num_objects + 1))
        largest_label = np.argmax(label_areas) + 1
        print(f"Largest object label: {largest_label}, Area: {label_areas[largest_label - 1]}")
        largest_label_mask = (labels == largest_label)
        largest_label_mask = ndimage.binary_fill_holes(largest_label_mask)
        # Optionally erode for separation
        if self.separation_size > 0:
            largest_label_mask = binary_erosion(largest_label_mask, footprint=disk(self.separation_size))
        relabeled, count = ndimage.label(largest_label_mask)
        if count > 1:
            comp_areas = ndimage.sum(np.ones_like(relabeled), relabeled, index=np.arange(1, count + 1))
            largest_comp = np.argmax(comp_areas) + 1
            largest_label_mask = (relabeled == largest_comp)
        return largest_label_mask.astype(int)


class CellSegmentation():
    '''
    This class is intended to detect cells in microscope images using `Cellpose <https://github.com/MouseLand/cellpose>`_ . This class segments the nucleus and cytosol for every cell detected in the image. The class uses optimization to generate the meta-parameters used by cellpose. For a complete description of Cellpose check the `Cellpose documentation <https://cellpose.readthedocs.io/en/latest/>`_ .
    
    Parameters
    
    image : NumPy array
        Array of images with dimensions [Z, Y, X, C] or maximum projection with dimensions [Y, X, C], [Y,X].    
    channels_cytosol : List of int
        List with integers indicating the index of channels for the cytosol segmentation. The default is None.
    channels_nucleus : list of int
        List with integers indicating the index of channels for the nucleus segmentation. The default is None. 
    diameter_cytosol : int, optional
        Average cytosol size in pixels. The default is 150.
    diameter_nucleus : int, optional
        Average nucleus size in pixels. The default is 100.
    optimization_segmentation_method: str
        Method used for the segmentation. The options are: \'default\', \'intensity_segmentation\', \'z_slice_segmentation_marker\', \'gaussian_filter_segmentation\', 'diameter_segmentation', and None.
    remove_fragmented_cells: bool, optional
        If true, it removes masks in the border of the image. The default is False.
    show_plot : bool, optional
        If true, it shows a plot with the detected masks. The default is True.
    image_name : str or None.
        Name for the image with detected spots. The default is None.
    '''
    def __init__(self, image:np.ndarray, channels_cytosol = None, channels_nucleus= None, diameter_cytosol:float = 150, diameter_nucleus:float = 100, optimization_segmentation_method='default', remove_fragmented_cells:bool=False, show_plot: bool = True, image_name = None,NUMBER_OF_CORES=1, running_in_pipeline = False, model_nuc_segmentation= 'nuclei', model_cyto_segmentation = 'cyto3',pretrained_model_nuc_segmentation=None, pretrained_model_cyto_segmentation=None,selection_metric='max_cells_and_area'):
        self.image = image
        self.channels_cytosol = channels_cytosol
        self.channels_nucleus = channels_nucleus
        self.diameter_cytosol = diameter_cytosol
        self.diameter_nucleus = diameter_nucleus
        self.show_plot = show_plot
        self.remove_fragmented_cells = remove_fragmented_cells
        self.image_name = image_name
        self.number_z_slices = image.shape[0]
        self.NUMBER_OPTIMIZATION_VALUES= np.min((self.number_z_slices,8))
        self.optimization_segmentation_method = optimization_segmentation_method  # optimization_segmentation_method = 'intensity_segmentation' 'default', 'gaussian_filter_segmentation' , None # 'diameter_segmentation'
        if self.optimization_segmentation_method == 'z_slice_segmentation_marker':
            self.NUMBER_OPTIMIZATION_VALUES= self.number_z_slices
        self.NUMBER_OF_CORES=NUMBER_OF_CORES
        self.running_in_pipeline = running_in_pipeline
        self.model_nuc_segmentation=model_nuc_segmentation                                                                                       
        self.model_cyto_segmentation=model_cyto_segmentation
        self.pretrained_model_nuc_segmentation=pretrained_model_nuc_segmentation
        self.pretrained_model_cyto_segmentation=pretrained_model_cyto_segmentation
        self.MAX_PERCENTILE = 99.5
        self.MIN_PERCENTILE = 0.1
        self.selection_metric = selection_metric #'max_cells_and_area' # 'max_cells' or 'max_area' or 'max_cells_and_area'
        
    def calculate_masks(self):
        '''
        This method performs the process of cell detection for microscope images using **Cellpose**.
        
        Returns
        
        masks_complete_cells : NumPy array. np.uint8
            Image containing the masks for every cell detected in the image. Numpy array with format [Y, X].
        masks_nuclei : NumPy array. np.uint8
            Image containing the masks for every nuclei detected in the image. Numpy array with format [Y, X].
        masks_cytosol_no_nuclei : NumPy array. np.uint8
            Image containing the masks for every cytosol (removing the nucleus) detected in the image. Numpy array with format [Y, X].
        '''
        # function that determines if the nucleus is in the cytosol
        def is_nucleus_in_cytosol(mask_n, mask_c):
            mask_n[mask_n>1]=1
            mask_c[mask_c>1]=1
            size_mask_n = np.count_nonzero(mask_n)
            size_mask_c = np.count_nonzero(mask_c)
            min_size =np.min( (size_mask_n,size_mask_c) )
            mask_combined =  mask_n + mask_c
            sum_mask = np.count_nonzero(mask_combined[mask_combined==2])
            if (sum_mask> min_size*0.8) and (min_size>200): # the element is inside if the two masks overlap over the 80% of the smaller mask.
                return 1
            else:
                return 0
        ##### IMPLEMENTATION #####
        if len(self.image.shape) > 3:  # [ZYXC]
            if self.image.shape[0] ==1:
                max_image = self.image[0,:,:,:]
            else:
                center_slice = self.number_z_slices//2
                max_image = np.max(self.image[:,:,:,:],axis=0)    # taking the mean value
                max_image = RemoveExtrema(max_image,min_percentile=1, max_percentile=98).remove_outliers() 
        else:
            max_image = self.image # [YXC] 
            max_image = RemoveExtrema(max_image,min_percentile=1, max_percentile=98).remove_outliers() 
        # Function to calculate the approximated radius in a mask
        def approximated_radius(masks,diameter=100):
            n_masks = np.max(masks)
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                size_mask = []
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    approximated_radius = np.sqrt(np.sum(masks == nm)/np.pi)  # a=  pi r2
                    size_mask.append(approximated_radius) # creating a list with the size of each mask
                array_radii =np.array(size_mask)
            else:
                array_radii = np.sqrt(np.sum(masks == 1)/np.pi)
            if np.any(array_radii > diameter*1.5) |  np.any(array_radii < 10):
                masks_radii = 0
            else:
                masks_radii= np.prod (array_radii)
            return masks_radii
        def metric_paired_masks(masks_complete_cells,masks_nuclei):
            median_radii_complete_cells_complete_cells = approximated_radius(masks_complete_cells,diameter=self.diameter_cytosol)
            median_radii_nuclei = approximated_radius(masks_nuclei,diameter=self.diameter_nucleus)
            return  median_radii_nuclei * median_radii_complete_cells_complete_cells
        
        def metric_max_cells_and_area(masks, mode='max_cells_and_area'):
            """
            Calculates a metric based on the masks to optimize cell segmentation.

            Parameters:
            - masks: numpy array containing the segmented masks.
            - mode: str, optional
                - 'max_cells': Maximizes the number of cells detected.
                - 'max_area': Maximizes the area of the largest cell.
                - 'max_cells_and_area': Maximizes both the number of cells and their average area.

            Returns:
            - metric: int or float representing the calculated metric.
            """
            n_masks = np.max(masks)
            if n_masks >= 1:
                size_masks = []
                for nm in range(1, n_masks + 1):
                    # Calculate the area of each mask
                    area = np.sum(masks == nm)
                    size_masks.append(area)
                size_masks_array = np.array(size_masks)
                total_area = np.sum(size_masks_array)
                mean_area = np.mean(size_masks_array)
                max_area = np.max(size_masks_array)
                if mode == 'max_cells':
                    metric = n_masks
                elif mode == 'max_area':
                    metric = max_area
                elif mode == 'max_cells_and_area':
                    metric = n_masks * mean_area
                else:
                    raise ValueError("Invalid mode. Choose 'max_cells', 'max_area', or 'max_cells_and_area'.")
            else:
                metric = 0
            return metric
                
        # Function to find masks
        def function_to_find_masks (image, diameter_cytosol=None,diameter_nucleus=None): 
            if diameter_cytosol is None:
                diameter_cytosol = self.diameter_cytosol
            if diameter_nucleus is None:
                diameter_nucleus = self.diameter_nucleus
            if not (self.channels_cytosol in (None,[None])):
                masks_cyto = Cellpose(image[:, :, self.channels_cytosol],diameter = diameter_cytosol, model_type = self.model_cyto_segmentation, selection_method = self.selection_metric ,NUMBER_OF_CORES=self.NUMBER_OF_CORES,pretrained_model=self.pretrained_model_cyto_segmentation).calculate_masks()
            else:
                masks_cyto = np.zeros_like(image[:, :, 0])
            if not (self.channels_nucleus in (None,[None])):
                masks_nuclei = Cellpose(image[:, :, self.channels_nucleus],  diameter = diameter_nucleus, model_type = self.model_nuc_segmentation, selection_method = self.selection_metric,NUMBER_OF_CORES=self.NUMBER_OF_CORES,pretrained_model=self.pretrained_model_nuc_segmentation).calculate_masks()
            else:
                masks_nuclei= np.zeros_like(image[:, :, 0])
            if not (self.channels_cytosol in (None,[None])) and not(self.channels_nucleus in (None,[None])):
                # Function that removes masks that are not paired with a nucleus or cytosol
                def remove_lonely_masks(masks_0, masks_1,is_nuc=None):
                    n_mask_0 = np.max(masks_0)
                    n_mask_1 = np.max(masks_1)
                    if (n_mask_0>0) and (n_mask_1>0):
                        for ind_0 in range(1,n_mask_0+1):
                            tested_mask_0 = erosion(np.where(masks_0 == ind_0, 1, 0))
                            array_paired= np.zeros(n_mask_1)
                            for ind_1 in range(1,n_mask_1+1):
                                tested_mask_1 = erosion(np.where(masks_1 == ind_1, 1, 0))
                                array_paired[ind_1-1] = is_nucleus_in_cytosol(tested_mask_1, tested_mask_0)
                                if (is_nuc =='nuc') and (np.count_nonzero(tested_mask_0) > np.count_nonzero(tested_mask_1) ):
                                    # condition that rejects images with nucleus bigger than the cytosol
                                    array_paired[ind_1-1] = 0
                                elif (is_nuc is None ) and (np.count_nonzero(tested_mask_1) > np.count_nonzero(tested_mask_0) ):
                                    array_paired[ind_1-1] = 0
                            if any (array_paired) == False: # If the cytosol is not associated with any mask.
                                masks_0 = np.where(masks_0 == ind_0, 0, masks_0)
                            masks_pairs = masks_0
                    else:
                        masks_pairs = np.zeros_like(masks_0)
                    return masks_pairs
                # Function that reorder the index to make it continuos 
                def reorder_masks(mask_tested):
                    n_mask_0 = np.max(mask_tested)
                    mask_new =np.zeros_like(mask_tested)
                    if n_mask_0>0:
                        counter = 0
                        for ind_0 in range(1,n_mask_0+1):
                            if ind_0 in mask_tested:
                                counter = counter + 1
                                if counter ==1:
                                    mask_new = np.where(mask_tested == ind_0, -counter, mask_tested)
                                else:
                                    mask_new = np.where(mask_new == ind_0, -counter, mask_new)
                        reordered_mask = np.absolute(mask_new)
                    else:
                        reordered_mask = mask_new
                    return reordered_mask  
                # Cytosol masks
                masks_cyto = remove_lonely_masks(masks_cyto, masks_nuclei)
                masks_cyto = reorder_masks(masks_cyto)
                # Masks nucleus
                masks_nuclei = remove_lonely_masks(masks_nuclei, masks_cyto,is_nuc='nuc')
                masks_nuclei = reorder_masks(masks_nuclei)
                # Iterate for each cyto mask
                def matching_masks(masks_cyto, masks_nuclei):
                    n_mask_cyto = np.max(masks_cyto)
                    n_mask_nuc = np.max(masks_nuclei)
                    new_masks_nuclei = np.zeros_like(masks_cyto)
                    reordered_mask_nuclei = np.zeros_like(masks_cyto)     
                    if (n_mask_cyto>0) and (n_mask_nuc>0):
                        for mc in range(1,n_mask_cyto+1):
                            tested_mask_cyto = np.where(masks_cyto == mc, 1, 0)
                            for mn in range(1,n_mask_nuc+1):
                                mask_paired = False
                                tested_mask_nuc = np.where(masks_nuclei == mn, 1, 0)
                                mask_paired = is_nucleus_in_cytosol(tested_mask_nuc, tested_mask_cyto)
                                if mask_paired == True:
                                    if np.count_nonzero(new_masks_nuclei) ==0:
                                        new_masks_nuclei = np.where(masks_nuclei == mn, -mc, masks_nuclei)
                                    else:
                                        new_masks_nuclei = np.where(new_masks_nuclei == mn, -mc, new_masks_nuclei)
                            reordered_mask_nuclei = np.absolute(new_masks_nuclei)
                    else:
                        masks_cyto = np.zeros_like(masks_cyto)
                        reordered_mask_nuclei = np.zeros_like(masks_nuclei)
                    return masks_cyto, reordered_mask_nuclei
                # Matching nuclei and cytosol
                masks_cyto, masks_nuclei = matching_masks(masks_cyto, masks_nuclei)                
                #Generating mask for cyto without nuc
                masks_cytosol_no_nuclei = masks_cyto - masks_nuclei
                masks_cytosol_no_nuclei[masks_cytosol_no_nuclei < 0] = 0
                masks_cytosol_no_nuclei.astype(int)
                # Renaming 
                masks_complete_cells = masks_cyto
            else:
                if not (self.channels_cytosol in (None,[None])):
                    masks_complete_cells = masks_cyto
                    masks_nuclei = None 
                    masks_cytosol_no_nuclei = None
                if not (self.channels_nucleus in (None, [None])):
                    masks_complete_cells = masks_nuclei
                    masks_cytosol_no_nuclei = None
            return masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei
        # OPTIMIZATION METHODS FOR SEGMENTATION
        if (self.optimization_segmentation_method == 'intensity_segmentation') and (len(self.image.shape) > 3) and (self.image.shape[0]>1):
            # Intensity Based Optimization to find the maximum number of index_paired_masks. 
            sigma_value_for_gaussian = 2
            list_masks_complete_cells = []
            list_masks_nuclei = []
            list_masks_cytosol_no_nuclei = []
            if not (self.channels_cytosol in (None,[None])) or not(self.channels_nucleus in (None,[None])):
                tested_thresholds = np.round(np.linspace(0, 5, self.NUMBER_OPTIMIZATION_VALUES), 0)
                array_number_paired_masks  = np.zeros( len(tested_thresholds))
                for idx, threshold in enumerate(tested_thresholds):
                    image_temp = RemoveExtrema(max_image,min_percentile=self.MIN_PERCENTILE, max_percentile=100-threshold).remove_outliers() 
                    image_filtered = stack.gaussian_filter(image_temp,sigma=sigma_value_for_gaussian)  #max_image.copy()
                    masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(image_filtered)
                    list_masks_complete_cells.append(masks_complete_cells)
                    list_masks_nuclei.append(masks_nuclei)
                    list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei)
                    metric = Utilities().metric_max_cells_and_area(masks_complete_cells, mode=self.selection_metric)
                    array_number_paired_masks[idx] = metric
                selected_index = np.argmax(array_number_paired_masks)
                masks_complete_cells = list_masks_complete_cells[selected_index]
                masks_nuclei = list_masks_nuclei [selected_index]
                masks_cytosol_no_nuclei = list_masks_cytosol_no_nuclei [selected_index]
            else:
                raise ValueError("Error: No nucleus or cytosol channels were selected. ")        
        

        elif (self.optimization_segmentation_method == 'z_slice_segmentation_marker') and (len(self.image.shape) > 3) and (self.image.shape[0] > 1):
            # Optimization based on selecting a z-slice or combination of z-slices to find the maximum number of index_paired_masks.
            list_idx = np.round(np.linspace(0, self.number_z_slices - 1, self.NUMBER_OPTIMIZATION_VALUES), 0).astype(int)
            list_idx = np.unique(list_idx)
            
            if not (self.channels_cytosol in (None, [None])) or not (self.channels_nucleus in (None, [None])):                
                # Lists to store results
                list_masks_complete_cells = []
                list_masks_nuclei = []
                list_masks_cytosol_no_nuclei = []
                array_number_paired_masks = np.zeros(len(list_idx) + 1)
                # Function to process each Z-slice or combination
                def process_slice(idx_value):
                    # You can adjust the range of slices combined here
                    slice_range = range(max(0, idx_value - 1), min(self.number_z_slices, idx_value + 2))
                    test_image_optimization = np.max(self.image[slice_range, :, :, :], axis=0)
                    test_image_optimization = RemoveExtrema(test_image_optimization, min_percentile=self.MIN_PERCENTILE, max_percentile=self.MAX_PERCENTILE).remove_outliers()
                    masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization)
                    metric = metric_max_cells_and_area(masks_complete_cells, mode=self.selection_metric)
                    return masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei, metric
                # Process the maximum projection of all slices first
                test_image_optimization = np.max(self.image[:, :, :, :], axis=0)
                test_image_optimization = RemoveExtrema(test_image_optimization, min_percentile=self.MIN_PERCENTILE, max_percentile=self.MAX_PERCENTILE).remove_outliers()
                masks_complete_cells_max, masks_nuclei_max, masks_cytosol_no_nuclei_max = function_to_find_masks(test_image_optimization)
                metric_max = metric_max_cells_and_area(masks_complete_cells_max, mode=self.selection_metric)
                # Store the result from the maximum projection
                list_masks_complete_cells.append(masks_complete_cells_max)
                list_masks_nuclei.append(masks_nuclei_max)
                list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei_max)
                array_number_paired_masks[0] = metric_max
                # Process slices in parallel
                results = Parallel(n_jobs=self.NUMBER_OF_CORES)(
                    delayed(process_slice)(idx_value) for idx_value in list_idx
                )
                # Collect results
                for idx, (masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei, metric) in enumerate(results):
                    list_masks_complete_cells.append(masks_complete_cells)
                    list_masks_nuclei.append(masks_nuclei)
                    list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei)
                    array_number_paired_masks[idx + 1] = metric
                # Select the best result
                selected_index = np.argmax(array_number_paired_masks)
                masks_complete_cells = list_masks_complete_cells[selected_index]
                masks_nuclei = list_masks_nuclei[selected_index]
                masks_cytosol_no_nuclei = list_masks_cytosol_no_nuclei[selected_index]
            else:
                raise ValueError("Error: No nucleus or cytosol channels were selected.")
        elif (self.optimization_segmentation_method == 'default') and (len(self.image.shape) > 3) and (self.image.shape[0]>1):
            # Optimization based on selecting a z-slice to find the maximum number of index_paired_masks.
            if self.number_z_slices > 20:
                num_slices_range = 3  # range to consider above and below a selected z-slice
            else:
                num_slices_range = 1
            list_idx = np.round(np.linspace(num_slices_range, self.number_z_slices-num_slices_range, self.NUMBER_OPTIMIZATION_VALUES), 0).astype(int)  
            list_idx = np.unique(list_idx)  #list(set(list_idx))
            # Optimization based on slice
            if not (self.channels_cytosol in (None,[None])) or not(self.channels_nucleus in (None,[None])):
                list_masks_complete_cells = []
                list_masks_nuclei = []
                list_masks_cytosol_no_nuclei = []
                array_number_paired_masks = np.zeros( len(list_idx) +1)
                # performing the segmentation on a maximum projection
                test_image_optimization = np.max(self.image[num_slices_range:-num_slices_range,:,:,:],axis=0) 
                test_image_optimization = RemoveExtrema(test_image_optimization,min_percentile=self.MIN_PERCENTILE, max_percentile=self.MAX_PERCENTILE).remove_outliers()  
                masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization)     
                list_masks_complete_cells.append(masks_complete_cells)
                list_masks_nuclei.append(masks_nuclei)
                list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei)           
                metric = Utilities().metric_max_cells_and_area(masks_complete_cells, mode=self.selection_metric)
                array_number_paired_masks[0] = metric
                # performing segmentation for a subsection of z-slices
                for idx, idx_value in enumerate(list_idx):
                    test_image_optimization = np.max(self.image[idx_value-num_slices_range:idx_value+num_slices_range,:,:,:],axis=0)  
                    test_image_optimization = RemoveExtrema(test_image_optimization,min_percentile=self.MIN_PERCENTILE, max_percentile=self.MAX_PERCENTILE).remove_outliers()  
                    masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization)
                    list_masks_complete_cells.append(masks_complete_cells)
                    list_masks_nuclei.append(masks_nuclei)
                    list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei)
                    metric = Utilities().metric_max_cells_and_area(masks_complete_cells, mode=self.selection_metric)
                    array_number_paired_masks[idx+1] = metric
                selected_index = np.argmax(array_number_paired_masks)
                masks_complete_cells = list_masks_complete_cells[selected_index]
                masks_nuclei = list_masks_nuclei [selected_index]
                masks_cytosol_no_nuclei = list_masks_cytosol_no_nuclei [selected_index]
            else:
                raise ValueError("Error: No nucleus or cytosol channels were selected. ")      
        elif (self.optimization_segmentation_method == 'center_slice') and (len(self.image.shape) > 3) and (self.image.shape[0]>1):
            # Optimization based on selecting a z-slice to find the maximum number of index_paired_masks. 
            center_slice = self.number_z_slices//2
            num_slices_range = np.min( (5,center_slice-1))  # range to consider above and below a selected z-slice
            # Optimization based on slice
            test_image_optimization = self.image[center_slice,:,:,:] 
            test_image_optimization = RemoveExtrema(test_image_optimization,min_percentile=self.MIN_PERCENTILE, max_percentile=self.MAX_PERCENTILE).remove_outliers()  
            masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization)
        elif (self.optimization_segmentation_method == 'gaussian_filter_segmentation') and (len(self.image.shape) > 3) and (self.image.shape[0]>1):
            # Optimization based on testing different sigmas in a gaussian filter to find the maximum number of index_paired_masks. 
            list_sigmas = np.round(np.linspace(0.5, 4, self.NUMBER_OPTIMIZATION_VALUES), 1) 
            # Optimization based on slice
            if not (self.channels_cytosol in (None,[None])) or not(self.channels_nucleus in (None,[None])):
                #list_sorting_number_paired_masks = []
                list_masks_complete_cells = []
                list_masks_nuclei = []
                list_masks_cytosol_no_nuclei = []
                array_number_paired_masks = np.zeros( len(list_sigmas)  )
                for idx, sigma_value in enumerate(list_sigmas):
                    test_image_optimization = stack.gaussian_filter(np.max(self.image[:,:,:,:],axis=0),sigma=sigma_value) 
                    test_image_optimization = RemoveExtrema(test_image_optimization,min_percentile=self.MIN_PERCENTILE, max_percentile=self.MAX_PERCENTILE).remove_outliers()   
                    masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(test_image_optimization)
                    list_masks_complete_cells.append(masks_complete_cells)
                    list_masks_nuclei.append(masks_nuclei)
                    list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei)  
                    #metric = Utilities().metric_max_cells_and_area(masks_complete_cells)
                    metric = Utilities().metric_max_cells_and_area(masks_complete_cells, mode=self.selection_metric)
                    array_number_paired_masks[idx] = metric
                #selected_threshold = list_sigmas[np.argmax(array_number_paired_masks)]
                selected_index = np.argmax(array_number_paired_masks)
                masks_complete_cells = list_masks_complete_cells[selected_index]
                masks_nuclei = list_masks_nuclei [selected_index]
                masks_cytosol_no_nuclei = list_masks_cytosol_no_nuclei [selected_index]
            else:
                raise ValueError("Error: No nucleus or cytosol channels were selected. ")
        else:
            # no optimization is applied if a 2D image is passed
            masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = function_to_find_masks(max_image)
        if self.running_in_pipeline == False :
            Plots().plotting_masks_and_original_image(image = max_image, 
                                                    masks_complete_cells = masks_complete_cells, 
                                                    masks_nuclei = masks_nuclei, 
                                                    channels_cytosol = self.channels_cytosol, 
                                                    channels_nucleus = self.channels_nucleus,
                                                    image_name = self.image_name,
                                                    show_plot = self.show_plot)
        return masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei 


class ManualSegmentation():
    '''
    Create a mask object from user-drawn polygons. The image format must be [Y, X].

    Parameters:
        image : NumPy array
            Image with dimensions [Y, X].
        cmap : str
            Color map to use for the image display. Default is 'Spectral'.
        polygon_color : tuple 
            RGB color of the polygon as a tuple (255,0,0).
    '''

    def __init__(self, image, cmap='Spectral', polygon_color=(255,0,0)):        
        self.ipython = get_ipython()
        if self.ipython:
            self.ipython.run_line_magic('matplotlib', 'widget')

        def process_image(image):
            processed_image = RemoveExtrema(image, min_percentile=0.1, max_percentile=99).remove_outliers()
            processed_image = (processed_image - processed_image.min())/(processed_image.max() - processed_image.min()) * 255
            return processed_image
        self.image = process_image(image)
        self.polygon_color = polygon_color
        self.selected_points = []
        self.figure_to_draw_points, self.axes_in_figure = plt.subplots(figsize=(5, 5))
        self.new_image = self.axes_in_figure.imshow(self.image, cmap=cmap)
        self.click = self.figure_to_draw_points.canvas.mpl_connect('button_press_event', self.onclick)

    def polygon(self, new_image, points_in_polygon):
        points_in_polygon = np.array(points_in_polygon, np.int32)
        points_in_polygon = points_in_polygon.reshape((-1, 1, 2))
        cv2.polylines(new_image, [points_in_polygon], isClosed=True, color=self.polygon_color, thickness=3)
        return new_image
    
    def switch_to_inline(self):
        if self.ipython:
            self.ipython.run_line_magic('matplotlib', 'inline')
            plt.show()  # Ensure that any existing plots are displayed
        return None

    def onclick(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.selected_points.append([int(event.xdata), int(event.ydata)])
            updated_image = self.polygon(np.copy(self.image), self.selected_points)
            for point in self.selected_points:
                cv2.circle(updated_image, tuple(point), radius=3, color=self.polygon_color, thickness=-1)
            self.new_image.set_data(updated_image)
        return None
    
    def close_and_save(self, filename='temp_mask.tif',save_mask=True):
        if self.selected_points:
            # Create an empty array with the same shape as the image slice
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            # Convert selected points to a numpy array for cv2.fillPoly
            mask_array = np.array([self.selected_points], dtype=np.int32)
            # Create the mask
            cv2.fillPoly(mask, mask_array, 255)  # Fill with white (255)
            if save_mask == True:   
                # Save the mask as a tif file
                print(f'Mask saved as {filename}')
                image = mask.astype(np.uint8)
                imageio.imwrite(filename, image)
            # close the figure and disconnect the click
            self.figure_to_draw_points.canvas.mpl_disconnect(self.click)
            plt.close(self.figure_to_draw_points)
            plt.close()
            self.switch_to_inline()  # Switch back to inline when closing the plot
            return mask.astype(bool)
        else:
            print('No points selected to create a mask.')
            plt.close()
            return None
        
class MultiManualSegmentation:
    """
    ManualSegmentation class for interactive segmentation of multiple objects in an image.
    
    Features:
    - Draw multiple polygons (each polygon defines one object region).
    - Each polygon is assigned a unique integer label in a combined segmentation mask.
    - After closing a polygon, prompts the user for a class name label, stored with the object.
    - Supports resetting the in-progress polygon (before closing) with Esc, without removing completed polygons.
    - Provides methods to retrieve the final mask and class name annotations for each labeled object.
    
    Usage:
      seg = ManualSegmentation(image)  # display image and start annotation
    - Left-click on the image to add polygon vertices.
    - Right-click or press Enter to finish the current polygon.
    - When prompted, type the class name for that object and press Enter.
    - Press Esc to cancel the polygon currently being drawn (if needed).
    - Repeat for multiple objects. 
    - After finishing, call seg.get_mask() and seg.get_class_map() to obtain results.
    """
    def __init__(self, image, cmap='Greys_r'):
        """
        Initialize the manual segmentation tool for a given image.
        
        Parameters:
        - image: numpy array of shape (H, W) or (H, W, C) representing the image to segment.
        """
        # Store image and set up mask
        self.ipython = get_ipython()
        if self.ipython:
            self.ipython.run_line_magic('matplotlib', 'widget')
        self.image = image
        self.height, self.width = image.shape[:2]
        # Mask initialized to 0 (background); will be filled with object labels
        self.mask = np.zeros((self.height, self.width), dtype=np.int32)
        # List to store annotations: each entry is {'label': int, 'class': str, 'polygon': [(x,y), ...]}
        self.objects = []
        self.label_count = 0  # Counter for labels (will assign 1,2,3,...)
        self.class_name_for_label = {}  # Dict mapping label -> class name
        # State for the polygon currently being drawn
        self.current_polygon = []           # list of (x, y) points for in-progress polygon
        self.current_polygon_artists = []   # matplotlib artists (dots/lines) for in-progress polygon
        self.awaiting_label = False         # flag to pause drawing while waiting for class name input
        self.cmap = cmap # default colormap for displaying the image
        
        # Set up figure and axes for the image
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image, cmap=cmap)
        # Add instruction text as title
        self.ax.set_title("Left-click: add point   Double-click/Enter: finish polygon   Esc: cancel polygon", fontsize=9)
        self.ax.set_xticks([]); self.ax.set_yticks([])  # hide axis ticks
        # Connect event handlers for mouse clicks and key presses
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_key   = self.fig.canvas.mpl_connect('key_press_event',   self._on_key_press)
        
        plt.show(block=False)  # display the interactive plot (non-blocking)

    def _on_click(self, event):
        """Handle mouse click events: add points or finish polygon on double-click."""
        # Don't process events if we're waiting on class name input
        if self.awaiting_label:
            return

        # Ignore clicks outside the image axes
        if event.inaxes != self.ax:
            return

        # Handle left-click events
        if event.button == 1:
            # If this is a double-click, check if we can finalize the polygon.
            if event.dblclick:
                if len(self.current_polygon) >= 3:
                    self._finalize_polygon()
                else:
                    print("Polygon must have at least 3 points. Continue drawing or press Esc to cancel.")
                return  # exit so a double-click doesn't also add a point
            
            # Process a normal single left-click: add a vertex to the polygon
            if event.xdata is None or event.ydata is None:
                return
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            # Clamp coordinates within image boundaries
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            self.current_polygon.append((x, y))
            # Mark the vertex with a red dot
            pt_artist, = self.ax.plot(x, y, 'ro')
            self.current_polygon_artists.append(pt_artist)
            # If this is not the first point, draw a line from the previous one
            if len(self.current_polygon) > 1:
                px, py = self.current_polygon[-2]
                line_artist, = self.ax.plot([px, x], [py, y], 'r-')
                self.current_polygon_artists.append(line_artist)
            self.fig.canvas.draw()
    
    def _on_key_press(self, event):
        """Handle key press events: Enter to finish polygon, Esc to cancel current polygon."""
        if self.awaiting_label:
            return  # ignore key presses while awaiting class name input
        if event.key == 'escape':
            # Cancel current polygon drawing (remove partial points/lines)
            self._reset_current_polygon()
        elif event.key in ['enter', 'return']:
            if len(self.current_polygon) >= 3:
                self._finalize_polygon()
            else:
                print("Polygon must have at least 3 points. Continue drawing or press Esc to cancel.")
    
    def _reset_current_polygon(self):
        """Clear the points and temporary drawings of the current in-progress polygon."""
        for artist in self.current_polygon_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.current_polygon_artists = []
        self.current_polygon = []
        self.fig.canvas.draw()
    
    def _finalize_polygon(self):
        """Complete the current polygon, assign a new label, and prompt for class name."""
        # Ensure polygon has at least 3 vertices
        if len(self.current_polygon) < 3:
            return
        # Increment label count for this new object
        self.label_count += 1
        label_val = self.label_count
        # Draw the polygon as a filled patch (with a random color) on the image
        color = (np.random.random(), np.random.random(), np.random.random())  # random color for this object
        poly_patch = Polygon(np.array(self.current_polygon), closed=True,
                                     facecolor=color, edgecolor=color, alpha=0.4, linewidth=2)
        self.ax.add_patch(poly_patch)
        # Remove all temporary vertex markers/lines used during drawing
        for artist in self.current_polygon_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.current_polygon_artists = []
        self.fig.canvas.draw()
        
        # Pause drawing and prompt user for class name via a TextBox widget
        self.awaiting_label = True
        axbox = self.fig.add_axes([0.25, 0.01, 0.5, 0.05])  # textbox at bottom of figure
        axbox.set_facecolor((0.9, 0.9, 0.9))
        prompt = f"Object {label_val} class name: "
        text_box = TextBox(axbox, prompt)
        
        # Define what happens when the user submits the class name
        def on_submit(text):
            class_name = text.strip()
            if class_name == "":
                class_name = f"object_{label_val}"
            # Record the class name and polygon info
            self.class_name_for_label[label_val] = class_name
            self.objects.append({
                "label": label_val,
                "class": class_name,
                "polygon": list(self.current_polygon)
            })
            # Fill the polygon area in the mask with the label value
            self._fill_polygon_mask(self.current_polygon, label_val)
            # Optionally, label the polygon on the plot with the class name
            cx = np.mean([p[0] for p in self.current_polygon])
            cy = np.mean([p[1] for p in self.current_polygon])
            self.ax.text(cx, cy, class_name, color='white', ha='center', va='center',
                         bbox=dict(facecolor=color, edgecolor='none', alpha=0.5, pad=1))
            # Cleanup: remove the TextBox and resume normal operation
            text_box.disconnect_events()
            axbox.remove()
            self.fig.canvas.draw()
            self.current_polygon = []  # reset current polygon list for the next object
            self.awaiting_label = False
        
        text_box.on_submit(on_submit)
    
    def _fill_polygon_mask(self, polygon_points, label_val):
        """Fill the given polygon area in the mask with the specified label value."""
        try:
            # Use PIL for efficient polygon filling
            mask_img = Image.fromarray(self.mask.astype('int32'), mode='I')
            draw = ImageDraw.Draw(mask_img)
            # Polygon points are in (x, y) format which matches ImageDraw coordinate system
            draw.polygon(polygon_points, outline=label_val, fill=label_val)
            # Update the numpy mask with the filled polygon
            self.mask = np.array(mask_img, dtype=np.int32)
        except ImportError:
            # Fallback: manual fill using matplotlib Path (less efficient)
            poly_path = matplotlib_path(polygon_points)
            y_grid, x_grid = np.mgrid[0:self.height, 0:self.width]
            coords = np.vstack((x_grid.flatten(), y_grid.flatten())).T  # shape (N_pixels, 2)
            inside = poly_path.contains_points(coords).reshape(self.height, self.width)
            self.mask[inside] = label_val
    
    def get_mask(self):
        """Return the segmentation mask array. Background is 0; objects have unique labels 1..N."""
        return self.mask.copy()
    
    def get_class_map(self):
        """Return a dictionary mapping each label (1..N) to its class name."""
        return dict(self.class_name_for_label)
    
    def get_mask_for_class(self, class_name):
        """
        Return a binary mask array for the specified class name.
        The binary mask is True for pixels that belong to any polygon with the
        specified class.

        Parameters:
            class_name (str): The class name to filter for (e.g., "cat").

        Returns:
            np.ndarray: A boolean array where pixels for the specified class are True.
        """
        # Retrieve the combined mask and class mapping
        combined_mask = self.get_mask()   # the mask with integer labels
        class_map = self.get_class_map()    # dictionary mapping labels (ints) to class names

        # Debug: print the current class map to check for duplicate entries.
        print("Current class map:", class_map)
        
        # Get all labels corresponding to the given class (ignoring case)
        labels = [label for label, cls in class_map.items() if cls.lower() == class_name.lower()]
        
        if not labels:
            print(f"No objects labeled '{class_name}' found.")
            return np.zeros_like(combined_mask, dtype=bool)
        
        # Debug: print out the labels found for the requested class.
        print(f"Labels for class '{class_name}':", labels)
        
        # Create a binary mask: accumulate True for each label region.
        binary_mask = np.zeros_like(combined_mask, dtype=bool)
        for lab in labels:
            binary_mask |= (combined_mask == lab)

        if self.ipython:
            plt.close(self.fig)
            self.ipython.run_line_magic('matplotlib', 'inline')
        
        return binary_mask
    


class LineProfile:
    """
    Interactive line profile tool: click two points on the displayed image to draw
    a line and compute pixel intensities along that line.

    After the second click, the line is drawn and the profile is calculated.
    Use get_profile() to retrieve (x_coords, y_coords, intensities).
    If max_pixels is set, the second point will be adjusted to match that distance.
    """

    def __init__(self, image, ax=None, cmap='gray', max_pixels=None):
        self.ipython = get_ipython()
        if self.ipython:
            self.ipython.run_line_magic('matplotlib', 'widget')
        self.image = image
        self.height, self.width = image.shape[:2]
        self.max_pixels = max_pixels  # maximum length of line in pixels
        # Set up figure and axes
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
        else:
            self.ax = ax
            self.fig = ax.figure
        self.ax.imshow(self.image, cmap=cmap)
        self.ax.set_title("Click start point, then end point for line profile", fontsize=10)
        self.start = None
        self.end = None
        self.line_artist = None
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.x = None
        self.y = None
        self.intensities = None

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        xpix = int(round(event.xdata))
        ypix = int(round(event.ydata))

        if self.start is None:
            self.start = (ypix, xpix)
            self.ax.plot(xpix, ypix, 'ro')
            self.fig.canvas.draw()
        else:
            y0, x0 = self.start
            y1, x1 = ypix, xpix

            dx = x1 - x0
            dy = y1 - y0
            dist = np.hypot(dx, dy)
            # Constrain to max_pixels if set
            if self.max_pixels is not None and dist > self.max_pixels:
                scale = self.max_pixels / dist
                dx = dx * scale
                dy = dy * scale
                x1 = int(round(x0 + dx))
                y1 = int(round(y0 + dy))
            self.end = (y1, x1)
            if self.line_artist:
                try:
                    self.line_artist.remove()
                except Exception:
                    pass
            self.line_artist, = self.ax.plot([x0, x1], [y0, y1], 'r-')
            self.fig.canvas.draw()
            # Compute intensity profile
            self.intensities = profile_line(self.image, (y0, x0), (y1, x1), mode='reflect')
            length = len(self.intensities)
            self.x = np.linspace(x0, x1, length)
            self.y = np.linspace(y0, y1, length)
            self.fig.canvas.mpl_disconnect(self.cid_click)

    def get_profile(self):
        """
        Returns three arrays: x_coords, y_coords, intensities.
        Must be called after two clicks to define the line.
        Also closes the figure and resets matplotlib to inline mode.
        """
        if self.ipython:
            plt.close(self.fig)
            self.ipython.run_line_magic('matplotlib', 'inline')
        return self.x, self.y, self.intensities


class TrackPyDetection:
    '''
    This class detects spots in microscope images using TrackPy.
    The format of the image must be [Z, Y, X, C] for 3D images or [Y, X, C] for 2D images.

    Parameters
    ----------
    image : NumPy array
        Array of images with dimensions [Z, Y, X, C] or [Y, X, C].
    channels_spots : int
        Specific channel index that contains spot signals.
    voxel_size_z : float, optional
        Height of a voxel along the z-axis in nanometers. Default is 300.
    voxel_size_yx : float, optional
        Size of a voxel on the yx-plane in nanometers. Default is 150.
    show_plot : bool, optional
        If True, displays plots of detected spots. Default is False.
    image_name : str or None, optional
        Base name for saving output images. Default is None.
    save_all_images : bool, optional
        If True, saves plots for all z-slices (for 3D images). Default is False.
    display_spots_on_multiple_z_planes : bool, optional
        If True, displays spots on adjacent z-planes. Default is False.
    use_log_filter_for_spot_detection : bool, optional
        If True, applies a Gaussian filter for spot detection. Default is True.
    threshold_for_spot_detection : float or None, optional
        Intensity threshold for spot detection. If None, an automatic threshold is calculated using Otsu's method. Default is None.
    save_files : bool, optional
        If True, saves output plots as image files. Default is True.
    '''

    def __init__(self, image, channels_spots, voxel_size_yx=150, yx_spot_size_in_px=5, 
                 show_plot=False, image_name=None, save_all_images=False, spot_diameter=5,
                 display_spots_on_multiple_z_planes=False, use_max_projection=True,
                 threshold_for_spot_detection=None, save_files=False):
        # Validate image dimensions
        if len(image.shape) < 4:
            image = np.expand_dims(image, axis=0)  # Add Z dimension if missing
        self.image = image
        self.channels_spots = channels_spots
        self.voxel_size_yx = voxel_size_yx
        self.yx_spot_size_in_px = yx_spot_size_in_px
        self.show_plot = show_plot
        self.image_name = image_name
        self.save_all_images = save_all_images
        self.display_spots_on_multiple_z_planes = display_spots_on_multiple_z_planes
        self.threshold_for_spot_detection = threshold_for_spot_detection
        self.save_files = save_files
        self.spot_diameter = spot_diameter

    def detect(self):
        '''
        Detects spots using TrackPy.

        Returns
        -------
        clusters_and_spots : np.ndarray
            Array with shape (nb_clusters, 4). Each row contains (z, y, x, num_spots_in_cluster).
            Since clustering is not performed, num_spots_in_cluster is always 1.
        rna_filtered : np.ndarray
            Filtered image array, same shape as input.
        threshold : float
            Threshold used for spot detection.
        '''
        if len(self.image.shape) == 4:
            # 3D image: [Z, Y, X, C]
            spot_channel = np.max(self.image[:, :, :, self.channels_spots],axis=0)   
        # Calculating Sigma with  the parameters for the PSF.
        spot_radius_px = detection.get_object_radius_pixel(
                        voxel_size_nm=(self.voxel_size_yx, self.voxel_size_yx), 
                        object_radius_nm=(self.voxel_size_yx*(self.yx_spot_size_in_px//2) , self.voxel_size_yx*(self.yx_spot_size_in_px//2)), ndim=2)  
        sigma = spot_radius_px
        ## SPOT DETECTION
        try:
            rna_filtered = stack.log_filter(spot_channel, sigma) # LoG filter
        except ValueError:
            print('Error during the log filter calculation, try using larger parameters values for the psf')
            rna_filtered = stack.remove_background_gaussian(spot_channel, sigma)

        # Determine threshold for spot detection
        if self.threshold_for_spot_detection is not None:
            threshold = self.threshold_for_spot_detection
        else:
            # Automatic threshold using Otsu's method
            image_flat = rna_filtered.flatten()
            threshold = threshold_otsu(image_flat)
        # Detect spots using TrackPy
        f = tp.locate(rna_filtered, diameter=self.spot_diameter, minmass=threshold,characterize=False)
        
        # Extract coordinates and intensity
        x = f['x'].values  # x-coordinate
        y = f['y'].values  # y-coordinate
        # Since the image is 2D after projection, z can be set to 0
        z = np.zeros_like(x)
        size = np.ones_like(x, dtype=int)
        clusters_and_spots = np.column_stack((z, y, x, size))
       
        return clusters_and_spots, rna_filtered, threshold


class BigFISH():
    '''
    This class is intended to detect spots in microscope images using `Big-FISH <https://github.com/fish-quant/big-fish>`_ Copyright © 2020, Arthur Imbert. The format of the image must be  [Z, Y, X, C].
    
    Parameters
    
    The description of the parameters is taken from `Big-FISH <https://github.com/fish-quant/big-fish>`_ BSD 3-Clause License. Copyright © 2020, Arthur Imbert. For a complete description of the parameters used check the `Big-FISH documentation <https://big-fish.readthedocs.io/en/stable/>`_ .
    
    image : NumPy array
        Array of images with dimensions [Z, Y, X, C]  or [Y, X, C].
    channels_spots : int
        Specific channel with spots that are used for the quantification
    voxel_size_z : int, optional
        Height of a voxel, along the z axis, in nanometers. The default is 300.
    voxel_size_yx : int, optional
        Size of a voxel on the yx plan in nanometers. The default is 150.
    cluster_radius_nm : int, optional
        Maximum distance between two samples for one to be considered as in the neighborhood of the other. Radius expressed in nanometer.
    minimum_spots_cluster : int, optional
        Number of spots in a neighborhood for a point to be considered as a core point (from which a cluster is expanded). This includes the point itself.
    show_plot : bool, optional
        If True shows a 2D maximum projection of the image and the detected spots. The default is False
    image_name : str or None.
        Name for the image with detected spots. The default is None.
    save_all_images : Bool, optional.
        If true, it shows a all planes for the plot detection. The default is False.
    display_spots_on_multiple_z_planes : Bool, optional.
        If true, it shows a spots on the plane below and above the selected plane. The default is False.
    use_log_filter_for_spot_detection : bool, optional
        Uses Big_FISH log_filter. The default is True.
    threshold_for_spot_detection: scalar or None.
        Indicates the intensity threshold used for spot detection, the default is None, and indicates that the threshold is calculated automatically.
    '''
    def __init__(self,image, channels_spots , voxel_size_z = 300,voxel_size_yx = 103, cluster_radius_nm = 350,yx_spot_size_in_px=5, z_spot_size_in_px=2, show_plot =False,image_name=None,save_all_images=False,display_spots_on_multiple_z_planes=False,use_log_filter_for_spot_detection=True,threshold_for_spot_detection=None,save_files=True):
        if len(image.shape)<4:
            image= np.expand_dims(image,axis =0)
        self.image = image
        self.channels_spots = channels_spots
        self.voxel_size_z = voxel_size_z
        self.voxel_size_yx = voxel_size_yx
        self.yx_spot_size_in_px = yx_spot_size_in_px
        self.z_spot_size_in_px = z_spot_size_in_px
        self.cluster_radius_nm = cluster_radius_nm
        self.minimum_spots_cluster = 1 #minimum_spots_cluster is set to one to detect all spots as clusters and then 
        self.show_plot = show_plot
        self.image_name=image_name
        self.save_all_images = save_all_images                                  # Displays all the z-planes
        self.display_spots_on_multiple_z_planes = display_spots_on_multiple_z_planes  # Displays the ith-z_plane and the detected spots in the planes ith-z_plane+1 and ith-z_plane
        self.use_log_filter_for_spot_detection =use_log_filter_for_spot_detection
        self.threshold_for_spot_detection=threshold_for_spot_detection
        self.decompose_dense_regions=True
        self.save_files=save_files
        self.neighborhood_min_spots_for_cluster = 2
    def detect(self):
        '''
        This method is intended to detect RNA spots in the cell and Transcription Sites (Clusters) using `Big-FISH <https://github.com/fish-quant/big-fish>`_ Copyright © 2020, Arthur Imbert.
        
        Returns
        
        clusters_and_spots : np.int64 Array with shape (nb_clusters, 5) or (nb_clusters, 4). 
            One coordinate per dimension for the centroid of the cluster (zyx or yx coordinates), the number of spots detected in the clusters.
       
        '''
        # Setting the colormap
        mpl.rc('image', cmap='viridis')
        rna=self.image[:,:,:,self.channels_spots] # [Z,Y,X]
        
        # Calculating Sigma with  the parameters for the PSF.
        spot_radius_px = detection.get_object_radius_pixel(
                        voxel_size_nm=(self.voxel_size_z, self.voxel_size_yx, self.voxel_size_yx), 
                        object_radius_nm=(self.voxel_size_z*(self.z_spot_size_in_px//2), self.voxel_size_yx*(self.yx_spot_size_in_px//2) , self.voxel_size_yx*(self.yx_spot_size_in_px//2)), ndim=3)     
        
        sigma = spot_radius_px
        spot_radius_nm =detection.get_object_radius_nm(voxel_size_nm=(self.voxel_size_z, self.voxel_size_yx, self.voxel_size_yx), 
                                                       object_radius_px=spot_radius_px, 
                                                       ndim=3)
        ## SPOT DETECTION
        if self.use_log_filter_for_spot_detection== True:
            try:
                rna_filtered = stack.log_filter(rna, sigma) # LoG filter
            except ValueError:
                print('Error during the log filter calculation, try using larger parameters values for the psf')
                rna_filtered = stack.remove_background_gaussian(rna, sigma)
        else:
            rna_filtered = stack.remove_background_gaussian(rna, sigma)
        # Automatic threshold detection.
        mask = detection.local_maximum_detection(rna_filtered, min_distance=sigma) # local maximum detection        
        if not (self.threshold_for_spot_detection is None):
            threshold = self.threshold_for_spot_detection
        else:
            threshold = detection.automated_threshold_setting(rna_filtered, mask) # thresholding
        spots, _ = detection.spots_thresholding(rna_filtered, mask, threshold, remove_duplicate=True)
        
        # Decomposing dense regions        
        if self.decompose_dense_regions == True:
            try:
                # return spots (z,y,x) dense_regions , and refrence spot
                spots_post_decomposition, _, _ = detection.decompose_dense(image=rna, 
                                                                        spots=spots, 
                                                                        voxel_size = (self.voxel_size_z, self.voxel_size_yx, self.voxel_size_yx), 
                                                                        spot_radius = spot_radius_nm,
                                                                        alpha=0.5,   # alpha impacts the number of spots per candidate region
                                                                        beta=1,      # beta impacts the number of candidate regions to decompose
                                                                        gamma=5)     # gamma the filtering step to denoise the image
            except:
                spots_post_decomposition = spots
        else:
            spots_post_decomposition = spots
        ### CLUSTER DETECTION
        # clusters_and_spots, Array with shape (nb_clusters, 5) or (nb_clusters, 4). One coordinate per dimension for the clusters centroid (zyx or yx coordinates),
        #  the number of spots detected in the clusters, and the cluster index
        # clusters, Array with shape (nb_clusters, 5) or (nb_clusters, 4). One coordinate per dimension for the clusters centroid (zyx or yx coordinates), the number of spots detected in the clusters and its index.
        
        spots_post_decomposition = spots_post_decomposition.astype(np.float64)

        clusters_and_spots_big_fish, clusters = detection.detect_clusters(spots_post_decomposition, 
                                            voxel_size=(self.voxel_size_z, self.voxel_size_yx, self.voxel_size_yx),
                                            radius= self.cluster_radius_nm,
                                            nb_min_spots = self.neighborhood_min_spots_for_cluster ) #[1] #z,yx,idx, 
        

        # select only those clusters_and_spots where last column is -1
        spots_no_clusters = clusters_and_spots_big_fish[clusters_and_spots_big_fish[:,-1]<0] # ensure to get -1 values.
        # replace the last column with 1 indicating the cluster size of 1. and this means is a spot, and not a cluster.
        spots_no_clusters[:,-1] = 1        
        # remove from clusters_and_spots the spots_no_clusters where the last column is less than 2.
        clusters_no_spots = clusters[clusters[:,-2]>1]

        clusters_no_spots = clusters_no_spots[:,0:-1]
        # concatenate the spots_no_clusters with the clusters
        clusters_and_spots = np.concatenate((spots_no_clusters, clusters_no_spots), axis=0)
        ## PLOTTING
        try:
            if self.save_files == True:
                if not(self.image_name is None):
                    path_output_elbow= str(self.image_name) +'__elbow_'+ '_ch_' + str(self.channels_spots) + '.png'
                    plot.plot_elbow(rna, 
                                    voxel_size=(self.voxel_size_z, self.voxel_size_yx,self.voxel_size_yx), 
                                    spot_radius= spot_radius_nm,
                                    path_output = path_output_elbow, show=bool(self.show_plot) )
                    if self.show_plot ==True:
                        plt.show()
                    else:
                        plt.close()
        except:
            print('not showing elbow plot')
        central_slice = rna.shape[0]//2
        if self.save_all_images:
            range_plot_images = range(0, rna.shape[0])
        else:
            range_plot_images = range(central_slice,central_slice+1)      
        for i in range_plot_images:
            if (i==central_slice) and (self.show_plot ==True):
                print('Z-Slice: ', str(i))
            image_2D = rna_filtered[i,:,:]
            if i > 1 and i<rna.shape[0]-1:
                if self.display_spots_on_multiple_z_planes == True:
                    # Displays the ith-z_plane and the detected spots in the planes ith-z_plane+1 and ith-z_plane-1
                    clusters_and_spots_z = clusters_and_spots[(clusters_and_spots[:,0]>=i-1) & (clusters_and_spots[:,0]<=i+2) ] 
                else:
                    clusters_and_spots_z = clusters_and_spots[clusters_and_spots[:,0]==i]
            else:
                clusters_and_spots_z = clusters_and_spots[clusters_and_spots[:,0]==i]
            if self.save_all_images:
                path_output= str(self.image_name) + '_ch_' + str(self.channels_spots) +'_slice_'+ str(i) +'.png'
            else:
                path_output= str(self.image_name) + '_ch_' + str(self.channels_spots) +'.png'
                
            if not(self.image_name is None) and (i==central_slice) and (self.show_plot ==True): # saving only the central slice
                show_figure_in_cli = True
            else:
                show_figure_in_cli = False                
            if not(self.image_name is None):
                if self.save_files == True:
                    try:
                        # spots_to_plot are clusters where clusters_to_plot[3]< 2
                        spots_to_plot = clusters_and_spots_z[clusters_and_spots_z[:,3]<=1]
                        clusters_to_plot = clusters_and_spots_z[clusters_and_spots_z[:,3]>1]

                        plot.plot_detection(image_2D, 
                                        spots=[spots_to_plot[:, :3], clusters_to_plot[:, :3]], 
                                        shape=["circle", "polygon"], 
                                        radius=[3, 6], 
                                        color=["orangered", "blue"],
                                        linewidth=[1, 1], 
                                        fill=[False, False], 
                                        framesize=(12, 7), 
                                        contrast=True,
                                        rescale=True,
                                        show=show_figure_in_cli,
                                        path_output = path_output)
                    except:
                        pass
                    if self.show_plot ==True:
                        plt.show()
                    else:
                        plt.close()
            del clusters_and_spots_z
        
        return clusters_and_spots, rna_filtered, threshold


class SpotDetection():
    '''
    This class is intended to detect spots in microscope images using `Big-FISH <https://github.com/fish-quant/big-fish>`_. The format of the image must be  [Z, Y, X, C].
    This class is intended to extract data from the class SpotDetection and return the data as a dataframe. 
    This class contains parameter description obtained from `Big-FISH <https://github.com/fish-quant/big-fish>`_ Copyright © 2020, Arthur Imbert.
    For a complete description of the parameters used check the `Big-FISH documentation <https://big-fish.readthedocs.io/en/stable/>`_ .
    
    Parameters
    
    image : NumPy array
        Array of images with dimensions [Z, Y, X, C] .
    channels_spots : int, or List
        List of channels with spots that are used for the quantification
    channels_cytosol : List of int
        List with integers indicating the index of channels for the cytosol segmentation. The default is None.
    channels_nucleus : list of int
        List with integers indicating the index of channels for the nucleus segmentation. The default is None. 
    cluster_radius_nm : int, optional
        Maximum distance between two samples for one to be considered as in the neighborhood of the other. Radius expressed in nanometer.
    minimum_spots_cluster : int, optional
        Number of spots in a neighborhood for a point to be considered as a core point (from which a cluster is expanded). This includes the point itself.
    masks_complete_cells : NumPy array
        Masks for every cell detected in the image are indicated by the array\'s values, where 0 indicates the background in the image, and integer numbers indicate the ith mask in the image. Array with format [Y, X].
    masks_nuclei: NumPy array
        Masks for every nucleus detected in the image are indicated by the array\'s values, where 0 indicates the background in the image, and integer numbers indicate the ith mask in the image. Array with format [Y, X].
    masks_cytosol_no_nuclei :  NumPy array
        Masks for every cytosol detected in the image are indicated by the array\'s values, where 0 indicates the background in the image, and integer numbers indicate the ith mask in the image. Array with format [Y, X].
    dataframe : Pandas Dataframe 
        Pandas dataframe with the following columns. image_id, cell_id, spot_id, nuc_loc_y, nuc_loc_x, cyto_loc_y, cyto_loc_x, nuc_area_px, cyto_area_px, cell_area_px, z, y, x, is_nuc, is_cluster, cluster_size, spot_type, is_cell_fragmented. The default is None.
    image_counter : int, optional
        counter for the number of images in the folder. The default is zero.
    list_voxels : List of tupples or None
        list with a tuple with two elements (voxel_size_z,voxel_size_yx ) for each channel.
        voxel_size_z is the height of a voxel, along the z axis, in nanometers. The default is 300.
        voxel_size_yx is the size of a voxel on the yx plan in nanometers. The default is 150.
    show_plot : bool, optional
        If True, it shows a 2D maximum projection of the image and the detected spots. The default is False.
    image_name : str or None.
        Name for the image with detected spots. The default is None.
    save_all_images : Bool, optional.
        If true, it shows a all planes for the plot detection. The default is False.
    display_spots_on_multiple_z_planes : Bool, optional.
        If true, it shows a spots on the plane below and above the selected plane. The default is False.
    use_log_filter_for_spot_detection : bool, optional
        Uses Big_FISH log_filter. The default is True.
    threshold_for_spot_detection: scalar or None.
        Indicates the intensity threshold used for spot detection, the default is None, and indicates that the threshold is calculated automatically.
    use_trackpy : bool, optional
        If True, it uses trackpy for spot detection. The default is False.
    calculate_intensity : book, optional.
        If True, calculate the intensity of the spots.
    use_fixed_size_for_intensity_calculation : bool, optional
        If True, it uses a fixed size for the intensity calculation. The default is True.

    
    '''
    def __init__(self,image,  channels_spots ,channels_cytosol,channels_nucleus, cluster_radius_nm=500,masks_complete_cells = None, masks_nuclei  = None, masks_cytosol_no_nuclei = None,
                dataframe=None, image_counter=0, list_voxels=[500,160], show_plot=True,image_name=None,save_all_images=True,display_spots_on_multiple_z_planes=False,
                use_log_filter_for_spot_detection=True,threshold_for_spot_detection=None,save_files=True,yx_spot_size_in_px=None, z_spot_size_in_px=None, 
                use_trackpy=False,use_maximum_projection=False,calculate_intensity=True,use_fixed_size_for_intensity_calculation=True):
        if len(image.shape)<4:
            image= np.expand_dims(image,axis =0)
        # calculate the maximum projection of the axis 0 keeping the same dimensions.
        if use_maximum_projection == True:
            image = np.max(image, axis=0, keepdims=True)
        self.image = image
        self.number_color_channels = image.shape[-1]
        self.channels_cytosol=channels_cytosol
        self.channels_nucleus=channels_nucleus
        if not (masks_complete_cells is None):
            self.list_masks_complete_cells = Utilities().separate_masks(masks_complete_cells)
        elif (masks_complete_cells is None) and not(masks_nuclei is None):
            self.list_masks_complete_cells = Utilities().separate_masks(masks_nuclei)            
        if not (masks_nuclei is None):    
            self.list_masks_nuclei = Utilities().separate_masks(masks_nuclei)
        else:
            self.list_masks_nuclei = None
        if not (masks_complete_cells is None) and not (masks_nuclei is None):
            self.list_masks_cytosol_no_nuclei = Utilities().separate_masks(masks_cytosol_no_nuclei)
        else:
            self.list_masks_cytosol_no_nuclei = None
        
        self.dataframe = dataframe
        self.image_counter = image_counter
        self.show_plot = show_plot
        self.list_voxels = list_voxels  
        # converting spot channels to a list
        self.channels_spots = channels_spots
        if not (type(channels_spots) is list):
            self.list_channels_spots = [channels_spots]
        else:
            self.list_channels_spots = channels_spots
        # spot properties
        self.cluster_radius_nm = cluster_radius_nm
        self.yx_spot_size_in_px =yx_spot_size_in_px
        self.z_spot_size_in_px = z_spot_size_in_px
        self.image_name = image_name
        self.save_all_images = save_all_images                                  # Displays all the z-planes
        self.display_spots_on_multiple_z_planes = display_spots_on_multiple_z_planes  # Displays the ith-z_plane and the detected spots in the planes ith-z_plane+1 and ith-z_plane
        self.use_log_filter_for_spot_detection =use_log_filter_for_spot_detection
        if not isinstance(threshold_for_spot_detection, list):
            threshold_for_spot_detection=[threshold_for_spot_detection]
        self.threshold_for_spot_detection=threshold_for_spot_detection
        self.save_files = save_files
        self.MINIMUM_SPOT_SIZE_IN_PX = 3
        self.use_maximum_projection = use_maximum_projection
        self.spot_radius_px = detection.get_object_radius_pixel(
                voxel_size_nm=(list_voxels[0], list_voxels[1], list_voxels[1]),
                object_radius_nm=(list_voxels[0]*(z_spot_size_in_px//2), list_voxels[1]*(yx_spot_size_in_px//2), list_voxels[1]*(yx_spot_size_in_px//2)), ndim=3)
        self.calculate_intensity =calculate_intensity
        self.use_fixed_size_for_intensity_calculation = use_fixed_size_for_intensity_calculation

    def get_dataframe(self):
        list_images = []
        list_thresholds_spot_detection = []
        for i in range(0,len(self.list_channels_spots)):
            if (i ==0):
                df_detected_spots = self.dataframe 
                reset_cell_counter = False
            if self.use_maximum_projection == True:
                # ensure that the image is a 2D image
                print('Using trackpy for spot detection')
                clusters_and_spots, image_filtered, threshold = TrackPyDetection(self.image, 
                                                                                self.list_channels_spots[i],
                                                                                voxel_size_yx = self.list_voxels[1], 
                                                                                yx_spot_size_in_px=self.yx_spot_size_in_px, 
                                                                                show_plot=self.show_plot,
                                                                                image_name=self.image_name,
                                                                                save_all_images=self.save_all_images,
                                                                                display_spots_on_multiple_z_planes=self.display_spots_on_multiple_z_planes, 
                                                                                spot_diameter=self.yx_spot_size_in_px,
                                                                                threshold_for_spot_detection=self.threshold_for_spot_detection[i],save_files=self.save_files).detect()
                
            else:
                print('Using BigFISH for spot detection')
                clusters_and_spots, image_filtered, threshold = BigFISH(self.image, 
                                                                        self.list_channels_spots[i], 
                                                                        voxel_size_z = self.list_voxels[0],
                                                                        voxel_size_yx = self.list_voxels[1],
                                                                        cluster_radius_nm=self.cluster_radius_nm, 
                                                                        show_plot=self.show_plot,image_name=self.image_name,
                                                                        yx_spot_size_in_px=self.yx_spot_size_in_px, 
                                                                        z_spot_size_in_px=self.z_spot_size_in_px, 
                                                                        save_all_images=self.save_all_images,
                                                                        display_spots_on_multiple_z_planes=self.display_spots_on_multiple_z_planes,
                                                                        use_log_filter_for_spot_detection =self.use_log_filter_for_spot_detection,
                                                                        threshold_for_spot_detection=self.threshold_for_spot_detection[i],save_files=self.save_files).detect()
                
                

            list_thresholds_spot_detection.append(threshold)
            # converting the psf to pixles
            spot_diameter_for_intensity_px = int(np.max((self.spot_radius_px[1]*2, self.MINIMUM_SPOT_SIZE_IN_PX)))
            
            if self.calculate_intensity:
                df_detected_spots = DataProcessing(clusters_and_spots, self.image, self.list_masks_complete_cells, self.list_masks_nuclei, self.list_masks_cytosol_no_nuclei, self.channels_cytosol,self.channels_nucleus,
                                            yx_spot_size_in_px=spot_diameter_for_intensity_px, dataframe =df_detected_spots,reset_cell_counter=reset_cell_counter,image_counter = self.image_counter ,spot_type=i,use_fixed_size_for_intensity_calculation=self.use_fixed_size_for_intensity_calculation,
                                            number_color_channels=self.number_color_channels, use_maximum_projection=self.use_maximum_projection ).get_dataframe()
            else:
                # If intensity calculation is skipped, only append spot locations
                df_detected_spots = pd.DataFrame(clusters_and_spots, columns=['z', 'y', 'x', 'cluster_size'])
                df_detected_spots['image_id'] = self.image_counter
                df_detected_spots['spot_type'] = i
                # remove spots that are not inside the mask
                
                mask_selected = (self.list_masks_complete_cells[0] >0).astype(int)
                df_in_mask = Utilities().spots_in_mask(df_detected_spots, mask_selected)
                df_detected_spots = df_in_mask[df_in_mask['In Mask'] == True]

            # reset counter for image and cell number
            reset_cell_counter = True
            list_images.append(image_filtered)
            # ensure that no spots are detected in the origin of the image x and y = 0
            df_detected_spots = df_detected_spots[(df_detected_spots['y']!=0) & (df_detected_spots['x']!=0)]

        return df_detected_spots, list_images, list_thresholds_spot_detection



class ParticleTracking:
    '''
    This class detects particles in a 3D time‐lapse image and optionally links them into trajectories.
    The image must be provided with dimensions [T, Z, Y, X, C].

    Parameters
    ----------
    image : ndarray
        The 5D image array with dimensions [T, Z, Y, X, C].
    channels_spots : list
        List of channel indices that contain spot signals.
    list_voxels : list
        List of voxel sizes for each dimension [z, y/x].
    channels_cytosol : list
        List of channel indices that contain cytosol signals.
    channels_nucleus : list
        List of channel indices that contain nucleus signals.
    remove_clusters : bool, optional
        Whether to remove clusters. Default is False.
    maximum_spots_cluster : int, optional
        Maximum number of spots in a cluster. Default is None.
    min_length_trajectory : int, optional
        Minimum trajectory length to be considered valid. Default is 10.
    threshold_for_spot_detection : int, optional
        Threshold value for spot detection. Default is 100.
    masks : ndarray, optional
        Array of masks indicating regions of interest. Default is None.
    memory : int, optional
        Number of frames a particle can disappear and reappear. Default is 0.
    yx_spot_size_in_px : int, optional
        Spot size in pixels. Default is 5.
    z_spot_size_in_px : int, optional
        Spot size in z. Default is 2.
    cluster_radius_nm : float or None, optional
        Cluster radius in nm; if None, defaults to (voxel_yx * 4).
    link_particles : bool, optional
        Whether to link particles into trajectories. Default is True.
    use_trackpy : bool, optional
        Whether to use trackpy for detection/linking. Default is False.
    use_maximum_projection : bool, optional
        Whether to project the image along Z for detection/linking. Default is False.
    separate_clusters_and_spots : bool, optional
        Whether to separate clusters and spots for linking. Default is False.
    maximum_range_search_pixels : int, optional
        Maximum search range (in pixels) for linking. Default is 10.
    link_using_3d_coordinates : bool, optional
        Whether to link particles using 3D coordinates. Default is False.
    neighbor_strategy : str, optional
        Strategy for neighbor search. Default is 'KDTree' (other option: 'BTree').
    generate_random_particles : bool, optional
        If True, generate random trajectories (i.e. random spot locations that remain constant over time). Default is False.
    number_of_random_particles_trajectories : int or None, optional
        Number of random trajectories to generate. Default is None.
    step_size_in_sec : float, optional
        Time step size in seconds for random trajectories. Default is 1.0.
    '''
    def __init__(self, image, channels_spots, list_voxels, channels_cytosol, channels_nucleus,
                 
                 remove_clusters=False, maximum_spots_cluster=None, min_length_trajectory=10,
                 threshold_for_spot_detection=100, masks=None, memory=0, yx_spot_size_in_px=5, z_spot_size_in_px=2,
                 cluster_radius_nm=None, link_particles=True, use_trackpy=False,
                 use_fixed_size_for_intensity_calculation=True, number_cores=None,
                 use_maximum_projection=False, separate_clusters_and_spots=False,
                 maximum_range_search_pixels=10, link_using_3d_coordinates=False,
                 neighbor_strategy='KDTree', generate_random_particles=False,
                 number_of_random_particles_trajectories=None,step_size_in_sec=1.0):

        if len(image.shape) != 5:
            raise ValueError('The image must have 5 dimensions [T, Z, Y, X, C].')

        self.image = image
        # Ensure channels_spots is a list.
        if not isinstance(channels_spots, list):
            self.channels_spots = [channels_spots]
        else:
            self.channels_spots = channels_spots

        self.masks = masks if masks is not None else np.ones(image[0].shape[:3], dtype=bool)
        self.channels_cytosol = channels_cytosol
        self.channels_nucleus = channels_nucleus
        self.threshold_for_spot_detection = threshold_for_spot_detection
        self.number_time_points = image.shape[0]
        self.number_color_channels = image.shape[-1]
        self.min_length_trajectory = min_length_trajectory
        self.memory = memory
        if number_cores is None:
            self.NUMBER_OF_CORES = cpu_count()
        else:
            self.NUMBER_OF_CORES = number_cores

        # Spot properties.
        self.list_voxels = list_voxels
        self.yx_spot_size_in_px = yx_spot_size_in_px
        self.z_spot_size_in_px = z_spot_size_in_px

        # Cluster properties.
        if cluster_radius_nm is None:
            self.cluster_radius_nm = int(list_voxels[1] * 4)
        else:
            self.cluster_radius_nm = cluster_radius_nm
        self.remove_clusters = remove_clusters
        self.maximum_spots_cluster = maximum_spots_cluster
        self.separate_clusters_and_spots = separate_clusters_and_spots

        # Compute spot radius in pixels.
        self.spot_radius_px = detection.get_object_radius_pixel(
            voxel_size_nm=(list_voxels[0], list_voxels[1], list_voxels[1]),
            object_radius_nm=(list_voxels[0]*(z_spot_size_in_px//2),
                              list_voxels[1]*(yx_spot_size_in_px//2),
                              list_voxels[1]*(yx_spot_size_in_px//2)),
            ndim=3
        )
        self.link_particles = link_particles
        self.use_trackpy = use_trackpy
        self.maximum_range_search_pixels = maximum_range_search_pixels
        self.use_maximum_projection = use_maximum_projection
        self.use_fixed_size_for_intensity_calculation = use_fixed_size_for_intensity_calculation
        self.link_using_3d_coordinates = link_using_3d_coordinates
        self.neighbor_strategy = neighbor_strategy

        # Random control parameters.
        self.generate_random_particles = generate_random_particles
        if self.generate_random_particles and (number_of_random_particles_trajectories is None):
            print("Warning: Number of random particles not specified; defaulting to 50.")
            number_of_random_particles_trajectories = 50
        self.number_of_random_particles_trajectories = number_of_random_particles_trajectories
        self.step_size_in_sec = step_size_in_sec

    def run(self):
        # --- RANDOM MODE: Process frame-by-frame for random spot trajectories ---
        if self.generate_random_particles:
            T = self.number_time_points
            rows = []
            # Build rows in the same format as TrackPyDetection output:
            # Each row: [image_id, cell_id, spot_id, nuc_loc_y, nuc_loc_x, cyto_loc_y, cyto_loc_x,
            #            nuc_area_px, cyto_area_px, cell_area_px, z, y, x, cluster_size, spot_type, is_cell_fragmented, frame]
            expected_columns = ['image_id', 'cell_id', 'spot_id', 'nuc_loc_y', 'nuc_loc_x',
                                'cyto_loc_y', 'cyto_loc_x', 'nuc_area_px', 'cyto_area_px', 'cell_area_px',
                                'z', 'y', 'x', 'cluster_size', 'spot_type', 'is_cell_fragmented', 'frame']
            if self.use_maximum_projection:
                # For 2D mode, use the 2D mask directly.
                mask_2d = self.masks if self.masks.ndim == 2 else self.masks[0]
                indices = np.argwhere(mask_2d)
                n_rand = self.number_of_random_particles_trajectories
                if len(indices) == 0:
                    raise ValueError("The mask is empty; cannot generate random locations.")
                replace = n_rand > len(indices)
                chosen = indices[np.random.choice(len(indices), size=n_rand, replace=replace)]
                spot_id = 0
                for i in range(n_rand):
                    y, x = chosen[i]
                    for t in range(T):
                        row = {
                            'image_id': 0,
                            'cell_id': 0,
                            'spot_id': spot_id,
                            'nuc_loc_y': np.nan,
                            'nuc_loc_x': np.nan,
                            'cyto_loc_y': np.nan,
                            'cyto_loc_x': np.nan,
                            'nuc_area_px': np.nan,
                            'cyto_area_px': np.nan,
                            'cell_area_px': np.nan,
                            'z': 0,  # In 2D mode, z is set to 0.
                            'y': y,
                            'x': x,
                            'cluster_size': 1,
                            'spot_type': 0,
                            'is_cell_fragmented': 0,
                            'frame': t
                        }
                        rows.append(row)
                    spot_id += 1
            else:
                # In non-max-projection mode, we expect a 3D mask.
                # If the provided mask is 2D, replicate it to create a 3D mask.
                if self.masks.ndim == 2:
                    mask_3d = np.expand_dims(self.masks, axis=0)
                elif self.masks.ndim >= 3:
                    mask_3d = self.masks if self.masks.ndim == 3 else self.masks[0]
                else:
                    raise ValueError("Invalid mask dimensions.")
                indices = np.argwhere(mask_3d)
                n_rand = self.number_of_random_particles_trajectories
                if len(indices) == 0:
                    raise ValueError("The mask is empty; cannot generate random locations.")
                replace = n_rand > len(indices)
                chosen = indices[np.random.choice(len(indices), size=n_rand, replace=replace)]
                spot_id = 0
                for i in range(n_rand):
                    # Here, we expect each chosen element to have three values: (z, y, x)
                    z, y, x = chosen[i]
                    for t in range(T):
                        row = {
                            'image_id': 0,
                            'cell_id': 0,
                            'spot_id': spot_id,
                            'nuc_loc_y': np.nan,
                            'nuc_loc_x': np.nan,
                            'cyto_loc_y': np.nan,
                            'cyto_loc_x': np.nan,
                            'nuc_area_px': np.nan,
                            'cyto_area_px': np.nan,
                            'cell_area_px': np.nan,
                            'z': z,
                            'y': y,
                            'x': x,
                            'cluster_size': 1,
                            'spot_type': 0,
                            'is_cell_fragmented': 0,
                            'frame': t
                        }
                        rows.append(row)
                    spot_id += 1
            df_random = pd.DataFrame(rows, columns=expected_columns)
            # Prepare image and mask for intensity extraction.
            if self.use_maximum_projection:
                image_proj = np.max(self.image, axis=1, keepdims=False)  # [T, Y, X, C]
                mask_proj = np.max(self.masks, axis=0) if self.masks.ndim > 2 else self.masks
            else:
                image_proj = self.image
                mask_proj = self.masks

            # Process each frame in parallel.
            def process_frame(t):
                df_frame = df_random[df_random['frame'] == t].copy()
                # DataProcessing expects an array with 4 columns: [z, y, x, cluster_size]
                clusters_array = df_frame[['z', 'y', 'x', 'cluster_size']].to_numpy()
                dp = DataProcessing(
                    clusters_and_spots=clusters_array,
                    image=image_proj[t],      # t-th frame; shape [Y, X, C]
                    masks_complete_cells=mask_proj,
                    masks_nuclei=None,
                    masks_cytosol_no_nuclei=None,
                    channels_cytosol=self.channels_cytosol,
                    channels_nucleus=self.channels_nucleus,
                    yx_spot_size_in_px=self.yx_spot_size_in_px,
                    spot_type=0,
                    dataframe=None,
                    reset_cell_counter=True,
                    image_counter=0,
                    number_color_channels=self.number_color_channels,
                    use_maximum_projection=self.use_maximum_projection,
                    use_fixed_size_for_intensity_calculation=self.use_fixed_size_for_intensity_calculation
                )
                df_processed = dp.get_dataframe()
                df_processed['frame'] = t
                return df_processed

            processed_frames = Parallel(n_jobs=self.NUMBER_OF_CORES)(
                delayed(process_frame)(t) for t in range(T)
            )
            df_complete = pd.concat(processed_frames, ignore_index=True)
            # Create a 'particle' column based on the original 'spot_id'
            if 'spot_id' in df_complete.columns:
                df_complete['particle'] = pd.factorize(df_complete['spot_id'])[0]
            
            # create the 'time' column
            df_complete['time'] = df_complete['frame'] * self.step_size_in_sec
            return [df_complete], self.image

        # --- NORMAL MODE (unchanged) ---
        else:
            def process_time_point(i):
                dataframe, imgs, _ = SpotDetection(
                    self.image[i],
                    channels_spots=self.channels_spots,
                    channels_cytosol=self.channels_cytosol,
                    channels_nucleus=self.channels_nucleus,
                    masks_complete_cells=self.masks,
                    list_voxels=self.list_voxels,
                    show_plot=False,
                    save_files=False,
                    cluster_radius_nm=self.cluster_radius_nm,
                    threshold_for_spot_detection=self.threshold_for_spot_detection,
                    yx_spot_size_in_px=self.yx_spot_size_in_px,
                    z_spot_size_in_px=self.z_spot_size_in_px,
                    use_trackpy=self.use_trackpy,
                    use_maximum_projection=self.use_maximum_projection,
                    use_fixed_size_for_intensity_calculation=self.use_fixed_size_for_intensity_calculation
                ).get_dataframe()
                filtered_images = []
                for ch in range(self.number_color_channels):
                    filtered_image_ch = stack.log_filter(
                        self.image[i, :, :, :, ch], sigma=self.spot_radius_px)
                    filtered_images.append(filtered_image_ch)
                dataframe['frame'] = i
                return dataframe, filtered_images

            results = Parallel(n_jobs=self.NUMBER_OF_CORES)(
                delayed(process_time_point)(i) for i in range(self.number_time_points)
            )
            dataframes, list_filtered_images = zip(*results)
            df_all = pd.concat(dataframes, ignore_index=True)
            df_all['time'] = df_all['frame'] * self.step_size_in_sec
            filtered_image_stack = np.zeros_like(self.image)
            for i in range(self.number_time_points):
                for ch in range(self.number_color_channels):
                    filtered_image_stack[i, :, :, :, ch] = list_filtered_images[i][ch]

            # def particle_linking(df, search_range=np.linspace(0.5, 5, 5),
            #                     min_length_trajectory=10, memory=0, pos_columns=['x', 'y', 'z']):
            #     list_df = []
            #     quality_metrics = []
            #     for search_distance in search_range:
            #         try:
            #             linked = tp.link(df, search_distance, pos_columns=pos_columns,
            #                             memory=memory, neighbor_strategy=self.neighbor_strategy)
            #             filtered = tp.filter_stubs(linked, threshold=min_length_trajectory)
            #             if 'particle' in filtered.columns:
            #                 num_trajectories = len(filtered['particle'].unique())
            #                 avg_length = filtered.groupby('particle').size().mean() if num_trajectories > 0 else 0
            #                 metric = num_trajectories * avg_length
            #             else:
            #                 metric = 0
            #             list_df.append(filtered)
            #             quality_metrics.append(metric)
            #         except Exception:
            #             list_df.append(pd.DataFrame())
            #             quality_metrics.append(0)
            #     return list_df[np.argmax(quality_metrics)]

            # def linking_2D(df, search_distance=10, min_length_trajectory=10, memory=0, pos_columns=['x', 'y']):
            #     linked = tp.link(df, search_distance, pos_columns=pos_columns,
            #                     memory=memory, neighbor_strategy=self.neighbor_strategy)
            #     return tp.filter_stubs(linked, threshold=min_length_trajectory)
            

            def particle_linking(df, search_range=np.linspace(0.5, 5, 5),
                    min_length_trajectory=10, memory=0, pos_columns=['x', 'y', 'z']):
                list_df = []
                quality_metrics = []
                for search_distance in search_range:
                    try:
                        linked = tp.link(df, search_distance, pos_columns=pos_columns,
                                        memory=memory, neighbor_strategy=self.neighbor_strategy,
                                        adaptive_stop=0.5, adaptive_step=0.95, )
                    except Exception:
                        try:
                            linked = tp.link(df, search_distance, pos_columns=pos_columns,
                                            memory=0, neighbor_strategy=self.neighbor_strategy,
                                        adaptive_stop=0.5, adaptive_step=0.95, )
                        except Exception:
                            list_df.append(pd.DataFrame())
                            quality_metrics.append(0)
                            continue
                    filtered = tp.filter_stubs(linked, threshold=min_length_trajectory)
                    if 'particle' in filtered.columns:
                        num_trajectories = len(filtered['particle'].unique())
                        avg_length = (filtered.groupby('particle').size().mean()
                                    if num_trajectories > 0 else 0)
                        metric = num_trajectories * avg_length
                    else:
                        metric = 0
                    list_df.append(filtered)
                    quality_metrics.append(metric)
                return list_df[np.argmax(quality_metrics)]


            def linking_2D(df, search_distance=10, min_length_trajectory=10,
                        memory=0, pos_columns=['x', 'y']):
                try:
                    linked = tp.link(df, search_distance, pos_columns=pos_columns,
                                    memory=memory, neighbor_strategy=self.neighbor_strategy,
                                    adaptive_stop=0.5, adaptive_step=0.95)
                except Exception:
                    try:
                        linked = tp.link(df, search_distance, pos_columns=pos_columns,
                                        memory=0, neighbor_strategy=self.neighbor_strategy,
                                        adaptive_stop=0.5, adaptive_step=0.95)
                    except Exception:
                        return pd.DataFrame()
                return tp.filter_stubs(linked, threshold=min_length_trajectory)


        # def particle_linking(df,
        #                     search_range: np.ndarray = np.linspace(0.5, 5, 5),
        #                     min_length_trajectory: int = 10,
        #                     memory: int = 0,
        #                     pos_columns: list = ['x', 'y', 'z']):
        #     """
        #     Try linking over a sweep of search distances, pick the result
        #     with the highest (num_trajectories * avg_length) metric.
        #     Allows a 1-frame gap if memory > 0, else falls back to memory=0.
        #     """
        #     list_df = []
        #     quality_metrics = []

        #     for search_distance in search_range:
        #         # first attempt with user memory
        #         try:
        #             linked = tp.link_df(
        #                 df,
        #                 search_distance,
        #                 memory=memory,
        #                 pos_columns=pos_columns,
        #                 neighbor_strategy=self.neighbor_strategy,
        #                 adaptive_stop=0.5,
        #                 adaptive_step=0.95,
        #             )
        #         except Exception:
        #             # fallback: no memory
        #             try:
        #                 linked = tp.link_df(
        #                     df,
        #                     search_distance,
        #                     memory=0,
        #                     pos_columns=pos_columns,
        #                     neighbor_strategy=self.neighbor_strategy,
        #                     adaptive_stop=0.5,
        #                     adaptive_step=0.95,
        #                 )
        #             except Exception:
        #                 list_df.append(pd.DataFrame())
        #                 quality_metrics.append(0)
        #                 continue

        #         # filter out short tracks
        #         filtered = tp.filter_stubs(linked, threshold=min_length_trajectory)

        #         # compute quality metric
        #         if 'particle' in filtered:
        #             num = filtered['particle'].nunique()
        #             avg_len = filtered.groupby('particle').size().mean() if num > 0 else 0
        #             metric = num * avg_len
        #         else:
        #             metric = 0

        #         list_df.append(filtered)
        #         quality_metrics.append(metric)

        #     # return the best result
        #     best_idx = int(np.argmax(quality_metrics))
        #     return list_df[best_idx]


        # def linking_2D(df,
        #             search_distance: float = 10,
        #             min_length_trajectory: int = 10,
        #             memory: int = 0,
        #             pos_columns: list = ['x', 'y']):
        #     """
        #     Single‐distance 2D linking with fallback to memory=0.
        #     """
        #     try:
        #         linked = tp.link_df(
        #             df,
        #             search_distance,
        #             memory=memory,
        #             pos_columns=pos_columns,
        #             neighbor_strategy=self.neighbor_strategy,
        #             adaptive_stop=0.5,
        #             adaptive_step=0.95,
        #         )
        #     except Exception:
        #         try:
        #             linked = tp.link_df(
        #                 df,
        #                 search_distance,
        #                 memory=0,
        #                 pos_columns=pos_columns,
        #                 neighbor_strategy=self.neighbor_strategy,
        #                 adaptive_stop=0.5,
        #                 adaptive_step=0.95,
        #             )
        #         except Exception:
        #             return pd.DataFrame()

        #     return tp.filter_stubs(linked, threshold=min_length_trajectory)

            list_dfs_traj = []
            counter = 0
            for spot_type in range(len(self.channels_spots)):
                df_spot = df_all[df_all['spot_type'] == spot_type]
                number_masks = np.max(self.masks)
                if number_masks > 0:
                    mask_dfs = []
                    for index_mask in range(1, int(number_masks) + 1):
                        mask_sel = (self.masks == index_mask).astype(int)
                        df_in_mask = Utilities().spots_in_mask(df_spot, mask_sel)
                        df_particles = df_in_mask[df_in_mask['In Mask'] == True]
                        if self.link_particles:
                            if self.use_maximum_projection:
                                linked_df = linking_2D(
                                    df=df_particles,
                                    search_distance=self.maximum_range_search_pixels,
                                    min_length_trajectory=self.min_length_trajectory,
                                    memory=self.memory,
                                    pos_columns=['x', 'y']
                                )
                            else:
                                if self.link_using_3d_coordinates:
                                    linked_df = particle_linking(
                                        df=df_particles,
                                        search_range=np.linspace(3, self.maximum_range_search_pixels, 3),
                                        min_length_trajectory=self.min_length_trajectory,
                                        memory=self.memory,
                                        pos_columns=['x', 'y', 'z']
                                    )
                                else:
                                    linked_df = linking_2D(
                                        df=df_particles,
                                        search_distance=self.maximum_range_search_pixels,
                                        min_length_trajectory=self.min_length_trajectory,
                                        memory=self.memory,
                                        pos_columns=['x', 'y']
                                    )
                            if not linked_df.empty and 'particle' in linked_df.columns:
                                linked_df['particle'] = pd.factorize(linked_df['particle'])[0]
                            else:
                                linked_df = pd.DataFrame()
                        else:
                            if not df_particles.empty:
                                linked_df = df_particles.copy()
                                num_p = linked_df.shape[0]
                                linked_df['particle'] = range(counter, counter + num_p)
                                counter += num_p
                            else:
                                linked_df = pd.DataFrame()
                        if not linked_df.empty:
                            linked_df['cell_id'] = index_mask - 1
                            if 'spot_id' in linked_df.columns:
                                linked_df = linked_df.drop(columns=['spot_id', 'In Mask'], errors='ignore')
                        mask_dfs.append(linked_df)
                    if mask_dfs:
                        df_traj = pd.concat(mask_dfs, ignore_index=True)
                        #
                    else:
                        df_traj = pd.DataFrame()
                    if (not df_traj.empty) and (self.maximum_spots_cluster is not None):
                        if 'particle' in df_traj.columns and 'cluster_size' in df_traj.columns:
                            avg_cluster = df_traj.groupby('particle')['cluster_size'].mean()
                            keep_particles = avg_cluster[avg_cluster < self.maximum_spots_cluster].index
                            df_traj = df_traj[df_traj['particle'].isin(keep_particles)].copy()
                            df_traj['particle'] = pd.factorize(df_traj['particle'])[0]
                            df_traj.reset_index(drop=True, inplace=True)
                    list_dfs_traj.append(df_traj)
                else:
                    list_dfs_traj.append(pd.DataFrame())
            return list_dfs_traj, filtered_image_stack

class DataProcessing():
    '''
    This class is intended to extract data from the class SpotDetection and return the data as a dataframe. 
    This class contains parameter descriptions obtained from `Big-FISH <https://github.com/fish-quant/big-fish>`_ Copyright © 2020, Arthur Imbert. For a complete description of the parameters used check the `Big-FISH documentation <https://big-fish.readthedocs.io/en/stable/>`_ .
    
    Parameters
    
    clusters_and_spots : np.int64 with shape (nb_spots, 4) or (nb_spots, 3).
        Coordinates of the detected spots . One coordinate per dimension (zyx or yx coordinates) plus the number of spots detected in the cluster. If no cluster was assigned, the value is -1.
    image : NumPy array
        Array of images with dimensions [Z, Y, X, C] .
    masks_complete_cells : List of NumPy arrays or a single NumPy array
        Masks for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
    masks_nuclei: List of NumPy arrays or a single NumPy array
        Masks for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
    masks_cytosol_no_nuclei : List of NumPy arrays or a single NumPy array
        Masks for every cell detected in the image. The list contains the mask arrays consisting of one or multiple Numpy arrays with format [Y, X].
    channels_cytosol : List of int
        List with integers indicating the index of channels for the cytosol segmentation. The default is None.
    channels_nucleus : list of int
        List with integers indicating the index of channels for the nucleus segmentation. The default is None. 
    yx_spot_size_in_px : int
        Size of the spot in pixels.
    spot_type : int, optional
        A label indicating the spot type, this counter starts at zero, increasing with the number of channels containing spots. The default is zero.
    dataframe : Pandas dataframe or None.
        Pandas dataframe with the following columns. image_id, cell_id, spot_id, nuc_loc_y, nuc_loc_x, cyto_loc_y, cyto_loc_x, nuc_area_px, cyto_area_px, cell_area_px, z, y, x, is_nuc, is_cluster, cluster_size, spot_type, is_cell_fragmented. The default is None.
    reset_cell_counter : bool
        This number is used to reset the counter of the number of cells. The default is False.
    image_counter : int, optional
        counter for the number of images in the folder. The default is zero.
    use_maximum_projection : bool, optional
        If True, it uses the maximum projection of the image will be used to calculate the intensity. The default is False.
    use_fixed_size_for_intensity_calculation : bool, optional
        If True, it uses the fixed size for the intensity calculation. The default is True and uses the yx_spot_size_in_px. Else it uses the yx_spot_size_in_px times the cluster size.

    '''
    def __init__(self, clusters_and_spots, image, masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei,  channels_cytosol, channels_nucleus, yx_spot_size_in_px,  spot_type=0, dataframe =None,reset_cell_counter=False,image_counter=0,number_color_channels=None,use_maximum_projection=False,use_fixed_size_for_intensity_calculation=True):
        #self.spotDetectionCSV=spotDetectionCSV 
        self.clusters_and_spots=clusters_and_spots
        self.channels_cytosol=channels_cytosol
        self.channels_nucleus=channels_nucleus
        self.number_color_channels=number_color_channels
        self.yx_spot_size_in_px =yx_spot_size_in_px
        if len(image.shape)<4:
            image= np.expand_dims(image,axis =0)
        #if use_maximum_projection == True:
        #    image = np.max(image, axis=0, keepdims=True)
        self.image = image
        if isinstance(masks_complete_cells, list) or (masks_complete_cells is None):
            self.masks_complete_cells=masks_complete_cells
        else:
            self.masks_complete_cells=Utilities().separate_masks(masks_complete_cells)
        if isinstance(masks_nuclei, list) or (masks_nuclei is None):
            self.masks_nuclei=masks_nuclei
        else:
            self.masks_nuclei=Utilities().separate_masks(masks_nuclei)  
        if isinstance(masks_cytosol_no_nuclei, list) or (masks_cytosol_no_nuclei is None):
            self.masks_cytosol_no_nuclei=masks_cytosol_no_nuclei
        else:
            self.masks_cytosol_no_nuclei= Utilities().separate_masks(masks_cytosol_no_nuclei)
        self.dataframe=dataframe
        self.spot_type = spot_type
        self.reset_cell_counter = reset_cell_counter
        self.image_counter = image_counter
        self.use_maximum_projection = use_maximum_projection
        self.use_fixed_size_for_intensity_calculation = use_fixed_size_for_intensity_calculation
        
        # This number represent the number of columns that doesnt change with the number of color channels in the image
        self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME = 18
    def get_dataframe(self):
        '''
        This method extracts data from the class SpotDetection and returns the data as a dataframe.
        
        Returns
        
        dataframe : Pandas dataframe
            Pandas dataframe with the following columns. image_id, cell_id, spot_id, nuc_loc_y, nuc_loc_x, cyto_loc_y, cyto_loc_x, nuc_area_px, cyto_area_px, cell_area_px, z, y, x, is_nuc, is_cluster, cluster_size, spot_type, is_cell_fragmented.
        '''
        def mask_selector(mask,calculate_centroid= True):
            mask_area = np.count_nonzero(mask)
            if calculate_centroid == True:
                centroid_y,centroid_x = ndimage.measurements.center_of_mass(mask)
            else:
                centroid_y,centroid_x = 0,0
            return  mask_area, int(centroid_y), int(centroid_x)
        def replace_border_px_zeros(mask,number_of_pixels_to_replace_in_border=7):
            # Ensure mask is writable
            if not mask.flags.writeable:
                mask = mask.copy()
                mask.setflags(write=1)
            mask[:number_of_pixels_to_replace_in_border, :] = 0
            mask[-number_of_pixels_to_replace_in_border:, :] = 0
            mask[:, :number_of_pixels_to_replace_in_border] = 0
            mask[:, -number_of_pixels_to_replace_in_border:] = 0
            return mask
        
        def separate_clusters_and_spots_in_mask(clusters_and_spots,mask):
            mask = replace_border_px_zeros(mask)
            coords = np.array([clusters_and_spots[:,1], clusters_and_spots[:,2]]).T # These are the points detected by trackpy
            coords_int = np.round(coords).astype(int)  # or np.floor, depends
            values_at_coords = mask[tuple(coords_int.T)] # If 1 the value is in the mask
            clusters_in_mask = clusters_and_spots[values_at_coords==1]  # [Z,Y,X,size,idx_foci]
            spots = clusters_in_mask[clusters_in_mask[:,3]<=1]  # [Z,Y,X,size,idx_foci]
            clusters = clusters_in_mask[clusters_in_mask[:,3]>1] # [Z,Y,X,size,idx_foci]
            return spots, clusters #spots[:,:-1], clusters[:,:-1]

        def data_to_df(df, clusters_and_spots, mask_nuc = None, mask_cytosol_only=None,masks_complete_cells=None, nuc_area = 0, cyto_area =0, cell_area=0,
                        nuc_centroid_y=0, nuc_centroid_x=0, cyto_centroid_y=0, cyto_centroid_x=0, image_counter=0, is_cell_in_border = 0, spot_type=0, cell_counter =0,
                        nuc_int=None, cyto_int = None, complete_cell_int=None,pseudo_cyto_int=None,nucleus_cytosol_intensity_ratio=None,nucleus_pseudo_cytosol_intensity_ratio=None):

            # detect spots in nucleus
            if not (self.channels_nucleus in (None,[None]) ):
                spots_nuc,ts=separate_clusters_and_spots_in_mask(clusters_and_spots,mask=mask_nuc)
            else:
                spots_nuc, ts = None, None
            # Detecting spots in the cytosol only
            if not (self.channels_cytosol in (None,[None]) ) and not (self.channels_nucleus in (None,[None]) ):
                spots_cytosol_only, clusters_cytosol_only=separate_clusters_and_spots_in_mask(clusters_and_spots,mask=mask_cytosol_only)
            # detecting spots in complete cell if no nucleus is detected
            elif not (self.channels_cytosol in (None,[None]))  and (self.channels_nucleus in (None,[None]) ):
                spots_cytosol_only,clusters_cytosol_only=separate_clusters_and_spots_in_mask(clusters_and_spots,mask=masks_complete_cells)
            else:
                spots_cytosol_only = None
                clusters_cytosol_only = None
            
            # spots and clusters are reported in the format [Z,Y,X,size]
            number_columns = len(df.columns)
            if not(spots_nuc is None):
                num_ts = ts.shape[0]
                num_nuc = spots_nuc.shape[0]
            else:
                num_ts = 0
                num_nuc = 0
            if not(spots_cytosol_only is None) :
                num_cyto = spots_cytosol_only.shape[0] 
                num_cyto_clusters = clusters_cytosol_only.shape[0]
            else:
                num_cyto = 0
                num_cyto_clusters = 0
            # creating empty arrays if spots are detected in nucleus and cytosol
            if num_ts > 0:
                array_ts =                  np.zeros( ( num_ts, number_columns)  )
                spot_idx_ts =  np.arange(0,                  num_ts                                                 ,1 )
                detected_ts = True
            else:
                spot_idx_ts = []
                detected_ts = False
            
            if num_nuc > 0:
                array_spots_nuc =           np.zeros( ( num_nuc, number_columns) )
                spot_idx_nuc = np.arange(num_ts,             num_ts + num_nuc                                       ,1 )
                detected_nuc = True
            else:
                spot_idx_nuc = []
                detected_nuc = False
            
            if num_cyto>0:
                array_spots_cytosol_only =  np.zeros( ( num_cyto  ,number_columns) )
                spot_idx_cyt = np.arange(num_ts + num_nuc,   num_ts + num_nuc + num_cyto  ,1 )
                detected_cyto = True
            else:
                spot_idx_cyt = []
                detected_cyto = False

            if num_cyto_clusters>0:
                array_clusters_cytosol_only =  np.zeros( ( num_cyto_clusters  ,number_columns) )
                cluster_idx_cyt = np.arange(num_ts + num_nuc + num_cyto,   num_ts + num_nuc + num_cyto + num_cyto_clusters  ,1 )
                detected_cyto_clusters = True
            else:
                cluster_idx_cyt = []
                detected_cyto_clusters = False
            # Spot index
            spot_idx = np.concatenate((spot_idx_ts,  spot_idx_nuc, spot_idx_cyt,cluster_idx_cyt )).astype(int)
            
            # Populating arrays
            if not (self.channels_nucleus in (None,[None]) ):
                if detected_ts == True:
                    array_ts[:,10:13] = ts[:,:3]        # populating coord 
                    array_ts[:,13] = 1                  # is_nuc
                    array_ts[:,14] = 1                  # is_cluster
                    array_ts[:,15] =  ts[:,3]           # cluster_size
                    array_ts[:,16] = spot_type          # spot_type
                    array_ts[:,17] = is_cell_in_border  # is_cell_fragmented
                if (detected_nuc == True):
                    array_spots_nuc[:,10:13] = spots_nuc[:,:3]   # populating coord 
                    array_spots_nuc[:,13] = 1                   # is_nuc
                    array_spots_nuc[:,14] = 0                   # is_cluster
                    array_spots_nuc[:,15] = 0                   # cluster_size
                    array_spots_nuc[:,16] =  spot_type          # spot_type
                    array_spots_nuc[:,17] =  is_cell_in_border  # is_cell_fragmented
            
            if not (self.channels_cytosol in (None,[None]) ) :
                if (detected_cyto == True): # and (detected_cyto_clusters == False):
                    array_spots_cytosol_only[:,10:13] = spots_cytosol_only[:,:3]    # populating coord 
                    array_spots_cytosol_only[:,13] = 0                             # is_nuc
                    array_spots_cytosol_only[:,14] = 0                             # is_cluster
                    array_spots_cytosol_only[:,15] = 1                            # cluster_size
                    array_spots_cytosol_only[:,16] =  spot_type                    # spot_type
                    array_spots_cytosol_only[:,17] =  is_cell_in_border            # is_cell_fragmented
                if (detected_cyto_clusters == True): #(detected_cyto == True) and 
                    array_clusters_cytosol_only[:,10:13] = clusters_cytosol_only[:,:3]    # populating coord
                    array_clusters_cytosol_only[:,13] = 0                             # is_nuc
                    array_clusters_cytosol_only[:,14] = 1                             # is_cluster
                    array_clusters_cytosol_only[:,15] = clusters_cytosol_only[:,3]    # cluster_size
                    array_clusters_cytosol_only[:,16] =  spot_type                    # spot_type
                    array_clusters_cytosol_only[:,17] =  is_cell_in_border            # is_cell_fragmented
            
            # Combine arrays
            arrays_to_combine = []
            # Append arrays conditionally based on detection flags
            if detected_ts:
                arrays_to_combine.append(array_ts)
            if detected_nuc:
                arrays_to_combine.append(array_spots_nuc)
            if detected_cyto:
                arrays_to_combine.append(array_spots_cytosol_only)
            if detected_cyto_clusters:
                arrays_to_combine.append(array_clusters_cytosol_only)
            # Check if any arrays have been added; if none, initialize with zeros
            if arrays_to_combine:
                array_complete = np.vstack(arrays_to_combine)
            else:
                # If no arrays meet the conditions, initialize with zeros for structure
                array_complete = np.zeros((1, number_columns))
            # Saves a dataframe with zeros when no spots are detected on the cell.
            if array_complete.shape[0] ==1:
                # if NO spots are detected populate  with -1
                array_complete[:,2] = -1     # spot_id
                array_complete[:,8:self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME] = -1
                array_complete[:,13] = 0                             # is_nuc
                array_complete[:,14] = 0                             # is_cluster
                array_complete[:,15] = 0                             # cluster_size
                array_complete[:,16] = -1                           # spot_type
                array_complete[:,17] =  is_cell_in_border            # is_cell_fragmented
            else:
                # if spots are detected populate  the reported  array
                array_complete[:,2] = spot_idx.T     # spot_id
            # populating  array with cell  information
            array_complete[:,0] = image_counter  # image_id
            array_complete[:,1] = cell_counter   # cell_id
            array_complete[:,3] = nuc_centroid_y     #'nuc_y_centoid'
            array_complete[:,4] = nuc_centroid_x     #'nuc_x_centoid'
            array_complete[:,5] = cyto_centroid_y     #'cyto_y_centoid'
            array_complete[:,6] = cyto_centroid_x     #'cyto_x_centoid'
            array_complete[:,7] = nuc_area       #'nuc_area_px'
            array_complete[:,8] = cyto_area      # cyto_area_px
            array_complete[:,9] = cell_area      #'cell_area_px'
            
            # Populating array to add the average intensity in the cell
            for c in range (self.number_color_channels):
                if not (self.channels_nucleus in (None,[None]) ):
                    array_complete[:,self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME+c] = nuc_int[c] 
                if not (self.channels_cytosol in (None,[None]) ) :
                    array_complete[:,self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME+self.number_color_channels+c] = cyto_int[c]    
                if not (self.channels_cytosol in (None,[None]) ) :
                    # populating with complete_cell_int
                    array_complete[:,self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME+self.number_color_channels*2 +c] = complete_cell_int[c]  
                # populating with pseudo_cyto_int
                array_complete[:,self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME+self.number_color_channels*3 +c] = pseudo_cyto_int[c]  
                # populating with nucleus_cytosol_intensity_ratio
                array_complete[:,self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME+self.number_color_channels*4 +c] = nucleus_cytosol_intensity_ratio[c]  
                # populating with nucleus_pseudo_cytosol_intensity_ratio
                array_complete[:,self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME+self.number_color_channels*5 +c] = nucleus_pseudo_cytosol_intensity_ratio[c]  
            
            NUMBER_INTENSITY_MEASUREMENTS = 6  # Update this if you add/remove types of measurements. This considers the following columns: nuc_int, cyto_int, complete_cell_int, pseudo_cyto_int, nucleus_cytosol_intensity_ratio, nucleus_pseudo_cytosol_intensity_ratio                        
            # This section calculates the intenisty fo each spot and cluster
            # ts                     n x [Z,Y,X,size,idx_ts]
            # spots_nuc              n x [Z,Y,X]
            # spots_cytosol_only     n x [Z,Y,X]
            if num_ts >0:
                if self.use_fixed_size_for_intensity_calculation == False:
                    #cluster_spot_size = (ts[:,3]*self.yx_spot_size_in_px).astype('int')
                    area_per_spot_px2 = np.pi * (self.yx_spot_size_in_px / 2) ** 2
                    cluster_spot_size = 2 * np.sqrt(
                    ts[:,3] * area_per_spot_px2 / np.pi
                    )
                else:
                    cluster_spot_size = self.yx_spot_size_in_px
                intensity_ts,_,snr_ts, _, _, psf_amplitude_ts, psf_sigma_ts,intensities_total_ts = Intensity(original_image=self.image, spot_size=cluster_spot_size, array_spot_location_z_y_x=ts[:,0:3],  use_max_projection=self.use_maximum_projection).calculate_intensity()
            if num_nuc >0:
                intensity_spots_nuc, _ ,snr_spots_nuc, _, _, psf_amplitude_nuc, psf_sigma_nuc,intensities_total_spots_nuc = Intensity(original_image=self.image, spot_size=self.yx_spot_size_in_px, array_spot_location_z_y_x=spots_nuc[:,0:3],  use_max_projection=self.use_maximum_projection).calculate_intensity()
            if num_cyto >0 :
                intensity_spots_cyto, _ ,snr_spots_cyto, _, _,psf_amplitude_cyto, psf_sigma_cyto,intensities_total_spots_cyto = Intensity(original_image=self.image, spot_size=self.yx_spot_size_in_px, array_spot_location_z_y_x=spots_cytosol_only[:,0:3],  use_max_projection=self.use_maximum_projection).calculate_intensity()
            if num_cyto_clusters >0:
                if self.use_fixed_size_for_intensity_calculation == False:
                    #cluster_cyto_spot_size = (clusters_cytosol_only[:,3]*self.yx_spot_size_in_px).astype('int')
                    area_per_spot_px2 = np.pi * (self.yx_spot_size_in_px / 2) ** 2
                    cluster_cyto_spot_size = 2 * np.sqrt(
                    clusters_cytosol_only[:,3] * area_per_spot_px2 / np.pi
                    )
                else:
                    cluster_cyto_spot_size = self.yx_spot_size_in_px
                intensity_clusters_cytosol_only, _ ,snr_clusters_cytosol_only,_,_,psf_amplitude_clusters_cytosol_only, psf_sigma_clusters_cytosol_only, intensities_total_clusters_cyto_only= Intensity(original_image=self.image, spot_size=cluster_cyto_spot_size, array_spot_location_z_y_x=clusters_cytosol_only[:,0:3], use_max_projection=self.use_maximum_projection).calculate_intensity()

            # Check each condition and append the relevant arrays if detected
            intensity_arrays = []
            snr_arrays = []
            amplitude_psf_arrays = []
            sigma_psf_arrays = []
            total_intensity_arrays = []
            if detected_ts:
                intensity_arrays.append(intensity_ts)
                snr_arrays.append(snr_ts)
                amplitude_psf_arrays.append(psf_amplitude_ts)
                sigma_psf_arrays.append(psf_sigma_ts)
                total_intensity_arrays.append(intensities_total_ts)
            if detected_nuc:
                intensity_arrays.append(intensity_spots_nuc)
                snr_arrays.append(snr_spots_nuc)
                amplitude_psf_arrays.append(psf_amplitude_nuc)
                sigma_psf_arrays.append(psf_sigma_nuc)
                total_intensity_arrays.append(intensities_total_spots_nuc)
            if detected_cyto:
                intensity_arrays.append(intensity_spots_cyto)
                snr_arrays.append(snr_spots_cyto)
                amplitude_psf_arrays.append(psf_amplitude_cyto)
                sigma_psf_arrays.append(psf_sigma_cyto)
                total_intensity_arrays.append(intensities_total_spots_cyto)
            if detected_cyto_clusters:
                intensity_arrays.append(intensity_clusters_cytosol_only)
                snr_arrays.append(snr_clusters_cytosol_only)
                amplitude_psf_arrays.append(psf_amplitude_clusters_cytosol_only)
                sigma_psf_arrays.append(psf_sigma_clusters_cytosol_only)
                total_intensity_arrays.append(intensities_total_clusters_cyto_only)
            # Check if any arrays have been added; if none, initialize arrays with zeros
            if intensity_arrays:
                array_spot_int = np.vstack(intensity_arrays)
                array_snr = np.vstack(snr_arrays)
                array_amplitude_psf = np.vstack(amplitude_psf_arrays)
                array_sigma_psf = np.vstack(sigma_psf_arrays)
                array_total_intensity = np.vstack(total_intensity_arrays)
            else:
                # No detections at all, initialize with zeros
                array_spot_int = np.zeros((1, self.number_color_channels))
                array_snr = np.zeros((1, self.number_color_channels))
                array_amplitude_psf = np.zeros((1, self.number_color_channels))
                array_sigma_psf = np.zeros((1, self.number_color_channels))
                array_total_intensity = np.zeros((1, self.number_color_channels))

            # INDEX OF LAST COLUMNS WITH INTENSITY VALUES 
            INDEX_COLUMN_INTENSITY = self.NUMBER_OF_CONSTANT_COLUMNS_IN_DATAFRAME +  (self.number_color_channels * NUMBER_INTENSITY_MEASUREMENTS)
            # this section populates the intensity of the spots in the cell 
            array_complete[:, INDEX_COLUMN_INTENSITY: INDEX_COLUMN_INTENSITY+self.number_color_channels] = array_spot_int
            # this section populates the amplitude of the psf in the cell
            INDEX_COLUMN_PSF_AMPLITUDE = INDEX_COLUMN_INTENSITY + self.number_color_channels
            array_complete[:, INDEX_COLUMN_PSF_AMPLITUDE: INDEX_COLUMN_PSF_AMPLITUDE+self.number_color_channels] = array_amplitude_psf
            # this section populates the sigma of the psf in the cell
            INDEX_COLUMN_PSF_SIGMA = INDEX_COLUMN_PSF_AMPLITUDE + self.number_color_channels
            array_complete[:, INDEX_COLUMN_PSF_SIGMA: INDEX_COLUMN_PSF_SIGMA+self.number_color_channels] = array_sigma_psf
            # this section populates the signal to noise ratio of the spots in the cell
            INDEX_COLUMN_SNR = INDEX_COLUMN_PSF_SIGMA + self.number_color_channels
            array_complete[:, INDEX_COLUMN_SNR: INDEX_COLUMN_SNR+self.number_color_channels] = array_snr
            # this section populates the total intensity of the spots in the cell
            INDEX_COLUMN_TOTAL_INTENSITY = INDEX_COLUMN_SNR + self.number_color_channels
            array_complete[:, INDEX_COLUMN_TOTAL_INTENSITY: INDEX_COLUMN_TOTAL_INTENSITY+self.number_color_channels] = array_total_intensity

            # Creating the dataframe  
            new_dataframe = pd.DataFrame(array_complete, columns=df.columns)
            # Add spot size columns for each color channel using the circular approximation
            # A common way to relate the sigma values of a Gaussian fit to a spot size is by using the full width at half maximum (FWHM). For a circular approximation:
            #\text{FWHM} = 2 \sqrt{2 \ln 2}\,\sigma \approx 2.355\,\sigma
            # Add spot size columns for each color channel using the exact FWHM formula
            fwhm_factor = 2 * np.sqrt(2 * np.log(2))
            for c in range(self.number_color_channels):
                sigma_col = 'psf_sigma_ch_' + str(c)
                spot_size_col = 'spot_size_ch_' + str(c)
                #new_dataframe[spot_size_col] = np.round( 2.355 * new_dataframe[sigma_col],4)
                new_dataframe[spot_size_col] = (fwhm_factor * new_dataframe[sigma_col]).round(4)
            # Continue with any further processing or type casting if needed
            df = pd.concat([df, new_dataframe], ignore_index=True)
            
            new_dtypes = {'image_id':int, 'cell_id':int, 'spot_id':int,'is_nuc':int,'is_cluster':int,'nuc_loc_y':int, 'nuc_loc_x':int,'cyto_loc_y':int, 'cyto_loc_x':int,'nuc_area_px':int,'cyto_area_px':int, 'cell_area_px':int,'cluster_size':int,'spot_type':int,'is_cell_fragmented':int} # 'x':int,'y':int,'z':int,
            df = df.astype(new_dtypes)
            return df
        
        if not (self.masks_nuclei is None):
            n_masks = len(self.masks_nuclei)
        else:
            n_masks = len(self.masks_complete_cells)  
            
        # Initializing Dataframe
        if (not ( self.dataframe is None))   and  ( self.reset_cell_counter == False): # IF the dataframe exist and not reset for multi-channel is passed
            new_dataframe = self.dataframe
            counter_total_cells = np.max( self.dataframe['cell_id'].values) +1
        
        elif (not ( self.dataframe is None)) and (self.reset_cell_counter == True):    # IF dataframe exist and reset is passed
            new_dataframe = self.dataframe
            counter_total_cells = np.max( self.dataframe['cell_id'].values) - n_masks +1   # restarting the counter for the number of cells
        
        elif self.dataframe is None: # IF the dataframe does not exist.
            # Generate columns for the number of color channels
            list_columns_intensity_nuc = []
            list_columns_intensity_cyto = []
            list_columns_intensity_complete_cell =[]
            list_intensity_spots = []
            list_intensity_clusters = []
            list_nucleus_cytosol_intensity_ratio =[]
            list_columns_intensity_pseudo_cyto=[]
            list_nucleus_pseudo_cytosol_intensity_ratio=[]
            list_psfs_amplitude =[]
            list_psfs_sigma =[]
            list_snr =[]
            list_total_intensity = []
            for c in range(self.number_color_channels):
                list_columns_intensity_nuc.append( 'nuc_int_ch_' + str(c) )
                list_columns_intensity_cyto.append( 'cyto_int_ch_' + str(c) )
                list_columns_intensity_complete_cell.append( 'complete_cell_int_ch_' + str(c) )
                list_intensity_spots.append( 'spot_int_ch_' + str(c) )
                list_nucleus_cytosol_intensity_ratio.append('nuc_cyto_int_ratio_ch_' + str(c) )
                list_columns_intensity_pseudo_cyto.append('pseudo_cyto_int_ch_' + str(c) )
                list_nucleus_pseudo_cytosol_intensity_ratio.append('nuc_pseudo_cyto_int_ratio_ch_' + str(c) )
                list_psfs_amplitude.append('psf_amplitude_ch_' + str(c) )
                list_psfs_sigma.append('psf_sigma_ch_' + str(c) )
                list_snr.append('snr_ch_' + str(c) )
                list_total_intensity.append('total_spot_int_ch_' + str(c) )
            # creating the main dataframe with column names
            new_dataframe = pd.DataFrame( columns= ['image_id', 'cell_id', 'spot_id','nuc_loc_y', 'nuc_loc_x','cyto_loc_y', 'cyto_loc_x','nuc_area_px','cyto_area_px', 'cell_area_px','z', 'y', 'x','is_nuc','is_cluster','cluster_size','spot_type','is_cell_fragmented'] + list_columns_intensity_nuc + list_columns_intensity_cyto +list_columns_intensity_complete_cell+list_columns_intensity_pseudo_cyto + list_nucleus_cytosol_intensity_ratio+list_nucleus_pseudo_cytosol_intensity_ratio +list_intensity_spots+list_intensity_clusters + list_psfs_amplitude+list_psfs_sigma+list_snr + list_total_intensity)
            counter_total_cells = 0
        # loop for each cell in image
        
        num_pixels_to_dilate = 30
        for id_cell in range (0,n_masks): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
            # calculating nuclear area and center of mass
            if not (self.channels_nucleus in  (None, [None])):
                nuc_area, nuc_centroid_y, nuc_centroid_x = mask_selector(self.masks_nuclei[id_cell], calculate_centroid=True)
                selected_mask_nuc = self.masks_nuclei[id_cell]
                dilated_image_mask = binary_dilation(selected_mask_nuc, iterations=num_pixels_to_dilate).astype('int')
                pseudo_cytosol_mask = np.subtract(dilated_image_mask, selected_mask_nuc)
                pseudo_cyto_int = np.zeros( (self.number_color_channels ))
                tested_mask_for_border =  self.masks_nuclei[id_cell]
                nuc_int = np.zeros( (self.number_color_channels ))
                for k in range(self.number_color_channels ):
                    temp_img = np.max (self.image[:,:,:,k ],axis=0)
                    temp_masked_img = temp_img * self.masks_nuclei[id_cell]
                    temp_masked_img_pseudo_cytosol_mask = temp_img * pseudo_cytosol_mask
                    nuc_int[k] =  np.round( temp_masked_img[np.nonzero(temp_masked_img)].mean() , 5)
                    pseudo_cyto_int[k] =  np.round( temp_masked_img_pseudo_cytosol_mask[np.nonzero(temp_masked_img_pseudo_cytosol_mask)].mean() , 5)
                    del temp_img, temp_masked_img,temp_masked_img_pseudo_cytosol_mask
            else:
                nuc_area, nuc_centroid_y, nuc_centroid_x = 0,0,0
                selected_mask_nuc = None
                nuc_int = None
                pseudo_cyto_int = np.zeros( (self.number_color_channels )) 
            # calculating cytosol area and center of mass
            if not (self.channels_cytosol in (None, [None])):
                cell_area, cyto_centroid_y, cyto_centroid_x  = mask_selector(self.masks_complete_cells[id_cell],calculate_centroid=True)
                tested_mask_for_border =  self.masks_complete_cells[id_cell]
                complete_cell_int = np.zeros( (self.number_color_channels ))
                cyto_int = np.zeros( (self.number_color_channels ))
                for k in range(self.number_color_channels ):
                    temp_img = np.max (self.image[:,:,:,k ],axis=0)
                    # calculating cytosol intensity for complete cell mask
                    temp_masked_img = temp_img * self.masks_complete_cells[id_cell]
                    complete_cell_int[k] =  np.round( temp_masked_img[np.nonzero(temp_masked_img)].mean() , 5) 
                    # calculate cytosol intensity only if masks_cytosol_no_nuclei is not None
                    if not (self.masks_cytosol_no_nuclei in (None, [None])):
                        temp_masked_img_cyto_only = temp_img * self.masks_cytosol_no_nuclei[id_cell]
                        cyto_int[k]=  np.round( temp_masked_img_cyto_only[np.nonzero(temp_masked_img_cyto_only)].mean() , 5)
                    else:
                        cyto_int[k] = None
                        temp_masked_img_cyto_only= None
                    del temp_img, temp_masked_img, temp_masked_img_cyto_only
            else:
                complete_cell_int = None
                cell_area, cyto_centroid_y, cyto_centroid_x  = 0,0,0
                cyto_int = None
            
            # Calculating ratio between nucleus and cytosol intensity
            nucleus_cytosol_intensity_ratio = np.zeros( (self.number_color_channels ))
            nucleus_pseudo_cytosol_intensity_ratio = np.zeros( (self.number_color_channels ))
            # case where nucleus and cyto are passed 
            if not (self.channels_cytosol in (None, [None])) and not (self.channels_nucleus in  (None, [None])):
                for k in range(self.number_color_channels ):
                    nucleus_cytosol_intensity_ratio[k] = nuc_int[k]/ cyto_int[k]
                    nucleus_pseudo_cytosol_intensity_ratio[k] = nuc_int[k]/ pseudo_cyto_int[k]
            # case where nucleus is  passed but not cyto
            elif (self.channels_cytosol in (None, [None])) and not (self.channels_nucleus in  (None, [None])):
                for k in range(self.number_color_channels ):
                    nucleus_pseudo_cytosol_intensity_ratio[k] = nuc_int[k]/ pseudo_cyto_int[k]
            # case where nucleus and cyto are passed 
            if not (self.channels_cytosol in (None, [None])) and not (self.channels_nucleus in  (None, [None])):
                slected_masks_cytosol_no_nuclei = self.masks_cytosol_no_nuclei[id_cell]
                cyto_area,_,_ = mask_selector(self.masks_cytosol_no_nuclei[id_cell],calculate_centroid=False)
                selected_masks_complete_cells = self.masks_complete_cells[id_cell]
            # case where nucleus is  passed but not cyto
            elif (self.channels_cytosol in (None, [None])) and not (self.channels_nucleus in  (None, [None])):
                slected_masks_cytosol_no_nuclei = None
                cyto_area = 0
                selected_masks_complete_cells = None
            # case where cyto is passed but not nucleus
            elif not (self.channels_cytosol in (None, [None])) and (self.channels_nucleus in  (None, [None])):
                slected_masks_cytosol_no_nuclei,_,_ = mask_selector( self.masks_complete_cells[id_cell],calculate_centroid=False) 
                cyto_area, _, _  = mask_selector(self.masks_complete_cells[id_cell],calculate_centroid=False) # if not nucleus channel is passed the cytosol is consider the complete cell.
                selected_masks_complete_cells = self.masks_complete_cells[id_cell]
            else:
                slected_masks_cytosol_no_nuclei = None
                cyto_area = 0 
                selected_masks_complete_cells = None
            # determining if the cell is in the border of the image. If true the cell is in the border.
            is_cell_in_border =  np.any( np.concatenate( ( tested_mask_for_border[:,0],tested_mask_for_border[:,-1],tested_mask_for_border[0,:],tested_mask_for_border[-1,:] ) ) )  
            # Data extraction
            new_dataframe = data_to_df( new_dataframe, 
                                        self.clusters_and_spots, 
                                        mask_nuc = selected_mask_nuc, 
                                        mask_cytosol_only=slected_masks_cytosol_no_nuclei, 
                                        masks_complete_cells = selected_masks_complete_cells,
                                        nuc_area=nuc_area,
                                        cyto_area=cyto_area, 
                                        cell_area=cell_area, 
                                        nuc_centroid_y = nuc_centroid_y, 
                                        nuc_centroid_x = nuc_centroid_x,
                                        cyto_centroid_y = cyto_centroid_y, 
                                        cyto_centroid_x = cyto_centroid_x,
                                        image_counter=self.image_counter,
                                        is_cell_in_border = is_cell_in_border,
                                        spot_type = self.spot_type ,
                                        cell_counter =counter_total_cells,
                                        nuc_int=nuc_int,
                                        cyto_int=cyto_int,
                                        complete_cell_int = complete_cell_int,
                                        pseudo_cyto_int=pseudo_cyto_int,
                                        nucleus_cytosol_intensity_ratio=nucleus_cytosol_intensity_ratio,
                                        nucleus_pseudo_cytosol_intensity_ratio=nucleus_pseudo_cytosol_intensity_ratio)
            counter_total_cells +=1
        

        return new_dataframe


class ParticleMotion:

    def __init__(self, trackpy_dataframe, microns_per_pixel=1, step_size_in_sec=1, max_lagtime=100, show_plot=True, remove_drift=False, spot_type=0, plot_name=None):
        self.microns_per_pixel = microns_per_pixel
        self.step_size_in_sec = step_size_in_sec
        self.show_plot = show_plot 
        self.remove_drift = remove_drift
        self.plot_name = plot_name
        if 'spot_type' in trackpy_dataframe.columns:
            if len(trackpy_dataframe['spot_type'].unique()) > 1:
                self.trackpy_dataframe = trackpy_dataframe[trackpy_dataframe['spot_type'] == spot_type]
            else:
                self.trackpy_dataframe = trackpy_dataframe
        else:
            self.trackpy_dataframe = trackpy_dataframe
        self.trackpy_dataframe = self.trackpy_dataframe[['particle', 'frame', 'x', 'y']].copy()

        # use self.max_lagtime = max_lagtime but test it is not longer than max_lagtime in the dataframe.
        if max_lagtime is None:
            self.max_lagtime = int(self.trackpy_dataframe['frame'].max() )
        else:
            self.max_lagtime = min(max_lagtime, int(self.trackpy_dataframe['frame'].max()))

    def calculate_msd(self):
        # Calculation code (as provided)
        if self.remove_drift == True:
            temp_trackpy_df = self.trackpy_dataframe.copy()
            drift = tp.compute_drift(temp_trackpy_df)
            trackpy_df = tp.subtract_drift(temp_trackpy_df.copy(), drift)
            if self.show_plot == True: 
                drift.plot()
                plt.show()
        else:
            trackpy_df = self.trackpy_dataframe.copy()
        
        # Calculate the MSD
        em_px = tp.emsd(trackpy_df, mpp=1 , fps=1 / self.step_size_in_sec, max_lagtime=self.max_lagtime)
        # Calculate the diffusion coefficient
        slope = np.linalg.lstsq(np.array(em_px.index)[:, np.newaxis], em_px.values, rcond=None)[0][0]
        D_px2_s = slope / 4
        time_range = em_px.index  # Use the lag times from the MSD data
        model_fit = slope * em_px.index
        D_um2_s = D_px2_s * (self.microns_per_pixel ** 2)
        # if the user provides microns_per_pixel ==1 , print a warning
        if self.microns_per_pixel == 1:
            print("Warning: microns_per_pixel is set to 1. Results are in pixel units.")
        # Plotting
        if self.show_plot:
            plt.style.use(['default', 'fivethirtyeight'])
            fig, ax = plt.subplots(figsize=(6, 4))
            em_px.plot(style='o', label= r'D = {0:.3f} px²/s'.format(D_px2_s) + '\n' + r'D = {0:.3f} um²/s'.format(D_um2_s)  , ax=ax)
            # Use em.index directly for plotting the fit
            ax.plot(em_px.index, slope * em_px.index, label='Linear fit')
            ax.set(
                ylabel=r'$\langle \Delta r^2 \rangle$ [px$^2$]',
                xlabel='time (s)'
            )
            ax.legend(loc='upper left')
            if self.plot_name is not None:
                fig.savefig(self.plot_name, transparent=False, dpi=360, bbox_inches='tight', format='png')
            plt.show()
        return D_um2_s, D_px2_s, em_px, time_range, model_fit, trackpy_df


class CropArray():
    def __init__(self, image, df_crops, crop_size, remove_outliers=True, max_percentile=99.5, selected_time_point=None,normalize_each_particle=False):
        """
        Create a croparray and mean_crop from the given image and dataframe of crops.
        Additionally, extract the first appearance of each spot.

        Parameters:
        - image (ndarray): The input image array. The shape should be (frames, z, y, x, channels). TZYXC.
        - df_crops (DataFrame): The dataframe containing information about the crops.
        - crop_size (int): The size of the crop.
        - remove_outliers (bool): Flag indicating whether to remove outliers from the croparray. Default is True.
        - max_percentile (float): The percentile value to clip the croparray and mean_crop. Default is 98.5.
        - selected_time_point (int): The selected time point to crop. Default is None.

        Returns:
        - croparray (ndarray): The croparray containing cropped images for each particle and frame.
        - mean_crop (ndarray): The mean crop for each particle and channel.
        - first_snapshots (ndarray): Array of first appearance snapshots for each particle.
        """
        if len(image.shape) != 5:
            raise ValueError('The image must have 5 dimensions [T, Z, Y, X, C]. If the image has only 4 dimensions, expand the image axis to the desired shape using np.expand_dims(image, axis=0) or as needed')
        
        self.max_frame = df_crops['frame'].max()
        self.crop_size = crop_size
        self.half_spot = crop_size // 2
        self.number_color_channels = image.shape[-1]
        self.remove_outliers = remove_outliers
        self.max_percentile = max_percentile
        if selected_time_point is not None:
            image = image[selected_time_point]
            image = np.expand_dims(image, axis=0)
            df_crops = df_crops[df_crops['frame'] == selected_time_point]
            df_crops['frame'] = 0
        self.df_crops = df_crops
        self.image = image
        self.number_particles = len(df_crops['particle'].unique())
        self.normalize_each_particle = normalize_each_particle
    
    def run(self):
        croparray = np.full((self.number_particles * self.crop_size, (self.max_frame + 1) * self.crop_size, self.number_color_channels), np.nan)
        mean_crop = np.full((self.crop_size * self.number_particles, self.crop_size, self.number_color_channels), np.nan)
        first_snapshots = np.full((self.crop_size * self.number_particles, self.crop_size, self.number_color_channels), np.nan)

        for particle_idx, particle_id in enumerate(self.df_crops['particle'].unique()):
            df_particle = self.df_crops[self.df_crops['particle'] == particle_id]
            list_crops = [[] for _ in range(self.number_color_channels)]
            found_first_appearance = [False] * self.number_color_channels  # Track first appearance for each channel

            for i in range(self.max_frame + 1):
                df_frame = df_particle[df_particle['frame'] == i]
                if len(df_frame) > 0:
                    x_center, y_center, z_value = int(df_frame['x']), int(df_frame['y']), int(df_frame['z'])
                else:
                    x_center, y_center, z_value = None, None, None

                if x_center is not None and y_center is not None:
                    x_start, y_start = max(0, x_center - self.half_spot), max(0, y_center - self.half_spot)
                    x_end, y_end = min(self.image.shape[3], x_center + self.half_spot + 1), min(self.image.shape[2], y_center + self.half_spot + 1)
                    if x_end > x_start and y_end > y_start:
                        y_croparray_start = particle_idx * self.crop_size
                        x_croparray_start = i * self.crop_size
                        for ch in range(self.number_color_channels):
                            crop = self.image[i, z_value, y_start:y_end, x_start:x_end, ch]
                            if self.normalize_each_particle:
                                crop = crop / np.nanpercentile(crop,self.max_percentile)  * 255
                            croparray[y_croparray_start:y_croparray_start + self.crop_size, x_croparray_start:x_croparray_start + self.crop_size, ch] = crop
                            list_crops[ch].append(crop)

                            if not found_first_appearance[ch]:
                                first_snapshots[particle_idx * self.crop_size:(particle_idx + 1) * self.crop_size, :, ch] = crop.T
                                found_first_appearance[ch] = True

            for ch in range(self.number_color_channels):
                stacked_arrays = np.stack(list_crops[ch], axis=0)
                mean_array = np.nanmean(stacked_arrays, axis=0) # use the median to avoid artifacts caused by outliers.
                mean_crop[particle_idx * self.crop_size:(particle_idx + 1) * self.crop_size, :, ch] = mean_array.T

        if self.remove_outliers:
            croparray_non_nan = np.nan_to_num(croparray)
            mean_crop = np.nan_to_num(mean_crop)
            first_snapshots = np.nan_to_num(first_snapshots)
            for ch in range(self.number_color_channels):
                croparray[:, :, ch] = np.clip(croparray_non_nan[:, :, ch], 0, np.nanpercentile(croparray_non_nan[:, :, ch], self.max_percentile))
                mean_crop[:, :, ch] = np.clip(mean_crop[:, :, ch], 0, np.nanpercentile(mean_crop[:, :, ch], self.max_percentile))
                first_snapshots[:, :, ch] = np.clip(first_snapshots[:, :, ch], 0, np.nanpercentile(first_snapshots[:, :, ch], self.max_percentile))
            croparray[croparray == 0] = np.nan
            mean_crop[mean_crop == 0] = np.nan
            first_snapshots[first_snapshots == 0] = np.nan

        return croparray, mean_crop, first_snapshots, self.crop_size


class Metadata():
    '''
    This class is intended to generate a metadata file containing used dependencies, user information, and parameters used to run the code.
    
    Parameters
    
    data_dir: str or PosixPath
        Directory containing the images to read.
    channels_cytosol : List of int
        List with integers indicating the index of channels for the cytosol segmentation. 
    channels_nucleus : list of int
        List with integers indicating the index of channels for the nucleus segmentation.
    channels_spots  : list of int
        List with integers indicating the index of channels for the spot detection using.
    diameter_cytosol : int
        Average cytosol size in pixels. The default is 150.
    diameter_nucleus : int
        Average nucleus size in pixels. The default is 100.
    minimum_spots_cluster : int
        Number of spots in a neighborhood for a point to be considered as a core point (from which a cluster is expanded). This includes the point itself.
    list_voxels : List of lists or None
        List with a tuple with two elements (voxel_size_z,voxel_size_yx ) for each spot channel.

    file_name_str : str
        Name used for the metadata file. The final name has the format metadata_<<file_name_str>>.txt
    list_counter_cell_id : str
        Counter that keeps track of the number of images in the folder.
    threshold_for_spot_detection : int
        Threshold value used to discriminate background noise from mRNA spots in the image.
    '''
    def __init__(self,data_dir, channels_cytosol, channels_nucleus, channels_spots, diameter_nucleus, diameter_cytosol, minimum_spots_cluster, list_voxels=None, list_spot_size_px=None, file_name_str=None,list_segmentation_successful=True,list_counter_image_id=[],threshold_for_spot_detection=[],number_of_images_to_process=None,remove_z_slices_borders=False,NUMBER_Z_SLICES_TO_TRIM=0,cluster_radius_nm=0,list_thresholds_spot_detection=[None],list_average_spots_per_cell=[None],list_number_detected_cells=[None],list_is_image_sharp=[None],list_metric_sharpeness_images=[None],remove_out_of_focus_images=False,sharpness_threshold=None):
        
        self.list_images, self.path_files, self.list_files_names, self.number_images = ReadImages(data_dir,number_of_images_to_process).read()
        self.channels_cytosol = channels_cytosol
        self.channels_nucleus = channels_nucleus
        if isinstance(channels_spots, list): 
            self.channels_spots = channels_spots
        else:
            self.channels_spots = [channels_spots]
        self.diameter_nucleus = diameter_nucleus
        self.diameter_cytosol = diameter_cytosol
        self.list_voxels = list_voxels
        self.list_spot_size_px = list_spot_size_px
        self.file_name_str=file_name_str
        self.minimum_spots_cluster = minimum_spots_cluster
        self.threshold_for_spot_detection=threshold_for_spot_detection
        if  (not str(data_dir.name)[0:5] ==  'temp_') and (self.file_name_str is None):
            self.filename = 'metadata_'+ str(data_dir.name).replace(" ", "")  +'.txt'
            self.filename_csv = 'images_report_'+ str(data_dir.name).replace(" ", "")+'.csv'
        elif not(self.file_name_str is None):
            self.filename = 'metadata_'+ str(file_name_str).replace(" ", "") +'.txt'
            self.filename_csv = 'images_report_'+ str(file_name_str).replace(" ", "")+'.csv'
        else:
            self.filename = 'metadata_'+ str(data_dir.name[5:].replace(" ", "")) +'.txt'
            self.filename_csv = 'images_report_'+ str(file_name_str).replace(" ", "")+'.csv'
        self.data_dir = data_dir
        self.list_segmentation_successful =list_segmentation_successful
        self.list_counter_image_id=list_counter_image_id
        self.remove_z_slices_borders=remove_z_slices_borders
        self.NUMBER_Z_SLICES_TO_TRIM=NUMBER_Z_SLICES_TO_TRIM
        self.cluster_radius_nm=cluster_radius_nm
        self.list_thresholds_spot_detection=list_thresholds_spot_detection
        self.list_average_spots_per_cell=list_average_spots_per_cell
        self.list_number_detected_cells=list_number_detected_cells
        self.list_is_image_sharp=list_is_image_sharp
        self.list_metric_sharpeness_images=list_metric_sharpeness_images
        self.remove_out_of_focus_images=remove_out_of_focus_images
        self.sharpness_threshold=sharpness_threshold
    def write_metadata(self):
        '''
        This method writes the metadata file.
        '''
        installed_modules = [str(module).replace(" ","==") for module in pkg_resources.working_set]
        important_modules = [ 'tqdm', 'torch','tifffile', 'setuptools', 'scipy', 'scikit-learn', 'scikit-image', 'PyYAML', 'pysmb', 'pyfiglet', 'pip', 'Pillow', 'pandas', 'opencv-python-headless', 'numpy', 'numba', 'natsort', 'mrc', 'matplotlib', 'llvmlite', 'jupyter-core', 'jupyter-client', 'joblib', 'ipython', 'ipython-genutils', 'ipykernel', 'cellpose', 'big-fish']
        def create_data_file(filename):
            if sys.platform == 'linux' or sys.platform == 'darwin':
                os.system('touch  ' + filename)
            elif sys.platform == 'win32':
                os.system('echo , > ' + filename)
        number_spaces_pound_sign = 75
        
        def write_data_in_file(filename):
            list_processing_image=[]
            list_image_id =[]
            with open(filename, 'w') as fd:
                fd.write('#' * (number_spaces_pound_sign)) 
                fd.write('\nAUTHOR INFORMATION  ')
                fd.write('\n    Author: ' + getpass.getuser())
                fd.write('\n    Created: ' + datetime.datetime.today().strftime('%d %b %Y'))
                fd.write('\n    Time: ' + str(datetime.datetime.now().hour) + ':' + str(datetime.datetime.now().minute) )
                fd.write('\n    Operative System: ' + sys.platform )
                fd.write('\n    Hostname: ' + socket.gethostname() + '\n')
                fd.write('#' * (number_spaces_pound_sign) ) 
                fd.write('\nPARAMETERS USED  ')
                fd.write('\n    channels_cytosol: ' + str(self.channels_cytosol) )
                fd.write('\n    channels_nucleus: ' + str(self.channels_nucleus) )
                fd.write('\n    channels_spots: ' + str(self.channels_spots) )
                fd.write('\n    diameter_nucleus: ' + str(self.diameter_nucleus) )
                fd.write('\n    diameter_cytosol: ' + str(self.diameter_cytosol) )
                fd.write('\n    Spot parameters')
                for k in range (0,len(self.channels_spots)):
                    fd.write('\n      For Channel ' + str(self.channels_spots[k]) )
                    fd.write('\n        voxel_size_z: ' + str(self.list_voxels[k][0]) )
                    fd.write('\n        voxel_size_yx: ' + str(self.list_voxels[k][1]) )
                    fd.write('\n        spot_size__px_z: ' + str(self.list_spot_size_px[k][0]) )
                    fd.write('\n        spot_size_px_yx: ' + str(self.list_spot_size_px[k][1]) )
                    if not(self.threshold_for_spot_detection in (None, [None]) ):
                        fd.write('\n        threshold_spot_detection: ' + str(self.threshold_for_spot_detection[k]) )
                    else:
                        fd.write('\n        threshold_spot_detection: ' + 'automatic value using BIG-FISH' )
                fd.write('\n    minimum_spots_cluster: ' + str(self.minimum_spots_cluster) )
                fd.write('\n    remove_z_slices_borders: ' + str(self.remove_z_slices_borders) )
                fd.write('\n    number of z-slices trimmed at each border: ' + str(self.NUMBER_Z_SLICES_TO_TRIM) )
                fd.write('\n    cluster radius: ' + str(self.cluster_radius_nm) )
                fd.write('\n    remove_out_of_focus_images: ' + str(self.remove_out_of_focus_images))
                if self.remove_out_of_focus_images == True:
                    fd.write('\n    sharpness_threshold: ' + str(self.sharpness_threshold))
                fd.write('\n') 
                fd.write('#' * (number_spaces_pound_sign) ) 
                fd.write('\nFILES AND DIRECTORIES USED ')
                fd.write('\n    Directory path: ' + str(self.data_dir) )
                fd.write('\n    Folder name: ' + str(self.data_dir.name)  )
                # for loop for all the images.
                fd.write('\n    Images in the directory :'  )
                # size of longest name string
                file_name_len =0
                max_file_name_len =0
                for _, img_name in enumerate (self.list_files_names):
                    if len(img_name) > file_name_len:
                        max_file_name_len = len(img_name)
                    else:
                        max_file_name_len =0
                
                str_label_img = '| Image Name'
                size_str_label_img = len(str_label_img)
                space_for_image_name = np.min((size_str_label_img, (size_str_label_img-max_file_name_len)))+1
                fd.write('\n        '+ str_label_img+' '* space_for_image_name + '      '+ '| Sharpness metric' + '      ' +'| Image Id'  )
                counter=0
                for indx, img_name in enumerate (self.list_files_names):
                    file_name_len = len(img_name)
                    difference_name_len = max_file_name_len-file_name_len
                    if (self.list_segmentation_successful[indx]== True) and (self.list_is_image_sharp[indx]== True):
                        fd.write('\n        '+ img_name + (' '*(difference_name_len+4))+ ' '*8+ str(self.list_metric_sharpeness_images[indx]) +  '        ' + str(self.list_counter_image_id[counter]) )
                        list_image_id.append(self.list_counter_image_id[counter])
                        counter+=1
                        list_processing_image.append('successful')
                    elif self.list_is_image_sharp[indx]== False:
                        fd.write('\n        '+ img_name + (' '*(difference_name_len+4)) + ' '*8 + str(self.list_metric_sharpeness_images[indx])+ '      - error out of focus.')
                        list_processing_image.append('error out of focus')
                        list_image_id.append(-1)
                    else:
                        fd.write('\n        '+ img_name + (' '*(difference_name_len+4))+ ' '*8 + str(self.list_metric_sharpeness_images[indx])+ '      - error segmentation.')
                        list_processing_image.append('error segmentation')
                        list_image_id.append(-1)
                fd.write('\n') 
                fd.write('#' * (number_spaces_pound_sign)) 
                fd.write('\nSUMMARY RESULTS')
                # iterate for all processed images and printing the obtained threshold intensity value.
                for k in range(len(self.channels_spots)):
                    fd.write('\n    For Channel ' + str(self.channels_spots[k]) )
                    fd.write('\n             Image Id    |    threshold    |    number cells    |  mean spots per cell |' )
                    for i,image_id in enumerate(self.list_counter_image_id) :
                        image_id_str = str(image_id)
                        len_id = len(image_id_str)
                        threshold_str = str(int(self.list_thresholds_spot_detection[i][k]))
                        len_ts= len(threshold_str)
                        number_cells_str = str(int(self.list_number_detected_cells[i]))
                        len_nc = len(number_cells_str)
                        average_spots_per_cells_str = str(int(self.list_average_spots_per_cell[i][k]))
                        fd.write('\n                ' +'    '+image_id_str + ' '* np.max((1,(13-len_id))) +
                                                '    '+threshold_str +  ' '* np.max((1,(14-len_ts))) +
                                                '    '+number_cells_str + ' '* np.max((1,(17-len_nc))) +
                                                '    '+average_spots_per_cells_str )
                    
                    total_average_number_cells = str(int(np.mean(self.list_number_detected_cells)))
                    total_detected_cells = str(int(np.sum(self.list_number_detected_cells)))
                    fd.write('\n              ' +'Average:' + ' '* np.max((1,(13-len_id))) +
                                                '    '+' '*len_ts +  ' '* np.max((1,(14-len_ts))) +
                                                '    '+ total_average_number_cells + ' '* np.max((1,(17-len_nc))) )
                    
                    fd.write('\n              ' +'Total  :' + ' '* np.max((1,(13-len_id))) +
                                                '    '+' '*len_ts +  ' '* np.max((1,(14-len_ts))) +
                                                '    '+ total_detected_cells + ' '* np.max((1,(17-len_nc)))  )
                fd.write('\n') 
                fd.write('#' * (number_spaces_pound_sign)) 
                fd.write('\nREPRODUCIBILITY ')
                fd.write('\n    Platform: \n')
                fd.write('        Python: ' + str(platform.python_version()) )
                fd.write('\n    Dependencies: ')
                # iterating for all modules
                for module_name in installed_modules:
                    if any(module_name[0:4] in s for s in important_modules):
                        fd.write('\n        '+ module_name)
                fd.write('\n') 
                fd.write('#' * (number_spaces_pound_sign) ) 
            return list_processing_image,list_image_id
        create_data_file(self.filename)
        list_processing_image, list_image_id = write_data_in_file(self.filename)
        data = {'Image_id': list_image_id, 'Image_name': self.list_files_names, 'Processing': list_processing_image}
        df = pd.DataFrame(data)
        df.to_csv(self.filename_csv)
        return None


class ReportPDF():
    '''
    This class intended to create a PDF report including the images generated during the pipeline.
    
    Parameters
    
    directory_results: str or PosixPath
        Directory containing the images to include in the report.
    channels_spots  : list of int
        List with integers indicating the index of channels for the spot detection using.
    save_all_images : Bool, optional.
        If true, it shows a all planes for the spot plot detection. The default is True.
    list_z_slices_per_image : int
        List containing all z-slices for each figure.
        
    .. image:: images/pdf_report.png
    
    This PDF file is generated, and it contains the processing steps for each image in the folder.
    
    '''    
    def __init__(self, directory,filenames_for_pdf_report, channels_spots,save_all_images,list_z_slices_per_image,threshold_for_spot_detection,list_segmentation_successful=True):
        self.directory = directory
        if isinstance(channels_spots, list): 
            self.channels_spots = channels_spots
        else:
            self.channels_spots = [channels_spots]
        self.save_all_images = save_all_images
        self.list_z_slices_per_image = list_z_slices_per_image
        self.threshold_for_spot_detection=threshold_for_spot_detection
        self.list_segmentation_successful =list_segmentation_successful
        self.filenames_for_pdf_report=filenames_for_pdf_report
        
    def create_report(self):
        '''
        This method creates a PDF with the original images, images for cell segmentation and images for the spot detection.
        '''
        pdf = FPDF()
        WIDTH = 210
        HEIGHT = 297
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        # code that reads the main file names
        list_files_names = self.filenames_for_pdf_report #[]
        # Main loop that reads each image and makes the pdf
        for i,temp_file_name in enumerate(list_files_names):
            pdf.cell(w=0, h=10, txt='Original image: ' + temp_file_name,ln =2,align = 'L')
            # code that returns the path of the original image
            temp_original_img_name = pathlib.Path().absolute().joinpath( self.directory, 'ori_' + temp_file_name +'.png' )
            pdf.image(str(temp_original_img_name), x=0, y=20, w=WIDTH-30)
            # creating some space
            for text_idx in range(0, 12):
                pdf.cell(w=0, h=10, txt='',ln =1,align = 'L')
            pdf.cell(w=0, h=10, txt='Cell segmentation: ' + temp_file_name,ln =1,align = 'L')
            # code that returns the path of the segmented image
            if self.list_segmentation_successful[i]==True:
                temp_segmented_img_name = pathlib.Path().absolute().joinpath( self.directory, 'seg_' + temp_file_name +'.png' )
                pdf.image(str(temp_segmented_img_name), x=0, y=HEIGHT/2, w=WIDTH-30)
            else:
                pdf.cell(w=0, h=20, txt='Segmentation was not possible for image: ' + temp_file_name,ln =1,align = 'L')
                pdf.add_page()
            # Code that plots the detected spots.
            if (self.save_all_images==True) and (self.list_segmentation_successful[i]==True):
                for id_channel, channel in enumerate(self.channels_spots):
                    counter=1
                    pdf.add_page() # adding a page
                    for z_slice in range(0, self.list_z_slices_per_image[i]):
                        temp_seg_name = pathlib.Path().absolute().joinpath( self.directory, 'det_' + temp_file_name + '_ch_'+str(channel) + '_slice_'+ str(z_slice) +'.png' )
                        # Plotting bottom image
                        if counter%2==0: # Adding space if is an even counter
                            # adding some space to plot the bottom image
                            for j in range(0, 11):
                                pdf.cell(w=0, h=10, txt='',ln =1,align = 'L')
                            # Plotting the image
                            pdf.cell(w=0, h=0, txt='Spots Ch_ ' + str(channel) + '_slice_'+ str(z_slice) +': '+ temp_file_name,ln =2,align = 'L') 
                            pdf.image(str(temp_seg_name), x=0, y=HEIGHT//2, w=WIDTH-80)
                            pdf.add_page()
                        # plotting top image
                        else:
                            pdf.cell(w=0, h=10, txt='Spots Ch_ ' + str(channel) + '_slice_'+ str(z_slice) +': '+ temp_file_name,ln =2,align = 'L') 
                            pdf.image(str(temp_seg_name), x=0, y=20, w=WIDTH-80)
                        counter=counter+1
                    pdf.add_page()
                    try:
                        if (self.threshold_for_spot_detection[id_channel] is None):
                            temp_elbow_name = pathlib.Path().absolute().joinpath( self.directory, 'det_' + temp_file_name + '__elbow_'+ '_ch_'+str(channel)+'.png' )
                            pdf.image(str(temp_elbow_name), x=0, y=HEIGHT//2, w=WIDTH-140)
                        else:
                            pdf.cell(w=0, h=10, txt='Used intensity threshold = '+str(self.threshold_for_spot_detection[id_channel]) ,ln =2,align = 'L')
                    except:
                        pdf.cell(w=0, h=10, txt='Error during the calculation of the elbow plot',ln =2,align = 'L')
                    pdf.add_page()
            elif self.list_segmentation_successful[i]==True:
                pdf.add_page()
                for id_channel, channel in enumerate(self.channels_spots):
                    # Plotting the image with detected spots
                    temp_seg_name = pathlib.Path().absolute().joinpath( self.directory, 'det_' + temp_file_name + '_ch_'+str(channel)+'.png' )
                    pdf.cell(w=0, h=10, txt='Spots Ch_ ' + str(channel) + ': '+ temp_file_name,ln =2,align = 'L') 
                    try:
                        pdf.image(str(temp_seg_name), x=0, y=20, w=WIDTH-30)  
                    except:
                        pdf.cell(w=0, h=10, txt='Error during the calculation of the elbow plot',ln =2,align = 'L')                  
                    # adding some space
                    for j in range(0, 12):
                        pdf.cell(w=0, h=10, txt='',ln =1,align = 'L')
                    # Plotting the elbow plot
                    try:
                        if (self.threshold_for_spot_detection[id_channel] is None):
                            temp_elbow_name = pathlib.Path().absolute().joinpath( self.directory, 'det_' + temp_file_name + '__elbow_'+ '_ch_'+str(channel)+'.png' )
                            pdf.image(str(temp_elbow_name), x=0, y=HEIGHT//2, w=WIDTH-140)
                        else:
                            pdf.cell(w=0, h=10, txt='Used intensity threshold = '+str(self.threshold_for_spot_detection[id_channel]) ,ln =2,align = 'L')
                    except:
                        pdf.cell(w=0, h=10, txt='Error during the calculation of the elbow plot',ln =2,align = 'L')
                    pdf.add_page()                
        pdf_name =  'pdf_report_' + self.directory.name[13:].replace(" ", "") + '.pdf'
        pdf.output(pdf_name, 'F')
        return None
    
    
class PipelineFISH():
    '''
    This class is intended to perform complete FISH analyses including cell segmentation and spot detection.
    
    Parameters
    data_folder_path : str or Pathlib object,
        Path to the folder with the images to process.
    parameter: bool, optional
        parameter description. The default is True. 
    voxel_size_z : int, optional
        Microscope conversion px to nanometers in the z axis. The default is 500.
    voxel_size_yx : int, optional
        Microscope conversion px to nanometers in the xy axis.   The default is 160.
    psf_z : int, optional
        Theoretical size of the PSF emitted by a [rna] spot in the z plan, in nanometers.  The default is 350.
    psf_yx: int, optional
        Theoretical size of the PSF emitted by a [rna] spot in the yx plan, in nanometers.  The default is 160.
    list_masks : List of Numpy or None.
        list of Numpy arrays where each array has values from 0 to n where n is the number of masks in  the image.
    save_all_images : Bool, optional.
        If true, it shows a all planes for the spots plot detection. The default is True.
    display_spots_on_multiple_z_planes : Bool, optional.
        If true, it shows a spots on the plane below and above the selected plane. The default is False.
    use_log_filter_for_spot_detection : bool, optional
        Uses Big_FISH log_filter. The default is True.
    threshold_for_spot_detection: scalar, list, or None.
        Indicates the intensity threshold used for spot detection, the default is None, and indicates that the threshold is calculated automatically.
    list_selected_z_slices : list or None
    number_of_images_to_process: int or None, optional
        This number indicates a subset of images to process from a given repository. The default is None, and this indicates that the code will process all images in the given repository.
    remove_z_slices_borders : bool optional
        This flag indicates the removal of the two first and last 2 z-slices from the segmentation and quantification. This needed to avoid processing images out of focus. The default is True.
    '''

    def __init__(self,data_folder_path=None, channels_cytosol=None, channels_nucleus=None, channels_spots=None,diameter_nucleus=100, diameter_cytosol=200, minimum_spots_cluster=5,  image=None, masks_dir=None, show_plot=True, voxel_size_z=500, voxel_size_yx=160 ,z_spot_size_in_px=2,yx_spot_size_in_px=5,cluster_radius_nm= None, file_name_str =None,optimization_segmentation_method='default',save_all_images=False,display_spots_on_multiple_z_planes=False,use_log_filter_for_spot_detection=True,threshold_for_spot_detection=[None],NUMBER_OF_CORES=1,list_selected_z_slices=None,save_filtered_images=False,number_of_images_to_process=None,remove_z_slices_borders=False,remove_out_of_focus_images = False,sharpness_threshold =1.05,save_pdf_report=False,folder_name='temp',save_files=True,model_nuc_segmentation='nuclei',model_cyto_segmentation='cyto3',pretrained_model_nuc_segmentation=None, pretrained_model_cyto_segmentation=None):
        
        if type(data_folder_path)== pathlib.PosixPath or isinstance(data_folder_path, str) or type(data_folder_path)== pathlib.WindowsPath:
            list_images, _ , self.list_files_names, self.number_images = ReadImages(data_folder_path,number_of_images_to_process).read()
        else:
            #list_images =[image]
            self.list_files_names = ['temp.tif']
            self.number_images = 1 
            
        #if Utilities().is_None(image) == False:
        if not (image is None):
            if len(image.shape)<=3:
                image = np.expand_dims(image,axis=0)
            list_images =[image.astype(np.uint16)]
        
        self.number_of_images_to_process = self.number_images
        if len(list_images[0].shape) < 4:
            list_images_extended = [ np.expand_dims(img,axis=0) for img in list_images ] 
            list_images = list_images_extended
        else:
            list_images = list_images
        # Trimming the z-slices in each image based on 
        list_images_trimmed = []
        num_z_silces = list_images[0].shape[0] 
        MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE = 10 # This constant is only used to remove the extre z-slices on the original image.
        if (remove_z_slices_borders == True) and  (list_selected_z_slices is None) and (num_z_silces>=MINIMAL_NUMBER_OF_Z_SLICES_TO_CONSIDER_A_3D_IMAGE): 
            NUMBER_Z_SLICES_TO_TRIM = 1 # This constant indicates the number of z_slices to remove from the border.
            list_selected_z_slices = np.arange(NUMBER_Z_SLICES_TO_TRIM,num_z_silces-NUMBER_Z_SLICES_TO_TRIM,1)
            self.remove_z_slices_borders = True
        else:
            NUMBER_Z_SLICES_TO_TRIM = 0
            self.remove_z_slices_borders = False
        self.NUMBER_Z_SLICES_TO_TRIM = NUMBER_Z_SLICES_TO_TRIM
        if not (list_selected_z_slices is None):
            number_images = len(list_images)
            for i in range (number_images):
                if len(list_selected_z_slices) > list_images[i].shape[0]:
                    raise ValueError("Error: You are selecting z-slices that are outside the size of your image. In PipelineFISH, please use this option list_selected_z_slices=None ")
                list_images_trimmed.append(list_images[i][list_selected_z_slices,:,:,:]   )
            self.list_images = list_images_trimmed
        else:
            self.list_images = list_images
        self.list_z_slices_per_image = [ img.shape[0] for img in self.list_images] # number of z-slices in the figure
        self.channels_cytosol = Utilities().make_it_a_list(channels_cytosol) #channels_cytosol
        self.channels_nucleus = Utilities().make_it_a_list(channels_nucleus) #channels_nucleus
        channels_spots = Utilities().make_it_a_list(channels_spots) #channels_spots
        self.channels_spots = channels_spots
        self.diameter_nucleus = diameter_nucleus
        self.diameter_cytosol = diameter_cytosol

        self.z_spot_size_in_px=z_spot_size_in_px
        self.yx_spot_size_in_px=yx_spot_size_in_px
        self.list_voxels = [voxel_size_z,voxel_size_yx]
        self.list_spot_size_px = [z_spot_size_in_px, yx_spot_size_in_px]
        self.minimum_spots_cluster = minimum_spots_cluster
        self.show_plot = show_plot
        if cluster_radius_nm is None:
            self.cluster_radius_nm = int(self.list_voxels[1] * 4)
        else:
            self.cluster_radius_nm = cluster_radius_nm
        
        if not(data_folder_path is None):
            self.data_folder_path = data_folder_path
        else:
            data_folder_path = pathlib.Path().absolute().joinpath(folder_name)
            if not data_folder_path.exists() and (save_files == True):
                data_folder_path.mkdir(parents=False, exist_ok=True)
            self.data_folder_path = data_folder_path
        
        if not(file_name_str is None):
            self.name_for_files = file_name_str
        else:
            self.name_for_files = self.data_folder_path.name
        self.masks_dir=masks_dir
        # saving the masks if they are not passed as a directory
        if (masks_dir is None):
            self.save_masks_as_file = True
        else:
            self.save_masks_as_file = False
        self.optimization_segmentation_method = optimization_segmentation_method # optimization_segmentation_method = 'default', 'intensity_segmentation' 'z_slice_segmentation_marker', 'gaussian_filter_segmentation' , None
        if np.min(self.list_z_slices_per_image) < 5:
            self.optimization_segmentation_method = 'center_slice'
        self.save_all_images = save_all_images                                  # Displays all the z-planes
        self.display_spots_on_multiple_z_planes = display_spots_on_multiple_z_planes  # Displays the ith-z_plane and the detected spots in the planes ith-z_plane+1 and ith-z_plane
        self.use_log_filter_for_spot_detection =use_log_filter_for_spot_detection
        self.NUMBER_OF_CORES=NUMBER_OF_CORES
        self.save_filtered_images= save_filtered_images
        self.sharpness_threshold = sharpness_threshold
        # Testing sharpness in images
        self.remove_out_of_focus_images = remove_out_of_focus_images
        if remove_out_of_focus_images == True:
            list_metric_sharpeness_images, list_is_image_sharp,list_sharp_images = Utilities().calculate_sharpness(list_images, channels_spots=channels_spots,threshold=sharpness_threshold)
        else:
            list_sharp_images= list_images
            list_is_image_sharp=np.ones(len(list_images))
            list_is_image_sharp = [bool(x) for x in list_is_image_sharp]
            list_metric_sharpeness_images= Utilities().calculate_sharpness(list_images, channels_spots=channels_spots,threshold=sharpness_threshold)[0]
        self.list_is_image_sharp = list_is_image_sharp
        self.list_metric_sharpeness_images =list_metric_sharpeness_images
        
        # Section that creates an automated intensity threshold for spot detection by using the average values obtained from processing all the directiory of images. 
        MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD = 3
        if (threshold_for_spot_detection == None) and (len(list_sharp_images)>MINUMUM_NUMBER_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD):
            MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD = 50
            number_images_to_test = np.min((MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD,len(list_sharp_images)))
            sub_section_images_to_test =list_sharp_images[:number_images_to_test]
            threshold_for_spot_detection =[]
            for i in range(len(channels_spots)):
                list_thresholds=[]
                voxel_size_z = self.list_voxels[0]
                voxel_size_yx = self.list_voxels[1]
                
                for _, image_selected in enumerate(sub_section_images_to_test):
                    threshold = BigFISH(image_selected,
                                        channels_spots[i], 
                                        voxel_size_z = voxel_size_z,
                                        voxel_size_yx = voxel_size_yx, 
                                        yx_spot_size_in_px=self.yx_spot_size_in_px, 
                                        z_spot_size_in_px=self.z_spot_size_in_px, 
                                        cluster_radius_nm=self.cluster_radius_nm,
                                        minimum_spots_cluster=self.minimum_spots_cluster, 
                                        use_log_filter_for_spot_detection =self.use_log_filter_for_spot_detection,
                                        threshold_for_spot_detection=None).detect()[2]
                    list_thresholds.append(threshold)
                # calculating the average threshold for all images removing min and max values.
                array_threshold_spot_detection = np.array(list_thresholds)
                min_val = np.min(array_threshold_spot_detection)
                max_val = np.max(array_threshold_spot_detection)
                mask_ts = (array_threshold_spot_detection != min_val) & (array_threshold_spot_detection != max_val)
                average_threshold_spot_detection= int(np.mean(array_threshold_spot_detection[mask_ts]))
                threshold_for_spot_detection.append(average_threshold_spot_detection)
            print('Most images are noisy. An average threshold value for spot detection has been calculated using all images:', threshold_for_spot_detection)
        else:
            threshold_for_spot_detection = Utilities().create_list_thresholds(channels_spots,threshold_for_spot_detection)
        self.threshold_for_spot_detection = threshold_for_spot_detection
        self.save_pdf_report = save_pdf_report
        self.save_files = save_files
        self.model_nuc_segmentation=model_nuc_segmentation
        self.model_cyto_segmentation=model_cyto_segmentation
        self.pretrained_model_nuc_segmentation=pretrained_model_nuc_segmentation
        self.pretrained_model_cyto_segmentation=pretrained_model_cyto_segmentation
        
        
    def run(self):
        # Creating folder to store outputs.
        if self.save_files == True:
            output_identification_string = Utilities().create_output_folders(self.data_folder_path, self.diameter_nucleus, self.diameter_cytosol, self.psf_z, self.psf_yx, self.threshold_for_spot_detection, self.channels_spots, self.threshold_for_spot_detection)
        else:
            output_identification_string = ''
        MINIMAL_NUMBER_OF_PIXELS_IN_MASK = 1000
        # Prealocating arrays
        list_masks_complete_cells=[]
        list_masks_nuclei=[]
        list_masks_cytosol_no_nuclei=[]
        list_segmentation_successful=[]
        list_counter_image_id=[]
        list_thresholds_spot_detection =[]
        list_number_detected_cells = []
        list_average_spots_per_cell =[]
        
        # if (self.save_files is None):
        temp_folder_name = str('temp_results_'+ self.name_for_files)
        if not os.path.exists(temp_folder_name) and (self.save_files == True):
            os.makedirs(temp_folder_name)
        if (self.save_masks_as_file ==True) and (self.save_files == True):
            masks_folder_name = str('masks_'+ self.name_for_files)
            if not os.path.exists(masks_folder_name):
                os.makedirs(masks_folder_name)
        if self.save_filtered_images == True:
            filtered_folder_name = str('filtered_images_'+ self.name_for_files)
            if not os.path.exists(filtered_folder_name):
                os.makedirs(filtered_folder_name)           
        # Running the pipeline.
        counter=0
        for i in range (0, self.number_images ):
            print('')
            print( ' ###################### ' )
            print( '        IMAGE : '+ str(i) )
            print( ' ###################### ' )
            if i ==0:
                dataframe = None
            #print('- ORIGINAL IMAGE')
            print('    Image Name :  ', self.list_files_names[i])
            temp_file_name = self.list_files_names[i][:self.list_files_names[i].rfind('.')] # slcing the name of the file. Removing after finding '.' in the string.
            temp_original_img_name = pathlib.Path().absolute().joinpath( temp_folder_name, 'ori_' + temp_file_name +'.png' )
            if self.save_files == True:
                Plots().plot_images(self.list_images[i],figsize=(15, 10) ,image_name=  temp_original_img_name, show_plot = self.show_plot)            
            #print('    Image Shape :                            ', list(self.list_images[i].shape ))
            img_shape = list(self.list_images[i].shape )
            if self.remove_z_slices_borders == True:
                img_shape = list(self.list_images[i].shape )
                img_shape[0]=img_shape[0]+2*(self.NUMBER_Z_SLICES_TO_TRIM)
                print('    Orginal Image Shape :                    ', img_shape)
                print('    Trimmed z_slices at each border :        ', self.NUMBER_Z_SLICES_TO_TRIM)
            else:
                print('    Original Image Shape :                   ', img_shape)
            print('    Image sharpness metric :                 ',self.list_metric_sharpeness_images[i])
            
            if self.list_is_image_sharp[i] == False: 
                print('    Image out of focus.')
                list_segmentation_successful.append(False)
            else:
                # Cell segmentation
                temp_segmentation_img_name = pathlib.Path().absolute().joinpath( temp_folder_name, 'seg_' + temp_file_name +'.png' )
                #print('- CELL SEGMENTATION')
                if (self.masks_dir is None):
                    masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei = CellSegmentation(self.list_images[i],
                                                                                                self.channels_cytosol, 
                                                                                                self.channels_nucleus, 
                                                                                                diameter_cytosol=self.diameter_cytosol, 
                                                                                                diameter_nucleus=self.diameter_nucleus, 
                                                                                                show_plot=self.show_plot,
                                                                                                optimization_segmentation_method=self.optimization_segmentation_method,
                                                                                                image_name = temp_segmentation_img_name,
                                                                                                NUMBER_OF_CORES=self.NUMBER_OF_CORES, 
                                                                                                running_in_pipeline = True,
                                                                                                model_nuc_segmentation=self.model_nuc_segmentation,
                                                                                                model_cyto_segmentation=self.model_cyto_segmentation,
                                                                                                pretrained_model_nuc_segmentation=self.pretrained_model_nuc_segmentation,
                                                                                                pretrained_model_cyto_segmentation=self.pretrained_model_cyto_segmentation).calculate_masks() 
                # test if segmentation was succcesful
                    if Utilities().is_None(self.channels_cytosol) ==True: #(self.channels_cytosol is None):
                        detected_mask_pixels = np.count_nonzero([masks_nuclei.flatten()])
                        number_detected_cells = np.max(masks_nuclei)
                    if Utilities().is_None(self.channels_nucleus) ==True: #(self.channels_nucleus  is None):
                        detected_mask_pixels = np.count_nonzero([masks_complete_cells.flatten()])
                        number_detected_cells = np.max(masks_complete_cells)
                    if (Utilities().is_None(self.channels_nucleus) == False) and (Utilities().is_None(self.channels_cytosol) ==False):#not (self.channels_nucleus  is None) and not(self.channels_cytosol  is None):
                        detected_mask_pixels =np.count_nonzero([masks_complete_cells.flatten(), masks_nuclei.flatten(), masks_cytosol_no_nuclei.flatten()])
                        number_detected_cells = np.max(masks_complete_cells)
                    # Counting pixels
                    if  detected_mask_pixels > MINIMAL_NUMBER_OF_PIXELS_IN_MASK:
                        segmentation_successful = True
                    else:
                        segmentation_successful = False   
                else:
                    # Paths to masks
                    if Utilities().is_None(self.channels_nucleus) == False: #not (self.channels_nucleus in (None,[None])) :
                        mask_nuc_path = self.masks_dir.absolute().joinpath('masks_nuclei_' + temp_file_name +'.tif' )
                        try:
                            masks_nuclei = imread(str(mask_nuc_path)) 
                            segmentation_successful = True
                            number_detected_cells = np.max(masks_nuclei)
                        except:
                            segmentation_successful = False
                    if Utilities().is_None(self.channels_cytosol) ==False: #not (self.channels_cytosol is None):
                        mask_cyto_path = self.masks_dir.absolute().joinpath( 'masks_cyto_' + temp_file_name +'.tif' )
                        try:
                            masks_complete_cells = imread(str( mask_cyto_path   )) 
                            segmentation_successful = True
                            number_detected_cells = np.max(masks_complete_cells)
                        except:
                            segmentation_successful = False
                    if  (Utilities().is_None(self.channels_nucleus) == False) and (Utilities().is_None(self.channels_cytosol) ==False): # not (self.channels_cytosol is None) and not (self.channels_nucleus is None) :
                        mask_cyto_no_nuclei_path = self.masks_dir.absolute().joinpath('masks_cyto_no_nuclei_' + temp_file_name +'.tif' )
                        try:
                            masks_cytosol_no_nuclei = imread(str(mask_cyto_no_nuclei_path  ))
                        except:
                            segmentation_successful = False
                    # test all masks exist, if not create the variable and set as None.
                    if not 'masks_nuclei' in locals():
                        masks_nuclei=None
                    if not 'masks_complete_cells' in locals():
                        masks_complete_cells=None
                    if not 'masks_cytosol_no_nuclei' in locals():
                        masks_cytosol_no_nuclei=None
                # saving masks
                if (self.save_masks_as_file ==True) and (segmentation_successful==True) and (self.save_files == True):
                    number_detected_cells = np.max(masks_complete_cells)
                    print('    Number of detected cells:                ', number_detected_cells)
                    if Utilities().is_None(self.channels_nucleus) == False: #not (self.channels_nucleus is None):
                        mask_nuc_path = pathlib.Path().absolute().joinpath( masks_folder_name, 'masks_nuclei_' + temp_file_name +'.tif' )
                        tifffile.imwrite(mask_nuc_path, masks_nuclei)
                    if Utilities().is_None(self.channels_cytosol) ==False: #not (self.channels_cytosol is None):
                        mask_cyto_path = pathlib.Path().absolute().joinpath( masks_folder_name, 'masks_cyto_' + temp_file_name +'.tif' )
                        tifffile.imwrite(mask_cyto_path, masks_complete_cells)
                    if (Utilities().is_None(self.channels_nucleus) == False) and (Utilities().is_None(self.channels_cytosol) ==False): #not (self.channels_cytosol is None) and not (self.channels_nucleus is None):
                        mask_cyto_no_nuclei_path = pathlib.Path().absolute().joinpath( masks_folder_name, 'masks_cyto_no_nuclei_' + temp_file_name +'.tif' )
                        tifffile.imwrite(mask_cyto_no_nuclei_path, masks_cytosol_no_nuclei)
                #else:
                if segmentation_successful==False:
                    number_detected_cells = 0
                
                list_number_detected_cells.append(number_detected_cells)
                #print('- SPOT DETECTION')
                if segmentation_successful==True:
                    temp_detection_img_name = pathlib.Path().absolute().joinpath( temp_folder_name, 'det_' + temp_file_name )
                    df_detected_spots, list_images,list_thresholds_spot_detection_in_image = SpotDetection(self.list_images[i],
                                                                                            self.channels_spots,
                                                                                            self.channels_cytosol,
                                                                                            self.channels_nucleus, 
                                                                                            cluster_radius_nm=self.cluster_radius_nm,
                                                                                            minimum_spots_cluster=self.minimum_spots_cluster,
                                                                                            masks_complete_cells=masks_complete_cells,
                                                                                            masks_nuclei=masks_nuclei, 
                                                                                            masks_cytosol_no_nuclei=masks_cytosol_no_nuclei, 
                                                                                            dataframe=dataframe,
                                                                                            image_counter=counter, 
                                                                                            list_voxels=self.list_voxels,
                                                                                            yx_spot_size_in_px=self.yx_spot_size_in_px,
                                                                                            z_spot_size_in_px=self.z_spot_size_in_px,
                                                                                            show_plot=self.show_plot,
                                                                                            image_name = temp_detection_img_name,
                                                                                            save_all_images=self.save_all_images,
                                                                                            display_spots_on_multiple_z_planes=self.display_spots_on_multiple_z_planes,
                                                                                            use_log_filter_for_spot_detection=self.use_log_filter_for_spot_detection,
                                                                                            threshold_for_spot_detection=self.threshold_for_spot_detection,
                                                                                            save_files=self.save_files ).get_dataframe()
                    dataframe = df_detected_spots
                    list_masks_complete_cells.append(masks_complete_cells)
                    list_masks_nuclei.append(masks_nuclei)
                    list_masks_cytosol_no_nuclei.append(masks_cytosol_no_nuclei)
                    list_counter_image_id.append(counter)
                    list_thresholds_spot_detection.append(list_thresholds_spot_detection_in_image)
                    print('    Intensity threshold for spot detection : ', str(list_thresholds_spot_detection[-1]))
                    # Create the image with labels.
                    df_test = dataframe.loc[dataframe['image_id'] == counter]
                    test_cells_ids = np.unique(df_test['cell_id'].values)
                    # Saving the average number of spots per cell
                    list_number_of_spots_per_cell_for_each_spot_type=[]
                    list_max_number_of_spots_per_cell_for_each_spot_type=[]
                    for sp in range(len(self.channels_spots)):
                        detected_spots = np.asarray([len( dataframe.loc[  (dataframe['cell_id']==cell_id_test)  & (dataframe['spot_type']==sp) & (dataframe['is_cell_fragmented']!=-1)].spot_id) for i,cell_id_test in enumerate(test_cells_ids)])
                        average_number_of_spots_per_cell = int(np.mean(detected_spots))
                        max_number_of_spots_per_cell = int(np.max(detected_spots))
                        list_number_of_spots_per_cell_for_each_spot_type.append(average_number_of_spots_per_cell)
                        list_max_number_of_spots_per_cell_for_each_spot_type.append(max_number_of_spots_per_cell)
                    print('    Average detected spots per cell :        ', list_number_of_spots_per_cell_for_each_spot_type)
                    print('    Maximum detected spots per cell :        ', list_max_number_of_spots_per_cell_for_each_spot_type)
                    list_average_spots_per_cell.append(list_number_of_spots_per_cell_for_each_spot_type)
                    # saving Spot images
                    if self.save_filtered_images == True:
                        for j in range(len(self.channels_spots)):
                            filtered_image_path = pathlib.Path().absolute().joinpath( filtered_folder_name, 'filter_Ch_' + str(self.channels_spots[j]) +'_'+ temp_file_name +'.tif' )
                            tifffile.imwrite(filtered_image_path, list_images[j])
                    # Create the image with labels.
                    df_subset = df_detected_spots.loc[df_detected_spots['image_id'] == counter]
                    df_labels = df_subset.drop_duplicates(subset=['cell_id'])
                    # Plotting cells 
                    if self.save_files == True:
                        Plots().plotting_masks_and_original_image(image= self.list_images[i], 
                                                            masks_complete_cells=masks_complete_cells, 
                                                            masks_nuclei=masks_nuclei, 
                                                            channels_cytosol=self.channels_cytosol, 
                                                            channels_nucleus = self.channels_nucleus,
                                                            image_name=temp_segmentation_img_name,
                                                            show_plot=self.show_plot,
                                                            df_labels=df_labels)
                    del masks_complete_cells, masks_nuclei, masks_cytosol_no_nuclei, list_images,df_subset,df_labels
                    counter+=1
                # appending cell segmentation flag
                list_segmentation_successful.append(segmentation_successful)
        
        # Creating a list storing if segmenation and sharpness selection were successful
        list_processing_successful = [a and b for a, b in zip(list_segmentation_successful, self.list_is_image_sharp)]
        
        # Saving all original images as a PDF
        #print('- CREATING THE PLOT WITH ORIGINAL IMAGES')
        image_name= 'original_images_' + self.name_for_files +'.pdf'
        if self.save_files == True:
            Plots().plotting_all_original_images(self.list_images,self.list_files_names,image_name,show_plot=self.show_plot)
        # Creating an image with all segmentation results
        image_name= 'segmentation_images_' + self.name_for_files +'.pdf'
        if self.save_files == True:
            Plots().plotting_segmentation_images(directory=pathlib.Path().absolute().joinpath(temp_folder_name),
                                           list_files_names=self.list_files_names,
                                           list_segmentation_successful=list_processing_successful,
                                           image_name=image_name,
                                           show_plot=False)
        # Saving all cells in a single image file
        if self.save_files == True:
            for k in range (len(self.channels_spots)):
                Plots().plot_all_cells_and_spots(list_images=self.list_images, 
                                            complete_dataframe=dataframe, 
                                            selected_channel=self.channels_spots[k], 
                                            list_masks_complete_cells = list_masks_complete_cells,
                                            list_masks_nuclei = list_masks_nuclei,
                                            spot_type=k,
                                            list_segmentation_successful=list_processing_successful,
                                            image_name='cells_channel_'+ str(self.channels_spots[k])+'_'+ self.name_for_files +'.pdf',
                                            microns_per_pixel=None,
                                            show_legend = True,
                                            show_plot= False)
        # Creating the dataframe    
        if self.save_files == True:   
            if  (not str(self.name_for_files)[0:5] ==  'temp_') and np.sum(list_processing_successful)>0:
                dataframe.to_csv('df_' + self.name_for_files +'.csv')
            elif np.sum(list_processing_successful)>0:
                dataframe.to_csv('df_' + self.name_for_files[5:] +'.csv')        
        # Creating the metadata
        if self.save_files == True:
            Metadata(self.data_folder_path, 
                self.channels_cytosol, 
                self.channels_nucleus, 
                self.channels_spots,
                self.diameter_nucleus, 
                self.diameter_cytosol, 
                self.minimum_spots_cluster,
                list_voxels=self.list_voxels, 
                file_name_str=self.name_for_files,
                list_segmentation_successful=list_segmentation_successful,
                list_counter_image_id=list_counter_image_id,
                threshold_for_spot_detection=self.threshold_for_spot_detection,
                number_of_images_to_process=self.number_of_images_to_process,
                remove_z_slices_borders=self.remove_z_slices_borders,
                NUMBER_Z_SLICES_TO_TRIM=self.NUMBER_Z_SLICES_TO_TRIM,
                cluster_radius_nm=self.cluster_radius_nm,
                list_thresholds_spot_detection=list_thresholds_spot_detection,
                list_average_spots_per_cell=list_average_spots_per_cell,
                list_number_detected_cells=list_number_detected_cells,
                list_is_image_sharp=self.list_is_image_sharp,
                list_metric_sharpeness_images=self.list_metric_sharpeness_images,
                remove_out_of_focus_images=self.remove_out_of_focus_images,
                sharpness_threshold=self.sharpness_threshold).write_metadata()
        # Creating a PDF report
        #print('CREATING THE PDF REPORT')
        if (self.save_pdf_report ==True) and (self.save_files == True):
            filenames_for_pdf_report = [ f[:-4] for f in self.list_files_names]
            ReportPDF(directory=pathlib.Path().absolute().joinpath(temp_folder_name), 
                    filenames_for_pdf_report=filenames_for_pdf_report, 
                    channels_spots=self.channels_spots, 
                    save_all_images=self.save_all_images, 
                    list_z_slices_per_image=self.list_z_slices_per_image,
                    threshold_for_spot_detection=self.threshold_for_spot_detection,
                    list_segmentation_successful=list_processing_successful ).create_report()
        return dataframe, list_masks_complete_cells, list_masks_nuclei, list_masks_cytosol_no_nuclei, output_identification_string



class ColocalizationDistance():
    '''
    This class is intended to calculate the Euclidean 2nd norm distance between the spots detected in two spot channels.
    
    Parameters
    
    dataframe : Pandas Dataframe 
        Pandas dataframe with the following columns. image_id, cell_id, spot_id, nuc_loc_y, nuc_loc_x, cyto_loc_y, cyto_loc_x, nuc_area_px, cyto_area_px, cell_area_px, z, y, x, is_nuc, is_cluster, cluster_size, spot_type, is_cell_fragmented. 
        The default must contain spots detected in two different color channels.
    list_spot_type_to_compare : list, optional
        List indicating the combination of two values in spot_type to compare from the dataframe. The default is list_spot_type_to_compare =[0,1] indicating that spot_types 0 and 1 are compared.
    time_point : int, optional.
        Integer indicating the time point at which the data was collected. This number is displayed as a column in the final dataframe. The default value is 0.
    threshold_intensity_0 : int, optional
        Integer indicating the intensity threshold used to collected the data for the first color channel. This number is displayed as a column in the final dataframe. The default value is 0.
    threshold_intensity_1 : int, optional
        Integer indicating the intensity threshold used to collected the data for the second color channel. This number is displayed as a column in the final dataframe. The default value is 0.
    threshold_distance : float, optional.
        This number indicates the threshold distance in pixels that is used to determine if two spots are co-located in two different color channels if they are located inside this threshold_distance. The default value is 2.
    show_plot : Bool, optional.
        If true, it shows a spots on the plane below and above the selected plane. The default is False.
    voxel_size_z, voxel_size_yx: float, optional.
        These values indicate the microscope voxel size. These parameters are optional and should be included only if a normalization to the z-axis is needed to calculate distance.
    psf_z, psf_yx: float, optional.
        These values indicate the microscope point spread function value. These parameters are optional and should be included only if a normalization to the z-axis is needed to calculate distance.
    report_codetected_spots_in_both_channels : bool, optional
        This option report the number of co-detected spots in channel both channels. Notice that this represents the total number of codetected spots in ch0 and ch1. The default is True.
    '''
    def __init__(self, df,list_spot_type_to_compare =[0,1], time_point=0,threshold_intensity_0=0,threshold_intensity_1=0,threshold_distance=2,show_plot = False,voxel_size_z=None,psf_z=None,voxel_size_yx=None,psf_yx=None,report_codetected_spots_in_both_channels=False):
        self.df = df
        self.time_point= time_point
        self.threshold_intensity_0 = threshold_intensity_0
        self.threshold_intensity_1 = threshold_intensity_1
        self.threshold_distance = threshold_distance
        self.show_plot = show_plot
        self.list_spot_type_to_compare = list_spot_type_to_compare
        if not (voxel_size_z is None):
            self.scale = np.array ([ voxel_size_z/psf_z, voxel_size_yx/psf_yx, voxel_size_yx/psf_yx ])
        else:
            self.scale = 1
        self.report_codetected_spots_in_both_channels = report_codetected_spots_in_both_channels
    
    def extract_spot_classification_from_df(self):
        '''
        This method calculates the distance between the spots detected in two color channnels.
        
        Returns
        
        dataframe : Pandas dataframe
            Pandas dataframe with the following columns: [time, ts_intensity_0, ts_intensity_1, ts_distance, image_id, cell_id, num_0_only, num_1_only, num_0_1, num_0, num_1, total]. 
                num_0_only = num_type_0_only
                num_1_only = num_type_1_only
                num_0_1 = num_type_0_1
                num_0 = num_type_0_only + num_type_0_1
                num_1 = num_type_1_only + num_type_0_1
                num_0_total = total number of spots detected on ch 0.
                num_1_total = total number of spots detected on ch 1.
                total = num_type_0_only + num_type_1_only + num_type_0_1
                
        '''
        number_cells = self.df['cell_id'].nunique()
        array_spot_type_per_cell = np.zeros((number_cells, 14)).astype(int) # this array will store the spots separated  as types: spot_0_only, spot_1_only, or spot_0_1
        list_coordinates_colocalized_spots=[]
        list_coordinates_spots_0_only = []
        list_coordinates_spots_1_only = []
        list_coordinates_spots_0_all = []
        list_coordinates_spots_1_all = []

        for cell_id in range(number_cells):
            image_id = self.df[self.df["cell_id"] == cell_id]['image_id'].values[0]
            # retrieving the coordinates for spots type 0 and 1 for each cell 
            spot_type_0 = self.list_spot_type_to_compare[0] 
            spot_type_1 = self.list_spot_type_to_compare[1]
            array_spots_0 = np.asarray( self.df[['z','y','x']][(self.df["cell_id"] == cell_id) & (self.df["spot_type"] == spot_type_0)] ) # coordinates for spot_type_0 with shape [num_spots_type_0, 3]
            array_spots_1 = np.asarray( self.df[['z','y','x']][(self.df["cell_id"] == cell_id) & (self.df["spot_type"] == spot_type_1)] ) # coordinates for spot_type_1 with shape [num_spots_type_1, 3]
            total_spots0 = array_spots_0.shape[0]
            total_spots1 = array_spots_1.shape[0]            
            # Concatenating arrays from spots 0 and 1
            array_all_spots = np.concatenate((array_spots_0,array_spots_1), axis=0) 
            # Calculating a distance matrix. 
            distance_matrix = np.zeros( (array_all_spots.shape[0], array_all_spots.shape[0])) #  the distance matrix is an square matrix resulting from the concatenation of both spot  types.
            for i in range(len(array_all_spots)):
                for j in range(len(array_all_spots)):
                    if j<i:
                        distance_matrix[i,j] = np.linalg.norm( ( array_all_spots[i,:]-array_all_spots[j,:] ) * self.scale )
            # masking the distance matrix. Ones indicate the distance is less or equal than threshold_distance
            mask_distance_matrix = (distance_matrix <= self.threshold_distance) 
            # Selecting the right-lower quadrant as a subsection of the distance matrix that compares one spot type versus the other. 
            subsection_mask_distance_matrix = mask_distance_matrix[total_spots0:, 0:total_spots0].copy()
            index_true_distance_matrix = np.transpose((subsection_mask_distance_matrix==1).nonzero())
            # To calulate 0 and 1 spots only the negation (NOT) of the subsection_mask_distance_matrix is used.
            negation_subsection_mask_distance_matrix = ~subsection_mask_distance_matrix
            # creating a subdataframe containing the coordinates of colocalized spots
            colocalized_spots_in_spots0 = index_true_distance_matrix[:,1] # selecting the x-axis in [Y,X] matrix
            coordinates_colocalized_spots = array_spots_0[ colocalized_spots_in_spots0]
            #coordinates_colocalized_spots = array_spots_0[ index_true_distance_matrix[:,1]]
            column_cell_id = np.zeros((coordinates_colocalized_spots.shape[0], 1))+ cell_id # zeros column as 2D array
            coordinates_colocalized_spots = np.hstack((coordinates_colocalized_spots, column_cell_id))   # append column
            list_coordinates_colocalized_spots.append(coordinates_colocalized_spots)
            # creating a subdataframe containing the coordinates of 0_only spots
            is_spot_only_type_0 = np.all(negation_subsection_mask_distance_matrix, axis =0 ) # Testing if all the columns are ones of inv(subsection_mask_distance_matrix). Representing spot type 0. Notice that np.all(arr, axis=0) does the calculation along the columns.
            localized_spots_in_spots_0_only = (is_spot_only_type_0 > 0).nonzero() #index_false_distance_matrix[:,1] # selecting the x-axis in [Y,X] matrix for 0_only spots
            coordinates_spots_0_only = array_spots_0[ localized_spots_in_spots_0_only]
            column_cell_id_0_only = np.zeros((coordinates_spots_0_only.shape[0], 1))+ cell_id # zeros column as 2D array
            coordinates_spots_0_only = np.hstack((coordinates_spots_0_only, column_cell_id_0_only))   # append column
            list_coordinates_spots_0_only.append(coordinates_spots_0_only)
            # creating a subdataframe containing the coordinates of 1_only spots
            is_spot_only_type_1 = np.all(negation_subsection_mask_distance_matrix, axis =1 ) #  Testing if all the rows are ones of inv(subsection_mask_distance_matrix). Representing spot type 1. Notice that np.all(arr, axis=1) does the calculation along the rows.    
            localized_spots_in_spots_1_only = (is_spot_only_type_1 > 0).nonzero() # index_false_distance_matrix[:,0] # selecting the y-axis in [Y,X] matrix for 1_only spots
            coordinates_spots_1_only = array_spots_1[ localized_spots_in_spots_1_only]
            column_cell_id_1_only = np.zeros((coordinates_spots_1_only.shape[0], 1))+ cell_id # zeros column as 2D array
            coordinates_spots_1_only = np.hstack((coordinates_spots_1_only, column_cell_id_1_only))   # append column
            list_coordinates_spots_1_only.append(coordinates_spots_1_only)
            
            # creating a subdataframe containing the coordinates of all spots in channel 0
            column_cell_id_0_all = np.zeros((array_spots_0.shape[0], 1))+ cell_id # zeros column as 2D array
            coordinates_spots_0_all = np.hstack((array_spots_0, column_cell_id_0_all))   # append column
            list_coordinates_spots_0_all.append(coordinates_spots_0_all)
            # creating a subdataframe containing the coordinates of all spots in channel 1
            column_cell_id_1_all = np.zeros((array_spots_1.shape[0], 1))+ cell_id # zeros column as 2D array
            coordinates_spots_1_all = np.hstack((array_spots_1, column_cell_id_1_all))   # append column
            list_coordinates_spots_1_all.append(coordinates_spots_1_all)
            # plotting the distance matrix. True values indicate that the combination of spots are inside the minimal selected radius.
            if self.show_plot == True:
                print('Cell_Id: ', str(cell_id))
                plt.imshow(subsection_mask_distance_matrix, cmap='Greys_r')
                plt.title('Subsection bool mask distance matrix') 
                plt.xlabel('Spots 0')
                plt.ylabel('Spots 1')   
                plt.show()        
            # Calculating each type of spots in cell
            num_type_0_only = coordinates_spots_0_only.shape[0]#np.sum(is_spot_only_type_0) 
            num_type_1_only = coordinates_spots_1_only.shape[0]#np.sum(is_spot_only_type_1) 
            if self.report_codetected_spots_in_both_channels == True:
                num_type_0_1 =  (total_spots0 - num_type_0_only) + (total_spots1 - num_type_1_only) # Number of spots in both channels
                total_spots = num_type_0_only+num_type_1_only+num_type_0_1
            else:
                num_type_0_1 =  coordinates_colocalized_spots.shape[0] # This will display the number of colocalized spots only in channel 0
                total_spots = num_type_0_only+num_type_1_only+num_type_0_1
            array_spot_type_per_cell[cell_id,:] = np.array([self.time_point, 
                                                            self.threshold_intensity_0, 
                                                            self.threshold_intensity_1, 
                                                            self.threshold_distance, 
                                                            image_id, 
                                                            cell_id, 
                                                            num_type_0_only, 
                                                            num_type_1_only, 
                                                            num_type_0_1, 
                                                            num_type_0_only+num_type_0_1, 
                                                            num_type_1_only+num_type_0_1, 
                                                            total_spots0,
                                                            total_spots1,
                                                            total_spots]).astype(int)
            list_labels = ['time','ts_intensity_0','ts_intensity_1','ts_distance','image_id','cell_id','num_0_only','num_1_only','num_0_1','num_0', 'num_1','num_0_total','num_1_total','total']
            # creating a dataframe
            df_spots_classification = pd.DataFrame(data=array_spot_type_per_cell, columns=list_labels)
            del coordinates_colocalized_spots,is_spot_only_type_0,is_spot_only_type_1,coordinates_spots_0_only,coordinates_spots_1_only
        # Creating dataframes for coordinates
        list_labels_coordinates = ['z','y','x','cell_id']
        new_dtypes = { 'cell_id':int, 'z':int,'y':int,'x':int}
        # Colocalized spots
        coordinates_colocalized_spots_all_cells = np.concatenate(list_coordinates_colocalized_spots)
        df_coordinates_colocalized_spots = pd.DataFrame(data=coordinates_colocalized_spots_all_cells, columns=list_labels_coordinates)
        df_coordinates_colocalized_spots = df_coordinates_colocalized_spots.astype(new_dtypes)
        # 0-only spots
        coordinates_0_only_spots_all_cells = np.concatenate(list_coordinates_spots_0_only)
        df_coordinates_0_only_spots = pd.DataFrame(data=coordinates_0_only_spots_all_cells, columns=list_labels_coordinates)
        df_coordinates_0_only_spots = df_coordinates_0_only_spots.astype(new_dtypes)
        # 1-only spots
        coordinates_1_only_spots_all_cells = np.concatenate(list_coordinates_spots_1_only)
        df_coordinates_1_only_spots = pd.DataFrame(data=coordinates_1_only_spots_all_cells, columns=list_labels_coordinates)
        df_coordinates_1_only_spots = df_coordinates_1_only_spots.astype(new_dtypes)
        
        # 0 all spots
        coordinates_0_all_spots_all_cells = np.concatenate(list_coordinates_spots_0_all)
        df_coordinates_0_all_spots = pd.DataFrame(data=coordinates_0_all_spots_all_cells, columns=list_labels_coordinates)
        df_coordinates_0_all_spots = df_coordinates_0_all_spots.astype(new_dtypes)
        # 1 all spots
        coordinates_1_all_spots_all_cells = np.concatenate(list_coordinates_spots_1_all)
        df_coordinates_1_all_spots = pd.DataFrame(data=coordinates_1_all_spots_all_cells, columns=list_labels_coordinates)
        df_coordinates_1_all_spots = df_coordinates_1_all_spots.astype(new_dtypes)

        return df_spots_classification, df_coordinates_colocalized_spots, df_coordinates_0_only_spots, df_coordinates_1_only_spots, df_coordinates_0_all_spots, df_coordinates_1_all_spots


class PointSpreadFunction():
    """
    Create a croparray and mean_crop from the given image and dataframe of crops.

    Parameters:
    - image (ndarray): The input image array. The shape should be (frames, z, y, x, channels). TZYXC.
    - df_crops (DataFrame): The dataframe containing information about the crops.
    - crop_size (int): The size of the crop.
    - remove_outliers (bool): Flag indicating whether to remove outliers from the croparray. Default is True.
    - max_percentile (float): The percentile value to clip the croparray and mean_crop. Default is 98.5.
    - min_percentile (float): The percentile value to clip the croparray and mean_crop. Default is 1.5. 
    - show_plot (bool): Flag indicating whether to show the plot. Default is False.
    - plot_name (str): The name of the plot. Default is 'temp.png'.
    - save_plots (bool): Flag indicating whether to save the plots. Default is False.
    - selected_color_channel (int): The selected color channel to calculate the PSF. Default is None.
    """
    def __init__(self, image, df_crops, crop_size_xy=5, crop_size_z=3, remove_outliers=True, 
                 selected_color_channel=None, min_percentile=0.5, max_percentile=99, show_plot=False, 
                 plot_name='temp.png', save_plots=False):
        self.half_spot_xy = crop_size_xy // 2
        self.half_spot_z = crop_size_z // 2
        self.crop_size_xy = crop_size_xy
        self.crop_size_z = crop_size_z
        self.number_particles = len(df_crops['particle'].unique())
        self.df_crops = df_crops
        self.max_frame = df_crops['frame'].max()
        self.image = image
        self.number_color_channels = image.shape[-1]
        self.remove_outliers = remove_outliers
        self.max_percentile = max_percentile
        self.min_percentile = min_percentile
        self.show_plot = show_plot
        self.plot_name = plot_name
        self.save_plots = save_plots
        self.selected_color_channel = selected_color_channel

    def gaussian_3d(self, coords, amplitude, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset):
        x, y, z = coords
        exponent = (
            ((x - x0) ** 2) / (2 * sigma_x ** 2) +
            ((y - y0) ** 2) / (2 * sigma_y ** 2) +
            ((z - z0) ** 2) / (2 * sigma_z ** 2)
        )
        return amplitude * np.exp(-exponent) + offset
    
    def fit_3D_gaussian(self, data):
        # Generate x, y, z coordinates
        z_size, y_size, x_size = data.shape
        z, y, x = np.indices((z_size, y_size, x_size))
        # Flatten the data and coordinates
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        data = data.ravel()

        # Mask out zero values (padded regions)
        mask = data > 0  # Exclude zero values
        x = x[mask]
        y = y[mask]
        z = z[mask]
        data = data[mask]
        coords = np.vstack((x, y, z))

        # Estimate initial parameters from data
        amplitude_initial = data.max() - data.min()
        offset_initial = data.min()
        x0_initial = np.sum(x * data) / np.sum(data)
        y0_initial = np.sum(y * data) / np.sum(data)
        z0_initial = np.sum(z * data) / np.sum(data)

        # Estimate initial sigma
        sigma_x_initial = np.sqrt(np.sum(data * (x - x0_initial)**2) / np.sum(data))
        sigma_y_initial = np.sqrt(np.sum(data * (y - y0_initial)**2) / np.sum(data))
        sigma_z_initial = np.sqrt(np.sum(data * (z - z0_initial)**2) / np.sum(data))

        initial_guess = (
            amplitude_initial,  # amplitude
            x0_initial,         # x0
            y0_initial,         # y0
            z0_initial,         # z0
            sigma_x_initial,    # sigma_x
            sigma_y_initial,    # sigma_y
            sigma_z_initial,    # sigma_z
            offset_initial      # offset
        )

        # Set bounds for parameters
        lower_bounds = [0, 0, 0, 0, 0.1, 0.1, 0.1, -np.inf]
        upper_bounds = [np.inf, x_size, y_size, z_size, self.crop_size_xy, self.crop_size_xy, self.crop_size_z, np.inf]

        # Fit the data
        try:
            popt, _ = curve_fit(
                self.gaussian_3d, coords, data, p0=initial_guess, bounds=(lower_bounds, upper_bounds)
            )
            fit_result = {
                'amplitude': popt[0],
                'x0': popt[1],
                'y0': popt[2],
                'z0': popt[3],
                'sigma_x': popt[4],
                'sigma_y': popt[5],
                'sigma_z': popt[6],
                'offset': popt[7]
            }
        except RuntimeError:
            fit_result = {
                'amplitude': np.nan,
                'x0': np.nan,
                'y0': np.nan,
                'z0': np.nan,
                'sigma_x': np.nan,
                'sigma_y': np.nan,
                'sigma_z': np.nan,
                'offset': np.nan
            }
        return fit_result

    def calculate(self):
        def calculate_particle_size(sigma_x, sigma_y, sigma_z):
            """
            Calculate the particle size based on the sigma values from the PSF.
            
            Parameters:
            sigma_x (float): Sigma value in the x-axis.
            sigma_y (float): Sigma value in the y-axis.
            sigma_z (float, optional): Sigma value in the z-axis for 3D calculation.
            
            Returns:
            dict: A dictionary containing FWHM values and particle size.
            """
            
            # Convert sigma to FWHM using the formula: FWHM = 2.355 * sigma
            FWHM_x = 2.355 * sigma_x
            FWHM_y = 2.355 * sigma_y
            if sigma_z is not None:
                FWHM_z = 2.355 * sigma_z
            else:
                FWHM_z = None
            return FWHM_x,FWHM_y,FWHM_z
        # Initialize variables
        list_mean_temporal_crops = [[] for _ in range(self.number_color_channels)]
        list_average_crop = []
        list_all_crops = [[[[] for _ in range(self.max_frame + 1)] for _ in range(self.number_color_channels)] for _ in range(self.number_particles)]
        
        for particle_idx, particle_id in enumerate(self.df_crops['particle'].unique()):
            df_particle = self.df_crops[self.df_crops['particle'] == particle_id]
            list_crops = [[] for _ in range(self.number_color_channels)]
            
            for i in range(self.max_frame + 1):
                df_frame = df_particle[df_particle['frame'] == i]
                if len(df_frame) > 0:
                    x_center, y_center, z_center = int(df_frame['x']), int(df_frame['y']), int(df_frame['z'])
                else:
                    x_center, y_center, z_center = None, None, None
                if x_center is not None and y_center is not None and z_center is not None:
                    x_start = max(0, x_center - self.half_spot_xy)
                    x_end = min(self.image.shape[3], x_center + self.half_spot_xy + 1)
                    y_start = max(0, y_center - self.half_spot_xy)
                    y_end = min(self.image.shape[2], y_center + self.half_spot_xy + 1)
                    z_start = max(0, z_center - self.half_spot_z)
                    z_end = min(self.image.shape[1], z_center + self.half_spot_z + 1)
                    for ch in range(self.number_color_channels):
                        crop = self.image[i, z_start:z_end, y_start:y_end, x_start:x_end, ch]
                        # Calculate padding
                        pad_z_before = max(0, self.half_spot_z - (z_center - z_start))
                        pad_z_after = max(0, self.crop_size_z - crop.shape[0] - pad_z_before)
                        pad_y_before = max(0, self.half_spot_xy - (y_center - y_start))
                        pad_y_after = max(0, self.crop_size_xy - crop.shape[1] - pad_y_before)
                        pad_x_before = max(0, self.half_spot_xy - (x_center - x_start))
                        pad_x_after = max(0, self.crop_size_xy - crop.shape[2] - pad_x_before)

                        pad_width = (
                            (pad_z_before, pad_z_after),
                            (pad_y_before, pad_y_after),
                            (pad_x_before, pad_x_after)
                        )
                        # Apply padding
                        crop_padded = np.pad(crop, pad_width, mode='constant', constant_values=0)
                        list_crops[ch].append(crop_padded)
                        list_all_crops[particle_idx][ch][i].append(crop_padded)
            for ch in range(self.number_color_channels):
                if list_crops[ch]:
                    stacked_arrays = np.stack(list_crops[ch], axis=0)  # [N_crops, Z, Y, X]
                    mean_array = np.mean(stacked_arrays, axis=0)       # [Z, Y, X]
                    list_mean_temporal_crops[ch].append(mean_array)
        for ch in range(self.number_color_channels):
            if list_mean_temporal_crops[ch]:
                list_average_crop.append(np.mean(list_mean_temporal_crops[ch], axis=0))
        # Fitting 3D Gaussians
        list_amplitude = []
        list_sigma_x = []
        list_sigma_y = []
        list_sigma_z = []
        list_offset = []
        if self.selected_color_channel is not None:
            number_color_channels = 1
            list_mean_temporal_crops = [list_mean_temporal_crops[self.selected_color_channel]]
        else:
            number_color_channels = self.number_color_channels
        for ch in range(number_color_channels):
            number_crops = len(list_mean_temporal_crops[ch])
            amplitude_vector = np.zeros(number_crops)
            sigma_x_vector = np.zeros(number_crops)
            sigma_y_vector = np.zeros(number_crops)
            sigma_z_vector = np.zeros(number_crops)
            offset_vector = np.zeros(number_crops)
            for i in range(number_crops):
                selected_crop = list_mean_temporal_crops[ch][i]
                try:
                    fit_result = self.fit_3D_gaussian(selected_crop)
                    amplitude_vector[i] = fit_result['amplitude']
                    sigma_x_vector[i] = fit_result['sigma_x']
                    sigma_y_vector[i] = fit_result['sigma_y']
                    sigma_z_vector[i] = fit_result['sigma_z']
                    offset_vector[i] = fit_result['offset']
                except:
                    amplitude_vector[i] = np.nan
                    sigma_x_vector[i] = np.nan
                    sigma_y_vector[i] = np.nan
                    sigma_z_vector[i] = np.nan
                    offset_vector[i] = np.nan
            # Remove NaNs
            amplitude_vector = amplitude_vector[~np.isnan(amplitude_vector)]
            sigma_x_vector = sigma_x_vector[~np.isnan(sigma_x_vector)]
            sigma_y_vector = sigma_y_vector[~np.isnan(sigma_y_vector)]
            sigma_z_vector = sigma_z_vector[~np.isnan(sigma_z_vector)]
            offset_vector = offset_vector[~np.isnan(offset_vector)]
            # Remove outliers based on percentiles
            if self.remove_outliers and amplitude_vector.size > 0:
                # Amplitude
                amp_lower = np.percentile(amplitude_vector, self.min_percentile)
                amp_upper = np.percentile(amplitude_vector, self.max_percentile)
                amplitude_vector = amplitude_vector[(amplitude_vector >= amp_lower) & (amplitude_vector <= amp_upper)]
                # Sigma X
                sigma_x_lower = np.percentile(sigma_x_vector, self.min_percentile)
                sigma_x_upper = np.percentile(sigma_x_vector, self.max_percentile)
                sigma_x_vector = sigma_x_vector[(sigma_x_vector >= sigma_x_lower) & (sigma_x_vector <= sigma_x_upper)]
                # Sigma Y
                sigma_y_lower = np.percentile(sigma_y_vector, self.min_percentile)
                sigma_y_upper = np.percentile(sigma_y_vector, self.max_percentile)
                sigma_y_vector = sigma_y_vector[(sigma_y_vector >= sigma_y_lower) & (sigma_y_vector <= sigma_y_upper)]
                # Sigma Z
                sigma_z_lower = np.percentile(sigma_z_vector, self.min_percentile)
                sigma_z_upper = np.percentile(sigma_z_vector, self.max_percentile)
                sigma_z_vector = sigma_z_vector[(sigma_z_vector >= sigma_z_lower) & (sigma_z_vector <= sigma_z_upper)]
                # Offset
                offset_lower = np.percentile(offset_vector, self.min_percentile)
                offset_upper = np.percentile(offset_vector, self.max_percentile)
                offset_vector = offset_vector[(offset_vector >= offset_lower) & (offset_vector <= offset_upper)]
            list_amplitude.append(amplitude_vector)
            list_sigma_x.append(sigma_x_vector)
            list_sigma_y.append(sigma_y_vector)
            list_sigma_z.append(sigma_z_vector)
            list_offset.append(offset_vector)
        if self.show_plot:
            Plots().plot_pixel_properties(list_amplitude, list_sigma_x, list_sigma_y, list_sigma_z, list_offset,
                                       self.crop_size_xy, plot_name=self.plot_name, save_plots=self.save_plots)
        list_FWHM_x = []
        list_FWHM_y = []
        list_FWHM_z = []
        # Calculate particle size in both 2D and 3D (if sigma_z is provided)
        for ch in range(number_color_channels): 
            sigma_x = np.mean(list_sigma_x[ch])
            sigma_y = np.mean(list_sigma_y[ch])
            sigma_z = np.mean(list_sigma_z[ch]) #if list_sigma_z[ch].size > 0 else None

            FWHM_x,FWHM_y,FWHM_z = calculate_particle_size(sigma_x, sigma_y, sigma_z)
            list_FWHM_x.append(FWHM_x)
            list_FWHM_y.append(FWHM_y)
            list_FWHM_z.append(FWHM_z)
        result = {
            'mean_temporal_crops': list_mean_temporal_crops,
            'all_crops': list_all_crops,
            'average_crop': list_average_crop,
            'amplitude': list_amplitude,
            'sigma_x': list_sigma_x,
            'sigma_y': list_sigma_y,
            'sigma_z': list_sigma_z,
            'offset': list_offset,
            'FWHM_x': list_FWHM_x,
            'FWHM_y': list_FWHM_y,
            'FWHM_z': list_FWHM_z
        }
        return result


# class Correlation:
#     """
#     A class for calculating the autocorrelation or cross-correlation of datasets.

#     Attributes:
#         primary_data (np.ndarray): Primary dataset for autocorrelation, shape [sample, time].
#         secondary_data (np.ndarray, optional): Secondary dataset for cross-correlation, same shape as primary.
#         max_lag (int, optional): Maximum lag to compute correlation, defaults to half time series length.
#         nan_handling (str, optional): Strategy to handle NaN values. Options: 'zeros', 'mean', 'forward_fill', 'ignore'.
#         return_full (bool, optional): Whether to return the full correlation array or only positive lags.
#         use_bootstrap (bool, optional): Whether to use bootstrap for error estimation.
#         shift_data (bool, optional): Whether to shift data based on leading NaNs.
#         show_plot (bool, optional): Whether to display the plot.
#         save_plots (bool, optional): Whether to save the plots.
#         plot_name (str, optional): Name of the plot file.
#         time_interval_between_frames_in_seconds (int, optional): Time interval between frames.
#         index_max_lag_for_fit (int, optional): Index for maximum lag for fitting.
#         color_channel (int, optional): Color channel for plotting.
#         start_lag (int, optional): Starting lag for plateau finding.
#         line_color (str, optional): Color of the plot line.
#         plot_title (str, optional): Title of the plot.
#         fit_type (str, optional): Type of fit for the plot.
#         de_correlation_threshold (float, optional): Threshold for decorrelation.
#         use_linear_projection_for_lag_0 (bool, optional): Whether to use linear projection for lag 0.
#         correct_baseline (bool, optional): If True, subtract baseline from mean correlation.
#         use_global_mean (bool, optional): If True, use a global mean for correlation normalization.
#         use_normalization_factor (bool, optional): If True, multiply correlation by 1/(global_means * #time_points).
#         remove_outliers (bool, optional): If True, remove “extreme” outlier trajectories before computing mean.
#         MAD_THRESHOLD_FACTOR (float, optional): Threshold factor for outlier removal.

#     Methods:
#         run():
#             Executes the correlation computation based on initialized settings.

#             Returns:
#                 (mean_correlation, error_correlation, lags, correlations_array, dwell_time)
#                 where
#                   mean_correlation      : 1D np.ndarray of shape [2*max_lag + 1] (or shorter if return_full=False)
#                   error_correlation     : same shape as mean_correlation
#                   lags                  : 1D np.ndarray with lag values in seconds
#                   correlations_array    : shape [N, 2*max_lag + 1] with one correlation per trajectory (N=filtered).
#                   dwell_time            : from your plotting/fitting method
#     """
#     def __init__(
#         self,
#         primary_data,
#         secondary_data=None,
#         max_lag=None,
#         nan_handling='zeros',
#         return_full=True,
#         use_bootstrap=True,
#         shift_data=False,
#         show_plot=False,
#         save_plots=False,
#         plot_name='temp_AC.png',
#         time_interval_between_frames_in_seconds=1,
#         index_max_lag_for_fit=None,
#         color_channel=0,
#         start_lag=0,
#         line_color='blue',
#         correct_baseline=False,
#         baseline_offset = None,
#         use_global_mean=False,
#         plot_title=None,
#         fit_type='linear',
#         de_correlation_threshold=0.01,
#         use_linear_projection_for_lag_0=True,
#         normalize_plot_with_g0=False,
#         remove_outliers=True,
#         MAD_THRESHOLD_FACTOR = 6.0,
#         plot_individual_trajectories=False,
#         y_axes_min_max_list_values = None,
#         x_axes_min_max_list_values=None,

        
#     ):
#         def shift_and_fill(data1, data2=None, min_nan_threshold=3, fill_with_nans=True):
#             """
#             Processes two 1D NumPy arrays by removing leading NaNs that exceed a given threshold,
#             then shifts both arrays left using the shift determined from the first array, and fills
#             the rightmost part with NaNs or zeros to maintain the original shape.
#             """
#             if data1.ndim != 1:
#                 raise ValueError("Both data1 and data2 must be 1D arrays.")

#             nan_count = 0
#             for value in data1:
#                 if np.isnan(value):
#                     nan_count += 1
#                 else:
#                     break
#             if nan_count >= min_nan_threshold:
#                 fill_value = np.nan if fill_with_nans else 0

#                 new_data1 = np.full_like(data1, fill_value)
#                 new_data1[: len(data1) - nan_count] = data1[nan_count:]

#                 if data2 is not None:
#                     new_data2 = np.full_like(data2, fill_value)
#                     new_data2[: len(data2) - nan_count] = data2[nan_count:]
#                 else:
#                     new_data2 = None
#                 return new_data1, new_data2
#             return data1, data2
#         if shift_data:
#             primary_data_shifted = np.zeros_like(primary_data)
#             if secondary_data is not None:
#                 secondary_data_shifted = np.zeros_like(secondary_data)
#             else:
#                 secondary_data_shifted = None

#             for i in range(primary_data.shape[0]):
#                 if secondary_data is None:
#                     primary_data_shifted[i, :], _ = shift_and_fill(
#                         primary_data[i, :], None, min_nan_threshold=2
#                     )
#                 else:
#                     primary_data_shifted[i, :], secondary_data_shifted[i, :] = shift_and_fill(
#                         primary_data[i, :], secondary_data[i, :], min_nan_threshold=2
#                     )
#             primary_data = primary_data_shifted
#             if secondary_data is not None:
#                 secondary_data = secondary_data_shifted

#         # Store attributes
#         self.primary_data = primary_data
#         self.secondary_data = secondary_data
#         self.max_lag = max_lag
#         self.nan_handling = nan_handling
#         self.return_full = return_full
#         self.use_bootstrap = use_bootstrap
#         self.BOOTSTRAP_ITERATIONS = 1000
#         self.time_interval_between_frames_in_seconds = float(time_interval_between_frames_in_seconds)
#         self.index_max_lag_for_fit = index_max_lag_for_fit
#         self.plot_name = plot_name
#         self.save_plots = save_plots
#         self.show_plot = show_plot
#         self.color_channel = color_channel
#         self.start_lag = start_lag
#         self.line_color = line_color
#         self.plot_title = plot_title
#         self.fit_type = fit_type
#         self.de_correlation_threshold = de_correlation_threshold
#         self.use_linear_projection_for_lag_0 = use_linear_projection_for_lag_0
#         self.normalize_plot_with_g0 = normalize_plot_with_g0
#         self.correct_baseline = correct_baseline
#         if baseline_offset is None:
#             self.baseline_offset=int(primary_data.shape[1]//2)  # offset for baseline correction it uses only half of the time series. To avoid the effect of large variations at the end of time serises.
#         else :
#             self.baseline_offset = baseline_offset
#         self.use_global_mean = use_global_mean
#         if correct_baseline:
#             plot_individual_trajectories = False
#             print('Baseline correction is enabled. Plotting individual trajectories is disabled due to baseline correction.')
#         self.remove_outliers = remove_outliers
#         self.MAD_THRESHOLD_FACTOR= MAD_THRESHOLD_FACTOR
#         self.plot_individual_trajectories = plot_individual_trajectories
#         self.y_axes_min_max_list_values = y_axes_min_max_list_values
#         self.x_axes_min_max_list_values = x_axes_min_max_list_values

#     def run(self):
#         """
#         Execute the correlation calculations with optional bootstrap error estimation.

#         Returns:
#             mean_correlation, error_correlation, lags, correlations_array, dwell_time
#         """
#         if self.max_lag is None:
#             self.max_lag = self.primary_data.shape[1] - 1
#         else:
#             if self.max_lag >= self.primary_data.shape[1]:
#                 raise ValueError("Max lag cannot be greater than the length of the time series.")
#         def find_plateau(correlation, threshold=0.005, start_lag=2):
#             differences = np.abs(np.diff(correlation[start_lag:]))
#             plateau_index = np.where(differences < threshold)[0]
#             if len(plateau_index) > 0:
#                 return plateau_index[0] + start_lag
#             return None

#         def trim_nans_from_edges(data):
#             mask = ~np.isnan(data)
#             if not np.any(mask):
#                 return np.array([])
#             start_idx = np.argmax(mask)
#             end_idx = len(mask) - np.argmax(mask[::-1])
#             return data[start_idx:end_idx]

#         # We’ll define a correlation_function that returns the correlation array
#         def correlation_function(primary_data, secondary_data):
#             """
#             Return an array of shape [N, 2*max_lag+1], where N = number of trajectories.
#             """
#             num_samples = primary_data.shape[0]
#             global_mean_data1 = np.nanmean(primary_data)
#             if secondary_data is not None:
#                 global_mean_data2 = np.nanmean(secondary_data)
#             else:
#                 global_mean_data2 = global_mean_data1

#             if self.nan_handling == "forward_fill":
#                 def forward_fill(data):
#                     not_nan = ~np.isnan(data)
#                     if not np.any(not_nan):
#                         return np.array([])
#                     first_valid_index = np.argmax(not_nan)
#                     last_valid_index = len(data) - np.argmax(not_nan[::-1]) - 1
#                     trimmed_data = data[first_valid_index:last_valid_index + 1]
#                     mask_ = np.isnan(trimmed_data)
#                     idx = np.where(~mask_, np.arange(len(trimmed_data)), 0)
#                     np.maximum.accumulate(idx, out=idx)
#                     filled_data = trimmed_data[idx]
#                     result = np.full_like(data, np.nan)
#                     result[first_valid_index:last_valid_index + 1] = filled_data
#                     return result
#                 local_forward_fill = forward_fill
#             else:
#                 local_forward_fill = lambda arr: arr

#             def process_sample(i):
#                 try:
#                     # Retrieve the time series for sample i.
#                     data1 = primary_data[i, :]
#                     data2 = secondary_data[i, :] if secondary_data is not None else data1

#                     # Trim edges using the provided function.
#                     data1 = trim_nans_from_edges(data1)
#                     data2 = trim_nans_from_edges(data2)

#                     # Handle NaNs based on the chosen strategy.
#                     if self.nan_handling == "mean":
#                         mean_val1 = np.nanmean(data1) if len(data1) > 0 else 0.0
#                         mean_val2 = np.nanmean(data2) if len(data2) > 0 else 0.0
#                         data1 = np.nan_to_num(data1, nan=mean_val1)
#                         data2 = np.nan_to_num(data2, nan=mean_val2)
#                     elif self.nan_handling == "forward_fill":
#                         data1 = local_forward_fill(data1)
#                         data2 = local_forward_fill(data2)
#                     elif self.nan_handling == "ignore":
#                         valid_mask = ~np.isnan(data1) & ~np.isnan(data2)
#                         data1 = data1[valid_mask]
#                         data2 = data2[valid_mask]
#                     elif self.nan_handling == "zeros":
#                         data1 = np.nan_to_num(data1)
#                         data2 = np.nan_to_num(data2)

#                     effective_number_time_points = np.sum(~np.isnan(data1))
#                     if effective_number_time_points < 1:
#                         return np.full(2 * self.max_lag + 1, np.nan)
#                     # Center the data using the full-sample mean.
#                     if self.use_global_mean:
#                         local_mean1 = np.nanmean(primary_data)
#                         local_mean2 = np.nanmean(secondary_data) if secondary_data is not None else local_mean1
#                     else:
#                         local_mean1 = np.nanmean(data1)
#                         local_mean2 = np.nanmean(data2)
#                     cdata1 = data1 - local_mean1
#                     cdata2 = data2 - local_mean2
#                     # Remove any residual NaNs.
#                     cdata1 = cdata1[~np.isnan(cdata1)]
#                     cdata2 = cdata2[~np.isnan(cdata2)]
#                     if len(cdata1) == 0 or len(cdata2) == 0:
#                         return np.full(2 * self.max_lag + 1, np.nan)
#                     # Compute the raw cross-correlation.
#                     raw_corr = np.correlate(cdata1, cdata2, mode="full")
#                     N = len(cdata1)
#                     mid = N - 1
#                     # Define a minimum overlap threshold (e.g. at least 5% of N must overlap)
#                     min_overlap = max(5, int(0.2 * N))
#                     final_corr = np.empty_like(raw_corr, dtype=np.float64)
#                     for j in range(len(raw_corr)):
#                         lag = j - mid
#                         overlap = N - abs(lag)
#                         if overlap < min_overlap:
#                             final_corr[j] = np.nan
#                             continue
#                         # For local normalization, compute means over the overlapping region
#                         if lag >= 0:
#                             local_seg1 = data1[:N - lag]
#                             local_seg2 = data2[lag:]
#                         else:
#                             local_seg1 = data1[-lag:]
#                             local_seg2 = data2[:N + lag]
#                         local_norm = np.nanmean(local_seg1) * np.nanmean(local_seg2)
#                         # Avoid division by zero
#                         if local_norm == 0:
#                             final_corr[j] = np.nan
#                         else:
#                             final_corr[j] = (raw_corr[j] / overlap) / local_norm
#                     # Slice the final correlation array to length 2*self.max_lag+1.
#                     mid_point = len(final_corr) // 2
#                     desired_length = 2 * self.max_lag + 1
#                     current_length = len(final_corr)
#                     if current_length < desired_length:
#                         out_array = np.full(desired_length, np.nan)
#                         start_idx = (desired_length - current_length) // 2
#                         out_array[start_idx:start_idx + current_length] = final_corr
#                         return out_array
#                     else:
#                         start_idx = mid_point - self.max_lag
#                         end_idx = mid_point + self.max_lag + 1
#                         return final_corr[start_idx:end_idx]
#                 except Exception as e:
#                     print(f"Error in process_sample for sample {i}: {e}")
#                     return np.full(2 * self.max_lag + 1, np.nan)


#             # Run the correlations in parallel
#             correlations = Parallel(n_jobs=-1)(
#                 delayed(process_sample)(i) for i in range(num_samples)
#             )
#             return np.array(correlations, dtype=np.float64)
#         correlations_array = correlation_function(self.primary_data, self.secondary_data)
#         # Removing outliers
#         if self.remove_outliers and correlations_array.size > 0:
#             # Compute the mean correlation for each trajectory (ignoring NaNs)
#             traj_means = np.nanmean(correlations_array, axis=1)
#             # Compute the median and the median absolute deviation (MAD)
#             median_mean = np.nanmedian(traj_means)
#             mad = np.nanmedian(np.abs(traj_means - median_mean))
#             # If MAD is zero, fallback to not removing any (or use a small constant)
#             if mad == 0:
#                 keep_mask = np.ones_like(traj_means, dtype=bool)
#             else:
#                 # Remove trajectories whose mean is far from the median:
#                 # keeping only those within threshold_factor * MAD.
#                 keep_mask = np.abs(traj_means - median_mean) < self.MAD_THRESHOLD_FACTOR * mad
#             num_removed = np.sum(~keep_mask)
#             num_total = len(traj_means)
#             if num_removed > 0:
#                 print(f"Warning: Removed {num_removed} outlier trajectories (out of {num_total}) based on a threshold of {self.MAD_THRESHOLD_FACTOR} MAD from the median mean correlation. These trajectories had unusually low or high mean values, which may indicate unreliable data.")
#             # Filter the correlations_array accordingly.
#             correlations_array = correlations_array[keep_mask, :]

#         # If everything got removed, or no valid data
#         if correlations_array.shape[0] == 0:
#             # Return empty arrays
#             mean_correlation = np.full(2 * self.max_lag + 1, np.nan)
#             error_correlation = np.full(2 * self.max_lag + 1, np.nan)
#             lags = np.arange(-self.max_lag, self.max_lag + 1) * self.time_interval_between_frames_in_seconds
#             return mean_correlation, error_correlation, lags, correlations_array, None
#         # ---------------------------------------------------
#         mean_correlation = np.nanmean(correlations_array, axis=0)
#         def exp_func(t, A, tau, B):
#             return A * np.exp(-t / tau) + B
#         if self.correct_baseline:
#             # Assume mean_correlation is the positive-lag portion (index 0 corresponds to lag 0)
#             L = len(mean_correlation) - 1  # Total number of lags (excluding lag 0, L+1 elements total)
#             # Define the fit region: use data from frame 1 up to 75% of the maximum lag.
#             start_idx_fit = 2
#             time_range = int(L * 0.99)
#             # Ensure time_range > start_idx
#             if time_range <= start_idx_fit:
#                 time_range = start_idx_fit + 1
#             y_fit = mean_correlation[start_idx_fit:time_range]
#             # Construct time vector for the fit, using the time interval between frames.
#             t_fit = np.arange(start_idx_fit, time_range) * self.time_interval_between_frames_in_seconds
#             # Set initial guesses:
#             B_guess = y_fit[-1]
#             # Guess A so that at t = 0, f(0)=A+B is equal to the original value at frame 0.
#             A_guess = mean_correlation[0] - B_guess
#             # Guess tau as half the time span of the fit region.
#             tau_guess = (t_fit[-1] - t_fit[0]) / 2.0 if (t_fit[-1] - t_fit[0]) > 0 else 1.0
#             initial_guess = [A_guess, tau_guess, B_guess]
#             # Set parameter bounds:
#             # A must be non-negative (if we assume a decreasing function), tau > 0, and B is between the minimum of y_fit and the frame 0 value.
#             lower_bounds = [0, 1e-6, np.min(y_fit)]
#             upper_bounds = [np.inf, np.inf, mean_correlation[0]]

#             mask_array = np.isfinite(t_fit) & np.isfinite(y_fit)
#             t_clean = t_fit[mask_array]
#             y_clean = y_fit[mask_array]
#             if t_clean.size < 3:
#                 warnings.warn(f"Too few valid points ({t_clean.size}) for exponential fit—using fallback.")
#                 fitted_B = np.nanpercentile(y_fit, 10)
#             else:
#                 try:
#                     popt, pcov = curve_fit(
#                         exp_func,
#                         t_clean,
#                         y_clean,
#                         p0=initial_guess,
#                         bounds=(lower_bounds, upper_bounds),
#                         # allow more iterations in tough cases
#                         maxfev=10_000
#                     )
#                     fitted_A, fitted_tau, fitted_B = popt
#                     print(f"Exponential fit parameters: A={fitted_A:.3g}, τ={fitted_tau:.3g}, B={fitted_B:.3g}")
#                 except ValueError as ve:
#                     # this will catch “array must not contain infs or NaNs” (and other ValueErrors)
#                     warnings.warn(f"Exp fit ValueError: {ve}  → using 10th percentile fallback.")
#                     fitted_B = np.nanpercentile(y_fit, 10)
#                 except Exception as e:
#                     warnings.warn(f"Exp fit failed ({type(e).__name__}): {e}  → using 10th percentile fallback.")
#                     fitted_B = np.nanpercentile(y_fit, 10)
#             # Subtract the estimated plateau value from the entire mean correlation.
#             mean_correlation = mean_correlation - fitted_B
#         num_kept = correlations_array.shape[0]
#         if self.use_bootstrap and num_kept > 1:
#             def single_bootstrap_iteration(_):
#                 # Create a new random generator for this iteration
#                 rng = np.random.default_rng()  
#                 indices = rng.choice(num_kept, size=num_kept, replace=True)
#                 sample = correlations_array[indices, :]  # shape [num_kept, 2*max_lag+1]
#                 # Compute mean for that sample
#                 m = np.nanmean(sample, axis=0)
#                 if self.correct_baseline:
#                     center_idx = self.max_lag
#                     offset = min(self.baseline_offset, center_idx)
#                     neg_region = m[center_idx - offset : center_idx]
#                     pos_region = m[center_idx + 1 : center_idx + 1 + offset]
#                     baseline_value = np.nanpercentile(np.concatenate([neg_region, pos_region]), 10)
#                     m = m - baseline_value
#                 return m

#             all_means = Parallel(n_jobs=-1)(
#                 delayed(single_bootstrap_iteration)(_) for _ in range(self.BOOTSTRAP_ITERATIONS)
#             )
#             all_means = np.array(all_means, dtype=np.float64)  # shape [BOOTSTRAP_ITERATIONS, 2*max_lag+1]
#             # Standard deviation across bootstrap draws (standard error of the mean)
#             error_correlation = np.nanstd(all_means, axis=0)
#         else:
#             error_correlation = np.nanstd(correlations_array, axis=0)/ np.sqrt(num_kept)
#         lags = np.arange(-self.max_lag, self.max_lag + 1) * self.time_interval_between_frames_in_seconds
#         if self.use_linear_projection_for_lag_0:
#             center_idx = self.max_lag
#             # For autocorrelation
#             if self.secondary_data is None:
#                 if center_idx - 6 >= 0 and center_idx - 1 >= 0:
#                     x = lags[center_idx - 6 : center_idx - 1]
#                     y = mean_correlation[center_idx - 5 : center_idx]
#                     slope, intercept, _, _, _ = linregress(x, y)
#                     mean_correlation[center_idx] = intercept  # slope*0 + intercept
#                 if center_idx < len(error_correlation):
#                     error_correlation[center_idx] = 0
#             else:
#                 # For crosscorrelation: compare projection before & after lag=0
#                 if center_idx - 6 >= 0 and center_idx - 1 >= 0:
#                     x_bef = lags[center_idx - 6 : center_idx - 1]
#                     y_bef = mean_correlation[center_idx - 5 : center_idx]
#                     slope_bef, intercept_bef, _, _, _ = linregress(x_bef, y_bef)
#                     corr_before = intercept_bef  # slope_bef*0+intercept_bef
#                 else:
#                     corr_before = mean_correlation[center_idx]

#                 if center_idx + 6 <= len(mean_correlation):
#                     x_aft = lags[center_idx + 1 : center_idx + 6]
#                     y_aft = mean_correlation[center_idx + 1 : center_idx + 6]
#                     slope_aft, intercept_aft, _, _, _ = linregress(x_aft, y_aft)
#                     corr_after = intercept_aft
#                 else:
#                     corr_after = mean_correlation[center_idx]

#                 # take max
#                 mean_correlation[center_idx] = np.max([corr_before, corr_after])
#                 if center_idx < len(error_correlation):
#                     error_correlation[center_idx] = 0
#         if not self.return_full:
#             # Slice to keep only lag >= 0
#             mean_correlation = mean_correlation[self.max_lag :]
#             error_correlation = error_correlation[self.max_lag :]
#             correlations_array = correlations_array[:, self.max_lag :]
#             lags = lags[self.max_lag :]
#         dwell_time = None
#         if self.show_plot:
#             # If you have a Plots() class or similar, call it here:
#             if self.secondary_data is None:
#                 # Autocorrelation
#                 dwell_time = Plots().plot_autocorrelation(
#                     mean_correlation=mean_correlation,
#                     error_correlation=error_correlation,
#                     lags=lags,
#                     correlations_array=correlations_array,
#                     time_interval_between_frames_in_seconds=self.time_interval_between_frames_in_seconds,
#                     index_max_lag_for_fit=self.index_max_lag_for_fit,
#                     start_lag=self.start_lag,
#                     plot_name=self.plot_name,
#                     save_plots=self.save_plots,
#                     line_color=self.line_color,
#                     plot_title=self.plot_title,
#                     fit_type=self.fit_type,
#                     de_correlation_threshold=self.de_correlation_threshold,
#                     normalize_plot_with_g0=self.normalize_plot_with_g0,
#                     plot_individual_trajectories = self.plot_individual_trajectories,
#                     y_axes_min_max_list_values = self.y_axes_min_max_list_values,
#                     x_axes_min_max_list_values = self.x_axes_min_max_list_values,
#                 )
#             else:
#                 # Crosscorrelation
#                 dwell_time = Plots().plot_crosscorrelation(
#                     intensity_array_ch0=self.primary_data,
#                     intensity_array_ch1=self.secondary_data,
#                     mean_correlation=mean_correlation,
#                     error_correlation=error_correlation,
#                     lags=lags,
#                     time_interval_between_frames_in_seconds=self.time_interval_between_frames_in_seconds,
#                     plot_name=self.plot_name,
#                     save_plots=self.save_plots,
#                     line_color=self.line_color,
#                     plot_title=self.plot_title,
#                     normalize_plot_with_g0=self.normalize_plot_with_g0,
#                     y_axes_min_max_list_values = self.y_axes_min_max_list_values,
#                     x_axes_min_max_list_values = self.x_axes_min_max_list_values,
#                 )

#         # Done
#         return mean_correlation, error_correlation, lags, correlations_array, dwell_time

class Correlation:
    """
    A class for calculating the autocorrelation or cross-correlation of datasets.

    Attributes:
        primary_data (np.ndarray): Primary dataset for autocorrelation, shape [sample, time].
        secondary_data (np.ndarray, optional): Secondary dataset for cross-correlation, same shape as primary.
        max_lag (int, optional): Maximum lag to compute correlation, defaults to half time series length.
        nan_handling (str, optional): Strategy to handle NaN values. Options: 'zeros', 'mean', 'forward_fill', 'ignore'.
        return_full (bool, optional): Whether to return the full correlation array or only positive lags.
        use_bootstrap (bool, optional): Whether to use bootstrap for error estimation.
        shift_data (bool, optional): Whether to shift data based on leading NaNs.
        show_plot (bool, optional): Whether to display the plot.
        save_plots (bool, optional): Whether to save the plots.
        plot_name (str, optional): Name of the plot file.
        time_interval_between_frames_in_seconds (int, optional): Time interval between frames.
        index_max_lag_for_fit (int, optional): Index for maximum lag for fitting.
        color_channel (int, optional): Color channel for plotting.
        start_lag (int, optional): Starting lag for plateau finding.
        line_color (str, optional): Color of the plot line.
        plot_title (str, optional): Title of the plot.
        fit_type (str, optional): Type of fit for the plot.
        de_correlation_threshold (float, optional): Threshold for decorrelation.
        use_linear_projection_for_lag_0 (bool, optional): Whether to use linear projection for lag 0.
        correct_baseline (bool, optional): If True, subtract baseline from mean correlation.
        use_global_mean (bool, optional): If True, use a global mean for correlation normalization.
        use_normalization_factor (bool, optional): If True, multiply correlation by 1/(global_means * #time_points).
        remove_outliers (bool, optional): If True, remove “extreme” outlier trajectories before computing mean.
        MAD_THRESHOLD_FACTOR (float, optional): Threshold factor for outlier removal.
        multi_tau (bool, optional): If True, use multi-tau algorithm for correlation (non-uniform, positive lags only).

    Methods:
        run():
            Executes the correlation computation based on initialized settings.

            Returns:
                (mean_correlation, error_correlation, lags, correlations_array, dwell_time)
    """
    def __init__(
        self,
        primary_data,
        secondary_data=None,
        max_lag=None,
        nan_handling='zeros',
        return_full=True,
        use_bootstrap=True,
        shift_data=False,
        show_plot=False,
        save_plots=False,
        plot_name='temp_AC.png',
        time_interval_between_frames_in_seconds=1,
        index_max_lag_for_fit=None,
        color_channel=0,
        start_lag=0,
        line_color='blue',
        correct_baseline=False,
        baseline_offset=None,
        use_global_mean=False,
        plot_title=None,
        fit_type='linear',
        de_correlation_threshold=0.01,
        use_linear_projection_for_lag_0=True,
        normalize_plot_with_g0=False,
        remove_outliers=True,
        MAD_THRESHOLD_FACTOR=6.0,
        plot_individual_trajectories=False,
        y_axes_min_max_list_values=None,
        x_axes_min_max_list_values=None,
        multi_tau=False
    ):
        def shift_and_fill(data1, data2=None, min_nan_threshold=3, fill_with_nans=True):
            """
            Remove leading NaNs beyond a threshold and shift arrays left, filling the end with NaNs or zeros.
            """
            if data1.ndim != 1:
                raise ValueError("Both data1 and data2 must be 1D arrays.")
            nan_count = 0
            for value in data1:
                if np.isnan(value):
                    nan_count += 1
                else:
                    break
            if nan_count >= min_nan_threshold:
                fill_value = np.nan if fill_with_nans else 0
                new_data1 = np.full_like(data1, fill_value)
                new_data1[: len(data1) - nan_count] = data1[nan_count:]
                if data2 is not None:
                    new_data2 = np.full_like(data2, fill_value)
                    new_data2[: len(data2) - nan_count] = data2[nan_count:]
                else:
                    new_data2 = None
                return new_data1, new_data2
            return data1, data2

        if shift_data:
            primary_data_shifted = np.zeros_like(primary_data)
            secondary_data_shifted = np.zeros_like(secondary_data) if secondary_data is not None else None
            for i in range(primary_data.shape[0]):
                if secondary_data is None:
                    primary_data_shifted[i, :], _ = shift_and_fill(primary_data[i, :], None, min_nan_threshold=2)
                else:
                    primary_data_shifted[i, :], secondary_data_shifted[i, :] = shift_and_fill(primary_data[i, :], secondary_data[i, :], min_nan_threshold=2)
            primary_data = primary_data_shifted
            if secondary_data is not None:
                secondary_data = secondary_data_shifted

        # Store attributes
        self.primary_data = primary_data
        self.secondary_data = secondary_data
        self.max_lag = max_lag
        self.nan_handling = nan_handling
        self.return_full = return_full
        self.use_bootstrap = use_bootstrap
        self.BOOTSTRAP_ITERATIONS = 1000
        self.time_interval_between_frames_in_seconds = float(time_interval_between_frames_in_seconds)
        self.index_max_lag_for_fit = index_max_lag_for_fit
        self.plot_name = plot_name
        self.save_plots = save_plots
        self.show_plot = show_plot
        self.color_channel = color_channel
        self.start_lag = start_lag
        self.line_color = line_color
        self.plot_title = plot_title
        self.fit_type = fit_type
        self.de_correlation_threshold = de_correlation_threshold
        self.use_linear_projection_for_lag_0 = use_linear_projection_for_lag_0
        self.normalize_plot_with_g0 = normalize_plot_with_g0
        self.correct_baseline = correct_baseline
        if baseline_offset is None:
            # Default: use half the time series length as baseline offset for fitting baseline
            self.baseline_offset = int(primary_data.shape[1] // 2)
        else:
            self.baseline_offset = baseline_offset
        if correct_baseline:
            plot_individual_trajectories = False
            print('Baseline correction is enabled. Plotting individual trajectories is disabled due to baseline correction.')
        self.use_global_mean = use_global_mean
        self.remove_outliers = remove_outliers
        self.MAD_THRESHOLD_FACTOR = MAD_THRESHOLD_FACTOR
        self.plot_individual_trajectories = plot_individual_trajectories
        self.y_axes_min_max_list_values = y_axes_min_max_list_values
        self.x_axes_min_max_list_values = x_axes_min_max_list_values
        self.multi_tau = multi_tau

    def run(self):
        """
        Execute the correlation calculations with optional bootstrap error estimation.
        """
        if self.max_lag is None:
            self.max_lag = self.primary_data.shape[1] - 1
        else:
            if self.max_lag >= self.primary_data.shape[1]:
                raise ValueError("Max lag cannot be greater than the length of the time series.")

        # Helper functions
        def trim_nans_from_edges(data):
            mask = ~np.isnan(data)
            if not np.any(mask):
                return np.array([])
            start_idx = np.argmax(mask)
            end_idx = len(mask) - np.argmax(mask[::-1])
            return data[start_idx:end_idx]

        # Prepare forward fill function if needed
        if self.nan_handling == "forward_fill":
            def forward_fill_func(data):
                not_nan = ~np.isnan(data)
                if not np.any(not_nan):
                    return np.array([])
                first_valid_index = np.argmax(not_nan)
                last_valid_index = len(data) - np.argmax(not_nan[::-1]) - 1
                trimmed = data[first_valid_index:last_valid_index + 1]
                mask_ff = np.isnan(trimmed)
                idx = np.where(~mask_ff, np.arange(len(trimmed)), 0)
                np.maximum.accumulate(idx, out=idx)
                filled = trimmed[idx]
                result = np.full_like(data, np.nan)
                result[first_valid_index:last_valid_index + 1] = filled
                return result
            local_forward_fill = forward_fill_func
        else:
            local_forward_fill = lambda arr: arr

        # Global means for normalization if needed
        global_mean_data1 = np.nanmean(self.primary_data)
        global_mean_data2 = np.nanmean(self.secondary_data) if self.secondary_data is not None else global_mean_data1

        if not self.multi_tau:
            # Linear correlation for each sample (symmetric output)
            def process_sample_linear(i):
                try:
                    data1 = trim_nans_from_edges(self.primary_data[i, :])
                    data2 = trim_nans_from_edges(self.secondary_data[i, :]) if self.secondary_data is not None else None
                    if data2 is None:
                        data2 = data1
                    # Handle NaNs according to strategy
                    if self.nan_handling == "mean":
                        mean_val1 = np.nanmean(data1) if len(data1) > 0 else 0.0
                        mean_val2 = np.nanmean(data2) if len(data2) > 0 else 0.0
                        data1 = np.nan_to_num(data1, nan=mean_val1)
                        data2 = np.nan_to_num(data2, nan=mean_val2)
                    elif self.nan_handling == "forward_fill":
                        data1 = local_forward_fill(data1)
                        data2 = local_forward_fill(data2)
                    elif self.nan_handling == "ignore":
                        valid_mask = ~np.isnan(data1) & ~np.isnan(data2)
                        data1 = data1[valid_mask]
                        data2 = data2[valid_mask]
                    elif self.nan_handling == "zeros":
                        data1 = np.nan_to_num(data1)
                        data2 = np.nan_to_num(data2)
                    effective_points = np.sum(~np.isnan(data1))
                    if effective_points < 1:
                        return np.full(2 * self.max_lag + 1, np.nan)
                    # Center data by mean
                    if self.use_global_mean:
                        local_mean1 = global_mean_data1
                        local_mean2 = global_mean_data2
                    else:
                        local_mean1 = np.nanmean(data1)
                        local_mean2 = np.nanmean(data2)
                    cdata1 = data1 - local_mean1
                    cdata2 = data2 - local_mean2
                    # Remove any residual NaNs
                    cdata1 = cdata1[~np.isnan(cdata1)]
                    cdata2 = cdata2[~np.isnan(cdata2)]
                    if len(cdata1) == 0 or len(cdata2) == 0:
                        return np.full(2 * self.max_lag + 1, np.nan)
                    N = len(cdata1)
                    # Compute raw full cross-correlation
                    raw_corr = np.correlate(cdata1, cdata2, mode="full")
                    mid = N - 1
                    min_overlap = max(5, int(0.2 * N))
                    final_corr = np.empty_like(raw_corr, dtype=np.float64)
                    for j in range(len(raw_corr)):
                        lag = j - mid
                        overlap = N - abs(lag)
                        if overlap < min_overlap:
                            final_corr[j] = np.nan
                            continue
                        if lag >= 0:
                            seg1 = data1[:N - lag]
                            seg2 = data2[lag:]
                        else:
                            seg1 = data1[-lag:]
                            seg2 = data2[:N + lag]
                        local_norm = np.nanmean(seg1) * np.nanmean(seg2)
                        if local_norm == 0:
                            final_corr[j] = np.nan
                        else:
                            final_corr[j] = (raw_corr[j] / overlap) / local_norm
                    mid_point = len(final_corr) // 2
                    desired_length = 2 * self.max_lag + 1
                    current_length = len(final_corr)
                    if current_length < desired_length:
                        # Pad with NaNs if needed
                        out = np.full(desired_length, np.nan)
                        start_idx = (desired_length - current_length) // 2
                        out[start_idx:start_idx + current_length] = final_corr
                        return out
                    else:
                        start_idx = mid_point - self.max_lag
                        end_idx = mid_point + self.max_lag + 1
                        return final_corr[start_idx:end_idx]
                except Exception as e:
                    print(f"Error in process_sample_linear for sample {i}: {e}")
                    return np.full(2 * self.max_lag + 1, np.nan)

            correlations_array = np.array(
                Parallel(n_jobs=-1)(delayed(process_sample_linear)(i) for i in range(self.primary_data.shape[0])),
                dtype=np.float64
            )
        else:
            # Multi-tau correlation for each sample (positive lags, non-uniform spacing)
            # Set parameter m (channels per stage after initial). m=8 gives 16 initial points, then groups of 8.
            m = 8
            N0 = self.primary_data.shape[1]
            global_lags_idx = []
            current_length = N0
            dt_factor = 1
            stage = 0
            while True:
                if stage == 0:
                    start_i = 0
                    end_i = min(2 * m - 1, self.max_lag, current_length - 1)
                else:
                    start_i = m
                    end_i = min(2 * m - 1, int(self.max_lag // dt_factor), current_length - 1)
                if start_i > end_i:
                    break
                for i_val in range(start_i, end_i + 1):
                    global_lags_idx.append(i_val * dt_factor)
                if end_i < 2 * m - 1 or (end_i * dt_factor) >= self.max_lag or current_length < 2:
                    break
                new_length = current_length // 2
                if new_length < 1:
                    break
                current_length = new_length
                dt_factor *= 2
                stage += 1
            global_lags_idx = sorted(set(global_lags_idx))
            global_lags_idx = np.array(global_lags_idx, dtype=int)
            idx_map = {lag: idx for idx, lag in enumerate(global_lags_idx)}

            def process_sample_multi_tau(i):
                try:
                    data1 = trim_nans_from_edges(self.primary_data[i, :])
                    data2 = trim_nans_from_edges(self.secondary_data[i, :]) if self.secondary_data is not None else None
                    if data2 is None:
                        data2 = data1
                    # Handle NaNs according to strategy
                    if self.nan_handling == "mean":
                        mean_val1 = np.nanmean(data1) if len(data1) > 0 else 0.0
                        mean_val2 = np.nanmean(data2) if len(data2) > 0 else 0.0
                        data1 = np.nan_to_num(data1, nan=mean_val1)
                        data2 = np.nan_to_num(data2, nan=mean_val2)
                    elif self.nan_handling == "forward_fill":
                        data1 = local_forward_fill(data1)
                        data2 = local_forward_fill(data2)
                    elif self.nan_handling == "ignore":
                        valid_mask = ~np.isnan(data1) & ~np.isnan(data2)
                        data1 = data1[valid_mask]
                        data2 = data2[valid_mask]
                    elif self.nan_handling == "zeros":
                        data1 = np.nan_to_num(data1)
                        data2 = np.nan_to_num(data2)
                    effective_points = np.sum(~np.isnan(data1))
                    if effective_points < 1:
                        # Return all-NaN array of global length
                        return np.full(len(global_lags_idx), np.nan)
                    # Center data by mean
                    if self.use_global_mean:
                        local_mean1 = global_mean_data1
                        local_mean2 = global_mean_data2
                    else:
                        local_mean1 = np.nanmean(data1)
                        local_mean2 = np.nanmean(data2)
                    cdata1 = data1 - local_mean1
                    cdata2 = data2 - local_mean2
                    # Remove any residual NaNs
                    mask_valid = ~np.isnan(cdata1) & ~np.isnan(cdata2)
                    cdata1 = cdata1[mask_valid]
                    cdata2 = cdata2[mask_valid]
                    data1_valid = data1[mask_valid]
                    data2_valid = data2[mask_valid]
                    current_N = len(cdata1)
                    if current_N == 0:
                        return np.full(len(global_lags_idx), np.nan)
                    # Multi-tau loop
                    output_corr = np.full(len(global_lags_idx), np.nan, dtype=np.float64)
                    current_data_raw1 = data1_valid.copy()
                    current_data_raw2 = data2_valid.copy()
                    current_cdata1 = cdata1.copy()
                    current_cdata2 = cdata2.copy()
                    dt_factor = 1
                    stage = 0
                    while True:
                        if stage == 0:
                            start_i = 0
                            end_i = min(2 * m - 1, self.max_lag, current_N - 1)
                        else:
                            start_i = m
                            end_i = min(2 * m - 1, int(self.max_lag // dt_factor), current_N - 1)
                        if start_i > end_i:
                            break
                        min_overlap = max(5, int(0.2 * current_N))
                        for j in range(start_i, end_i + 1):
                            overlap = current_N - j
                            if overlap < min_overlap:
                                continue
                            raw_sum = np.nansum(current_cdata1[:current_N - j] * current_cdata2[j:current_N])
                            seg1 = current_data_raw1[:current_N - j]
                            seg2 = current_data_raw2[j:current_N]
                            local_norm = np.nanmean(seg1) * np.nanmean(seg2)
                            if local_norm == 0:
                                corr_val = np.nan
                            else:
                                corr_val = (raw_sum / overlap) / local_norm
                            lag_index = j * dt_factor
                            if lag_index in idx_map:
                                output_corr[idx_map[lag_index]] = corr_val
                        if end_i < 2 * m - 1 or (end_i * dt_factor) >= self.max_lag or current_N < 2:
                            break
                        new_length = current_N // 2
                        if new_length < 1:
                            break
                        # Downsample data
                        if self.secondary_data is None:
                            new_data_raw1 = 0.5 * (current_data_raw1[:2 * new_length:2] + current_data_raw1[1:2 * new_length:2])
                            new_data_raw2 = new_data_raw1  # autocorrelation
                        else:
                            new_data_raw1 = 0.5 * (current_data_raw1[:2 * new_length:2] + current_data_raw1[1:2 * new_length:2])
                            new_data_raw2 = 0.5 * (current_data_raw2[:2 * new_length:2] + current_data_raw2[1:2 * new_length:2])
                        if self.use_global_mean:
                            new_mean1 = global_mean_data1
                            new_mean2 = global_mean_data2
                        else:
                            new_mean1 = np.nanmean(new_data_raw1)
                            new_mean2 = np.nanmean(new_data_raw2) if self.secondary_data is not None else new_mean1
                        new_cdata1 = new_data_raw1 - new_mean1
                        new_cdata2 = new_data_raw2 - new_mean2
                        # Update for next stage
                        current_data_raw1 = new_data_raw1
                        current_data_raw2 = new_data_raw2
                        current_cdata1 = new_cdata1 = new_cdata1  # (just to use consistent naming; not needed separately)
                        current_cdata2 = new_cdata2 = new_cdata2
                        current_N = new_length
                        dt_factor *= 2
                        stage += 1
                    return output_corr
                except Exception as e:
                    print(f"Error in process_sample_multi_tau for sample {i}: {e}")
                    return np.full(len(global_lags_idx), np.nan)

            correlations_array = np.array(
                Parallel(n_jobs=-1)(delayed(process_sample_multi_tau)(i) for i in range(self.primary_data.shape[0])),
                dtype=np.float64
            )

        # Remove outlier trajectories if required
        if self.remove_outliers and correlations_array.size > 0:
            traj_means = np.nanmean(correlations_array, axis=1)
            median_mean = np.nanmedian(traj_means)
            mad = np.nanmedian(np.abs(traj_means - median_mean))
            if mad == 0 or np.isnan(mad):
                keep_mask = np.ones_like(traj_means, dtype=bool)
            else:
                keep_mask = np.abs(traj_means - median_mean) < self.MAD_THRESHOLD_FACTOR * mad
            num_removed = np.sum(~keep_mask)
            num_total = len(traj_means)
            if num_removed > 0:
                print(f"Warning: Removed {num_removed} outlier trajectories (out of {num_total}) based on a threshold of {self.MAD_THRESHOLD_FACTOR} MAD from the median mean correlation.")
            correlations_array = correlations_array[keep_mask, :]

        # If all data removed or no valid points
        if correlations_array.shape[0] == 0:
            length = correlations_array.shape[1] if correlations_array.ndim > 1 else (len(global_lags_idx) if self.multi_tau else (2 * self.max_lag + 1))
            mean_correlation = np.full(length, np.nan)
            error_correlation = np.full_like(mean_correlation, np.nan)
            if not self.multi_tau:
                lags = np.arange(-self.max_lag, self.max_lag + 1) * self.time_interval_between_frames_in_seconds
            else:
                lags = (global_lags_idx if 'global_lags_idx' in locals() else np.arange(0, self.max_lag + 1)) * self.time_interval_between_frames_in_seconds
            return mean_correlation, error_correlation, lags, correlations_array, None

        # Compute mean correlation and error bars
        mean_correlation = np.nanmean(correlations_array, axis=0)
        # Correct baseline via exponential fit if requested
        if self.correct_baseline:
            L = len(mean_correlation) - 1
            start_idx_fit = 2
            time_range = int(L * 0.99)
            if time_range <= start_idx_fit:
                time_range = start_idx_fit + 1
            if not self.multi_tau:
                # Construct positive lags array (assuming mean_correlation corresponds to lag 0..max_lag if return_full=False, or symmetrical otherwise)
                if self.return_full:
                    # If still symmetric, convert to positive lags portion for fitting
                    lags_array = np.arange(0, L + 1) * self.time_interval_between_frames_in_seconds
                else:
                    lags_array = np.arange(0, len(mean_correlation)) * self.time_interval_between_frames_in_seconds
            else:
                lags_array = global_lags_idx * self.time_interval_between_frames_in_seconds
            y_fit = mean_correlation[start_idx_fit:time_range]
            t_fit = lags_array[start_idx_fit:time_range]
            B_guess = y_fit[-1] if len(y_fit) > 0 else 0.0
            A_guess = mean_correlation[0] - B_guess
            tau_guess = (t_fit[-1] - t_fit[0]) / 2.0 if len(t_fit) > 1 else 1.0
            initial_guess = [A_guess, tau_guess, B_guess]
            lower_bounds = [0, 1e-6, np.min(y_fit) if len(y_fit) > 0 else 0.0]
            upper_bounds = [np.inf, np.inf, mean_correlation[0]]
            mask_fit = np.isfinite(y_fit) & np.isfinite(t_fit)
            t_clean = t_fit[mask_fit]
            y_clean = y_fit[mask_fit]
            if len(t_clean) < 3:
                warnings.warn(f"Too few valid points ({len(t_clean)}) for exponential fit—using fallback.")
                fitted_B = np.nanpercentile(y_fit, 10) if len(y_fit) > 0 else 0.0
            else:
                try:
                    popt, _ = curve_fit(lambda t, A, tau, B: A * np.exp(-t / tau) + B,
                                        t_clean, y_clean, p0=initial_guess,
                                        bounds=(lower_bounds, upper_bounds),
                                        maxfev=10000)
                    fitted_B = popt[2]
                    print(f"Exponential fit parameters: A={popt[0]:.3g}, τ={popt[1]:.3g}, B={popt[2]:.3g}")
                except Exception as e:
                    warnings.warn(f"Exp fit failed ({type(e).__name__}: {e}) → using 10th percentile fallback.")
                    fitted_B = np.nanpercentile(y_fit, 10) if len(y_fit) > 0 else 0.0
            mean_correlation = mean_correlation - fitted_B

        num_kept = correlations_array.shape[0]
        if self.use_bootstrap and num_kept > 1:
            def single_bootstrap_iteration(_):
                rng = np.random.default_rng()
                indices = rng.choice(num_kept, size=num_kept, replace=True)
                sample = correlations_array[indices, :]
                m = np.nanmean(sample, axis=0)
                if self.correct_baseline:
                    if not self.multi_tau:
                        center_idx = self.max_lag
                        offset = min(self.baseline_offset, center_idx)
                        neg_region = m[center_idx - offset : center_idx]
                        pos_region = m[center_idx + 1 : center_idx + 1 + offset]
                        baseline_value = np.nanpercentile(np.concatenate([neg_region, pos_region]), 10)
                        m = m - baseline_value
                    else:
                        offset = min(self.baseline_offset, len(m))
                        tail_region = m[-offset:] if offset > 0 else m
                        baseline_value = np.nanpercentile(tail_region, 10)
                        m = m - baseline_value
                return m
            all_means = np.array(
                Parallel(n_jobs=-1)(delayed(single_bootstrap_iteration)(_) for _ in range(self.BOOTSTRAP_ITERATIONS)),
                dtype=np.float64
            )
            error_correlation = np.nanstd(all_means, axis=0)
        else:
            error_correlation = np.nanstd(correlations_array, axis=0) / np.sqrt(num_kept)

        # Construct lags array (in seconds)
        if not self.multi_tau:
            lags = np.arange(-self.max_lag, self.max_lag + 1) * self.time_interval_between_frames_in_seconds
        else:
            lags = global_lags_idx * self.time_interval_between_frames_in_seconds

        # Linear projection adjustment for lag=0
        if self.use_linear_projection_for_lag_0:
            if not self.multi_tau:
                center_idx = self.max_lag
                if self.secondary_data is None:
                    # Autocorrelation: use negative side to project to 0
                    if center_idx - 6 >= 0 and center_idx - 1 >= 0:
                        x = lags[center_idx - 6 : center_idx - 1]
                        y = mean_correlation[center_idx - 5 : center_idx]
                        if len(x) > 1 and np.all(np.isfinite(y)):
                            _, intercept, _, _, _ = linregress(x, y)
                            mean_correlation[center_idx] = intercept
                    if center_idx < len(error_correlation):
                        error_correlation[center_idx] = 0
                else:
                    # Cross-correlation: project both sides and take max
                    if center_idx - 6 >= 0 and center_idx - 1 >= 0:
                        x_bef = lags[center_idx - 6 : center_idx - 1]
                        y_bef = mean_correlation[center_idx - 5 : center_idx]
                        corr_before = linregress(x_bef, y_bef).intercept if len(x_bef) > 1 and np.all(np.isfinite(y_bef)) else mean_correlation[center_idx]
                    else:
                        corr_before = mean_correlation[center_idx]
                    if center_idx + 6 < len(mean_correlation):
                        x_aft = lags[center_idx + 1 : center_idx + 6]
                        y_aft = mean_correlation[center_idx + 1 : center_idx + 6]
                        corr_after = linregress(x_aft, y_aft).intercept if len(x_aft) > 1 and np.all(np.isfinite(y_aft)) else mean_correlation[center_idx]
                    else:
                        corr_after = mean_correlation[center_idx]
                    mean_correlation[center_idx] = np.nanmax([corr_before, corr_after])
                    if center_idx < len(error_correlation):
                        error_correlation[center_idx] = 0
            else:
                if self.secondary_data is None:
                    # Autocorrelation multi-tau: use first few lags to project to 0
                    if len(lags) > 5:
                        x = lags[1:6]
                        y = mean_correlation[1:6]
                        if len(x) > 1 and np.all(np.isfinite(y)):
                            _, intercept, _, _, _ = linregress(x, y)
                            mean_correlation[0] = intercept
                    if len(error_correlation) > 0:
                        error_correlation[0] = 0
                else:
                    # Cross-correlation multi-tau: no adjustment (lack negative lags)
                    pass

        # For linear correlation, handle return_full flag (for positive lags only)
        if not self.multi_tau and not self.return_full:
            mean_correlation = mean_correlation[self.max_lag:]
            error_correlation = error_correlation[self.max_lag:]
            correlations_array = correlations_array[:, self.max_lag:]
            lags = lags[self.max_lag:]

        dwell_time = None
        if self.show_plot:
            if self.secondary_data is None:
                dwell_time = Plots().plot_autocorrelation(
                    mean_correlation=mean_correlation,
                    error_correlation=error_correlation,
                    lags=lags,
                    correlations_array=correlations_array,
                    time_interval_between_frames_in_seconds=self.time_interval_between_frames_in_seconds,
                    index_max_lag_for_fit=self.index_max_lag_for_fit,
                    start_lag=self.start_lag,
                    plot_name=self.plot_name,
                    save_plots=self.save_plots,
                    line_color=self.line_color,
                    plot_title=self.plot_title,
                    fit_type=self.fit_type,
                    de_correlation_threshold=self.de_correlation_threshold,
                    normalize_plot_with_g0=self.normalize_plot_with_g0,
                    plot_individual_trajectories=self.plot_individual_trajectories,
                    y_axes_min_max_list_values=self.y_axes_min_max_list_values,
                    x_axes_min_max_list_values=self.x_axes_min_max_list_values,
                )
            else:
                dwell_time = Plots().plot_crosscorrelation(
                    intensity_array_ch0=self.primary_data,
                    intensity_array_ch1=self.secondary_data,
                    mean_correlation=mean_correlation,
                    error_correlation=error_correlation,
                    lags=lags,
                    time_interval_between_frames_in_seconds=self.time_interval_between_frames_in_seconds,
                    plot_name=self.plot_name,
                    save_plots=self.save_plots,
                    line_color=self.line_color,
                    plot_title=self.plot_title,
                    normalize_plot_with_g0=self.normalize_plot_with_g0,
                    y_axes_min_max_list_values=self.y_axes_min_max_list_values,
                    x_axes_min_max_list_values=self.x_axes_min_max_list_values,
                )
        return mean_correlation, error_correlation, lags, correlations_array, dwell_time


class MicroscopeSimulation():
    def __init__(self):
        pass
    
    def initialize (self,cell_library_folder_path):
        def read_files(directory):
            list_files_names_complete = sorted([f for f in listdir(directory) if isfile(join(directory, f)) and ('cell_') in f], key=str.lower)  # reading all files in the folder with prefix 'cell_'
            list_files_names_complete.sort(key=lambda f: int(re.sub(r'\D', '', f)))  # sorting the index in numerical order
            path_files_complete = [ str(directory.joinpath(f).resolve()) for f in list_files_names_complete ] # creating the complete path for each file
            list_library_cells =  [ np.load(f) for f in path_files_complete ]
            return list_library_cells
        # Path to data
        background_library_path = cell_library_folder_path.joinpath('background_pixels_library.npy')
        df_library_path = cell_library_folder_path.joinpath('df_library.csv')
        # extracting library data
        background_pixels_library = np.load(background_library_path)   # Reading the background library [C, Number_pixels]
        df_cell_library = pd.read_csv(df_library_path)   # Returns a dataframe with the following columns [cell_id, size, number_of_spots,ts_size] and each row represents a cell.
        list_library_cells = read_files(cell_library_folder_path)      # Returns a list of cells where each cell has the shape [Z,Y,X,C]
        return list_library_cells,df_cell_library,background_pixels_library
    
    def generate_simulated_positions (self,image_size_Y_X,number_of_cells_in_simulation,list_library_cells,df_cell_library,generate_cells_close_to_each_other=True):
        initial_dictionary_for_df = {
            'start_y_position': [],
            'start_x_position': [],
            'centroid_y': [],
            'centroid_x': [],
            'z_size': [],
            'y_size': [],
            'x_size': [],
            'nucleus_area': [],
            'number_of_spots': [],
            'ts_size_0': [],
            'ts_size_1': [],
            'ts_size_2': [],
            'ts_size_3': [],
            'library_id': [],
        }
        
        # this statement generate a large number of cells if generate_cells_close_to_each_other is true.
        if generate_cells_close_to_each_other == True:
            large_number_initial_simulation = number_of_cells_in_simulation*3
        else:
            large_number_initial_simulation = number_of_cells_in_simulation
        # Create the DataFrame
        number_cells_in_library = len(list_library_cells)
        max_cell_size = np.max( [np.max(cell.shape[1:3]) for _, cell in enumerate(list_library_cells)] )
        simulation_dataframe = pd.DataFrame(initial_dictionary_for_df)
        #max_cell_size 
        MAX_NUM_ITERATIONS = 20000
        printed_cells=0
        min_position_image_edge = max_cell_size
        max_y_position = image_size_Y_X[0]-min_position_image_edge
        max_x_position = image_size_Y_X[1]-min_position_image_edge
        counter=0
        # random indexes for selecting a cell from library
        number_cells_in_library = len(list_library_cells)
        rnd_index_cells = np.random.randint(0, number_cells_in_library,size=MAX_NUM_ITERATIONS).astype(int)
        # This creates a random positions with a len MAX_NUM_ITERATIONS
        y_positions = np.random.randint(min_position_image_edge, max_y_position-max_cell_size, size=MAX_NUM_ITERATIONS).astype(int)
        x_positions = np.random.randint(min_position_image_edge, max_x_position-max_cell_size, size=MAX_NUM_ITERATIONS).astype(int)
        z_positions = np.zeros(MAX_NUM_ITERATIONS,dtype=int)
        cell_size_Z_Y_X = np.zeros((number_cells_in_library,3))
        for i in range (number_cells_in_library):
            cell_size_Z_Y_X[i,:] = list_library_cells[i][:,:,:,0].shape
        # Main while loop that iterates until number_of_cell_in_image is reached or counter>MAX_NUM_ITERATIONS
        list_cells_position = []
        while (counter< MAX_NUM_ITERATIONS-1) and (printed_cells<=large_number_initial_simulation-1):
            add_cell = False
            tested_positions=[]
            if printed_cells >0:
                # Test cell positions
                cell_Z_Y_X_positions = [z_positions[counter], y_positions[counter], x_positions[counter]]
                tested_positions = list_cells_position.copy()
                tested_positions.append(cell_Z_Y_X_positions) 
                array_tested_positions = np.asarray( tested_positions)
                # Calculating a distance matrix. 
                distance_matrix = np.zeros( (array_tested_positions.shape[0], array_tested_positions.shape[0])) 
                for i in range(len(array_tested_positions)):
                    for j in range(len(array_tested_positions)):
                        if j<i:
                            distance_matrix[i,j] = np.linalg.norm( ( array_tested_positions[i,:]-array_tested_positions[j,:] )  )
                # Masking the distance matrix. Ones indicate the distance is less or equal than threshold_distance
                mask_distance_matrix = (distance_matrix <= max_cell_size) 
                # Negation (NOT) of the distance_matrix .
                negation_subsection_mask_distance_matrix = ~mask_distance_matrix
                lower_diagonal_mask_distance_matrix = np.tril(negation_subsection_mask_distance_matrix, k=-1)
                add_cell = np.all(lower_diagonal_mask_distance_matrix[-1,:-1])
                del array_tested_positions
            else:
                cell_Z_Y_X_positions = [z_positions[counter], y_positions[counter], x_positions[counter]]
                add_cell = True
            if add_cell == True: 
                library_cell_index = rnd_index_cells[counter]
                list_cells_position.append(cell_Z_Y_X_positions)
                centroid_y = y_positions[counter] + cell_size_Z_Y_X[library_cell_index,1]//2
                centroid_x = x_positions[counter] + cell_size_Z_Y_X[library_cell_index,2]//2
                # extracting information about a given cell
                nucleus_area = df_cell_library.loc[   (df_cell_library['cell_id']==library_cell_index) ].nucleus_area.values[0]   
                number_of_spots = df_cell_library.loc[   (df_cell_library['cell_id']==library_cell_index) ].number_of_spots.values[0]   
                ts_array = np.zeros(4,dtype=int)
                list_ts = [df_cell_library.loc[   (df_cell_library['cell_id']==library_cell_index) ].ts_size_0.values ,
                            df_cell_library.loc[   (df_cell_library['cell_id']==library_cell_index) ].ts_size_1.values,
                            df_cell_library.loc[   (df_cell_library['cell_id']==library_cell_index) ].ts_size_2.values ,
                            df_cell_library.loc[   (df_cell_library['cell_id']==library_cell_index) ].ts_size_3.values ] 
                ts_array[:] = list_ts[:]
                cell_data = pd.Series([ y_positions[counter], x_positions[counter], centroid_y, centroid_x ,cell_size_Z_Y_X[library_cell_index,0], cell_size_Z_Y_X[library_cell_index,1], cell_size_Z_Y_X[library_cell_index,2], nucleus_area, number_of_spots]+ts_array.tolist()+[library_cell_index ], index=simulation_dataframe.columns)
                #simulation_dataframe = simulation_dataframe.append(cell_data, ignore_index=True)
                cell_data_df = cell_data.to_frame().T
                simulation_dataframe = pd.concat([simulation_dataframe, cell_data_df], ignore_index=True)
                printed_cells+=1
            counter+=1
        new_dtypes = { 'start_y_position':int,'start_x_position':int,'centroid_y':int,'centroid_x':int,'z_size':int,'y_size':int,'x_size':int,'nucleus_area':int,'number_of_spots':int,'ts_size_0':int,'ts_size_1':int,'ts_size_2':int,'ts_size_3':int,'library_id':int}
        simulation_dataframe = simulation_dataframe.astype(new_dtypes)
        
        if generate_cells_close_to_each_other == True:
            # Calculating the distance matrix of selected cells
            tested_positions = simulation_dataframe[['start_y_position', 'start_x_position']]
            tested_positions.values.shape
            array_tested_positions = np.asarray( tested_positions)
            # Calculating a distance matrix. 
            distance_matrix = np.zeros( (array_tested_positions.shape[0], array_tested_positions.shape[0])) 
            for i in range(len(array_tested_positions)):
                for j in range(len(array_tested_positions)):
                    distance_matrix[i,j] = np.linalg.norm( ( array_tested_positions[i,:]-array_tested_positions[j,:] )  )
            # Calculating the distance of the closest N cells around a given cell
            sum_rows = []
            number_neighbor_cell = 8
            for i in range(distance_matrix.shape[0]):
                row_values = distance_matrix[i]
                n_min_values_indices = np.argsort(row_values)[:number_neighbor_cell]
                sum_rows.append( np.sum(row_values[n_min_values_indices]))
            sum_rows = np.asarray(sum_rows)
            # Selecting only number_of_cells_in_simulation 
            selected_indices = np.argsort(sum_rows)[:number_of_cells_in_simulation]
            simulation_df_new = simulation_dataframe.iloc[selected_indices].copy()
            simulation_df_new = simulation_df_new.reset_index(drop=True)
            simulation_df_new
            simulation_dataframe = simulation_df_new
            simulation_dataframe
        # Creating a new df selecting only the cells 
        size_Z = list_library_cells[0].shape[0]
        complete_image_size_Z_Y_X= [size_Z]+image_size_Y_X
        return simulation_dataframe,complete_image_size_Z_Y_X
    
    def make_simulated_image(self, z_position, y_position, x_position, x_size, y_size, complete_image_size_Z_Y_X, simulation_dataframe, list_library_cells, background_pixels_library = None,alpha_0=0,alpha_1=0,alpha_2=0,remove_elements_low_intensity=False):
        number_color_channels = list_library_cells[0].shape[3]
        # Re-centering z_position index
        length_z_indices = complete_image_size_Z_Y_X[0]
        z_array = np.arange(0,length_z_indices,1)
        # Moving image out of focus
        z_position_hat = int( alpha_0 + (alpha_1 * x_position) + (alpha_2 * y_position) + z_position)
        z_position_center_as_zero = complete_image_size_Z_Y_X[0]//2
        z_position_original = z_position_center_as_zero + z_position_hat
        z_array = [int(i - z_position_center_as_zero) if i < z_position_center_as_zero else int(i - z_position_center_as_zero) for i in range(length_z_indices)] 
        list_mean_background_pixels_library=[]
        if not background_pixels_library is None: 
            list_mean_background_pixels_library = [np.mean(background_pixels_library[i,:]) for i in range(number_color_channels)  ]
        y_range = [y_position, y_position+ y_size]
        x_range = [x_position, x_position+x_size]

        def min_edge_value_full_image (tested_value,edge_values,original_edge):
            if tested_value<edge_values:
                new_range = 0
            else:
                new_range = tested_value
            moved_pixels =abs(original_edge-new_range)
            return new_range,moved_pixels
        def max_edge_value_full_image (tested_value,edge_values,original_edge):
            if tested_value>edge_values:
                new_range = edge_values
            else:
                new_range = tested_value
            moved_pixels =abs(new_range-original_edge-1)
            return new_range,moved_pixels
        # extending the image range to consider cell on the image border
        additional_range = 200
        extended_y_min_range,moved_px_y_min = min_edge_value_full_image(y_range[0]-additional_range,0,y_range[0])
        extended_x_min_range,moved_px_x_min = min_edge_value_full_image(x_range[0]-additional_range,0,x_range[0])
        extended_y_max_range,moved_px_y_max = max_edge_value_full_image(y_range[1]+additional_range,complete_image_size_Z_Y_X[1],y_range[1])
        extended_x_max_range,moved_px_x_max = max_edge_value_full_image(x_range[1]+additional_range,complete_image_size_Z_Y_X[2],x_range[1])
        extended_y_pixels = extended_y_max_range - extended_y_min_range
        extended_x_pixels = extended_x_max_range - extended_x_min_range
        # Function to calculate ranges
        def return_ranges(selected_row, initial_x_range=None, initial_y_range=None):
            tested_x_size = selected_row.x_size
            tested_y_size = selected_row.y_size
            is_even_x = tested_x_size%2== 0
            is_even_y = tested_y_size%2== 0
            if not (initial_x_range is None):
                tested_x_position = selected_row.start_x_position - initial_x_range
            else:
                tested_x_position = selected_row.start_x_position
            if not (initial_y_range is None): 
                tested_y_position = selected_row.start_y_position - initial_y_range
            else:
                tested_y_position = selected_row.start_y_position 
            min_y_value = tested_y_position - tested_y_size//2
            max_y_value = tested_y_position + tested_y_size//2 + int(is_even_x)
            min_x_value = tested_x_position - tested_x_size//2
            max_x_value = tested_x_position + tested_x_size//2 + int(is_even_y)
            return min_y_value,max_y_value,min_x_value,max_x_value
        # Test one by one if a cell is located inside the extended area
        list_is_inside_range =[]
        for _, selected_row in simulation_dataframe.iterrows():
            min_y_value,max_y_value,min_x_value,max_x_value = return_ranges(selected_row,initial_x_range=None,initial_y_range=None)
            is_inside_range = (min_x_value >= extended_x_min_range) & (max_x_value <= extended_x_max_range) & (min_y_value >= extended_y_min_range) & (max_y_value <= extended_y_max_range)
            list_is_inside_range.append(is_inside_range)
        condition = np.array(list_is_inside_range)
        df_cells_in_image = simulation_dataframe[condition]
        # take the image position and the cell location
        number_cells_in_library = len(list_library_cells)
        volume_simulated_image = np.zeros ((extended_y_pixels,extended_x_pixels,number_color_channels ),dtype=int)
        # Repetitive calculation performed over library of cells. Including cell shapes, cell_indexes, simulated volumes
        list_volume_tested_cell=[]    
        for i in range (number_cells_in_library):
            # creating the image if z_position is inside z_array
            if np.isin(z_position_hat, z_array):
                list_volume_tested_cell.append(list_library_cells[i][z_position_original,:,:,:].astype(np.uint16))
            else:
                # iterating for each color channel 
                temp_image = np.zeros_like(list_library_cells[i][0,:,:,:],dtype=np.uint16)
                for ch in range(number_color_channels):
                    # using the center slice
                    temp_image[:,:,ch] = list_library_cells[i][z_position_center_as_zero,:,:,ch]
                list_volume_tested_cell.append(temp_image)
                del temp_image
        # Lambda function to calculate edges in simulation
        min_edge_simulation = lambda tested_value:  0 if tested_value<0 else tested_value
        # main loop that creates the simulated image
        for _, selected_row in df_cells_in_image.iterrows():
            library_id_selected = selected_row.library_id
            volume_selected_cell = list_volume_tested_cell[library_id_selected]
            min_y_value,max_y_value,min_x_value,max_x_value = return_ranges(selected_row,initial_x_range=extended_x_min_range,initial_y_range= extended_y_min_range)
            # Positions in final simulation
            y_min_in_simulation = min_edge_simulation(min_y_value)
            x_min_in_simulation = min_edge_simulation(min_x_value) 
            # Subsection of the volume to add to the final image
            sub_volume_selected_cell = volume_selected_cell.copy()
            sim_y_max = y_min_in_simulation + sub_volume_selected_cell.shape[0]
            sim_x_max = x_min_in_simulation + sub_volume_selected_cell.shape[1]
            # adding the cell to the image
            volume_simulated_image[y_min_in_simulation:sim_y_max,x_min_in_simulation:sim_x_max, :] =  sub_volume_selected_cell 
            del sub_volume_selected_cell
        # Loop that creates the final dataframe only if the nucleus centroid is inside the desired area.
        list_is_inside_range =[]
        for _, selected_row in df_cells_in_image.iterrows():
            centroid_y = selected_row.centroid_y
            centroid_x = selected_row.centroid_x
            is_inside_range = (centroid_x >= x_range[0]) & (centroid_x <= x_range[1]) & (centroid_y >= y_range[0]) & (centroid_y <= y_range[1])
            list_is_inside_range.append(is_inside_range)
        # Test one by one if a cell is located inside the 
        condition_inside_final_area = np.array(list_is_inside_range)
        df_cells_in_image = df_cells_in_image[condition_inside_final_area]
        df_cells_in_image.reset_index(drop=True,inplace=True)
        if not background_pixels_library is None: 
            # adding background noise
            simulated_image = np.zeros_like(volume_simulated_image)
            for i in range (number_color_channels):
                temp_simulated_image = volume_simulated_image[:,:,i].copy()
                zero_indices = np.where(temp_simulated_image == 0)
                random_elements = np.random.choice(background_pixels_library[i,:], size=len(zero_indices[0]))
                # Replace zero elements with random elements
                temp_simulated_image[zero_indices] = random_elements
                simulated_image[:,:,i] = temp_simulated_image
        else:
            simulated_image =volume_simulated_image        
        # add a filter to the image if z is out of bounds
        if not np.isin(z_position_hat, z_array):
            z_distance_from_edge = np.abs(z_position_hat)-np.max(z_array)
            scaling_factor = 1*z_distance_from_edge
            sigma = 10  # The standard deviation of the Gaussian distribution
            for ch in range(number_color_channels):
                simulated_image[:,:,ch] =  gaussian_filter(simulated_image[:,:,ch], sigma*scaling_factor)
                if (not background_pixels_library is None) and (remove_elements_low_intensity==True): 
                    temp_simulated_image=simulated_image[:,:,ch].copy()
                    indices_to_replace = np.where(temp_simulated_image < list_mean_background_pixels_library[ch])
                    random_elements = np.random.choice(background_pixels_library[ch,:], size=len(indices_to_replace[0]))
                #    # Replace zero elements with random elements
                    temp_simulated_image[indices_to_replace] = random_elements
                    simulated_image[:,:,ch] = temp_simulated_image
        # Reshaping the final image
        simulated_image =simulated_image[moved_px_y_min:-moved_px_y_max-1,moved_px_x_min:-moved_px_x_max-1,:].copy()
        return simulated_image,df_cells_in_image



class Testing():
    '''
    This class contains miscellaneous methods to perform tasks needed in multiple classes. No parameters are necessary for this class.
    '''
    def __init__(self):
        pass

    def test_particle_colocalization(self):
        # simulate two colocalized trajectories
        image_size = (1000, 1000)  # width, height
        simulation_time = 20  # total frames
        num_particles = 100  # total particles in the first dataframe
        copy_percentage = 0.3  # 30% of the first dataframe's particles are copied to the second
        diffusion_rate = 5  # particles can move up to 20 units per frame
        df_trajectories_0_test, df_trajectories_1_test = Utilities().generate_random_colocalized_trajectories(image_size, simulation_time, num_particles, copy_percentage, diffusion_rate)

        # colocalization.
        radius = 5  # Define your radius r
        number_overlapping_frames_threshold = 19 # Define the minimum number of frames a pair should appear in to be considered as overlapping
        df_merged_trajectories_test,  df_non_overlapping_test, df_overlapping_test = Utilities().merge_trajectories(df_trajectories_0_test, df_trajectories_1_test, radius=radius, number_overlapping_frames_threshold=number_overlapping_frames_threshold)

        # print the number of particles in each dataframe
        print('Number of particles in df_trajectories_0:', df_trajectories_0_test['particle'].nunique())
        print('Number of particles in df_trajectories_1:', df_trajectories_1_test['particle'].nunique())
        print('Number of particles in df_merged_trajectories:', df_merged_trajectories_test['particle'].nunique())
        print('Number of particles in df_non_overlapping:', df_non_overlapping_test['particle'].nunique())
        print('Number of particles in df_overlapping:', df_overlapping_test['particle'].nunique())
        return None



class Utilities():
    '''
    This class contains miscellaneous methods to perform tasks needed in multiple classes. No parameters are necessary for this class.
    '''
    def __init__(self):
        pass

    def get_one_drive_dir():
        if sys.platform.startswith('win'):
            return Path(os.environ.get("OneDrive"))
        else:
            base = Path.home() / "Library" / "CloudStorage"
            for folder in base.iterdir():
                if folder.is_dir() and "OneDrive" in folder.name:
                    return folder
        return None


    def downsample_array(self, arr: np.ndarray, factor: int, method: str = 'drop') -> np.ndarray:
        """
        Downsample a 2D array (samples x time) by grouping columns.

        Parameters
        ----------
        arr : np.ndarray
            2D array (samples x time) to be downsampled. May contain NaNs.
        factor : int
            Downsampling factor. For example, factor=2 groups every 2 columns.
        method : str, optional
            Downsampling method, either:
            - 'drop': Keep every factor-th column.
            - 'average': Compute the mean (ignoring NaNs) of each block of consecutive columns.

        Returns
        -------
        np.ndarray
            Downsampled 2D array.

        Raises
        ------
        ValueError
            If an invalid method is specified.
        """
        n_samples, n_time = arr.shape

        if method == 'drop':
            # Keep every factor-th column
            return arr[:, ::factor]

        elif method == 'average':
            # Compute the number of groups (blocks) needed; include the last incomplete block if any.
            n_groups = int(np.ceil(n_time / factor))
            downsampled = np.full((n_samples, n_groups), np.nan, dtype=arr.dtype)
            for i in range(n_groups):
                start = i * factor
                end = min((i + 1) * factor, n_time)
                block = arr[:, start:end]
                downsampled[:, i] = np.nanmean(block, axis=1)
            return downsampled

        else:
            raise ValueError("Invalid method. Please choose 'drop' or 'average'.")



    def simulate_missing_data(self, matrix1, matrix2=None, percentage_to_remove_data=0, replace_with='nan'):
        if percentage_to_remove_data ==0: 
            return matrix1, matrix2
        if matrix2 is not None:
            if matrix1.shape != matrix2.shape:
                raise ValueError("Both matrices must have the same shape.")
        num_rows, num_cols = matrix1.shape
        new_matrix1 = matrix1.copy()
        if matrix2 is not None:
            new_matrix2 = matrix2.copy()
        if replace_with == 'zeros':
            replacement_value = 0
        elif replace_with == 'nan':
            replacement_value = np.nan
        else:
            raise ValueError("Invalid replace_with argument. Use 'zeros' or 'nan'.")
        for i in range(num_rows):
            # Randomly select columns to remove between 20% of the percentage_to_remove_data
            rand_percentage_to_remove_data = np.random.randint(int(0.5*percentage_to_remove_data), int(1.5*percentage_to_remove_data))
            total_cols_to_remove = int(num_cols * (rand_percentage_to_remove_data / 100))
            total_cols_to_remove = min(total_cols_to_remove, num_cols)  # Ensure not removing more columns than available
            # Randomly split the total columns to remove between left and right
            left_cols_to_remove = np.random.randint(0, total_cols_to_remove + 1)
            right_cols_to_remove = total_cols_to_remove - left_cols_to_remove
            # Replace the columns from the extremes in both matrices
            if left_cols_to_remove > 0:
                new_matrix1[i, :left_cols_to_remove] = replacement_value
                if matrix2 is not None:
                    new_matrix2[i, :left_cols_to_remove] = replacement_value
            if right_cols_to_remove > 0:
                new_matrix1[i, num_cols - right_cols_to_remove:] = replacement_value
                if matrix2 is not None:
                    new_matrix2[i, num_cols - right_cols_to_remove:] = replacement_value
        if matrix2 is None:
            return new_matrix1, None # Return only the first matrix if the second one is None
        else:   
            return new_matrix1, new_matrix2

    def log_filter(self, image_TZYXC, spot_radius_px=1):
            image_TZYXC_filtered =np.zeros_like(image_TZYXC)
            for t in range (image_TZYXC.shape[0]):
                for ch in range (image_TZYXC.shape[-1]):
                    image_TZYXC_filtered[t,:,:,:,ch] = stack.log_filter(image_TZYXC[t,:,:,:,ch], sigma=spot_radius_px)
            return image_TZYXC_filtered

    def find_last_valid_column(self,data):
        # Initialize an array to hold the index of the last valid data point for each row
        last_valid_indices = np.zeros(data.shape[0], dtype=int)
        # Process each row individually
        for idx, row in enumerate(data):
            # Reverse the row to make counting consecutive NaNs from the end easier
            reversed_row = np.flip(row)
            # Find the first non-NaN value in the reversed row
            valid_index = np.argmax(~np.isnan(reversed_row))
            # If the entire reversed row is NaN, the valid_index will point to a NaN
            if np.isnan(reversed_row[valid_index]):
                # If no valid data points are found, mark the index as -1 or another flag value
                last_valid_indices[idx] = -1
            else:
                # Calculate the last valid index in the original row
                last_valid_indices[idx] = len(row) - 1 - valid_index
        return np.max(last_valid_indices)


    def shift_initial_nans(self,data):
        # Create a new array of the same shape filled with NaNs
        new_data = np.full(data.shape, np.nan)
        # Iterate over each row
        for idx, row in enumerate(data):
            # Find the index of the first non-NaN value
            first_non_nan_index = np.argmax(~np.isnan(row))
            # Check if the row has any non-NaNs at all
            if not np.isnan(row[first_non_nan_index]):
                # Number of elements to shift
                elements_to_shift = len(row) - first_non_nan_index
                # Shift the elements from the first non-NaN to the left
                new_data[idx, :elements_to_shift] = row[first_non_nan_index:]
        return new_data

    
    def show_metadta_and_plot_imeges(self, data_folder_path, show_metadata=True):
        list_images, list_names, voxel_xy_um, voxel_z_um, channel_names, number_color_channels, list_time_intervals, bit_depth = \
                ReadLif(data_folder_path, show_metadata=show_metadata, save_tif=False, save_png=False, format='TZYXC').read()
        # Iterate over all loaded images
        for i, image in enumerate(list_images):
            plt.figure(figsize=(10, 5 * number_color_channels))  # Adjust figure size dynamically
            for channel in range(number_color_channels):
                ax = plt.subplot(1, number_color_channels, channel + 1)
                # Calculate the 95th percentile of the maximum intensity for scaling
                max_value_95_percentile = np.percentile(image[0, :, :, :, channel].max(axis=0), 95)
                # Display the maximum projection of the image
                ax.imshow(image[0, :, :, :, channel].max(axis=0), cmap='gray', vmax=max_value_95_percentile)
                ax.axis('off')
                # Set title for each channel, assuming channel_names is correctly ordered
                if channel_names:
                    ax.set_title(f'Channel {channel}, {channel_names[channel]}')
                else:
                    ax.set_title(f'Channel {channel}')
            # Set the title for the entire figure
            plt.suptitle(list_names[i])
            plt.show()  # Show the plot for this image
            print('Image:', list_names[i], 'Image shape:', image.shape)
        return None


    def shift_trajectories(
        self,
        array_ch0: np.ndarray,
        array_ch1: np.ndarray = None,
        cut_off_index: int = None,
        min_percentage_data_in_trajectory: float = 0.1,
        max_missing_frames: int = None  # maximum allowed missing (NaN) values *internally*
    ) -> np.ndarray:
        """
        Shift trajectories (rows) to remove leading NaNs and trim them to a common length.
        Additionally, remove any trajectory that contains more than a specified number
        of internal NaNs (i.e. missing values between the first and last valid data points).

        Parameters:
            array_ch0 (np.ndarray): 2D array (trajectories x time) for channel 0.
            array_ch1 (np.ndarray, optional): 2D array for channel 1 (same shape as array_ch0).
            cut_off_index (int, optional): Maximum number of columns (time points) to keep.
            min_percentage_data_in_trajectory (float, optional): Minimum fraction of valid (non-NaN) data required.
            max_missing_frames (int, optional): Maximum number of missing values allowed within the internal 
                part of the trajectory (i.e. ignoring leading and trailing NaNs).

        Returns:
            np.ndarray: Tuple (array_ch0, array_ch1) if array_ch1 is provided; otherwise just array_ch0,
                        after shifting, trimming, and filtering.
        """
        if cut_off_index is not None and cut_off_index > array_ch0.shape[1]:
            raise ValueError("The cut_off_index is larger than the number of frames in the trajectories.")

        n_time = array_ch0.shape[1]

        # Relative threshold: require at most a given fraction of NaNs overall.
        relative_threshold = n_time * (1 - min_percentage_data_in_trajectory)
        mask_relative = np.count_nonzero(np.isnan(array_ch0), axis=1) <= relative_threshold

        # Define a helper to count the "internal" NaNs (ignoring leading and trailing NaNs).
        def count_internal_nans(row):
            valid_idx = np.where(~np.isnan(row))[0]
            if valid_idx.size == 0:
                return 0  # row is entirely NaN (should be filtered later by mask_relative)
            first_valid = valid_idx[0]
            last_valid = valid_idx[-1]
            # Count NaNs only between the first and last valid data points (inclusive)
            return np.count_nonzero(np.isnan(row[first_valid:last_valid + 1]))

        if max_missing_frames is not None:
            mask_absolute = np.array([count_internal_nans(row) <= max_missing_frames for row in array_ch0])
            mask = mask_relative & mask_absolute
        else:
            mask = mask_relative

        array_ch0 = array_ch0[mask]
        if array_ch1 is not None:
            array_ch1 = array_ch1[mask]

        if array_ch0.shape[0] == 0:
            raise ValueError("All trajectories have more than the allowed missing data (relative and/or internal).")

        # Shift the data: remove initial NaNs (your helper function should preserve internal NaNs).
        array_ch0 = Utilities().shift_initial_nans(array_ch0)
        if array_ch1 is not None:
            array_ch1 = Utilities().shift_initial_nans(array_ch1)

        # Trim trajectories to the last valid column.
        last_valid_columns = Utilities().find_last_valid_column(array_ch0)
        array_ch0 = array_ch0[:, :last_valid_columns]
        if array_ch1 is not None:
            array_ch1 = array_ch1[:, :last_valid_columns]

        # Determine the cut-off time (using the last valid index).
        max_index = int(np.nanmax(np.argwhere(~np.isnan(array_ch0))[:, 1])) + 1
        if cut_off_index is None:
            cut_off_index = max_index
        else:
            cut_off_index = min(max_index, cut_off_index)
        array_ch0 = array_ch0[:, :cut_off_index]
        if array_ch1 is not None:
            array_ch1 = array_ch1[:, :cut_off_index]

        return (array_ch0, array_ch1) if array_ch1 is not None else array_ch0
        


    def parse_bool_or_int(self, value):
        # Try to convert the value to an integer first
        try:
            int_value = int(value)
            if int_value in [0, 1]:
                return bool(int_value)
        except ValueError:
            # If conversion to int fails, try to understand it as a boolean string
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            else:
                raise ValueError("The provided value must be either 'true', 'false', '0', or '1'.")
        # If all else fails, raise an exception
        raise ValueError("The provided value must be either 'true', 'false', '0', or '1'.")

    @staticmethod
    def metadata_decorator(metadata_folder_func=None, metadata_filename=None, exclude_args=None):
        """
        A decorator to manage metadata operations, including logging function arguments and execution status.

        Args:
            metadata_folder_func (function, optional): Function to get the metadata folder path.
            metadata_filename (str, optional): Filename for metadata storage.
            exclude_args (list, optional): List of argument names to exclude from metadata.
        """
        if exclude_args is None:
            exclude_args = []

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Capture function arguments
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Build metadata dictionary
                metadata = {}
                # Add Date and Time
                metadata["Date and Time"] = pd.Timestamp.now().round('min')
                # Add computer user name
                metadata["computer_user_name"] = getpass.getuser()
                # Add function arguments, excluding specified ones
                for name, value in bound_args.arguments.items():
                    if name not in exclude_args:
                        metadata[name] = value
                try:
                    # Call the original function
                    result = func(*args, **kwargs)
                    # Optionally, you can log that the function executed successfully
                    metadata["Execution Status"] = "Success"
                except Exception as e:
                    # Capture exception details
                    metadata["Exception"] = str(e)
                    metadata["Traceback"] = traceback.format_exc()
                    metadata["Execution Status"] = "Failed"
                    raise  # Re-raise the exception after logging
                finally:
                    # Determine metadata folder path
                    if metadata_folder_func is not None:
                        metadata_folder = metadata_folder_func(*args, **kwargs)
                    else:
                        metadata_folder = 'temp_metadata_folder'  # default folder
                    # Create metadata folder if it doesn't exist
                    metadata_folder_path = Path(metadata_folder)
                    metadata_folder_path.mkdir(parents=True, exist_ok=True)

                    # Customize metadata file name
                    if metadata_filename:
                        metadata_file = f"{metadata_filename}.txt"
                    else:
                        metadata_file = f"{func.__name__}_metadata.txt"
                    metadata_file_path = metadata_folder_path / metadata_file
                    # Write metadata to text file
                    with open(metadata_file_path, 'w') as f:
                        for key, value in metadata.items():
                            f.write(f"{key}: {value}\n")
                return result
            return wrapper
        return decorator

    @staticmethod
    def get_metadata_folder(*args, **kwargs):
        """
        Returns the metadata folder path based on provided arguments.

        Returns:
            str: Path to the metadata folder.
        """
        # Example implementation
        if 'data_folder_path' in kwargs and 'selected_image' in kwargs:
            data_folder_path = Path(kwargs['data_folder_path'])
            selected_image = kwargs['selected_image']
        else:
            raise ValueError("Missing required arguments: 'data_folder_path' and 'selected_image'")
        results_name = 'results_' + data_folder_path.stem + '_cell_id_' + str(selected_image)
        current_dir = Path().absolute()
        results_folder = current_dir.joinpath('results_live_cell', results_name)
        return results_folder
    
    def metric_max_cells_and_area(self,masks, mode='max_cells_and_area'):
            """
            Calculates a metric based on the masks to optimize cell segmentation.

            Parameters:
            - masks: numpy array containing the segmented masks.
            - mode: str, optional
                - 'max_cells': Maximizes the number of cells detected.
                - 'max_area': Maximizes the area of the largest cell.
                - 'max_cells_and_area': Maximizes both the number of cells and their average area.

            Returns:
            - metric: int or float representing the calculated metric.
            """
            n_masks = np.max(masks)
            if n_masks >= 1:
                size_masks = []
                for nm in range(1, n_masks + 1):
                    # Calculate the area of each mask
                    area = np.sum(masks == nm)
                    size_masks.append(area)
                size_masks_array = np.array(size_masks)
                total_area = np.sum(size_masks_array)
                mean_area = np.mean(size_masks_array)
                max_area = np.max(size_masks_array)
                if mode == 'max_cells':
                    metric = n_masks
                elif mode == 'max_area':
                    metric = max_area
                elif mode == 'max_cells_and_area':
                    metric = n_masks * mean_area
                else:
                    raise ValueError("Invalid mode. Choose 'max_cells', 'max_area', or 'max_cells_and_area'.")
            else:
                metric = 0
            return metric
    def clear_folder_except_substring(self, directory, substring):
        """
        Check if a folder exists and remove all contents except files containing a specific substring.

        Parameters:
        - directory (str): The path to the directory to check and clear.
        - substring (str): The substring that must be included in the filenames to keep.
        """
        # Check if the directory exists
        if os.path.exists(directory):
            # List all files and directories in the folder
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                # Check if it is a file or directory and does not contain the substring
                if os.path.isfile(file_path) and substring not in filename:
                    os.remove(file_path)
                elif os.path.isdir(file_path) and substring not in filename:
                    shutil.rmtree(file_path)  # Remove the directory and all its contents
        return None

    def test_particle_presence_all_frames_with_ML(self,croparray,crop_size,selected_color_channel=0,minimal_number_spots_in_time=3,ml_threshold=0.51):
        number_particles_in_crops = croparray.shape[0]//crop_size
        number_time_points_in_crops = croparray.shape[1]//crop_size
        if number_time_points_in_crops < minimal_number_spots_in_time:
            minimal_number_spots_in_time = number_time_points_in_crops
            print('The number of time points is less than the minimal number of spots in time. The minimal number of spots in time is set to the number of time points.')
        list_crops_selected_particle_all_time_points = []
        for particle_id in range(number_particles_in_crops):
            crop_all_time_points = croparray[particle_id*crop_size:(particle_id+1)*crop_size, :, :]
            crop_all_time_points = np.transpose(crop_all_time_points, (1, 0, 2))
            list_crops_selected_particle_all_time_points.append(crop_all_time_points)
        flag_vector = np.zeros(number_particles_in_crops, dtype=bool)
        for particle_to_test in range(0, number_particles_in_crops, 1):
            tested_particle_array = list_crops_selected_particle_all_time_points[particle_to_test][:,:,:]
            # iterate for all time points to determine if the crop has nans
            list_crops_without_nans = []
            for time_point in range(number_time_points_in_crops):
                crop_test_nan = tested_particle_array[time_point*crop_size:(time_point+1)*crop_size, :, :].flatten()
                if np.isnan(crop_test_nan).any():
                    list_crops_without_nans.append(False)
                else:
                    list_crops_without_nans.append(True)
            array_crops_without_nans = np.array(list_crops_without_nans)
            list_crops_nomalized = Utilities().normalize_crop_return_list(array_crops_YXC=tested_particle_array,crop_size=crop_size,selected_color_channel=selected_color_channel,normalize_to_255=True)
            flag_vector_all_time = ML.predict_crops(model_ML, list_crops_nomalized,threshold=ml_threshold).astype(bool)
            flag_vector_particle_particle = flag_vector_all_time & array_crops_without_nans
            if np.sum(flag_vector_particle_particle) > minimal_number_spots_in_time:
                flag_vector[particle_to_test] = True
            else:
                flag_vector[particle_to_test]= False
            #print('tested_particle',particle_to_test,'value',flag_vector[particle_to_test])
        return flag_vector
    


    def normalize_crop_return_list(self,array_crops_YXC,crop_size,selected_color_channel=0, normalize_to_255 = False):
        list_crops = []
        number_crops = array_crops_YXC.shape[0] // crop_size
        for crop_id in range(number_crops):
            crop = array_crops_YXC[crop_id * crop_size:(crop_id + 1) * crop_size, :, selected_color_channel]
            crop = (crop - np.min(crop)) / (np.max(crop) - np.min(crop))
            if normalize_to_255:
                crop = (crop * 255).astype(np.uint8)
            list_crops.append(crop)
        return list_crops


    def pad_image(self, image, pixels_to_pad):
        if pixels_to_pad < 1:
            return image
        # Create a mask of the same shape as the image and fill the borders with 0
        mask = np.ones_like(image)
        mask[:pixels_to_pad, :] = 0
        mask[-pixels_to_pad:, :] = 0
        mask[:, :pixels_to_pad] = 0
        mask[:, -pixels_to_pad:] = 0
        # Apply the mask to the image
        padded_image = image * mask
        return padded_image


    def is_spot_in_crop(self, selected_crop_id, crop_size, selected_color_channel,
                        array_crops_YXC, show_plot=False, snr_threshold=1.0):
        """
        Improved method to determine if a spot is present in a crop using blob detection.
        
        Parameters:
            selected_crop_id (int): The index of the crop to analyze.
            crop_size (int): The size of the crop.
            selected_color_channel (int): The channel index to analyze.
            array_crops_YXC (numpy.ndarray): Array containing image crops (shape: [N*crop_size, X, Channels]).
            show_plot (bool, optional): If True, display the crop with detected blobs.
            snr_threshold (float, optional): The SNR threshold a blob must exceed.
        
        Returns:
            bool: True if at least one blob is detected with SNR >= snr_threshold.
        """
        # Extract the crop for analysis
        crop = array_crops_YXC[selected_crop_id * crop_size : (selected_crop_id + 1) * crop_size, :, selected_color_channel]
        # Optionally smooth the crop to reduce noise
        crop_smooth = gaussian(crop, sigma=1)
        # Use Laplacian-of-Gaussian blob detection.
        # min_sigma and max_sigma can be tuned based on expected spot size.
        blobs = blob_log(crop_smooth, min_sigma=1, max_sigma=crop_size/4, num_sigma=10, threshold=0.1)
        # blob_log returns an array with each row: [y, x, sigma].
        detected_spot = False
        detected_blobs = []
        for blob in blobs:
            y, x, sigma = blob
            # Define the spot (inner circle) with radius ~ sigma*sqrt(2)
            radius_blob = sigma * np.sqrt(2)
            # Define an annular region for background: for example, an outer radius of 1.5 times the blob radius
            radius_outer = 1.5 * radius_blob
            # Create coordinate grids
            y_grid, x_grid = np.ogrid[:crop.shape[0], :crop.shape[1]]
            mask_blob = (x_grid - x)**2 + (y_grid - y)**2 <= radius_blob**2
            mask_outer = (x_grid - x)**2 + (y_grid - y)**2 <= radius_outer**2
            mask_annulus = mask_outer & (~mask_blob)
            if np.sum(mask_blob) == 0 or np.sum(mask_annulus) == 0:
                continue
            mean_blob = np.mean(crop[mask_blob])
            mean_bg = np.mean(crop[mask_annulus])
            std_bg = np.std(crop[mask_annulus])
            snr = (mean_blob - mean_bg) / std_bg if std_bg > 0 else 0
            if snr >= snr_threshold:
                detected_spot = True
                detected_blobs.append((x, y, sigma, snr))
        if show_plot:
            fig, ax = plt.subplots()
            ax.imshow(crop, cmap='gray')
            for x, y, sigma, snr in detected_blobs:
                c = plt.Circle((x, y), sigma*np.sqrt(2), color='red', linewidth=2, fill=False)
                ax.add_patch(c)
                ax.text(x, y, f"{snr:.2f}", color='yellow', fontsize=8)
            plt.title("Detected Blobs")
            plt.show()
        
        return detected_spot

        

    def calculate_SNR(self,mean_array, spot_size):
        """
        Calculate the signal-to-noise ratio for a given area with a central disk and surrounding donut.

        Parameters:
        mean_array (np.ndarray): A 2D array representing the cropped area around a particle.
        spot_size (int): The diameter of the central disk from which the signal is measured.

        Returns:
        float: The calculated signal-to-noise ratio.
        """
        center = mean_array.shape[0] // 2
        radius = spot_size // 2

        # Creating masks for the disk and donut
        y, x = np.ogrid[-center:center + 1, -center:center + 1]
        mask_disk = x**2 + y**2 <= radius**2
        mask_donut = (x**2 + y**2 > radius**2) & (x**2 + y**2 <= (center)**2)

        # Extract disk and donut values from the mean_array
        disk_values = mean_array[mask_disk]
        donut_values = mean_array[mask_donut]

        # Calculate mean and standard deviation for the disk and donut
        mean_intensity_disk = np.mean(disk_values)
        mean_intensity_donut = np.mean(donut_values)
        std_intensity_donut = np.std(donut_values)

        # Calculate SNR
        if std_intensity_donut > 0:
            snr = (mean_intensity_disk - mean_intensity_donut) / std_intensity_donut
        else:
            snr = 0  # Handle division by zero if standard deviation is zero

        return snr

    def calculate_threshold_for_spot_detection(self, image_TZYXC, list_spot_size_px, list_voxels, channels_spots,
                                            max_spots_for_threshold=2000, show_plot=True, plot_name=None):
        thresholds_per_channel = []
        number_random_images = min(7, image_TZYXC.shape[0])
        sigma_smoothing_vectors = 7

        voxel_size_z = list_voxels[0]
        voxel_size_yx = list_voxels[1]
        z_spot_size_in_px = list_spot_size_px[0]
        yx_spot_size_in_px = list_spot_size_px[1]

        # Ensure channels_spots is a list
        if not isinstance(channels_spots, list):
            channels_spots = [channels_spots]

        for idx_channel, channel in enumerate(channels_spots):
            # Initialize lists to store data for averaging
            counts_spots_list = []
            first_derivative_list = []
            thresholds_list = []
            min_thresholds = []
            max_thresholds = []

            for iteration in range(number_random_images):
                # Select a random time point
                selected_time = np.random.randint(0, image_TZYXC.shape[0])
                image_single_channel = image_TZYXC[selected_time, :, :, :, channel]

                # Calculate spot radius
                spot_radius_px = detection.get_object_radius_pixel(
                    voxel_size_nm=(voxel_size_z, voxel_size_yx, voxel_size_yx),
                    object_radius_nm=(voxel_size_z*(z_spot_size_in_px//2), voxel_size_yx*(yx_spot_size_in_px//2) , voxel_size_yx*(yx_spot_size_in_px//2)), ndim=3) 
                
                spot_radius_nm = detection.get_object_radius_nm(
                    voxel_size_nm=(voxel_size_z, voxel_size_yx, voxel_size_yx),
                    object_radius_px=spot_radius_px, ndim=3)

                # Get thresholds and counts
                thresholds, count_spots, _ = detection.get_elbow_values(
                    image_single_channel, voxel_size=(voxel_size_z, voxel_size_yx, voxel_size_yx),
                    spot_radius=spot_radius_nm)

                # Smooth the curve
                smoothed_counts = gaussian_filter(count_spots, sigma=sigma_smoothing_vectors)
                first_derivative = np.gradient(smoothed_counts, thresholds)
                first_derivative = gaussian_filter(first_derivative, sigma=sigma_smoothing_vectors)

                # Normalize the first derivative to be between 0 and 1
                if first_derivative.max() != first_derivative.min():
                    first_derivative_norm = (first_derivative - first_derivative.min()) / (
                        first_derivative.max() - first_derivative.min())
                else:
                    first_derivative_norm = np.zeros_like(first_derivative)

                # Store data for averaging
                counts_spots_list.append(count_spots)
                first_derivative_list.append(first_derivative_norm)
                thresholds_list.append(thresholds)
                min_thresholds.append(thresholds[0])
                max_thresholds.append(thresholds[-1])

            # Define common thresholds for interpolation
            min_threshold = max(min_thresholds)
            max_threshold = min(max_thresholds)
            thresholds_common = np.linspace(min_threshold, max_threshold, num=100)

            # Interpolate and average the data
            counts_spots_interp = []
            first_derivative_interp = []
            for counts_spots, first_derivative_norm, thresholds in zip(counts_spots_list, first_derivative_list, thresholds_list):
                counts_interp = np.interp(thresholds_common, thresholds, counts_spots)
                derivative_interp = np.interp(thresholds_common, thresholds, first_derivative_norm)
                counts_spots_interp.append(counts_interp)
                first_derivative_interp.append(derivative_interp)

            average_counts_spots = np.mean(counts_spots_interp, axis=0)
            # smooth the average counts spots
            
            average_first_derivative = np.mean(first_derivative_interp, axis=0)
            average_first_derivative = gaussian_filter(average_first_derivative, sigma=5)

            # Use the average first derivative to find peaks and select threshold
            peaks, _ = find_peaks(-average_first_derivative, distance=10, prominence=0.05)
            threshold = None
            for index in peaks:
                spots_detected = np.exp(average_counts_spots[index])
                if spots_detected < max_spots_for_threshold:
                    threshold = thresholds_common[index]
                    break  # Select the first suitable threshold

            # Fallback threshold if none found
            if threshold is None:
                threshold = thresholds_common[len(thresholds_common) // 2]
            thresholds_per_channel.append(threshold)

            # Plotting the average data
            if show_plot:
                fig, ax = plt.subplots(1, 2, figsize=(8, 3))
                ax[0].plot(thresholds_common, average_counts_spots, color='gray', linewidth=3)
                ax[0].set_title(f'Number Spots - Channel {channel}')
                # Plot the selected threshold
                idx = np.argmin(np.abs(thresholds_common - threshold))
                ax[0].plot(thresholds_common[idx], average_counts_spots[idx],
                        'o', color='g', label='Threshold', markersize=10)
                ax[0].legend()
                ax[1].plot(thresholds_common, average_first_derivative, color='dodgerblue', linewidth=3)
                ax[1].plot(thresholds_common[peaks], average_first_derivative[peaks], 'x', markersize=10)
                ax[1].plot(thresholds_common[idx], average_first_derivative[idx],
                        'o', color='g', label='Threshold', markersize=10)
                ax[1].set_title(f' d(Number Spots) - Channel {channel}')
                ax[1].legend()
                plt.tight_layout()
                plt.show()
                if plot_name is not None:
                    # Modify plot_name to include channel number
                    plot_name_channel = f"{plot_name}_channel_{channel}.png"
                    fig.savefig(plot_name_channel)

        return thresholds_per_channel

    
    def gaussian_laplace_filter_image(self,image_TZYXC,list_spot_size_px,list_voxels):
        spot_radius_px = detection.get_object_radius_pixel(
                voxel_size_nm=(list_voxels[0], list_voxels[1], list_voxels[1]),
                object_radius_nm=(list_voxels[0]*(list_spot_size_px[0]//2), list_voxels[1]*(list_spot_size_px[1]//2), list_voxels[1]*(list_spot_size_px[1]//2)), ndim=3)
            
        image_TZYXC_filtered =np.zeros_like(image_TZYXC)
        for t in range (image_TZYXC.shape[0]):
            for ch in range (image_TZYXC.shape[-1]):
                image_TZYXC_filtered[t,:,:,:,ch] = stack.log_filter(image_TZYXC[t,:,:,:,ch], sigma=spot_radius_px)
        return image_TZYXC_filtered

    def find_folders_by_keywords(self,base_path, keywords):
        """
        Search for folders in the given base directory that contain all specified keywords 
        and return them sorted by a numerical suffix.

        Parameters:
        - base_path (str or Path): The directory to search within.
        - keywords (list of str): List of keywords to match in folder names.

        Returns:
        - list of str: Sorted list of folder names containing all keywords.
        """
        base_path = Path(base_path)
        if not base_path.exists() or not base_path.is_dir():
            raise ValueError(f"The path {base_path} does not exist or is not a directory.")

        # Filter directories that contain all keywords
        folders = [x for x in base_path.iterdir() if x.is_dir() and all(kw in x.name for kw in keywords)]

        # Sorting function to sort by numerical suffix assumed to be after the last underscore '_'
        def sort_key(name):
            parts = name.name.rsplit('_', 1)
            if len(parts) > 1 and parts[-1].isdigit():
                return int(parts[-1])
            return 0  # Default to 0 if no clear numeric suffix
        sorted_folders = sorted(folders, key=sort_key)
        return [folder.name for folder in sorted_folders]
    
    def calculate_threshold_from_percentage(self,tested_image, masks, target_percentage=5):
        """
        Calculate the intensity threshold from a percentage of pixel intensities above the threshold.
        
        Args:
        tested_image (ndarray): The image to calculate the threshold from.
        masks (ndarray): The mask indicating the region of interest.
        target_percentage (float): The percentage of pixel intensities above the threshold.
        
        Returns:
        int: The calculated intensity threshold.
        """
        masked_data = tested_image * masks[np.newaxis, np.newaxis, :, :, np.newaxis].astype(float)
        intensity_data = np.mean(masked_data[:,:,:,:,1], axis=0).flatten()  # Example data
        intensity_data = intensity_data[intensity_data < np.percentile(intensity_data, 99.99)]
        intensity_data = intensity_data[intensity_data > 0]
        sorted_data = np.sort(intensity_data)
        target_index = int((1 - target_percentage / 100) * len(sorted_data))
        threshold = sorted_data[target_index]
        return int(threshold)


    def combine_images_vertically(self,image_paths, save_path, delete_originals=False,show_image=False):
        # Load images
        images = [Image.open(path) for path in image_paths]
        # Calculate maximum width to standardize image widths
        max_width = max(img.width for img in images)
        # Resize images to have the same width
        resized_images = [img.resize((max_width, int(img.height * max_width / img.width)), Image.Resampling.LANCZOS) for img in images]
        # Calculate the total height for the new combined image
        total_height = sum(img.height for img in resized_images)
        # Create a new image to combine all
        combined_image = Image.new("RGB", (max_width, total_height))
        # Paste each image into the combined image
        y_offset = 0
        for img in resized_images:
            combined_image.paste(img, (0, y_offset))
            y_offset += img.height
        # Save or display the final image
        combined_image.save(save_path)
        if show_image:
            combined_image.show()
        # Optional: Delete the original image files
        if delete_originals:
            for path in image_paths:
                path_obj = Path(path)
                # Ensure the path is not a directory and exists before deleting
                if path_obj.is_file():
                    path_obj.unlink()
                else:
                    print(f"Skipped deleting {path} as it is not a file.")
        return None

    def two_dimensional_gaussian(self,x_y, amplitude, xo, yo, sigma_x, sigma_y, offset):
        xo = float(xo)
        yo = float(yo)    
        x, y = x_y
        g = offset + amplitude * np.exp(-(((x - xo)**2 / (2 * sigma_x**2)) + ((y - yo)**2 / (2 * sigma_y**2))))
        return g.ravel()

    def generate_gaussian_data(self,amplitude, sigma_x, sigma_y, x_position=0, y_position=0, offset=0, size=(9,9), resolution=100):
        x = np.linspace(-size[0]//2, size[0]//2+1, resolution) + x_position
        y = np.linspace(-size[1]//2, size[1]//2+1, resolution) + y_position
        x, y = np.meshgrid(x, y)
        z = self.two_dimensional_gaussian((x, y), amplitude, x_position, y_position, sigma_x, sigma_y, offset)
        return x, y, z.reshape(x.shape)


    def fit_2D_gaussian(self, data):
        y_size, x_size = data.shape
        x = np.linspace(0, x_size - 1, x_size)
        y = np.linspace(0, y_size - 1, y_size)
        x, y = np.meshgrid(x, y)
        
        # Compute centroid for initial guess
        total_intensity = data.sum() + 1e-9
        x_center = (np.arange(x_size) * data.sum(axis=0)).sum() / total_intensity
        y_center = (np.arange(y_size) * data.sum(axis=1)).sum() / total_intensity
        
        amplitude_guess = float(data.max())
        offset_guess = float(data.min())
        sigma_guess = min(x_size, y_size) / 4.0
        
        initial_guess = (amplitude_guess, x_center, y_center, sigma_guess, sigma_guess, offset_guess)
        
        # Set parameter bounds
        bounds = (
            [0,             0,         0,          0.1,       0.1,        data.min()-abs(data.min())],
            [amplitude_guess*5, x_size-1, y_size-1, max(x_size,y_size), max(x_size,y_size), data.max()+abs(data.max())]
        )
        
        try:
            popt, pcov = curve_fit(
                self.two_dimensional_gaussian,
                (x, y),
                data.ravel(),
                p0=initial_guess,
                bounds=bounds
            )
            # Compute sum of squared residuals for goodness-of-fit
            fitted_data = self.two_dimensional_gaussian((x, y), *popt).reshape(data.shape)
            residuals = data - fitted_data
            ss_res = np.sum(residuals**2)
            
            amplitude, x_pos, y_pos, sigma_x, sigma_y, offset = popt
            
            # Check validity
            if amplitude < 0 or sigma_x <= 0 or sigma_y <= 0:
                return None, np.inf  # Invalid fit, return infinite residual
            return {
                'amplitude': amplitude,
                'x_position': x_pos,
                'y_position': y_pos,
                'sigma_x': abs(sigma_x),
                'sigma_y': abs(sigma_y),
                'offset': offset,
            }, ss_res
        except:
            # Fit failed
            return None, np.inf


    def optimize_spot_size(self, frame_data, x_pos, y_pos, use_maximum_projection=False):
        """
        Try spot sizes from 3 to 15 pixels (odd integers), fit the Gaussian 
        for each, and select the best fit based on minimal residuals.
        """
        best_fit = None
        best_residual = np.inf
        best_spot_size = None
        # Iterate over odd sizes from 3 to 15
        for spot_size in range(3, 16, 2):
            half = spot_size // 2
            # Extract the spot data (ensure it's within image bounds)
            y_min = max(y_pos - half, 0)
            y_max = min(y_pos + half + 1, frame_data.shape[0])
            x_min = max(x_pos - half, 0)
            x_max = min(x_pos + half + 1, frame_data.shape[1])
            spot_data = frame_data[y_min:y_max, x_min:x_max]
            fit_result, ss_res = self.fit_2D_gaussian(spot_data)
            if ss_res < best_residual:
                best_residual = ss_res
                best_fit = fit_result
                best_spot_size = spot_size
                
        return best_fit, best_spot_size
        


    def generate_random_colocalized_trajectories(self, image_size=(512,512), simulation_time=30, num_particles=100, copy_percentage=0.3, diffusion_rate=3):
        """
        Generate two dataframes with random trajectories simulating diffusion behavior.
        
        Args:
        image_size (tuple): Size of the image (width, height).
        simulation_time (int): Total time of the simulation in frames.
        num_particles (int): Number of particles in the first dataframe.
        copy_percentage (float): Percentage of particles from the first dataframe to copy into the second.
        diffusion_rate (float): Max distance a particle can move in one time step in any direction.
        
        Returns:
        tuple: Two dataframes with trajectories.
        """
        # Generate initial positions
        initial_positions = np.random.rand(num_particles, 2) * image_size
        # Simulate movements for the first dataframe
        steps_0 = np.random.normal(0, diffusion_rate, (num_particles, simulation_time, 2))
        paths_0 = np.cumsum(steps_0, axis=1)
        paths_0 += initial_positions[:, None, :]  # Adding initial position to each step
        # Flatten arrays to create DataFrame
        df_trajectories_0 = pd.DataFrame({
            'particle': np.repeat(np.arange(num_particles), simulation_time),
            'frame': np.tile(np.arange(simulation_time), num_particles),
            'x': paths_0[:,:,0].flatten(),
            'y': paths_0[:,:,1].flatten()
        })
        # Apply boundary conditions to keep particles within the image size
        df_trajectories_0['x'] = np.clip(df_trajectories_0['x'], 0, image_size[0])
        df_trajectories_0['y'] = np.clip(df_trajectories_0['y'], 0, image_size[1])
        # Decide how many particles to copy
        num_copy = int(num_particles * copy_percentage)
        copied_particles = np.random.choice(num_particles, num_copy, replace=False)
        # Generate trajectories for the second dataframe with new particles plus copied ones
        steps_1 = np.random.normal(0, diffusion_rate, (num_particles, simulation_time, 2))
        paths_1 = np.cumsum(steps_1, axis=1)
        paths_1 += initial_positions[:, None, :]  # Each particle starts at the same initial position as in df_0 but diffuses independently
        df_trajectories_1 = pd.DataFrame({
            'particle': np.repeat(np.arange(num_particles, num_particles * 2), simulation_time),
            'frame': np.tile(np.arange(simulation_time), num_particles),
            'x': paths_1[:,:,0].flatten(),
            'y': paths_1[:,:,1].flatten()
        })
        # Apply boundary conditions
        df_trajectories_1['x'] = np.clip(df_trajectories_1['x'], 0, image_size[0])
        df_trajectories_1['y'] = np.clip(df_trajectories_1['y'], 0, image_size[1])
        # Copy selected particles from the first dataframe to the second
        copied_df = df_trajectories_0[df_trajectories_0['particle'].isin(copied_particles)].copy()
        copied_df['particle'] += num_particles  # Update particle identifiers to prevent collisions
        # Combine the copied particles with the second dataframe
        df_trajectories_1 = pd.concat([df_trajectories_1, copied_df]).reset_index(drop=True)
        return df_trajectories_0, df_trajectories_1



    def merge_trajectories(self, df_trajectories_0, df_trajectories_1, radius=5, number_overlapping_frames_threshold=5,show_plot=False,save_plots=False,plot_name='temp_merged.png'):
        # Generate new unique particle identifiers for both dataframes to prevent ID collisions
        offset = df_trajectories_0['particle'].max() + 1
        df_trajectories_1['particle'] += offset
        # Calculate trajectory lengths
        df_0_particle_length = df_trajectories_0.groupby('particle')['frame'].nunique().reset_index()
        df_0_particle_length.columns = ['particle', 'trajectory_length']
        df_1_particle_length = df_trajectories_1.groupby('particle')['frame'].nunique().reset_index()
        df_1_particle_length.columns = ['particle', 'trajectory_length']
        # Initialize data structures
        overlapping_particles_count = {}
        best_pairs = {}
        # Determine overlaps
        for frame in set(df_trajectories_0['frame']).union(set(df_trajectories_1['frame'])):
            data_0 = df_trajectories_0[df_trajectories_0['frame'] == frame]
            data_1 = df_trajectories_1[df_trajectories_1['frame'] == frame]
            if not data_0.empty and not data_1.empty:
                coords_0 = data_0[['x', 'y']].values
                coords_1 = data_1[['x', 'y']].values
                distances = cdist(coords_0, coords_1)
                for i, row in enumerate(distances):
                    overlapping_indices = np.where(row < radius)[0]
                    for j in overlapping_indices:
                        pair = (int(data_0.iloc[i]['particle']), int(data_1.iloc[j]['particle']))
                        if pair in overlapping_particles_count:
                            overlapping_particles_count[pair] += 1
                        else:
                            overlapping_particles_count[pair] = 1
        # Select pairs exceeding the threshold and with the highest trajectory sum
        for pair, count in overlapping_particles_count.items():
            if count > number_overlapping_frames_threshold:
                traj_length_0 = df_0_particle_length.loc[df_0_particle_length['particle'] == pair[0], 'trajectory_length'].values[0]
                traj_length_1 = df_1_particle_length.loc[df_1_particle_length['particle'] == pair[1], 'trajectory_length'].values[0]
                total_length = traj_length_0 + traj_length_1
                if pair not in best_pairs or total_length > best_pairs[pair][1]:
                    best_pairs[pair] = total_length
        # Create DataFrame for overlapping particles
        df_overlapping = pd.DataFrame()
        if best_pairs:
            for pair in best_pairs.keys():
                if df_0_particle_length.loc[df_0_particle_length['particle'] == pair[0], 'trajectory_length'].values[0] > \
                df_1_particle_length.loc[df_1_particle_length['particle'] == pair[1], 'trajectory_length'].values[0]:
                    df_selected = df_trajectories_0[df_trajectories_0['particle'] == pair[0]].copy()
                else:
                    df_selected = df_trajectories_1[df_trajectories_1['particle'] == pair[1]].copy()
                df_selected['particle_new'] = f"{pair[0]}_{pair[1]}"
                df_overlapping = pd.concat([df_overlapping, df_selected])
        # Adjust particle identifiers
        if not df_overlapping.empty:
            df_overlapping.drop(columns=['particle'], inplace=True)
            df_overlapping.rename(columns={'particle_new': 'particle'}, inplace=True)
            df_overlapping['particle'] = pd.factorize(df_overlapping['particle'])[0]
            # Filter out overlapping particles to create non-overlapping DataFrame
            overlapping_particles = set(df_overlapping['particle'])
            df_non_overlapping = pd.concat([
                df_trajectories_0[~df_trajectories_0['particle'].isin(overlapping_particles)],
                df_trajectories_1[~df_trajectories_1['particle'].isin(overlapping_particles)]
            ])
        else:
            df_non_overlapping = pd.concat([df_trajectories_0, df_trajectories_1])
        # Concatenate and clean up the DataFrames
        df_merged_trajectories = pd.concat([df_non_overlapping, df_overlapping])

        if show_plot:
            Plots().plot_merged_trajectories(df_trajectories_0,df_trajectories_1,df_merged_trajectories,df_overlapping,df_non_overlapping,save_plots=save_plots,plot_name=plot_name)


        return df_merged_trajectories, df_non_overlapping, df_overlapping


    def spots_in_mask(self, df, mask):
        """
        Checks which spots are inside the given 3D mask.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing at least the columns 'z', 'y', 'x' for spot coordinates.
        mask : np.ndarray
            A 3D binary mask of shape [Z, Y, X] where 1 indicates inside the mask and 0 outside.

        Returns
        -------
        df : pd.DataFrame
            The same DataFrame with a new column 'In Mask' that is True/1 if the spot is inside
            the mask, and False/0 otherwise.
        """
        # expand the mask to 3d
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        # number of z slices in the df.
        # test if column z is present in the dataframe
        if 'z' not in df.columns:
        # add a column z with zeros
            df['z'] = 0
        n_z = df.z.nunique()
        # repeat the values of the mask to match the number of z slices in the df.
        mask = np.repeat(mask, n_z, axis=0)

        # Extract spot coordinates from dataframe
        coords = np.stack([df['z'].values, df['y'].values, df['x'].values], axis=1)

        # Round to nearest integer indices
        coords_int = np.round(coords).astype(int)

        # Ensure indices are within mask bounds
        # This step avoids indexing errors if spots lie outside image boundaries
        z_valid = (0 <= coords_int[:, 0]) & (coords_int[:, 0] < mask.shape[0])
        y_valid = (0 <= coords_int[:, 1]) & (coords_int[:, 1] < mask.shape[1])
        x_valid = (0 <= coords_int[:, 2]) & (coords_int[:, 2] < mask.shape[2])
        valid_mask = z_valid & y_valid & x_valid

        # Initialize 'In Mask' as False
        df['In Mask'] = False

        # For valid coordinates, check if spot is inside the mask
        # Only index mask for valid coordinates
        valid_coords = coords_int[valid_mask]
        values_at_coords = mask[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]]

        # Assign True to 'In Mask' where coordinates are valid and inside mask
        df.loc[valid_mask, 'In Mask'] = (values_at_coords == 1)

        return df

    def masks_to_contours(self, masks,downsample_factor=1):
        number_masks = np.max(masks)
        list_contours = []  
        for index_mask in range(1, number_masks + 1):
            #mask_selected = np.where(masks == index_mask, 1, 0)
            binary_mask = np.where(masks == index_mask, 1, 0)
            # Find contours in the binary mask
            contours = find_contours(binary_mask, 0.5)
            # Extract the largest contour
            largest_contour = max(contours, key=len)
            # Downsample the contour
            downsampled_contour = largest_contour[::downsample_factor]
            list_contours.append(downsampled_contour)
        return list_contours

    def contours_to_maks(self, contours, image_shape):
        mask = np.zeros(image_shape, dtype=np.uint8)
        for contour in contours:
            mask[contour[:, 0].astype(int), contour[:, 1].astype(int)] = 1
        return mask            


    def find_src_directory(current_directory: Path) -> Path:
        # Loop through the parent directories
        for parent in current_directory.parents:
            potential_src = parent / 'src'
            if potential_src.is_dir():
                return potential_src
        return None

    def calculate_projection(self, image, axis=0, projection_method='mean'):
        if projection_method== 'max':
            projected_image = np.max(image, axis=axis)
        elif projection_method== 'mean':
            projected_image = np.mean(image, axis=axis)
        elif projection_method== 'sum':
            projected_image = np.sum(image, axis=axis)
        else:
            projected_image = np.mean(image, axis=axis)
            print('Projection_method not defined. Using mean as default.')
        return projected_image
 
    
    def erode_mask(self,img,px_to_remove = 1):
        img[0:px_to_remove, :] = 0;img[:, 0:px_to_remove] = 0;img[img.shape[0]-px_to_remove:img.shape[0]-1, :] = 0; img[:, img.shape[1]-px_to_remove: img.shape[1]-1 ] = 0#This line of code ensures that the corners are zeros.
        return erosion(img)

    def convert_str_to_path(file_path):
        if type(file_path)== pathlib.PosixPath or type(file_path)== pathlib.WindowsPath:
            file_path = file_path
        else:
            file_path = pathlib.Path(file_path)
        return file_path
    
    def remove_images_not_processed(images_metadata, list_images):
        if images_metadata is None:
            return list_images
        else:
            selected_images = []
            max_image_id = images_metadata['Image_id'].max()+1
            for i in range (max_image_id):
                processing_status = images_metadata[images_metadata['Image_id'] == i].Processing.values[0]
                if processing_status == 'successful':
                    selected_images.append(list_images[i])
        return selected_images
    

    def calculate_sharpness(self,list_images, channels_spots, neighborhood_size=31, threshold=1.12):
        list_mean_sharpeness_image = []
        list_is_image_sharp=[]
        list_sharp_images =[]
        for _ , image in enumerate(list_images):
            temp = image[:,:,:,channels_spots[0]].astype(np.uint16)
            focus = stack.compute_focus(temp, neighborhood_size=neighborhood_size)
            mean_sharpeness_image = np.round(np.mean(focus.mean(axis=(1, 2))),3)
            if mean_sharpeness_image > threshold:
                is_image_sharp = True
                list_sharp_images.append(image)
            else:
                is_image_sharp = False
            list_mean_sharpeness_image.append(mean_sharpeness_image)
            list_is_image_sharp.append(is_image_sharp)
        return list_mean_sharpeness_image, list_is_image_sharp,list_sharp_images
    
    def remove_outliers(self, array,min_percentile=1,max_percentile=98):
        max_val = np.percentile(array, max_percentile)
        if np.isnan(max_val) == True:
            max_val = np.percentile(array, max_percentile+0.1)
        min_val = np.percentile(array, min_percentile)
        if np.isnan(min_val) == True:
            min_val = np.percentile(array, min_percentile+0.1)
        array = array [array > min_val]
        array = array [array < max_val]
        return array 
    
    def is_None(self,variable_to_test):
        if (type(variable_to_test) is list):
            variable_to_test = variable_to_test[0]
        if variable_to_test in (None, 'None', 'none',['None'],['none'],[None]):
            is_none = True
        else:
            is_none = False
        return is_none
    
    def make_it_a_list(self,variable_to_test):
        if not (type(variable_to_test) is list):
            list_variable = [variable_to_test]
        else:
            list_variable = variable_to_test
        return list_variable
    
    def reorder_mask_image(self,mask_image_tested):
        number_masks = np.max(mask_image_tested)
        mask_new =np.zeros_like(mask_image_tested)
        if number_masks>0:
            counter = 0
            for index_mask in range(1,number_masks+1):
                if index_mask in mask_image_tested:
                    counter = counter + 1
                    if counter ==1:
                        mask_new = np.where(mask_image_tested == index_mask, -counter, mask_image_tested)
                    else:
                        mask_new = np.where(mask_new == index_mask, -counter, mask_new)
            reordered_mask = np.absolute(mask_new)
        else:
            reordered_mask = mask_new
        return reordered_mask  
    
    
    def return_n_masks(self,mask_image_tested, number_of_selected_masks=1):
        number_masks = np.max(mask_image_tested)
        mask_new =np.zeros_like(mask_image_tested)
        if number_masks>0:
            # calculate the size of each mask
            mask_sizes = []
            for index_mask in range(1,number_masks+1):
                mask_sizes.append(np.sum(mask_image_tested == index_mask))
            # select the n masks with the largest size and create a new mask with the selected masks
            if number_of_selected_masks < number_masks:
                for i in range(number_of_selected_masks):
                    mask_new = np.where(mask_image_tested == np.argmax(mask_sizes)+1, -i-1, mask_new)
                    mask_sizes[np.argmax(mask_sizes)] = 0
            else:
                mask_new = mask_image_tested
                print('The number of selected masks is larger than the number of masks detected.')
        return np.absolute(mask_new)



    
    # Function that reorder the index to make it continuos 
    def remove_artifacts_from_mask_image(self,mask_image_tested, minimal_mask_area_size = 2000):
        number_masks = np.max(mask_image_tested)
        if number_masks>0:
            for index_mask in range(1,number_masks+1):
                mask_size = np.sum(mask_image_tested == index_mask)
                if mask_size <= minimal_mask_area_size:
                    #mask_image_tested = np.where(mask_image_tested == index_mask, mask_image_tested, 0)
                    mask_image_tested = np.where(mask_image_tested == index_mask,0,mask_image_tested )
            reordered_mask = Utilities().reorder_mask_image(mask_image_tested)
        else:
            reordered_mask=mask_image_tested
        return reordered_mask  
    
    def convert_to_standard_format(self,data_folder_path,path_to_config_file, number_color_channels=2,number_of_fov=1, download_data_from_NAS = True, use_metadata=False, is_format_FOV_Z_Y_X_C=True):
        path_to_masks_dir = None
        # Creating a folder to store all plots
        destination_folder = pathlib.Path().absolute().joinpath('temp_'+data_folder_path.name+'_sf')
        if pathlib.Path.exists(destination_folder):
            shutil.rmtree(str(destination_folder))
            destination_folder.mkdir(parents=True, exist_ok=True)
        else:
            destination_folder.mkdir(parents=True, exist_ok=True)
        local_data_dir, _, _, _, list_files_names_all_fov, list_images_all_fov = Utilities().read_images_from_folder(path_to_config_file, data_folder_path, path_to_masks_dir,  download_data_from_NAS)
        if download_data_from_NAS == False:
            local_data_dir = data_folder_path
        # Downloading data
        if use_metadata == True:
            try:
                metadata = pycro.Dataset(str(local_data_dir))
                number_z_slices = max(metadata.axes['z'])+1
                number_color_channels = max(metadata.axes['channel'])+1
                number_of_fov = max(metadata.axes['position'])+1
                detected_metadata = True
                print('Number of z slices: ', str(number_z_slices), '\n',
                    'Number of color channels: ', str(number_color_channels) , '\n'
                    'Number of FOV: ', str(number_of_fov) , '\n', '\n', '\n')
            except:
                detected_metadata = False
                raise ValueError('The metadata file is not found. Please check the path to the metadata file.')
        if is_format_FOV_Z_Y_X_C == True:
            #_, _, _, _, list_files_names_all_fov, list_images_all_fov = Utilities().read_images_from_folder(path_to_config_file, data_folder_path, path_to_masks_dir,  download_data_from_NAS)
            number_images_all_fov = len(list_files_names_all_fov)
            # Re-arranging the image from shape [FOV, Z, Y, X, C] to multiple tifs with shape [Z, Y, X, C]  FOV_Z_Y_X_C
            list_images_standard_format= []
            list_files_names = []
            number_images =0
            for i in range(number_images_all_fov):
                for j in range (number_of_fov):
                    temp_image_fov = list_images_all_fov[i]
                    if use_metadata == False:
                        number_z_slices = temp_image_fov.shape[0]//2
                        if number_z_slices > 50:
                            raise ValueError('The number of automatically detected z slices is '+str(number_z_slices)+', double-check the number_of_fov and number_color_channels.' )
                    y_shape, x_shape = temp_image_fov.shape[2], temp_image_fov.shape[3]
                    list_files_names.append(  list_files_names_all_fov[i].split(".")[0]+'_fov_'+str(j) +'.tif' )
                    temp_image = np.zeros((number_z_slices,y_shape, x_shape,number_color_channels))
                    
                    temp_image = temp_image_fov[j,:,:,:] # format [Z,Y,X,C]
                    list_images_standard_format.append(temp_image)
                    number_images+=1
            for k in range(number_images):
                tifffile.imsave(str(destination_folder.joinpath(list_files_names[k])), list_images_standard_format[k])
            masks_dir = None
        else:
            if number_of_fov > 1:  
                number_images_all_fov = len(list_files_names_all_fov)
                number_images = 0
                # This option sections a single tif file containing multiple fov.
                # The format of the original FOV is [FOV_0:Ch_0-Ch_1-Z_1...Z_N, ... FOV_N:Ch_0-Ch_1-Z_1...Z_N]  
                for k in range(number_images_all_fov):
                    # Section that separaters all fov into single tif files
                    image_all_fov = list_images_all_fov[k]
                    number_total_images_in_fov = image_all_fov.shape[0]
                    if detected_metadata == False:
                        if (number_total_images_in_fov % (number_color_channels*number_of_fov)) == 0:
                            number_z_slices = int(number_total_images_in_fov / (number_color_channels*number_of_fov))
                        else:
                            raise ValueError('The number of z slices is not defined correctly double-check the number_of_fov and number_color_channels.' )
                        if number_z_slices > 50:
                            raise ValueError('The number of automatically detected z slices is '+str(number_z_slices)+', double-check the number_of_fov and number_color_channels.' )
                    number_elements_on_fov = number_color_channels*number_z_slices
                    list_files_names = []
                    y_shape, x_shape = image_all_fov.shape[1], image_all_fov.shape[2]                
                    # Iterating for each image. Note that the color channels are intercalated in the original image. For that reason a for loop is needed and then selecting even and odd indexes.
                    list_images_standard_format= []
                    counter=0
                    for i in range(number_of_fov):
                        list_files_names.append(  list_files_names_all_fov[k].split(".")[0]+'_img_'+str(k)+'_fov_'+str(i) +'.tif' )
                        temp_image_fov = np.zeros((number_elements_on_fov,y_shape, x_shape))
                        temp_image_fov = image_all_fov[counter*number_elements_on_fov:number_elements_on_fov*(counter+1),:,:]
                        temp_image = np.zeros((number_z_slices,y_shape, x_shape,number_color_channels))
                        for ch in range(number_color_channels):
                            temp_image[:,:,:,ch] = temp_image_fov[ch::number_color_channels,:,:] 
                        list_images_standard_format.append(temp_image)
                        counter+=1
                        number_images+=1
                        del temp_image, temp_image_fov
            elif number_of_fov == 1:
                # This option takes multiple tif files containing multiple images with format [FOV_0:Ch_0-Ch_1-Z_1...Z_N, ... FOV_N:Ch_0-Ch_1-Z_1...Z_N]
                number_images = len(list_files_names_all_fov)
                # Re-arranging the image
                list_images_standard_format= []
                list_files_names = []
                for i in range(number_images):
                    temp_image_fov = list_images_all_fov[i]
                    number_z_slices = temp_image_fov.shape[0]//2
                    if number_z_slices > 50:
                        raise ValueError('The number of automatically detected z slices is '+str(number_z_slices)+', double-check the number_of_fov and number_color_channels.' )
                    y_shape, x_shape = temp_image_fov.shape[1], temp_image_fov.shape[2]
                    list_files_names.append(  list_files_names_all_fov[i].split(".")[0]+'_fov_'+str(i) +'.tif' )
                    temp_image = np.zeros((number_z_slices,y_shape, x_shape,number_color_channels))
                    for ch in range(number_color_channels):
                        temp_image[:,:,:,ch] = temp_image_fov[ch::number_color_channels,:,:] 
                    list_images_standard_format.append(temp_image)
        # Saving images as tif files
        for i in range(number_images):
            tifffile.imsave(str(destination_folder.joinpath(list_files_names[i])), list_images_standard_format[i])
        masks_dir = None
        return destination_folder,masks_dir, list_files_names, list_images_all_fov, list_images_standard_format
    
    def create_output_folders(self,data_folder_path,diameter_nucleus,diameter_cytosol,psf_z,psf_yx,threshold_for_spot_detection,channels_spots,list_threshold_for_spot_detection):
        # testing if the images were merged.
        if data_folder_path.name == 'merged':
            data_folder_path = data_folder_path.parents[0]
        # Testing if a temporal folder was created.
        if data_folder_path.name[0:5] == 'temp_':
            original_folder_name = data_folder_path.name[5:]
        else:
            original_folder_name= data_folder_path.name
        # Creating the output_identification_string
        if (threshold_for_spot_detection is None):
            output_identification_string = original_folder_name+'___nuc_' + str(diameter_nucleus) +'__cyto_' + str(diameter_cytosol) +'__psfz_' + str(psf_z) +'__psfyx_' + str(psf_yx)+'__ts_auto'
        else:
            output_identification_string = original_folder_name +'___nuc_' + str(diameter_nucleus) +'__cyto_' + str(diameter_cytosol) +'__psfz_' + str(psf_z) +'__psfyx_' + str(psf_yx)+'__ts'
            for i in range (len(channels_spots)):
                output_identification_string+='_'+ str(list_threshold_for_spot_detection[i])
                print ('\n Output folder name : ' , output_identification_string)
        # Output folders
        analysis_folder_name = 'analysis_'+ output_identification_string
        # Removing directory if exist
        if os.path.exists(analysis_folder_name):
            shutil.rmtree(analysis_folder_name)
        # Creating the directory
        os.makedirs(analysis_folder_name) 
        return output_identification_string
    
    
    def  create_list_thresholds(self,channels_spots,threshold_for_spot_detection=None):
        # If more than one channel contain spots. This section will create a list of thresholds for spot detection and for each channel. 
        if not(isinstance(channels_spots, list)):
            channels_spots=Utilities().make_it_a_list(channels_spots)
        list_threshold_for_spot_detection=[]
        if not isinstance(threshold_for_spot_detection, list):
            for i in range (len(channels_spots)):
                list_threshold_for_spot_detection.append(threshold_for_spot_detection)
        else:
            list_threshold_for_spot_detection = threshold_for_spot_detection
        # Lists for thresholds. If the list is smaller than the number of spot channels and it uses the same value for all channels.
        if (isinstance(list_threshold_for_spot_detection, list)) and (len(list_threshold_for_spot_detection) < len(channels_spots)):
            for i in range (len(channels_spots)):
                list_threshold_for_spot_detection.append(list_threshold_for_spot_detection[0])
        return list_threshold_for_spot_detection
    
    # This function is intended to merge masks in a single image
    def merge_masks (self,list_masks):
        '''
        This method is intended to merge a list of images into a single image (Numpy array) where each cell is represented by an integer value.
        
        Parameters
        
        list_masks : List of Numpy arrays.
            List of Numpy arrays, where each array has dimensions [Y, X] with values 0 and 1, where 0 represents the background and 1 the cell mask in the image.
        '''
        n_masks = len(list_masks)
        if not ( n_masks is None):
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                base_image = np.zeros_like(list_masks[0])
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    tested_mask = np.where(list_masks[nm-1] == 1, nm, 0)
                    base_image = base_image + tested_mask
            # making zeros all elements outside each mask, and once all elements inside of each mask.
            else:  # do nothing if only a single mask is detected per image.
                base_image = list_masks[0]
        else:
            base_image =[]
        masks = base_image.astype(np.uint8)
        return masks
    
    def separate_masks (self,masks):
        '''
        This method is intended to separate an image (Numpy array) with multiple masks into a list of Numpy arrays where each cell is represented individually in a new NumPy array.
        
        Parameters
        
        masks : Numpy array.
            Numpy array with dimensions [Y, X] with values from 0 to n where n is the number of masks in the image.
        '''
        list_masks = []
        n_masks = np.max(masks)
        if not ( n_masks is None):
            if n_masks > 1: # detecting if more than 1 mask are detected per cell
                for nm in range (1, n_masks+1): # iterating for each mask in a given cell. The mask has values from 0 for background, to int n, where n is the number of detected masks.
                    mask_copy = masks.copy()
                    tested_mask = np.where(mask_copy == nm, 1, 0) # making zeros all elements outside each mask, and once all elements inside of each mask.
                    list_masks.append(tested_mask)
            else:  # do nothing if only a single mask is detected per image.
                list_masks.append(masks)
        else:
            list_masks.append(masks)
        return list_masks
    
   
    def convert_to_int8(self, image, rescale=True, min_percentile=1, max_percentile=98, padding_zeros=True):
        '''
        This method converts images from int16 (or any other type) to uint8. Optionally, the image can be rescaled and stretched.

        Parameters
        ----------
        image : NumPy array
            NumPy array with dimensions [Y, X] or [Y, X, C]. The code expects 3 channels (RGB) for color images. If less than 3 channels are passed, the array can be padded with zeros.
            If a 2D image is passed, the function will return a 2D uint8 image, unless padding_zeros is True.
        rescale : bool, optional
            If True, it rescales the image to stretch intensity values between `min_percentile` and `max_percentile`, and then rescales the min and max intensity to 0 and 255. The default is True.
        min_percentile : float, optional
            The lower percentile to use for intensity scaling. Default is 1.
        max_percentile : float, optional
            The upper percentile to use for intensity scaling. Default is 98.
        padding_zeros : bool, optional
            If True, pads the channel dimension with zeros to reach 3 channels. This applies to both 2D and 3D images. For 2D images, it will create a 3D image with shape [Y, X, 3].
            The default is True.
        '''

        if image.ndim == 2:
            # The image is 2D (Y, X)
            if rescale:
                if np.max(image) > 0:
                    image = RemoveExtrema(image, min_percentile=min_percentile, max_percentile=max_percentile).remove_outliers()
            if np.max(image) > 0:
                temp = image.copy()
                denom = np.max(temp) - np.min(temp)
                if denom == 0:
                    image_new = np.zeros_like(temp)
                else:
                    image_new = ((temp - np.min(temp)) / denom) * 255
            else:
                image_new = image.copy()
            image_new = np.uint8(image_new)

            if padding_zeros:
                # Expand dims to (Y, X, 1)
                image_new = np.expand_dims(image_new, axis=2)
                # Padding with zeros to reach 3 channels
                while image_new.shape[2] < 3:
                    zeros_plane = np.zeros_like(image_new[:, :, 0])
                    image_new = np.concatenate((image_new, zeros_plane[:, :, np.newaxis]), axis=2)
            return image_new
        elif image.ndim == 3:
            # The image has channels (Y, X, C)
            if rescale:
                im_zeros = np.zeros_like(image)
                for ch in range(image.shape[2]):
                    if np.max(image[:, :, ch]) > 0:
                        im_zeros[:, :, ch] = RemoveExtrema(image[:, :, ch], min_percentile=min_percentile, max_percentile=max_percentile).remove_outliers()
                image = im_zeros
            image_new = np.zeros_like(image)
            for i in range(image.shape[2]):  # Iterate over each channel
                if np.max(image[:, :, i]) > 0:
                    temp = image[:, :, i].copy()
                    denom = np.max(temp) - np.min(temp)
                    if denom == 0:
                        image_new[:, :, i] = np.zeros_like(temp)
                    else:
                        image_new[:, :, i] = ((temp - np.min(temp)) / denom) * 255
            image_new = np.uint8(image_new)
            # Padding with zeros to reach 3 channels
            if padding_zeros:
                while image_new.shape[2] < 3:
                    zeros_plane = np.zeros_like(image_new[:, :, 0])
                    image_new = np.concatenate((image_new, zeros_plane[:, :, np.newaxis]), axis=2)
            return image_new
        else:
            raise ValueError("Image must be either 2D or 3D with channels.")

    def read_zipfiles_from_NAS(self,list_dirs,path_to_config_file,share_name,mandatory_substring,local_folder_path):
        # This function iterates over all zip files in a remote directory and download them to a local directory
        list_remote_files=[]
        list_local_files =[]
        if (isinstance(list_dirs, tuple)==False) and (isinstance(list_dirs, list)==False):
            list_dirs = [list_dirs]
        for folder in list_dirs:
            print(folder)
            list_files = NASConnection(path_to_config_file,share_name = share_name).read_files(folder,timeout=60)
            for file in list_files:
                if ('.zip' in file) and (mandatory_substring in file):   # add an argument with re conditions 
                    # Listing all zip files
                    zip_file_path = pathlib.Path().joinpath(folder,file)
                    list_remote_files.append (zip_file_path)
                    list_local_files.append(pathlib.Path().joinpath(local_folder_path,zip_file_path.name)) 
                    # downloading the zip files from NAS
                    NASConnection(path_to_config_file,share_name = share_name).download_file(zip_file_path, local_folder_path,timeout=200)
        return list_local_files
    
    def unzip_local_folders(self,list_local_files,local_folder_path):
        list_local_folders =[]
        for zip_folder in list_local_files:
            # Reads from a list of zip files
            file_to_unzip = zipfile.ZipFile(str(zip_folder)) # opens zip
            temp_folder_name = pathlib.Path().joinpath(local_folder_path, zip_folder.stem)
            if (os.path.exists(temp_folder_name)) :
                shutil.rmtree(temp_folder_name)
                os.makedirs(temp_folder_name) # make a new directory
            # Iterates for each file in zip file
            for file_in_zip in file_to_unzip.namelist():
                # Extracts data to specific folder
                file_to_unzip.extract(file_in_zip,temp_folder_name)
            # Closes the zip file
            file_to_unzip.close()
            # removes the original zip file
            os.remove(pathlib.Path().joinpath(local_folder_path, zip_folder.name))
            list_local_folders.append(temp_folder_name)
        return list_local_folders
    

    def summary_df_by_spot_type(self, df):
        # Grouping by 'image_id' and 'cell_id' and then getting counts of 'spot_type'
        summary_df = df.groupby(['image_id', 'cell_id', 'spot_type']).size().unstack(fill_value=0)
        # Resetting index to make 'image_id' and 'cell_id' columns instead of index
        summary_df.reset_index(inplace=True)
        # Dynamically creating column names based on existing spot_types
        column_names = ['image_id', 'cell_id']
        column_names.extend([f'count_spot_type_{i}' for i in summary_df.columns[2:]])
        # Renaming columns
        summary_df.columns = column_names
        # Output the final DataFrame
        return summary_df
    
    def remove_cells_below_spots_threshold(self, df_detected_spots, threshold_min_number_spots = 100, spot_type = 'all',reorder_cell_ids=True):
        ''' This function takes the dataframe 'df_detected_spots' and a threshold for the minimum number of spots in the cell to be considered for analysis. 
            If the number of spots in the cell is greater than the threshold, the function returns the dataframe with the spots in the cell.
            Otherwise, it removes the cell from the dataframe. use option None to remove cells with less than the threshold in all spot types, 0 for spot type 0, 1 for spot type 1, etc.'''
        if spot_type == 'all':
            number_of_spots = df_detected_spots.groupby('cell_id').size().reset_index(name='counts')
        else:
            number_of_spots = df_detected_spots.loc[df_detected_spots['spot_type'] == spot_type].groupby('cell_id').size().reset_index(name='counts')    
        cells_above_threshold = number_of_spots.loc[number_of_spots['counts'] > threshold_min_number_spots, 'cell_id']
        # Filtering the cells in the dataframe
        df_detected_spots_filtered = df_detected_spots.loc[df_detected_spots['cell_id'].isin(cells_above_threshold)]
        # reorder the cell_id in the dataframe to make it a values from 0 to n
        if reorder_cell_ids == True:
            df_detected_spots_filtered['cell_id'], _ = pd.factorize(df_detected_spots_filtered['cell_id'])
        return df_detected_spots_filtered

    
    def df_trajectories_to_array(
        self, 
        dataframe, 
        selected_field, 
        fill_value='nans', 
        total_frames=None
    ):
        """
        Extracts the selected_field (column) for each particle across frames.

        Parameters:
            dataframe (pandas.DataFrame): Dataframe containing particle tracking data.
            selected_field (str): The field to extract from the dataframe.
            fill_value (str): 'zeros' to fill missing data with 0, 
                            'nans' to fill with np.nan. Default is 'nans'.
            total_frames (int, optional): Force the returned array to have 
                                        'total_frames' columns (i.e. frames).
                                        If None, this defaults to (max_frame + 1).

        Returns:
            data_array (np.ndarray): Array with shape (n_particles, total_frames)
                                    containing the selected field values.
        """

        # Make a copy to avoid modifying the original
        df = dataframe.copy()

        # Check optional columns
        has_image_id = ('image_id' in df.columns)
        has_cell_id  = ('cell_id'  in df.columns)

        # Create a unique label for each particle
        if has_image_id and has_cell_id:
            df['unique_particle'] = (
                df['image_id'].astype(str) + '_' +
                df['cell_id'].astype(str) + '_' +
                df['particle'].astype(str)
            )
        elif has_image_id:
            df['unique_particle'] = (
                df['image_id'].astype(str) + '_' +
                df['particle'].astype(str)
            )
        elif has_cell_id:
            df['unique_particle'] = (
                df['cell_id'].astype(str) + '_' +
                df['particle'].astype(str)
            )
        else:
            df['unique_particle'] = df['particle'].astype(str)

        # Unique ID for each particle
        particles = df['unique_particle'].unique()

        # The highest frame index actually in the data
        max_frame = int(df['frame'].max())  # e.g. 279

        # If total_frames is unspecified, default to max_frame + 1
        if total_frames is None:
            total_frames = max_frame + 1  # e.g. 280 columns

        # Create mapping from particle identifier -> row index
        particle_to_index = {p: i for i, p in enumerate(particles)}
        n_particles = len(particles)

        # Initialize data_array with either np.nan or zeros
        if fill_value == 'zeros':
            data_array = np.zeros((n_particles, total_frames), dtype=float)
        else:
            data_array = np.full((n_particles, total_frames), np.nan, dtype=float)

        # Fill the array
        for particle in particles:
            particle_data = df[df['unique_particle'] == particle]
            frames = particle_data['frame'].values.astype(int)
            values = particle_data[selected_field].values
            row_idx = particle_to_index[particle]

            # Make sure we only assign into columns < total_frames
            frames_in_range = frames[frames < total_frames]

            # Truncate the values array to match the frames_in_range
            # (in case some frames >= total_frames)
            values_to_assign = values[:len(frames_in_range)]

            data_array[row_idx, frames_in_range] = values_to_assign

        # Return just data_array (or also list of particle IDs, if desired)
        return data_array


    def remove_nan_rows(self,array, nan_percentage=0.3):
        """
        Removes rows from the array where the percentage of NaNs exceeds a given threshold.

        Parameters:
            array (np.ndarray): The input array with potential NaN values.
            nan_percentage (float): The threshold percentage of NaNs required to remove a row.
        
        Returns:
            np.ndarray: The array with rows having excessive NaNs removed.
        """
        # Number of columns in the array
        total_columns = array.shape[1]
        
        # Calculate the number of NaNs in each row
        nan_counts = np.sum(np.isnan(array), axis=1)
        
        # Determine the maximum allowed number of NaNs based on the percentage
        max_allowed_nans = total_columns * nan_percentage
        
        # Create a mask for rows where the number of NaNs is less than or equal to the maximum allowed
        mask = nan_counts <= max_allowed_nans
        
        # Filter the array to keep only rows below the NaN threshold
        filtered_array = array[mask]
        
        return filtered_array
    

    
    def df_extract_data(self,dataframe,spot_type, minimum_spots_cluster=2):
        ''' This function is intended to read a dataframe and returns 
            number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, number_of_TS_per_cell, ts_size, cell_size
        '''
        # Number of cells
        number_cells = dataframe['cell_id'].nunique()
        # Number of spots in cytosol
        number_of_spots_per_cell_cytosol = np.asarray([len( dataframe.loc[  (dataframe['cell_id']==i) & (dataframe['is_nuc']==False) & (dataframe['spot_type']==spot_type)  & (dataframe['is_cell_fragmented']!=-1) ].spot_id) for i in range(0, number_cells)])
        # Number of spots in nucleus.  Spots without TS.
        number_of_spots_per_cell_nucleus = np.asarray([len( dataframe.loc[  (dataframe['cell_id']==i) & (dataframe['is_nuc']==True) & (dataframe['spot_type']==spot_type)  & (dataframe['is_cell_fragmented']!=-1)    ].spot_id) for i in range(0, number_cells)])
        # Number of spots
        number_of_spots_per_cell = np.asarray([len( dataframe.loc[  (dataframe['cell_id']==i)  & (dataframe['spot_type']==spot_type) & (dataframe['is_cell_fragmented']!=-1)].spot_id) for i in range(0, number_cells)])
        # Number of TS per cell.
        number_of_TS_per_cell = [len( dataframe.loc[  (dataframe['cell_id']==i) &  (dataframe['is_cluster']==True) & (dataframe['is_nuc']==True) & (dataframe['spot_type']==spot_type)  &   (dataframe['cluster_size']>=minimum_spots_cluster)  & (dataframe['is_cell_fragmented']!=-1)  ].spot_id) for i in range(0, number_cells)]
        number_of_TS_per_cell= np.asarray(number_of_TS_per_cell)
        # Number of RNA in a TS
        ts_size =  dataframe.loc[ (dataframe['is_cluster']==True) & (dataframe['is_nuc']==True)  & (dataframe['spot_type']==spot_type) &   (dataframe['cluster_size']>=minimum_spots_cluster)  & (dataframe['is_cell_fragmented']!=-1)   ].cluster_size.values
        # Size of each cell
        cell_size = [dataframe.loc[   (dataframe['cell_id']==i) ].cell_area_px.values[0] for i in range(0, number_cells)]
        cell_size = np.asarray(cell_size)
        # Cyto size
        cyto_size = [dataframe.loc[   (dataframe['cell_id']==i) ].cyto_area_px.values[0] for i in range(0, number_cells)]
        cyto_size = np.asarray(cyto_size)
        # Size of the nucleus of each cell
        nuc_size = [dataframe.loc[   (dataframe['cell_id']==i) ].nuc_area_px.values[0] for i in range(0, number_cells)]
        nuc_size = np.asarray(nuc_size)
        # removing values less than zeros
        number_of_spots_per_cell.clip(0)
        number_of_spots_per_cell_cytosol.clip(0)
        number_of_spots_per_cell_nucleus.clip(0)
        number_of_TS_per_cell.clip(0)
        ts_size.clip(0)
        return number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, number_of_TS_per_cell, ts_size,cell_size, number_cells, nuc_size, cyto_size
    
    def extracting_data_for_each_df_in_directory(self,list_local_folders, current_dir,spot_type=0, minimum_spots_cluster=2):
        '''
        This method is intended to extract data from the dataframe
        '''
        # Extracting data from dataframe and converting it into lists for each directory.
        list_spots_total=[]
        list_spots_nuc=[]
        list_spots_cytosol=[]
        list_number_cells =[]
        list_transcription_sites =[]
        list_cell_size=[]
        list_nuc_size =[]
        list_dataframes =[]
        list_cyto_size =[]
        for i in range (0, len (list_local_folders)):
            df_dir = current_dir.joinpath('analyses',list_local_folders[i])    # loading files from "analyses" folder
            df_file = glob.glob( str(df_dir.joinpath('df_*')) )[0]
            dataframe = pd.read_csv(df_file)
            # Extracting values from dataframe
            number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, number_of_TS_per_cell, ts_size, cell_size, number_cells, nuc_size, cyto_size = Utilities().df_extract_data(dataframe,spot_type,minimum_spots_cluster=minimum_spots_cluster)            
            # Appending each condition to a list
            list_spots_total.append(number_of_spots_per_cell)  # This list includes spots and TS in the nucleus
            list_spots_nuc.append(number_of_spots_per_cell_nucleus)   #
            list_spots_cytosol.append(number_of_spots_per_cell_cytosol)
            list_number_cells.append(number_cells)
            list_transcription_sites.append(number_of_TS_per_cell)
            list_cell_size.append(cell_size)
            list_nuc_size.append(nuc_size)
            list_dataframes.append(dataframe)
            list_cyto_size.append(cyto_size)
            # Deleting variables
            del number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, number_of_TS_per_cell, ts_size, cell_size, number_cells,nuc_size
        return list_spots_total, list_spots_nuc, list_spots_cytosol, list_number_cells, list_transcription_sites,list_cell_size,list_dataframes,list_nuc_size,list_cyto_size
    
    def extract_data_interpretation(self,list_dirs, path_to_config_file, current_dir, mandatory_substring, local_folder_path, list_labels, share_name='share',minimum_spots_cluster=2, connect_to_NAS=0, spot_type=0, remove_extreme_values=False):
        if connect_to_NAS == True:
            # Reading the data from NAS, unziping files, organizing data as single dataframe for comparison. 
            list_local_files = Utilities().read_zipfiles_from_NAS(list_dirs,path_to_config_file,share_name, mandatory_substring, local_folder_path)
            list_local_folders = Utilities().unzip_local_folders(list_local_files,local_folder_path)
        else: 
            list_local_folders = list_dirs # Use this line to process files from a local repository
        # Extracting data from each repository
        list_spots_total, list_spots_nuc, list_spots_cytosol, list_number_cells, list_transcription_sites, list_cell_size, list_dataframes, list_nuc_size, list_cyto_size = Utilities().extracting_data_for_each_df_in_directory(  list_local_folders=list_local_folders,current_dir=current_dir,spot_type=spot_type,minimum_spots_cluster=minimum_spots_cluster)
        # Final dataframes for nuc, cyto and total spots
        df_all = Utilities().convert_list_to_df (list_number_cells, list_spots_total, list_labels, remove_extreme_values= remove_extreme_values)
        df_cyto = Utilities().convert_list_to_df (list_number_cells, list_spots_cytosol, list_labels, remove_extreme_values= remove_extreme_values)
        df_nuc = Utilities().convert_list_to_df (list_number_cells, list_spots_nuc, list_labels, remove_extreme_values= remove_extreme_values)
        df_transcription_sites = Utilities().convert_list_to_df (list_number_cells, list_transcription_sites, list_labels, remove_extreme_values= remove_extreme_values)
        return df_all, df_cyto, df_nuc, df_transcription_sites, list_spots_total, list_spots_nuc, list_spots_cytosol, list_number_cells, list_transcription_sites, list_cell_size, list_dataframes, list_nuc_size, list_cyto_size 
    
    def function_get_df_columns_as_array(self,df, colum_to_extract, extraction_type='all_values'):
        '''This method is intended to extract a column from a dataframe and convert its values to an array format.
            The argument <<<extraction_type>>> accepts two possible values. 
                values_per_cell: this returns an unique value that represents a cell parameter and is intended to be used with the following columns 
                        'nuc_int_ch", cyto_int_ch', 'nuc_loc_y', 'nuc_loc_x', 'cyto_loc_y', 'cyto_loc_x', 'nuc_area_px', 'cyto_area_px', 'cell_area_px'
                all_values: this returns all fields in the dataframe for the specified column.  
        '''
        number_cells = df['cell_id'].nunique()
        if extraction_type == 'values_per_cell':
            return np.asarray( [       df.loc[(df['cell_id']==i)][colum_to_extract].values[0]            for i in range(0, number_cells)] )
        elif extraction_type == 'all_values' :
            return np.asarray( [       df.loc[(df['cell_id']==i)][colum_to_extract].values          for i in range(0, number_cells)] )      
    
    def convert_list_to_df (self,list_number_cells, list_spots, list_labels, remove_extreme_values= False,max_quantile=0.98) :
        # defining the dimensions for the array.
        max_number_cells = max(list_number_cells)
        number_conditions = len(list_number_cells)
        # creating an array with the same dimensions
        spots_array = np.empty((max_number_cells, number_conditions))
        spots_array[:] = np.NaN
        # replace the elements in the array
        for i in range(0, number_conditions ):
            spots_array[0:list_number_cells[i], i] = list_spots[i] 
        # creating a dataframe
        df = pd.DataFrame(data = spots_array, columns=list_labels)
        # Removing 1% extreme values.
        if remove_extreme_values == True:
            for col in df.columns:
                max_data_value = df[col].quantile(max_quantile)
                df[col] = np.where(df[col] >= max_data_value, np.nan, df[col])
        return df

    def download_data_NAS(self,path_to_config_file,data_folder_path, path_to_masks_dir,share_name,timeout=200):
        '''
        This method is inteded to download data from a NAS. to a local directory.
        path_to_config_file
        data_folder_path
        path_to_masks_dir
        share_name,timeout
        '''
        # Downloading data from NAS
        local_folder_path = pathlib.Path().absolute().joinpath('temp_' + data_folder_path.name)
        NASConnection(path_to_config_file,share_name = share_name).copy_files(data_folder_path, local_folder_path,timeout=timeout)
        local_data_dir = local_folder_path     # path to a folder with images.
        # Downloading masks from NAS
        if not (path_to_masks_dir is None):
            local_folder_path_masks = pathlib.Path().absolute().joinpath( path_to_masks_dir.stem  )
            zip_file_path = local_folder_path_masks.joinpath( path_to_masks_dir.stem +'.zip')
            NASConnection(path_to_config_file,share_name = share_name).download_file(path_to_masks_dir, local_folder_path_masks,timeout=timeout)
            # Unzip downloaded images and update mask directory
            file_to_unzip = zipfile.ZipFile(str(zip_file_path)) # opens zip
            # Iterates for each file in zip file
            for file_in_zip in file_to_unzip.namelist():
                # Extracts data to specific folder
                file_to_unzip.extract(file_in_zip,local_folder_path_masks)
            # Closes the zip file
            file_to_unzip.close()
            # removes the original zip file
            os.remove(zip_file_path)
            masks_dir = local_folder_path_masks
        else:
            masks_dir = None
        return local_data_dir, masks_dir
    
    def read_images_from_folder(self, path_to_config_file, data_folder_path, path_to_masks_dir=None, download_data_from_NAS=False, substring_to_detect_in_file_name = '.*_C0.tif'):
        # Download data from NAS
        if download_data_from_NAS == True:
            share_name = 'share'
            local_data_dir, masks_dir = Utilities().download_data_NAS(path_to_config_file,data_folder_path, path_to_masks_dir,share_name,timeout=200)
        else:
            local_data_dir = data_folder_path 
            masks_dir = path_to_masks_dir 
        # Detecting if images need to be merged
        is_needed_to_merge_images = MergeChannels(local_data_dir, substring_to_detect_in_file_name = substring_to_detect_in_file_name, save_figure =1).checking_images()
        if is_needed_to_merge_images == True:
            _, _, number_images, _ = MergeChannels(local_data_dir, substring_to_detect_in_file_name = substring_to_detect_in_file_name, save_figure =1).merge()
            local_data_dir = local_data_dir.joinpath('merged')
            list_images, path_files, list_files_names, number_images = ReadImages(directory= local_data_dir).read()
        else:
            list_images, path_files, list_files_names, number_images = ReadImages(directory= local_data_dir).read()  # list_images, path_files, list_files_names, number_files        
        # Printing image properties
        if len(list_images[0].shape) < 4:
            number_color_channels = None
        else:
            number_color_channels = list_images[0].shape[-1] 
        print('Image shape: ', list_images[0].shape , '\n')
        print('Number of images: ',number_images , '\n')
        print('Local directory with images: ', local_data_dir, '\n')
        return local_data_dir, masks_dir, number_images, number_color_channels, list_files_names,list_images
        
    def save_output_to_folder (self, output_identification_string, data_folder_path,
                                list_files_distributions=None,
                                file_plots_bleed_thru = None,
                                file_plots_int_ratio=None,
                                file_plots_int_pseudo_ratio=None,
                                channels_spots=None,
                                save_pdf_report=True):
        #  Moving figures to the final folder 
        if not (list_files_distributions is None) and (type(list_files_distributions) is list):
            file_plots_distributions = list_files_distributions[0]
            file_plots_cell_size_vs_num_spots = list_files_distributions[1]
            file_plots_cell_intensity_vs_num_spots = list_files_distributions[2]
            file_plots_spot_intensity_distributions = list_files_distributions[3]
            for i in range (len(file_plots_distributions)):
                if not (file_plots_distributions is None):
                    pathlib.Path().absolute().joinpath(file_plots_distributions[i]).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_distributions[i]))
                if not (file_plots_cell_size_vs_num_spots is None):
                    pathlib.Path().absolute().joinpath(file_plots_cell_size_vs_num_spots[i]).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_cell_size_vs_num_spots[i]))
                if not (file_plots_cell_intensity_vs_num_spots is None):
                    pathlib.Path().absolute().joinpath(file_plots_cell_intensity_vs_num_spots[i]).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_cell_intensity_vs_num_spots[i]))
                if not (file_plots_spot_intensity_distributions is None):
                    pathlib.Path().absolute().joinpath(file_plots_spot_intensity_distributions[i]).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_spot_intensity_distributions[i]))
            
        if not (file_plots_bleed_thru is None):
            pathlib.Path().absolute().joinpath(file_plots_bleed_thru).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_bleed_thru))
        if not (file_plots_int_ratio is None):
            pathlib.Path().absolute().joinpath(file_plots_int_ratio).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_int_ratio))
        if not (file_plots_int_pseudo_ratio is None):
            pathlib.Path().absolute().joinpath(file_plots_int_pseudo_ratio).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),file_plots_int_pseudo_ratio))
        
        # all original images
        pathlib.Path().absolute().joinpath('original_images_' + data_folder_path.name +'.pdf').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string    ),'original_images_'+ data_folder_path.name +'.pdf'))
        # all cell images
        for i in range (len(channels_spots)):
            temp_plot_name = 'cells_channel_'+ str(channels_spots[i])+'_'+ data_folder_path.name +'.pdf'
            pathlib.Path().absolute().joinpath(temp_plot_name).rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string    ),temp_plot_name))
        #metadata_path
        pathlib.Path().absolute().joinpath('images_report_'+ data_folder_path.name +'.csv').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),'images_report_'+ data_folder_path.name +'.csv'))
        pathlib.Path().absolute().joinpath('metadata_'+ data_folder_path.name +'.txt').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),'metadata_'+ data_folder_path.name +'.txt'))
        #df_path 
        pathlib.Path().absolute().joinpath('df_' + data_folder_path.name +'.csv').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string),'df_'+ data_folder_path.name +'.csv'))
        #pdf_path 
        if save_pdf_report == True:
            pathlib.Path().absolute().joinpath('pdf_report_' + data_folder_path.name +'.pdf').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string    ),'pdf_report_'+ data_folder_path.name +'.pdf'))
        #pdf_path segmentation 
        pathlib.Path().absolute().joinpath('segmentation_images_' + data_folder_path.name +'.pdf').rename(pathlib.Path().absolute().joinpath(str('analysis_'+ output_identification_string    ),'segmentation_images_'+ data_folder_path.name +'.pdf'))
        return None

    def sending_data_to_NAS(self,output_identification_string, data_folder_path, path_to_config_file, path_to_masks_dir, diameter_nucleus, diameter_cytosol, send_data_to_NAS = False, masks_dir = None, share_name = 'share'):
        # Writing analyses data to NAS
        analysis_folder_name = 'analysis_'+ output_identification_string
        if send_data_to_NAS == True:
            shutil.make_archive(analysis_folder_name,'zip',pathlib.Path().absolute().joinpath(analysis_folder_name))
            local_file_to_send_to_NAS = pathlib.Path().absolute().joinpath(analysis_folder_name+'.zip')
            NASConnection(path_to_config_file,share_name = share_name).write_files_to_NAS(local_file_to_send_to_NAS, data_folder_path)
            os.remove(pathlib.Path().absolute().joinpath(analysis_folder_name+'.zip'))
        # Writing masks to NAS
        ## Creating mask directory name
        if path_to_masks_dir == None: 
            mask_folder_created_by_pipeline = 'masks_'+ data_folder_path.name # default name by pipeline
            name_final_masks = data_folder_path.name +'___nuc_' + str(diameter_nucleus) + '__cyto_' + str(diameter_cytosol) 
            mask_dir_complete_name = 'masks_'+ name_final_masks # final name for masks dir
            shutil.move(mask_folder_created_by_pipeline, mask_dir_complete_name ) # remaing the masks dir
        elif masks_dir is None:
            mask_dir_complete_name = None
        else: 
            mask_dir_complete_name = masks_dir.name
        ## Sending masks to NAS
        if (send_data_to_NAS == True) and (path_to_masks_dir == None) :
            shutil.make_archive( mask_dir_complete_name , 'zip', pathlib.Path().absolute().joinpath(mask_dir_complete_name))
            local_file_to_send_to_NAS = pathlib.Path().absolute().joinpath(mask_dir_complete_name+'.zip')
            NASConnection(path_to_config_file,share_name = share_name).write_files_to_NAS(local_file_to_send_to_NAS, data_folder_path)
            os.remove(pathlib.Path().absolute().joinpath(mask_dir_complete_name+'.zip'))
        return analysis_folder_name, mask_dir_complete_name
    
    def move_results_to_analyses_folder(self, output_identification_string,  data_folder_path,mask_dir_complete_name,path_to_masks_dir, save_filtered_images = False, download_data_from_NAS = False):
        # Moving all results to "analyses" folder
        if not os.path.exists(str('analyses')):
            os.makedirs(str('analyses'))
        # Subfolder name
        analysis_folder_name = 'analysis_'+ output_identification_string
        final_dir_name =pathlib.Path().absolute().joinpath('analyses', analysis_folder_name)
        # Removing directory if exist
        if os.path.exists(str(final_dir_name)):
            shutil.rmtree(str(final_dir_name))
        # Moving results to a subdirectory in 'analyses' folder
        pathlib.Path().absolute().joinpath(analysis_folder_name).rename(final_dir_name )
        # Moving masks to a subdirectory in 'analyses' folder
        if (download_data_from_NAS == True) or (path_to_masks_dir == None):
            final_mask_dir_name = pathlib.Path().absolute().joinpath('analyses', mask_dir_complete_name)
            if os.path.exists(str(final_mask_dir_name)):
                shutil.rmtree(str(final_mask_dir_name))
            pathlib.Path().absolute().joinpath(mask_dir_complete_name).rename(final_mask_dir_name )
        if save_filtered_images == True:
            filtered_folder_name = 'filtered_images_' + data_folder_path.name 
            pathlib.Path().absolute().joinpath(filtered_folder_name).rename(pathlib.Path().absolute().joinpath('analyses',str('analysis_'+ output_identification_string    ),filtered_folder_name))
        # Delete local temporal files
        temp_results_folder_name = pathlib.Path().absolute().joinpath('temp_results_' + data_folder_path.name)
        shutil.rmtree(temp_results_folder_name)
        # Removing directory if exist
        std_format_folder_name = 'temp_'+data_folder_path.name+'_sf'
        std_format_folder_name_dir_name =pathlib.Path().absolute().joinpath(std_format_folder_name)
        if os.path.exists(str(std_format_folder_name_dir_name)):
            shutil.rmtree(str(std_format_folder_name_dir_name))
        
        if (download_data_from_NAS == True):
            # Delete temporal images downloaded from NAS
            shutil.rmtree('temp_'+data_folder_path.name)
        return None
    
    def export_data_to_CSV(self,list_spots_total, list_spots_nuc, list_spots_cytosol, destination_folder, plot_title_suffix=''):
        # Exporting data to CSV. 
        # ColumnA = time, 
        # ColumnB= #RNA in nucleus, 
        # ColumnC = #RNA in cytoplasm, 
        # ColumnD = total RNA.
        num_time_points = len(list_spots_total)
        num_columns = 4 # time, RNA_nuc, RNA_cyto, total
        array_data_spots =  np.empty(shape=(0, num_columns))
        for i in range(0, num_time_points):
            num_cells = len(list_spots_total[i])
            temp_array_data_spots = np.zeros((num_cells,num_columns))
            temp_array_data_spots[:,0] = i
            temp_array_data_spots[:,1] = list_spots_nuc[i] # nuc
            temp_array_data_spots[:,2] = list_spots_cytosol[i] # cyto
            temp_array_data_spots[:,3] = list_spots_total[i] # all spots
            array_data_spots = np.append(array_data_spots, temp_array_data_spots, axis=0)
        array_data_spots.shape
        # final data frame with format for the model
        df_for_model = pd.DataFrame(data=array_data_spots, columns =['time_index', 'RNA_nuc','RNA_cyto','RNA_total'] )
        new_dtypes = {'time_index':int, 'RNA_nuc':int, 'RNA_cyto':int,'RNA_total':int}
        df_for_model = df_for_model.astype(new_dtypes)
        # Save to csv
        df_for_model.to_csv(pathlib.Path().absolute().joinpath(destination_folder,plot_title_suffix+'.csv'))
        return df_for_model
    def extract_images_masks_dataframe( self,data_folder_path, mandatory_substring, path_to_config_file,connect_to_NAS,path_to_masks_dir=None, rescale=False,max_percentile=99.5):
        local_folder_path = pathlib.Path().absolute().joinpath('temp_local__'+data_folder_path.name)
        # This section downloads results including the dataframe
        if connect_to_NAS == True:
            list_local_files = Utilities().read_zipfiles_from_NAS(list_dirs=data_folder_path,path_to_config_file=path_to_config_file,share_name='share', mandatory_substring=mandatory_substring, local_folder_path=local_folder_path)
            list_local_folders = Utilities().unzip_local_folders(list_local_files,local_folder_path)
        else: 
            list_local_folders = data_folder_path # Use this line to process files from a local repository
        # Extracting the dataframe
        df_file_path = glob.glob( str(list_local_folders[0].joinpath('df_*')) )[0]
        dataframe = pd.read_csv(df_file_path)
        # Extracting the dataframe with cell ids
        try:
            df_file_path_metadata = glob.glob( str(list_local_folders[0].joinpath('images_report_*')) )[0]     
            images_metadata = pd.read_csv(df_file_path_metadata)
        except:
            images_metadata = None
        # Extracting Original images
        local_data_dir, masks_dir, number_images, number_color_channels, list_files_names,_ = Utilities().read_images_from_folder( path_to_config_file, data_folder_path = data_folder_path, path_to_masks_dir = path_to_masks_dir,  download_data_from_NAS = connect_to_NAS, substring_to_detect_in_file_name = '.*_C0.tif')        
        # Reading images from folders
        list_images, path_files, list_files_names, _ = ReadImages(directory= local_data_dir).read()
        if not (path_to_masks_dir is None):
            list_masks, path_files_masks, list_files_names_masks, _ = ReadImages(directory= masks_dir).read()
        else:
            list_masks = None
        # Converting the images to int8
        return list_images, list_masks, dataframe, number_images, number_color_channels,list_local_folders,local_data_dir, images_metadata
    
    
    def image_cell_selection(self,cell_id, list_images, dataframe, mask_cell=None, mask_nuc=None, scaling_value_radius_cell=1.1):
        SCALING_RADIUS_NUCLEUS = scaling_value_radius_cell #1.1
        SCALING_RADIUS_CYTOSOL = scaling_value_radius_cell
        # selecting only the dataframe containing the values for the selected field
        df_selected_cell = dataframe.loc[   (dataframe['cell_id']==cell_id)]
        selected_image_id = df_selected_cell.image_id.values[0]
        y_max_image_shape = list_images[selected_image_id].shape[1]-1
        x_max_image_shape = list_images[selected_image_id].shape[2]-1
        # Cell location in image
        scaling_value_radius_cell = scaling_value_radius_cell # use this parameter to increase or decrease the number of radius to plot from the center of the cell.
        nuc_loc_x = df_selected_cell.nuc_loc_x.values[0]
        nuc_loc_y = df_selected_cell.nuc_loc_y.values[0]
        cyto_loc_x = df_selected_cell.cyto_loc_x.values[0]
        cyto_loc_y = df_selected_cell.cyto_loc_y.values[0]
        nuc_radius_px =  int(np.sqrt(df_selected_cell.nuc_area_px.values[0])*SCALING_RADIUS_NUCLEUS)
        cyto_radius_px = int(np.sqrt(df_selected_cell.cyto_area_px.values[0])*SCALING_RADIUS_CYTOSOL)
        # Detecting if a mask for the cytosol was used. If true, the code will plot the complete cell. Else, it will only plot the cell nucleus.
        if cyto_loc_x:
            plot_complete_cell = True
        else:
            plot_complete_cell = False
        if plot_complete_cell == True:
            x_min_value = cyto_loc_x - cyto_radius_px
            x_max_value = cyto_loc_x + cyto_radius_px
            y_min_value = cyto_loc_y - cyto_radius_px
            y_max_value = cyto_loc_y + cyto_radius_px
        else:
            x_min_value = nuc_loc_x - nuc_radius_px
            x_max_value = nuc_loc_x + nuc_radius_px
            y_min_value = nuc_loc_y - nuc_radius_px
            y_max_value = nuc_loc_y + nuc_radius_px
        # making sure that the selection doesnt go outside the limits of the original image
        x_min_value = np.max((0,x_min_value ))
        y_min_value = np.max((0,y_min_value ))
        x_max_value = np.min((x_max_value,x_max_image_shape))
        y_max_value = np.min((y_max_value,y_max_image_shape))
        # coordinates to select in the image 
        subsection_image_selected_cell = list_images[selected_image_id][:,y_min_value: y_max_value,x_min_value:x_max_value,:]
        # coordinates to select in the masks image
        if not (mask_cell is None):
            subsection_mask_cell = mask_cell[y_min_value: y_max_value,x_min_value:x_max_value]
            subsection_mask_cell[0, :] = 0; subsection_mask_cell[-1, :] = 0; subsection_mask_cell[:, 0] = 0; subsection_mask_cell[:, -1] = 0
        else:
            subsection_mask_cell = None
        if not (mask_nuc is None):
            subsection_mask_nuc = mask_nuc[y_min_value: y_max_value,x_min_value:x_max_value]
            subsection_mask_nuc[0, :] = 0; subsection_mask_nuc[-1, :] = 0; subsection_mask_nuc[:, 0] = 0; subsection_mask_nuc[:, -1] = 0
        else:
            subsection_mask_nuc = None 
        # spots
        df_spots = df_selected_cell[['spot_id', 'z', 'y', 'x','is_nuc', 'is_cluster','cluster_size','spot_type']]
        df_spots = df_spots.reset_index(drop=True)
        # Removing columns with -1. 
        df_spots = df_spots[df_spots.spot_id >= 0]
        # Re-organizing the origin of the image based on the subsection.
        df_spots_subsection_coordinates = df_spots.copy()
        df_spots_subsection_coordinates['y'] = df_spots_subsection_coordinates['y'] - y_min_value
        df_spots_subsection_coordinates['x'] = df_spots_subsection_coordinates['x'] - x_min_value
        return subsection_image_selected_cell, df_spots_subsection_coordinates,subsection_mask_cell, subsection_mask_nuc,selected_image_id
    
    
    def extract_spot_location_from_cell(self,df, spot_type=0, min_ts_size= None,z_slice=None):
        df_spots_all_z = df.loc[ (df['spot_type']==spot_type)  & (df['is_cluster']==0) ] 
        number_spots = len (  df_spots_all_z  )
        
        # Locating spots in the dataframe
        if (z_slice is None):
            df_spots = df.loc[ (df['spot_type']==spot_type)  & (df['is_cluster']==0) ] 
        else:
            df_spots = df.loc[ (df['spot_type']==spot_type)  & (df['is_cluster']==0) & (df['z']==z_slice) ] 
        number_spots_selected_z = len (  df_spots  )
        
        if number_spots_selected_z >0:
            y_spot_locations = df_spots['y'].values
            x_spot_locations = df_spots['x'].values
        else:
            y_spot_locations = None
            x_spot_locations = None
        
        # locating the TS in  the dataframe 
        if not (min_ts_size is None):
            df_TS = df.loc[ (df['spot_type']==spot_type)  & (df['is_cluster']==1) & (df['cluster_size']>=min_ts_size) ] 
        else:
            df_TS = df.loc[ (df['spot_type']==spot_type)  & (df['is_cluster']==1)]
        number_TS = len (  df_TS  )
        # TS location
        if number_TS >0:
            y_TS_locations = df_TS['y'].values
            x_TS_locations = df_TS['x'].values
        else:
            y_TS_locations = None
            x_TS_locations = None
        return y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations,number_spots, number_TS,number_spots_selected_z
    
    def spot_crops (self,image,df,number_crops_to_show,spot_size=5):
        number_crops_to_show = np.min((number_crops_to_show, len(df)/2))
        def return_crop(image, x, y,spot_size):
            spot_range = np.linspace(-(spot_size - 1) / 2, (spot_size - 1) / 2, spot_size,dtype=int)
            crop_image = image[y+spot_range[0]:y+(spot_range[-1]+1), x+spot_range[0]:x+(spot_range[-1]+1)].copy()
            return crop_image
        list_crops_0 = []
        list_crops_1 = []
        counter =0
        i =0
        while counter <= number_crops_to_show:
            x_co = df['x'].values[i]
            y_co = df['y'].values[i]
            if (x_co> (spot_size - 1) / 2) and (y_co>(spot_size - 1) / 2):
                crop_0 = return_crop(image=image[:,:,0], x=x_co, y=y_co, spot_size=spot_size)
                crop_1 = return_crop(image=image[:,:,1], x=x_co, y=y_co, spot_size=spot_size)
                list_crops_0.append(crop_0) 
                list_crops_1.append(crop_1)
                counter+=1
            i+=1
        return list_crops_0, list_crops_1



class Plots():
    '''
    This class contains miscellaneous methods to generate plots. No parameters are necessary for this class.
    '''
    def __init__(self):
        pass


    def display_visualization_plot(self, ax, frame_idx, image_data,
                                channels=None, merge=False,
                                min_percent=1.0, max_percent=99.0,
                                particle_id=None, df_tracking=None,
                                detected_spots_frame=None):
        ax.clear(); ax.axis('off')
        # remove old insets
        for a in ax.figure.axes:
            if a is not ax: a.remove()
        # image compose
        nch = image_data.shape[0] if image_data.ndim>2 else 1
        idxs = [i for i,ch in enumerate(channels or [True]*nch) if ch] or [0]
        if merge and len(idxs)>1:
            # merge style from display/tracking
            img = np.zeros((image_data.shape[1],image_data.shape[2],3))
            for i,ch in enumerate(idxs[:3]):
                c = image_data[ch]; lo=np.percentile(c,min_percent); hi=np.percentile(c,max_percent)
                norm=np.clip((c-lo)/(hi-lo+1e-8),0,1)
                color=list_colors_default[ch]; img+=np.dstack([norm*c for c in color])
            img=np.clip(img,0,1); ax.imshow(img)
        else:
            ch=idxs[0]
            c=image_data[ch]; lo=np.percentile(c,min_percent); hi=np.percentile(c,max_percent)
            norm=np.clip((c-lo)/(hi-lo+1e-8),0,1)
            color=list_colors_default[ch]; ax.imshow(np.dstack([norm*c for c in color]))
        # particle inset panels
        if particle_id is not None and df_tracking is not None:
            dfm=df_tracking[(df_tracking['particle']==particle_id)&(df_tracking['frame']==frame_idx)]
            if not dfm.empty:
                r=dfm.iloc[0]; x,y=r['x'],r['y']
                ax.plot(x,y,'o',mfc='none',mec='red',mew=2)
                h,w = image_data.shape[1], image_data.shape[2]
                ws,hs=int(w*0.05),int(h*0.05)
                left,top=int(x-ws/2),int(y-hs/2)
                for i,ch in enumerate(idxs):
                    c=image_data[ch]; lo=np.percentile(c,min_percent); hi=np.percentile(c,max_percent)
                    norm=np.clip((c-lo)/(hi-lo+1e-8),0,1)
                    reg=norm[top:top+hs,left:left+ws]
                    zoom=cv2.resize(reg,(ws*2,hs*2),interpolation=cv2.INTER_NEAREST)
                    color=list_colors_default[ch]; tinted=np.dstack([zoom*c for c in color])
                    axi=ax.inset_axes([0.75,0.75-i*0.3,0.2,0.2]); axi.imshow(tinted); axi.axis('off')
    
    def visualize_image_widget(self, image_TZYXC):
        """
        Visualize the image with widgets to scan time and select color channels.

        Parameters:
        image_TZYXC: numpy array of shape (T, Z, Y, X, C)
        """
        # Compute max projection over Z axis
        max_proj_TYXC = np.max(image_TZYXC, axis=1)  # Shape: (T, Y, X, C)
        # Rearrange dimensions to (T, C, Y, X) for Napari
        max_proj_TCYX = np.transpose(max_proj_TYXC, (0, 3, 1, 2))

        with napari.gui_qt():
            viewer = napari.Viewer()
            # Add the image; Napari will automatically add time and channel sliders
            viewer.add_image(
                max_proj_TCYX,
                name='Max Projection',
                channel_axis=1,
                colormap='gray',
                blending='additive',
                visible=True,
            )

    def plot_matrix_sample_time(self, array1, array2=None, plot_name=None,cmap='hot'):
        """
        Plots one or two matrices. If the second matrix is None, only the first matrix is plotted.
        
        Args:
        array1 (np.ndarray): First matrix to plot.
        array2 (np.ndarray, optional): Second matrix to plot. Defaults to None.
        """
        # Determine the number of subplots
        if array2 is None:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.matshow(array1, aspect='auto', cmap=cmap)
            ax.set_title('First Signal',fontsize=10)
            ax.grid(False)
        else:
            fig, axs = plt.subplots(2, 1, figsize=(10, 4))
            # Plot the first array
            axs[0].matshow(array1, aspect='auto', cmap='hot')
            axs[0].set_title('First Signal',fontsize=10)
            axs[0].grid(False)
            # Plot the second array
            axs[1].matshow(array2, aspect='auto', cmap=cmap)
            axs[1].set_title('Second Signal',fontsize=10)
            axs[1].grid(False)
        plt.tight_layout()  # Adjust layout to prevent overlap
        if plot_name is not None:
            plt.savefig(plot_name, transparent=False, dpi=360, bbox_inches='tight', format='png')
        plt.show()


    
    def Napari_Visualizer(self,tested_image_TZYXC, tracks_df, z_correction=7,channels_spots=0, plot_name='test.gif'):
        viewer = napari.Viewer(ndisplay=3)
        channel_visibility = [channels_spots == i for i in range(2)]  # List of boolean visibility for channels
        number_frames = tested_image_TZYXC.shape[0]
        if number_frames > 30:
            step_size = 2
        else:
            step_size = 1
        def modify_gif_to_loop(filepath):
            reader = imageio.get_reader(filepath)
            frames = [frame for frame in reader]
            writer = imageio.get_writer(filepath, loop=0)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        
        # Add images for each channel
        for i in range(2):
            viewer.add_image(
                tested_image_TZYXC[..., i],
                name=f'Channel {i}',
                colormap=['seismic', 'red'][i], # 'seismic' , 'bwr'
                scale=(1, z_correction, 1, 1),
                visible=channel_visibility[i]
            )

        # Convert DataFrame to numpy once
        track_data = tracks_df[['particle', 'frame', 'z', 'y', 'x']].to_numpy()

        # Add tracks
        tracks_layer = viewer.add_tracks(
            track_data,
            name='Tracks',
            visible=False,
            scale=(1, 1, 1, 1),
            tail_width=2,
            tail_length=0,
            head_length=0
        )
        # Prepare to add points for detected spots at each time point
        point_layers = []
        for t in range(tested_image_TZYXC.shape[0]):
            frame_data = tracks_df[tracks_df['frame'] == t]
            if not frame_data.empty:
                positions = frame_data[['z', 'y', 'x']].to_numpy()
                positions[:, 0] *= z_correction  # Correct Z-axis scaling
                layer = viewer.add_points(
                    positions,
                    name=f'Spots at t={t}',
                    face_color='transparent',
                    edge_color='y',
                    border_width=0.2,
                    size=6,
                    symbol='disc',
                    visible=False  # Start with all points invisible
                )
                point_layers.append((t, layer))
        # Initialize animation
        animation = Animation(viewer)
        animation.capture_keyframe()
        # Loop through time points to capture frames
        for t in range(tested_image_TZYXC.shape[0]):
            viewer.dims.set_current_step(0, t)
            if t % step_size == 0:  # Reduce frequency of keyframe capture
                animation.capture_keyframe(steps=5)
        # Rotate to show 3D view and capture frames
        show_3d_image = False
        if show_3d_image:
            viewer.dims.ndisplay = 3
            viewer.camera.angles = (0, 0, 90)
            animation.capture_keyframe(steps=10)
            viewer.camera.angles = (30, 30, 90)
            animation.capture_keyframe(steps=60)
        # Show tracks progressively
        for t in range(tested_image_TZYXC.shape[0]):
            viewer.dims.set_current_step(0, t)  # Update the time step
            if t % step_size == 0:  # Reduce frequency of keyframe capture
                for time_index, layer in point_layers:
                    layer.visible = (time_index == t)  # Show only the current time point's spots
                animation.capture_keyframe(steps=5)
        # Finalize and save the animation
        animation.animate(plot_name, canvas_only=True)
        modify_gif_to_loop(plot_name)
        # Close the Napari viewer
        viewer.close()
    
    
    def plot_cell_zoom_selected_crop(self, image_TZYXC, df, figsize=(6,6), time_point=0, show_spots=True, use_gaussian_filter=True, show_spots_ids=True, spot_color='w', spot_mark_size=4,
                                    title=None, image_name=None, microns_per_pixel=None, list_channel_order_to_plot=[0,1,2], 
                                    min_percentile=0.01, list_max_percentile=[99.9], save_image=False, zoom_coords=None, use_standard_color_map = False, 
                                    zoom_size=11, selected_spot=None, plot_title=None, show_plot=True, use_max_projection=False):
        font_props = {'size': 12}
        number_color_channels = image_TZYXC.shape[-1]
        image_size_x = image_TZYXC.shape[3]
        image_size_y = image_TZYXC.shape[2]

        if len(list_max_percentile) == 1:
            list_max_percentile = list_max_percentile * number_color_channels

        # Check if time_point is within the valid range
        if time_point is not None:
            if time_point >= image_TZYXC.shape[0]:
                print("Error: 'time_point' is out of the image stack range.")
                return
        if time_point is None:
            image_to_plot = image_TZYXC[0,... ]
        else:
            image_to_plot = image_TZYXC[time_point,... ]

        # Filter data at this time point
        if time_point is None:
            df_time = df
        else:
            df_time = df[df['frame'] == time_point]
        number_spots = len(df_time)
        print('Number of spots detected at this time point: ', number_spots)    

        # Extract spot positions
        if number_spots > 0:
            # If a selected_spot is provided, find its index
            if selected_spot is not None:
                particles_detected = df_time.particle.values
                if selected_spot in particles_detected:
                    index_selected_spot = np.where(particles_detected == selected_spot)[0][0]
                else:
                    time_points_spots_is_present = df[df['particle'] == selected_spot]['frame'].values
                    if time_points_spots_is_present.size > 0:
                        print('Particle : '+ str(selected_spot)+' is only present in the following frames: ', time_points_spots_is_present)
                        selected_spot = None
                    else:
                        print('Particle : '+ str(selected_spot)+' is not present in the dataframe.')
                        print('The particles present in the dataframe at this time_point are : \n', np.sort(particles_detected))
                        selected_spot = None
                x_spot_locations = df_time['x'].values
                y_spot_locations = df_time['y'].values
            else:
                y_spot_locations = df_time['y'].values
                x_spot_locations = df_time['x'].values
        else:
            y_spot_locations = []
            x_spot_locations = []

        # Determine zoom coords if not provided
        if zoom_coords is None:
            if selected_spot is not None and number_spots > 0 and selected_spot in df_time['particle'].values:
                zoom_coords = (int(x_spot_locations[index_selected_spot]) - zoom_size // 2, 
                            int(y_spot_locations[index_selected_spot]) - zoom_size // 2)
            else:
                zoom_coords = None

        # Apply gaussian filter if requested
        if use_gaussian_filter:
            filtered_image = gaussian_filter(image_to_plot, sigma=1)
            # Use max projection along Z (axis=0), resulting shape [Y,X,C]
            max_subsection_image = np.max(filtered_image, axis=0)
        else:
            max_subsection_image = np.max(image_to_plot, axis=0)

        # Define custom colormaps and cmap_list_imagej as previously described
        cmap_list_imagej = [green_colormap, magenta_colormap, yellow_colormap, red_colormap]

        # Create combined image from custom colormaps
        combined_image = np.zeros((image_size_y, image_size_x, 3), dtype=np.float32)

        if number_color_channels > 0:
            channel_0_img = max_subsection_image[:, :, 0]
            min_val_ch0 = np.percentile(channel_0_img, min_percentile)
            max_val_ch0 = np.percentile(channel_0_img, list_max_percentile[0])
            channel_0_img = np.clip(channel_0_img, min_val_ch0, max_val_ch0)
            norm_ch0 = (channel_0_img - min_val_ch0) / (max_val_ch0 - min_val_ch0 + 1e-8)
        else:
            norm_ch0 = None

        # Channel 1 normalized:
        if number_color_channels > 1:
            channel_1_img = max_subsection_image[:, :, 1]
            min_val_ch1 = np.percentile(channel_1_img, min_percentile)
            max_val_ch1 = np.percentile(channel_1_img, list_max_percentile[1])
            channel_1_img = np.clip(channel_1_img, min_val_ch1, max_val_ch1)
            norm_ch1 = (channel_1_img - min_val_ch1) / (max_val_ch1 - min_val_ch1 + 1e-8)
        else:
            norm_ch1 = None

        # Construct combined image from selected channels
        for i, ch in enumerate(list_channel_order_to_plot):
            if ch >= number_color_channels:
                print("Warning: channel order index out of range.")
                continue
            channel_img = max_subsection_image[:, :, ch]
            # Compute percentile-based normalization
            min_val = np.percentile(channel_img, min_percentile)
            max_val = np.percentile(channel_img, list_max_percentile[ch])
            channel_img = np.clip(channel_img, min_val, max_val)
            norm_channel = (channel_img - min_val) / (max_val - min_val + 1e-8)
            # Apply custom colormap
            if use_standard_color_map:
                # use the standard rgb colormap and not viridis
                combined_image [:, :, i ] = norm_channel 
            else:
                colored_channel = cmap_list_imagej[ch](norm_channel)
                colored_channel_rgb = colored_channel[:, :, :3]
                combined_image += colored_channel_rgb

        # Clip final combined image to [0,1]
        combined_image = np.clip(combined_image, 0, 1)

        # Plot the combined image
        fig, ax = plt.subplots(figsize=figsize, facecolor='black')
        ax.set_facecolor('black')
        ax.imshow(combined_image)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add scalebar if microns_per_pixel provided
        if microns_per_pixel:
            scalebar = ScaleBar(dx=microns_per_pixel, units='um', length_fraction=0.2, 
                                location='lower right', box_color='black', color='white', 
                                font_properties=font_props)
            ax.add_artist(scalebar)

        # Plot spots
        if show_spots and number_spots > 0:
            for i in range(number_spots):
                circ = plt.Circle((x_spot_locations[i], y_spot_locations[i]), spot_mark_size, 
                                color=spot_color, fill=False, lw=1)
                ax.add_artist(circ)

        # Plot spot IDs
        if show_spots_ids and number_spots > 0:
            for i, txt in enumerate(df_time['particle'].values):
                ax.annotate(txt, (df_time['x'].iloc[i] + 4, df_time['y'].iloc[i] + 4), 
                            fontsize=8, color='white')

        # Highlight selected spot
        if selected_spot is not None and selected_spot in df_time['particle'].values:
            rec_spot_mark_size = spot_mark_size * 3
            index_selected_spot = np.where(df_time['particle'].values == selected_spot)[0][0]
            rect = plt.Rectangle((x_spot_locations[index_selected_spot] - rec_spot_mark_size, 
                                y_spot_locations[index_selected_spot] - rec_spot_mark_size),
                                rec_spot_mark_size*2, rec_spot_mark_size*2, linewidth=2.5, 
                                edgecolor='w', facecolor='none')
            ax.add_patch(rect)

        # Zoom insets
        if zoom_coords is not None and norm_ch0 is not None and norm_ch1 is not None:
            xz, yz = zoom_coords
            # Clip coords so we don't go out of range
            xz = np.clip(xz, 0, image_size_x - zoom_size)
            yz = np.clip(yz, 0, image_size_y - zoom_size)
            # Extract zoomed regions
            zoomed_ch0 = norm_ch0[yz:yz+zoom_size, xz:xz+zoom_size]
            zoomed_ch1 = norm_ch1[yz:yz+zoom_size, xz:xz+zoom_size]
            zoomed_merged = combined_image[yz:yz+zoom_size, xz:xz+zoom_size, :]
            # Positions for the three insets
            x_position = 0.78
            size_crop = 0.08
            y_position = 0.65
            # Channel 0 inset
            ax_inset_ch0 = fig.add_axes([x_position, y_position, size_crop, size_crop])
            ax_inset_ch0.imshow(zoomed_ch0, cmap='gray')
            rect_ch0 = plt.Rectangle((0, 0), zoom_size, zoom_size, linewidth=1, edgecolor='white', facecolor='none')
            ax_inset_ch0.add_patch(rect_ch0)
            ax_inset_ch0.set_title('Ch 0', fontsize=8, color='white', fontweight='bold')
            ax_inset_ch0.axis('off')
            # Channel 1 inset
            ax_inset_ch1 = fig.add_axes([x_position, y_position - size_crop*1.4, size_crop, size_crop])
            ax_inset_ch1.imshow(zoomed_ch1, cmap='gray')
            rect_ch1 = plt.Rectangle((0, 0), zoom_size, zoom_size, linewidth=1, edgecolor='white', facecolor='none')
            ax_inset_ch1.add_patch(rect_ch1)
            ax_inset_ch1.set_title('Ch 1', fontsize=8, color='white', fontweight='bold')
            ax_inset_ch1.axis('off')
            # Merged inset
            show_merged = False
            if show_merged:
                ax_inset_merged = fig.add_axes([x_position, y_position - size_crop*2.8, size_crop, size_crop])
                ax_inset_merged.imshow(zoomed_merged)
                rect_merged = plt.Rectangle((0, 0), zoom_size, zoom_size, linewidth=1, edgecolor='white', facecolor='none')
                ax_inset_merged.add_patch(rect_merged)
                ax_inset_merged.set_title('Merged', fontsize=8, color='white', fontweight='bold')
                ax_inset_merged.axis('off')
        if title:
            ax.set_title(title, fontsize=16, color='white')
        if plot_title:
            ax.set_title(plot_title, fontsize=16, color='white')
        if save_image and image_name:
            plt.savefig(image_name, transparent=False, dpi=900, bbox_inches='tight', facecolor='black', format='pdf')
        if show_plot:
            plt.show()
        return fig, ax


    def plot_cell_zoom_with_timecourse_horizontal(self, image_TZYXC, df, 
                                    selected_spot=None, time_point=0, 
                                    figsize=(18,6), show_spots=True, 
                                    use_gaussian_filter=True, show_spots_ids=True, 
                                    spot_color='w', spot_mark_size=4,
                                    title=None, image_name=None, microns_per_pixel=None, 
                                    list_channel_order_to_plot=[0,1,2], 
                                    min_percentile=0.01, list_max_percentile=[99.9], 
                                    save_image=False, zoom_coords=None, 
                                    zoom_size=11, plot_title=None, show_plot=True,  use_standard_color_map = False, 
                                    use_max_projection=False):

        font_props = {'size': 12}
        number_color_channels = image_TZYXC.shape[-1]
        image_size_x = image_TZYXC.shape[3]
        image_size_y = image_TZYXC.shape[2]

        if selected_spot is not None:
            frames = sorted(df['frame'].unique())
            spot_int_values_ch0 = []
            spot_int_values_ch1 = []
            for f in frames:
                df_frame_spot = df[(df['frame'] == f) & (df['particle'] == selected_spot)]
                if len(df_frame_spot) > 0:
                    spot_int_values_ch0.append(df_frame_spot['spot_int_ch_0'].iloc[0])
                    spot_int_values_ch1.append(df_frame_spot['spot_int_ch_1'].iloc[0])
                else:
                    spot_int_values_ch0.append(0)
                    spot_int_values_ch1.append(0)

        if len(list_max_percentile) == 1:
            list_max_percentile = list_max_percentile * number_color_channels

        # Check time_point
        if time_point is not None and (time_point < 0 or time_point >= image_TZYXC.shape[0]):
            print("Error: 'time_point' is out of the image stack range.")
            return

        if time_point is None:
            image_to_plot = image_TZYXC[0,...]
            df_time = df
        else:
            image_to_plot = image_TZYXC[time_point,...]
            df_time = df[df['frame'] == time_point]
        number_spots = len(df_time)
        print('Number of spots detected at this time point: ', number_spots)

        # Spots
        if number_spots > 0:
            particles_detected = df_time.particle.values
            if selected_spot is not None and selected_spot in particles_detected:
                index_selected_spot = np.where(particles_detected == selected_spot)[0][0]
            else:
                index_selected_spot = None
            x_spot_locations = df_time['x'].values
            y_spot_locations = df_time['y'].values
        else:
            x_spot_locations = []
            y_spot_locations = []
            index_selected_spot = None

        # Determine zoom coords if not provided
        if zoom_coords is None and selected_spot is not None and index_selected_spot is not None:
            zoom_coords = (int(x_spot_locations[index_selected_spot]) - zoom_size // 2, 
                        int(y_spot_locations[index_selected_spot]) - zoom_size // 2)
        if use_gaussian_filter:
            filtered_image = gaussian_filter(image_to_plot, sigma=1)
            max_subsection_image = np.max(filtered_image, axis=0)
        else:
            max_subsection_image = np.max(image_to_plot, axis=0)

        # Custom colormaps assumed defined: green_colormap, magenta_colormap, yellow_colormap, red_colormap
        cmap_list_imagej = [green_colormap, magenta_colormap, yellow_colormap, red_colormap]

        # Normalize channels for zoom
        if number_color_channels > 0:
            channel_0_img = max_subsection_image[:, :, 0]
            min_val_ch0 = np.percentile(channel_0_img, min_percentile)
            max_val_ch0 = np.percentile(channel_0_img, list_max_percentile[0])
            channel_0_img = np.clip(channel_0_img, min_val_ch0, max_val_ch0)
            norm_ch0 = (channel_0_img - min_val_ch0) / (max_val_ch0 - min_val_ch0 + 1e-8)
        else:
            norm_ch0 = None

        if number_color_channels > 1:
            channel_1_img = max_subsection_image[:, :, 1]
            min_val_ch1 = np.percentile(channel_1_img, min_percentile)
            max_val_ch1 = np.percentile(channel_1_img, list_max_percentile[1])
            channel_1_img = np.clip(channel_1_img, min_val_ch1, max_val_ch1)
            norm_ch1 = (channel_1_img - min_val_ch1) / (max_val_ch1 - min_val_ch1 + 1e-8)
        else:
            norm_ch1 = None

        # Combined image
        combined_image = np.zeros((image_size_y, image_size_x, 3), dtype=np.float32)
        for i, ch in enumerate(list_channel_order_to_plot):
            if ch >= number_color_channels:
                print("Warning: channel order index out of range.")
                continue
            channel_img = max_subsection_image[:, :, ch]
            min_val = np.percentile(channel_img, min_percentile)
            max_val = np.percentile(channel_img, list_max_percentile[ch])
            channel_img = np.clip(channel_img, min_val, max_val)
            norm_channel = (channel_img - min_val) / (max_val - min_val + 1e-8)
            if use_standard_color_map == False:
                colored_channel = cmap_list_imagej[ch](norm_channel)
                colored_channel_rgb = colored_channel[:, :, :3]
                combined_image += colored_channel_rgb
            else:
                combined_image [:, :, i ] = norm_channel
        combined_image = np.clip(combined_image, 0, 1)

        # Figure and gridspec
        fig = plt.figure(figsize=figsize, facecolor='black')
        # width_ratios=[1,2]: first subplot square, second double width than height
        gs = fig.add_gridspec(1, 2, width_ratios=[1,2]) 
        ax = fig.add_subplot(gs[0,0])
        ax_plot = fig.add_subplot(gs[0,1], facecolor='black')

        ax.set_facecolor('black')
        ax.imshow(combined_image)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Scalebar
        if microns_per_pixel:
            scalebar = ScaleBar(dx=microns_per_pixel, units='um', length_fraction=0.2, 
                                location='lower right', box_color='black', color='white', 
                                font_properties=font_props)
            ax.add_artist(scalebar)

        # Spots
        if show_spots and number_spots > 0:
            for i in range(number_spots):
                circ = plt.Circle((x_spot_locations[i], y_spot_locations[i]), spot_mark_size, 
                                color=spot_color, fill=False, lw=1)
                ax.add_artist(circ)

        # Spot IDs
        if show_spots_ids and number_spots > 0:
            for i, txt in enumerate(df_time['particle'].values):
                ax.annotate(txt, (df_time['x'].iloc[i] + 4, df_time['y'].iloc[i] + 4), 
                            fontsize=8, color='white')

        # Highlight selected spot
        if selected_spot is not None and selected_spot in df_time['particle'].values:
            rec_spot_mark_size = spot_mark_size * 3
            rect = plt.Rectangle((x_spot_locations[index_selected_spot] - rec_spot_mark_size, 
                                y_spot_locations[index_selected_spot] - rec_spot_mark_size),
                                rec_spot_mark_size*2, rec_spot_mark_size*2, linewidth=3, 
                                edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        # Plot zoom insets outside the plot area (above second subplot) if zoom_coords available
        if zoom_coords is not None:
            # Clip zoom coords to image bounds
            xz, yz = zoom_coords
            xz = np.clip(xz, 0, image_size_x - zoom_size)
            yz = np.clip(yz, 0, image_size_y - zoom_size)
            # Build list of normalized per-channel arrays
            norm_channels = []
            for ch in list_channel_order_to_plot:
                if ch < number_color_channels:
                    channel_img = max_subsection_image[:, :, ch]
                    lo = np.percentile(channel_img, min_percentile)
                    hi = np.percentile(channel_img, list_max_percentile[ch])
                    clipped = np.clip(channel_img, lo, hi)
                    norm_channels.append((ch, (clipped - lo) / (hi - lo + 1e-8)))
            # Inset sizing and positions
            size_crop = 0.1
            y_pos = 0.9
            x_start = 0.5
            x_gap = size_crop + 0.02
            # Plot each channel inset
            for idx, (ch, norm_arr) in enumerate(norm_channels):
                zoomed = norm_arr[yz:yz+zoom_size, xz:xz+zoom_size]
                ax_inset = fig.add_axes([x_start + idx * x_gap, y_pos, size_crop, size_crop])
                ax_inset.imshow(zoomed, cmap='gray' if norm_arr.ndim==2 else None)
                rect = plt.Rectangle((0, 0), zoom_size, zoom_size,
                                     linewidth=1, edgecolor='white', facecolor='none')
                ax_inset.add_patch(rect)
                ax_inset.set_title(f'Ch {ch}', fontsize=8, color='white', fontweight='bold')
                ax_inset.axis('off')

        if title:
            ax.set_title(title, fontsize=16, color='white')
        if plot_title:
            ax.set_title(plot_title, fontsize=16, color='white')

        # Plot the time course on the right subplot (ax_plot)
        if selected_spot is not None:
            ax_plot.set_facecolor('black')
            if use_standard_color_map == False:
                ax_plot.plot(frames, spot_int_values_ch0, marker='o', color=cmap_list_imagej[0](1.0), label='Ch0', lw = 2)
                ax_plot.plot(frames, spot_int_values_ch1, marker='o', color=cmap_list_imagej[1](1.0), label='Ch1', lw = 2)
            else:
                ax_plot.plot(frames, spot_int_values_ch0, marker='o', color='r', label='Ch0', lw = 2)
                ax_plot.plot(frames, spot_int_values_ch1, marker='o', color='g', label='Ch1', lw = 2)
            ax_plot.set_xlabel('Frame', fontsize=12, color='white')
            ax_plot.set_ylabel('Spot intensity', fontsize=12, color='white')
            ax_plot.tick_params(colors='white', which='both')
            ax_plot.xaxis.label.set_color('white')
            ax_plot.yaxis.label.set_color('white')
            ax_plot.spines['bottom'].set_color('white')
            ax_plot.spines['top'].set_color('white')
            ax_plot.spines['left'].set_color('white')
            ax_plot.spines['right'].set_color('white')
            # Legend at top right of ax_plot
            legend = ax_plot.legend(loc='upper right', facecolor='black', edgecolor='white', fontsize=10)
            for text in legend.get_texts():
                text.set_color('white')
            # grid off
            ax_plot.grid(False)
            # Highlight current frame
            if time_point is not None:
                ax_plot.axvline(x=time_point, color='yellow', linestyle='-', lw=3)
                ch0_val = spot_int_values_ch0[frames.index(time_point)]
                ch1_val = spot_int_values_ch1[frames.index(time_point)]
                ax_plot.text(time_point, ch0_val, f"{ch0_val:.1f}", color='cyan', fontsize=8)
                ax_plot.text(time_point, ch1_val, f"{ch1_val:.1f}", color='magenta', fontsize=8)

        if save_image and image_name:
            image_name += '.png' if not image_name.endswith('.png') else ''
            plt.savefig(image_name, transparent=False, dpi=360, bbox_inches='tight', facecolor='black', format='png')

        if show_plot:
            plt.show()

        return fig, (ax, ax_plot)
    
    
    def plot_cell_zoom_with_timecourse(
        self,
        image_TZYXC,
        df,
        selected_spot=None,
        time_point=0,
        figsize=(18, 6),
        show_spots=True,
        use_gaussian_filter=True,
        show_spots_ids=True,
        spot_color='w',
        spot_mark_size=4,
        title=None,
        image_name=None,
        microns_per_pixel=None,
        list_channel_order_to_plot=[0, 1, 2],
        min_percentile=0.01,
        list_max_percentile=[99.9],
        save_image=False,
        zoom_coords=None,
        zoom_size=11,
        plot_title=None,
        show_plot=True,
        use_standard_color_map=False,
        use_max_projection=False,
        facecolor='black',
        show_legend=True,
        width_ratios=[1, 3],
        frame_rate_sec=1,
        frame_units='s',
    ):
        # —————————————— Styling to match plot_trajectories ——————————————
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"]   = "white"
        # black color for the fonts
        plt.rcParams["text.color"] = "black"
        plt.rcParams["axes.labelcolor"] = "black"
        plt.rcParams["xtick.color"] = "black"
        plt.rcParams["ytick.color"] = "black"

        font_props = {'size': 20}
        n_ch = image_TZYXC.shape[-1]
        # — Build intensity vectors for the selected spot —
        if selected_spot is not None:
            frames = sorted(df['frame'].unique())
            spot_int_ch0 = []
            spot_int_ch1 = []
            for f in frames:
                row = df[(df['frame'] == f) & (df['particle'] == selected_spot)]
                spot_int_ch0.append(
                    row['spot_int_ch_0'].iloc[0] if 'spot_int_ch_0' in row.columns and not row.empty else 0
                )
                spot_int_ch1.append(
                    row['spot_int_ch_1'].iloc[0] if 'spot_int_ch_1' in row.columns and not row.empty else 0
                )

        # — Ensure per-channel percentiles —
        if len(list_max_percentile) == 1:
            list_max_percentile = list_max_percentile * n_ch

        # — Pick the frame to show —
        if time_point is None or time_point >= image_TZYXC.shape[0]:
            img = image_TZYXC[0, ...]
            df_t = df
        else:
            img = image_TZYXC[time_point, ...]
            df_t = df[df['frame'] == time_point]

        # — Max-project + optional Gaussian —
        if use_gaussian_filter:
            filtered = gaussian_filter(img, sigma=1)
            max_proj = np.max(filtered, axis=0)
        else:
            max_proj = np.max(img, axis=0)

        H, W = max_proj.shape[:2]
        combined = np.zeros((H, W, 3), dtype=float)
        cmap_list = [green_colormap, magenta_colormap, yellow_colormap, red_colormap]
        for idx, ch in enumerate(list_channel_order_to_plot):
            if ch >= n_ch:
                continue
            ch_img = max_proj[..., ch]
            vmin = np.percentile(ch_img, min_percentile)
            vmax = np.percentile(ch_img, list_max_percentile[ch])
            norm = np.clip((ch_img - vmin) / (vmax - vmin + 1e-8), 0, 1)
            if not use_standard_color_map:
                rgb = cmap_list[ch](norm)[:, :, :3]
            else:
                rgb = np.zeros_like(combined)
                rgb[..., idx] = norm
            combined += rgb
        combined = np.clip(combined, 0, 1)

        # — Create 1×3 layout: Image | Time-course | Histogram —
        fig = plt.figure(figsize=figsize, facecolor=facecolor)
        gs = fig.add_gridspec(1, 3, width_ratios=[width_ratios[0], width_ratios[1], 1])
        ax      = fig.add_subplot(gs[0, 0])
        ax_tc   = fig.add_subplot(gs[0, 1], facecolor=facecolor)
        ax_hist = fig.add_subplot(gs[0, 2], facecolor=facecolor)

        # — Display combined image —
        ax.set_facecolor('black')
        ax.imshow(combined)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)

        # — Scalebar if desired —
        if microns_per_pixel:
            sb = ScaleBar(dx=microns_per_pixel, units='um', length_fraction=0.2,
                        location='lower right', box_color='black',
                        color='white', font_properties=font_props)
            ax.add_artist(sb)
        # — Define xs/ys always so we can highlight later —
        if not df_t.empty and 'x' in df_t.columns and 'y' in df_t.columns:
            xs = df_t['x'].values
            ys = df_t['y'].values
        else:
            xs = np.array([])
            ys = np.array([])
        if show_spots and xs.size:
            for x, y in zip(xs, ys):
                circ = plt.Circle((x, y), spot_mark_size,
                                edgecolor=spot_color, facecolor='none', lw=1)
                ax.add_patch(circ)
            if show_spots_ids:
                for i, pid in enumerate(df_t['particle'].values):
                    ax.annotate(pid, (xs[i]+4, ys[i]+4), color='white', fontsize=8)
        if selected_spot is not None and selected_spot in df_t['particle'].values:
            idx0 = list(df_t['particle']).index(selected_spot)
            x0, y0 = xs[idx0], ys[idx0]
            rect = plt.Rectangle((x0-spot_mark_size*3, y0-spot_mark_size*3),
                                spot_mark_size*6, spot_mark_size*6,
                                linewidth=3, edgecolor='w', facecolor='none')
            ax.add_patch(rect)
        if title:
            ax.set_title(title, fontsize=16, color='white')
        if plot_title:
            ax.set_title(plot_title, fontsize=16, color='white')
        if selected_spot is not None:
            ax_tc.set_facecolor(facecolor)
            if frame_units == 's':
                xvals = [f * frame_rate_sec for f in frames]
                ax_tc.set_xlabel('Time (s)', fontsize=26, color = 'black')
            elif frame_units == 'min':
                xvals = [(f * frame_rate_sec) / 60 for f in frames]
                ax_tc.set_xlabel('Time (min)', fontsize=26, color = 'black')
            else:
                xvals = frames
                ax_tc.set_xlabel('Frame', fontsize=26, color = 'black')
            ax_tc.plot(xvals, spot_int_ch0, label='Ch0',
                    color='green', linewidth=2)
            if n_ch > 1:
                ax_tc.plot(xvals, spot_int_ch1, label='Ch1',
                        color='k', linewidth=3)
            ax_tc.set_ylabel('Intensity (a.u.)', fontsize=26, color = 'black')
            ax_tc.tick_params(axis='both', which='major', labelsize=24, color = 'black') # font color black for tick_params
            ax_tc.xaxis.label.set_color('black')
            for sp in ax_tc.spines.values():
                sp.set_visible(True); sp.set_linewidth(3); sp.set_color('black')
            if show_legend:
                leg = ax_tc.legend(loc='upper right', fontsize=20, color = 'black')
                leg.get_frame().set_edgecolor('black')
                leg.get_frame().set_linewidth(1.5)
                for txt in leg.get_texts():
                    txt.set_color('black')
            if time_point in frames:
                ax_tc.axvline(xvals[frames.index(time_point)],
                            color='yellow', linestyle='--', lw=1)
            ax_tc.set_xlim(-50, 1850)
        if selected_spot is not None:
            ax_hist.hist(spot_int_ch0, bins=100, orientation='horizontal',
                        color='green', alpha=0.9)
            ax_hist.set_xlabel('Counts', fontsize=26, color = 'black')
            ax_hist.set_ylabel('Intensity (a.u.)', fontsize=26, color = 'black')
            ax_hist.set_ylim(ax_tc.get_ylim())
            ax_hist.tick_params(axis='both', which='major', labelsize=24, color = 'black') # black color font 
            ax_hist.xaxis.label.set_color('black')
            for sp in ax_hist.spines.values():
                sp.set_visible(True); sp.set_linewidth(3); sp.set_color('black')
            ax_hist.grid(False)
        plt.tight_layout()
        if save_image and image_name:
            plt.savefig(image_name, dpi=360, bbox_inches='tight',
                        facecolor=facecolor, format='pdf')
        if show_plot:
            plt.show()

        return fig, (ax, ax_tc, ax_hist)
    

    def plot_histograms_from_df(
        self,
        df_tracking,
        selected_field,
        axes=None,
        figsize=(7, 4),
        plot_name='temp.png',
        save_plot=False,
        bin_count=100,
        list_colors=['r', 'g', 'b', 'm', 'c'],
        remove_outliers=False
        ):
        plt.rcdefaults()
        pattern = re.compile(r'{}_ch_(\d+)'.format(selected_field))
        channel_indices = sorted(set(
            int(match.group(1)) for column in df_tracking.columns
            if (match := pattern.search(column)) is not None
        ))
        array_list = []
        titles = []
        x_labels = []
        y_labels = []
        list_mean_values = []
        for ch in channel_indices:
            field_name = f'{selected_field}_ch_{ch}'
            if field_name in df_tracking.columns:
                data_array = Utilities().df_trajectories_to_array(df_tracking, selected_field=field_name, fill_value='nans')
                # remove nans from the array
                array_list.append(data_array)
                titles.append(f'Channel {ch}')
                x_labels.append(selected_field)
                y_labels.append('Frequency')
        if not array_list:
            print("No valid channels found.")
            return []
        number_of_channels = len(array_list)
        if axes is None:
            fig, axes = plt.subplots(1, number_of_channels, figsize=figsize)
            if number_of_channels == 1:
                axes = [axes]  # Make it iterable if there's only one plot.
        else:
            # Ensure axes is always an iterable
            if not isinstance(axes, (list, tuple, np.ndarray)):
                axes = [axes]
        for i, data in enumerate(array_list):
            # Process data
            flat_data = data[~np.isnan(data)]
            if remove_outliers:
                # remove large outliers
                flat_data = flat_data[flat_data < np.percentile(flat_data, 99.9)]
                # remove small outliers
                flat_data = flat_data[flat_data > np.percentile(flat_data, 0.1)]
            sorted_data = np.sort(flat_data)
            # Plotting
            axes[i].hist(sorted_data, bins=bin_count, color=list_colors[i % len(list_colors)], alpha=0.7, rwidth=0.85)
            axes[i].set_title(titles[i], fontsize=12)
            axes[i].set_xlabel(x_labels[i], fontsize=12)
            axes[i].set_ylabel(y_labels[i], fontsize=12)
            # Plotting the mean value as a vertical line
            mean_value = np.nanmean(flat_data)
            list_mean_values.append(mean_value)
            axes[i].axvline(mean_value, color='k', linestyle='dashed', linewidth=1.5)
            # Adding the mean value as label
            axes[i].text(0.95, 0.9, f'Mean: {mean_value:.2f}', color='k', fontsize=6,
                        ha='right', va='center', transform=axes[i].transAxes)
            # Adding the standard deviation as label
            std_value = np.nanstd(flat_data)
            axes[i].text(0.95, 0.8, f'Std: {std_value:.2f}', color='k', fontsize=6,
                        ha='right', va='center', transform=axes[i].transAxes)
            del sorted_data, flat_data
        plt.tight_layout()
        if save_plot:
            plt.savefig(plot_name, dpi=96)
        if axes is None:
            plt.show()
        return list_mean_values



    def plot_crops_properties(self,list_particles_arrays,figsize=(15, 4), plot_name='temp_spot_intensity.png', save_plots=False,selection_threshold=None,label='Mean Intensity',log_scale=False,list_colors = ['r', 'g', 'b', 'm', 'c']):
        """
        Plot mean intensities of spots across multiple channels.

        Args:
        list_particle_arrays (list of np.ndarray): List containing arrays of intensities, snr, psf, etc for each channel.
        plot_name (str): Name of the file to save the plot.
        save_plots (bool): Whether to save the plot to disk.

        Returns:
        None
        """
        plt.rcdefaults()
        fig, ax = plt.subplots(figsize=figsize)
        for idx, intensity_array in enumerate(list_particles_arrays):
            # Compute statistics
            mean_value = np.nanmean(intensity_array, axis=1)
            std_value = np.nanstd(intensity_array, axis=1)
            sem_value = std_value / np.sqrt(intensity_array.shape[1])
            ax.errorbar(range(len(mean_value)), mean_value, yerr=sem_value,
                        fmt='o', color=list_colors[idx % len(list_colors)], linewidth=1,
                        label=f'Channel {idx}',alpha=0.5)
            for i, mean_val in enumerate(mean_value):
                ax.text(i + 0.1, mean_val + 0.1, str(i), fontsize=6, color=list_colors[idx % len(list_colors)])
        if selection_threshold is not None:
            ax.axhline(y=selection_threshold, color='k', linestyle='--', linewidth=1, label='Threshold '+str(selection_threshold))
        ax.set_xlabel('Particle')
        ax.set_ylabel( label)
        ax.legend(loc = 'upper right')
        ax.set_xlim(-0.5, len(mean_value) - 0.5)  # Assumes all channels have the same number of spots
        if log_scale:
            ax.set_yscale('log')
        plt.tight_layout()        
        if save_plots:
            plt.savefig(plot_name, dpi=96)
        plt.show()
        return None


    def plot_autocorrelation(self, mean_correlation, 
                            error_correlation, 
                            lags, 
                            correlations_array = None,
                            time_interval_between_frames_in_seconds=1, 
                            channel_label=0, 
                            index_max_lag_for_fit=None, 
                            start_lag=0,
                            line_color='blue', 
                            plot_title=None, 
                            fit_type='linear', 
                            de_correlation_threshold=0.05, 
                            normalize_plot_with_g0=False, 
                            save_plots=True,
                            plot_name=None,
                            max_lag_index=None,
                            plot_individual_trajectories=False,
                            y_axes_min_max_list_values = None,
                            x_axes_min_max_list_values = None,
                            ):

        def single_exponential_decay(tau, A, tau_c, C):
            return A * np.exp(-tau / tau_c) + C
        start_lag = int(start_lag)
        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        # Normalize correlation if requested
        if normalize_plot_with_g0:
            normalized_correlation = mean_correlation / mean_correlation[start_lag]
        else:
            normalized_correlation = mean_correlation
        # Plot correlation with error bands
        ax.plot(lags[start_lag:], normalized_correlation[start_lag:], 'o-', color=line_color, linewidth=2, label='Mean', alpha=0.5)
        ax.fill_between(lags[start_lag:], 
                        normalized_correlation[start_lag:] - error_correlation[start_lag:], 
                        normalized_correlation[start_lag:] + error_correlation[start_lag:], 
                        color=line_color, alpha=0.1)
        dwell_time = 0
        if fit_type == 'linear':
            decorrelation_successful = False
            if index_max_lag_for_fit is None:
                index_max_lag_for_fit = normalized_correlation.shape[0]
            else: 
                index_max_lag_for_fit = int(index_max_lag_for_fit)
            try:
                decorrelation_successful = True
                de_correlation_threshold_value = normalized_correlation[index_max_lag_for_fit + start_lag]
                print(f"Decorrelation threshold value: {de_correlation_threshold_value}")
            except:
                print('Could not find the decorrelation point automatically. Please provide the index_max_lag_for_fit')
                index_max_lag_for_fit = normalized_correlation.shape[0]
                decorrelation_successful = False
            if decorrelation_successful:
                autocorrelations = normalized_correlation[start_lag:] 
                # Fit only up to the found index
                selected_lags = lags[start_lag+1:start_lag+index_max_lag_for_fit]
                selected_autocorrelations = autocorrelations[1:index_max_lag_for_fit]
                try:
                    slope, intercept, _, _, _ = linregress(selected_lags, selected_autocorrelations)
                    total_lags = np.arange(-1, index_max_lag_for_fit+1)*time_interval_between_frames_in_seconds
                    line = slope * total_lags + intercept
                    dwell_time = (-intercept / slope)
                    
                    dt = time_interval_between_frames_in_seconds
                    proj_lags = np.arange(start_lag, dwell_time + dt, dt)
                    proj_vals = slope * proj_lags + intercept
                    mask = proj_vals >= 0
                    proj_lags = proj_lags[mask]
                    proj_vals = proj_vals[mask]
                    ax.plot(proj_lags, proj_vals, 'r-', label='Linear Fit')
                    #ax.plot(total_lags, line, 'r-', label='Linear Fit')
                    max_value = autocorrelations[0]*0.8
                    text_str = f"Dwell Time: {dwell_time:.1f}"
                    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
                    ax.text(total_lags[-1]/2, max_value, s=text_str, color='black', bbox=props, fontsize=10)
                except:
                    pass
                ax.axvline(x=start_lag, color='r', linestyle='--', linewidth=1)
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
            G_tau = np.nan_to_num(G_tau)  # Ensure no NaNs
            if len(G_tau) < 3:
                print("Not enough data points for fitting.")
            else:
                tail_length = max(1, len(G_tau)//10)
                C_guess = np.mean(G_tau[-tail_length:])
                G0 = G_tau[0]
                A_guess = G0 - C_guess

                # Ensure A_guess is reasonable
                if A_guess == 0:
                    A_guess = (np.max(G_tau) - np.min(G_tau)) / 2.0

                A_guess = max(A_guess, 1e-6)  # Ensure A_guess is not zero

                target_value = C_guess + A_guess / np.e
                idx_tau_c = np.argmin(np.abs(G_tau - target_value))
                if idx_tau_c == 0:
                    tau_c_guess = (taus[-1] / 2) if len(taus) > 1 else 1.0
                else:
                    tau_c_guess = taus[idx_tau_c]
                tau_c_guess = max(tau_c_guess, 1e-6)
                initial_guess = [A_guess, tau_c_guess, C_guess]
                # Try curve_fit with some bounds or increased maxfev
                try:
                    params, _ = curve_fit(single_exponential_decay, taus, G_tau, p0=initial_guess,
                                        maxfev=100000,  # Increase maxfev
                                        bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf])) 
                    # Adjust bounds as needed
                    A_fitted, tau_c_fitted, C_fitted = params
                    G_fitted = single_exponential_decay(taus, *params)
                    G0_fitted = single_exponential_decay(0, A_fitted, tau_c_fitted, C_fitted)
                    print("Fitted G(0):", G0_fitted)
                    threshold_value = de_correlation_threshold
                    below_threshold = np.where(G_fitted < threshold_value)[0]

                    if len(below_threshold) > 0:
                        dw_index = below_threshold[0]
                        dwell_time = taus[dw_index]
                        ax.plot(taus, G_fitted, color='r', linestyle='-', 
                                label=f'Fit: tau_c={tau_c_fitted:.1f}, Decorr={dwell_time:.1f}')
                        ax.plot(dwell_time, G_fitted[dw_index], 'ro', markersize=10)
                        #ax.axvline(x=0, color='r', linestyle='--', linewidth=1)
                        ax.axhline(y=G_fitted[dw_index], color='r', linestyle='--', linewidth=1)
                        if plot_title is None:
                            plot_title = f'Exponential Fit (Signal {channel_label})'
                        ax.set_title(plot_title, fontsize=10)
                    else:
                        print("Could not find a time where G(τ) falls below threshold.")
                except RuntimeError as e:
                    print("Exponential fit failed:", e)
        # plotting individual trajectories.
        if plot_individual_trajectories and correlations_array is not None:    
            for i in range(correlations_array.shape[0]):
                ax.plot(lags[start_lag:], correlations_array[i][start_lag:], '-', color='blue', linewidth=0.1, alpha=0.5)
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
            ax.set_xlim(lags[start_lag]-1, lags[max_lag_index])
        # y axis limits
        if y_axes_min_max_list_values is not None:
            ax.set_ylim(y_axes_min_max_list_values[0], y_axes_min_max_list_values[1])
        # x axis limits
        if x_axes_min_max_list_values is not None:
            ax.set_xlim(x_axes_min_max_list_values[0], x_axes_min_max_list_values[1])
        fig.tight_layout()
        if save_plots and plot_name is not None:
            plt.savefig(plot_name, dpi=96)
        plt.show()
        return dwell_time
    

    def plot_crosscorrelation(self,intensity_array_ch0,intensity_array_ch1,mean_correlation,error_correlation,lags,
                            time_interval_between_frames_in_seconds=1,plot_name='temp_CC.png', 
                            save_plots=False,line_color='blue',plot_title=None,normalize_plot_with_g0=True,
                            #plot_individual_trajectories=False,
                            y_axes_min_max_list_values = None,
                            x_axes_min_max_list_values = None,
                              ):

        fig, axes = plt.subplots(1, 3, figsize=(14, 3))
        font_size = 10
        # plot the intensity of the spots in the trajectories using df_trajectories for each particle
        lags_ch0 = np.arange(intensity_array_ch0.shape[1])* time_interval_between_frames_in_seconds
        axes[0].plot(lags_ch0, intensity_array_ch0.T, c='k', linewidth=0.1, alpha=0.1)
        axes[0].set_xlabel(r"$\tau$(au)")
        axes[0].set_ylabel('Intensity(au)')
        axes[0].set_title('Intensity of Spots (Signal 0)', fontsize=font_size)
        # adding the mean and standard deviation of the intensity
        #time_points_ch0 = np.arange(intensity_array_ch0.shape[1])* time_interval_between_frames_in_seconds
        mean_time_intensity_ch0 = np.nanmean(intensity_array_ch0, axis=0)
        std_time_intensity_values_ch0  = np.nanstd(intensity_array_ch0, axis=0)
        axes[0].plot(lags_ch0, mean_time_intensity_ch0, 'o-', color=line_color,linewidth=2, label='Mean', alpha=0.5)
        axes[0].fill_between(lags_ch0, mean_time_intensity_ch0 - std_time_intensity_values_ch0 , mean_time_intensity_ch0 + std_time_intensity_values_ch0 , color=line_color, alpha=0.1)
        # plot the intensity of the spots in the trajectories using df_trajectories for each particle
        lags_ch1 = np.arange(intensity_array_ch1.shape[1])* time_interval_between_frames_in_seconds
        axes[1].plot(lags_ch1, intensity_array_ch1.T, c='k', linewidth=0.1, alpha=0.1)
        axes[1].set_xlabel(r"$\tau$(au)")
        axes[1].set_ylabel('Intensity (au)')
        axes[1].set_title('Intensity of Spots (Signal 1)', fontsize=font_size)
        mean_time_intensity_ch1 = np.nanmean(intensity_array_ch1, axis=0)
        std_time_intensity_values_ch1 = np.nanstd(intensity_array_ch1, axis=0)
        axes[1].plot(lags_ch1, mean_time_intensity_ch1, 'o-', color=line_color,linewidth=2, label='Mean', alpha=0.5)
        axes[1].fill_between(lags_ch1, mean_time_intensity_ch1 - std_time_intensity_values_ch1, mean_time_intensity_ch1 + std_time_intensity_values_ch1, color=line_color, alpha=0.1) 
        # plot the crosscorrelation function
        
        axes[2].axvline(x=0, color='k', linestyle='-', linewidth=1)
        axes[2].axhline(y=0, color='k', linestyle='-', linewidth=1)
        # find the index where lag is 0
        start_lag = np.where(lags == 0)[0][0]
        if normalize_plot_with_g0:
            normalized_correlation = mean_correlation / mean_correlation[start_lag]
        else:
            normalized_correlation = mean_correlation
        axes[2].plot(lags, normalized_correlation, 'o-', color=line_color,linewidth=2, label='Mean', alpha=0.5)
        axes[2].fill_between(lags, normalized_correlation - error_correlation, normalized_correlation + error_correlation, color='b', alpha=0.1)
        axes[2].set_xlabel(r"$\tau$(au)")
        if normalize_plot_with_g0:
            axes[2].set_ylabel(r"$G(\tau)/G(0)$")
        else:
            axes[2].set_ylabel(r"$G(\tau)$")

        # smooth the crosscorrelation function using a moving average of n points
        number_points_to_smooth = 5
        mean_correlation_smoothed = np.convolve(normalized_correlation, np.ones(number_points_to_smooth)/number_points_to_smooth, mode='same')
        axes[2].plot(lags, mean_correlation_smoothed, color= line_color, label='Smoothed', alpha=0.5)
        if plot_title is None:
            axes[2].set_title('Cross-correlation', fontsize=font_size)
        else:
            axes[2].set_title(plot_title, fontsize=font_size)
        # plot a vertical line in the x value that is the maximum of the crosscorrelation function
        max_lag = lags[np.nanargmax(mean_correlation_smoothed)]
        # value at the maximum of the crosscorrelation function
        max_value = mean_correlation_smoothed[np.nanargmax(mean_correlation_smoothed)]
        axes[2].axvline(x=max_lag, color='r', linestyle='--', linewidth=2)
        # display in a text box the lag value of the maximum of the crosscorrelation function
        text = r'$\tau_{max}$ = ' + str(max_lag) + ' au'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        axes[2].text(lags[1], max_value, s=text, color='black', bbox=props, fontsize=10)
        axes[2].grid(False)
        fig.tight_layout()
        # y_axes_min_max_list_values
        if y_axes_min_max_list_values is not None:
            axes[2].set_ylim(y_axes_min_max_list_values[0], y_axes_min_max_list_values[1])
        # x_axes_min_max_list_values
        if x_axes_min_max_list_values is not None:
            axes[2].set_xlim(x_axes_min_max_list_values[0], x_axes_min_max_list_values[1])
        plt.show()
        if save_plots:
            fig.savefig(plot_name, dpi=300)
        return max_lag


    def plot_merged_trajectories(self,df_trajectories_0,df_trajectories_1,df_merged_trajectories,df_overlapping,df_non_overlapping,save_plots=True,plot_name='temp_merged.png'):
        # Calculate the min and max values for x and y to use the same range for all plots
        min_x = min(df_trajectories_0['x'].min(), df_trajectories_1['x'].min())
        max_x = max(df_trajectories_0['x'].max(), df_trajectories_1['x'].max())
        min_y = min(df_trajectories_0['y'].min(), df_trajectories_1['y'].min())
        max_y = max(df_trajectories_0['y'].max(), df_trajectories_1['y'].max())
        # Create a figure and axes with 3 subplots
        plt.figure(figsize=(15, 4))  # Increase width to accommodate three subplots
        # Plot for df_trajectories_0
        plt.subplot(1, 5, 1)
        grouped = df_trajectories_0.groupby('particle')
        for name, group in grouped:
            plt.plot(group['x'], group['y'], marker='o', linestyle='-', linewidth=6, markersize=1, alpha=0.1, color='b')
        plt.title('Channel 0: spots:' + ' ' + str(df_trajectories_0['particle'].nunique()))
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        # invert the y-axis
        plt.gca().invert_yaxis()
        # Plot for df_trajectories_1
        plt.subplot(1, 5, 2)
        grouped = df_trajectories_1.groupby('particle')
        for name, group in grouped:
            plt.plot(group['x'], group['y'], marker='o', linestyle='-', linewidth=6, markersize=1, alpha=0.1, color='r')
        plt.title('Channel 1: spots:' + ' ' + str(df_trajectories_1['particle'].nunique()))
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.gca().invert_yaxis()
        # Plot trajectories 1 and 2
        plt.subplot(1, 5, 3)
        grouped = df_trajectories_0.groupby('particle')
        for name, group in grouped:
            plt.plot(group['x'], group['y'], marker='o', linestyle='-', linewidth=6, markersize=1, alpha=0.1, color='b')
        grouped = df_trajectories_1.groupby('particle')
        for name, group in grouped:
            plt.plot(group['x'], group['y'], marker='o', linestyle='-', linewidth=6, markersize=1, alpha=0.1, color='r')
        plt.title('Merged: spots:' + ' ' + str(df_merged_trajectories['particle'].nunique()))
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.gca().invert_yaxis()
        # Plot overlaping particles
        plt.subplot(1, 5, 4)
        grouped = df_overlapping.groupby('particle')
        for name, group in grouped:
            plt.plot(group['x'], group['y'], marker='o', linestyle='-', linewidth=6, markersize=1, alpha=0.1, color='g')
        plt.title('Overlapping:' + ' ' + str(df_overlapping['particle'].nunique()))
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.tight_layout()
        plt.gca().invert_yaxis()
        # Plot non-overlapping
        plt.subplot(1, 5, 5)
        grouped = df_non_overlapping.groupby('particle')
        for name, group in grouped:
            plt.plot(group['x'], group['y'], marker='o', linestyle='-', linewidth=6, markersize=1, alpha=0.1, color='m')
        plt.title('Non-overlapping:' + ' ' + str(df_non_overlapping['particle'].nunique()))
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.tight_layout()
        plt.gca().invert_yaxis()
        # Save plot if required
        if save_plots:
            plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        plt.show()
        return None

    def plot_pixel_properties(self, list_amplitude, list_sigma_x, list_sigma_y, list_sigma_z, list_offset,
                              crop_size, plot_name='temp.png', save_plots=False):
        plt.rcdefaults()
        number_of_channels = len(list_amplitude)
        fig_height = 4.5 * number_of_channels
        fig = plt.figure(figsize=(14, fig_height))
        subtitles_fotsize = 10

        for ch in range(number_of_channels):
            # Vectors
            amplitude_vector = list_amplitude[ch]
            sigma_x_vector = list_sigma_x[ch]
            sigma_y_vector = list_sigma_y[ch]
            sigma_z_vector = list_sigma_z[ch]
            offset_vector = list_offset[ch]

            # Skip if vectors are empty
            if amplitude_vector.size == 0 or offset_vector.size == 0:
                continue

            # Means
            mean_amplitude = np.mean(amplitude_vector)
            mean_sigma_x = np.mean(sigma_x_vector)
            mean_sigma_y = np.mean(sigma_y_vector)
            mean_sigma_z = np.mean(sigma_z_vector)
            mean_offset = np.mean(offset_vector)

            # Define gridspec
            gs = gridspec.GridSpec(number_of_channels, 5, width_ratios=[0.5, 1, 0.5, 0.5, 0.75])

            ax0 = fig.add_subplot(gs[ch, 0])
            ax1 = fig.add_subplot(gs[ch, 1], projection='3d')
            ax2 = fig.add_subplot(gs[ch, 2])
            ax3 = fig.add_subplot(gs[ch, 3])
            ax4 = fig.add_subplot(gs[ch, 4])

            # Reconstructed 2D spot (central slice)
            size = crop_size
            x = np.linspace(0, size - 1, size)
            y = np.linspace(0, size - 1, size)
            z = np.linspace(0, size - 1, size)
            X, Y = np.meshgrid(x, y)

            # Generate 3D Gaussian data
            def gaussian_3d(x, y, z, amplitude, x0, y0, z0, sigma_x, sigma_y, sigma_z, offset):
                return amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2)
                                          + ((y - y0) ** 2) / (2 * sigma_y ** 2)
                                          + ((z - z0) ** 2) / (2 * sigma_z ** 2))) + offset

            # Central z slice index
            z0 = size // 2

            # Compute the central slice
            Z = gaussian_3d(X, Y, z0, mean_amplitude, size / 2, size / 2, size / 2,
                            mean_sigma_x, mean_sigma_y, mean_sigma_z, mean_offset)

            # Plot central z slice
            ax0.imshow(Z, cmap='binary')
            props = dict(boxstyle='round', facecolor='white', alpha=0.9)
            text_str = f"Sigma X: {mean_sigma_x:.2f}\nSigma Y: {mean_sigma_y:.2f}\nSigma Z: {mean_sigma_z:.2f}"
            ax0.text(0.95, 0.95, text_str, color='black', bbox=props, fontsize=6,
                     ha='right', va='top', transform=ax0.transAxes)
            ax0.axis('off')
            ax0.set_aspect('equal')
            ax0.grid(False)
            ax0.set_title(f'Central Slice of 3D Gaussian, Ch: {ch}', fontsize=subtitles_fotsize)

            # Plotting the 3D Gaussian Isosurface
            X3D, Y3D, Z3D = np.meshgrid(x, y, z)
            V = gaussian_3d(X3D, Y3D, Z3D, mean_amplitude, size / 2, size / 2, size / 2,
                            mean_sigma_x, mean_sigma_y, mean_sigma_z, mean_offset)

            # Flatten the data for plotting
            X_flat = X3D.flatten()
            Y_flat = Y3D.flatten()
            Z_flat = Z3D.flatten()
            V_flat = V.flatten()

            # Define an isosurface value
            iso_value = mean_amplitude * 0.2

            # Select points where V >= iso_value
            indices = V_flat >= iso_value

            # Plot the isosurface using scatter plot
            ax1.scatter(X_flat[indices], Y_flat[indices], Z_flat[indices],
                        c=V_flat[indices], cmap='seismic', alpha=0.5, s=5)
            ax1.set_xlim(0, size - 1)
            ax1.set_ylim(0, size - 1)
            ax1.set_zlim(0, size - 1)
            ax1.set_title(f'3D Gaussian Isosurface, Ch: {ch}', fontsize=subtitles_fotsize)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_zticks([])

            # Adding text annotation inside the plot
            props = dict(boxstyle='round', facecolor='white', alpha=0.9)
            text_str = f"Amplitude: {mean_amplitude:.2f}"
            ax1.text2D(0.05, 0.95, text_str, color='black', bbox=props, fontsize=6,
                       transform=ax1.transAxes)

            max_intensity = np.max((np.max(amplitude_vector), np.max(offset_vector)))
            min_intensity = np.min((np.min(amplitude_vector), np.min(offset_vector)))

            # Ensure min_intensity and max_intensity are finite
            if not np.isfinite(max_intensity) or not np.isfinite(min_intensity):
                continue

            # Plotting the amplitude box plot
            x_positions = np.random.normal(1, 0.02, size=amplitude_vector.shape[0])
            ax2.boxplot(amplitude_vector, patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor='royalblue', alpha=0.2),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        medianprops=dict(color='black', linewidth=1.5))
            ax2.scatter(x_positions, amplitude_vector, color='k', alpha=0.9, s=3)

            if min_intensity > 0:
                ax2.set_ylim(min_intensity * 0.9, max_intensity * 1.1)
            else:
                ax2.set_ylim(min_intensity * 1.15, max_intensity * 1.1)
            ax2.set_title(f'Amplitude, Ch: {ch}', fontsize=subtitles_fotsize)
            ax2.set_xticklabels([''])

            # Plotting the background offset
            x_positions = np.random.normal(1, 0.02, size=offset_vector.shape[0])
            ax3.boxplot(offset_vector, patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor='royalblue', alpha=0.2),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        medianprops=dict(color='black', linewidth=1.5))
            ax3.scatter(x_positions, offset_vector, color='k', alpha=0.9, s=3)
            if min_intensity > 0:
                ax3.set_ylim(min_intensity * 0.9, max_intensity * 1.1)
            else:
                ax3.set_ylim(min_intensity * 1.15, max_intensity * 1.1)
            ax3.set_title(f'Background, Ch: {ch}', fontsize=subtitles_fotsize)
            ax3.set_xticklabels([''])

            # Plotting the sigma_x, sigma_y, sigma_z box plot
            x_positions_sigma_x = np.random.normal(1, 0.02, size=sigma_x_vector.shape[0])
            x_positions_sigma_y = np.random.normal(2, 0.02, size=sigma_y_vector.shape[0])
            x_positions_sigma_z = np.random.normal(3, 0.02, size=sigma_z_vector.shape[0])
            x_positions = np.hstack((x_positions_sigma_x, x_positions_sigma_y, x_positions_sigma_z))
            std_spot_size = np.hstack((sigma_x_vector, sigma_y_vector, sigma_z_vector))
            labels = ['Sigma X', 'Sigma Y', 'Sigma Z']

            ax4.boxplot([sigma_x_vector, sigma_y_vector, sigma_z_vector], positions=[1, 2, 3],
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor='royalblue', alpha=0.2),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        medianprops=dict(color='black', linewidth=1.5))
            ax4.scatter(x_positions, std_spot_size, color='k', alpha=0.9, s=3)
            ax4.set_xticks([1, 2, 3])
            ax4.set_xticklabels(labels)
            ax4.set_title(f'Sigma Values, Ch: {ch}', fontsize=subtitles_fotsize)

        # Add supertitle
        fig.suptitle(f'Spot Properties', fontsize=12)
        plt.tight_layout()

        # Save plot if required
        if save_plots:
            plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        plt.show()
        return None
    

    def plot_croparray(self, croparray, crop_size, plot_name=None, suptitle=None, save_plots=False,
                   show_particle_labels=True, cmap='binary_r', max_percentile=99.5,
                   flag_vector=None, selected_channel=None, axes=None):
        # 1) eliminate NaNs
        croparray = np.nan_to_num(croparray, nan=0.0)

        # 2) cap #particles to avoid huge canvases
        n_particles = croparray.shape[0] // crop_size
        if n_particles > 200:
            n_particles = 200
            croparray = croparray[: n_particles * crop_size, :, :]
        h, w, chans = croparray.shape
        if w > h:
            pad = np.zeros((w - h, w, chans), dtype=croparray.dtype)
            croparray = np.vstack((croparray, pad))
            h = w
        elif h > w:
            pad = np.zeros((h, h - w, chans), dtype=croparray.dtype)
            croparray = np.hstack((croparray, pad))
            w = h
        if axes is None:
            n_axes = 1 if selected_channel is not None else chans + 1
            fig, axes = plt.subplots(1, n_axes, figsize=(5 * n_axes, 5), dpi=150)
            if n_axes == 1:
                axes = [axes]
        else:
            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
        def _safe_vmax(arr):
            p = np.percentile(arr, max_percentile)
            if p <= 0:
                m = arr.max()
                return m if m > 0 else 1.0
            return p
        if selected_channel is None:
            for c in range(chans):
                vmax = _safe_vmax(croparray[:, :, c])
                axes[c].imshow(
                    croparray[:, :, c],
                    cmap=cmap,
                    vmax=vmax,
                    interpolation='none',  # no smoothing
                    aspect='equal'
                )
                axes[c].set_title(f'Ch {c}')
                axes[c].axis('off')
            # merged RGB
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for c in range(min(3, chans)):
                m = croparray[:, :, c].max()
                if m > 0:
                    rgb[:, :, c] = (croparray[:, :, c] / m * 255).astype('uint8')
            vmax = _safe_vmax(rgb)
            axes[chans].imshow(
                rgb,
                vmax=vmax,
                interpolation='none',
                aspect='equal'
            )
            axes[chans].set_title('Merged')
            axes[chans].axis('off')
        else:
            c = selected_channel
            vmax = _safe_vmax(croparray[:, :, c])
            axes[0].imshow(
                croparray[:, :, c],
                cmap=cmap,
                vmax=vmax,
                interpolation='none',
                aspect='equal'
            )
            axes[0].axis('off')
        if show_particle_labels:
            for i in range(n_particles):
                y = i * crop_size + crop_size / 2
                color = 'r' if (flag_vector is not None and i < len(flag_vector) and flag_vector[i]) else 'k'
                for ax in axes:
                    ax.text(-5, y, str(i), color=color, fontsize=5, va='center', ha='right')
        if save_plots and plot_name:
            plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        if axes is None:
            plt.show()

    def plot_average_crops(self, mean_crop, crop_size=11, plot_name=None, plot_orientation='horizontal', 
                        save_plots=False, show_particle_labels=False, cmap='binary_r', max_percentile=99.5, 
                        max_crops_to_display=50, flag_vector=None):
        number_color_channels = mean_crop.shape[2]
        mean_crop_int8 = np.zeros((mean_crop.shape[0], mean_crop.shape[1], 3), dtype='uint8')
        num_particles = mean_crop.shape[0] // crop_size
        num_original_particles = num_particles
        # this section normalizes each crop to the 99.5 percentile of the max value of the crop. This is done to improve visualization.
        normalized_crops = np.zeros_like(mean_crop)
        for ch in range(number_color_channels):
            for i in range(num_particles):
                temp_crop = mean_crop[i*crop_size:(i+1)*crop_size, :, ch] 
                # perform min max normalization
                normalized_crops[i*crop_size: (i+1)*crop_size,:,ch] = ( temp_crop  / np.percentile(temp_crop,max_percentile) ) * 255
        mean_crop = normalized_crops

        # Prepare the data array with NaN padding if necessary
        if num_particles < max_crops_to_display:
            pad_size = (max_crops_to_display - num_particles) * crop_size
            padded_mean_crop = np.full((pad_size, mean_crop.shape[1], mean_crop.shape[2]), np.nan)
            mean_crop = np.concatenate((mean_crop, padded_mean_crop), axis=0)
            num_particles = max_crops_to_display  # Update num_particles to reflect padded size
        else: #       if num_particles > max_crops_to_display:
            print('WARNING: Too many particles to plot. The plot may be too crowded. Only the first '+ str(max_crops_to_display) + ' particles will be plotted.')
            num_particles = max_crops_to_display
            mean_crop = mean_crop[:num_particles*crop_size, :, :]
            mean_crop_int8 = mean_crop_int8[:num_particles*crop_size, :, :]
            num_original_particles = max_crops_to_display

        mean_crop_int8 = np.zeros((mean_crop.shape[0], mean_crop.shape[1], 3), dtype='uint8')
        for ch in range(number_color_channels):
            valid_data = mean_crop[:,:,ch][~np.isnan(mean_crop[:,:,ch])]
            max_value = valid_data.max() if valid_data.size > 0 else 1
            mean_crop_int8[:,:,ch] = (mean_crop[:,:,ch] / max_value * 255).astype('uint8')

        plot_size_crops = (mean_crop.shape[0]//crop_size)*1
        if plot_orientation == 'horizontal':
            figsize = (plot_size_crops, 1.7)
            _, ax = plt.subplots(number_color_channels+1, 1, figsize=figsize)
            for ch in range(number_color_channels):
                # calculate the max value percentile for each channel ignoring NaNs
                max_value_percentile = np.nanpercentile(mean_crop[:,:,ch], max_percentile)
                ax[ch].matshow(mean_crop[:,:,ch].T, cmap=cmap,vmax=max_value_percentile)  # Show the first channel
                ax[ch].set_ylabel(f'Ch{ch}')
                ax[ch].set_xticks([])
                ax[ch].set_yticks([])
                ax[ch].grid(False)
                # Draw a box around flagged particles
                if flag_vector is not None:
                    if ch == 0:
                        for i in range(num_original_particles):
                            if flag_vector[i] == 1:
                                rect = plt.Rectangle((i * crop_size, 0), crop_size, 1, linewidth=3, edgecolor='red', facecolor='none')
                                ax[ch].add_patch(rect)
            # Plotting the two channels together using croparray_int8
            transposed_arr = np.transpose(mean_crop_int8, (1, 0, 2)) # Transpose the array to have the same orientation as the crops
            max_value_percentile = np.nanpercentile(transposed_arr, max_percentile)
            ax[number_color_channels].imshow(transposed_arr, vmax=max_value_percentile)  # Show the combined channels
            ax[number_color_channels].set_xticks([])
            ax[number_color_channels].set_yticks([])
            ax[number_color_channels].grid(False)
            
            if show_particle_labels:
                color_text = 'k'
                for i in range(num_original_particles):
                    x_position = i * crop_size + crop_size / 2  # Center of each crop
                    if flag_vector is not None:
                        if flag_vector[i] == 1:
                            color_text = 'r'
                        else:
                            color_text = 'k'
                    ax[number_color_channels].text(x_position, transposed_arr.shape[0] + 5, f'{i}', ha='center', va='bottom', color=color_text, fontsize=6)
                ax[number_color_channels].set_xlabel('Particle', labelpad=20)
            else:
                ax[number_color_channels].set_xlabel('Particle')
        else:
            figsize = (1.7, plot_size_crops)
            _, ax = plt.subplots(1, number_color_channels+1, figsize=figsize)
            for ch in range(number_color_channels):
                max_value_percentile = np.percentile(mean_crop[:,:,ch], max_percentile)
                ax[ch].matshow(mean_crop[:,:,ch], cmap=cmap,vmax=max_value_percentile)  # Show the first channel
                if ch == 0:
                    ax[ch].set_ylabel('Particle')
                ax[ch].set_title(f'Ch{ch}', fontsize=12)
                ax[ch].set_xticks([])
                ax[ch].set_yticks([])
                ax[ch].grid(False)

                # Draw a box around flagged particles
                if flag_vector is not None:
                    for i in range(num_original_particles):
                        if flag_vector[i] == 1:
                            rect = plt.Rectangle((0, i * crop_size), mean_crop.shape[1], crop_size, linewidth=2, edgecolor='red', facecolor='none')
                            ax[ch].add_patch(rect)
            # Plotting the two channels together using croparray_int8
            max_value_percentile = np.nanpercentile(mean_crop_int8, max_percentile)
            ax[number_color_channels].imshow(mean_crop_int8, vmax=max_value_percentile)  # Show the combined channels
            ax[number_color_channels].set_xticks([])
            ax[number_color_channels].set_yticks([])
            ax[number_color_channels].grid(False)
            if show_particle_labels:
                color_text = 'k'
                for i in range(num_original_particles):
                    y_position = i * crop_size + crop_size / 2  # Center of each crop vertically
                    if flag_vector is not None:
                        if flag_vector[i] == 1:
                            color_text = 'r'
                        else:
                            color_text = 'k'
                    ax[0].text(-5, y_position, f' {i}', rotation=90, ha='right', va='center', color=color_text, fontsize=6)
                ax[0].set_ylabel('Particle', labelpad=20)
            else:
                ax[0].set_ylabel('Particle')
        # save plot
        if save_plots == True:
            plt.savefig(plot_name, dpi=96, bbox_inches='tight')
        plt.show()


    def plot_matrix_crops(self, mean_crop, crop_size=11, plot_name=None, plot_orientation='horizontal',
                        save_plots=False, show_particle_labels=False, cmap='gray', max_percentile=99.5,
                        max_crops_to_display=None, flag_vector=None, max_crops_per_column=10, selected_channel='RGB'):
        """
        Plot the cropped images with the option to select a specific color channel or display the RGB image.
        """

        number_color_channels = mean_crop.shape[-1]
        num_particles = mean_crop.shape[0] // crop_size
        if max_crops_to_display is None:
            max_crops_to_display = num_particles
        num_rows = (min(num_particles, max_crops_to_display) + max_crops_per_column - 1) // max_crops_per_column
        num_cols = min(max_crops_per_column, num_particles)

        figsize = (num_cols * 0.75, num_rows * 0.75)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

        for idx in range(min(num_particles, max_crops_to_display)):
            row, col = divmod(idx, max_crops_per_column)
            ax = axs[row, col]

            crop_img = mean_crop[idx * crop_size: (idx + 1) * crop_size, :, :]

            if selected_channel == 'RGB':
                if number_color_channels == 1:
                    # Duplicate single channel across RGB
                    crop_img = np.repeat(crop_img, 3, axis=-1)
                elif number_color_channels == 2:
                    # Add a third channel filled with zeros
                    zero_channel = np.zeros_like(crop_img[:, :, 0])
                    crop_img = np.stack([crop_img[:, :, 0], crop_img[:, :, 1], zero_channel], axis=-1)
                elif number_color_channels == 3 or number_color_channels == 4:
                    # Use first three channels (ignore alpha if exists)
                    crop_img = crop_img[:, :, :3]

                # Normalize each channel independently
                norm_crop_img = np.zeros_like(crop_img).astype(np.uint8)
                for ch in range(3):
                    ch_min = np.nanmin(crop_img[:, :, ch])
                    ch_max = np.nanmax(crop_img[:, :, ch])
                    ch_range = ch_max - ch_min
                    if ch_range > 0:
                        norm_crop_img[:, :, ch] = ((crop_img[:, :, ch] - ch_min) / ch_range * 255).astype(np.uint8)
                    else:
                        norm_crop_img[:, :, ch] = np.zeros_like(crop_img[:, :, ch])

                ax.imshow(norm_crop_img, aspect='auto')
            else:
                # Display a specific single channel
                if isinstance(selected_channel, int) and selected_channel < number_color_channels:
                    channel_img = crop_img[:, :, selected_channel]
                    ch_min = np.nanmin(channel_img)
                    ch_max = np.nanmax(channel_img)
                    ch_range = ch_max - ch_min
                    norm_channel_img = ((channel_img - ch_min) / ch_range * 255 if ch_range > 0 else np.zeros_like(channel_img)).astype(np.uint8)
                    ax.imshow(norm_channel_img, cmap=cmap, aspect='auto')
                else:
                    print("Warning: Invalid channel selection.")
            ax.set_title(f'id {idx}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axs.ravel()[num_particles:]:
            ax.set_visible(False)  # Hide unused axes
        plt.tight_layout()
        if save_plots and plot_name:
            plt.savefig(plot_name)

        plt.show()
    


    def plot_matrix_pair_crops(self, mean_crop, crop_size=11, plot_name=None, save_plots=False,
                            max_crops_to_display=None, flag_vector=None, selected_channels=(0, 1),
                            spacer_size=2, figure=None, show_text_ds=False):
        """
        Plot the cropped images with the option to select specific color channels.

        Parameters:
        - mean_crop: np.ndarray
            The array containing the mean crops of particles.
        - crop_size: int
            The size of each crop (assumed square).
        - plot_name: str
            The name of the plot file to save (if saving).
        - save_plots: bool
            Whether to save the plot to a file.
        - max_crops_to_display: int
            Maximum number of crops to display.
        - flag_vector: np.ndarray
            A boolean vector indicating which crops to highlight.
        - selected_channels: tuple
            The indices of the channels to display side by side.
        - spacer_size: int
            The size of the spacer between images.
        - figure: matplotlib.figure.Figure
            The figure object to use for plotting.
        - show_text_ds: bool
            Whether to show text descriptions.
        """

        def resize_image_to_target(image, target_size):
            """
            Resize the image to the target size without changing its aspect ratio.
            """
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize(target_size, Image.LANCZOS)
            return np.array(image_pil)

        number_color_channels = mean_crop.shape[-1]
        num_particles = mean_crop.shape[0] // crop_size

        if max_crops_to_display is None:
            max_crops_to_display = num_particles
        num_crops = min(num_particles, max_crops_to_display)

        # Fixed grid size
        fixed_num_cols = 10
        fixed_num_rows = 6
        fixed_total_plots = fixed_num_cols * fixed_num_rows  # 60

        if num_crops > fixed_total_plots:
            # Calculate additional rows needed
            additional_crops = num_crops - fixed_total_plots
            additional_rows = int(np.ceil(additional_crops / fixed_num_cols))
            num_rows = fixed_num_rows + additional_rows
            num_cols = fixed_num_cols
        else:
            num_rows = fixed_num_rows
            num_cols = fixed_num_cols

        # Adjust figure size
        figsize = (num_cols, num_rows * 0.75)

        if figure is None:
            fig = Figure(figsize=figsize)
        else:
            fig = figure
            fig.clear()  # Clear the figure before reusing it
            fig.set_size_inches(figsize)

        # Create a gridspec for subplots
        #canvas = FigureCanvas(fig)  # Needed to render the figure in memory
        axs = fig.subplots(num_rows, num_cols, squeeze=False)

        idx = 0
        total_subplots = num_rows * num_cols
        for row in range(num_rows):
            for col in range(num_cols):
                ax = axs[row, col]
                if idx < num_crops:
                    crop_img = mean_crop[idx * crop_size: (idx + 1) * crop_size, :, :]
                    # Create a combined image with the two selected channels
                    combined_img_list = []
                    for ch in selected_channels:
                        if ch < number_color_channels:
                            channel_img = crop_img[:, :, ch]
                            # Normalize the channel image
                            ch_min = np.nanmin(channel_img)
                            ch_max = np.nanmax(channel_img)
                            ch_range = ch_max - ch_min
                            if ch_range > 0:
                                norm_channel_img = ((channel_img - ch_min) / ch_range * 255).astype(np.uint8)
                            else:
                                norm_channel_img = np.zeros_like(channel_img, dtype=np.uint8)
                            combined_img_list.append(norm_channel_img)
                        else:
                            #print(f"Warning: Channel {ch} not found in crop ID {idx}.")
                            combined_img_list.append(np.zeros_like(crop_img[:, :, 0], dtype=np.uint8))
                    # Create spacer image (white)
                    spacer_value = 255  # White color for spacer
                    spacer_shape = (crop_size, spacer_size)
                    spacer = np.full(spacer_shape, spacer_value, dtype=np.uint8)
                    # Concatenate images with spacer
                    combined_img = np.concatenate([combined_img_list[0], spacer, combined_img_list[1]], axis=1)
                    # Resize combined image
                    target_size = (crop_size * 2 + spacer_size, crop_size)
                    combined_img = resize_image_to_target(combined_img, target_size)
                    # Convert combined_img to RGB
                    combined_img_rgb = np.stack([combined_img, combined_img, combined_img], axis=-1)
                    # Add red line at the top if flag_vector[idx] is True
                    if flag_vector is not None and flag_vector[idx]:
                        start_col = crop_size + spacer_size
                        end_col = start_col + crop_size
                        combined_img_rgb[0:2, start_col:end_col, 0] = 255  # Red channel
                        combined_img_rgb[0:2, start_col:end_col, 1] = 0    # Green channel
                        combined_img_rgb[0:2, start_col:end_col, 2] = 0    # Blue channel

                    # Display the combined image
                    ax.imshow(combined_img_rgb)
                    if show_text_ds:
                        ax.set_title(f'ID {idx}', fontsize=8)
                else:
                    if num_crops < fixed_total_plots:
                        # Fill empty plots if total subplots are fixed
                        empty_image = np.zeros((crop_size, crop_size * 2 + spacer_size, 3), dtype=np.uint8)
                        ax.imshow(empty_image)
                        ax.set_title('Empty', fontsize=8)
                    else:
                        # Hide unused subplots
                        ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal')
                idx += 1

        # Adjust layout
        fig.tight_layout()

        if save_plots and plot_name:
            fig.savefig(plot_name, bbox_inches='tight')


    def plot_single_particle(self, croparray, crop_size, selected_particle_idx, plot_name=None, suptitle=None,
                                    save_plots=False, cmap='binary_r', max_percentile=99.5):
        """
        Plot the croparray image data for a single particle across all color channels vertically.

        Args:
            croparray (ndarray): The croparray image data.
            crop_size (int): The size of each spot in the plot. It must be an odd number.
            selected_particle_idx (int): Index of the particle to plot.
            plot_name (str, optional): The name of the plot. Defaults to None.
            save_plots (bool, optional): Whether to save the plots. Defaults to False.
            show_particle_labels (bool, optional): Whether to show particle labels. Defaults to True.
            cmap (str, optional): The colormap to use. Defaults to 'binary_r'.
            max_percentile (float, optional): The maximum percentile to use for the colormap. Defaults to 99.5.

        Returns:
            None
        """
        plt.rcdefaults()
        # Replace NaNs with zeros in croparray
        croparray = np.nan_to_num(croparray)
        number_color_channels = croparray.shape[-1]
        time_points = croparray.shape[1]
        figsize = ( time_points * 2,2)  # Assuming each channel height of 2 inches and width of 3 inches
        fig, axs = plt.subplots(number_color_channels, 1, figsize=figsize, squeeze=False)
        particle_start_idx = selected_particle_idx * crop_size
        particle_end_idx = particle_start_idx + crop_size
        for ch in range(number_color_channels):
            ax = axs[ch, 0]
            particle_crop = croparray[particle_start_idx:particle_end_idx, :, ch]
            max_value_percentile = np.percentile(particle_crop, max_percentile)
            ax.imshow(particle_crop, cmap=cmap, vmax=max_value_percentile)
            for t in range(time_points):
                rect = Rectangle((t*crop_size, -0.5), crop_size, crop_size, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            ax.set_title(f'Channel {ch}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        if suptitle:
            plt.suptitle(suptitle, fontsize=16)
        plt.tight_layout()
        if save_plots and plot_name:
            plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        plt.show()


    
    def plot_trajectories_and_mask(self, df, masks=None, filtered_image=None, plot_type='both', figsize=(20, 6),x_range=None,y_range=None, z_range=None,show_legend=True,z_view=None,time_point=None,voxel_xy_um=None,max_percentile=99.7,save_plots=False,plot_name='temp_trajectories.png'):
        plt.rcdefaults()
        if 'Cell_ID' in df.columns:
            elements_to_group = ['Cell_ID','particle']
        else:
            elements_to_group = 'particle'
        ###################################
        ########     For 3D plots    ######
        ###################################
        if masks is not None:
            number_masks = np.max(masks)
            list_contours = Utilities().masks_to_contours(masks)
        else:
            number_masks = 0
        max_z = df['z'].max().astype(int)
        
        # function to plot the trajectories
        if plot_type == '3d' or plot_type == 'both':
            fig = plt.figure(figsize=figsize)
            if plot_type == 'both':
                ax3d = fig.add_subplot(131, projection='3d')
            else:
                ax3d = fig.add_subplot(111, projection='3d')
            if number_masks > 0:
                list_contours = Utilities().masks_to_contours(masks)
                for index_mask in range(0, number_masks):
                    downsampled_contour = list_contours[index_mask]
                    # Swap columns to rows
                    downsampled_contour[:, [0, 1]] = downsampled_contour[:, [1, 0]]
                    # Repeat the 2D polygon in the z-direction
                    num_z_layers = max_z
                    z_values = np.linspace(0, num_z_layers, num_z_layers)  # Adjust as needed for the z-spacing
                    vertices_3d = np.zeros((len(downsampled_contour) * num_z_layers, 3))
                    for i, z in enumerate(z_values):
                        vertices_3d[i * len(downsampled_contour):(i + 1) * len(downsampled_contour), :2] = downsampled_contour
                        vertices_3d[i * len(downsampled_contour):(i + 1) * len(downsampled_contour), 2] = z
                    # Connect the vertices to form faces
                    faces = []
                    for i in range(num_z_layers - 1):
                        for j in range(len(downsampled_contour)):
                            a = i * len(downsampled_contour) + j
                            b = i * len(downsampled_contour) + (j + 1) % len(downsampled_contour)
                            c = (i + 1) * len(downsampled_contour) + (j + 1) % len(downsampled_contour)
                            d = (i + 1) * len(downsampled_contour) + j
                            faces.append([a, b, c, d])
                    # Plot XY connections with one color
                    for i in range(num_z_layers):
                        for j in range(len(downsampled_contour) - 1):
                            x = vertices_3d[i * len(downsampled_contour) + j:i * len(downsampled_contour) + j + 2, 0]
                            y = vertices_3d[i * len(downsampled_contour) + j:i * len(downsampled_contour) + j + 2, 1]
                            z = vertices_3d[i * len(downsampled_contour) + j:i * len(downsampled_contour) + j + 2, 2]
                            ax3d.plot(x, y, z, color='b', linewidth=0.5, linestyle='--')
                    # Plot Z connections with another color
                    for i in range(num_z_layers - 1):
                        for j in range(len(downsampled_contour)):
                            x = vertices_3d[i * len(downsampled_contour) + j, 0], vertices_3d[(i + 1) * len(downsampled_contour) + j, 0]
                            y = vertices_3d[i * len(downsampled_contour) + j, 1], vertices_3d[(i + 1) * len(downsampled_contour) + j, 1]
                            z = vertices_3d[i * len(downsampled_contour) + j, 2], vertices_3d[(i + 1) * len(downsampled_contour) + j, 2]
                            ax3d.plot(x, y, z, color='k', linewidth=0.2, linestyle='--')
            if not z_view is None:
                ax3d.view_init(elev=z_view[0], azim=z_view[1])
            else:
                ax3d.view_init(elev=40, azim=80) # Adjust the elevation and azimuth angles as needed
            # Plotting trajectories of particles in 3D
            for particle, group in df.groupby(elements_to_group):
                ax3d.plot(group['x'], group['y'], group['z'], label=f'Particle {particle}', linewidth=1, linestyle='-')
                ax3d.scatter(group['x'].iloc[0], group['y'].iloc[0], group['z'].iloc[0], color='b', marker='s', s=10,alpha=0.5)  # Square marker at the start
            ax3d.set_xlabel('X')
            ax3d.set_ylabel('Y')
            ax3d.set_zlabel('Z')
            ax3d.set_title('Trajectories of Particles (3D)')
            if not x_range is None:
                ax3d.set_xlim(x_range)
            if not y_range is None:
                ax3d.set_ylim(y_range)
            if not z_range is None:
                ax3d.set_zlim(z_range)    
            if show_legend:
                ax3d.legend()
            ax3d.invert_xaxis()
        
        ###################################
        ########    For 2D plots    ######
        ###################################
        
        # Plotting trajectories of particles in 2D
        if plot_type == '2d' or plot_type == 'both':
            if plot_type == 'both':
                ax2d = fig.add_subplot(132)
            else:
                fig = plt.figure(figsize=figsize)
                ax2d = fig.add_subplot(111)
            if number_masks > 0:
                list_contours = Utilities().masks_to_contours(masks)
                for index_mask in range(0, number_masks ):
                    downsampled_contour = list_contours[index_mask]
                    # swaping columns to rows
                    ax2d.plot(downsampled_contour[:, 1], downsampled_contour[:, 0], color='r', linewidth=2, linestyle='-')
            # Plotting particles
            for particle, group in df.groupby(elements_to_group):
                ax2d.plot(group['x'], group['y'], label=f'Particle {particle}')
                ax2d.scatter(group['x'].iloc[0], group['y'].iloc[0], color='b', marker='s', s=10,alpha=0.5)  # Square marker at the start
            ax2d.set_xlabel('X')
            ax2d.set_ylabel('Y')
            ax2d.set_title('Trajectories of Particles (2D)')
            if not x_range is None:
                ax2d.set_xlim(x_range)
            if not y_range is None:
                ax2d.set_ylim(y_range)
            if show_legend:
                ax2d.legend()
            ax2d.invert_yaxis()
            ax2d.set_aspect('equal')
        
        ###################################
        ########  Filtered image     ######
        ###################################

        # Ploting the original image
        if plot_type == 'both':
            ax_img = fig.add_subplot(133)
            if time_point is None:
                max_filtered_image = np.max(filtered_image[0], axis=0)
                ax_img.imshow(max_filtered_image, cmap='binary_r')
            else:
                max_filtered_image = np.max(filtered_image[time_point],axis=0)
            filtered_images_remove_extrema_first = RemoveExtrema(max_filtered_image,min_percentile=0.01, max_percentile=max_percentile).remove_outliers() 
            ax_img.imshow(filtered_images_remove_extrema_first, cmap='binary_r')

            ax_img.set_title('Filtered Image')
            # Plotting all spots detected as circles, filtered by the given time point if specified
            if time_point is not None:
                filtered_df = df[df['frame'] == time_point]
            else:
                filtered_df = df
            for particle, group in filtered_df.groupby(elements_to_group):
                ax_img.scatter(group['x']-1, group['y']-1, label=f'Particle {particle}', s=30, marker='o', linewidths=1, edgecolors='r', facecolors='none')
            if not x_range is None:
                ax_img.set_xlim(x_range)
            if not y_range is None:
                ax_img.set_ylim(y_range)
            if number_masks > 0:
                list_contours = Utilities().masks_to_contours(masks)
                for index_mask in range(0, number_masks ):
                    downsampled_contour = list_contours[index_mask]
                    ax_img.plot(downsampled_contour[:, 1], downsampled_contour[:, 0], color='r', linewidth=2, linestyle='-')
            ax_img.invert_yaxis()
            ax_img.set_aspect('equal')
            # remove grid
            ax_img.grid(False)
            if voxel_xy_um is not None:
                scalebar = ScaleBar(dx = voxel_xy_um, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w')
                ax_img.add_artist(scalebar) 
        plt.tight_layout()
        if save_plots == True:
            plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        plt.show()
        return None
    




    def plot_3d_video_detected_spots(self, original_image, filtered_image, spots=None, df_spots=None, colocalized_spots=None, df_colocalized_spots=None, clusters=None, masks=None, cmap='binary_r', threshold_for_spot_detection=None, voxel_xy_um=None, show_intensity_distribution=False, remove_outliers=True, maximum_percentile=99.7, bins=40, color='orangered', save_plots=False, show_plot=True, image_name='temp.pdf'):
        """
        Plots a 3D video of detected spots.

        Args:
            original_image (ndarray): The original image.
            filtered_image (ndarray): The filtered image.
            spots (ndarray, optional): Array of detected spots. Defaults to None.
            df_spots (DataFrame, optional): DataFrame of detected spots. Defaults to None.
            colocalized_spots (ndarray, optional): Array of colocalized spots. Defaults to None.
            df_colocalized_spots (DataFrame, optional): DataFrame of colocalized spots. Defaults to None.
            clusters (ndarray, optional): Array of clusters. Defaults to None.
            masks (ndarray, optional): Array of masks. Defaults to None.
            cmap (str, optional): The colormap to use for the images. Defaults to 'binary_r'.
            threshold_for_spot_detection (float, optional): The threshold for spot detection. Defaults to None.
            voxel_xy_um (float, optional): The pixel size in micrometers. Defaults to None.
            show_intensity_distribution (bool, optional): Whether to show the intensity distribution. Defaults to False.
            remove_outliers (bool, optional): Whether to remove outliers from the intensity distribution. Defaults to True.
            maximum_percentile (float, optional): The maximum percentile for clipping the intensity distribution. Defaults to 99.7.
            bins (int, optional): The number of bins for the intensity distribution histogram. Defaults to 40.
            color (str, optional): The color for the intensity distribution histogram. Defaults to 'orangered'.
            save_plots (bool, optional): Whether to save the plots to a file. Defaults to False.
            show_plot (bool, optional): Whether to show the plots. Defaults to True.
            image_name (str, optional): The name of the image file to save. Defaults to 'temp.pdf'.

        Returns:
            None
        """
    #def plot_3d_video_detected_spots(self,original_image, filtered_image, spots=None,df_spots=None,colocalized_spots=None,df_colocalized_spots=None,  clusters=None,masks=None, cmap='binary_r',threshold_for_spot_detection=None,voxel_xy_um=None, show_intensity_distribution=False, remove_outliers= True, maximum_percentile=99.7,bins=40,color='orangered',save_plots=False,show_plot=True, image_name='temp.pdf'):
        plt.rcdefaults()
        if (spots is None or len(spots) == 0) and (df_spots is None or df_spots.empty) and (clusters is None or len(clusters) == 0):
            error_message = "Warning: 'spots', 'df_spots', and 'clusters'  all be None or empty."
            print(error_message)  # This will stop the function and show the error
        if show_intensity_distribution ==True:
            number_columns = 5
        else:
            number_columns = 4
        # Initialize spot and cluster variables
        number_detected_spots = 0
        number_clusters = 0
        number_co_detected_spots = 0
        x_spots = y_spots = z_spots = x_clusters = y_clusters = z_clusters = []
        if spots is not None and len(spots) > 0:
            number_detected_spots = spots.shape[0]  # Number of detected spots
            # Sample coordinates
            y_spots = spots[:, 1]  # 
            x_spots = spots[:, 2]  # 
            z_spots = spots[:, 0]  # 
        if clusters is not None and len(clusters) > 0:
            x_clusters = clusters[:, 1]
            y_clusters = clusters[:, 2]
            z_clusters = clusters[:, 0]
            number_clusters = clusters.shape[0]
        else:
            number_clusters = 0
        
        if df_spots is not None and not df_spots.empty:
            number_detected_spots = len(df_spots)
            y_spots = df_spots.y #spots[:, 1]  # 
            x_spots = df_spots.x #spots[:, 2]  #
            z_spots = df_spots.z #spots[:, 3]  #
        
        if df_colocalized_spots is not None and not df_colocalized_spots.empty:
            number_co_detected_spots = len(df_colocalized_spots)
            y_spots_colocalized = df_colocalized_spots.y
            x_spots_colocalized = df_colocalized_spots.x
            z_spots_colocalized = df_colocalized_spots.z
        
        if colocalized_spots is not None and len(colocalized_spots) > 0:
            number_co_detected_spots = colocalized_spots.shape[0]
            y_spots_colocalized = colocalized_spots[:, 1]
            x_spots_colocalized = colocalized_spots[:, 2]
            z_spots_colocalized = colocalized_spots[:, 0]  
        # Plotting
        fig = plt.figure(figsize=(25, 5))
        # Create the first subplot for 'max_log_filter'
        ax0 = fig.add_subplot(1,number_columns,1)
        temp= np.max(original_image, axis=0)
        #temp= np.max(original_image, axis=0)
        ax0.imshow(temp,vmin = np.percentile(temp,0.0), vmax=np.percentile(temp,99.7) , cmap=cmap)
        ax0.set_title('original', fontsize=16)
        ax0.grid(False)
        if voxel_xy_um is not None:
            scalebar = ScaleBar(dx = voxel_xy_um, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w')
            ax0.add_artist(scalebar) 
        # Create the first subplot for 'max_log_filter'
        ax1 = fig.add_subplot(1,number_columns,2)
        temp= np.max(filtered_image, axis=0)
        ax1.imshow(temp,vmin = np.percentile(temp,0.1), vmax=np.percentile(temp,99.7), cmap=cmap)
        ax1.set_title('max_log_filter', fontsize=16)
        ax1.grid(False)
        # Create the second subplot for 'max_log_filter + spots'
        ax2 = plt.subplot(1,number_columns,3)
        ax2.imshow(temp,vmin = np.percentile(temp,0.1) ,vmax=np.percentile(temp,99.7),cmap=cmap)
        if threshold_for_spot_detection is not None:
            ax2.set_title(f'max_log_filter + spots, threshold: {str(int(threshold_for_spot_detection))}', fontsize=16)
        else:
            ax2.set_title('max_log_filter + spots', fontsize=16)
        ax2.grid(False)
        # Plot clusters as circles on top of the 2D image
        if number_clusters > 0:
            radii = clusters[:, 3]*2
            for x, y, r in zip(y_clusters, x_clusters, radii):
                circle = Circle((y, x), r, color='cyan', fill=False)
                ax2.add_patch(circle)
        if masks is not None:
            NUM_POINTS_MASK_EDGE_LINE = 50
            n_masks =np.max(masks)                 
            for i in range(1, n_masks+1 ):
                # Removing the borders just for plotting
                tested_mask = np.where(masks == i, 1, 0).astype(bool)
                # Remove border for plotting
                temp_mask=Utilities().erode_mask(tested_mask)
                temp_mask[0, :] = 0; temp_mask[-1, :] = 0; temp_mask[:, 0] = 0; temp_mask[:, -1] = 0
                temp_contour_n = find_contours(temp_mask, 0.1, fully_connected='high',positive_orientation='high')
                contours_connected_n = np.vstack((temp_contour_n))
                contour = np.vstack((contours_connected_n[-1,:],contours_connected_n))
                if contour.shape[0] > NUM_POINTS_MASK_EDGE_LINE :
                    contour = signal.resample(contour, num = NUM_POINTS_MASK_EDGE_LINE)
                ax2.fill(contour[:, 1], contour[:, 0], facecolor = 'none', edgecolor = 'w', linewidth=1.5) # mask nucleus
        if number_detected_spots >0:
            # Plot circles on top of the 2D image
            radii = len(y_spots) * [3]
            for x, y, r in zip(y_spots, x_spots, radii):
                circle = Circle((y, x), r, color='red', fill=False)
                ax2.add_patch(circle)
        if number_co_detected_spots >0:
            # Plot circles on top of the 2D image
            radii = len(y_spots_colocalized) * [6]
            for x, y, r in zip(y_spots_colocalized, x_spots_colocalized, radii):
                circle = Circle((y, x), r, color='cyan', fill=False, linewidth=2)
                ax2.add_patch(circle)
        # Create the third subplot for '3D Spots'
        ax3 = fig.add_subplot(1,number_columns,4, projection='3d')
        if number_detected_spots >0:
            ax3.scatter(x_spots, y_spots, z_spots, c='r', marker='o')
        if number_clusters > 0:
            ax3.scatter(x_clusters, y_clusters, z_clusters, c='k', marker='o')
        if number_co_detected_spots >0:
            ax3.scatter(x_spots_colocalized, y_spots_colocalized, z_spots_colocalized, c='b', marker='o')
        ax3.set_xlabel('Y', fontsize=14)
        ax3.set_ylabel('X', fontsize=14)
        ax3.set_zlabel('Z', fontsize=14)
        ax3.set_title(f'spots: {str(int(number_detected_spots))}'+f' co-spots: {str(int(number_co_detected_spots))}'  , fontsize=16)
        ax3.view_init(elev=30, azim=-2)
        ax3.invert_zaxis()  # Invert z-axis
        ax3.set_xlim(0, filtered_image.shape[2])
        ax3.set_ylim(0, filtered_image.shape[1])
        ax3.grid(True)
        if show_intensity_distribution:
            ax4 = fig.add_subplot(1, number_columns, 5)
            pixels_intensity = original_image.ravel()
            if remove_outliers:
                pixels_intensity = np.clip(pixels_intensity, 0, np.percentile(pixels_intensity, maximum_percentile))
            hist = sns.histplot(pixels_intensity.ravel(), bins=bins, ax=ax4, color=color)
            ax4.set_title("Pixel intensity")
            ax4.set_xlabel('Intensity')
            ax4.set_ylabel('Frequency')
            # Check if threshold_for_spot_detection is not None and add a vertical line and legend
            if threshold_for_spot_detection is not None:
                ax4.axvline(x=threshold_for_spot_detection, color='blue', label=f'Threshold: {threshold_for_spot_detection}', linewidth=2)
                ax4.legend()  # This will show the legend on the plot
        plt.tight_layout()
        if save_plots == True:
            plt.savefig(image_name, transparent=False,dpi=300, bbox_inches = 'tight', format='png')
        if show_plot ==True:
            plt.show()
        else:
            plt.close()    
        return None
    
    def plot_colocalized_spots (self,filtered_images,df_coordinates_colocalized_spots=None, df_coordinates_0_only_spots=None, df_coordinates_1_only_spots=None, masks= None, voxel_xy_um=None, figsize = (20, 12),max_percentile=99.9,save_plots=False,show_plot=True, image_name='temp.pdf'):
        """
        This function is intended to plot the colocalized spots detected in two spot channels.
        The function will plot the maximum projections of the filtered images and the colocalized spots detected in both channels.
        filtered_images : List of numpy arrays where each array is a filtered image of a channel.
        """
        plt.rcdefaults()
        detected_spots = True
        if (df_coordinates_colocalized_spots is None) and (df_coordinates_0_only_spots is None) and (df_coordinates_1_only_spots is None):
            print('No spots in sample')
            number_colocalized_spots = 0
            number_spots_0 = 0
            number_spots_1 = 0
            x_spots_0, y_spots_0 = None, None
            x_spots_1, y_spots_1 = None, None
            detected_spots =False

        if (df_coordinates_colocalized_spots is not None) and (detected_spots ==True):
            y_spots = df_coordinates_colocalized_spots.y #spots[:, 1]  # 
            x_spots = df_coordinates_colocalized_spots.x #spots[:, 2]  # 
            #if 'z' in df_coordinates_colocalized_spots.columns:
            #    z_spots = df_coordinates_colocalized_spots.z
            number_colocalized_spots = len(df_coordinates_colocalized_spots)
        else:
            y_spots,x_spots = None, None
            number_colocalized_spots = 0
            #TODO
        print('Number of colocalized spots:', number_colocalized_spots)
        if (df_coordinates_0_only_spots is not None) and (detected_spots ==True):
            y_spots_0 = df_coordinates_0_only_spots.y
            x_spots_0 = df_coordinates_0_only_spots.x
            number_spots_0 = len(df_coordinates_0_only_spots)
            print('Number of spots in channel 0:', number_spots_0)
        if (df_coordinates_1_only_spots is not None) and (detected_spots ==True) :
            y_spots_1 = df_coordinates_1_only_spots.y
            x_spots_1 = df_coordinates_1_only_spots.x
            number_spots_1 = len(df_coordinates_1_only_spots)
            print('Number of spots in channel 1:', number_spots_1)  
        # plot image with detected spots
        min_percentile = 0.1
        circles_radius = 5
        # maximum projections of the filtered images
        filtered_images_remove_extrema_first = RemoveExtrema(np.max(filtered_images[0][:,:,:],axis=0),min_percentile=min_percentile, max_percentile=max_percentile).remove_outliers() 
        filtered_images_remove_extrema_second = RemoveExtrema(np.max(filtered_images[1][:,:,:],axis=0),min_percentile=min_percentile, max_percentile=max_percentile).remove_outliers() 
        # merging both channels
        merged_filtered_images = np.stack( ( filtered_images_remove_extrema_first, filtered_images_remove_extrema_second), axis=-1  )  #  (filtered_images[0][0,:,:], filtered_images[1][0,:,:]), axis=-1)
        merged_filtered_images_remove_extrema= Utilities().convert_to_int8(merged_filtered_images, rescale=True, min_percentile=min_percentile, max_percentile=max_percentile)  
        selected_cmap ='Greys_r'
        
        # Plot filtered images with spots
        _, axes = plt.subplots(nrows = 1, ncols = 3, figsize = figsize)
        axes[0].imshow( filtered_images_remove_extrema_first,cmap=selected_cmap)
        axes[0].grid(False)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title('Spots Channel 0 only' )
        # Plot circles on top of the 2D image
        if detected_spots ==True:
            if df_coordinates_0_only_spots is None:
                radii = len(y_spots) * [circles_radius]
                for x, y, r in zip(y_spots, x_spots, radii):
                    circle = Circle((y, x), r, color='r', fill=False,linewidth=1)
                    axes[0].add_patch(circle)
                axes[0].set_title('Colocalized Spots in Channel 0' )
            else:
                radii = len(y_spots_0) * [circles_radius]
                for x, y, r in zip(y_spots_0, x_spots_0, radii):
                    circle = Circle((y, x), r, color='r', fill=False, linewidth=1.5)
                    axes[0].add_patch(circle)
        axes[1].imshow( filtered_images_remove_extrema_second,cmap=selected_cmap)
        axes[1].grid(False)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_title('Spots Channel 1 only' )
        if detected_spots ==True:
            # Plot circles on top of the 2D image
            if df_coordinates_0_only_spots is None:
                radii = len(y_spots) * [circles_radius]
                for x, y, r in zip(y_spots, x_spots, radii):
                    circle = Circle((y, x), r, color='r', fill=False, linewidth=1.5)
                    axes[1].add_patch(circle)
                axes[1].set_title('Colocalized Spots in Channel 1' )
            else:
                radii = len(y_spots_1) * [circles_radius]
                for x, y, r in zip(y_spots_1, x_spots_1, radii):
                    circle = Circle((y, x), r, color='r', fill=False, linewidth=1.5)
                    axes[1].add_patch(circle)
        axes[2].imshow( merged_filtered_images_remove_extrema,cmap=selected_cmap)
        axes[2].grid(False)
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        # Plot circles on top of the 2D image
        if detected_spots ==True:
            radii = len(y_spots) * [circles_radius]
            for x, y, r in zip(y_spots, x_spots, radii):
                circle = Circle((y, x), r, color='w', fill=False, linewidth=1.5)
                axes[2].add_patch(circle)
        axes[2].set_title('Colocalized Spots' )
        if masks is not None:
            NUM_POINTS_MASK_EDGE_LINE = 50
            n_masks =np.max(masks)                 
            for i in range(1, n_masks+1 ):
                # Removing the borders just for plotting
                tested_mask = np.where(masks == i, 1, 0).astype(bool)
                # Remove border for plotting
                temp_mask=Utilities().erode_mask(tested_mask)
                temp_mask[0, :] = 0; temp_mask[-1, :] = 0; temp_mask[:, 0] = 0; temp_mask[:, -1] = 0
                temp_contour_n = find_contours(temp_mask, 0.1, fully_connected='high',positive_orientation='high')
                contours_connected_n = np.vstack((temp_contour_n))
                contour = np.vstack((contours_connected_n[-1,:],contours_connected_n))
                if contour.shape[0] > NUM_POINTS_MASK_EDGE_LINE :
                    contour = signal.resample(contour, num = NUM_POINTS_MASK_EDGE_LINE)
                axes[2].fill(contour[:, 1], contour[:, 0], facecolor = 'none', edgecolor = 'w', linewidth=2) # mask nucleus
        if voxel_xy_um is not None:
            scalebar = ScaleBar(dx = voxel_xy_um, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w')
            axes[2].add_artist(scalebar) 
        if save_plots == True:
            plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        if show_plot ==True:
            plt.show()
        else:
            plt.close()    
        return None




    def plot_image_pixel_intensity_distribution(self, image, figsize=(8, 4),bins=50,remove_outliers=True,remove_zeros=False, single_color=None,maximum_percentile=99.95,save_plots=False,show_plot=True,plot_name='temp.pdf',list_colors=['r','g','b','m','c','k'],tracking_channel = None,threshold_tracking=None):
        '''
        Plot the pixel intensity distribution of an image. The accepted formats are YX, YXC, ZYX, ZYXC and TZYXC.
        '''
        plt.rcdefaults()
        if len(image.shape) >2 :
            number_color_channels = image.shape[-1]      
        else:
            number_color_channels = 1
            image = np.expand_dims(image, axis=-1)
        # Create subplots if more than one color channel
        if number_color_channels > 1:
            _ , ax = plt.subplots(1, number_color_channels, figsize=figsize)
            if number_color_channels == 1:
                ax = [ax]  # Make ax iterable if only one subplot
        else:
            _ , ax = plt.subplots(figsize=figsize)
            ax = [ax]  # Make ax iterable for consistency     
        list_median_intensity = []     
        for c in range(number_color_channels):
            pixels_intensity = image[..., c].ravel()
            # remove nan values
            pixels_intensity = pixels_intensity[~np.isnan(pixels_intensity)]
            # remove zeros
            if remove_zeros == True:    
                pixels_intensity = pixels_intensity[pixels_intensity != 0]
            # remove extrema the 99th percentile using numpy function
            if remove_outliers ==True:
                pixels_intensity = np.clip(pixels_intensity, 0, np.percentile(pixels_intensity, maximum_percentile))
            list_median_intensity.append(np.median(pixels_intensity))
            if single_color is not None: 
                sns.histplot(pixels_intensity.ravel(), bins=bins, ax=ax[c], color=single_color)
            else:
                sns.histplot(pixels_intensity.ravel(), bins=bins, ax=ax[c], color=list_colors[c])
            ax[c].set_facecolor('#f0f0f0')  # Light grey background
            ax[c].set_title(f"Channel {c}", fontsize=8)
            if remove_zeros ==True:
                ax[c].set_xlabel(r'Intensity > 0', fontsize=8)
            else:
                ax[c].set_xlabel('Intensity',  fontsize=8)
            ax[c].set_ylabel('Frequency', fontsize=8)
            if tracking_channel is not None and c == tracking_channel and threshold_tracking is not None:
                ax[c].axvline(x=threshold_tracking, color='blue', label=f'Threshold: {threshold_tracking}', linewidth=1)
                ax[c].text(0.95, 0.95, f"Tracking ts: {threshold_tracking:.2f}", fontsize=10, ha='right', va='top', transform=ax[c].transAxes)
                # calculate the percentil of the pixel intensity that is above the threshold
                percentil_above_threshold = 1 - (np.sum(pixels_intensity > threshold_tracking)/len(pixels_intensity))
                ax[c].text(0.95, 0.85, f"Percentile ts: {percentil_above_threshold:.2f}", fontsize=10, ha='right', va='top', transform=ax[c].transAxes)
        plt.tight_layout()
        if save_plots == True:
            plt.savefig(plot_name, transparent=False,dpi=96, bbox_inches = 'tight')
        if show_plot ==True:
            plt.show()
        else:
            plt.close()
        return list_median_intensity
    

    def plotting_segmentation_images(self,directory,list_files_names,list_segmentation_successful=[None],image_name='temp.pdf',show_plot=True):
        plt.rcdefaults()
        number_images = len(list_files_names)
        NUM_COLUMNS = 1
        NUM_ROWS = number_images
        # Plotting
        _, axes = plt.subplots(nrows = NUM_ROWS, ncols = NUM_COLUMNS, figsize = (15, NUM_ROWS*3))
        # Prealocating plots
        for i in range (0, NUM_ROWS):
            if NUM_ROWS == 1:
                axis_index = axes
            else:
                axis_index = axes[i]
            axis_index.grid(False)
            axis_index.set_xticks([])
            axis_index.set_yticks([])
        # Plotting each image
        for i in range(0, number_images):
            if NUM_ROWS == 1:
                axis_index = axes
            else:
                axis_index = axes[i]
            if (list_segmentation_successful[i] == True): # or (list_segmentation_successful[i] is None):
                temp_segmented_img_name = directory.joinpath('seg_' + list_files_names[i].split(".")[0] +'.png' )
                temp_img =  imread(str( temp_segmented_img_name ))
                axis_index.imshow( temp_img)
            img_title= list_files_names[i]
            axis_index.grid(False)
            axis_index.set_xticks([])
            axis_index.set_yticks([])
            axis_index.set_title(img_title[:-4], fontsize=6 )
        plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        if show_plot ==True:
            plt.show()
        else:
            plt.close()
        plt.show()
        return None
        
        
    def plotting_all_original_images(self,list_images,list_files_names,image_name,show_plot=True):
        plt.rcdefaults()
        number_images = len(list_images)
        NUM_COLUMNS = 5
        NUM_ROWS = math.ceil(number_images/ NUM_COLUMNS)
        # Plotting
        _, axes = plt.subplots(nrows = NUM_ROWS, ncols = NUM_COLUMNS, figsize = (15, NUM_ROWS*3))
        # Prealocating plots
        for i in range (0, NUM_ROWS):
            for j in range(0,NUM_COLUMNS):
                if NUM_ROWS == 1:
                    axis_index = axes[j]
                else:
                    axis_index = axes[i,j]
                axis_index.grid(False)
                axis_index.set_xticks([])
                axis_index.set_yticks([])
        # Plotting each image
        r = 0
        c = 0
        counter = 0
        for i in range(0, number_images):
            if NUM_ROWS == 1:
                axis_index = axes[c]
            else:
                axis_index = axes[r,c]
            temp_img =  list_images[i] #imread(str( local_data_dir.joinpath(list_files_names[i]) ))
            max_image = np.max (temp_img,axis =0)
            max_nun_channels = np.min([3, max_image.shape[2]])
            img_title= list_files_names[i]
            image_int8 = Utilities().convert_to_int8(max_image[ :, :, 0:max_nun_channels], rescale=True, min_percentile=1, max_percentile=95)  
            axis_index.imshow( image_int8)
            axis_index.grid(False)
            axis_index.set_xticks([])
            axis_index.set_yticks([])
            axis_index.set_title(img_title[:-4], fontsize=6 )
            c+=1
            if (c>0) and (c%NUM_COLUMNS ==0):
                c=0
                r+=1
            counter +=1
        plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        if show_plot ==True:
            plt.show()
        else:
            plt.close()
        plt.show()
        return None


    def plot_images(self, image, df=None, masks=None, figsize=(8.5, 5), selected_time=None, 
                suptitle=None, show_trajectories=False, save_plots=False, plot_name='temp', 
                show_plot=True, use_maximum_projection=False, use_gaussian_filter=False, gaussian_sigma=1,microns_per_pixel=None,
                cmap='Spectral', min_max_percentile=[0.5, 99.5], show_gird=False, vmax=None):
        '''
        This method is intended to plot all the channels from an image.
        
        It accepts images in either of these formats:
        - [Z, Y, X, C]  (4D)
        - [T, Z, X, Y, C]  (5D, with a time dimension)
        
        If the image has a time dimension (5D), the `selected_time` parameter is used to select a
        specific time slice. The selected time slice is then converted from [Z, X, Y, C] to 
        [Z, Y, X, C] (swapping the X and Y axes) so that the rest of the plotting code works as before.
        '''
        # Determine if the image is 4D or 5D
        if image.ndim == 4:
            # Format: [Z, Y, X, C]
            image_ZYXC = image
        elif image.ndim == 5:
            # Format: [T, Z, X, Y, C]
            if selected_time is None:
                selected_time = 0
            # Extract the time slice; result has shape [Z, X, Y, C]
            image_slice = image[selected_time, ...]
            # Reorder axes from [Z, X, Y, C] to [Z, Y, X, C]
            image_ZYXC = np.transpose(image_slice, (0, 2, 1, 3))
        else:
            raise ValueError("Image must be either 4D ([Z, Y, X, C]) or 5D ([T, Z, X, Y, C]).")
        
        number_channels = image_ZYXC.shape[3]
        number_z_slices = image_ZYXC.shape[0]
        if number_z_slices == 1:
            center_slice = 0
        else:
            center_slice = image_ZYXC.shape[0] // 2
        
        # Create subplots: one subplot per channel
        _, axes = plt.subplots(nrows=1, ncols=number_channels, figsize=figsize)
        if number_channels == 1:
            axes = [axes]
        
        for i in range(number_channels):
            if number_z_slices > 1:
                if use_maximum_projection:
                    # Compute maximum projection across all Z-slices for the i-th channel
                    temp_max = np.max(image_ZYXC[:, :, :, i], axis=0)
                    rescaled_image = RemoveExtrema(temp_max, min_percentile=min_max_percentile[0],
                                                    max_percentile=min_max_percentile[1]).remove_outliers()
                else:
                    rescaled_image = RemoveExtrema(image_ZYXC[center_slice, :, :, i],
                                                min_percentile=min_max_percentile[0],
                                                max_percentile=min_max_percentile[1]).remove_outliers()
            else:
                rescaled_image = RemoveExtrema(image_ZYXC[center_slice, :, :, i],
                                            min_percentile=min_max_percentile[0],
                                            max_percentile=min_max_percentile[1]).remove_outliers()
            if use_gaussian_filter:
                rescaled_image = gaussian_filter(rescaled_image, sigma=gaussian_sigma)
            axis_index = axes[i]
            if vmax is not None:
                axis_index.imshow(rescaled_image, cmap=cmap, vmax=vmax)
            else:
                axis_index.imshow(rescaled_image, cmap=cmap)
            #axis_index.set_title('Channel_' + str(i), fontsize=8)
            if show_gird:
                axis_index.grid(color='k', ls='-.', lw=0.5)
            else:
                axis_index.grid(False)
            if show_trajectories and df is not None:
                for particle, group in df.groupby('particle'):
                    axis_index.plot(group['x'], group['y'], linewidth=0.5, color='r')
            if df is not None:
                if selected_time is not None:
                    df_time = df[df['frame'] == selected_time]
                else:
                    df_time = df
                for particle, group in df_time.groupby('particle'):
                    axis_index.scatter(group['x'] - 1, group['y'] - 1, s=20, marker='o',
                                    linewidth=0.5, edgecolors='r', facecolors='none')
            if masks is not None and np.any(masks):
                border_width = 3  # Padding to avoid the mask outline appearing at the image edge.
                masks[-border_width:, :] = 0
                masks[:, :border_width] = 0
                masks[:, -border_width:] = 0
                list_contours = Utilities().masks_to_contours(masks)
                for contour in list_contours:
                    axis_index.plot(contour[:, 1], contour[:, 0], color='r', linewidth=1, linestyle='-')
        
        # Add super title if provided
        if suptitle is not None:
            if '[LOW QUALITY IMAGE]' in suptitle:
                plt.suptitle(suptitle, fontsize=10, verticalalignment='top', color='red')
            else:
                plt.suptitle(suptitle, fontsize=10, verticalalignment='top')
            plt.subplots_adjust(top=0.85)

        font_props = {'size': 10, }
        if microns_per_pixel is not None:
            scalebar = ScaleBar(dx=microns_per_pixel, units='um', length_fraction=0.25,
                                location='lower right', box_color='k', color='w',
                                font_properties=font_props, label='')
            axis_index.add_artist(scalebar)

        # remove axis labels and ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

        
        if save_plots:
            plt.savefig(plot_name, bbox_inches='tight', dpi=96)
        if show_plot:
            plt.show()
        else:
            plt.close()
        return None

    
    def plotting_masks_and_original_image(self,image, masks_complete_cells=None, masks_nuclei=None, channels_cytosol=None, channels_nucleus=None,image_name='temp',figsize = (15, 10),show_plot=True,df_labels=None,text_color='k',fontsize=12):
        # This functions makes zeros the border of the mask, it is used only for plotting.
        plt.rcdefaults()
        NUM_POINTS_MASK_EDGE_LINE = 50
        def erode_mask(img,px_to_remove = 1):
            img[0:px_to_remove, :] = 0;img[:, 0:px_to_remove] = 0;img[img.shape[0]-px_to_remove:img.shape[0]-1, :] = 0; img[:, img.shape[1]-px_to_remove: img.shape[1]-1 ] = 0#This line of code ensures that the corners are zeros.
            return erosion(img) # performin erosion of the mask to remove not connected pixeles.
        # This section converst the image into a 2d maximum projection.
        if len(image.shape) > 3:  # [ZYXC]
            if image.shape[0] ==1:
                max_image = image[0,:,:,:]
            else:
                max_image = np.max(image[:,:,:,:],axis=0)    # taking the mean value
        else:
            max_image = image # [YXC] 
        # give an error if the masks are not provided
        if (masks_nuclei is None) and (masks_complete_cells is None):
            raise ValueError('At least one of the masks should be provided.')
        if (masks_complete_cells is None):
            masks_complete_cells = masks_nuclei
            print('masks_complete_cells is None, using masks_nuclei.')
        # Plotting
        n_channels = np.min([3, max_image.shape[2]])
        im = Utilities().convert_to_int8(max_image[ :, :, 0:n_channels], rescale=True, min_percentile=1, max_percentile=95)  
        if np.max(masks_complete_cells) != 0 and not(channels_cytosol in (None,[None])) and not(channels_nucleus in (None,[None])):
            _, axes = plt.subplots(nrows = 1, ncols = 4, figsize = figsize)
            masks_plot_cyto= masks_complete_cells 
            masks_plot_nuc = masks_nuclei              
            axes[0].imshow(im)
            axes[0].set(title = 'All channels')
            axes[1].imshow(masks_plot_cyto)
            axes[1].set(title = 'Cytosol mask')
            if not (df_labels is None):
                cell_ids_labels = df_labels.loc[ :,'cell_id'].values
                for _, label in enumerate(cell_ids_labels):
                    cell_idx_string = str(label)
                    Y_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_y'].item()
                    X_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_x'].item()
                    axes[1].text(x=X_cell_location, y=Y_cell_location, s=cell_idx_string, fontsize=fontsize, color=text_color, fontweight='bold')
            axes[2].imshow(masks_plot_nuc)
            axes[2].set(title = 'Nuclei mask')
            if not (df_labels is None):
                cell_ids_labels = df_labels.loc[ :,'cell_id'].values
                for _, label in enumerate(cell_ids_labels):
                    cell_idx_string = str(label)
                    Y_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_y'].item()
                    X_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_x'].item()
                    axes[2].text(x=X_cell_location, y=Y_cell_location, s=cell_idx_string, fontsize=fontsize, color=text_color, fontweight='bold')
            axes[3].imshow(im)
            n_masks =np.max(masks_complete_cells)                 
            for i in range(1, n_masks+1 ):
                # Removing the borders just for plotting
                tested_mask_cyto = np.where(masks_complete_cells == i, 1, 0).astype(bool)
                tested_mask_nuc = np.where(masks_nuclei == i, 1, 0).astype(bool)
                # Remove border for plotting
                temp_nucleus_mask= erode_mask(tested_mask_nuc)
                temp_complete_mask = erode_mask(tested_mask_cyto)
                temp_nucleus_mask[0, :] = 0; temp_nucleus_mask[-1, :] = 0; temp_nucleus_mask[:, 0] = 0; temp_nucleus_mask[:, -1] = 0
                temp_complete_mask[0, :] = 0; temp_complete_mask[-1, :] = 0; temp_complete_mask[:, 0] = 0; temp_complete_mask[:, -1] = 0
                temp_contour_n = find_contours(temp_nucleus_mask, 0.1, fully_connected='high',positive_orientation='high')
                temp_contour_c = find_contours(temp_complete_mask, 0.1, fully_connected='high',positive_orientation='high')
                contours_connected_n = np.vstack((temp_contour_n))
                contour_n = np.vstack((contours_connected_n[-1,:],contours_connected_n))
                if contour_n.shape[0] > NUM_POINTS_MASK_EDGE_LINE :
                    contour_n = signal.resample(contour_n, num = NUM_POINTS_MASK_EDGE_LINE)
                contours_connected_c = np.vstack((temp_contour_c))
                contour_c = np.vstack((contours_connected_c[-1,:],contours_connected_c))
                if contour_c.shape[0] > NUM_POINTS_MASK_EDGE_LINE :
                    contour_c = signal.resample(contour_c, num = NUM_POINTS_MASK_EDGE_LINE)
                axes[3].fill(contour_n[:, 1], contour_n[:, 0], facecolor = 'none', edgecolor = 'red', linewidth=2) # mask nucleus
                axes[3].fill(contour_c[:, 1], contour_c[:, 0], facecolor = 'none', edgecolor = 'red', linewidth=2) # mask cytosol
                axes[3].set(title = 'Paired masks')
            if not (df_labels is None):
                cell_ids_labels = df_labels.loc[ :,'cell_id'].values
                for _, label in enumerate(cell_ids_labels):
                    cell_idx_string = str(label)
                    Y_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_y'].item()
                    X_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_x'].item()
                    axes[3].text(x=X_cell_location, y=Y_cell_location, s=cell_idx_string, fontsize=fontsize, color=text_color, fontweight='bold')
        else:
            if not(channels_cytosol in (None,[None])) and (channels_nucleus in (None,[None])):
                masks_plot_cyto= masks_complete_cells 
                n_channels = np.min([3, max_image.shape[2]])
                _, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 10))
                axes[0].imshow(im)
                axes[0].set(title = 'All channels')
                axes[1].imshow(masks_plot_cyto)
                axes[1].set(title = 'Cytosol mask')
                if not (df_labels is None):
                    cell_ids_labels = df_labels.loc[ :,'cell_id'].values
                    for _, label in enumerate(cell_ids_labels):
                        cell_idx_string = str(label)
                        Y_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_y'].item()
                        X_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_x'].item()
                        axes[1].text(x=X_cell_location, y=Y_cell_location, s=cell_idx_string, fontsize=fontsize, color=text_color, fontweight='bold')
                axes[2].imshow(im)
                n_masks =np.max(masks_complete_cells)                 
                for i in range(1, n_masks+1 ):
                    # Removing the borders just for plotting
                    tested_mask_cyto = np.where(masks_complete_cells == i, 1, 0).astype(bool)
                    # Remove border for plotting
                    temp_complete_mask = erode_mask(tested_mask_cyto)
                    temp_complete_mask[0, :] = 0; temp_complete_mask[-1, :] = 0; temp_complete_mask[:, 0] = 0; temp_complete_mask[:, -1] = 0
                    temp_contour_c = find_contours(temp_complete_mask, 0.1, fully_connected='high',positive_orientation='high')
                    try:
                        contours_connected_c = np.vstack((temp_contour_c))
                        contour_c = np.vstack((contours_connected_c[-1,:],contours_connected_c))
                        if contour_c.shape[0] > NUM_POINTS_MASK_EDGE_LINE :
                            contour_c = signal.resample(contour_c, num = NUM_POINTS_MASK_EDGE_LINE)
                            axes[2].fill(contour_c[:, 1], contour_c[:, 0], facecolor = 'none', edgecolor = 'red', linewidth=2) # mask cytosol
                    except:
                        contour_c = 0
                    axes[2].set(title = 'Original + Masks')
                if not (df_labels is None):
                    cell_ids_labels = df_labels.loc[ :,'cell_id'].values
                    for _, label in enumerate(cell_ids_labels):
                        cell_idx_string = str(label)
                        Y_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'cyto_loc_y'].item()
                        X_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'cyto_loc_x'].item()
                        axes[2].text(x=X_cell_location, y=Y_cell_location, s=cell_idx_string, fontsize=fontsize, color=text_color, fontweight='bold')
            if (channels_cytosol in (None,[None])) and not(channels_nucleus in (None,[None])):
                masks_plot_nuc = masks_nuclei    
                _, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 10))
                axes[0].imshow(im)
                axes[0].set(title = 'All channels')
                axes[1].imshow(masks_plot_nuc)
                if not (df_labels is None):
                    cell_ids_labels = df_labels.loc[ :,'cell_id'].values
                    for _, label in enumerate(cell_ids_labels):
                        cell_idx_string = str(label)
                        Y_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_y'].item()
                        X_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_x'].item()
                        axes[1].text(x=X_cell_location, y=Y_cell_location, s=cell_idx_string, fontsize=fontsize, color=text_color, fontweight='bold')
                axes[1].set(title = 'Nuclei mask')
                axes[2].imshow(im)
                n_masks =np.max(masks_nuclei)                 
                for i in range(1, n_masks+1 ):
                    # Removing the borders just for plotting
                    tested_mask_nuc = np.where(masks_nuclei == i, 1, 0).astype(bool)
                    # Remove border for plotting
                    temp_nucleus_mask= erode_mask(tested_mask_nuc)
                    temp_nucleus_mask[0, :] = 0; temp_nucleus_mask[-1, :] = 0; temp_nucleus_mask[:, 0] = 0; temp_nucleus_mask[:, -1] = 0
                    temp_contour_n = find_contours(temp_nucleus_mask, 0.1, fully_connected='high',positive_orientation='high')
                    contours_connected_n = np.vstack((temp_contour_n))
                    contour_n = np.vstack((contours_connected_n[-1,:],contours_connected_n))
                    if contour_n.shape[0] > NUM_POINTS_MASK_EDGE_LINE :
                        contour_n = signal.resample(contour_n, num = NUM_POINTS_MASK_EDGE_LINE)
                    axes[2].fill(contour_n[:, 1], contour_n[:, 0], facecolor = 'none', edgecolor = 'red', linewidth=2) # mask nucleus
                    axes[2].set(title = 'Original + Masks')
                if not (df_labels is None):
                    cell_ids_labels = df_labels.loc[ :,'cell_id'].values
                    for _, label in enumerate(cell_ids_labels):
                        cell_idx_string = str(label)
                        Y_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_y'].item()
                        X_cell_location = df_labels.loc[df_labels['cell_id'] == label, 'nuc_loc_x'].item()
                        axes[2].text(x=X_cell_location, y=Y_cell_location, s=cell_idx_string, fontsize=12, color='black')
        if not(image_name is None):
            plt.savefig(image_name,bbox_inches='tight',dpi=180)
        if show_plot == 1:
            plt.show()
        else:
            plt.close()
        return None


    def dist_plots(self, df, plot_title,destination_folder,y_lim_values=None ):
        plt.rcdefaults()
        stacked_df = df.stack()
        pct = stacked_df.quantile(q=0.99)
        #color_palete = 'colorblind'
        color_palete = 'CMRmap'
        #color_palete = 'OrRd'
        sns.set_style("white")
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        max_x_val = df.max().max()
        # Distribution
        plt.figure(figsize=(10,5))
        sns.set(font_scale = 1)
        sns.set_style("white")
        p_dist =sns.kdeplot(data=df,palette=color_palete,cut=0,lw=5)
        p_dist.set_xlabel("Spots")
        p_dist.set_ylabel("Kernel Density Estimator (KDE)")
        p_dist.set_title(plot_title)
        p_dist.set(xlim=(0, pct))
        name_plot = 'Dist_'+plot_title+'.pdf'
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))
        # ECDF
        plt.figure(figsize=(10,5))
        sns.set(font_scale = 1)
        sns.set_style("white")
        p_dist =sns.ecdfplot(data=df,palette=color_palete,lw=5)
        p_dist.set_xlabel("Spots")
        p_dist.set_ylabel("Proportion")
        p_dist.set_title(plot_title)
        p_dist.set_ylim(0,1.05)
        p_dist.set(xlim=(0, pct))
        name_plot = 'ECDF_'+ plot_title+'.pdf'
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))
        # Whisker Plots
        plt.figure(figsize=(7,9))
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        p = sns.stripplot(data=df, size=3, color='0.5', jitter=0.2)
        plt.xticks(rotation=45, ha="right")
        sns.set(font_scale = 1.5)
        bp=sns.boxplot( 
                    meanprops={'visible': True,'color': 'r', 'ls': 'solid', 'lw': 4},
                    whiskerprops={'visible': True, 'color':'k','ls': 'solid', 'lw': 1},
                    data=df,
                    showcaps={'visible': False, 'color':'orangered', 'ls': 'solid', 'lw': 1}, # Q1-Q3 25-75%
                    ax=p,
                    showmeans=True,meanline=True,zorder=10,showfliers=False,showbox=True,linewidth=1,color='w')
        p.set_xlabel("Time After Treatment")
        p.set_ylabel("Spot Count")
        p.set_title(plot_title)
        if not (y_lim_values is None):
            p.set(ylim=y_lim_values)
        sns.set(font_scale = 1.5)
        name_plot = 'Whisker_'+plot_title +'.pdf'  
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))
        # Joy plots
        plt.figure(figsize=(7,5))
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        fig, axes = joypy.joyplot(df,x_range=[-5,pct],bins=25,hist=False, overlap=0.8, linewidth=1, figsize=(7,5), colormap=cm.CMRmap) #
        name_plot = 'JoyPlot_'+ plot_title+'.pdf'
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))
        return None

    def plot_comparing_df(self, df_all,df_cyto,df_nuc,plot_title,destination_folder):
        plt.rcdefaults()
        #color_palete = 'CMRmap'
        color_palete = 'Dark2'
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        # This code creates a single colum for all conditions and adds a 'location' column.
        df_all_melt = df_all.melt()
        df_all_melt['location'] = 'all' 
        df_cyto_melt = df_cyto.melt()
        df_cyto_melt['location']= 'cyto'
        df_nuc_melt = df_nuc.melt()
        df_nuc_melt['location']= 'nuc' 
        data_frames_list = [df_all_melt, df_cyto_melt, df_nuc_melt]
        data_frames = pd.concat(data_frames_list)       
        # Plotting
        plt.figure(figsize=(12,7))
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        b= sns.barplot(data=data_frames, x= 'variable',y='value', hue = 'location',palette=color_palete)
        b.set_xlabel("time after treatment")
        b.set_ylabel("Spot Count")
        b.set_title(plot_title)
        plt.xticks(rotation=45, ha="right") 
        name_plot = plot_title +'.pdf'  
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))
        return None


    def plot_TS(self, df_original,plot_title,destination_folder,minimum_spots_cluster,remove_zeros=False):
        plt.rcdefaults()
        color_palete = 'CMRmap'
        #color_palete = 'Accent'
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        df= df_original.copy()
        if remove_zeros == True:
            for col in df.columns:
                df[col] = np.where(df[col]==0, np.nan, df[col])
        plt.figure(figsize=(12,7))
        b= sns.stripplot(data=df, size=4, jitter=0.3, dodge=True,palette=color_palete)
        b.set_xlabel('time after treatment')
        b.set_ylabel('No. Cells with TS (Int. >= ' +str (minimum_spots_cluster) +' <RNAs>)' )
        b.set_title(plot_title)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
        plt.xticks(rotation=45, ha="right")
        name_plot = plot_title +'.pdf'  
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))
        return None


    def plot_TS_bar_stacked(self, df_original,plot_title,destination_folder,minimum_spots_cluster,remove_zeros=False,normalize=True):
        plt.rcdefaults()
        if (normalize == True) and (remove_zeros == True):
            warnings.warn("Warining: notice that normalization is only possible if zeros are not removed. To normalize the output use the options as follows: remove_zeros=False, normalize=True ")
        df= df_original.copy()
        color_palete = 'OrRd'
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        if remove_zeros == True:
            for col in df.columns:
                df[col] = np.where(df[col]==0, np.nan, df[col])
            min_range = 1
            num_labels =4
            column_labels =['1 TS','2 TS','>2 TS']
            ts_values = list(range(min_range,num_labels )) # 1,2,3
            max_ts_count = 1
        else:
            min_range = 0
            num_labels =4
            column_labels =['0 TS','1 TS','2 TS','>2 TS']
            max_ts_count = 2
            ts_values = list(range(min_range, num_labels )) # 0,1,2,3
        num_columns = len(list(df.columns))
        table_data = np.zeros((len(ts_values),num_columns)) 
        
        for i, col in enumerate(df.columns):
            for indx, ts_size in enumerate (ts_values):
                if indx<=max_ts_count:
                    table_data[indx,i] = df.loc[df[col] == ts_size, col].count()
                else:
                    table_data[indx,i] = df.loc[df[col] >= ts_size, col].count()
        if (normalize == True) and (remove_zeros == False):
            number_cells = np.sum(table_data,axis =0)
            normalized_table = table_data/number_cells
            df_new = pd.DataFrame(normalized_table.T, columns = column_labels,index=list(df.columns))
            ylabel_text = ' TS (Int. >= ' +str (minimum_spots_cluster) +' <RNAs>) / Cell' 
        else:
            df_new = pd.DataFrame(table_data.T, columns = column_labels,index=list(df.columns))
            ylabel_text = 'No. Cells with TS (Int. >= ' +str (minimum_spots_cluster) +' <RNAs>)' 
        # Plotting
        b= df_new.plot(kind='bar', stacked=True,figsize=(12,7)) #, cmap=color_palete
        b.legend(fontsize=12)
        b.set_xlabel('time after treatment')
        b.set_ylabel(ylabel_text )
        b.set_title(plot_title)
        if normalize == False:
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
        plt.xticks(rotation=45, ha="right")
        name_plot = plot_title +'.pdf'  
        plt.savefig(name_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,name_plot))
        return None


    def plotting_results_as_distributions(self, number_of_spots_per_cell,number_of_spots_per_cell_cytosol,number_of_spots_per_cell_nucleus,ts_size,number_of_TS_per_cell,minimum_spots_cluster,numBins=20,output_identification_string=None, spot_type=0):
        plt.rcdefaults()
        # Creating a name for the plot
        if not (output_identification_string is None):
            index_string = output_identification_string.index('__')
            if index_string >=0:
                title_string = output_identification_string[0:index_string]
            else :
                title_string = ''
        else:
            title_string = ''
        # Plotting intensity distributions
        def plot_probability_distribution(data_to_plot, numBins = 10, title='', xlab='', ylab='', color='r', subplots=False, show_grid=True, fig=plt.figure() ):
            n, bins, _ = plt.hist(data_to_plot,bins=numBins,density=False,color=color)
            plt.xlabel(xlab, size=16)
            plt.ylabel(ylab, size=16)
            plt.grid(show_grid)
            plt.text(bins[(len(bins)//2)],(np.amax(n)//2).astype(int),'mean = '+str(round( np.mean(data_to_plot) ,1) ), fontsize=14,bbox=dict(facecolor='w', alpha=0.5) )
            plt.title(title, size=16)
            return (f)
        # Section that generates each subplot
        number_subplots = int(np.any(number_of_spots_per_cell)) + int(np.any(number_of_spots_per_cell_cytosol)) + int(np.any(number_of_spots_per_cell_nucleus)) + int(np.any(ts_size)) + int(np.any(number_of_TS_per_cell))
        file_name = 'spot_distributions_'+title_string+'_spot_type_'+str(spot_type)+'.pdf'
        #Plotting
        fig_size = (25, 5)
        f = plt.figure(figsize=fig_size)
        #ylab='Probability'
        ylab='Frequency Count' 
        selected_color = '#1C00FE' 
        # adding subplots
        subplot_counter = 0
        if np.any(number_of_spots_per_cell):
            subplot_counter+=1
            f.add_subplot(1,number_subplots,subplot_counter) 
            plot_probability_distribution( number_of_spots_per_cell, numBins=20,  title='Total Num Spots per cell', xlab='Number', ylab=ylab, fig=f, color=selected_color)
        if np.any(number_of_spots_per_cell_cytosol):
            subplot_counter+=1
            f.add_subplot(1,number_subplots,subplot_counter)  
            plot_probability_distribution(number_of_spots_per_cell_cytosol,   numBins=20,  title='Num Spots in Cytosol', xlab='Number', ylab=ylab, fig=f, color=selected_color)
        if np.any(number_of_spots_per_cell_nucleus):
            subplot_counter+=1
            f.add_subplot(1,number_subplots,subplot_counter)  
            plot_probability_distribution(number_of_spots_per_cell_nucleus, numBins=20,    title='Num Spots in Nucleus', xlab='Number', ylab=ylab, fig=f, color=selected_color)
        if np.any(ts_size):
            subplot_counter+=1
            f.add_subplot(1,number_subplots,subplot_counter)  
            plot_probability_distribution(ts_size, numBins=20,    title='Clusters in nucleus', xlab='RNA per Cluster', ylab=ylab, fig=f, color=selected_color)
        if np.any(number_of_TS_per_cell):
            subplot_counter+=1
            f.add_subplot(1,number_subplots,subplot_counter)  
            plot_probability_distribution(number_of_TS_per_cell ,  numBins=20, title='Number TS per cell', xlab='[TS (>= '+str(minimum_spots_cluster)+' rna)]', ylab=ylab, fig=f, color=selected_color)
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf' )
        plt.show()
        return file_name



    def plot_scatter_and_distributions(self, x,y, plot_title,  x_label_scatter='cell_size', y_lable_scatter = 'number_of_spots_per_cell', destination_folder=None, selected_color = '#1C00FE',save_plot=False,temporal_figure=False):
        plt.rcdefaults()
        r, p = pearsonr(x, y)
        df_join_distribution = pd.DataFrame({x_label_scatter:x,y_lable_scatter:y})
        sns.set(font_scale = 1.3)
        b = sns.jointplot(data=df_join_distribution, y=y_lable_scatter, x=x_label_scatter, color= selected_color , marginal_kws=dict(bins=40, rug=True))
        b.plot_joint(sns.rugplot, height=0, color=[0.7,0.7,0.7], clip_on=True)
        b.plot_joint(sns.kdeplot, color=[0.5,0.5,0.5], levels=5)
        b.plot_joint(sns.regplot,scatter_kws={'color': 'orangered',"s":10, 'marker':'o'}, line_kws={'color': selected_color,'lw': 2} )
        blank_plot, = b.ax_joint.plot([], [], linestyle="", alpha=0)
        b.ax_joint.legend([blank_plot],['r={:.2f}'.format( np.round(r,2))],loc='upper left',)
        b.ax_joint.set_xlim(np.percentile(x,0.01), np.percentile(x,99.9))
        b.ax_joint.set_ylim(np.percentile(y,0.01), np.percentile(y,99.9))
        b.fig.suptitle(plot_title)
        b.ax_joint.collections[0].set_alpha(0)
        b.fig.tight_layout()
        b.fig.subplots_adjust(top=0.92) 
        name_plot = plot_title 
        if temporal_figure == True:
            file_name = 'temp__'+str(np.random.randint(1000, size=1)[0])+'__'+name_plot+'.png' # generating a random name for the temporal plot
            plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='png')
            plt.close(b.fig)
        else:
            file_name = name_plot+'.pdf'
        if (save_plot == True) and (temporal_figure == False):
            plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
            plt.show()
        if not (destination_folder is None) and (save_plot == True) and (temporal_figure==False):
            pathlib.Path().absolute().joinpath(name_plot).rename(pathlib.Path().absolute().joinpath(destination_folder,file_name))
        return b.fig, file_name

    def plot_cell_size_spots(self, channels_cytosol, channels_nucleus, cell_size, number_of_spots_per_cell, cyto_size, number_of_spots_per_cell_cytosol, nuc_size, number_of_spots_per_cell_nucleus,output_identification_string=None,spot_type=0):  
        '''
        This function is intended to plot the spot count as a function of the cell size. 
        
        '''
        # Creating a name for the plot
        plt.rcdefaults()
        if not (output_identification_string is None):
            index_string = output_identification_string.index('__')
            if index_string >=0:
                title_string = output_identification_string[0:index_string]
            else :
                title_string = ''
        else:
            title_string = ''
        
        if not channels_cytosol in (None, 'None', 'none',['None'],['none'],[None]):
            cyto_exists = True
        else:
            cyto_exists = False
        if not channels_nucleus in (None, 'None', 'none',['None'],['none'],[None]):
            nuc_exists = True
        else:
            nuc_exists = False
        # Plot title
        title_plot='cell'
        file_name = 'scatter_cell_size_vs_spots_'+title_string+'_spot_type_'+str(spot_type)+'.pdf'
        # Complete cell
        if (cyto_exists == True) and (nuc_exists == True):
            x = cell_size
            y = number_of_spots_per_cell
            _,fig1_temp_name = Plots().plot_scatter_and_distributions(x,y,title_plot,x_label_scatter='cell_size', y_lable_scatter = 'number_of_spots_per_cell',temporal_figure=True)
        # Cytosol
        if cyto_exists == True:
            x = cyto_size
            y = number_of_spots_per_cell_cytosol
            title_plot='cytosol'
            _,fig2_temp_name = Plots().plot_scatter_and_distributions(x,y,title_plot ,x_label_scatter='cyto_size', y_lable_scatter = 'number_of_spots_per_cyto',temporal_figure=True)
        # Nucleus
        if nuc_exists == True:
            x = nuc_size
            y = number_of_spots_per_cell_nucleus
            title_plot='nucleus'
            _,fig3_temp_name = Plots().plot_scatter_and_distributions(x,y,title_plot ,x_label_scatter='nuc_size', y_lable_scatter = 'number_of_spots_per_nuc',temporal_figure=True)
        # Plotting
        _, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 10))
        counter = 0
        if cyto_exists == True:
            axes[counter].imshow(plt.imread(fig2_temp_name))
            os.remove(fig2_temp_name)
            counter +=1
        if nuc_exists == True:
            axes[counter].imshow(plt.imread(fig3_temp_name))
            os.remove(fig3_temp_name)
        if (cyto_exists == True) and (nuc_exists == True):
            axes[2].imshow(plt.imread(fig1_temp_name))
            os.remove(fig1_temp_name)
        # removing axis
        axes[0].grid(False)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].grid(False)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[2].grid(False)
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        return file_name

    def plot_cell_intensity_spots(self, dataframe, number_of_spots_per_cell_nucleus = None, number_of_spots_per_cell_cytosol = None,output_identification_string=None,spot_type=0):  
        plt.rcdefaults()
        # Creating a name for the plot
        if not (output_identification_string is None):
            index_string = output_identification_string.index('__')
            if index_string >=0:
                title_string = output_identification_string[0:index_string]
            else :
                title_string = ''
        else:
            title_string = ''
        # Counting the number of color channels in the dataframe
        pattern = r'^spot_int_ch_\d'
        string_list = dataframe.columns
        number_color_channels = 0
        for string in string_list:
            match = re.match(pattern, string)
            if match:
                number_color_channels += 1
        # Detecting if the nucleus and cytosol are detected
        if np.any(number_of_spots_per_cell_cytosol):
            cyto_exists = True
        else:
            cyto_exists = False
        if np.any(number_of_spots_per_cell_nucleus):
            nucleus_exists = True
        else:
            nucleus_exists = False
        if (nucleus_exists==True) and (cyto_exists==True):
            number_rows = 2
        else:
            number_rows = 1
        # Creating plot
        file_name  = 'ch_int_vs_spots_'+title_string+'_spot_type_'+str(spot_type)+'.pdf'
        counter = 0
        _, axes = plt.subplots(nrows = number_rows, ncols = number_color_channels, figsize = (15, 10))
        for j in range(number_rows):
            for i in range(number_color_channels):
                if number_rows==1 and (number_color_channels==1):
                    axis_index = axes
                elif number_rows==1 and (number_color_channels>=1):
                    axis_index = axes[j]
                elif number_rows==2 and (number_color_channels>1):
                    axis_index = axes[j,i]                
                if (nucleus_exists==True) and (counter ==0):
                    column_intensity = 'nuc_int_ch_'+str(i)
                    title_plot='nucleus'
                    x = Utilities().function_get_df_columns_as_array(df=dataframe, colum_to_extract=column_intensity, extraction_type='values_per_cell') 
                    y = number_of_spots_per_cell_nucleus
                if ((cyto_exists==True) and (counter ==1)) or ((cyto_exists==True) and (counter ==0) and (number_rows==1)):
                    column_intensity = 'cyto_int_ch_'+str(i)
                    title_plot='cytosol'
                    x = Utilities().function_get_df_columns_as_array(df=dataframe, colum_to_extract=column_intensity, extraction_type='values_per_cell') 
                    y = number_of_spots_per_cell_cytosol
                _,fig_temp_name = Plots().plot_scatter_and_distributions(x,y,title_plot,x_label_scatter='Intensity_Ch_'+str(i), y_lable_scatter = 'number_of_spots',temporal_figure=True)
                if number_rows ==1:
                    axis_index.imshow(plt.imread(fig_temp_name))
                    axis_index.grid(False)
                    axis_index.set_xticks([])
                    axis_index.set_yticks([])
                else:
                    axes[j,i].imshow(plt.imread(fig_temp_name))
                    axes[j,i].grid(False)
                    axes[j,i].set_xticks([])
                    axes[j,i].set_yticks([])
                os.remove(fig_temp_name)
                del x, y
            counter +=1
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        return file_name

    def plot_scatter_bleed_thru (self, dataframe,channels_cytosol, channels_nucleus,output_identification_string=None):
        plt.rcdefaults()
        # Creating a name for the plot
        if not (output_identification_string is None):
            index_string = output_identification_string.index('__')
            if index_string >=0:
                title_string = output_identification_string[0:index_string]
            else :
                title_string = ''
        else:
            title_string = ''
        # Counting the number of color channels in the dataframe
        pattern = r'^spot_int_ch_\d'
        string_list = dataframe.columns
        number_color_channels = 0
        for string in string_list:
            match = re.match(pattern, string)
            if match:
                number_color_channels += 1
        # Calculating the number of combination of color channels
        combinations_channels = list(itertools.combinations(range(number_color_channels), 2))
        _, axes = plt.subplots(nrows = 1, ncols = len(combinations_channels), figsize = (20, 10))
        for i in range(len(combinations_channels)):
            if len(combinations_channels) == 1:
                axis_index = axes
            else:
                axis_index = axes[i]
            title_plot=title_string
            file_name  = 'bleed_thru_'+title_string+'.pdf'
            if not channels_cytosol in (None, 'None', 'none',['None'],['none'],[None]):
                x = Utilities().function_get_df_columns_as_array(df=dataframe, colum_to_extract='cyto_int_ch_'+str(combinations_channels[i][0]), extraction_type='values_per_cell') 
                y = Utilities().function_get_df_columns_as_array(df=dataframe, colum_to_extract='cyto_int_ch_'+str(combinations_channels[i][1]), extraction_type='values_per_cell') 
            if not channels_nucleus in (None, 'None', 'none',['None'],['none'],[None]):
                x = Utilities().function_get_df_columns_as_array(df=dataframe, colum_to_extract='nuc_int_ch_'+str(combinations_channels[i][0]), extraction_type='values_per_cell') 
                y = Utilities().function_get_df_columns_as_array(df=dataframe, colum_to_extract='nuc_int_ch_'+str(combinations_channels[i][1]), extraction_type='values_per_cell') 
            _,fig_temp_name = Plots().plot_scatter_and_distributions(x,y,title_plot,x_label_scatter='intensity_Ch_'+str(combinations_channels[i][0]), y_lable_scatter = 'intensity_Ch_'+str(combinations_channels[i][1]),temporal_figure=True)
            axis_index.imshow(plt.imread(fig_temp_name))
            axis_index.grid(False)
            axis_index.set_xticks([])
            axis_index.set_yticks([])
            del x, y 
            os.remove(fig_temp_name)
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        return file_name
    
    def plot_interpretation_distributions (self, df_all, df_cyto, df_nuc, destination_folder, plot_title_suffix='',y_lim_values_all_spots=None, y_lim_values_cyto=None,y_lim_values_nuc=None):
        plt.rcdefaults()
        if (df_cyto.dropna().any().any() == True) and (df_nuc.dropna().any().any() == True):  # removing nans from df and then testing if any element is non zero. If this is true, the plot is generated
            plot_title_complete = 'all_spots__'+plot_title_suffix
            Plots().dist_plots(df_all, plot_title_complete, destination_folder,y_lim_values_all_spots)
        
        if df_cyto.dropna().any().any() == True:  # removing nans from df and then testing if any element is non zero. If this is true, the plot is generated
            # Plotting for all Cytosol only
            plot_title_cyto = 'cyto__'+plot_title_suffix
            Plots().dist_plots(df_cyto, plot_title_cyto, destination_folder,y_lim_values_cyto)
        
        if df_nuc.dropna().any().any() == True:  # removing nans from df and then testing if any element is non zero. If this is true, the plot is generated
            # Plotting for all nucleus
            plot_title_nuc = 'nuc__'+plot_title_suffix
            Plots().dist_plots(df_nuc, plot_title_nuc, destination_folder,y_lim_values_nuc)
        return None


    def plot_spot_intensity_distributions(self, dataframe,output_identification_string=None,remove_outliers=True, spot_type=0):
        plt.rcdefaults()
        # Creating a name for the plot
        if not (output_identification_string is None):
            index_string = output_identification_string.index('__')
            if index_string >=0:
                title_string = output_identification_string[0:index_string]
            else :
                title_string = ''
        else:
            title_string=''
        # Counting the number of color channels in the dataframe
        pattern = r'^spot_int_ch_\d'
        string_list = dataframe.columns
        number_color_channels = 0
        for string in string_list:
            match = re.match(pattern, string)
            if match:
                number_color_channels += 1
        # Plotting
        _, axes = plt.subplots(nrows = 1, ncols = number_color_channels, figsize = (25, 5))
        max_percentile = 99
        min_percentile = 0.5
        title_plot  = 'spot_intensities'
        file_name = title_plot +'_'+title_string+'_spot_type_'+str(spot_type)+'.pdf'
        colors = ['r','g','b','m']
        for i in range (0,number_color_channels ):
            if number_color_channels ==1:
                axis_index = axes
            else:
                axis_index = axes[i]
            column_name = 'spot_int_ch_'+str(i)
            df_spot_intensity = dataframe.loc[   (dataframe['is_cluster']==False) & (dataframe['spot_type']==spot_type)]
            spot_intensity = df_spot_intensity[column_name].values
            if remove_outliers ==True:
                spot_intensity =Utilities().remove_outliers( spot_intensity,min_percentile=1,max_percentile=98)
            axis_index.hist(x=spot_intensity, bins=30, density = True, histtype ='bar',color = colors[i],label = 'spots')
            axis_index.set_xlabel('spot intensity Ch_'+str(i) )
            axis_index.set_ylabel('probability' )
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        return file_name
    
    
    def plot_nuc_cyto_int_ratio_distributions(self, dataframe,output_identification_string=None, plot_for_pseudo_cytosol=False,remove_outliers=True):
        plt.rcdefaults()
        # Creating a name for the plot
        if not (output_identification_string is None):
            index_string = output_identification_string.index('__')
            if index_string >=0:
                title_string = output_identification_string[0:index_string]
            else :
                title_string = ''
        else:
            title_string=''
        # Counting the number of color channels in the dataframe
        if plot_for_pseudo_cytosol == True:
            pattern = r'nuc_pseudo_cyto_int_ratio_ch_\d'
            title_plot  = 'nuc_pseudo_cyto_ratio'
            prefix_column_to_extract = 'nuc_pseudo_cyto_int_ratio_ch_'
            prefix_x_label = 'nuc_pseudo_cyto_int_ratio_ch_'
        else:
            pattern = r'^nuc_cyto_int_ratio_ch_\d'
            title_plot  = 'nuc_cyto_ratio'
            prefix_column_to_extract = 'nuc_cyto_int_ratio_ch_'
            prefix_x_label = 'nuc_cyto_int_ratio_ch_'
        string_list = dataframe.columns
        number_color_channels = 0
        for string in string_list:
            match = re.match(pattern, string)
            if match:
                number_color_channels += 1
        # Plotting
        _, ax = plt.subplots(nrows = 1, ncols = number_color_channels, figsize = (25, 5))
        file_name = title_plot +'_'+title_string+'.pdf'
        colors = ['r','g','b','m']
        number_cells = dataframe['cell_id'].nunique()
        for i in range (0,number_color_channels ):
            colum_to_extract = prefix_column_to_extract+str(i)
            int_ratio = np.asarray( [  dataframe.loc[(dataframe['cell_id']==i)][colum_to_extract].values[0]  for i in range(0, number_cells)] )
            if remove_outliers ==True:
                int_ratio =Utilities().remove_outliers( int_ratio,min_percentile=1,max_percentile=99)
            ax[i].hist(x=int_ratio, bins=30, density = True, histtype ='bar',color = colors[i],label = 'spots')
            ax[i].set_xlabel(prefix_x_label+str(i) )
            ax[i].set_ylabel('probability' )
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        return file_name
        
    def plot_all_distributions (self, dataframe,channels_cytosol, channels_nucleus,channels_spots,minimum_spots_cluster,output_identification_string ):
        plt.rcdefaults()
        if isinstance(channels_spots, list):
            number_channels_spots = (len(channels_spots))
        else:
            number_channels_spots = 1
        list_file_plots_spot_intensity_distributions =[]
        list_file_plots_distributions =[]
        list_file_plots_cell_size_vs_num_spots =[]
        list_file_plots_cell_intensity_vs_num_spots =[]
        # extracting data for each spot type
        for i in range (number_channels_spots):
            number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, number_of_TS_per_cell, ts_size, cell_size, number_cells, nuc_size, cyto_size = Utilities().df_extract_data(dataframe,spot_type=i,minimum_spots_cluster=minimum_spots_cluster)
            file_plots_cell_intensity_vs_num_spots = Plots().plot_cell_intensity_spots(dataframe, number_of_spots_per_cell_nucleus, number_of_spots_per_cell_cytosol,output_identification_string,spot_type=i)
            file_plots_spot_intensity_distributions = Plots().plot_spot_intensity_distributions(dataframe,output_identification_string=output_identification_string,remove_outliers=True,spot_type=i) 
            file_plots_distributions = Plots().plotting_results_as_distributions(number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, ts_size, number_of_TS_per_cell, minimum_spots_cluster, output_identification_string=output_identification_string,spot_type=i)
            file_plots_cell_size_vs_num_spots = Plots().plot_cell_size_spots(channels_cytosol, channels_nucleus, cell_size, number_of_spots_per_cell, cyto_size, number_of_spots_per_cell_cytosol, nuc_size, number_of_spots_per_cell_nucleus,output_identification_string=output_identification_string,spot_type=i)
            # Appending list of files
            list_file_plots_spot_intensity_distributions.append(file_plots_spot_intensity_distributions)
            list_file_plots_distributions.append(file_plots_distributions)
            list_file_plots_cell_size_vs_num_spots.append(file_plots_cell_size_vs_num_spots)
            list_file_plots_cell_intensity_vs_num_spots.append(file_plots_cell_intensity_vs_num_spots)
            del number_of_spots_per_cell, number_of_spots_per_cell_cytosol, number_of_spots_per_cell_nucleus, number_of_TS_per_cell, ts_size
            list_files_distributions = [list_file_plots_spot_intensity_distributions,list_file_plots_distributions,list_file_plots_cell_size_vs_num_spots,list_file_plots_cell_intensity_vs_num_spots]
        return list_files_distributions #list_file_plots_spot_intensity_distributions,list_file_plots_distributions,list_file_plots_cell_size_vs_num_spots,list_file_plots_cell_intensity_vs_num_spots
    
    def compare_intensities_spots_interpretation(self, merged_dataframe, list_dataframes, list_number_cells,  list_labels, plot_title_suffix, destination_folder, column_name, remove_extreme_values= True,max_quantile=0.97,color_palete='CMRmap'):
        plt.rcdefaults()
        file_name = 'ch_int_vs_spots_'+plot_title_suffix+'__'+column_name+'.pdf'
        sns.set(font_scale = 1.5)
        sns.set_style("white")
        # Detecting the number of columns in the dataset
        my_list = list_dataframes[0].columns
        filtered_list = [elem for elem in my_list if column_name in elem]
        list_column_names = sorted(filtered_list)
        number_color_channels = len(list_column_names)
        # Iterating for each color channel
        y_value_label = 'Spot_Count'
        list_file_names =[]
        for i,column_intensity in enumerate(list_column_names):
            x_value_label = 'Channel '+str(i) +' Intensity'
            title_plot='temp__'+str(np.random.randint(1000, size=1)[0])+'_ch_'+str(i)+'_spots.png'
            list_file_names.append(title_plot)
            #column_intensity = column_name +str(i)
            list_cell_int = []
            for j in range (len(list_dataframes)):
                list_cell_int.append( Utilities().function_get_df_columns_as_array(df=list_dataframes[j], colum_to_extract=column_intensity, extraction_type='values_per_cell')  )
            df_cell_int = Utilities().convert_list_to_df (list_number_cells, list_cell_int, list_labels, remove_extreme_values= remove_extreme_values,max_quantile=max_quantile)
            # This code creates a single column for all conditions and adds a 'location' column.
            df_all_melt = merged_dataframe.melt()
            df_all_melt.rename(columns={'value' : y_value_label}, inplace=True)
            df_int_melt = df_cell_int.melt()
            df_int_melt.rename(columns={'value' : x_value_label}, inplace=True)
            data_frames_list = [df_all_melt, df_int_melt[x_value_label]]
            data_frames = pd.concat(data_frames_list, axis=1)
            data_frames
            # Plotting
            plt.figure(figsize=(5,5))
            sns.set(font_scale = 1.5)
            b= sns.scatterplot( data = data_frames, x = x_value_label, y = y_value_label, hue = 'variable',  alpha = 0.9, palette = color_palete)
            b.set_xlabel(x_value_label)
            b.set_ylabel(y_value_label)
            b.legend(fontsize=10)
            plt.xticks(rotation=45, ha="right")
            plt.savefig(title_plot, transparent=False,dpi=360, bbox_inches = 'tight', format='png')
            plt.close()
            del b, data_frames, df_int_melt, df_all_melt, df_cell_int
        # Saving a plot with all channels
        _, axes = plt.subplots(nrows = 1, ncols = number_color_channels, figsize = (15, 7))
        for i in range(number_color_channels):
            axes[i].imshow(plt.imread(list_file_names[i]))
            os.remove(list_file_names[i])
            axes[i].grid(False)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        plt.savefig(file_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        plt.show()
        pathlib.Path().absolute().joinpath(file_name).rename(pathlib.Path().absolute().joinpath(destination_folder,file_name))
        return None
        
    def plot_single_cell_all_channels(self, image, df=None, spot_type=0,min_ts_size=4,show_spots=False,image_name=None,microns_per_pixel=None,max_percentile=99.8):
        plt.rcdefaults()
        # Extracting spot localization
        if not (df is None):
            y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations, number_spots, number_TS, number_spots_selected_z = Utilities().extract_spot_location_from_cell(df=df, spot_type=spot_type, min_ts_size= min_ts_size)
        number_color_channels = image.shape[3]
        # Plotting
        _, axes = plt.subplots(nrows = 1, ncols = number_color_channels, figsize = (25, 7))
        for i in range(0, number_color_channels):
            temp_image = np.max(image[:,: ,:,i],axis=0)
            max_visualization_value = np.percentile(temp_image,max_percentile)
            min_visualization_value = np.percentile(temp_image, 0)
            axes[i].imshow( temp_image,cmap = 'plasma', vmin=min_visualization_value,vmax=max_visualization_value)
            axes[i].grid(False)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(r'$_{max}$z (channel '+str(i)+')')
            if (show_spots == True) and (not(df is None)):
            # Plotting spots on image
                for sp in range (number_spots):
                    circle1=plt.Circle((x_spot_locations[sp], y_spot_locations[sp]), 2, color = 'k', fill = False,lw=1)
                    axes[i].add_artist(circle1)     
                # Plotting TS
                if number_TS >0:
                    for ts in range (number_TS):
                        circleTS=plt.Circle((x_TS_locations[ts], y_TS_locations[ts]), 6, color = 'b', fill = False,lw=3)
                        axes[i].add_artist(circleTS)  
            if not (microns_per_pixel is None): 
                scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
                axes[i].add_artist(scalebar)
        # Saving the image
        if not (image_name is None):               
            if image_name[-4:] != '.png':
                image_name = image_name+'.png'
            plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='png')
        plt.show()
        return None
    
    def plot_single_cell(self, image, df, selected_channel, spot_type=0,min_ts_size=4,show_spots=True,image_name=None,microns_per_pixel=None,show_legend = True,max_percentile=99.5,selected_colormap = 'plasma',show_title=True):
        plt.rcdefaults()
        # Extracting spot localization
        y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations, number_spots, number_TS, number_spots_selected_z = Utilities().extract_spot_location_from_cell(df=df, spot_type=spot_type, min_ts_size= min_ts_size)
        # maximum and minimum values to plot
        max_visualization_value = np.percentile(np.max(image[:,: ,:,selected_channel],axis=0),max_percentile)
        min_visualization_value = np.percentile(np.max(image[:,: ,:,selected_channel],axis=0), 0)
        # Section that detects the number of subplots to show
        if show_spots == True:
            number_columns = 2
            x_plot_size =18
        else:
            number_columns = 1
            x_plot_size =9
        # Plotting
        _, axes = plt.subplots(nrows = 1, ncols = number_columns, figsize = (x_plot_size, 6))
        if show_spots == True:
            axis_index = axes[0]
        else:
            axis_index = axes
        # Visualizing image only
        axis_index.imshow( np.max(image[:,: ,:,selected_channel],axis=0),cmap = selected_colormap,
                    vmin=min_visualization_value, vmax=max_visualization_value)
        axis_index.grid(False)
        axis_index.set_xticks([])
        axis_index.set_yticks([])
        if show_title == True:
            axis_index.set_title(r'$_{max}$z (channel '+str(selected_channel) +')')
        if not (microns_per_pixel is None): 
            scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
            axis_index.add_artist(scalebar)
        # Visualization image with detected spots
        if show_spots == True:
            axes[1].imshow( np.max(image[:,: ,:,selected_channel],axis=0),cmap = selected_colormap,
                            vmin=min_visualization_value, vmax=max_visualization_value)
            axes[1].grid(False)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            axes[1].set_title(r'$_{max}$z channel ('+str(selected_channel) + ') and detected spots')
            if not (microns_per_pixel is None): 
                scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
                axes[1].add_artist(scalebar)
            if show_spots == True:
                # Plotting spots on image
                for i in range (number_spots):
                    if i < number_spots-1:
                        circle1=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'k', fill = False,lw=1)
                    else:
                        circle1=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'k', fill = False,lw=1, label='Spots = '+str(number_spots))
                    axes[1].add_artist(circle1)     
                # Plotting TS
                if number_TS >0:
                    for i in range (number_TS):
                        if i < number_TS-1:
                            circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'b', fill = False,lw=3 )
                        else:
                            circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'b', fill = False,lw=3, label= 'TS = '+str(number_TS) )
                        axes[1].add_artist(circleTS )
                # showing label with number of spots and ts.
                if show_legend == True: 
                    legend = axes[1].legend(loc='upper right',facecolor= 'white')
                    legend.get_frame().set_alpha(None)
        # Saving the image
        if not (image_name is None):                
            if image_name[-4:] != '.png':
                image_name = image_name+'.png'
            plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='png')
        plt.show()
        return None
    
    def plot_cell_all_z_planes(self, image, image_name=None ):
        plt.rcdefaults()
        number_color_channels = image.shape[3]
        number_z_slices = image.shape[0]
        _, axes = plt.subplots(nrows = number_color_channels , ncols = number_z_slices, figsize = ( number_z_slices*2, 10 ))
        for i in range(0, number_z_slices):
            for j in range(0, number_color_channels):
                temp_image = image[i,: ,:,j]
                max_visualization_value = np.percentile(temp_image,99.5)
                min_visualization_value = np.percentile(temp_image, 0)
                axes[j,i].imshow( temp_image,cmap='plasma', vmin=min_visualization_value,vmax=max_visualization_value)
                axes[j,i].grid(False)
                axes[j,i].set_xticks([])
                axes[j,i].set_yticks([])
                if i ==0:
                    axes[j,i].set_ylabel('Channel '+str(j) )
                if j == 0:
                    axes[j,i].set_title(r'$z_{plane}$ '+str(i) )
            # Saving the image
        if not (image_name is None):             
            if image_name[-4:] != '.png':
                image_name = image_name+'.png'
            plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='png')
        plt.show()
        return None
    
    
    def plot_selected_cell_colors(self, image, df, spot_type=0, min_ts_size=None, show_spots=True,use_gaussian_filter = True, spot_color ='k',spot_mark_size=2,image_name=None,microns_per_pixel=None, show_legend=True,list_channel_order_to_plot=[0,1,2], max_percentile=99.8,save_image=False):
        plt.rcdefaults()
        # Extracting spot location
        y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations, number_spots, number_TS, number_spots_selected_z = Utilities().extract_spot_location_from_cell(df=df, spot_type=spot_type, min_ts_size= min_ts_size)
        # Applying Gaussian filter
        if use_gaussian_filter == True:
            filtered_image_selected_cell = GaussianFilter(video=image, sigma = 1).apply_filter()
            max_subsection_image_selected_cell = np.max(filtered_image_selected_cell,axis=0)
        else:
            max_subsection_image_selected_cell = np.max(image[:,: ,:,:],axis=0)
        # Converting to int8
        print('max_sub',np.max(max_subsection_image_selected_cell))
        subsection_image_selected_cell_int8 = Utilities().convert_to_int8(max_subsection_image_selected_cell, rescale=True, min_percentile=0.5, max_percentile=max_percentile)
        print('max',np.max(subsection_image_selected_cell_int8))
        print('shape',subsection_image_selected_cell_int8.shape)
        #print('test', subsection_image_selected_cell_int8.shape[2]<3  )
        # padding with zeros the channel dimension.
        while subsection_image_selected_cell_int8.shape[2]<3:
            zeros_plane = np.zeros_like(subsection_image_selected_cell_int8[:,:,0])
            subsection_image_selected_cell_int8 = np.concatenate((subsection_image_selected_cell_int8,zeros_plane[:,:,np.newaxis]),axis=2)
        # Plot maximum projection
        if show_spots == True:
            number_columns = 2
            x_plot_size =12
        else:
            number_columns = 1
            x_plot_size =6
        _, axes = plt.subplots(nrows = 1, ncols = number_columns, figsize = (x_plot_size, 6))
        if show_spots == True:
            axis_index = axes[0]
        else:
            axis_index = axes
        # Plotting original image
        axis_index.imshow( subsection_image_selected_cell_int8[:,:,list_channel_order_to_plot])
        axis_index.grid(False)
        axis_index.set_xticks([])
        axis_index.set_yticks([])
        if not (microns_per_pixel is None): 
            scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
            axis_index.add_artist(scalebar)
        if show_spots == True:
            # Plotting image with detected spots
            axes[1].imshow( subsection_image_selected_cell_int8[:,:,list_channel_order_to_plot])
            axes[1].grid(False)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            if not (microns_per_pixel is None): 
                scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
                axes[1].add_artist(scalebar)
        if show_spots == True:
            # Plotting spots on image
                for i in range (number_spots):
                    if i < number_spots-1:
                        circle=plt.Circle((x_spot_locations[i], y_spot_locations[i]), spot_mark_size, color = spot_color, fill = False,lw=1)
                    else:
                        circle=plt.Circle((x_spot_locations[i], y_spot_locations[i]), spot_mark_size, color = spot_color, fill = False,lw=1, label='Spots = '+str(number_spots))
                    axes[1].add_artist(circle)     
                # Plotting TS
                if number_TS >0:
                    for i in range (number_TS):
                        if i < number_TS-1:
                            circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'y', fill = False,lw=3 )
                        else:
                            circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'y', fill = False,lw=3, label= 'TS = '+str(number_TS) )
                        axes[1].add_artist(circleTS ) 
                if show_legend == True: 
                    legend = axes[1].legend(loc='upper right',facecolor= 'white')
                    legend.get_frame().set_alpha(None)
        # Saving the image
        if save_image == True and not (image_name is None):
            image_name = str(image_name)
            if image_name[-4:] != '.png':
                image_name = image_name+'.png'
            plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='png')
        plt.show()

        return None
    
    
    def plot_complete_fov(self, list_images, df, number_of_selected_image, use_GaussianFilter=True,microns_per_pixel = None,image_name=None,show_cell_ids=True,list_channel_order_to_plot=None,min_percentile=10, max_percentile=99.5):
        plt.rcdefaults()
        df_selected_cell = df.loc[   (df['image_id']==number_of_selected_image)]
        if use_GaussianFilter == True:
            video_filtered = GaussianFilter(video=list_images[number_of_selected_image], sigma = 1).apply_filter()
            max_complete_image = np.max(video_filtered,axis=0)
        else:
            max_complete_image = np.max(list_images[number_of_selected_image],axis=0)
        max_complete_image_int8 = Utilities().convert_to_int8(max_complete_image, rescale=True, min_percentile=min_percentile, max_percentile=max_percentile)    
        # Plot maximum projection
        _, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 15))
        if not (list_channel_order_to_plot is None):
            axes.imshow( max_complete_image_int8[:,:,list_channel_order_to_plot])
        else:
            axes.imshow( max_complete_image_int8[:,:,[2,1,0]])
        axes.grid(False)
        axes.set_xticks([])
        axes.set_yticks([])
        if not (microns_per_pixel is None): 
            scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.5,location='lower right',box_color='k',color='w', font_properties=font_props)
            axes.add_artist(scalebar)
        if show_cell_ids == True:
            moving_scale =40 # This parameter moves the label position.
            cell_ids_labels = np.unique(df_selected_cell.loc[ :,'cell_id'].values)
            for _, label in enumerate(cell_ids_labels):
                cell_idx_string = str(label)
                Y_cell_location = df_selected_cell.loc[df_selected_cell['cell_id'] == label, 'nuc_loc_y'].values[0]-moving_scale
                X_cell_location = df_selected_cell.loc[df_selected_cell['cell_id'] == label, 'nuc_loc_x'].values[0]
                if X_cell_location>moving_scale:
                    X_cell_location = X_cell_location-moving_scale   
                axes.text(x=X_cell_location, y=Y_cell_location, s=cell_idx_string, fontsize=12, color='w')
        # Saving the image
        if not (image_name is None):                
            if image_name[-4:] != '.png':
                image_name = image_name+'.png'
            plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='png')
        plt.show()
        return None
    
    
    def plot_all_cells_and_spots(self, list_images, complete_dataframe, selected_channel, list_masks_complete_cells= [None], list_masks_nuclei=[None], spot_type=0,list_segmentation_successful=None,min_ts_size=4,image_name=None,microns_per_pixel=None,show_legend = True,show_plot=True,use_max_projection=True):
        plt.rcdefaults()
        # removing images where segmentation was not successful
        if not (list_segmentation_successful is None):
            list_images = [list_images[i] for i in range(len(list_images)) if list_segmentation_successful[i]]
        #Calculating number of subplots 
        number_cells = np.max(complete_dataframe['cell_id'].values)+1
        NUM_COLUMNS = 10
        NUM_ROWS =  math.ceil(number_cells/ NUM_COLUMNS) *2 
        max_size_y_image_size = 800
        y_image_size = np.min((max_size_y_image_size,NUM_ROWS*4))
        # Read the list of masks
        NUM_POINTS_MASK_EDGE_LINE = 100
        if not (list_masks_complete_cells[0] is None):
            list_cell_masks = []
            for _, masks_image in enumerate (list_masks_complete_cells):
                n_masks =np.max(masks_image)
                for i in range(1, n_masks+1 ):
                    tested_mask = np.where(masks_image == i, 1, 0).astype(bool)
                    list_cell_masks.append(tested_mask)
        else:
            list_cell_masks=[None]
        if not (list_masks_nuclei[0] is None):
            list_nuc_masks = []
            for _, masks_image in enumerate (list_masks_nuclei):
                n_masks =np.max(masks_image)
                for i in range(1, n_masks+1 ):
                    tested_mask = np.where(masks_image == i, 1, 0).astype(bool)
                    list_nuc_masks.append(tested_mask)
        else:
            list_nuc_masks=[None]
        _, axes = plt.subplots(nrows = NUM_ROWS, ncols = NUM_COLUMNS, figsize = (30, y_image_size))
        # Extracting image with cell and specific dataframe
        for i in range (0, NUM_ROWS):
            for j in range(0,NUM_COLUMNS):
                if NUM_ROWS == 1:
                    axis_index = axes[j]
                else:
                    axis_index = axes[i,j]
                axis_index.grid(False)
                axis_index.set_xticks([])
                axis_index.set_yticks([])
        # Plotting cells only
        r = 0
        c = 0
        for cell_id in range(0, number_cells):
            if NUM_ROWS == 1:
                axis_index = axes[r]
            else:
                axis_index = axes[r,c]
            image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, dataframe=complete_dataframe)
            # maximum and minimum values to plot
            central_z_slice = int(image.shape[0]/2)
            if use_max_projection ==True:
                temp_image = np.max(image[:,: ,:,selected_channel],axis=0)
            else:
                temp_image = image[central_z_slice,: ,:,selected_channel]
            max_visualization_value = np.percentile(temp_image,99.5)
            min_visualization_value = np.percentile(temp_image, 0)
            # Extracting spot localization
            axis_index.imshow( temp_image,cmap = 'Greys', vmin=min_visualization_value, vmax=max_visualization_value)
            axis_index.grid(False)
            axis_index.set_xticks([])
            axis_index.set_yticks([])
            axis_index.set_title('Cell_'+str(cell_id) )
            #Showing scale bar
            if not (microns_per_pixel is None): 
                scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
                axis_index.add_artist(scalebar)
            # Updating indexes
            c+=1
            if (c>0) and (c%NUM_COLUMNS ==0):
                c=0
                r+=2
        # Plotting cells with detected spots
        r = 1
        c = 0
        for cell_id in range(0, number_cells):
            if NUM_ROWS == 1:
                axis_index = axes[r]
            else:
                axis_index = axes[r,c]
            if not(list_nuc_masks[0] is None) and not(list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=list_cell_masks[cell_id], mask_nuc=list_nuc_masks[cell_id], dataframe=complete_dataframe)
            if (list_nuc_masks[0] is None) and not(list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=list_cell_masks[cell_id], mask_nuc=None, dataframe=complete_dataframe)
            if not(list_nuc_masks[0] is None) and (list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=None, mask_nuc=list_nuc_masks[cell_id], dataframe=complete_dataframe)
            if (list_nuc_masks[0] is None) and (list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=None, mask_nuc=None, dataframe=complete_dataframe)
            # Extracting spot localization
            #y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations, number_spots, number_TS, number_spots_selected_z = Utilities().extract_spot_location_from_cell(df=df, spot_type=spot_type, min_ts_size= min_ts_size)
            # maximum and minimum values to plot
            central_z_slice = int(image.shape[0]/2)
            if use_max_projection ==True:
                temp_image = np.max(image[:,: ,:,selected_channel],axis=0)
                z_slice =None
            else:
                temp_image = image[central_z_slice,: ,:,selected_channel]
                z_slice =central_z_slice
            y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations, number_spots, number_TS, number_spots_selected_z = Utilities().extract_spot_location_from_cell(df=df, spot_type=spot_type, min_ts_size= min_ts_size,z_slice=z_slice)
            max_visualization_value = np.percentile(temp_image,99.5)
            min_visualization_value = np.percentile(temp_image, 0)
            # Plotting
            axis_index.imshow( temp_image,cmap = 'Greys', vmin=min_visualization_value, vmax=max_visualization_value)
            axis_index.grid(False)
            axis_index.set_xticks([])
            axis_index.set_yticks([])
            axis_index.set_title('Cell '+str(cell_id) + ' - Detection')
            # plotting the mask if exitsts
            if not( cell_mask is None):
                temp_contour = find_contours(cell_mask, 0.5, fully_connected='high',positive_orientation='high')
                contour = np.asarray(temp_contour[0])
                downsampled_mask = signal.resample(contour, num = NUM_POINTS_MASK_EDGE_LINE)
                axis_index.fill(downsampled_mask[:, 1], downsampled_mask[:, 0], facecolor = 'none', edgecolor = 'm', linewidth=1.5) 
            if not (nuc_mask is None):
                temp_contour = find_contours(nuc_mask, 0.5, fully_connected='high',positive_orientation='high')
                contour = np.asarray(temp_contour[0])
                downsampled_mask = signal.resample(contour, num = NUM_POINTS_MASK_EDGE_LINE)
                axis_index.fill(downsampled_mask[:, 1], downsampled_mask[:, 0], facecolor = 'none', edgecolor = 'b', linewidth=1.5) 
            # Plotting spots on image
            if number_spots_selected_z >0:
                for i in range (number_spots_selected_z):
                    if i < number_spots_selected_z-1:
                        circle1=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'r', fill = False,lw=0.3)
                    else:
                        circle1=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'r', fill = False,lw=0.3, label='Spots: '+str(number_spots))
                    axis_index.add_artist(circle1)     
            # Plotting TS
            if number_TS >0:
                for i in range (number_TS):
                    if i < number_TS-1:
                        circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'cyan', fill = False,lw=2.5 )
                    else:
                        circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'cyan', fill = False,lw=2.5, label= 'TS: '+str(number_TS) )
                    axis_index.add_artist(circleTS )
            # showing label with number of spots and ts.
            if (show_legend == True) and (number_spots_selected_z>0): 
                legend = axis_index.legend(loc='upper right',facecolor= 'white',prop={'size': 9})
                legend.get_frame().set_alpha(None)
            #Showing scale bar
            if not (microns_per_pixel is None): 
                scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
                axis_index.add_artist(scalebar)
            # Updating indexes
            c+=1
            if (c>0) and (c%NUM_COLUMNS ==0):
                c=0
                r+=2
        # Saving the image
        if not (image_name is None):                
            if image_name[-4:] != '.pdf':
                image_name = image_name+'.pdf' 
            try:
                plt.savefig(image_name, transparent=False,dpi=120, bbox_inches = 'tight', format='pdf')
            except:
                plt.savefig(image_name, transparent=False,dpi=90, bbox_inches = 'tight', format='pdf')
        if show_plot == True:
            plt.show()
        else:
            plt.close()
        return None
    
    
    def plot_all_cells(self, list_images, complete_dataframe, selected_channel, list_masks_complete_cells=[None], list_masks_nuclei=[None],spot_type=0,list_segmentation_successful=None,min_ts_size=4,show_spots=True,image_name=None,microns_per_pixel=None,show_legend = True,show_plot=True):
        plt.rcdefaults()
        # removing images where segmentation was not successful
        if not (list_segmentation_successful is None):
            list_images = [list_images[i] for i in range(len(list_images)) if list_segmentation_successful[i]]
        #Calculating number of subplots 
        number_cells = np.max(complete_dataframe['cell_id'].values)+1
        NUM_COLUMNS = 10
        NUM_ROWS = math.ceil(number_cells/ NUM_COLUMNS)
        max_size_y_image_size = 400
        y_image_size = np.min((max_size_y_image_size,NUM_ROWS*4))
        # Read the list of masks
        NUM_POINTS_MASK_EDGE_LINE = 100
        if not (list_masks_complete_cells[0] is None):
            list_cell_masks = []
            for _, masks_image in enumerate (list_masks_complete_cells):
                n_masks =np.max(masks_image)
                for i in range(1, n_masks+1 ):
                    tested_mask = np.where(masks_image == i, 1, 0).astype(bool)
                    list_cell_masks.append(tested_mask)
        else:
            list_cell_masks=[None]
        if not (list_masks_nuclei[0] is None):
            list_nuc_masks = []
            for _, masks_image in enumerate (list_masks_nuclei):
                n_masks =np.max(masks_image)
                for i in range(1, n_masks+1 ):
                    tested_mask = np.where(masks_image == i, 1, 0).astype(bool)
                    list_nuc_masks.append(tested_mask)
        else:
            list_nuc_masks=[None]
        _, axes = plt.subplots(nrows = NUM_ROWS, ncols = NUM_COLUMNS, figsize = (30, y_image_size))
        # Extracting image with cell and specific dataframe
        for i in range (0, NUM_ROWS):
            for j in range(0,NUM_COLUMNS):
                if NUM_ROWS == 1:
                    axis_index = axes[j]
                else:
                    axis_index = axes[i,j]
                axis_index.grid(False)
                axis_index.set_xticks([])
                axis_index.set_yticks([])
        r = 0
        c = 0
        for cell_id in range(0, number_cells):
            if NUM_ROWS == 1:
                axis_index = axes[r]
            else:
                axis_index = axes[r,c]
            if not(list_nuc_masks[0] is None) and not(list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=list_cell_masks[cell_id], mask_nuc=list_nuc_masks[cell_id], dataframe=complete_dataframe)
            if (list_nuc_masks[0] is None) and not(list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=list_cell_masks[cell_id], mask_nuc=None, dataframe=complete_dataframe)
            if not(list_nuc_masks[0] is None) and (list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=None, mask_nuc=list_nuc_masks[cell_id], dataframe=complete_dataframe)
            if (list_nuc_masks[0] is None) and (list_cell_masks[0] is None):
                image, df, cell_mask, nuc_mask,_ = Utilities().image_cell_selection(cell_id=cell_id, list_images=list_images, mask_cell=None, mask_nuc=None, dataframe=complete_dataframe)
            # Extracting spot localization
            y_spot_locations, x_spot_locations, y_TS_locations, x_TS_locations, number_spots, number_TS, number_spots_selected_z   = Utilities().extract_spot_location_from_cell(df=df, spot_type=spot_type, min_ts_size= min_ts_size)
            # maximum and minimum values to plot
            temp_image = np.max(image[:,: ,:,selected_channel],axis=0)
            max_visualization_value = np.percentile(temp_image,99.5)
            min_visualization_value = np.percentile(temp_image, 0)
            # Plotting
            # Visualizing image only
            if show_spots == False:
                axis_index.imshow( temp_image,cmap = 'plasma', vmin=min_visualization_value, vmax=max_visualization_value)
                axis_index.grid(False)
                axis_index.set_xticks([])
                axis_index.set_yticks([])
                axis_index.set_title('Cell ID '+str(cell_id) )
            # Visualization image with detected spots
            else:
                axis_index.imshow( temp_image,cmap = 'Greys', vmin=min_visualization_value, vmax=max_visualization_value)
                axis_index.grid(False)
                axis_index.set_xticks([])
                axis_index.set_yticks([])
                axis_index.set_title('Cell ID '+str(cell_id))
                # plotting the mask if exitsts
                if not( cell_mask is None):
                    temp_contour = find_contours(cell_mask, 0.5, fully_connected='high',positive_orientation='high')
                    contour = np.asarray(temp_contour[0])
                    downsampled_mask = signal.resample(contour, num = NUM_POINTS_MASK_EDGE_LINE)
                    axis_index.fill(downsampled_mask[:, 1], downsampled_mask[:, 0], facecolor = 'none', edgecolor = 'm', linewidth=1.5) 
                if not (nuc_mask is None):
                    temp_contour = find_contours(nuc_mask, 0.5, fully_connected='high',positive_orientation='high')
                    contour = np.asarray(temp_contour[0])
                    downsampled_mask = signal.resample(contour, num = NUM_POINTS_MASK_EDGE_LINE)
                    axis_index.fill(downsampled_mask[:, 1], downsampled_mask[:, 0], facecolor = 'none', edgecolor = 'b', linewidth=1.5) 
                # Plotting spots on image
                if number_spots >0:
                    for i in range (number_spots):
                        if i < number_spots-1:
                            circle1=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'r', fill = False,lw=0.5)
                        else:
                            circle1=plt.Circle((x_spot_locations[i], y_spot_locations[i]), 2, color = 'r', fill = False,lw=0.5, label='Spots: '+str(number_spots))
                        axis_index.add_artist(circle1)     
                # Plotting TS
                if number_TS >0:
                    for i in range (number_TS):
                        if i < number_TS-1:
                            circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'cyan', fill = False,lw=3 )
                        else:
                            circleTS=plt.Circle((x_TS_locations[i], y_TS_locations[i]), 6, color = 'cyan', fill = False,lw=3, label= 'TS: '+str(number_TS) )
                        axis_index.add_artist(circleTS )
                # showing label with number of spots and ts.
                if (show_legend == True) and (number_spots>0): 
                    legend = axis_index.legend(loc='upper right',facecolor= 'white',prop={'size': 9})
                    legend.get_frame().set_alpha(None)
            #Showing scale bar
            if not (microns_per_pixel is None): 
                scalebar = ScaleBar(dx = microns_per_pixel, units= 'um', length_fraction=0.25,location='lower right',box_color='k',color='w', font_properties=font_props)
                axis_index.add_artist(scalebar)
            # Updating indexes
            c+=1
            if (c>0) and (c%NUM_COLUMNS ==0):
                c=0
                r+=1
        # Saving the image
        if not (image_name is None):                
            if image_name[-4:] != '.pdf':
                image_name = image_name+'.pdf'
            plt.savefig(image_name, transparent=False,dpi=360, bbox_inches = 'tight', format='pdf')
        if show_plot == True:
            plt.show()
        else:
            plt.close()
        return None
    
class SliderWidgetTracking:
    def __init__(self, image_TZYXC, masks=None, list_voxels=None, time_point=None,list_spot_size_px=None, channels_spots=None, channels_cytosol=None, channels_nucleus=None, min_length_trajectory=3, yx_spot_size_in_px=2, starting_threshold=500,channel_for_tracking=0):
        self.image_TZYXC = image_TZYXC
        self.masks = masks if masks is not None else np.ones(image_TZYXC.shape[1:-1])  # Default masks
        self.channels_spots = channels_spots if channels_spots is not None else [0, 1]
        self.channels_cytosol = channels_cytosol if channels_cytosol is not None else []
        self.channels_nucleus = channels_nucleus if channels_nucleus is not None else []
        self.list_voxels = list_voxels if list_voxels is not None else []
        self.list_spot_size_px = list_spot_size_px if list_spot_size_px is not None else []
        self.min_length_trajectory = min_length_trajectory
        self.threshold_for_spot_detection = starting_threshold
        self.yx_spot_size_in_px = yx_spot_size_in_px
        self.last_tracking_params = {}
        self.cached_dataframes = None
        self.cached_images = None
        self.starting_threshold = starting_threshold
        self.time_point = time_point
        self.channel_for_tracking=channel_for_tracking

    def plot_filtered_image(self, selected_time=0, max_percentile=99.5, threshold_for_spot_detection=500):
        plt.rcdefaults()
        # Checking if tracking parameters have changed or if cache is empty
        current_params = (threshold_for_spot_detection)
        if current_params != self.last_tracking_params or self.cached_dataframes is None:
            print("Updating particle tracking...")
            if self.time_point is None:
                self.cached_dataframes, self.cached_images = ParticleTracking(
                    image=self.image_TZYXC,
                    channels_spots=self.channels_spots,
                    masks=self.masks,
                    list_voxels=self.list_voxels,
                    yx_spot_size_in_px=self.yx_spot_size_in_px,
                    z_spot_size_in_px=self.z_spot_size_in_px,
                    channels_cytosol=0,
                    channels_nucleus=None,
                    #minimum_spots_cluster=self.minimum_spots_cluster,
                    min_length_trajectory=self.min_length_trajectory,
                    threshold_for_spot_detection=[threshold_for_spot_detection] * len(self.channels_spots) ).run()
            else:
                df, list_filtered_images, _ = SpotDetection(self.image_TZYXC[self.time_point, ...],
                                                                channels_spots=self.channels_spots,
                                                                channels_cytosol=0,
                                                                channels_nucleus=None,
                                                                masks_complete_cells=self.masks,
                                                                list_voxels=self.list_voxels,
                                                                yx_spot_size_in_px=self.yx_spot_size_in_px,
                                                                z_spot_size_in_px=self.z_spot_size_in_px,
                                                                show_plot=False,
                                                                save_files=False,
                                                                threshold_for_spot_detection=[threshold_for_spot_detection] * len(self.channels_spots),).get_dataframe()
                # reformating the filtered images. Now the dimensions are TZYXC.
                filtered_images_spot_detection = np.zeros(self.image_TZYXC.shape[1:5])
                for ch in range(len(list_filtered_images)):
                    filtered_images_spot_detection[:,:,:,ch] = list_filtered_images[ch]
                filtered_images_temp = np.expand_dims(filtered_images_spot_detection, axis=0)  
                                
                # add a column with continuous numbers indicating 'particle'
                df['particle'] = np.arange(df.shape[0])                
                # Expand dimension to match the shape of the image
                self.cached_dataframes = df
                self.cached_images = filtered_images_temp
            self.last_tracking_params = current_params

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for selected_color, ax in enumerate(axes):
            if self.time_point is None:
                filtered_image = RemoveExtrema(np.max(self.cached_images[selected_time, :, :, :, selected_color], axis=0), min_percentile=0.01, max_percentile=max_percentile).remove_outliers()
            else:
                filtered_image = RemoveExtrema(np.max(self.cached_images[0, :, :, :, selected_color], axis=0), min_percentile=0.01, max_percentile=max_percentile).remove_outliers()
            ax.imshow(filtered_image, cmap='binary_r')
            ax.set_title(f'Image and Spots Detected - Channel {selected_color}', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(False)
            if self.time_point is None: 
                df_tracking = self.cached_dataframes[self.channel_for_tracking] # THIS SELECTS ONLY THE DF_TRACKING THAT IS THE NUMBER 1 IN THE LIST
            else:
                df_tracking = self.cached_dataframes
            if df_tracking is not None:
                if self.time_point is None:
                    df_time = df_tracking[df_tracking['frame'] == selected_time]
                else:
                    df_time = df_tracking
                for particle, group in df_time.groupby('particle'):
                    ax.scatter(group['x']-1, group['y']-1, s=30, marker='o', linewidth=1, edgecolors='r', facecolors='none')
            if self.masks is not None and np.any(self.masks):
                list_contours = Utilities().masks_to_contours(self.masks)
                for contour in list_contours:
                    ax.plot(contour[:, 1], contour[:, 0], color='r', linewidth=2, linestyle='-')
            ax.invert_yaxis()
        plt.tight_layout()
    
    def get_threshold(self):
        return self.last_tracking_params
    def get_cached_dataframes(self):
        if self.time_point is None:
            selected_df = self.cached_dataframes[self.channel_for_tracking]
        else:
            selected_df = self.cached_dataframes
        return selected_df
    def get_cached_images(self):
        return self.cached_images
    def display(self):
        time_slider = widgets.IntSlider(value=0, min=0, max=self.image_TZYXC.shape[0]-1, step=1, description='Time Point:', continuous_update=False)
        threshold_slider = widgets.IntSlider(value=self.starting_threshold, min=350, max=self.starting_threshold*2, step=50, description='Threshold for Spot Detection:', continuous_update=False)
        percentile_slider = widgets.FloatSlider(value=99.95, min=90.0, max=100, step=0.05, description='Max Percentile:', continuous_update=False)
        if self.time_point is None:
            list_widgets = [time_slider, percentile_slider, threshold_slider]
            dic_widgets = {'selected_time': time_slider, 'max_percentile': percentile_slider, 'threshold_for_spot_detection': threshold_slider}
        else:
            list_widgets = [percentile_slider, threshold_slider]
            dic_widgets = {'max_percentile': percentile_slider, 'threshold_for_spot_detection': threshold_slider}
        controls = widgets.VBox(list_widgets, layout=widgets.Layout(align_items='center'))
        out = widgets.interactive_output(self.plot_filtered_image, dic_widgets)
        display(widgets.HBox([out, controls], layout=widgets.Layout(align_items='center')))


class SingleTimePointSpotDetection:
    def __init__(self, image_TZYXC, masks=None, list_voxels=None, list_spot_size_px=None, channels_spots=None, channels_cytosol=None, channels_nucleus=None, yx_spot_size_in_px=2, starting_threshold=500, channel_for_tracking=0):
        self.image_TZYXC = image_TZYXC
        self.masks = masks if masks is not None else np.ones(image_TZYXC.shape[1:-1])  # Default masks
        self.list_voxels = list_voxels if list_voxels is not None else []
        self.list_spot_size_px = list_spot_size_px if list_spot_size_px is not None else []
        self.channels_spots = channels_spots if channels_spots is not None else [0, 1]
        self.channels_cytosol = channels_cytosol if channels_cytosol is not None else []
        self.channels_nucleus = channels_nucleus if channels_nucleus is not None else []
        self.yx_spot_size_in_px = yx_spot_size_in_px
        self.starting_threshold = starting_threshold
        self.number_color_channels = image_TZYXC.shape[-1]
        self.channel_for_tracking = channel_for_tracking
        self.current_time_point = 0
        self.last_tracking_params = {}
        self.cached_dataframes = None
        self.cached_images = None
        self.spot_radius_px = detection.get_object_radius_pixel(
                        voxel_size_nm=(list_voxels[0], list_voxels[1], list_voxels[1]),
                        object_radius_nm=(list_voxels[0]*(list_spot_size_px[0]//2), list_voxels[1]*(list_spot_size_px[1]//2), list_voxels[1]*(list_spot_size_px[1]//2)), ndim=3)

    def detect_spots(self, time_point, threshold_for_spot_detection):
        plt.rcdefaults()
        # Checking if tracking parameters have changed or if cache is empty
        current_params = (threshold_for_spot_detection, time_point) 
        if current_params != self.last_tracking_params or self.cached_dataframes is None:
            print("Detecting spots at time point:", time_point)
            self.current_time_point = time_point
            df = SpotDetection(self.image_TZYXC[time_point, ...],
                                channels_spots=self.channels_spots,
                                channels_cytosol=self.channels_cytosol,
                                channels_nucleus=self.channels_nucleus,
                                masks_complete_cells=self.masks,
                                list_voxels=self.list_voxels,
                                yx_spot_size_in_px=self.yx_spot_size_in_px,
                                z_spot_size_in_px=self.z_spot_size_in_px,
                                show_plot=False,
                                save_files=False,
                                threshold_for_spot_detection=[threshold_for_spot_detection] * len(self.channels_spots)).get_dataframe()[0]
            list_filtered_images =[]
            for ch in range(self.number_color_channels):
                list_filtered_images.append( stack.log_filter(self.image_TZYXC[time_point,:,:,:,ch], sigma=self.spot_radius_px) )
            
            self.last_tracking_params = current_params
            self.cached_dataframes = df
            self.cached_images = list_filtered_images
            return df, list_filtered_images
        else:
            return self.cached_dataframes, self.cached_images
    
    def plot_spots(self, df, list_filtered_images, max_percentile=99.5):
        #num_channels = len(self.channels_spots)
        fig, axes = plt.subplots(1, self.number_color_channels, figsize=(10 * self.number_color_channels, 8))
        if self.number_color_channels == 1:
            axes = [axes]  # Ensure axes is iterable even for one subplot
        for i, ax in enumerate(axes):
            if i < len(list_filtered_images):  # Check if the channel index is within bounds
                # Select the filtered image for the current channel
                channel_image = list_filtered_images[i]
                filtered_image = RemoveExtrema(np.max(channel_image, axis=0), min_percentile=0.01, max_percentile=max_percentile).remove_outliers()
                ax.imshow(filtered_image, cmap='binary_r')
                # Filter dataframe for the current channel if needed
                df_cluster = df[(df['is_cluster'] == True) & (df['cluster_size'] > 1)]
                df_single = df[(df['is_cluster'] == True) & (df['cluster_size'] <= 1)]
                #df_single = df_channel[df_channel['is_cluster'] == False]
                ax.scatter(df_cluster['x'], df_cluster['y'],  label=f'Clustered Spots - Channel {[i]}', s=100, marker='o', linewidth=2, edgecolors='cyan', facecolors='none')
                ax.scatter(df_single['x'], df_single['y'],  label=f'Detected Spots - Channel {[i]}', s=100, marker='o', linewidth=1, edgecolors='magenta', facecolors='none')
                # plot as text the particle number next to the spot
                for i, txt in enumerate(df_single['spot_id']):
                    ax.annotate(txt, (df_single['x'].iloc[i]+2, df_single['y'].iloc[i]+2), fontsize=12, color='cyan')
                for i, txt in enumerate(df_cluster['spot_id']):
                    ax.annotate(txt, (df_cluster['x'].iloc[i]+2, df_cluster['y'].iloc[i]+2), fontsize=12, color='cyan')
                #ax.scatter(df_channel['x'], df_channel['y'], color='red', label=f'Detected Spots - Channel {self.channels_spots[i]}', s=30, marker='o', linewidth=0.5, edgecolors='r', facecolors='none')
                ax.set_title(f'Spots detected at time point {self.current_time_point} - Channel {i}')
                ax.legend()
            else:
                print(f"Channel {i} index out of bounds for filtered images.")
        
        plt.tight_layout()
        
    def get_threshold_and_time(self):
        return self.last_tracking_params
    def get_cached_dataframes(self):
        return self.cached_dataframes
    def get_cached_images(self):
        return self.cached_images
    def display(self):
        def update_plot(time_point, threshold_for_spot_detection, max_percentile):
            df, list_filtered_images = self.detect_spots(time_point, threshold_for_spot_detection)
            self.plot_spots(df, list_filtered_images, max_percentile)
        time_slider = widgets.IntSlider(value=0, min=0, max=self.image_TZYXC.shape[0]-1, step=1, description='Time:', continuous_update=False)
        threshold_slider = widgets.IntSlider(value=self.starting_threshold, min=400, max=5000, step=50, description='Threshold:', continuous_update=False)
        percentile_slider = widgets.FloatSlider(value=99.8, min=90.0, max=100, step=0.05, description='Max Percentile:', continuous_update=False)
        ui = widgets.VBox([time_slider, threshold_slider, percentile_slider])
        out = widgets.interactive_output(update_plot, {'time_point': time_slider, 'threshold_for_spot_detection': threshold_slider, 'max_percentile': percentile_slider})
        # Display all UI components together, ensuring only one output is displayed
        display(ui, out)


class SliderPlotting:
    def __init__(self, image_TZYXC, masks=None, cmap='custom', df_tracking=None, use_gaussian_filter=False, sigma=1.5):
        """
        Parameters:
            image_TZYXC: 5D numpy array with shape [T, Z, Y, X, C].
            masks: optional segmentation masks.
            cmap: Either a single colormap name/string, a list of colormaps, or the string "custom" to use cmap_list_imagej.
            df_tracking: optional tracking DataFrame.
            use_gaussian_filter: if True, apply gaussian filtering.
            sigma: sigma value for gaussian filtering.
        """
        self.image_TZYXC = image_TZYXC
        self.masks = masks
        self.time_points = image_TZYXC.shape[0]
        # If cmap is 'custom', use our predefined custom colormap list.
        if isinstance(cmap, str) and cmap.lower() == 'custom':
            # Use as many colormaps as there are channels, up to the length of cmap_list_imagej.
            num_channels = min(image_TZYXC.shape[-1], len(cmap_list_imagej))
            self.cmap = cmap_list_imagej[:num_channels]
        elif not isinstance(cmap, list):
            # If a single colormap string is given, duplicate it for each channel.
            self.cmap = [cmap] * min(image_TZYXC.shape[-1], 3)
        else:
            self.cmap = cmap
        
        self.df_tracking = df_tracking
        self.number_color_channels = min(image_TZYXC.shape[-1], 3)
        self.use_gaussian_filter = use_gaussian_filter
        self.sigma = sigma

    def convert_to_uint8(self, image, rescale=True, min_percentile=0.001, max_percentile=99.999):
        # Dummy implementation: you would replace this with your actual method.
        if rescale:
            # For example, clip and rescale the image.
            low = np.nanpercentile(image, min_percentile)
            high = np.nanpercentile(image, max_percentile)
            image = np.clip(image, low, high)
        max_val = image.max()
        if max_val > 0:
            image = (image / max_val * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)
        return image

    def plot_filtered_image(self, selected_time=0, max_percentiles=None, sigma=None):
        if max_percentiles is None:
            max_percentiles = [99.5] * self.number_color_channels
        if sigma is None:
            sigma = self.sigma
        T, Z, Y, X, C = self.image_TZYXC.shape
        max_projection_image = np.zeros((Y, X, 3), dtype=np.uint8)
        for ch in range(self.number_color_channels):
            image_channel = self.image_TZYXC[selected_time, :, :, :, ch]
            max_proj = np.max(image_channel, axis=0)
            max_proj_uint8 = self.convert_to_uint8(max_proj, min_percentile=0.001, max_percentile=max_percentiles[ch])
            if self.use_gaussian_filter:
                max_proj_uint8 = gaussian_filter(max_proj_uint8, sigma=sigma)
                max_proj_uint8 = self.convert_to_uint8(max_proj_uint8, rescale=False)
            max_projection_image[:, :, ch] = max_proj_uint8
        # Plotting the individual channels and a combined image.
        fig, axes = plt.subplots(1, self.number_color_channels + 1, figsize=(12, 5))
        for ch in range(self.number_color_channels):
            axes[ch].imshow(max_projection_image[:, :, ch], cmap=self.cmap[ch])
            axes[ch].set_title(f'Color Channel {ch}', fontsize=10)
            axes[ch].axis('off')
        # Combined image
        axes[self.number_color_channels].imshow(max_projection_image)
        axes[self.number_color_channels].set_title('Max Projection', fontsize=10)
        axes[self.number_color_channels].axis('off')
        if self.df_tracking is not None:
            df_time = self.df_tracking[self.df_tracking['frame'] == selected_time]
            for particle, group in df_time.groupby('particle'):
                spot_size = 10
                axes[self.number_color_channels].scatter(group['x']-1, group['y']-1,
                                                          s=spot_size*2, marker='o',
                                                          linewidth=0.5, edgecolors='w',
                                                          facecolors='none')
        if self.masks is not None and np.any(self.masks):
            # Assuming Utilities().masks_to_contours() is defined elsewhere.
            list_contours = Utilities().masks_to_contours(self.masks)
            for contour in list_contours:
                axes[self.number_color_channels].plot(contour[:, 1], contour[:, 0],
                                                       color='w', linewidth=0.5, linestyle='-')
        plt.tight_layout()
        plt.show()

    def display(self):
        # This method creates an interactive slider to update the image.
        time_slider = widgets.IntSlider(
            value=0, min=0, max=self.time_points - 1,
            step=1, description='Time Point:', continuous_update=False
        )
        max_percentile_sliders = [
            widgets.FloatSlider(
                value=99.95, min=90.0, max=100, step=0.01,
                description=f'Ch{ch}:', continuous_update=False
            ) for ch in range(self.number_color_channels)
        ]

        slider_sigma = widgets.FloatSlider(
            value=self.sigma, min=0.01, max=5.0, step=0.05,
            description='Sigma:', continuous_update=False
        )
        controls = widgets.VBox(
            [time_slider] + max_percentile_sliders + [slider_sigma],
            layout=widgets.Layout(align_items='center')
        )
        interactive_kwargs = {
            'selected_time': time_slider,
            'max_percentiles': widgets.fixed([slider.value for slider in max_percentile_sliders]),
            'sigma': slider_sigma
        }
        out = widgets.interactive_output(
            self.plot_filtered_image,
            interactive_kwargs
        )
        display(widgets.HBox([out, controls], layout=widgets.Layout(align_items='center')))

class VideoTracking:
    def __init__(self, image_TZYXC, df_tracking, voxel_xy_um, list_channel_order_to_plot, list_max_percentile, min_percentile, zoom_size, selected_spot, figsize=(10, 10), dpi=150):
        self.image_TZYXC = image_TZYXC
        self.df_tracking = df_tracking
        self.voxel_xy_um = voxel_xy_um
        self.list_channel_order_to_plot = list_channel_order_to_plot
        self.list_max_percentile = list_max_percentile
        self.min_percentile = min_percentile
        self.zoom_size = zoom_size
        self.selected_spot = selected_spot
        self.figsize = figsize
        self.dpi = dpi

    def generate_video_frames(self, max_percentile=99.5):
        video_frames = []
        for time_point in range(self.image_TZYXC.shape[0]):
            fig, ax = Plots().plot_cell_zoom_selected_crop(
                image_TZYXC=self.image_TZYXC,
                df=self.df_tracking,
                use_gaussian_filter=True,
                microns_per_pixel=self.voxel_xy_um,
                time_point=time_point,
                list_channel_order_to_plot=self.list_channel_order_to_plot,
                list_max_percentile=self.list_max_percentile,
                min_percentile=self.min_percentile,
                save_image=False,
                show_spots_ids=True,
                title=f'Time point: {time_point}',
                zoom_size=self.zoom_size,
                selected_spot=self.selected_spot,
                show_plot=False,
                figsize=self.figsize
            )
            # Set the background color of the plot
            ax.set_facecolor('black')
            # Convert the plot to an image array
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            video_frames.append(image)
            plt.close(fig)  # Close the figure to free memory
        return video_frames

    def display_video(self, video_frames):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        im = ax.imshow(video_frames[0])  # Initialize with the first frame
        ax.axis('off')  # Turn off axis

        def update(frame):
            im.set_data(video_frames[frame])
            return im,

        ani = FuncAnimation(fig, update, frames=len(video_frames), blit=True, repeat=True)
        plt.close(fig)  # Prevent extra empty plot from showing in notebook
        return HTML(ani.to_jshtml())

