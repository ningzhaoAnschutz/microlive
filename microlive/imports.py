# Import standard libraries
import sys
import os
import pathlib
import tempfile
from pathlib import Path
import warnings
import getpass
import importlib
import random
import time
import math
import socket
import datetime
from functools import partial, wraps
from itertools import compress
import io
import traceback
import subprocess
import re
import bisect
from numba import njit, types
from numba.typed import List as TypedList
import cv2
import io
import fpdf
import json

# Import third-party libraries
import shutil
import urllib.request
import gzip
import inspect
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.colors as mcolors
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from skimage.io import imread
from skimage.measure import regionprops
from skimage import exposure  # for adaptive histogram equalization
from skimage.filters import threshold_local
from skimage.morphology import disk, binary_opening, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.transform import hough_circle, hough_circle_peaks
import openpyxl
import bigfish.stack as stack
from PIL import Image
from joblib import Parallel, delayed
from scipy.ndimage import binary_opening as binary_opening_ndi 
from scipy.ndimage import binary_closing as binary_closing_ndi
from scipy.ndimage import gaussian_filter, binary_dilation,gaussian_filter1d , uniform_filter1d#label
from scipy.ndimage import label as ndi_label
from scipy.signal import find_peaks
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import curve_fit
from scipy.stats import linregress, pearsonr, gaussian_kde, mannwhitneyu,ttest_ind
from statsmodels.stats.multitest import multipletests
from snapgene_reader import snapgene_file_to_dict, snapgene_file_to_seqrecord
from IPython.display import Image as IPImage, display
import ipywidgets as widgets
import imageio
import contextlib, io
_f = io.StringIO()
with contextlib.redirect_stdout(_f), contextlib.redirect_stderr(_f):
    from cellpose import models, denoise
import seaborn as sns
from dna_features_viewer import GraphicFeature, GraphicRecord, CircularGraphicRecord
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import CodonAdaptationIndex
from Bio.SeqRecord import SeqRecord
from typing import List
import trackpy as tp
tp.quiet(suppress=True)
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgb

try:
    import torch
except ImportError:
    print("Warning: napari and/or napari-animation not found. Some functionality will not be available.")
    napari = None

import logging
logging.getLogger("root").setLevel(logging.ERROR)

# Suppress Cellpose verbose logging (only show warnings and errors)
logging.getLogger('cellpose').setLevel(logging.WARNING)
logging.getLogger('cellpose.core').setLevel(logging.WARNING)
logging.getLogger('cellpose.models').setLevel(logging.WARNING)

from mpl_toolkits.mplot3d import Axes3D  # For older versions of Matplotlib
import matplotlib.colors as mcolors


# Get the username
computer_user_name = getpass.getuser()

# =============================================================================
# Package-aware imports and paths (pip packaging compatible)
# =============================================================================

# Import MicroLive modules using direct module imports (avoid circular import via __init__.py)
from microlive.microscopy import *  # Direct module import, not through package __init__
import microlive.microscopy as mi
import microlive.ml_spot_detection as ML
from microlive.utils.resources import get_icon_path, get_model_path

# Get icon path (works both in development and installed modes)
icon_file = get_icon_path()
if icon_file is None:
    warnings.warn("Icon file not found. Some GUI features may not display correctly.")

# Load machine learning model
try:
    model_path = get_model_path()
    if model_path is not None and model_path.exists():
        model_ML = ML.ParticleDetectionCNN()
        ML.load_model(model_ML, model_path)
    else:
        model_ML = None
        warnings.warn("ML model not found. ML spot detection will not be available.")
except Exception as e:
    model_ML = None
    warnings.warn(f"Error loading machine learning model: {e}")



# --- Set microscope folder path based on operating system ---
if sys.platform.startswith('win'):
    # Windows example (adjust the path as needed)
    microscope_folder_path = Path(f"C:/Users/{computer_user_name}/OneDrive - TheUniversityofColoradoDenver/Microscope")
elif sys.platform.startswith('darwin'):
    # macOS path (your original path)
    microscope_folder_path = Path(
        f"/Users/{computer_user_name}/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/"
        "General - Zhao (NZ) Lab/Microscope"
    )
elif sys.platform.startswith('linux'):
    # Linux example (adjust as needed)
    microscope_folder_path = Path(f"/home/{computer_user_name}/Microscope")
else:
    # Fallback: use the home directory plus a default folder name
    microscope_folder_path = Path.home() / "Microscope"

# Suppress warnings
warnings.filterwarnings("ignore")



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

