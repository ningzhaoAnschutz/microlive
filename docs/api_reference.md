# API Reference - MicroLive GUI

## Overview

This document provides comprehensive technical documentation for the MicroLive main classes, methods, and data structures. The application is built using PyQt5 and follows object-oriented design principles for live-cell microscopy image analysis and single-molecule measurements.

## Architecture Overview

The application follows a modular design with the main `GUI` class coordinating between different functional modules:

- **Image I/O & Management**: Multi-format loading (LIF, TIFF, OME-TIFF) with metadata extraction
- **Display System**: Multi-channel visualization with custom colormaps and real-time parameter adjustment
- **Analysis Pipeline**: Segmentation → Tracking → Statistical Analysis with comprehensive quality control
- **Export System**: Comprehensive data and visualization export with metadata preservation

For a complete workflow walkthrough, see the [Tutorial](tutorial.md). For user-focused parameter explanations, see the [User Guide](user_guide.md).

## Main Classes

### GUI Class

The main application window class inheriting from `QMainWindow`.

```python
class GUI(QMainWindow):
    """
    MicroLive is a comprehensive GUI application for microscopy image analysis.
    A PyQt5 QMainWindow‐based application for interactive analysis of multi-dimensional 
    microscopy image data organized into multiple tabs for end-to-end workflows.
    """
```

#### Key Attributes

**Core Image Data:**
```python
image_stack: np.ndarray           # 5D array [T, Z, Y, X, C] - main image data
corrected_image: np.ndarray       # Photobleaching corrected image stack
segmentation_mask: np.ndarray     # Binary segmentation mask (2D)
total_frames: int                 # Number of time frames
number_color_channels: int        # Number of imaging channels
```

**Metadata & Properties:**
```python
voxel_yx_nm: float               # XY pixel size in nanometers
voxel_z_nm: float                # Z voxel size in nanometers
time_interval_value: float       # Time between frames in seconds
bit_depth: int                   # Image bit depth
channel_names: list              # Channel name labels
selected_image_name: str         # Current image identifier
```

**Analysis Results:**
```python
df_tracking: pd.DataFrame         # Particle tracking results
correlation_results: list        # Correlation analysis results
colocalization_results: dict     # Colocalization analysis results
df_random_spots: pd.DataFrame    # Random control spots data
has_tracked: bool                # Flag indicating tracking completion
```

**Display & Navigation:**
```python
current_frame: int                # Current time frame index (0-based)
current_channel: int              # Current channel index (0-based)
display_min_percentile: float     # Display intensity minimum (0.1)
display_max_percentile: float     # Display intensity maximum (99.95)
display_sigma: float              # Gaussian smoothing sigma (0.7)
low_display_sigma: float          # Low-pass smoothing sigma (0.15)
channelDisplayParams: dict        # Per-channel display settings
merged_mode: bool                 # Channel merging state
```

**Tracking Parameters:**
```python
# Detection Parameters (consistent with User Guide ranges)
yx_spot_size_in_px: int          # Spot size in XY pixels (range: 3-15, default: 5)
z_spot_size_in_px: int           # Spot size in Z pixels (range: 1-10, default: 2)
threshold_spot_detection: float  # Detection intensity threshold (image-dependent)
user_selected_threshold: float   # User-defined threshold override
use_maximum_projection: bool     # Use 2D projection for tracking (default: True)
max_spots_for_threshold: int     # Maximum spots for threshold calculation (range: 100-10000, default: 3000)

# Trajectory Linking (consistent with User Guide)
min_length_trajectory: int       # Minimum trajectory length (range: 1-200, default: 20)
maximum_range_search_pixels: int # Linking search range (range: 1-50, default: 7)
memory: int                      # Linking memory frames (range: 0-10, default: 1)
link_using_3d_coordinates: bool  # Use Z coordinates for linking (default: True)

# Clustering (3D mode only, consistent with User Guide)
cluster_radius_nm: int           # Cluster radius in nm (range: 100-2000, default: 500)
maximum_spots_cluster: int       # Max spots per cluster (range: 0-1000, default: None)
separate_clusters_and_spots: bool # Separate cluster/spot analysis (default: False)
```

**Segmentation Parameters:**
```python
segmentation_mode: str           # "manual", "watershed", "cellpose", "None"
use_max_proj_for_segmentation: bool # Use Z projection for segmentation
segmentation_current_channel: int    # Channel used for segmentation
segmentation_current_frame: int      # Frame used for segmentation
segmentation_maxproj: np.ndarray     # Maximum projection for segmentation
watershed_threshold_factor: float    # Watershed threshold factor
```

**Correlation Parameters:**
```python
correlation_fit_type: str        # "linear" or "exponential"
de_correlation_threshold: float  # Decorrelation threshold (default: 0.01)
min_percentage_data_in_trajectory: float # Min data requirement (default: 0.3)
correct_baseline: bool           # Apply baseline correction
remove_outliers: bool            # Remove outlier trajectories
correlation_min_percentile: float # Plot range minimum
correlation_max_percentile: float # Plot range maximum
selected_field_name_for_correlation: str # Field for correlation analysis
index_max_lag_for_fit: int       # Maximum lag for fitting
```

**Photobleaching Parameters:**
```python
photobleaching_calculated: bool  # Correction applied flag
photobleaching_mode: str         # "inside_cell", "outside_cell", "use_circular_region"
photobleaching_model: str        # "exponential", "linear", "double_exponential"
photobleaching_radius: int       # Circular region radius (default: 20)
photobleaching_number_removed_initial_points: int # Excluded initial points
photobleaching_data: dict        # Photobleaching analysis results
```

**Visualization Parameters:**
```python
# Tracking Visualization
tracking_vis_merged: bool        # Merged channel display mode
tracking_vis_channels: list      # Channel selection states
selected_particle_id: int       # Currently selected particle for visualization

# Display Options
tracking_remove_background_checkbox: bool # Remove background overlay
tracking_time_text_checkbox: bool        # Show timestamp overlay
show_trajectories_checkbox: bool         # Show particle trajectories
show_cluster_size_checkbox: bool         # Show cluster size annotations
show_particle_id_checkbox: bool          # Show particle ID labels
```

#### Core Methods

##### Image Loading and Management

```python
def load_lif_image(self, file_path: str, image_index: int) -> None:
    """
    Load specific scene from Leica LIF file.
    
    Extracts metadata (voxel sizes, time intervals, channel names), converts to
    standard 5D format, updates UI elements, and resets analysis state.
    
    Parameters
    ----------
    file_path : str
        Path to .lif file
    image_index : int
        Scene index to load (0-based)
    """

def load_tif_image(self, file_path: str) -> None:
    """
    Load TIFF or OME-TIFF file with metadata parsing.
    
    Attempts to extract embedded metadata (OME-XML, ImageJ), prompts user
    for missing fields, converts to standard format, and initializes GUI.
    
    Parameters
    ----------
    file_path : str
        Path to TIFF file
    """

def convert_to_standard_format(self, image_stack: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary-dimensional image to standard 5D [T, Z, Y, X, C] format.
    
    For non-5D inputs, opens dimension mapping dialog to assign file axes
    to standard dimensions. Inserts singleton dimensions as needed.
    
    Parameters
    ----------
    image_stack : np.ndarray
        Input image array of any dimensionality
        
    Returns
    -------
    np.ndarray
        Standardized 5D image array
    """

def open_dimension_mapping_dialog(self, file_shape: tuple) -> list:
    """
    Open modal dialog for mapping file dimensions to [T, Z, Y, X, C].
    
    Parameters
    ----------
    file_shape : tuple
        Shape of loaded image file
        
    Returns
    -------
    list or None
        Mapping list [T_idx, Z_idx, Y_idx, X_idx, C_idx] or None if cancelled
    """

def on_tree_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
    """Handle file tree selection to load images or scenes."""

def close_selected_file(self) -> None:
    """Remove selected file from tree and free memory."""
```

##### Display and Visualization

```python
def plot_image(self) -> None:
    """
    Update main image display with current frame, channel, and Z-slice.
    
    Applies intensity scaling, smoothing, and colormap based on current
    display parameters and channel settings.
    """

def update_frame(self, value: int) -> None:
    """
    Navigate to specific time frame and update all synchronized sliders.
    
    Parameters
    ----------
    value : int
        Frame index (0-based)
    """

def update_channel(self, channel: int) -> None:
    """
    Switch to specific imaging channel and update displays.
    
    Parameters
    ----------
    channel : int
        Channel index (0-based)
    """

def update_z(self, value: int) -> None:
    """Handle Z-slider changes for 3D navigation."""

def compute_merged_image(self) -> np.ndarray:
    """
    Generate RGB merged image from multiple channels.
    
    Applies channel-specific colormaps (green, magenta, yellow) and
    combines up to 3 channels into single RGB display.
    
    Returns
    -------
    np.ndarray
        RGB image array [Y, X, 3] with values in [0, 1]
    """

def colorize_single_channel(self, gray_img: np.ndarray, channel_index: int) -> np.ndarray:
    """
    Apply predefined colormap to single-channel image.
    
    Channel 0: Green, Channel 1: Magenta, Channel 2: Yellow
    
    Parameters
    ----------
    gray_img : np.ndarray
        Grayscale input image (uint8)
    channel_index : int
        Channel index for color assignment
        
    Returns
    -------
    np.ndarray
        3-channel colorized image
    """

def merge_color_channels(self) -> None:
    """Switch display to merged multi-channel mode."""

def play_pause(self) -> None:
    """Toggle time-lapse playback."""

def next_frame(self) -> None:
    """Advance to next frame with wraparound."""
```

##### Segmentation Methods

```python
def manual_segmentation(self) -> None:
    """
    Enter manual polygon segmentation mode.
    
    Displays current frame with preprocessing, clears previous mask,
    and connects mouse click handler for polygon drawing.
    """

def on_click_segmentation(self, event) -> None:
    """
    Handle mouse clicks during manual segmentation.
    
    Adds polygon vertices on left click, updates display with current polygon.
    """

def finish_segmentation(self) -> None:
    """
    Complete manual segmentation and generate binary mask.
    
    Disconnects click handler, creates mask from polygon vertices,
    and updates segmentation display.
    """

def run_watershed_segmentation(self) -> None:
    """Execute watershed segmentation with current threshold factor."""

def update_watershed_threshold_factor(self, value: int) -> None:
    """Update watershed threshold (slider value / 100)."""

def run_cellpose_segmentation(self) -> None:
    """Execute Cellpose segmentation (if available)."""

def update_segmentation_source(self, state: int) -> None:
    """Toggle between single frame and maximum projection for segmentation."""

def plot_segmentation(self) -> None:
    """Update segmentation display with current image and mask overlay."""
```

##### Tracking Methods

```python
def detect_spots_all_frames(self) -> None:
    """
    Perform spot detection across all time frames without linking.
    
    Uses current threshold and spot size parameters. Shows progress dialog
    and stores results for visualization and subsequent linking.
    """

def perform_particle_tracking(self) -> None:
    """
    Execute complete particle tracking workflow.
    
    Runs detection followed by trajectory linking with current parameters.
    Handles progress display and error recovery.
    """

def detect_spots_in_current_frame(self) -> None:
    """
    Run spot detection on currently displayed frame only.
    
    Used for parameter testing and threshold visualization.
    """

def update_threshold_value(self, value: float) -> None:
    """
    Update detection threshold and refresh histogram visualization.
    
    Parameters
    ----------
    value : float
        New intensity threshold value
    """

def update_threshold_histogram(self) -> None:
    """
    Generate and display intensity histogram for threshold selection.
    
    Computes histogram of current frame/channel within segmentation mask,
    updates slider range, and displays current threshold line.
    """

def track_particles(self, corrected_image: np.ndarray, mask: np.ndarray, 
                   parameters: dict, use_maximum_projection: bool) -> list:
    """
    Core particle tracking function with error handling.
    
    Parameters
    ----------
    corrected_image : np.ndarray
        Image data for tracking
    mask : np.ndarray
        Segmentation mask
    parameters : dict
        Complete tracking parameter set
    use_maximum_projection : bool
        Whether to use 2D projection
        
    Returns
    -------
    list
        List of trajectory DataFrames
    """

def detect_spots(self, image: np.ndarray, threshold: float, 
                list_voxels: list, mask: np.ndarray) -> pd.DataFrame:
    """
    Low-level spot detection function.
    
    Parameters
    ----------
    image : np.ndarray
        Single time point image
    threshold : float
        Detection threshold
    list_voxels : list
        [z_size, xy_size] in physical units
    mask : np.ndarray
        Binary segmentation mask
        
    Returns
    -------
    pd.DataFrame
        Detected spots with coordinates and properties
    """

def generate_random_spots(self, state: int) -> None:
    """Generate random control spots for validation."""

def scale_spots(self) -> float:
    """Determine platform-specific spot scaling factor."""

# Tracking parameter update methods
def update_yx_spot_size(self, value: int) -> None:
    """Update XY spot size, ensuring odd values."""

def update_z_spot_size(self, value: int) -> None:
    """Update Z spot size parameter."""

def update_min_length_trajectory(self, value: int) -> None:
    """Update minimum trajectory length filter."""

def update_max_range_search_pixels(self, value: int) -> None:
    """Update maximum search range for particle linking."""

def update_memory(self, value: int) -> None:
    """Update memory parameter for trajectory linking."""

def update_cluster_radius(self, value: int) -> None:
    """Update cluster radius in nanometers."""

def update_max_spots_cluster(self, value: int) -> None:
    """Update maximum spots per cluster (0 for None)."""

def update_use_maximum_projection(self, state: int) -> None:
    """Toggle 2D projection mode for tracking."""
```

##### Analysis Methods

```python
def compute_photobleaching(self) -> None:
    """
    Apply photobleaching correction with selected model.
    
    Fits decay curve to intensity time course and generates corrected
    image stack. Updates photobleaching_calculated flag and displays results.
    """

def plot_photobleaching(self) -> None:
    """Display photobleaching analysis results with fitted curves."""

def compute_correlations(self) -> None:
    """
    Execute auto- or cross-correlation analysis.
    
    Computes temporal correlations from tracking data with current
    field selection and fitting parameters.
    """

def compute_colocalization(self) -> None:
    """
    Perform automated colocalization analysis.
    
    Uses ML-based or intensity-based method to classify spot pairs
    across channels. Generates crop visualization matrix.
    """

def plot_intensity_histogram(self) -> None:
    """Generate distribution histogram for selected field and channel."""

def plot_intensity_time_course(self) -> None:
    """Display time course plot for selected data type and channel."""

def display_correlation_plot(self) -> None:
    """Render correlation analysis results with fitted curves."""

def display_colocalization_results(self, mean_crop: np.ndarray, crop_size: int,
                                 flag_vector: np.ndarray, ch1: int, ch2: int) -> None:
    """Display colocalization matrix visualization."""
```

##### Export Methods

```python
def export_tracking_data(self) -> None:
    """Export tracking DataFrame to CSV with default naming."""

def export_displayed_image_as_png(self) -> None:
    """Export current display as high-resolution PNG (300 DPI)."""

def export_tracking_video(self) -> None:
    """Export tracking visualization as MP4 or GIF video."""

def export_displayed_video(self) -> None:
    """Export Import tab time-lapse as video."""

def get_default_export_filename(self, prefix: str = None, extension: str = None) -> str:
    """
    Generate standardized filename for exports.
    
    Format: [prefix_]filename_imagename[.extension]
    
    Parameters
    ----------
    prefix : str, optional
        Descriptive prefix (e.g., "tracking", "colocalization")
    extension : str, optional
        File extension
        
    Returns
    -------
    str
        Formatted filename
    """

def export_selected_items(self) -> None:
    """Batch export selected items from Export tab."""

def export_metadata(self) -> None:
    """Export complete analysis parameters to text file."""
```

## Tracking Pipeline Classes

The particle tracking functionality in the GUI utilizes three main classes from `microscopy.py`:

### ParticleTracking Class

```python
class ParticleTracking:
    """
    Main class for detecting and linking particles in time-lapse microscopy data.
    
    Coordinates the complete tracking workflow from detection through trajectory
    linking, with support for clustering analysis and multi-channel intensity
    measurements.
    """
    
    def __init__(self, image, channels_spots, list_voxels, channels_cytosol, channels_nucleus,
                 remove_clusters=False, maximum_spots_cluster=None, min_length_trajectory=10,
                 threshold_for_spot_detection=100, masks=None, memory=0, yx_spot_size_in_px=5, 
                 z_spot_size_in_px=2, cluster_radius_nm=None, link_particles=True, 
                 use_trackpy=False, use_maximum_projection=False, separate_clusters_and_spots=False,
                 maximum_range_search_pixels=10, link_using_3d_coordinates=False,
                 neighbor_strategy='KDTree', generate_random_particles=False,
                 number_of_random_particles_trajectories=None):
        """
        Initialize particle tracking with comprehensive parameter set.
        
        Parameters
        ----------
        image : ndarray
            5D image array [T, Z, Y, X, C]
        channels_spots : list
            Channel indices containing spot signals
        list_voxels : list
            [z_size, xy_size] voxel dimensions in nm
        masks : ndarray, optional
            Binary mask defining analysis region
        threshold_for_spot_detection : float
            Intensity threshold for spot detection
        min_length_trajectory : int
            Minimum trajectory length for filtering
        memory : int
            Frames a particle can disappear and reappear
        maximum_range_search_pixels : int
            Maximum search distance for linking (pixels)
        cluster_radius_nm : float
            Clustering radius in nanometers
        link_using_3d_coordinates : bool
            Use Z coordinates for trajectory linking
        generate_random_particles : bool
            Generate random control trajectories
        """
    
    def run(self) -> list:
        """
        Execute complete tracking workflow.
        
        Returns
        -------
        list
            List of trajectory DataFrames, one per linked trajectory
        """
```

### SpotDetection Class

```python
class SpotDetection:
    """
    Spot detection and data extraction for microscopy images.
    
    Handles detection across multiple channels with clustering analysis
    and comprehensive intensity measurements per spot. Used internally
    by the ParticleTracking workflow.
    
    The detection process follows these steps:
    1. Image preprocessing and filtering
    2. Local maxima detection using specified spot sizes
    3. Threshold-based spot filtering  
    4. Clustering analysis (3D mode only)
    5. Intensity measurements with background correction
    6. Quality metric calculation (SNR, PSF fitting)
    
    This corresponds to the "Single Frame" and "Detection" buttons
    in the Tracking tab GUI.
    """
    
    def __init__(self, image, channels_spots, channels_cytosol, channels_nucleus, 
                 cluster_radius_nm=500, masks_complete_cells=None, masks_nuclei=None, 
                 masks_cytosol_no_nuclei=None, dataframe=None, image_counter=0, 
                 list_voxels=[500,160], show_plot=True, yx_spot_size_in_px=None, 
                 z_spot_size_in_px=None, use_trackpy=False, use_maximum_projection=False,
                 calculate_intensity=True, use_fixed_size_for_intensity_calculation=True):
        """
        Initialize spot detection with segmentation masks and parameters.
        
        Parameters match those available in the Tracking tab GUI:
        
        Parameters
        ----------
        image : ndarray
            4D image array [Z, Y, X, C] for single time point
        channels_spots : list
            Channel indices for spot detection (set by current channel in GUI)
        masks_complete_cells : ndarray, optional
            Cell segmentation masks from Segmentation tab
        masks_nuclei : ndarray, optional
            Nuclear segmentation masks (if available)
        cluster_radius_nm : float, default=500
            Clustering radius in nanometers (GUI: "Cluster radius (nm)")
        calculate_intensity : bool, default=True
            Whether to measure spot intensities using disk-doughnut method
        use_fixed_size_for_intensity_calculation : bool, default=True
            Use fixed aperture vs. cluster-size-dependent aperture
        use_maximum_projection : bool, default=False
            Use 2D projection (GUI: "Use 2D Projection for Tracking" checkbox)
        yx_spot_size_in_px : int, optional
            XY spot size from GUI parameter (default uses GUI setting)
        z_spot_size_in_px : int, optional
            Z spot size from GUI parameter (default uses GUI setting)
        """
```

### DataProcessing Class

```python
class DataProcessing:
    """
    Process detected spots and extract cellular localization information.
    
    Links spots to segmentation masks, calculates cellular properties,
    and generates comprehensive metadata for each detected spot.
    """
    
    def __init__(self, clusters_and_spots, image, masks_complete_cells, masks_nuclei, 
                 masks_cytosol_no_nuclei, channels_cytosol, channels_nucleus, 
                 yx_spot_size_in_px, spot_type=0, dataframe=None, reset_cell_counter=False,
                 image_counter=0, number_color_channels=None, use_maximum_projection=False,
                 use_fixed_size_for_intensity_calculation=True):
        """
        Initialize data processing with spot coordinates and segmentation masks.
        
        Parameters
        ----------
        clusters_and_spots : ndarray
            Detected spot coordinates [N, 3 or 4] with (z, y, x, cluster_size)
        image : ndarray
            Image data for intensity measurements
        masks_complete_cells : ndarray
            Cell segmentation masks
        masks_nuclei : ndarray
            Nuclear segmentation masks
        spot_type : int
            Identifier for spot channel/type
        use_fixed_size_for_intensity_calculation : bool
            Use fixed aperture vs. cluster-size-dependent aperture
        """
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Process spots and generate comprehensive results DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Processed results with cellular localization and intensity data
        """
```

## Machine Learning-Based Colocalization

The application implements a convolutional neural network (CNN) for automated spot colocalization analysis, located in the `modeling/machine_learning` directory. This ML approach provides an alternative to traditional intensity-based colocalization methods.

### CNN Architecture

The colocalization model uses a lightweight CNN architecture (`ParticleDetectionCNN`) designed for binary classification of small image crops:

```python
class ParticleDetectionCNN(nn.Module):
    """
    Convolutional Neural Network for particle detection in microscopy image crops.
    
    Architecture:
    - Input: 11x11 grayscale image crops (resized to 64x64 for training)
    - 3 Convolutional layers with ReLU activation and MaxPooling
    - 2 Fully connected layers with dropout
    - Output: Binary classification (particle present/absent)
    """
```

**Network Layers:**
- **Conv1**: 1→16 channels, 3×3 kernel, ReLU, 2×2 MaxPool
- **Conv2**: 16→32 channels, 3×3 kernel, ReLU, 2×2 MaxPool  
- **Conv3**: 32→64 channels, 3×3 kernel, ReLU, 2×2 MaxPool
- **FC1**: Flattened → 128 units, ReLU, 50% Dropout
- **FC2**: 128 → 1 unit, Sigmoid output

### Training Data Generation

The ML model is trained on multiple data sources to ensure robust performance across diverse experimental conditions:

#### 1. Real Microscopy Data
```python
def create_crops_from_image(real_image, df_tracking, minimal_snr=0.8, 
                           grid_size=11, selected_color_channel=0):
    """
    Extract training crops from real microscopy images using tracking data.
    
    This function generates positive and negative training examples from
    actual experimental data, ensuring the model learns from realistic
    imaging conditions and noise characteristics.
    
    Positive samples: High-SNR tracked particles (SNR > 0.8)
    Negative samples: Random image regions without particles
    
    Parameters
    ----------
    real_image : ndarray
        5D microscopy image stack [T, Z, Y, X, C]
    df_tracking : pd.DataFrame
        Tracking results from ParticleTracking analysis
    minimal_snr : float, default=0.8
        Minimum signal-to-noise ratio for positive examples
    grid_size : int, default=11
        Size of extracted crops (should match model input)
    selected_color_channel : int, default=0
        Channel index for crop extraction
        
    Returns
    -------
    ndarray
        Normalized crop image [11, 11] ready for training
    """
```

#### 2. Simulated Gaussian Spots
```python
def plot_spot(amplitude=None, sigma=None, grid_size=11, mu_x=None, mu_y=None,
              percentage_noise=None, create_spot=False, number_spots=1):
    """
    Generate synthetic particle crops with realistic noise characteristics.
    
    Creates simulated fluorescent spots matching typical microscopy PSF
    properties, including realistic noise models and variable parameters
    to improve model generalization.
    
    Features:
    - Variable amplitude (200-255 intensity units)
    - Gaussian PSF with σ = 0.5-2.0 pixels  
    - Random positioning within crop center ±4 pixels
    - Poisson noise modeling (1-10% of signal amplitude)
    - Support for single and double-particle spots
    - Background intensity variation
    
    This corresponds to the simulated data used in the ML training
    pipeline described in the Tutorial.
    """
```

#### 3. Human-Annotated Ground Truth
The model incorporates expert human annotations from multiple researchers to establish consensus ground truth:

```python
# Consensus ground truth from multiple annotators
flag_vector_consensus = np.sum([annotator_a, annotator_b, annotator_c, annotator_d], 
                              axis=0) >= (num_annotators / 2)
```

**Inter-annotator Agreement:**
- Pearson correlations between expert annotators: 0.65-0.85
- Consensus threshold: ≥50% agreement across annotators

### Training Pipeline

**Data Composition:**
- Real microscopy crops: 40% (1000 positive + 500 negative samples)
- Simulated single spots: 25% (500 samples)
- Simulated double spots: 10% (256 samples)  
- Human-validated crops: 25% (consensus annotations)

**Training Parameters:**
```python
batch_size = 256
num_epochs = 51,200  # (batch_size × 200)
learning_rate = 1e-6
optimizer = Adam
loss_function = BCELoss  # Binary Cross-Entropy
validation_split = 0.2
```

**Data Augmentation:**
- Random rotation (±180°)
- Intensity normalization to [0, 255]
- Gaussian noise injection
- Random positioning within crops

### Model Performance

The ML method demonstrates superior performance compared to traditional intensity-based approaches:

**Accuracy Metrics:**
- **ML (Real Data)**: 85-92% accuracy
- **ML (Simulated Data)**: 88-95% accuracy  
- **ML (Human-Validated)**: 90-96% accuracy
- **Traditional SNR Method**: 75-82% accuracy

**Validation Protocol:**
```python
def calculate_performance(predicted, ground_truth):
    TP = true_positives
    FP = false_positives  
    TN = true_negatives
    FN = false_negatives
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    return TP, FP, TN, FN, accuracy
```

### Implementation in Colocalization Analysis

The ML model is integrated into the colocalization workflow as follows:

1. **Crop Extraction**: Generate 11×11 pixel crops around each detected spot
2. **Preprocessing**: Normalize intensity to [0, 255] range
3. **Inference**: Apply trained CNN to classify each crop
4. **Thresholding**: Apply confidence threshold (default: 0.5-0.67)
5. **Visualization**: Display results in crop matrix with ML flags

```python
def compute_colocalization(self):
    """ML-based colocalization implementation in GUI."""
    if self.method_ml_radio.isChecked():
        # Load pre-trained model
        model = ML.ParticleDetectionCNN()
        ML.load_model(model, 'particle_detection_cnn.pth')
        
        # Generate normalized crops
        list_crops = AM.Utilities().normalize_crop_return_list(
            array_crops_YXC=mean_crop,
            crop_size=crop_size,
            selected_color_channel=selected_channel,
            normalize_to_255=True
        )
        
        # ML inference
        threshold = self.ml_threshold_input.value()
        flag_vector = ML.predict_crops(model, list_crops, threshold=threshold)
        method_used = "ML"
```

### Model Files and Training Notebooks

**Key Components:**
- `modeling/machine_learning/ML_SpotDetection.py`: CNN implementation and training functions
- `modeling/machine_learning/MachineLearning_spot_detection.ipynb`: Interactive model training
- `modeling/machine_learning/ML_Pipeline_and_Data_Validation.ipynb`: Complete training pipeline
- Pre-trained models: `particle_detection_cnn.pth`, `particle_detection_cnn_real_data.pth`

**Training Data Validation:**
The training pipeline includes comprehensive validation against human expert annotations, ensuring the model learns robust features for particle detection across diverse experimental conditions and imaging parameters.

## Data Structures

### Tracking DataFrame Structure

The tracking functionality generates a comprehensive pandas DataFrame (`df_tracking`) containing all particle information across time. This DataFrame is the primary output of the particle tracking pipeline and serves as input for subsequent analyses in the Distribution, Time Course, and Correlation tabs.

**Access in GUI**: Available after running "Tracking" button, exported via "Export DataFrame" button.

**Related Tutorial Section**: [Step 4: Particle Tracking](tutorial.md#step-4-particle-tracking)

**Related User Guide Section**: [Particle Tracking](user_guide.md#particle-tracking)

#### Core Spatial and Temporal Columns

These columns are always present after successful tracking:

| Column | Type | Description | Units | GUI Source |
|--------|------|-------------|-------|------------|
| `frame` | int | Time frame index (0-based) | frames | Time slider position |
| `x` | float | X coordinate of particle centroid | pixels | Image coordinates |
| `y` | float | Y coordinate of particle centroid | pixels | Image coordinates |
| `z` | float | Z coordinate of particle centroid | pixels | Z-slice or 3D detection |
| `particle` | int | Unique trajectory identifier | - | Trajectory linking results |

#### Intensity Measurements (Per Channel)

For each imaging channel N (where N = 0, 1, 2, ...), the following intensity columns are automatically generated. The specific measurements depend on the "Use fixed size for intensity calculation" setting in the Tracking tab.

**Background-Subtracted Intensity (Primary measurement):**
| Column | Type | Description | Units | Calculation Method |
|--------|------|-------------|-------|-------------------|
| `spot_int_ch_N` | float | Integrated intensity using disk-doughnut method | counts | See [User Guide: Distribution Analysis](user_guide.md#distribution-analysis) |

**Additional Intensity Measurements:**
| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `total_spot_int_ch_N` | float | Sum of all pixel values within spot region | counts |
| `psf_amplitude_ch_N` | float | Peak intensity from 2D Gaussian fit | counts |
| `psf_sigma_ch_N` | float | Standard deviation from 2D Gaussian fit | pixels |
| `snr_ch_N` | float | Signal-to-noise ratio | - |

**Quality Control Note**: If Gaussian fitting fails for a spot, `psf_amplitude_ch_N` and `psf_sigma_ch_N` will contain NaN values. The `snr_ch_N` calculation is robust to fitting failures.

#### Particle Properties and Clustering Information

These columns provide information about spot detection and clustering results:

| Column | Type | Description | Units | Available When |
|--------|------|-------------|-------|----------------|
| `cluster_size` | int | Number of spots grouped in cluster (1 = individual spot) | spots | Always (3D mode provides detailed clustering) |
| `is_cluster` | bool | Whether particle is part of a multi-spot cluster | - | When cluster_size > 1 |
| `spot_type` | int | Channel identifier for analysis | - | Multi-channel tracking |
| `spot_id` | int | Unique spot identifier within each frame | - | Always |

**Cluster Size Interpretation** (see [User Guide: Tracking Algorithms](user_guide.md#tracking-algorithms)):
- `cluster_size = 1`: Individual isolated spot
- `cluster_size = 2-5`: Small molecular complexes  
- `cluster_size > 5`: Large assemblies or transcriptional factories

#### Cellular Localization (When Segmentation Available)

| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `cell_id` | int | Unique cell identifier | - |
| `is_nuc` | bool | Whether spot is located in nucleus | - |
| `is_cell_fragmented` | int | Flag indicating if the Cell touches the border of the image (no cell -1, complete cell 0, or fragmented cell 1) | - |
| `nuc_loc_y` | float | Nuclear centroid Y coordinate | pixels |
| `nuc_loc_x` | float | Nuclear centroid X coordinate | pixels |
| `cyto_loc_y` | float | Cytoplasmic centroid Y coordinate | pixels |
| `cyto_loc_x` | float | Cytoplasmic centroid X coordinate | pixels |
| `nuc_area_px` | float | Nuclear area | pixels² |
| `cyto_area_px` | float | Cytoplasmic area | pixels² |
| `cell_area_px` | float | Total cell area | pixels² |

#### Intensity Context (When Segmentation Available)

For each channel N with segmentation:

| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `nuc_int_ch_N` | float | Average nuclear intensity | counts |
| `cyto_int_ch_N` | float | Average cytoplasmic intensity | counts |
| `pseudo_cyto_int_ch_N` | float | Pseudo-cytoplasm intensity (dilated nucleus) | counts |
| `nucleus_cytosol_intensity_ratio_ch_N` | float | Nuclear/cytoplasmic intensity ratio | - |
| `nucleus_pseudo_cytosol_intensity_ratio_ch_N` | float | Nuclear/pseudo-cytoplasm ratio | - |

#### Metadata Columns

| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `image_id` | int | Source image identifier | - |
| `spot_id` | int | Unique spot identifier within image | - |

### DataFrame Generation Workflow

The tracking DataFrame is generated through the following pipeline:

1. **Spot Detection**: `SpotDetection` class detects spots in each frame
2. **Trajectory Linking**: `ParticleTracking` class links spots across time
3. **Data Processing**: `DataProcessing` class adds cellular context and measurements
4. **Quality Filtering**: Remove short trajectories and apply quality filters

### Usage Examples

**Access specific particle trajectory:**
```python
# Get all data points for particle ID 42
particle_data = df_tracking[df_tracking['particle'] == 42]

# Sort by frame for time-course analysis
particle_timecourse = particle_data.sort_values('frame')
```

**Extract intensity time course (for Correlation Analysis):**
```python
# Get channel 0 intensity over time for particle 42
particle_42 = df_tracking[df_tracking['particle'] == 42].sort_values('frame')
intensity_timecourse = particle_42['spot_int_ch_0'].values
time_points = particle_42['frame'].values

# Convert to physical time units (if time interval known)
if time_interval_seconds is not None:
    time_seconds = time_points * time_interval_seconds
```

**Filter by cellular localization (requires segmentation):**
```python
# Get only nuclear spots
nuclear_spots = df_tracking[df_tracking['is_nuc'] == True]

# Get spots in specific cell
cell_5_spots = df_tracking[df_tracking['cell_id'] == 5]

# Filter by cell quality (complete cells only)
complete_cell_spots = df_tracking[df_tracking['is_cell_fragmented'] == 0]
```

**Quality control and analysis:**
```python
# Calculate trajectory lengths (for minimum length filtering)
trajectory_lengths = df_tracking.groupby('particle').size()

# Get high-quality spots (high SNR, good PSF fit)
high_quality = df_tracking[
    (df_tracking['snr_ch_0'] > 3.0) & 
    (df_tracking['psf_sigma_ch_0'].notna()) &
    (df_tracking['psf_sigma_ch_0'] < 3.0)  # Reasonable PSF width
]

# Analyze cluster properties (3D tracking mode)
cluster_summary = df_tracking[df_tracking['cluster_size'] > 1].groupby('particle').agg({
    'cluster_size': 'mean',
    'spot_int_ch_0': 'mean'
})
```

These examples demonstrate how to use the tracking DataFrame for the analyses available in the GUI tabs (Distribution, Time Course, Correlation) and for custom analysis scripts.

### Quality Control Columns

The DataFrame includes several quality control indicators:

- **`cluster_size`**: Spots with `cluster_size > 1` may represent clustered particles
- **`snr_ch_N`**: Signal-to-noise ratio for quality assessment
- **`is_cell_fragmented`**: Indicates cell boundary quality (-1: poor, 0: good, 1: excellent)
- **`psf_sigma_ch_N`**: PSF width for spot quality assessment

### Colocalization Results Dictionary

```python
colocalization_results = {
    'mean_crop_filtered': np.ndarray,    # Processed crop images [H, W, C]
    'crop_size': int,                    # Crop window size (pixels)
    'flag_vector': np.ndarray,           # Boolean colocalization flags
    'ch1_index': int,                    # Reference channel index
    'ch2_index': int,                    # Target channel index
    'num_spots_reference': int,          # Total reference spots
    'num_spots_colocalize': int,         # Colocalized spots count
    'colocalization_percentage': float,  # Percentage colocalized
    'threshold_value': float,            # Analysis threshold used
    'method': str                        # 'ML' or 'Intensity'
}
```
