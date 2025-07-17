# User Guide - MicroLive Application

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Loading and Managing Images](#loading-and-managing-images)
4. [Display and Visualization](#display-and-visualization)
5. [Segmentation](#segmentation)
6. [Photobleaching Correction](#photobleaching-correction)
7. [Particle Tracking](#particle-tracking)
8. [Statistical Analysis](#statistical-analysis)
9. [Colocalization Analysis](#colocalization-analysis)
10. [Data Export](#data-export)
11. [Best Practices](#best-practices)

## Getting Started

### Quick Start Workflow

For users new to MicroLive, follow this basic workflow:

1. **Load Data**: Open LIF or TIFF files via the Import tab
2. **Segment Cells**: Define regions of interest in the Segmentation tab  
3. **Track Particles**: Detect and link spots in the Tracking tab
4. **Analyze Results**: Generate plots and statistics in Distribution/Correlation tabs
5. **Export Data**: Save results and visualizations via the Export tab

For detailed step-by-step instructions, see the [Complete Tutorial](tutorial.md).

### File Format Support

**Input Formats:**
- **LIF Files**: Native Leica format with complete metadata extraction
- **TIFF/OME-TIFF**: Standard microscopy formats with embedded metadata support
- **Multi-dimensional Arrays**: Automatic dimension mapping for arbitrary file structures

**Metadata Handling:**
- Automatic extraction of voxel sizes, time intervals, and channel information
- Manual entry prompts for missing critical metadata
- Validation and unit conversion (μm to nm, etc.)

## Interface Overview

### Tab Organization
The application is organized into 12 main tabs, each serving specific functions:

**Core Modules:**
- **Import**: Primary image visualization and navigation
- **Segmentation**: Define regions of interest
- **Tracking**: Particle detection and trajectory analysis

**Analysis Modules:**
- **Distribution**: Statistical analysis of particle properties
- **Time Course**: Temporal analysis visualization
- **Correlation**: Auto- and cross-correlation analysis
- **Colocalization**: Automated and manual colocalization

**Specialized Tools:**
- **Photobleaching**: Intensity correction
- **Tracking Visualization**: Enhanced particle display
- **Crops**: Particle crop analysis
- **Export**: Comprehensive data export

### Common Interface Elements
- **Parameter Controls**: Spinboxes, sliders, and combo boxes for adjusting analysis parameters
- **Matplotlib Canvases**: Interactive plots with zoom, pan, and navigation tools
- **Progress Dialogs**: Visual feedback during long-running operations
- **Export Buttons**: Save results in various formats throughout the interface

## Loading and Managing Images

### File Format Support

**Input Formats:**
- **LIF Files**: Native Leica format with complete metadata extraction
- **TIFF/OME-TIFF**: Standard microscopy formats with embedded metadata support
- **Multi-dimensional Arrays**: Automatic dimension mapping for arbitrary file structures

**Metadata Handling:**
- Automatic extraction of voxel sizes, time intervals, and channel information
- Manual entry prompts for missing critical metadata
- Validation and unit conversion (μm to nm, etc.)

### Loading Workflow

1. **Select Files**: Use "Open File" to browse for LIF or TIFF files
2. **Handle Missing Metadata**: If prompted, enter missing values:
   - Pixel size in nanometers
   - Time interval in seconds
   - Z-step size in nanometers
   These values are crucial for quantitative analysis

3. **Dimension Mapping** (if required): For non-standard file formats, map file dimensions to standard microscopy axes:
   - Choose from **TCZYX**, **TZYXC**, or **XYZCT** (Time, Channels, Z, Y, X permutations)
   - Select **"Singleton"** for missing axes to insert placeholder dimensions
   - Standard format: [T, Z, Y, X, C] (Time, Z-planes, Y-pixels, X-pixels, Channels)

4. **File Tree Navigation**: Select specific scenes from LIF files using the hierarchical tree view
5. **Metadata Review**: Confirm and review extracted metadata in the info panel

For technical details on supported formats, see [API Reference](api_reference.md#file-format-support).

## Display and Visualization

### Basic Display Controls
- **Channel Selection**: Click channel buttons to view individual channels
- **Merge Channels**: Use "Merge Channels" to combine multiple channels
- **Time Navigation**: Use the time slider or play button for frame navigation
- **Z-plane Control**: Use the vertical Z-slider to navigate through Z-stacks

### Intensity Scaling
Each channel has independent intensity controls:
- **Min/Max Percentiles**: Adjust contrast using percentile values
- **Gaussian Filtering**: Apply smoothing with high and low sigma values
- **Per-channel Settings**: Use the channel tabs for individual channel control

### Advanced Display Options
- **Background Removal**: Toggle to remove background using segmentation mask
- **Time Stamps**: Display time information on images
- **Colormaps**: Automatic ImageJ-style colormaps for each channel
- **Export Options**: Save current view as PNG or export videos

### Theme and Appearance
- **Dark/Light Theme**: Toggle switch in the top-left of Import tab
- **Color Schemes**: Consistent color schemes across all analysis modules
- **Font Sizing**: Optimized for different screen resolutions

## Segmentation

### Manual Segmentation
1. **Select Channel**: Choose the most appropriate channel for segmentation
2. **Start Segmentation**: Click "Manual Segmentation" button
3. **Draw Polygon**: Click points to define the region boundary
4. **Finish**: Click "Finish Segmentation" to complete the mask

### Watershed Segmentation
1. **Threshold Factor**: Adjust the sensitivity slider (0.1-2.0)
2. **Run Watershed**: Click to execute automatic segmentation
3. **Review Results**: Check the segmentation overlay
4. **Export Mask**: Save the binary mask as a TIFF file

### Maximum Projection Options
- **Enable Max Projection**: Use maximum projection to create a temporal projection of the image in all frames.
- **Status Indicator**: Visual feedback shows current projection state
- **Frame Selection**: Choose specific time points for segmentation

## Photobleaching Correction

Photobleaching correction compensates for the gradual loss of fluorescence intensity over time during imaging. The analysis fits decay models to intensity time courses and generates corrected image stacks.

### Mathematical Models

The photobleaching correction implements three mathematical models to characterize intensity decay:

#### 1. Single Exponential Decay

The most commonly used model for photobleaching follows first-order kinetics:

```
I(t) = I₀ × exp(-t/τ) + C
```

Where:
- **I(t)**: Intensity at time t
- **I₀**: Initial amplitude (intensity at t=0)
- **τ**: Decay time constant (characteristic bleaching time)
- **C**: Background/baseline intensity
- **t**: Time

**Decay Rate**: k = 1/τ (rate constant, units: 1/time)

#### 2. Double Exponential Decay

For complex systems with multiple fluorophore populations or bleaching mechanisms:

```
I(t) = A₁ × exp(-t/τ₁) + A₂ × exp(-t/τ₂) + C
```

Where:
- **A₁, A₂**: Amplitudes of fast and slow components
- **τ₁, τ₂**: Fast and slow decay time constants (τ₁ < τ₂)
- **C**: Baseline intensity

**Component Analysis**:
- Fast component: A₁/(A₁ + A₂) × 100% (percentage)
- Slow component: A₂/(A₁ + A₂) × 100% (percentage)

#### 3. Linear Decay

For cases where bleaching appears linear over the observation period:

```
I(t) = I₀ - k × t + C
```

Where:
- **I₀**: Initial intensity
- **k**: Linear decay rate (intensity units/time)
- **C**: Baseline intensity
- **t**: Time
- **b**: Is the intercept. **b** = **I₀** + **C**
### Correction Methods

#### Region Selection:

**Inside Cell**: Use pixels within the segmentation mask
- Measures photobleaching of cellular fluorescence
- Accounts for intracellular environment effects
- Most appropriate for cytoplasmic or membrane proteins

**Outside Cell**: Use background pixels outside the mask
- Measures background fluorescence decay
- Controls for autofluorescence bleaching
- Useful for background subtraction applications

**Circular Region**: Define a circular region in the center of the mask with adjustable radius
- Focuses on specific cellular regions
- Reduces spatial heterogeneity effects
- Radius parameter (pixels): Size of analysis region. (1–200 pixels, default: 30)

**Entire Image**: Use all pixels from the full field of view
-	Measures global fluorescence decay
-	Does not require any segmentation mask
-	Captures both specimen and background bleaching


#### Model Selection Guidelines:

- **Single Exponential**: Most fluorescent proteins, single fluorophore systems
- **Double Exponential**: Mixed populations, complex cellular environments
- **Linear**: Short time courses, minimal bleaching

### Workflow

1. **Prerequisites**: Ensure segmentation is complete
2. **Configure Parameters**: 
   - Select region mode (inside_cell/outside_cell/circular_region)
   - Set circular radius (1-200 pixels, default: 30)
   - Choose model type
3. **Exclude Points**: Set number of initial time points to ignore (0-200, default: 0)
4. **Run Correction**: Execute photobleaching analysis
5. **Review Results**: Examine fit quality and corrected data

### Results Interpretation

#### Left Plots: Fitted Decay Curves

**Displayed Parameters (Single Exponential)**:
- **τ (tau)**: Decay time constant in acquisition units
- **k**: Decay rate = 1/τ (units: 1/time)
- **I₀**: Initial amplitude
- **C**: Baseline intensity

**Displayed Parameters (Double Exponential)**:
- **τ₁**: Fast component time constant
- **τ₂**: Slow component time constant  
- **A₁**: Fast component amplitude
- **A₂**: Slow component amplitude
- **Fast %**: A₁/(A₁ + A₂) × 100
- **Slow %**: A₂/(A₁ + A₂) × 100
- **C**: Baseline intensity

**Displayed Parameters (Linear)**:
- **Slope**: Linear decay rate k (intensity/time)
- **Intercept**: Initial intensity I₀

#### Right Plots: Original vs. Corrected Comparison

**Original Data** (blue):
- Raw intensity time course
- Shows actual photobleaching decay
- Error bars: Standard error across pixels

**Corrected Data** (orange):
- Intensity after bleaching compensation
- Should show stable intensity over time
- Correction formula: I_corrected(t) = I_original(t) × [I₀ / I_fitted(t)]

#### Quality Metrics

**Good Correction Indicators**:
- **High R²**: >0.8 for exponential, >0.7 for linear
- **Stable corrected intensity**: Minimal drift in orange curve
- **Reasonable parameters**: τ values consistent with fluorophore properties
- **Error reduction**: Smaller error bars in corrected data

**Poor Correction Indicators**:
- **Low R²**: <0.5 suggests inappropriate model choice
- **Unrealistic parameters**: Extremely short or long τ values
- **Overcorrection**: Corrected intensity increases over time
- **High residuals**: Large deviations from fitted curve

### Practical Guidelines

#### Parameter Selection:
- **Exclude initial points**: Remove frames with focus drift or settling artifacts
- **Circular radius**: 
  - Small radius (10-20 px): Nuclear regions, specific organelles
  - Medium radius (20-40 px): General cellular analysis
  - Large radius (40+ px): Whole-cell averaging

#### Model Choice:
- Start with single exponential for most applications
- Use double exponential if single exponential fits poorly (R² < 0.7)
- Linear model only for very short acquisitions or minimal bleaching

#### Validation:
- Compare uncorrected vs. corrected time courses in subsequent analyses
- Verify that biological conclusions remain consistent
- Check that correction doesn't introduce artifacts in quantitative measurements

### Integration with Tracking

When "Photobleaching Corrected" is selected as the image source in the Tracking tab:
- All subsequent analyses use the corrected image stack
- Intensity measurements reflect bleaching-compensated values
- Time courses and correlations show corrected dynamics
- Export functions save corrected data when specified

**Note**: Photobleaching correction is applied globally to the entire image stack. Region-specific analysis (inside vs. outside cell) determines the correction parameters, but the resulting correction factor is applied uniformly across all pixels.

## Particle Tracking

The Particle Tracking tab provides comprehensive tools for detecting and linking particles across time frames. The analysis pipeline consists of spot detection, trajectory linking, and optional clustering analysis.

### Tracking Algorithms

MicroLive utilizes different computational approaches depending on your tracking mode selection:

#### 2D Projection Tracking
When **"Use 2D Projection for Tracking"** is enabled:
- **Detection**: Uses **trackpy** library for 2D spot detection on maximum Z-projections
- **Tracking**: Uses **trackpy** for trajectory linking across time frames
- **Advantages**: Faster processing, suitable for thin samples or when Z-information is not critical
- **Use cases**: When computational speed is prioritized

#### 3D Tracking
When **"Use 2D Projection for Tracking"** is disabled:
- **Detection**: Uses **big-fish** library for 3D spot detection across all Z-planes
- **Tracking**: Uses **trackpy** for trajectory linking in 3D space
- **Advantages**: More accurate localization, better separation of overlapping spots, cluster detection
- **Use cases**: Thick samples, nuclear proteins, or when precise 3D coordinates are needed

### Algorithm Selection and Performance

**Tracking Algorithm Comparison:**

| Feature | 2D Projection Mode | 3D Tracking Mode |
|---------|-------------------|-------------------|
| **Detection** | TrackPy (2D) | Big-FISH (3D) |
| **Linking** | TrackPy | TrackPy |
| **Speed** | Faster | Slower |
| **Accuracy** | Good for thin samples | Better for thick samples |
| **Cluster Detection** | No | Yes |
| **Memory Usage** | Lower | Higher |

**When to Use Each Mode:**

**2D Projection (Default)**:
- Thin samples (≤5 μm)
- Membrane proteins
- High-speed imaging requirements
- Limited computational resources

**3D Tracking**:
- Thick samples (>5 μm) 
- Nuclear proteins
- Clustered/aggregated proteins
- When precise Z-localization is critical

### Cluster Detection and Analysis

**3D tracking mode enables advanced cluster detection capabilities:**

#### Cluster vs. Spot Classification
- **Individual Spots**: Single, isolated fluorescent puncta
- **Clusters**: Groups of spots within a defined radius that may represent:
  - Protein complexes
  - Transcriptional factories
  - Stress granules
  - Other biomolecular condensates

#### Cluster Parameters
- **Cluster Radius (nm)**: Maximum distance between spots to be considered part of the same cluster
- **Max Cluster Size**: Maximum number of spots allowed per cluster (helps filter artifacts)
- **Separate Analysis**: Option to analyze clusters and individual spots independently

#### Additional Data from 3D Analysis
When using 3D tracking with cluster detection, you get:
- **Cluster size information**: Number of spots per cluster
- **3D coordinates**: Precise X, Y, Z positions for each spot
- **Cluster membership**: Which spots belong to which clusters
- **Enhanced intensity measurements**: More accurate due to 3D localization
- **Spatial relationships**: Better understanding of molecular organization

### Tracking Workflow

### Detection Parameters
**Spot Detection:**
- **Threshold**: Intensity threshold for particle detection
- **Spot Size**: Expected particle size in XY and Z dimensions
- **Cluster Handling**: Radius and maximum size for clustered particles

**Quality Control:**
- **Minimum Trajectory Length**: Filter short trajectories
- **Maximum Projection**: Use 2D projection for detection
- **Random Spots**: Generate control spots for validation

### Linking Parameters
- **Search Range**: Maximum distance for linking particles between frames
- **Memory**: Number of frames a particle can disappear and reappear
- **3D Coordinates**: Use Z-information for linking (when available)

### Workflow
1. **Source Selection**: Choose original or photobleaching-corrected image
2. **Threshold Setting**: Use histogram to set detection threshold
3. **Parameter Tuning**: Adjust spot size and linking parameters
4. **Single Frame Test**: Test detection on current frame
5. **All Frames Detection**: Run detection across all time points
6. **Trajectory Linking**: Connect detections into trajectories

### Tracking Parameters Reference

#### Detection Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Threshold** | Image-dependent | Auto (99th percentile) | Intensity cutoff for spot detection |
| **YX Spot Size** | 3-15 pixels | 5 | Expected spot diameter in XY |
| **Z Spot Size** | 1-10 pixels | 2 | Expected spot extent in Z |
| **Max Spots for Threshold** | 100-10000 | 3000 | Spots used for automatic threshold calculation |
| **Random Spots** | Boolean | False | Generate control trajectories (see Quality Control section) |

#### Clustering Parameters (3D Mode Only)

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Cluster Radius** | 100-2000 nm | 500 | Maximum distance for spot grouping |
| **Max Cluster Size** | 0-1000 | None (0) | Maximum spots per cluster |
| **Separate Analysis** | Boolean | False | Analyze clusters and spots independently |

#### Trajectory Linking

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Min Trajectory Length** | 1-200 frames | 20 | Filter for minimum track duration |
| **Max Search Range** | 1-50 pixels | 7 | Maximum distance between frames |
| **Memory** | 0-10 frames | 1 | Frames a particle can disappear |
| **Use 3D Coordinates** | Boolean | True | Include Z dimension in linking |

#### Display Parameters (Tracking Tab)

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Min Intensity Percentile** | 0-50% | 1.0% | Lower display threshold for tracking visualization |
| **Max Intensity Percentile** | 90-100% | 99.95% | Upper display threshold for tracking visualization |

For algorithm implementation details, see [API Reference](api_reference.md#tracking-pipeline-classes).

### Visualization Options
- **Overlay Display**: Show detected spots and trajectories
- **Color Coding**: Different colors for different particle properties
- **Display Controls**: Toggle trajectories, cluster sizes, particle IDs
- **Export Options**: Save tracking data, images, and videos

## Statistical Analysis

### Distribution Analysis

**Available Metrics:**

- **Spot Intensity (Background Subtracted)**: Uses the disk and doughnut method where mean background intensity is subtracted from mean spot intensity:
  ```
  I_spot = (1/s_spot²) × Σ I(x,y) in D - (1/(s_bg² - s_spot²)) × Σ I(x,y) in R
  ```
  Where D is the spot region (s_spot × s_spot) and R is the background annulus region.

- **Total Spot Intensity**: Sum of all pixel intensities within the spot region D:
  ```
  I_total = Σ I(x,y) for all (x,y) in D
  ```

- **PSF Amplitude**: The peak intensity (I₀) obtained from 2D Gaussian fitting:
  ```
  I_spot(x,y) = I_bg + I₀ × exp[-1/2 × ((x-x₀)²/σₓ² + (y-y₀)²/σᵧ²)]
  ```

- **PSF Sigma**: The standard deviation (σₓ, σᵧ) from the 2D Gaussian fit, representing the spot width in pixels.

- **Signal-to-Noise Ratio (SNR)**: Calculated as the difference between mean spot and background intensities divided by background standard deviation:
  ```
  SNR = (mean_intensity_spot - mean_intensity_background) / std_background
  ```

- **Spot Size (FWHM-based)**: Physical size measurements of detected particles calculated using Full Width at Half Maximum principles:

#### Full Width at Half Maximum (FWHM) Calculation

The spot size measurement combines two complementary approaches to accurately determine particle dimensions:

**1. PSF-Based FWHM (for well-isolated spots):**

When a 2D Gaussian PSF fit is successful, the spot size is derived from the fitted standard deviations:
```
FWHM = 2√(2ln2) × σₓᵧ ≈ 2.355 × σₓᵧ
```

**2. Cluster-Based Size (for clustered particles):**

For spots detected as clusters (when `cluster_size > 1`), the cluster size represents the **number of individual spots grouped together** within the specified clustering radius using the Big-FISH library.

#### Big-FISH Clustering Process:

The clustering analysis follows this workflow:

1. **Individual Spot Detection**: Big-FISH first detects individual spots using local maxima detection and thresholding
2. **Cluster Identification**: Spots within `cluster_radius_nm` of each other are grouped into clusters using the `detect_clusters` function
3. **Size Assignment**:
   - **Individual spots** (`cluster_size = 1`): Isolated spots not part of any cluster
   - **Clustered spots** (`cluster_size > 1`): The number indicates how many individual spots were detected within the cluster

**3. Hybrid Size Calculation:**

The final `spot_size` value reported in the tracking DataFrame uses a context-dependent approach:

- **Individual spots** (`cluster_size = 1`): Uses PSF-based FWHM if fitting succeeded, otherwise uses default detection aperture size
- **Failed fits**: Falls back `np.NAN`

#### Physical Interpretation

The FWHM measurement provides the **effective diameter** of the fluorescent signal, which represents:

- **Point sources**: The apparent size due to the microscope's Point Spread Function (PSF)
- **Extended objects**: The actual physical size convolved with the PSF
- **Clusters**: The spatial distribution of multiple nearby fluorophores

#### Conversion to Physical Units

Spot sizes are automatically converted to nanometers using the pixel calibration:
```
spot_size_nm = spot_size_pixels × voxel_yx_nm
```

This approach ensures accurate size measurements across different imaging conditions and particle types, from single molecules to protein complexes and cellular structures.

**Usage:**
1. Select "spot_size" from the field dropdown menu
2. Choose the channel for analysis
3. Set percentile ranges to filter outliers (default: 0-99.5%)
4. Click "Plot Histogram" to generate the distribution

### Time Course Analysis
**Data Types:**
- **Particle Counts**: Number of particles over time
- **Intensity Metrics**: Various intensity measurements over time
- **Quality Metrics**: SNR and other quality indicators over time

**Controls:**
- **Channel Selection**: Choose specific channels for analysis
- **Percentile Filtering**: Remove outliers from visualization
- **Export Options**: Save time course plots and data

### Correlation Analysis

The Correlation tab provides tools for analyzing temporal correlations in particle intensity time courses, implementing fluorescence correlation spectroscopy (FCS) methods for single-molecule dynamics analysis.

### Autocorrelation Function (ACF) Analysis

The autocorrelation analysis follows established protocols for extracting kinetic parameters from single-molecule fluorescence trajectories. The implementation is based on methods described in fluorescence correlation spectroscopy literature.

#### Mathematical Foundation

The autocorrelation function is calculated as:

```
G(τ) = ⟨δI(t)·δI(t+τ)⟩ / ⟨I(t)⟩²
```

Where:
- **I(t)**: Fluorescence intensity at time t
- **⟨·⟩**: Temporal average over the trajectory
- **δI(t) = I(t) - ⟨I(t)⟩**: Intensity fluctuations around the mean
- **τ**: Discrete time lag (frame intervals)

#### Data Processing Pipeline

**1. Trajectory Filtering:**
- Minimum trajectory length filter removes short, potentially spurious tracks
- Low signal-to-noise ratio trajectories are excluded
- Only trajectories with sufficient data coverage are retained

**2. Noise Reduction:**
- **G(0) Correction**: The autocorrelation at zero lag often contains shot noise from the microscope
- Linear extrapolation from subsequent time points (typically 10 frames) replaces the noisy G(0) value
- **Baseline Correction**: Mean correlation values are shifted to account for basal fluorescence levels

**3. Statistical Analysis:**
- **Bootstrapping**: Subsets of trajectories are resampled to generate multiple correlation realizations
- Standard error is computed from bootstrap distributions
- **Outlier Removal**: Trajectories with correlations deviating >4× median absolute deviation are excluded (~5-10% of data)

#### Fitting Methods

**Linear Fit:**
- Suitable for simple exponential decay processes
- Provides decorrelation time from linear regression of log-transformed data

**Exponential Fit:**
- Single exponential: `G(τ) = A·exp(-τ/τ_c) + C`
- Captures characteristic decay time τ_c directly
- Better for complex kinetic processes

#### Quality Controls

**1. Random Location Control:**
- Correlation calculated from random positions within the cell mask
- Should show no clear decorrelation pattern
- Validates that observed correlations arise from genuine molecular processes

**2. Simulation Validation:**
- Synthetic intensity data with known kinetic parameters
- Tests ability to recover input elongation and initiation rates
- Confirms preprocessing pipeline accuracy

#### Cross-Correlation Analysis

For dual-channel experiments, cross-correlation analysis reveals:
- **Co-localization dynamics**: Temporal correlation between channels
- **Lag time analysis**: Delays between molecular processes
- **Coupling efficiency**: Strength of inter-channel correlation

The peak lag time in cross-correlation indicates the temporal relationship between processes in different channels.

#### Best Practices

**Parameter Selection:**
- Set minimum trajectory length based on expected process timescales
- Adjust decorrelation threshold based on signal-to-noise characteristics
- Use appropriate fitting model (linear vs. exponential) based on underlying kinetics

**Data Quality:**
- Ensure adequate sampling rate relative to process timescales
- Maintain consistent imaging conditions across experiments
- Verify control measurements show expected behavior

**Statistical Considerations:**
- Include sufficient trajectories for robust statistics (typically >50-100)
- Report confidence intervals from bootstrap analysis
- Document outlier removal criteria and percentages

## Colocalization Analysis

### Automated Analysis
**ML Method:**
- **Machine Learning**: Use trained models for spot classification
- **Threshold Control**: Adjust ML confidence threshold (0.5-1.0)
- **Consistency**: Best for uniform spot morphologies

**Intensity Method:**
- **Signal-to-Noise**: Based on SNR thresholds
- **Flexibility**: Better for variable spot appearances
- **Threshold Range**: Typically 2-5 for SNR values

### Manual Verification
**Interface:**
- **Crop Grid**: Visual inspection of individual spots
- **Checkboxes**: Manual marking of colocalized spots
- **Populate Function**: Auto-fill based on algorithmic results
- **Statistics**: Real-time percentage calculations

**Workflow:**
1. **Run Automated Analysis**: Execute ML or intensity-based detection
2. **Review Results**: Examine colocalization matrix
3. **Manual Verification**: Switch to manual tab for detailed inspection
4. **Populate Results**: Load algorithmic results for manual review
5. **Individual Review**: Check/uncheck spots based on visual inspection
6. **Export Data**: Save manual verification results

### Quality Control
- **Visual Matrix**: Color-coded representation of colocalization
- **Percentage Metrics**: Quantitative colocalization measurements
- **Export Options**: Save analysis images and data tables
- **Documentation**: Include analysis parameters in exported metadata

## Data Export

### Export Types
**Images:**
- **PNG**: High-resolution images at 300 DPI
- **OME-TIFF**: Complete image stacks with metadata
- **Video**: MP4 or GIF animations

**Data:**
- **CSV Files**: Tracking data, colocalization results, correlation data
- **Metadata**: Complete parameter logs and analysis settings
- **User Comments**: Custom annotations and notes

### Batch Export
1. **Export Tab**: Navigate to the Export tab
2. **Select Items**: Check desired export items from the table
3. **Add Comments**: Include user comments and observations
4. **Choose Folder**: Select parent directory for export
5. **Execute Export**: All selected items exported to organized subfolder

### File Organization
Exported data is organized in structured folders:
- **Images**: All image exports (PNG, TIFF)
- **Data**: CSV files and numerical results
- **Metadata**: Parameter files and analysis logs
- **Comments**: User annotations and notes

### Custom Naming
- **Automatic Names**: Based on original filename and analysis type
- **Timestamp Integration**: Includes creation dates
- **Parameter Integration**: Incorporates key analysis parameters

## Best Practices

### Experimental Design
- **Adequate Sampling**: Ensure sufficient temporal resolution
- **Control Conditions**: Include appropriate negative and positive controls
- **Documentation**: Record acquisition parameters and experimental conditions

### Image Quality
- **Signal-to-Noise**: Optimize acquisition parameters for good SNR
- **Photobleaching**: Minimize photobleaching during acquisition
- **Stability**: Ensure stable imaging conditions throughout acquisition

### Analysis Workflow
- **Parameter Validation**: Test parameters on subset of data first
- **Quality Control**: Always perform visual inspection of automated results
- **Reproducibility**: Document all analysis parameters for reproducibility
- **Controls**: Use random spot generation for validation

### Data Management
- **Backup**: Keep backups of original data files
- **Version Control**: Track different analysis versions
- **Documentation**: Maintain detailed analysis logs
- **Organization**: Use consistent file naming and folder structures

### Performance Optimization
- **Memory Management**: Close unused files to free memory
- **Processing Time**: Use maximum projection for faster analysis when appropriate
- **Batch Processing**: Process multiple files with consistent parameters

### Troubleshooting Common Issues
- **Poor Detection**: Adjust threshold and spot size parameters
- **Broken Trajectories**: Increase search range and memory parameters
- **Memory Issues**: Reduce image size or close unused files
- **Performance**: Use 2D projection mode for faster processing

## Data Structures

Understanding the structure of output data is crucial for effective analysis and interpretation of results. MicroLive generates structured DataFrames containing comprehensive information about detected particles, their trajectories, and colocalization relationships.

### Particle Tracking DataFrame (`df_tracking`)

The particle tracking analysis generates a comprehensive pandas DataFrame that serves as the primary output containing all particle information across time. This DataFrame is accessible through the "Export DataFrame" button in the Tracking tab.

#### Core Spatial and Temporal Columns

| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `frame` | int | Time frame index (0-based) | frames |
| `x` | float | X coordinate of particle centroid | pixels |
| `y` | float | Y coordinate of particle centroid | pixels |
| `z` | float | Z coordinate of particle centroid | pixels |
| `particle` | int | Unique trajectory identifier | - |

#### Intensity Measurements (Per Channel)

For each imaging channel N (where N = 0, 1, 2, ...), the following intensity columns are automatically generated:

**Background-Subtracted Intensity:**
| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `spot_int_ch_N` | float | Integrated intensity using disk-doughnut method | counts |

**Total Intensity:**
| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `total_spot_int_ch_N` | float | Sum of all pixel values within spot region | counts |

**PSF Fitting Results:**
| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `psf_amplitude_ch_N` | float | Peak intensity from 2D Gaussian fit | counts |
| `psf_sigma_ch_N` | float | Standard deviation from 2D Gaussian fit | pixels |

**Quality Metrics:**
| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `snr_ch_N` | float | Signal-to-noise ratio | - |

#### Particle Properties and Clustering Information

| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `cluster_size` | int | Number of spots grouped in cluster (1 = individual spot) | spots |
| `is_cluster` | bool | Whether particle is part of a multi-spot cluster | - |
| `spot_type` | int | Channel identifier for analysis | - |
| `spot_id` | int | Unique spot identifier within each frame | - |

#### Cellular Localization Data (When Segmentation Available)

**Cell Assignment:**
| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `cell_id` | int | Unique cell identifier | - |
| `is_nuc` | bool | Whether spot is located in nucleus | - |
| `is_cell_fragmented` | int | Cell boundary quality (-1: no cell, 0: complete, 1: fragmented) | - |

**Spatial References:**
| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `nuc_loc_y` | float | Nuclear centroid Y coordinate | pixels |
| `nuc_loc_x` | float | Nuclear centroid X coordinate | pixels |
| `cyto_loc_y` | float | Cytoplasmic centroid Y coordinate | pixels |
| `cyto_loc_x` | float | Cytoplasmic centroid X coordinate | pixels |

**Cellular Areas:**
| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `nuc_area_px` | float | Nuclear area | pixels² |
| `cyto_area_px` | float | Cytoplasmic area (excluding nucleus) | pixels² |
| `cell_area_px` | float | Total cell area | pixels² |

#### Cellular Intensity Context (Per Channel)

For each channel N with segmentation data:

**Regional Intensities:**
| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `nuc_int_ch_N` | float | Average nuclear intensity | counts |
| `cyto_int_ch_N` | float | Average cytoplasmic intensity | counts |
| `pseudo_cyto_int_ch_N` | float | Intensity in dilated nuclear region | counts |

**Intensity Ratios:**
| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `nucleus_cytosol_intensity_ratio_ch_N` | float | Nuclear/cytoplasmic intensity ratio | - |
| `nucleus_pseudo_cytosol_intensity_ratio_ch_N` | float | Nuclear/pseudo-cytoplasm ratio | - |

#### Data Usage Examples

**Access specific particle trajectory:**
```python
# Get all data points for particle ID 42
particle_data = df_tracking[df_tracking['particle'] == 42]
```

**Extract intensity time course:**
```python
# Get channel 0 intensity over time for particle 42
particle_42 = df_tracking[df_tracking['particle'] == 42].sort_values('frame')
intensity_timecourse = particle_42['spot_int_ch_0'].values
time_points = particle_42['frame'].values
```

**Filter by cellular localization:**
```python
# Get only nuclear spots
nuclear_spots = df_tracking[df_tracking['is_nuc'] == True]

# Get spots in specific cell
cell_5_spots = df_tracking[df_tracking['cell_id'] == 5]
```

**Analyze trajectory properties:**
```python
# Calculate trajectory lengths
trajectory_lengths = df_tracking.groupby('particle').size()

# Get mean intensity per trajectory
mean_intensities = df_tracking.groupby('particle')['spot_int_ch_0'].mean()
```

### Colocalization Results DataFrame

The colocalization analysis generates summary statistics stored in `df_colocalization`, accessible through the "Export Data" button in the Colocalization tab.

#### Colocalization Summary Structure

| Column | Type | Description |
|--------|------|-------------|
| `file name` | str | Source file name (without extension) |
| `image name` | str | Scene/image identifier |
| `reference channel` | int | Channel index used as reference |
| `colocalize channel` | int | Channel index tested for colocalization |
| `number of spots reference` | int | Total spots detected in reference channel |
| `number of spots colocalize` | int | Number of spots classified as colocalized |
| `colocalization percentage` | float | Percentage of reference spots that colocalize |
| `threshold value` | float | Threshold used for classification |
| `method` | str | Analysis method ("ML", "Intensity", or "Manual") |

#### Manual Colocalization DataFrame

The manual colocalization validation generates a similar structure in `df_manual_colocalization`:

| Column | Type | Description |
|--------|------|-------------|
| `file name` | str | Source file identifier |
| `image name` | str | Scene name |
| `reference channel` | int | Reference channel index |
| `colocalize channel` | int | Target channel index |
| `number of spots reference` | int | Total spots manually reviewed |
| `number of spots colocalize` | int | Spots marked as colocalized by user |
| `colocalization percentage` | float | Manual validation percentage |
| `threshold value` | NaN | Not applicable for manual method |
| `method` | str | Always "Manual" |

### Data Export and Integration

#### CSV Export Format

All DataFrames are exported as CSV files with UTF-8 encoding:

- **Tracking data**: `tracking_[filename]_[imagename].csv`
- **Colocalization results**: `colocalization_[filename]_[imagename].csv`
- **Manual colocalization**: `manual_colocalization_[filename]_[imagename].csv`
