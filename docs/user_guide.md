# User Guide - MicroLive Application

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Loading and Managing Images](#loading-and-managing-images)
4. [Display and Visualization](#display-and-visualization)
5. [Image Registration](#image-registration)
6. [Segmentation](#segmentation)
7. [Photobleaching Correction](#photobleaching-correction)
8. [Particle Tracking](#particle-tracking)
9. [MSD Analysis](#msd-mean-squared-displacement-analysis)
10. [Statistical Analysis (Distribution & Time Course)](#statistical-analysis)
11. [Correlation Analysis](#correlation-analysis)
12. [Colocalization Analysis](#colocalization-analysis)
13. [Tracking Visualization](#tracking-visualization)
14. [Data Structures](#data-structures)
15. [Data Export](#data-export)

## Getting Started

### Quick Start Workflow

For users new to MicroLive, follow this basic workflow:

1. **Load Data**: Open LIF, CZI, or TIFF files via the Import tab
2. **Register** *(optional)*: Correct for sample drift in the Registration tab
3. **Segment Cells**: Define regions of interest in the Segmentation tab
4. **Correct Photobleaching** *(optional)*: Apply intensity correction before tracking
5. **Track Particles**: Detect and link spots in the Tracking tab
6. **Analyze Results**: Generate MSD, distributions, correlations, and colocalization
7. **Export Data**: Save results and visualizations via the Export tab

For detailed step-by-step instructions, see the [Complete Tutorial](tutorial.md).

### File Format Support

**Input Formats:**

- **LIF Files**: Native Leica format with complete metadata extraction
- **CZI Files**: Zeiss format with metadata extraction; automatic Apotome grid removal for raw structured-illumination data
- **TIFF/OME-TIFF**: Standard microscopy formats with embedded metadata support
- **Multi-dimensional Arrays**: Automatic dimension mapping for arbitrary file structures

**Metadata Handling:**

- Automatic extraction of voxel sizes, time intervals, and channel information
- Manual entry prompts for missing critical metadata
- Validation and unit conversion (μm to nm, etc.)

## Interface Overview

### Tab Organization

The application is organized into **12 main tabs** in the following order:

| # | Tab | Purpose |
|---|-----|---------|
| 1 | **Import** | Image loading, navigation, and display |
| 2 | **Registration** | XY drift correction and channel alignment |
| 3 | **Segmentation** | Cell/ROI boundary definition (Watershed, Cellpose, Manual, Import, Edit) |
| 4 | **Photobleaching** | Intensity decay correction |
| 5 | **Tracking** | Spot detection and trajectory linking |
| 6 | **MSD** | Mean squared displacement / diffusion analysis |
| 7 | **Distribution** | Histograms of particle properties |
| 8 | **Time Course** | Temporal dynamics plots |
| 9 | **Correlation** | Auto- and cross-correlation (ACF/CCF) |
| 10 | **Colocalization** | Automated spatial colocalization (Visual, Distance, Verify sub-tabs) |
| 11 | **Visualization** | Enhanced per-trajectory inspection |
| 12 | **Export** | Batch export of all results |

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

1. **Select Files**: Use "Open File" to browse for LIF, CZI, or TIFF files
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
- **Colormaps**: Automatic ImageJ-style colormaps for each channel (Green, Magenta, Yellow)
- **Merge Brightness**: Multiple channels are automatically scaled to 60% brightness when merged in the Import tab to prevent oversaturation. The Visualization tab provides a slider to adjust brightness (10-100%)
- **Interactive Zoom**: Click and drag a rectangle on the image to zoom in; double-click to reset view
- **Export Options**: Save current view as PNG or export videos

### Theme and Appearance

- **Dark/Light Theme**: Toggle switch in the top-left of Import tab
- **Color Schemes**: Consistent color schemes across all analysis modules
- **Font Sizing**: Optimized for different screen resolutions

## Image Registration

The Registration tab corrects for XY sample drift or stage movement between time frames, ensuring cells remain aligned throughout the acquisition. If your cells do not visibly drift, this step can be skipped.

### Registration Workflow

1. **Draw ROI** (recommended): Click and drag on the left image panel to select a rectangular region containing a stable, bright reference structure (e.g., cell body, fiducial)
   - Avoid regions with moving particles or bright puncta that appear and disappear
   - A cyan rectangle confirms the ROI; the status indicator turns green
   - If no ROI is drawn, the full image is used (a warning will appear)
2. **Select Mode**: Choose the transformation type from the Mode dropdown:

   | Mode | Degrees of Freedom | When to Use |
   |------|--------------------|-------------|
   | `RIGID_BODY` (default) | Rotation + translation | Most live-cell experiments |
   | `TRANSLATION` | X/Y shift only | Minimal drift, no rotation |
   | `SCALED_ROTATION` | Rotation + uniform scale + translation | Optical zoom drift |
   | `AFFINE` | Full affine (shear, scale, rotation) | Complex distortions |

3. **Run Registration**: Click **"▶ Perform Registration"** (green button)
   - A progress bar shows frame-by-frame completion
   - The right panel updates to show the registered image when complete
4. **Verify**: Use the time slider and play button to confirm alignment
5. **Undo if Needed**: Click **"✕ Remove"** to discard and revert to the original image

### Integration with Other Tabs

Once registration is applied, all downstream tabs (Segmentation, Tracking, Photobleaching, etc.) automatically use the registered image stack. Removing registration reverts the entire pipeline to the original data.

### Technical Notes

- Registration is computed against **frame 0** as the reference
- The ROI is used only for computing the transformation; the correction is applied to the full image
- Pixels outside the valid registration area are set to zero (visible as a black border)
- Registration is not saved to the original file; it is applied in-memory only

---

## Segmentation

### Manual Segmentation

Manual segmentation uses a polygon-based drawing workflow where you click to place vertices that define cell boundaries.

#### Workflow

1. **Select Channel**: Choose the most appropriate channel for segmentation
2. **Navigate to Manual Tab**: Select the "Manual" sub-tab in the Segmentation tab
3. **Place Vertices**: Click on the image to place polygon vertices — each click adds a point connected by a line to the previous point
4. **Close the Polygon**: Either **double-click** anywhere (with at least 3 points) or **click near the first point** (within 15 pixels) to close the polygon and create the mask
5. **Add More Cells**: Continue clicking to define additional cell regions — each closed polygon is assigned a unique label
6. **Finish**: Click "Finish Segmentation" to finalize all drawn masks

> **Tip**: The polygon auto-closes when you click within 15 pixels of the first point, making it easy to complete the outline precisely.

### Mask Editing (Edit Tab)

After any segmentation (Watershed, Cellpose, Manual, or Imported), you can refine individual masks using the **Edit** sub-tab. This provides two specialized tools for correcting segmentation errors.

#### Selecting a Mask to Edit

1. Open the **Edit** sub-tab in the Segmentation panel
2. Use the **Mask Selector** dropdown to choose which segmentation result to edit (e.g., "Watershed", "Cellpose Cyto", "Cellpose Nuc")
3. The selected mask is displayed with colored overlays and white contours

#### Eraser Tool

The **Eraser** removes pixels from the mask by painting with a circular brush.

- **Activate**: Click the "🖌️ Eraser" button (active by default)
- **Use**: Click and drag on the image to erase pixels (sets them to background)
- **Brush Size**: Fixed 10-pixel circular brush
- **Undo**: Use the "↩ Undo" button to revert the last eraser stroke

#### Knife Tool (Cut/Split)

The **Knife** tool splits merged cells by drawing a multi-point cut line.

- **Activate**: Click the "🔪 Knife" button
- **Place Cut Points**: Click on the image to add cut points — red dots appear at each point, connected by red lines
- **Execute Cut**: **Double-click** to finalize and execute the cut along all placed points
- **Border Extension**: If an endpoint is within 15 pixels of the image border, the cut line automatically extends to the border edge for clean splits
- **Result**: After cutting, only the **largest** piece is kept; smaller fragments are automatically removed
- **Auto-Save**: The mask is automatically saved after each cut — no need to press "Apply & Save" manually

##### Knife Tool Workflow Example

1. Select a mask containing merged (touching) cells
2. Switch to the Knife tool
3. Click 2-4 points along the boundary between the merged cells
4. Double-click to execute the cut
5. The status bar shows how many pixels were kept vs. removed (e.g., "✂️ Kept largest (2,450 px), removed 1,230 px")

#### Saving Edits

- **Apply & Save**: Click to save all eraser edits to the underlying mask
- Knife tool cuts are saved automatically after execution
- All edits support **Undo** for easy correction

### Watershed Segmentation

The watershed algorithm automatically identifies cell boundaries based on intensity gradients. The algorithm uses adaptive histogram equalization (CLAHE) for local contrast enhancement, followed by thresholding and watershed-based region separation.

#### Workflow

1. **Threshold Factor**: Adjust the sensitivity slider (0.1-2.0)
2. **Run Watershed**: Click to execute automatic segmentation
3. **Review Results**: Check the segmentation overlay
4. **Export Mask**: Save the binary mask as a TIFF file

#### Segmentation Parameters Reference

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Threshold Factor** | 0.1-2.0 | 1.0 | Controls segmentation sensitivity. Lower values detect dimmer regions; higher values require brighter signals |
| **Expected Radius** | 10-500 px | 50 | Approximate cell radius used for smoothing and marker separation |
| **Min Object Size** | 100-50000 px² | 500 | Minimum area for valid objects; smaller regions are removed |
| **Separation Size** | 1-20 px | 3 | Erosion radius to separate touching cells when multiple objects detected |
| **Threshold Method** | otsu, li | otsu | Algorithm for automatic threshold calculation |

#### Parameter Selection Guidelines

**Threshold Factor:**

- **Low values (0.3-0.7)**: More sensitive detection, includes dimmer regions
  - Use for: Low-contrast images, dim cells, detecting cell periphery
  - Risk: May include background noise as cell area
- **Medium values (0.8-1.2)**: Balanced detection (recommended starting point)
  - Use for: Standard fluorescence images with moderate contrast
- **High values (1.3-2.0)**: More stringent, only brightest regions
  - Use for: High-background images, selecting only central bright regions
  - Risk: May exclude legitimate cell periphery

**Expected Radius:**

- Measure typical cell radius in pixels before segmentation
- Underestimating leads to over-segmentation (one cell split into many)
- Overestimating leads to under-segmentation (multiple cells merged)
- For U-2 OS cells at 63× with 130 nm pixels: typically 30-80 px
- For smaller cells (e.g., lymphocytes): typically 15-30 px

**Min Object Size:**

- Set based on smallest expected cell area
- Calculate as: π × (minimum_cell_radius)²
- Common values: 500-2000 px² for typical mammalian cells

### Maximum Projection Options

- **Enable Max Projection**: Use maximum projection to create a temporal projection of the image in all frames.
- **Status Indicator**: Visual feedback shows current projection state
- **Frame Selection**: Choose specific time points for segmentation

### Cellpose Segmentation

Cellpose provides deep learning-based cell segmentation that can automatically identify both cytosol and nucleus regions.

#### Workflow

1. **Select Model**: Choose the appropriate Cellpose model:
   - **cyto3** (default): Best for cytoplasm segmentation
   - **nuclei**: Optimized for nuclear segmentation
2. **Set Diameter**: Enter the expected cell/nucleus diameter in pixels
3. **Run Segmentation**: Click "Segment Cytosol" or "Segment Nucleus"
4. **Adjust Size**: Use the Size Adjust slider to expand or shrink masks as needed

#### Cellpose Parameters Reference

| Parameter            | Component | Range              | Default | Description                         |
|----------------------|-----------|--------------------| --------|-------------------------------------|
| **Model**            | Cytosol   | cyto3, cyto2, cyto | cyto3   | Cytoplasm segmentation model        |
| **Model**            | Nucleus   | nuclei, cyto3, etc.| nuclei  | Nuclear segmentation model          |
| **Diameter**         | Both      | 0-1000 px          | 150/60  | Expected cell/nucleus diameter      |
| **Optimize Parameters** | Both   | Boolean            | False   | Auto-optimize diameter              |
| **Size Adjust**      | Both      | -40 to +40 px      | 0       | Expand (+) or shrink (-) masks      |

#### Size Adjustment Slider

Each segmentation component (Cytosol and Nucleus) has an independent **Size Adjust** slider:

- **Center (0)**: Original mask size as detected by Cellpose
- **Positive values (+1 to +40)**: Expand the mask using Voronoi-like growth (safeguarded against exceeding image boundaries)
- **Negative values (-1 to -40)**: Shrink the mask using morphological erosion (safeguarded to prevent masks from disappearing below 10 pixels)

This allows you to independently adjust cytosol and nucleus masks. For example, you can expand the cytosol mask by 10 pixels to capture peripheral signals while keeping the nucleus mask at its original size.

#### Improve Segmentation Options

- **Remove cells touching border**: Filter out partial cells at image edges
- **Remove unpaired cells**: When both cytosol and nucleus are segmented, remove cells without matching nucleus (or vice versa)
- **Keep only center cell**: Filter masks to keep only the single cell whose centroid is closest to the image center

## Photobleaching Correction

Photobleaching correction compensates for the gradual loss of fluorescence intensity over time during imaging. The analysis fits decay models to intensity time courses and generates corrected image stacks.

### Mathematical Models

The photobleaching correction implements a single exponential decay model:

#### 1. Single Exponential Decay

```text
I(t) = I₀ × exp(-τ t) + C
```

Where:

- **I(t)**: Intensity at time t
- **I₀**: Initial amplitude (intensity at t=0)
- **τ**: Decay time constant (characteristic bleaching time,units: 1/time)
- **C**: Background/baseline intensity
- **t**: Time

### Correction Methods

#### Region Selection

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

- Measures global fluorescence decay
- Does not require any segmentation mask
- Captures both specimen and background bleaching

#### Model Selection Guidelines

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

#### Parameter Selection

- **Exclude initial points**: Remove frames with focus drift or settling artifacts
- **Circular radius**:
  - Small radius (10-20 px): Nuclear regions, specific organelles
  - Medium radius (20-40 px): General cellular analysis
  - Large radius (40+ px): Whole-cell averaging

#### Model Choice

- Start with single exponential for most applications
- Use double exponential if single exponential fits poorly (R² < 0.7)
- Linear model only for very short acquisitions or minimal bleaching

#### Validation

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
| **Cluster Detection** | Sigma-based | Geometric (DBSCAN) |
| **Memory Usage** | Lower | Higher |

**When to Use Each Mode:**

**2D Projection (Default)**:

- Thin samples (≤5 μm)
- High-speed imaging requirements
- Limited computational resources

**3D Tracking**:

- Thick samples (>5 μm)
- When precise Z-localization is critical

### Cluster Detection and Analysis

MicroLive provides cluster detection in both 2D and 3D tracking modes using different approaches optimized for each modality.

#### Cluster Classification Methods

**2D Projection Mode: Sigma-Based Classification**

When using 2D tracking, clusters are identified based on spot **spatial width (sigma)**, not intensity:

- **Physical basis**: Multiple overlapping mRNAs/spots that are too close to be resolved individually create a wider Gaussian profile than a single diffraction-limited spot
- **Method**: TrackPy measures each spot's spatial extent (`size` column)
- **Reference calculation**: The median of the lower 50% of measured sizes establishes the expected sigma for a single spot
- **Classification threshold**: Spots with `size > 1.5× reference_size` are classified as clusters
- **Cluster size estimation**: `cluster_size = round((size / reference_size)²)` based on the area ratio

This approach is self-calibrating and does not rely on intensity, which can vary due to expression levels.

**3D Tracking Mode: Geometric (DBSCAN) Classification**

When using 3D tracking with Big-FISH, clusters are identified based on **spatial proximity**:

- **Method**: DBSCAN clustering via `detection.detect_clusters()`
- **Cluster Radius (nm)**: Maximum distance between spots to be considered part of the same cluster
- **Minimum Spots**: Minimum number of nearby spots required to form a cluster (default: 2)
- **Cluster size**: The actual count of individual spots detected within the cluster region

This approach is appropriate for 3D data where individual spots can be spatially resolved.

#### Cluster vs. Spot Classification

- **Individual Spots**: Single, isolated fluorescent puncta (`cluster_size = 1`)
- **Clusters**: Groups of spots that may represent:
  - Multiple mRNAs at a transcription site
  - Protein complexes
  - Stress granules
  - Other biomolecular condensates

#### Cluster Parameters

- **Cluster Radius (nm)**: Maximum distance between spots to be considered part of the same cluster (3D mode only)
- **Max Cluster Size**: Maximum number of spots allowed per cluster (helps filter artifacts)
- **Separate Analysis**: Option to analyze clusters and individual spots independently

#### Additional Data from Cluster Analysis

When cluster detection is enabled, you get:

- **Cluster size information**: Number of spots per cluster
- **3D coordinates** (3D mode): Precise X, Y, Z positions for each spot
- **Cluster membership**: Which spots belong to which clusters
- **Enhanced intensity measurements**: More accurate due to proper cluster handling

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

### Automated Threshold Detection

MicroLive provides automated threshold detection using a hybrid approach that combines methods from Big-FISH and TrueSpot. The primary method generates a curve of spot counts versus threshold values and identifies the transition point where the curve changes from a steep decrease to a plateau. For images lacking a distinct transition, a fallback method analyzes the variability in the derivative of the spot count curve. Users can click the **"Auto"** button to calculate and apply the optimal threshold, with independent storage per channel for multi-channel experiments.

### Fixed Threshold Mode

By default, the detection threshold is normalized per-frame using percentile mapping, which adapts to each frame's intensity range. This works well when signal intensity is relatively constant across time. However, for experiments where signal genuinely decreases over time (e.g., after inhibitor treatment), this adaptive normalization can cause false positives in later frames by boosting sensitivity as intensity drops.

The **"Fixed"** button (next to "Auto") enables fixed threshold mode. When active, the threshold is computed from **frame 0 only** and applied unchanged to all subsequent frames. This ensures that as signal decreases, fewer spots are detected — matching the expected biological behavior.

**When to use Fixed Threshold:**

- Inhibitor treatment experiments where spots decrease over time
- Drug washout or degradation experiments
- Any time-lapse where signal is expected to decrease monotonically

**When to use default (adaptive) mode:**

- Constant-signal experiments with minor intensity fluctuations
- Photobleaching-corrected data (intensity already normalized)

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
| **Fixed Threshold** | Boolean | False | Use frame-0 threshold for all frames (for decreasing-signal experiments) |

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

```text
  I_spot = (1/s_spot²) × Σ I(x,y) in D - (1/(s_bg² - s_spot²)) × Σ I(x,y) in R
```

  Where D is the spot region (s_spot × s_spot) and R is the background annulus region.

- **Total Spot Intensity**: Sum of all pixel intensities within the spot region D:

```text
  I_total = Σ I(x,y) for all (x,y) in D
```

- **PSF Amplitude**: The peak intensity (I₀) obtained from 2D Gaussian fitting:

```text
  I_spot(x,y) = I_bg + I₀ × exp[-1/2 × ((x-x₀)²/σₓ² + (y-y₀)²/σᵧ²)]
```

- **PSF Sigma**: The standard deviation (σₓ, σᵧ) from the 2D Gaussian fit, representing the spot width in pixels.

- **Signal-to-Noise Ratio (SNR)**: MicroLive supports two methods for calculating SNR, selectable via the `snr_method` parameter:

  **Peak Method (Default)** — Standard SNR definition using maximum pixel value as signal:

```text
  SNR = (max_intensity_spot - mean_intensity_background) / std_background
```

  This is the standard definition of SNR in microscopy. It measures the contrast between the brightest pixel in the spot (the peak) and the background noise level. Recommended for most applications.

  **Disk-Doughnut Method** — Uses mean disk intensity instead of peak value:

```text
  SNR = (mean_intensity_spot - mean_intensity_background) / std_background
```

  This alternative method is more robust when data is very noisy or spots are dim, where the maximum pixel value may be unreliable due to noise spikes. Use this method when working with low-SNR imaging conditions.

- **Spot Size (FWHM-based)**: Physical size measurements of detected particles calculated using Full Width at Half Maximum principles:

#### Full Width at Half Maximum (FWHM) Calculation

The spot size measurement combines two complementary approaches to accurately determine particle dimensions:

**1. PSF-Based FWHM (for well-isolated spots):**

When a 2D Gaussian PSF fit is successful, the spot size is derived from the fitted standard deviations:

```text
FWHM = 2√(2ln2) × σₓᵧ ≈ 2.355 × σₓᵧ
```

**2. Cluster-Based Size (for clustered particles):**

For spots detected as clusters (when `cluster_size > 1`), the method depends on the tracking mode:

**2D Mode (TrackPy - Sigma-Based):**

The cluster size is estimated from the spot's spatial width compared to a reference single-spot width:

```text
cluster_size = round((measured_size / reference_size)²)
```

Where `reference_size` is the median of the lower 50% of measured spot sizes (self-calibrating).

**3D Mode (Big-FISH - Geometric):**

The cluster size represents the **actual count of individual spots** detected within the specified clustering radius using DBSCAN clustering.

#### Cluster Detection Workflows

**2D Sigma-Based Process (TrackPy):**

1. **Spot Detection**: TrackPy detects spots and measures their spatial extent (`size` column)
2. **Reference Calculation**: Median of lower 50% of sizes establishes single-spot reference
3. **Classification**: Spots with `size > 1.5× reference_size` are classified as clusters
4. **Size Estimation**: `cluster_size = round((size / reference_size)²)`

**3D Geometric Process (Big-FISH):**

1. **Individual Spot Detection**: Big-FISH first detects individual spots using local maxima detection and thresholding
2. **Cluster Identification**: Spots within `cluster_radius_nm` of each other are grouped into clusters using the `detect_clusters` function (DBSCAN)
3. **Size Assignment**:
   - **Individual spots** (`cluster_size = 1`): Isolated spots not part of any cluster
   - **Clustered spots** (`cluster_size > 1`): The number indicates how many individual spots were detected within the cluster

**3. PSF Size Calculation:**

The `psf_sigma_ch_N` column in the tracking DataFrame contains the spot size from PSF fitting:

- **Individual spots** (`cluster_size = 1`): Uses PSF-based sigma (σ) if fitting succeeded
- **Failed fits**: Contains `NaN` values

#### Physical Interpretation

The FWHM measurement provides the **effective diameter** of the fluorescent signal, which represents:

- **Point sources**: The apparent size due to the microscope's Point Spread Function (PSF)
- **Extended objects**: The actual physical size convolved with the PSF
- **Clusters**: The spatial distribution of multiple nearby fluorophores

#### Conversion to Physical Units

Spot sizes are automatically converted to nanometers using the pixel calibration:

```text
spot_size_nm = spot_size_pixels × voxel_yx_nm
```

This approach ensures accurate size measurements across different imaging conditions and particle types, from single molecules to protein complexes and cellular structures.

**Usage:**

1. Select "psf_sigma" from the field dropdown menu (this provides the PSF width used to calculate spot size)
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

---

## Correlation Analysis

The Correlation tab provides tools for analyzing temporal correlations in particle intensity time courses, implementing fluorescence correlation spectroscopy (FCS) methods for single-molecule dynamics analysis.

### Autocorrelation Function (ACF) Analysis

The autocorrelation analysis follows established protocols for extracting kinetic parameters from single-molecule fluorescence trajectories. The implementation is based on methods described in fluorescence correlation spectroscopy literature.

#### Mathematical Foundation

The autocorrelation function is calculated as:

```text
G(τ) = ⟨δI(t)·δI(t+τ)⟩ / ⟨I(t)⟩²
```

Where:

- **I(t)**: Fluorescence intensity at time t
- **⟨·⟩**: Temporal average over the trajectory
- **δI(t) = I(t) - ⟨I(t)⟩**: Intensity fluctuations around the mean
- **τ**: Discrete time lag (frame intervals)

#### Correlation Parameters Reference

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Min Trajectory Length** | 10-500 frames | 25 | Minimum number of frames for trajectory inclusion |
| **Max Lag Time** | 10-500 frames | 100 | Maximum time lag (τ) for correlation calculation |
| **G(0) Points** | 2-20 points | 10 | Number of points used for G(0) extrapolation |
| **Bootstrap Iterations** | 50-1000 | 100 | Number of resamples for error estimation |
| **Outlier Threshold** | 2-6 MAD | 4 | Median absolute deviation threshold for outlier removal |
| **Fit Method** | linear, exponential | exponential | Model for decorrelation time extraction |

#### Parameter Selection Guidelines

**Minimum Trajectory Length:**

- Set based on expected decorrelation time (τ_c)
- Rule of thumb: min_length ≥ 2 × expected τ_c (in frames)
- For translation imaging (τ_c ~ 100-400 s at 1 fps): use 50-200 frames
- For fast processes (τ_c ~ 10-50 s): use 20-100 frames
- Too short: Noisy correlations, unreliable fits
- Too long: May exclude many valid trajectories

**Max Lag Time:**

- Should extend beyond expected decorrelation time
- Typically set to 2-3 × expected τ_c
- For translation (gene length ~1900 codons, 5 aa/s): τ_c ≈ 380 s, use max_lag = 500+ frames at 1 fps
- Very long lag times increase noise at high τ

**Fitting Method Selection:**

- **Exponential fit**: Recommended for most biological processes
  - More accurate for long decorrelation times
  - Provides direct τ_c estimate and amplitude
- **Linear fit**: Use when:
  - Decorrelation is very fast relative to imaging rate
  - Log-linear relationship is clearly visible in log(G) vs τ plot
  - Quick estimation is needed

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

#### Extracting Biological Parameters

From the autocorrelation analysis, kinetic parameters can be estimated using the following relationships (Larson et al., 2011):

**Elongation Rate (k_e):**

```text
k_e ≈ L / τ_c
```

Where L is the gene length (in amino acids or codons) and τ_c is the decorrelation time.

**Initiation Rate (k_i):**

```text
k_i ≈ 1 / (G(0) × τ_c)
```

Where G(0) is the autocorrelation amplitude at zero lag.

**Example Calculation:**

- Gene length: 1,910 codons
- Measured τ_c: 365 seconds
- Measured G(0): 0.07
- Elongation rate: k_e = 1910/365 ≈ 5.2 aa/s
- Initiation rate: k_i = 1/(0.07 × 365) ≈ 0.039 s⁻¹

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

## MSD (Mean Squared Displacement) Analysis

MSD analysis quantifies particle mobility by measuring how far particles move over time, enabling extraction of diffusion coefficients and characterization of motion types (e.g., Brownian diffusion, confined motion, directed transport).

### Mathematical Foundation

The Mean Squared Displacement is defined as:

```text
MSD(τ) = ⟨|r(t + τ) - r(t)|²⟩
```

Where:

- **r(t)**: Position vector at time t
- **τ**: Time lag (lag time)
- **⟨·⟩**: Ensemble and time average over all particles and time points

For Brownian diffusion, MSD relates to the diffusion coefficient D:

**2D Diffusion:**

```text
MSD(τ) = 4Dτ
```

**3D Diffusion:**

```text
MSD(τ) = 6Dτ
```

### MSD Tab Overview

The MSD tab provides tools for calculating diffusion coefficients from tracked particle trajectories. The analysis requires completed particle tracking with linked trajectories.

#### Workflow

1. **Complete Tracking**: Run particle detection and trajectory linking in the Tracking tab
2. **Navigate to MSD Tab**: Open the MSD analysis tab
3. **Set Parameters**: Configure fit points and verify mode (2D/3D auto-detected)
4. **Calculate MSD**: Click "Calculate MSD" to run the analysis
5. **Review Results**: Examine the MSD plot and diffusion coefficient
6. **Export Data**: Save the MSD curve and per-trajectory data

### MSD Parameters Reference

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Fit Points** | 2 to available tracked lags | 5 | Number of initial lag points used for linear fitting |
| **Mode** | 2D, 3D | Auto-detect | Dimensionality for diffusion calculation (D = slope/4 for 2D, slope/6 for 3D) |
| **Max Lag Time** | 10-1000 frames | 100 | Maximum lag time for MSD calculation |
| **Remove Drift** | Boolean | False | Subtract ensemble drift before MSD calculation |
| **Log-Log Scale** | Boolean | False | Display MSD plot on logarithmic axes |

### Parameter Selection Guidelines

**Fit Points:**

- Determines the number of lag points used for extracting diffusion coefficient
- Only the initial linear regime should be used for fitting
- Start with the first 5 lag points and inspect whether that region is linear
- For confined motion, never include the later MSD plateau in a free-diffusion fit
- For directed motion: linear fitting may not be appropriate
- **Rule of thumb**: Fit approximately the first 10% of the movie/trajectory,
  while retaining enough lag points to estimate a line

**Mode Selection (Auto-detected):**

- **2D Mode**: Used when tracking on maximum Z-projections
  - Formula: D = slope / 4
  - Appropriate for: Membrane proteins, 2D cell cultures
- **3D Mode**: Used when Z-coordinates are available in tracking data
  - Formula: D = slope / 6
  - Appropriate for: Cytoplasmic proteins, thick samples
  - Requires valid, scene-specific XY and Z voxel sizes

**Calibration:**

- Coordinates enter the MSD calculation in pixels
- The Import-tab XY calibration is converted from nm/px to µm/px
- The movie interval is interpreted as seconds per frame
- The MSD panel displays XY, Δt, and the fitted lag-time range used for every result
- Calculation stops if a required calibration is missing or non-positive

**Remove Drift:**

- Enable when systematic cellular/stage drift is present
- Calculates ensemble mean displacement and subtracts from individual trajectories
- Important for slow-moving particles where drift dominates

### Results Interpretation

#### Diffusion Coefficient

The primary output is the pair-weighted ensemble diffusion coefficient D. The
separate per-track row reports the mean ± standard deviation of valid positive
single-trajectory fits; it is not the uncertainty of the ensemble coefficient.

| D (μm²/s) | Typical Interpretation |
|-----------|----------------------|
| 0.001-0.01 | Very slow/confined motion (e.g., chromatin-bound proteins) |
| 0.01-0.1 | Slow diffusion (e.g., large complexes, membrane proteins) |
| 0.1-1.0 | Moderate diffusion (e.g., cytoplasmic mRNPs) |
| 1.0-10.0 | Fast diffusion (e.g., small cytoplasmic proteins) |

#### Quality Metrics

**R² (Linear Fit):**

- **R² > 0.95**: Excellent linear fit, reliable D estimate
- **R² = 0.80-0.95**: Good fit, reasonable D estimate
- **R² < 0.80**: Poor fit, may indicate:
  - Non-Brownian motion (confined, directed)
  - Insufficient trajectory data
  - Too many fit points beyond linear regime

#### MSD Plot Interpretation

**Log-Log Scale Analysis:**

The slope of MSD vs. τ on log-log scale indicates motion type:

| Slope (α) | Motion Type | Interpretation |
|-----------|-------------|----------------|
| α ≈ 1.0 | Normal (Brownian) diffusion | Free diffusion, unconfined |
| α < 1.0 | Subdiffusion (anomalous) | Confined motion, molecular crowding |
| α > 1.0 | Superdiffusion | Directed motion, active transport |
| α ≈ 0 (plateau) | Confined diffusion | Particle trapped in domain |

### Per-Cell MSD Analysis

When segmentation is available, MSD is calculated separately for each cell:

- Each cell's trajectories are pooled for ensemble averaging
- Results show per-cell ensemble diffusion coefficients and linear-fit R²
- Color-coded curves differentiate cells in the plot
- A minimum of 10 particles per cell is required for display

### Export Options

**Export DataFrame:**

- CSV file containing MSD values for each trajectory
- Columns: time (lag), MSD per trajectory, cell assignment
- Enables custom downstream analysis

**Export Plot:**

- High-resolution PNG image of MSD curves
- Includes fit line and diffusion coefficient annotation

### Troubleshooting

**"Not enough data for MSD calculation":**

- Ensure minimum 20 frames per trajectory (MIN_FRAMES_MSD = 20)
- Ensure minimum 10 particles total (MIN_PARTICLES_MSD = 10)
- Increase "Memory" parameter in tracking to reduce trajectory breaks

**Very low or very high D values:**

- Verify pixel size calibration (voxel_yx_nm)
- Verify time interval calibration (step_size_in_sec)
- Check for tracking errors (false linking across cells)

**Poor R² fit:**

- Reduce number of fit points
- Check for non-Brownian motion (use log-log plot)
- Consider if particles are confined or actively transported

## Colocalization Analysis

MicroLive provides comprehensive colocalization analysis tools to quantify spatial relationships between fluorescent signals in different channels. The analysis combines automated detection methods with manual verification capabilities for robust and accurate results.

### Overview

Colocalization analysis determines whether particles detected in one channel (reference channel) are also present in another channel (colocalization channel).

### Analysis Workflow

#### 1. Prerequisites

- **Completed particle tracking**: Spots must be detected and tracked in at least one channel
- **Multi-channel data**: Minimum of 2 fluorescence channels required
- **Channel assignment**: Reference channel is automatically set to the tracking channel

#### 2. Automated Detection Methods

**Machine Learning (ML) Method:**

- **Technology**: Convolutional Neural Network (CNN) trained on manually-annotated image crops
- **Training data**: Combination of real experimental data and synthetic augmented datasets
- **Input**: Image crops centered on detected particles from both channels
- **Output**: Confidence score (0.5-1.0) indicating colocalization probability

**Intensity-Based Method:**

- **Technology**: Signal-to-noise ratio (SNR) thresholding
- **Calculation**: `SNR = (mean_spot - mean_background) / std_background`
- **Threshold range**: Typically 2-5 (adjustable 0-10)

#### 3. Parameter Configuration

**Channel Selection:**

- **Reference Channel**: Source of particle coordinates (auto-set from tracking)
- **Colocalization Channel**: Target channel for colocalization assessment
- **Validation**: Prevents selection of identical channels

**Method-Specific Parameters:**

*ML Method:*

- **ML Threshold**: Confidence cutoff (0.5-1.0, default: 0.50)
  - Higher values: More stringent, fewer false positives
  - Lower values: More permissive, captures weaker colocalizations

*Intensity Method:*

- **SNR Threshold**: Signal strength requirement (0-10, default: 3.0)
  - Threshold = 2: Permissive, includes dim spots
  - Threshold = 5: Stringent, only bright spots

**Display Options:**

- **Crop Columns**: Grid layout for visualization (1-100, default: 50)
- **Export Format**: PNG images, CSV data tables

#### 4. Results Interpretation

**Quantitative Metrics:**

- **Total Reference Spots**: Number of particles in reference channel
- **Colocalized Spots**: Number meeting colocalization criteria
- **Colocalization Percentage**: `(Colocalized / Total Reference) × 100`
- **Threshold Value**: Applied detection threshold
- **Method Used**: ML or Intensity-based classification

**Visual Output:**

- **Paired Crop Display**: Side-by-side comparison of both channels
- **Color Coding**: Flagged spots highlighted with colored borders
- **Grid Layout**: Organized display for systematic review
- **Summary Statistics**: Real-time percentage calculations

### Manual Verification Workflow

#### 1. Data Population

```text
Colocalization → Run Automated Analysis → Colocalization Manual → Populate
```

#### 2. Visual Inspection Interface

- **Thumbnail Grid**: Individual spot crops at 4× magnification
- **Side-by-Side Display**: Reference and colocalization channels paired
- **Interactive Checkboxes**: Toggle colocalization status per spot
- **Real-Time Statistics**: Updated percentages during review

#### 3. Quality Control Features

- **Scalable Thumbnails**: Adjustable crop size for detail examination
- **Consistent Scaling**: Normalized intensity display across crops
- **Systematic Layout**: Organized grid prevents missed spots
- **Progress Tracking**: Live count of reviewed vs. total spots

#### 4. Manual Verification Process

1. **Load Results**: Click "Populate" to import automated classifications
2. **Systematic Review**: Examine each crop pair sequentially
3. **Adjust Classifications**: Check/uncheck based on visual assessment
4. **Monitor Progress**: Track completion via statistics display
5. **Export Data**: Save final manual classifications

> **Note:** Manual verification overrides are automatically applied to the `is_colocalized` column in `df_tracking` when tracking data is exported (from the Export tab or the Tracking tab's "Export DataFrame" button). This means the exported tracking CSV always reflects the most recent manual verification state, not just the automated classification. The override is applied at export time, so you can re-verify and re-export without re-running colocalization.

### Distance-Based Colocalization

The Distance subtab provides colocalization analysis based on the physical proximity between spots in different channels, using the Euclidean distance between detected particles.

#### Mathematical Basis

Two particles are considered colocalized if their distance is within a specified threshold:

```text
Distance = √[(x₁ - x₂)² + (y₁ - y₂)² + z_scale × (z₁ - z₂)²]
```

Where:

- **(x₁, y₁, z₁)**: Coordinates of the reference channel spot
- **(x₂, y₂, z₂)**: Coordinates of the nearest target channel spot
- **z_scale**: Anisotropic scaling factor (voxel_z / voxel_xy) for 3D mode

#### Configuration Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Reference Channel** | 0-N | Ch0 | Channel containing reference particles |
| **Colocalize Channel** | 0-N | Ch1 | Channel to test for colocalization |
| **Distance Threshold** | 0.5-20 px | 2.0 | Maximum distance for colocalization |
| **Use 3D** | Boolean | False | Include Z-dimension in distance calculation |
| **Cell Selection** | All/Per-cell | All Cells (pooled) | Pool all cells or analyze per-cell |

#### Results Display

The Distance tab displays:

- **Summary Statistics**: Total colocalized spots, per-channel breakdown, percentages
- **Per-Cell Table**: Individual cell statistics when segmentation is available
- **Visualization Mode**:
  - **Scatter View**: XY positions of spots color-coded by classification
  - **Overlay View**: Image with spot markers (yellow circles = colocalized)

#### Interpretation Note

The Distance tab reports statistics based on **individual observations** (each detection in each frame), not unique particle tracks. For a time-lapse with 360 frames:

- A single particle detected in all frames contributes 360 observations
- The colocalization percentage reflects the ratio of colocalized observations to total observations

### Verify Distance Tab

The Verify Distance subtab provides manual review of distance-based colocalization results, displaying **unique particle tracks** rather than individual frame observations.

#### Key Concepts

**Observations vs. Tracks:**

| Metric | Distance Tab | Verify Distance Tab |
|--------|--------------|---------------------|
| **Unit** | Per-frame observation | Unique particle track |
| **Count** | Total detections across all frames | Unique tracked particles |
| **Typical ratio** | Many observations per track | One entry per track |

**Example:**

For a 360-frame video with 200 unique particles averaging 120 frames each:

- **Distance Tab**: Shows ~24,000 observations (200 × 120)
- **Verify Distance Tab**: Shows 200 unique tracks

#### Colocalization Logic

A particle track is marked as **colocalized** if:

- **ANY observation** of that particle (in any frame) is within the distance threshold of a target channel spot

This means a track that is colocalized in 50% of its frames will still appear as "colocalized" in Verify Distance.

#### Workflow

1. **Run Distance Colocalization**: Complete analysis in the Distance subtab
2. **Click "Populate"**: Load results into the Verify Distance view
3. **Review Thumbnails**: Each row shows a time-averaged crop of both channels
4. **Adjust Classifications**: Check/uncheck based on visual inspection
5. **Click "Sort"**: Orders spots by distance (closest to threshold first)
6. **Export Data**: Save verified classifications

#### Sorting Purpose

The **Sort** button orders spots by their minimum distance to the threshold:

- Spots closest to the threshold (most uncertain) appear first
- Enables efficient review of borderline cases
- Already clearly colocalized or non-colocalized spots are pushed to the end

#### Statistics Display

The header shows:

```text
[Distance: 2.0px / 260nm] Total: 200 | Colocalized: 195 (97.5%)
```

- **Distance**: Applied threshold in pixels and nanometers
- **Total**: Number of unique particle tracks
- **Colocalized**: Tracks with at least one colocalized observation

## Tracking Visualization

The Visualization tab provides detailed single-trajectory inspection with synchronized multi-channel display.

### Overview

- **Purpose**: Review individual particle trajectories in detail before export
- **Input**: Completed particle tracking (from Tracking tab)
- **Output**: Per-trajectory images and videos

### Interface

| Panel | Content |
|-------|---------|
| **Left** | Full-field view with all trajectories; selected particle highlighted |
| **Right (per channel)** | Cropped view centered on the selected particle |
| **Particle list** | Scrollable list of all trajectory IDs |

### Workflow

1. **Select a particle**: Click a trajectory ID in the right-side list
   - The full-field view marks the particle with a red circle
   - Cropped panels update to show all channels at the current frame
2. **Navigate time**: Use the time slider or play button to follow the particle
3. **Adjust display**: Use Min/Max intensity sliders and channel buttons for optimal contrast
4. **Display options**:
   - **Remove Background**: Apply segmentation mask to suppress out-of-cell signal
   - **Show Scalebar**: Overlay a physical scale bar
   - **Show Time Stamp**: Display the current time point
5. **Export**:
   - **Export Image**: Save the current multi-panel view as a PNG
   - **Export Video**: Save an MP4/GIF of the selected particle's dynamics over time

### Use Cases

- Verify that a trajectory is a genuine single particle and not a tracking artifact
- Compare particle appearance in reference vs. colocalization channel
- Generate figures for publications showing individual trajectory examples
- Identify outliers (e.g., particles that split, merge, or blink)

---

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
| `spot_type` | int | Actual imaging channel number used for spot detection (e.g., if tracking channel 2, spot_type=2) | - |
| `spot_id` | int | Unique spot identifier within each frame | - |
| `unique_particle` | str | Hierarchical unique particle identifier (format: `cell_id_spot_type_particle` when segmentation is available, or `particle` otherwise). This ensures particles are uniquely identified across cells AND channels in multi-cell, multi-channel experiments. | - |

#### Colocalization Columns (When Colocalization Available)

These columns are added to `df_tracking` when colocalization analysis is performed. They enable per-particle colocalization labeling directly in the tracking export.

| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `is_colocalized` | float/NaN | Visual (ML or Intensity) colocalization flag per particle | `True`/`False`/`NaN` |
| `is_colocalized_distance` | float/NaN | Distance-based colocalization flag per frame | `True`/`False`/`NaN` |

**Behavior:**

- **`is_colocalized`**: Set when visual colocalization (ML or Intensity method) is run. This is a **per-particle** flag — all frames of a given particle share the same value, since visual colocalization evaluates time-averaged crops.
- **`is_colocalized_distance`**: Set when distance-based colocalization is run. This is a **per-frame** flag — each observation is independently evaluated based on spatial proximity in that frame.
- Particles not analyzed (e.g., in cells not selected for colocalization in single-cell mode) retain `NaN`.
- If manual verification is performed, `is_colocalized` is updated at export time to reflect the user's checkbox selections (see [Manual Verification Workflow](#manual-verification-workflow)).
- Re-running tracking clears both columns; re-running colocalization regenerates them.

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

#### Distance Colocalization DataFrame

The Distance colocalization analysis exports per-cell classification summary via the "Export Data" button:

| Column | Type | Description |
|--------|------|-------------|
| `cell_id` | int | Cell identifier |
| `n_ch0_spots` | int | Total spots in reference channel |
| `n_ch1_spots` | int | Total spots in target channel |
| `n_colocalized` | int | Number of colocalized spots |
| `n_ch0_only` | int | Spots only in reference channel |
| `n_ch1_only` | int | Spots only in target channel |
| `coloc_pct` | float | Colocalization percentage |
| `method` | str | Always "distance" |
| `threshold_px` | float | Distance threshold in pixels |
| `threshold_nm` | float | Distance threshold in nanometers |
| `use_3d` | bool | Whether 3D distance was used |
| `channel_0` | int | Reference channel index |
| `channel_1` | int | Target channel index |

**Note:** Counts are based on individual frame observations (all spot detections across all time frames).

#### Verify Distance DataFrame

The Verify Distance manual verification exports per-track classification via the "Export Data" button:

| Column | Type | Description |
|--------|------|-------------|
| `spot_index` | int | Index of the unique particle track (0, 1, 2...) |
| `colocalized_manual` | bool | User-verified colocalization status |
| `method` | str | Always "distance" |
| `threshold_px` | float | Distance threshold in pixels |
| `threshold_nm` | float | Distance threshold in nanometers |
| `use_3d` | bool | Whether 3D distance was used |
| `image_name` | str | Name of the analyzed image |

**Note:** Counts are based on unique particle tracks, not individual frame observations. A track is marked colocalized if ANY of its frame observations met the distance threshold.

### Data Export and Integration

#### CSV Export Format

All DataFrames are exported as CSV files with UTF-8 encoding:

- **Tracking data**: `tracking_[filename]_[imagename].csv`
  - When colocalization has been run, the tracking CSV includes `is_colocalized` and/or `is_colocalized_distance` columns alongside the standard tracking data. If manual verification was performed, the exported values reflect the verified labels.
- **Colocalization results**: `colocalization_[filename]_[imagename].csv`
- **Manual colocalization**: `manual_colocalization_[filename]_[imagename].csv`
