# Tutorial - MicroLive

## Introduction

This tutorial provides comprehensive step-by-step instructions for all analysis workflows in the MicroLive GUI. We'll cover each tab in order, from basic image loading to advanced analysis and data export.

## Prerequisites

Before starting, ensure you have:
- Sample microscopy data (.lif or .tif format)
- Basic understanding of your imaging parameters
- Sufficient computational resources for your dataset

---

## Tutorial 1: Import Tab - Image Loading and Visualization

### Overview
Learn to load, visualize, and navigate through microscopy data with proper display settings.

### Step 1: Application Launch and File Loading

1. **Launch the Application**
   ```bash
   python gui/micro.py
   ```

2. **Open Image Files**
   - Click "Open File" button
   - Navigate to your microscopy file (.lif or .tif)
   - Select the file and click "Open"
   - For LIF files: multiple scenes will appear in the tree view
   - For TIFF files: single image will be loaded immediately

3. **Select Image Scene** (LIF files only)
   - Expand the file in the tree view on the right
   - Click on the desired scene/image
   - The image will load and display in the main canvas

### Step 2: Metadata Verification

1. **Check Image Information Panel**
   - Review file name, dimensions, and channels
   - Verify frames, Z-slices, and pixel counts
   - Check bit depth and time intervals
   - Confirm voxel sizes (XY and Z)

2. **Handle Missing Metadata**

If prompted, enter missing values:
- **PhysicalSizeX and PhysicalSizeY**: Pixel size in nanometers (nm) for both X and Y dimensions
- **PhysicalSizeZ**: Z-step size in nanometers (nm)
- **TimeIncrement**: Time interval between frames in seconds (s)

These values are crucial for quantitative analysis and proper scaling of measurements.

### Step 3: Navigation and Display Controls

1. **Time Navigation**
   - Use the horizontal time slider to browse frames
   - Click "Play" button for automatic playback
   - Observe temporal dynamics of your sample

2. **Z-plane Navigation**
   - Use the vertical Z-slider on the right of the image
   - Top position = maximum projection (recommended for tracking)
   - Lower positions = individual Z-planes

3. **Channel Selection**
   - Click individual channel buttons (Ch0, Ch1, etc.)
   - Each channel displays with its assigned colormap
   - Green (Ch0), Magenta (Ch1), Yellow (Ch2)

4. **Merge Channels**
   - Click "Merge Channels" for multi-channel overlay
   - Useful for assessing colocalization visually
   - Individual channel intensities blend additively

### Step 4: Display Optimization

1. **Adjust Channel Parameters**
   - Select channel tabs in the control panel
   - Adjust intensity percentiles (min: 0-50%, max: 90-100%)
   - Modify smoothing (high sigma for noise reduction)
   - Fine-tune low sigma for subtle enhancement

2. **Theme and Overlays**
   - Toggle Dark/Light theme with the switch
   - Enable "Time" checkbox for timestamp overlay
   - Enable "Background" checkbox for segmentation overlay

### Step 5: Export Options

1. **Export Current Image**
   - Click "Export Image" for high-resolution PNG (300 DPI)
   - Choose filename and location
   - Current display settings are preserved

2. **Export Time-lapse Video**
   - Click "Export Video" for MP4 or GIF format
   - All frames exported with current display settings
   - Scalebar included if voxel size is available

### Expected Results
- Properly loaded and calibrated microscopy data
- Optimized display settings for visualization
- Understanding of data structure and quality

---

## Tutorial 2: Segmentation Tab - Cell Boundary Definition

### Overview
Define cellular regions for masking subsequent analyses, ensuring particles are analyzed only within cells of interest.

### Step 1: Segmentation Setup

1. **Navigate to Segmentation Tab**
   - Click the "Segmentation" tab
   - The interface shows image on left, controls on right

2. **Select Image Source**
   - Choose appropriate channel for segmentation.
   - Use channel buttons to switch between channels
   - Use time slider to select representative frame

3. **Configure Maximum Projection**
   - Check "Use Max Projection for Segmentation" if needed
   - Useful for thick samples or when cells extend through Z
   - Status label indicates current setting

### Step 2: Manual Segmentation

1. **Start Manual Segmentation**
   - Click "Manual Segmentation" button
   - Image displays with preprocessing applied
   - Click handler activates for polygon drawing

2. **Draw Cell Boundary**
   - Click points around the cell perimeter
   - Points are connected automatically
   - Work clockwise or counter-clockwise around cell
   - Include entire cell body and extensions

3. **Complete Segmentation**
   - Click "Finish Segmentation" when polygon is complete
   - Binary mask is generated automatically
   - Red overlay shows segmented region

### Step 3: Watershed Segmentation

1. **Alternative Automated Method**
   - Adjust "Threshold Factor" slider (typically 1.0-2.0)
   - Higher values = more stringent segmentation
   - Click "Run Watershed Segmentation"

2. **Evaluate Results**
   - Check that cell boundaries are correctly identified
   - Adjust threshold and re-run if needed
   - May work better for well-contrasted cells

### Step 4: Quality Control and Export

1. **Verify Segmentation Quality**
   - Ensure mask covers intended cellular region
   - Check that background is properly excluded
   - Re-segment if boundaries are inaccurate

2. **Export Segmentation**
   - Click "Export Image" for segmentation visualization
   - Click "Export Mask" for binary mask as TIFF
   - Save documentation of segmentation approach

### Expected Results
- Binary mask defining cellular region
- Quality segmentation for subsequent masking
- Exported documentation of segmentation

---

## Tutorial 3: Photobleaching Tab - Intensity Correction

### Overview
Correct for photobleaching artifacts that affect quantitative fluorescence measurements over time.

### Step 1: Prerequisites Check

1. **Ensure Requirements are Met**
   - Must have segmentation mask from previous step
   - Time-series fluorescence data required
   - Sufficient time points for fitting (>20 frames recommended)

### Step 2: Photobleaching Analysis Setup

1. **Navigate to Photobleaching Tab**
   - Click "Photobleaching" tab
   - Review available parameters

2. **Select Analysis Mode**
   - **Inside Cell**: Use pixels within segmentation mask
   - **Outside Cell**: Use background pixels outside mask
   - **Circular Region**: Define custom circular analysis area

3. **Configure Parameters**
   - **Radius**: Size of analysis region (for circular mode)
   - **Remove Time Points**: Exclude initial frames if needed (typically 0-5)
   - **Model Type**: Choose appropriate decay model
     - **Exponential**: Most common, single exponential decay
     - **Linear**: For approximately linear bleaching
     - **Double Exponential**: Two-component decay (complex systems)

### Step 3: Execute Correction

1. **Run Photobleaching Analysis**
   - Click "Run Photobleaching" button
   - Wait for fitting to complete
   - Progress indicated in status

2. **Interpret Results**
   - **Left Panels**: Show fitted decay curves for each channel
   - **Right Panels**: Compare original vs. corrected intensities
   - **Fit Parameters**: Decay rates and amplitudes displayed
   - **Model Quality**: Assess goodness of fit

### Step 4: Evaluate Correction

1. **Check Correction Quality**
   - Corrected data should show stable intensities
   - Original data shows declining intensities
   - Fit should capture the decay trend well

2. **Adjust if Needed**
   - Try different model types if fit is poor
   - Adjust removed time points if early frames are problematic
   - Consider different analysis regions

### Step 5: Apply Corrected Data

1. **Use in Subsequent Analysis**
   - Photobleaching-corrected images now available
   - Select "Photobleaching Corrected" in tracking source
   - All subsequent quantitative analyses benefit from correction

2. **Export Results**
   - Click "Export Photobleaching Image" for documentation
   - Save plots showing correction effectiveness

### Expected Results
- Quantified photobleaching parameters
- Corrected image stack for quantitative analysis
- Documentation of correction effectiveness

---

## Tutorial 4: Tracking Tab - Particle Detection and Trajectory Linking

### Overview
Detect particles and link them into trajectories for dynamic analysis.

### Step 1: Tracking Setup and Image Source

1. **Navigate to Tracking Tab**
   - Click "Tracking" tab
   - Left panel shows image display, right panel shows parameters

2. **Configure Image Source**
   - Select "Original Image" or "Photobleaching Corrected"
   - Use corrected images for quantitative intensity measurements
   - 2D projection checkbox controls Z-processing

3. **Set Display Parameters**
   - Adjust "Min Int" and "Max Int" percentiles
   - Optimize contrast for particle visualization
   - Select appropriate channel for detection

### Step 2: Threshold Selection

1. **Generate Threshold Histogram**
   - Navigate through frames to see typical intensity
   - Histogram appears when parameters are set
   - Shows intensity distribution within segmentation mask

2. **Set Detection Threshold**
   - Use threshold slider to set detection level
   - Red line shows current threshold on histogram
   - Aim for 3-5× background level
   - Balance sensitivity vs. false detections

### Step 3: Detection Parameters

1. **Spot Size Configuration**
   - **YX Spot Size**: Expected particle size in pixels (typically 5-9)
   - **Z Spot Size**: Axial extent in pixels (typically 3-5)
   - Must be odd numbers (automatically adjusted)

2. **Clustering Parameters**
   - **Cluster Radius**: Distance for grouping spots (nm)
   - **Max Cluster Size**: Maximum spots per cluster (0 = no limit)
   - Helps handle aggregated particles

### Step 4: Trajectory Linking Parameters

1. **Search and Memory Settings**
   - **Max Range Search**: Maximum distance particles can move between frames
   - **Memory**: Frames a particle can disappear (typically 1-3)
   - **Min Length Trajectory**: Minimum trajectory duration (frames)

2. **Advanced Options**
   - **2D Projection**: Use maximum projection for detection
   - **Fixed Size Intensity**: Use consistent aperture for measurements
   - **3D Coordinates**: Include Z in linking (if available)

### Step 5: Detection and Tracking Execution

1. **Test Detection**
   - Click "Single Frame" to test current parameters
   - Red circles show detected particles
   - Adjust threshold if too few/many detections

2. **Detect All Frames**
   - Click "Detection" for complete detection
   - Progress dialog shows processing status
   - Review detection overlay across time

3. **Link Trajectories**
   - Click "Tracking" to link detections into trajectories
   - Most computationally intensive step
   - Creates complete trajectory dataset

### Step 6: Visualization and Quality Control

1. **Enable Visualization Options**
   - Check "Trajectories" to show particle paths
   - Check "Cluster Size" to show clustering info
   - Check "Particle ID" to show trajectory numbers
   - Check "Time Stamp" for temporal reference

2. **Quality Assessment**
   - Look for broken trajectories (insufficient linking)
   - Look for incorrect linking (excessive linking)
   - Use play button to observe trajectory dynamics

3. **Random Control Spots**
   - Enable "Generate Random Spots" for controls
   - Specify number of random locations
   - Provides background measurements for comparison

### Step 7: Export Tracking Results

1. **Export Data**
   - Click "Export DataFrame" for CSV trajectory data
   - Contains all particle measurements and properties
   - Use default naming or specify custom filename

2. **Export Visualizations**
   - Click "Export Image" for current frame visualization
   - Click "Export Video" for complete trajectory movie
   - Includes all enabled overlays and colormaps

### Expected Results
- Complete particle trajectory dataset
- Quantitative measurements for each detection
- Visual documentation of tracking quality
- CSV file ready for further analysis

---

## Tutorial 5: Distribution Tab - Statistical Analysis of Particle Properties

### Overview
Generate histograms and statistical summaries of measured particle properties.

### Step 1: Data Requirements

1. **Prerequisites**
   - Must have completed particle tracking
   - Non-empty trajectory dataset required
   - Multiple particles recommended for meaningful statistics

### Step 2: Distribution Analysis Setup

1. **Navigate to Distribution Tab**
   - Click "Distribution" tab
   - Left panel for plots, right panel for controls

2. **Select Data Field**
   - **Field dropdown**: Choose measurement to analyze
     - `spot_int`: Integrated intensity (most common)
     - `psf_amplitude`: Fitted peak intensity
     - `psf_sigma`: Particle size/width
     - `snr`: Signal-to-noise ratio
     - `total_spot_int`: Total intensity including background
     - `cluster_size`: Number of spots in cluster

3. **Select Channel**
   - Choose channel for analysis
   - Must match field selection (e.g., spot_int_ch_0)
   - Different channels may show different distributions

### Step 3: Histogram Configuration

1. **Set Percentile Range**
   - **Min Percentile**: Lower bound for histogram (0-50%)
   - **Max Percentile**: Upper bound for histogram (50-100%)
   - Helps exclude outliers and focus on main distribution

2. **Generate Histogram**
   - Click "Plot Histogram" to create distribution plot
   - 60 bins used by default
   - Statistics displayed: mean and median values

### Step 4: Interpretation

1. **Analyze Distribution Shape**
   - Normal distribution suggests homogeneous population
   - Bimodal distribution may indicate subpopulations
   - Skewed distribution suggests measurement artifacts or biology

2. **Compare Across Conditions**
   - Generate histograms for different channels
   - Compare different measurement fields
   - Document differences between experimental conditions

### Step 5: Export Results

1. **Save Distribution Plots**
   - Click "Export Distribution Image"
   - High-resolution PNG with statistics
   - Include in publications or reports

### Expected Results
- Quantitative distribution of particle properties
- Statistical summary (mean, median)
- Visual representation suitable for publication

---

## Tutorial 6: Time Course Tab - Temporal Dynamics Analysis

### Overview
Analyze how particle properties change over time, revealing dynamic processes.

### Step 1: Time Course Setup

1. **Navigate to Time Course Tab**
   - Click "Time Course" tab
   - Requires completed tracking data

2. **Select Analysis Parameters**
   - **Channel**: Choose channel for analysis
   - **Data Type**: Select measurement type
     - `particles`: Number of detected particles per frame
     - `spot_int`: Average spot intensity over time
     - `psf_amplitude`: Average amplitude over time
     - Other measurements available

### Step 2: Configure Analysis Range

1. **Set Percentile Filters**
   - **Min Percentile**: Exclude lowest values (typically 5%)
   - **Max Percentile**: Exclude highest values (typically 95%)
   - Helps remove outliers and focus on main population

### Step 3: Generate Time Course

1. **Execute Analysis**
   - Click "Plot Time Course" button
   - Processing time depends on dataset size
   - Multiple traces may appear for different particles/conditions

2. **Interpret Results**
   - **Y-axis**: Selected measurement values
   - **X-axis**: Time (frames or seconds if calibrated)
   - **Trends**: Look for increases, decreases, or oscillations
   - **Variability**: Assess consistency across time


### Step 4: Export Time Course Data

1. **Save Time Course Plots**
   - Click "Export Time Courses Image"
   - High-resolution plots with proper axes labels
   - Suitable for presentations and publications

### Expected Results
- Temporal profiles of particle properties
- Identification of dynamic processes
- Quantitative time course data for further analysis

---

## Tutorial 7: Correlation Tab - Temporal Correlation Analysis

### Overview
Analyze temporal correlations in particle dynamics to extract kinetic information and assess particle interactions.

### Step 1: Correlation Analysis Prerequisites

1. **Data Requirements**
   - Completed particle tracking with good quality trajectories
   - Sufficient time points (>50 frames recommended)
   - Long trajectories provide better correlation estimates

2. **Navigate to Correlation Tab**
   - Click "Correlation" tab
   - Left panel for plots, right panel for parameters

### Step 2: Select Correlation Type

1. **Autocorrelation Analysis**
   - Select "Auto" radio button
   - Analyzes single channel temporal dynamics
   - Reveals characteristic time scales of fluctuations

2. **Cross-correlation Analysis**
   - Select "Cross" radio button
   - Compares dynamics between two channels
   - Identifies temporal relationships and delays

### Step 3: Channel and Field Selection

1. **Select Channels**
   - Check appropriate channel checkboxes
   - For autocorrelation: select one channel
   - For cross-correlation: select two channels

2. **Choose Data Field**
   - **Field dropdown**: Select measurement for correlation
   - `spot_int`: Most common choice for intensity correlations
   - `psf_amplitude`: Alternative intensity measure
   - Other fields available for specialized analyses

### Step 4: Configure Correlation Parameters

1. **Data Quality Settings**
   - **Min % Data**: Require minimum data completeness per trajectory
   - **Threshold**: Decorrelation threshold for analysis
   - **Remove outliers**: Filter extreme values

2. **Fit Configuration**
   - **Fit Type**: Choose "Linear" or "Exponential"
   - **Index Max Lag for Plot**: Set range for visualization
   - **Index Max Lag for Fit**: Set range for curve fitting
   - **Start Lag**: Usually set to 1 to exclude τ=0

3. **Display Options**
   - **Min/Max Percentile**: Set plot range
   - **Normalize**: Normalize correlation amplitude
   - **Baseline correction**: Remove systematic trends

### Step 5: Execute Correlation Analysis

1. **Run Correlation Calculation**
   - Click "Run" button
   - Processing time depends on data size and lag range
   - Progress indicated during calculation

2. **Interpret Autocorrelation Results**
   - **G(τ) vs τ**: Correlation amplitude vs. time lag
   - **Decorrelation time**: Time scale of intensity fluctuations
   - **Amplitude**: Magnitude of fluctuations
   - **Exponential fit**: Provides characteristic time constant

3. **Interpret Cross-correlation Results**
   - **Peak position**: Time delay between channels
   - **Peak amplitude**: Strength of correlation
   - **Asymmetry**: Indicates directional relationships

### Step 6: Advanced Analysis Options

1. **Fitting Models**
   - **Linear**: For initial analysis or non-exponential behavior
   - **Exponential**: Most common for single-component kinetics
   - Fit parameters provide quantitative kinetic information

2. **Quality Assessment**
   - **Error bars**: Indicate statistical uncertainty
   - **Fit quality**: Assess how well model describes data
   - **Smoothed curves**: Help identify trends in noisy data

### Step 7: Export Correlation Results

1. **Save Correlation Analysis**
   - Click "Export Correlation Image"
   - High-resolution plots with fit parameters
   - Include both raw data and fitted curves

### Expected Results
- Quantitative correlation functions
- Kinetic parameters from curve fitting
- Temporal relationships between channels
- Publication-quality correlation plots

---

## Tutorial 8: Colocalization Tab - Automated Spatial Analysis

### Overview
Quantify spatial relationships between particles in different channels using automated methods.

### Step 1: Colocalization Prerequisites

1. **Data Requirements**
   - Multi-channel microscopy data (minimum 2 channels)
   - Completed particle tracking on reference channel
   - Good registration between channels

### Step 2: Colocalization Setup

1. **Navigate to Colocalization Tab**
   - Click "Colocalization" tab
   - Interface shows channel selection and method options

2. **Select Analysis Channels**
   - **Reference**: Channel containing tracked particles
   - **Colocalize**: Target channel for colocalization analysis
   - Typically reference is the tracking channel

### Step 3: Choose Analysis Method

1. **Machine Learning Method**
   - Select "ML" radio button
   - Uses trained neural network for classification
   - **ML Threshold**: Set confidence threshold (0.5-1.0)
   - Higher thresholds = more stringent colocalization

2. **Intensity-Based Method**
   - Select "Intensity" radio button
   - Uses signal-to-noise ratio criteria
   - **Threshold**: Set SNR threshold (typically 3-5)
   - Higher thresholds = more stringent requirements

### Step 4: Configure Display Options

1. **Crop Matrix Settings**
   - **Columns**: Number of columns in result matrix
   - More columns = more crops displayed
   - Balance between detail and overview

### Step 5: Execute Colocalization Analysis

1. **Run Analysis**
   - Click "Run" button
   - Processing generates crop matrix around each particle
   - Red borders indicate colocalized spots

2. **Interpret Results**
   - **Colocalization Percentage**: Quantitative measure
   - **Visual Matrix**: Side-by-side channel comparison
   - **Color Coding**: Flagged spots highlighted
   - **Crop Spacing**: Organized grid layout

### Step 6: Quality Control

1. **Assess Analysis Quality**
   - Visually inspect flagged vs. non-flagged spots
   - Check for false positives/negatives
   - Consider adjusting thresholds if needed

2. **Interactive Zoom**
   - Hover over matrix for magnified view
   - Red rectangle shows current zoom region
   - Detailed inspection of individual spots

### Step 7: Export Colocalization Results

1. **Export Quantitative Data**
   - Click "Export Data" for CSV summary
   - Contains percentages and spot counts
   - Includes method and threshold information

2. **Export Visualization**
   - Click "Export Image" for crop matrix
   - High-resolution documentation of results
   - Includes colocalization percentage in title

### Expected Results
- Quantitative colocalization percentage
- Visual matrix showing analysis results
- CSV file with detailed statistics
- Publication-ready visualization

---

## Tutorial 9: Colocalization Manual Tab - Expert Validation

### Overview
Manually verify and refine automated colocalization results through expert visual inspection.

### Step 1: Manual Verification Setup

1. **Navigate to Manual Tab**
   - Click "Colocalization Manual" tab
   - Requires completed automated colocalization

2. **Populate Initial Results**
   - Click "Populate" to load automated results
   - Checkboxes reflect ML or intensity analysis
   - Each spot pair shown as thumbnail image

### Step 2: Visual Inspection Process

1. **Review Individual Spots**
   - Scroll through spot pairs in thumbnail view
   - Each thumbnail shows both channels side-by-side
   - 4× scale provides detailed view

2. **Manual Classification**
   - Check boxes for true colocalization
   - Uncheck boxes for false positives
   - Base decisions on visual overlap and intensity


### Step 3: Statistical Tracking

1. **Monitor Progress**
   - Statistics update in real-time
   - Shows total spots, marked colocalized, percentage
   - Track progress through dataset

2. **Consistency Checks**
   - Review decisions periodically
   - Consider inter-observer variability
   - Document decision criteria

### Step 4: Refinement Tools

1. **Cleanup Function**
   - Click "Cleanup" to uncheck all boxes
   - Start fresh with manual inspection
   - Useful for complete re-analysis

2. **Batch Operations**
   - Work systematically through thumbnails
   - Focus on borderline cases
   - Accept obvious positives/negatives quickly

### Step 5: Export Manual Results

1. **Save Manual Analysis**
   - Click "Export Data" for refined CSV results
   - Includes manually verified percentages
   - Method marked as "Manual" for documentation

### Expected Results
- Expert-validated colocalization percentage
- Refined analysis with reduced false positives/negatives
- Documentation of manual validation process
- High-confidence colocalization measurements

---

## Tutorial 10: Tracking Visualization Tab - Advanced Particle Inspection

### Overview
Detailed visualization and analysis of individual particle trajectories with multi-channel context.

### Step 1: Tracking Visualization Setup

1. **Navigate to Tracking Visualization Tab**
   - Click "Tracking Visualization" tab
   - Requires completed particle tracking

2. **Populate Particle List**
   - Particle list shows all tracked particles
   - Listed by particle ID number
   - Select individual particles for detailed analysis

### Step 2: Particle Selection and Display

1. **Select Particle of Interest**
   - Click particle in list (right panel)
   - Main display updates to show selected particle
   - Red circle highlights particle position

2. **Multi-panel Display**
   - **Left Panel**: Full field view with particle marked
   - **Right Panels**: Cropped views of each channel
   - **Synchronized**: All panels show same time point

### Step 3: Display Optimization

1. **Intensity Controls**
   - Adjust "Min Int" and "Max Int" percentiles
   - Optimize contrast for particle visibility
   - Apply to all channels simultaneously

2. **Channel Selection**
   - Click individual channel buttons for single-channel view
   - Click "Merge Channels" for overlay view
   - Toggle between visualization modes

### Step 4: Temporal Navigation

1. **Time Control**
   - Use time slider for frame-by-frame analysis
   - Play button for automatic progression
   - Observe particle dynamics over time

2. **Display Options**
   - **Remove Background**: Apply segmentation mask
   - **Show Scalebar**: Add scale reference
   - **Show Time Stamp**: Display temporal information

### Step 5: Advanced Analysis

1. **Trajectory Inspection**
   - Follow particle movement through time
   - Assess trajectory quality and continuity
   - Identify potential tracking errors

2. **Multi-channel Comparison**
   - Compare particle appearance across channels
   - Assess intensity relationships
   - Identify channel-specific behaviors

### Step 6: Export Detailed Results

1. **Export Single Frame**
   - Click "Export Image" for current view
   - Includes all display settings and overlays
   - Documents specific particle behavior

2. **Export Particle Movie**
   - Click "Export Video" for temporal sequence
   - Shows selected particle dynamics
   - Useful for presentations and detailed analysis

### Expected Results
- Detailed inspection of individual particle behavior
- Multi-channel context for particle analysis
- High-quality visualizations for documentation
- Videos showing particle dynamics

---

## Tutorial 11: Crops Tab - Particle Crop Analysis

### Overview
Generate and analyze standardized crops around detected particles for detailed morphological analysis.

### Step 1: Crops Analysis Setup

1. **Navigate to Crops Tab**
   - Click "Crops" tab
   - Requires completed particle tracking

2. **Channel Selection**
   - Click channel buttons to select analysis channel
   - Choose channel with best particle contrast
   - Can generate crops for each channel separately

### Step 2: Generate Particle Crops

1. **Execute Crop Analysis**
   - Click "Plot Crops" button
   - Generates standardized crops around each detected particle
   - Crops are normalized for consistent comparison

2. **Crop Matrix Display**
   - Grid layout shows all particle crops
   - Consistent size and scale for comparison
   - Background normalized for visualization

### Step 3: Crop Analysis and Interpretation

1. **Visual Inspection**
   - Assess particle morphology consistency
   - Identify outliers or artifacts
   - Evaluate detection quality across dataset

2. **Pattern Recognition**
   - Look for morphological patterns
   - Identify different particle types
   - Assess size and shape distributions

### Step 4: Quality Control

1. **Detection Validation**
   - Verify that crops contain actual particles
   - Identify false positive detections
   - Assess background contamination

2. **Parameter Optimization**
   - Use crop analysis to refine detection parameters
   - Adjust threshold or spot size if needed
   - Re-run tracking with optimized parameters

### Step 5: Export Crop Analysis

1. **Save Crop Matrix**
   - Click "Export Crops Image"
   - High-resolution grid of all particle crops
   - Useful for presentations and documentation

### Expected Results
- Standardized particle crops for morphological analysis
- Quality assessment of particle detection
- Visual documentation of particle characteristics
- Basis for detection parameter optimization

---

## Tutorial 12: Export Tab - Comprehensive Data Management

### Overview
Organize and export all analysis results in a structured, documented format for sharing and archival.

### Step 1: Export Preparation

1. **Complete Desired Analyses**
   - Finish all relevant analysis tabs
   - Ensure results are satisfactory
   - Document any special conditions or issues

2. **Navigate to Export Tab**
   - Click "Export" tab
   - Review comprehensive export options

### Step 2: Documentation and Comments

1. **Select Predefined Comments**
   - Use dropdown to select common observations:
     - "Few or no spots were detected"
     - "Aggregates in cell"
     - "Cell died during acquisition"
     - "Cell divided during acquisition"
     - "The cell goes out of focus"
     - "Error during microscope acquisition"
     - "Error during tracking. Spots not linked correctly"

2. **Add Custom Comments**
   - Select "Custom" for specific observations
   - Document experimental conditions
   - Note any analysis challenges or decisions
   - Include relevant methodological details

### Step 3: Select Export Items

1. **Review Available Exports**
   - **Export Entire Image as OME-TIF**: Original data with metadata in tif format
   - **Export Displayed Image**: Current visualization
   - **Export Segmentation Image**: Segmentation overlay
   - **Export Mask as TIF**: Binary segmentation mask
   - **Export Photobleaching Image**: Correction analysis plots
   - **Export Tracking Data**: Complete trajectory CSV
   - **Export Tracking Image**: Tracking visualization
   - **Export Distribution Image**: Histogram analysis
   - **Export Time Course Image**: Temporal analysis plots
   - **Export Correlation Image**: Correlation analysis
   - **Export Colocalization Image**: Colocalization matrix
   - **Export Colocalization Data**: Colocalization CSV
   - **Export Manual Colocalization Image**: Manual verification
   - **Export Manual Colocalization Data**: Manual CSV
   - **Export Crops Image**: Particle crop matrix
   - **Export Metadata File**: Complete analysis parameters
   - **Export User Comments**: User comments as a text file
   - **Export Random Spots Data**: Control measurements for random positions inside the mask

2. **Use Selection Tools**
   - **Select All**: Check all available exports
   - **Deselect All**: Clear all selections
   - **Individual Selection**: Choose specific items

### Step 4: Execute Batch Export

1. **Choose Export Location**
   - Click "Export Selected Items"
   - Select parent directory for export
   - Organized subfolder created automatically

2. **Folder Organization**
   - Results folder named with file and image identifiers
   - All selected items exported with consistent naming
   - Metadata file includes complete parameter documentation

### Step 5: Verify Export Completeness

1. **Review Export Folder**
   - Check that all selected items were exported
   - Verify file naming consistency
   - Confirm metadata file contains all parameters

2. **Quality Assurance**
   - Open key files to verify integrity
   - Check that CSV files contain expected data
   - Ensure images display correctly

### Step 6: Data Management Best Practices

1. **File Organization**
   - Use consistent naming conventions
   - Include date and experiment identifiers
   - Maintain folder structure across experiments

2. **Documentation Standards**
   - Always export metadata with results
   - Include user comments for context
   - Document any deviations from standard protocols

3. **Backup and Archival**
   - Create backups of exported results
   - Store raw data separately from analysis results
   - Maintain version control for analysis parameters

### Expected Results
- Organized folder containing all analysis results
- Complete documentation of analysis parameters
- Consistent file naming and structure
- User comments and metadata preserved
- Ready for sharing, publication, or archival

---

## Troubleshooting Common Issues

### Loading and Display Issues
**Problem**: Image fails to load or displays incorrectly
**Solutions**: 
- Check file format compatibility (.lif, .tif, .ome.tif)
- Verify file path and permissions
- Ensure sufficient memory for large files
- Check metadata completeness

### Segmentation Problems
**Problem**: Poor segmentation quality
**Solutions**:
- Try different channels for segmentation
- Adjust watershed threshold factor
- Use maximum projection for thick samples
- Consider manual segmentation for complex shapes

### Detection Issues
**Problem**: Too few particles detected
**Solutions**:
- Lower detection threshold
- Check segmentation mask coverage
- Verify channel selection
- Adjust spot size parameters

**Problem**: Too many false detections
**Solutions**:
- Raise detection threshold
- Improve segmentation to exclude background
- Adjust spot size to match particle size
- Check for imaging artifacts

### Tracking Problems
**Problem**: Broken trajectories
**Solutions**:
- Increase maximum search range
- Add memory frames for tracking gaps
- Improve detection consistency
- Check for sample drift

**Problem**: Incorrect particle linking
**Solutions**:
- Decrease search range to prevent over-linking
- Reduce memory frames
- Improve detection specificity
- Check for high particle density issues

### Performance Issues
**Problem**: Slow processing or memory errors
**Solutions**:
- Use 2D projection mode
- Process smaller regions or time ranges
- Close unused files and applications
- Increase system memory
- Use lower resolution for initial optimization

### Analysis Quality Issues
**Problem**: Poor correlation results
**Solutions**:
- Increase minimum trajectory length
- Improve tracking quality
- Check temporal sampling rate
- Verify sufficient data points per trajectory

**Problem**: Inconsistent colocalization
**Solutions**:
- Verify channel registration
- Check detection parameters consistency
- Use manual verification for validation
- Consider using ML method for better accuracy

## Best Practices Summary

### Data Quality
1. **Start with high-quality data**: Proper acquisition parameters, sufficient signal-to-noise ratio
2. **Verify metadata**: Ensure pixel sizes and time intervals are correct
3. **Optimize imaging**: Balance temporal resolution, spatial resolution, and photobleaching

### Analysis Workflow
1. **Sequential approach**: Follow tab order for logical workflow progression
2. **Parameter testing**: Test on subsets before full analysis
3. **Quality control**: Visually inspect results at each step
4. **Documentation**: Export parameters and add comments throughout

### Result Validation
1. **Visual inspection**: Always verify automated results visually
2. **Control experiments**: Include appropriate negative and positive controls
3. **Reproducibility**: Test parameter sensitivity and biological replicates
4. **Statistical rigor**: Use appropriate sample sizes and statistical tests

### Data Management
1. **Consistent naming**: Use systematic file and folder naming
2. **Complete export**: Always export metadata with results
3. **Version control**: Track analysis parameters across iterations
4. **Backup strategy**: Maintain copies of raw data and analysis results

### Publication Preparation
1. **Method documentation**: Export complete parameter sets
2. **Visual documentation**: Export high-quality figures
3. **Statistical reporting**: Include appropriate error bars and sample sizes
4. **Reproducibility**: Provide sufficient detail for replication

For additional support, consult the [User Guide](user_guide.md) and [API Reference](api_reference.md) documentation.
