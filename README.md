# MicroLive <img src="docs/icons/icon_micro.png" alt="Micro Logo" width="150" align="right" />

**Authors:** Luis U. Aguilera, William S. Raymond, Rhiannon M. Sears, Nathan L. Nowling, Brian Munsky, Ning Zhao

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/user_guide.md)
[![Tutorial](https://img.shields.io/badge/tutorial-step--by--step-orange.svg)](docs/tutorial.md)
[![API Reference](https://img.shields.io/badge/API-reference-blue.svg)](docs/api_reference.md)

## About

**MicroLive** is a Python library and GUI application designed to process live-cell microscope images and perform single-molecule measurements. It provides an intuitive interface for particle tracking, colocalization analysis, correlation analysis, and advanced visualization tools, making it ideal for quantitative microscopy research.

<img src="docs/Microlive_video_720_fast.gif" alt="MicroLive Demo" width="900" />

---

## Features

### Image I/O and Management

- **Multi-format Support**: Load .lif (Leica), .tif, and .ome.tif files with automatic metadata extraction
- **Dimension Mapping**: Interactive mapping for arbitrary file dimensions to standard microscopy format [T, Z, Y, X, C]
- **File Management**: Tree view interface for managing multiple datasets simultaneously
- **Metadata Preservation**: Comprehensive metadata handling and export

### Image Registration

- **Drift Correction**: Correct for sample drift and translation across time-lapse acquisitions
- **ROI-based Alignment**: Select reference region for enhanced registration accuracy
- **Multiple Algorithms**: Support for phase correlation and intensity-based registration methods
- **Before/After Comparison**: Side-by-side visualization of original and registered images

### Segmentation

- **Manual ROI Drawing**: Draw polygonal regions of interest for cell boundaries
- **Watershed Segmentation**: Automated segmentation with adjustable threshold factor
- **Mask Overlay and Export**: Display segmented regions as overlays and export binary mask images (TIFF format)

### Cellpose Segmentation

- **Deep Learning Segmentation**: GPU-accelerated cell segmentation using Cellpose models (cyto3, nuclei)
- **Dual Mask Support**: Simultaneous cytoplasm and nucleus segmentation with automatic pairing
- **Time-varying Masks (TYX)**: Generate masks across multiple timepoints for dynamic cell tracking
- **Parameter Optimization**: Automated grid search for optimal diameter and flow threshold parameters
- **Border Cell Removal**: Option to exclude cells touching image boundaries

### Particle Tracking

- **Automated Detection**: Spot detection with customizable size and intensity filtering
- **Trajectory Linking**: Advanced linking algorithms with memory and search range controls
- **Clustering Analysis**: Handle aggregated particles by merging spots within a specified radius (in nm) and reporting cluster sizes
- **Quality Control**: Random spot generation for validation and trajectory length filtering
- **Real-time Visualization**: Live tracking overlays with customizable display options

### Mean Squared Displacement (MSD) Analysis

- **Diffusion Coefficient Calculation**: Compute ensemble-averaged MSD and extract diffusion coefficients (D)
- **Per-cell Analysis**: Calculate and compare D values across individual cells
- **2D and 3D Modes**: Support for both planar and volumetric diffusion analysis
- **Linear Fitting**: Adjustable number of fit points for slope calculation
- **Log-log Plotting**: Optional logarithmic scale for anomalous diffusion detection
- **Data Export**: Export MSD curves and D values as CSV

### Colocalization Analysis

- **Machine Learning-Assisted Analysis**: Machine learning model for automated colocalization classification between channels
- **Intensity-Based Methods**: Signal-to-noise ratio thresholds for spot colocalization
- **Manual Verification**: Grid-based interface for expert validation of automated results (Coloc Edit)
- **Threshold Control**: Adjustable ML confidence or SNR thresholds
- **Comprehensive Export**: Visualization matrices and quantitative data export

### Statistical Analysis

- **Distribution Analysis**: Histogram plots of particle properties (intensity, size, SNR, cluster size)
- **Time Course Analysis**: Temporal analysis of particle metrics with percentile filtering
- **Correlation Analysis**: Auto- and cross-correlation with linear/exponential fitting
- **Photobleaching Correction**: Model based on exponential decay

### Advanced Visualization

- **Multi-channel Display**: Individual channel selection with ImageJ-style colormaps
- **Channel Merging**: Combine up to 3 channels with color mapping (green, magenta, yellow)
- **Interactive Navigation**: Frame-by-frame and Z-plane navigation with independent play controls per tab
- **Background Removal**: Segmentation-based background removal and overlay options
- **Trajectory Visualization**: Display particle trajectories with customizable colors and trail lengths
- **Export Options**: High-resolution images, videos (MP4/GIF), and interactive plots

### Data Export and Documentation

- **Multiple Formats**: PNG (300 DPI), TIFF, OME-TIFF, CSV, and video formats
- **Batch Export**: Select and export multiple analysis results simultaneously
- **Metadata Logging**: Complete parameter documentation for reproducibility
- **User Annotations**: Custom comments and observations integrated with exports

---

## GUI Tabs Overview

| Tab | Description |
|-----|-------------|
| Import | Load images, manage files, adjust display settings |
| Registration | Correct drift and align time-lapse images |
| Segmentation | Cell segmentation (Watershed, Cellpose, Manual, Import sub-tabs) |
| Photobleaching | Correct for fluorescence decay over time |
| Tracking | Detect spots, link trajectories, analyze clusters |
| MSD | Calculate diffusion coefficients from trajectories |
| Distribution | Histogram analysis of particle properties |
| Time Course | Temporal analysis of spot intensities |
| Correlation | Auto- and cross-correlation analysis |
| Coloc | Automated colocalization analysis |
| Coloc Edit | Manual verification of colocalization results |
| Visualization | Trajectory display and video export |
| Export | Batch export of all analysis results |

---

## Documentation

For comprehensive information about using MicroLive, please refer to the detailed documentation:

- **[User Guide](docs/user_guide.md)** — Complete guide to using the MicroLive GUI interface
- **[Tutorial](docs/tutorial.md)** — Step-by-step tutorials for all analysis workflows
- **[API Reference](docs/api_reference.md)** — Technical documentation for developers and advanced users

---

## Installation

To install this repository and all its dependencies, we recommend using [Anaconda](https://www.anaconda.com).

### Quick Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/ningzhaoAnschutz/microlive.git  
cd microlive 
```

#### 2. Create Environment from YAML File

**macOS (Apple Silicon M1/M2/M3):**

```bash
conda env create -f micro_mac.yml
```

**Windows with NVIDIA GPU:**

```bash
conda env create -f micro_windows.yml
```

> **Note:** The Windows environment uses PyTorch with CUDA 12.4 for GPU acceleration. Ensure you have NVIDIA drivers and CUDA installed. To verify GPU support after installation:
>
> ```python
> import torch
> print(f"CUDA available: {torch.cuda.is_available()}")
> print(f"GPU: {torch.cuda.get_device_name(0)}")
> ```

#### 3. Activate Environment

```bash
conda activate micro_mac        # macOS
conda activate micro_windows    # Windows
```

### Manual Installation

#### 1. Create Virtual Environment

```bash
conda create -n micro_mac python=3.12 -y
conda activate micro_mac
```

#### 2. Install Dependencies

**macOS:**

```bash
pip install -r requirements.txt
```

**Windows with NVIDIA GPU:**

```bash
pip install -r requirements_windows.txt
```

See [requirements.txt](requirements.txt) (macOS) and [requirements_windows.txt](requirements_windows.txt) (Windows GPU) for the complete dependency lists.

---

## Launching the GUI

### Platform-Specific Launchers

- **macOS**: Double-click `micro_mac.command` in the `gui` directory
- **Windows**: Double-click `micro_windows.bat` in the `gui` directory

### Manual Launch

1. **Activate Environment**:

   ```bash
   conda activate micro_mac
   ```

2. **Navigate to GUI Directory**:

   ```bash
   cd gui
   ```

3. **Launch Application**:

   ```bash
   python micro.py
   ```

### First-Time Setup

When launching MicroLive for the first time:

1. **Theme Selection**: Choose between dark and light themes (toggle in Display tab)
2. **Load Sample Data**: Use "Open File" to load your microscopy images
3. **Check Documentation**: Refer to the [User Guide](docs/user_guide.md) for detailed instructions
4. **Follow Tutorials**: Start with [Tutorial](docs/tutorial.md) for guided examples

---

## Project Structure

```text
microlive/
├── src/                          # Core source code
│   ├── microscopy.py             # Main analysis classes and functions
│   ├── imports.py                # Central import management
│   ├── ML_SpotDetection.py       # Machine learning spot detection
│   └── pipelines/                # Analysis pipeline modules
├── gui/                          # Graphical user interface
│   ├── micro.py                  # Main GUI application
│   ├── micro_mac.command         # macOS launcher script
│   └── micro_windows.bat         # Windows launcher script
├── docs/                         # Documentation
│   ├── user_guide.md             # Complete user manual
│   ├── tutorial.md               # Step-by-step tutorials
│   ├── api_reference.md          # Technical API documentation
│   └── icons/                    # Application icons
├── notebooks/                    # Example Jupyter notebooks
│   └── converter.ipynb           # Format conversion notebook
├── modeling/                     # Machine learning models
│   └── machine_learning/         # ML-based analysis tools
├── micro_mac.yml                 # Conda environment file (macOS GPU)
├── micro_windows.yml             # Conda environment file (Windows CUDA GPU)
├── requirements.txt              # Python dependencies (macOS)
├── requirements_windows.txt      # Python dependencies (Windows CUDA GPU)
├── LICENSE                       # GPL v3 License
└── README.md                     # This file
```

---

## Quick Start Workflows

### Basic Particle Tracking

1. **Load Data**: Open your microscopy file (.lif or .tif)
2. **Segmentation**: Define cell boundaries using Cellpose or manual segmentation
3. **Detection**: Set threshold and spot parameters in the Tracking tab
4. **Tracking**: Link particles across time frames
5. **MSD Analysis**: Calculate diffusion coefficients in the MSD tab
6. **Export**: Save tracking data and visualizations

<img src="docs/tracking_video_fast.gif" alt="Tracking" width="400" />

### Colocalization Analysis

1. **Complete Tracking**: Perform particle tracking on reference channel
2. **Select Channels**: Choose reference and target channels in Coloc tab
3. **Run Analysis**: Execute ML-based or intensity-based colocalization
4. **Manual Verification**: Review and refine results in Coloc Edit tab
5. **Export Results**: Save quantitative data and visualization matrices

<img src="docs/tracking_visualization_fast.gif" alt="Tracking Visualization" width="600" />

### Correlation Analysis

1. **Quality Tracking**: Ensure good quality particle trajectories
2. **Select Data**: Choose field and channels for analysis
3. **Configure Parameters**: Set fit type and lag ranges
4. **Run Analysis**: Execute auto- or cross-correlation
5. **Interpret Results**: Analyze correlation curves and fitted parameters

---

## Contributing

We welcome contributions to MicroLive.

### Reporting Issues

- Use the [GitHub Issues](https://github.com/ningzhaoAnschutz/microlive/issues) page
- Provide detailed descriptions and steps to reproduce
- Include system information and error messages

### Contributing Code

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes and add tests
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/new-feature`)
6. Create a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new functionality
- Update documentation as needed

---

## License

This project is licensed under the GNU General Public License v3 (GPLv3). See the [LICENSE](LICENSE) file for details.

---

## Support and Contact

- **Documentation**: [User Guide](docs/user_guide.md) | [API Reference](docs/api_reference.md) | [Tutorial](docs/tutorial.md)
- **Issues**: [GitHub Issues](https://github.com/ningzhaoAnschutz/microlive/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ningzhaoAnschutz/microlive/discussions)

For questions about specific research applications or collaborations, please contact the development team.

---

## Citation

If you use MicroLive in your research, please cite it as follows:

> **Luis U. Aguilera, William S. Raymond, Rhiannon M. Sears, Nathan L. Nowling, Brian Munsky, Ning Zhao.** *MicroLive: An Image Processing Toolkit for Quantifying Live-cell Single-Molecule Microscopy.* GitHub, 2025.  
> [https://github.com/ningzhaoAnschutz/microlive](https://github.com/ningzhaoAnschutz/microlive)  
> Licensed under the GPL v3 License.

### BibTeX Entry

```bibtex
@misc{Aguilera2025MicroLive,
  author       = {Aguilera, Luis U. and Raymond, William S. and Sears, Rhiannon M. and Nowling, Nathan L. and Munsky, Brian and Zhao, Ning},
  title        = {MicroLive: An Image Processing Toolkit for Quantifying Live-cell Single-Molecule Microscopy},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/ningzhaoAnschutz/microlive}},
  note         = {Licensed under the GPL v3 License},
  keywords     = {microscopy, single-molecule, particle tracking, colocalization, image analysis}
}
```

---

For step-by-step examples, see the [Tutorial](docs/tutorial.md).
