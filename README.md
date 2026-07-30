# MicroLive <img src="https://raw.githubusercontent.com/ningzhaoAnschutz/microlive/main/docs/icons/icon_micro.png" alt="Micro Logo" width="150" align="right" />

**Authors:** Luis U. Aguilera, William S. Raymond, Rhiannon M. Sears, Nathan L. Nowling, Brian Munsky, Ning Zhao

[![PyPI version](https://img.shields.io/pypi/v/microlive.svg)](https://pypi.org/project/microlive/) [![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/user_guide.md) [![Tutorial](https://img.shields.io/badge/tutorial-step--by--step-orange.svg)](docs/tutorial.md) [![API Reference](https://img.shields.io/badge/API-reference-blue.svg)](docs/api_reference.md) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/) [![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)

## About

**MicroLive** is a Python-based GUI application for live-cell microscopy image analysis and single-molecule measurements. It provides an end-to-end workflow from image loading through particle tracking, colocalization analysis, and statistical analysis.

[![MicroLive Demo Video](https://img.youtube.com/vi/GU1ct5go-AU/maxresdefault.jpg)](https://youtu.be/GU1ct5go-AU?si=wEJWUaSYgNGj-Pi3)

*Click the image above to watch the demo video on YouTube*

---

## Features

- **Image I/O**: Load .lif, .czi, .tif, .ome.tif with metadata extraction and dimension mapping
- **Registration**: Drift correction via phase correlation
- **Segmentation**: Cellpose (GPU), watershed, manual ROI, or external mask import
- **Photobleaching correction**: Exponential decay modeling
- **Particle tracking**: 2D (TrackPy) and 3D (Big-FISH) detection with multi-channel support
- **Automated threshold detection**: Hybrid Big-FISH/TrueSpot method with fixed-threshold mode for inhibitor experiments
- **Trajectory linking**: Nearest-neighbor with memory and cluster analysis
- **Intensity quantification**: Background subtraction, PSF fitting, SNR calculation
- **Colocalization**: CNN-based, distance-based, and manual verification
- **MSD analysis**: Per-cell diffusion coefficient calculation
- **Correlation**: Auto- and cross-correlation with exponential/linear fitting
- **Export**: PNG, TIFF, CSV, MP4/GIF with full metadata logging
- **Filtered LIF export**: Export analyzed scenes once as renamed OME-TIFF/AVI files with an original-to-new-name manifest

---

## Documentation

- **[User Guide](docs/user_guide.md)** — Complete guide to using MicroLive
- **[Tutorial](docs/tutorial.md)** — Step-by-step tutorials for all workflows
- **[API Reference](docs/api_reference.md)** — Technical documentation for developers

---

## Installation

### Quick Install (Recommended)

```bash
# Create and activate conda environment
conda create -n microlive python=3.10 -y
conda activate microlive

# Install MicroLive
pip install microlive

# Launch the GUI
microlive
```

That's it! MicroLive will launch with GPU acceleration automatically enabled:

- **macOS (Apple Silicon)**: MPS GPU acceleration works automatically
- **Linux/Windows (CPU)**: Works out of the box

---

### GPU Acceleration for NVIDIA (Windows/Linux)

For NVIDIA GPU acceleration, install PyTorch with CUDA support **before** installing MicroLive:

```bash
# Create environment
conda create -n microlive python=3.10 -y
conda activate microlive

# Install PyTorch with CUDA 12.4 (adjust version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Then install MicroLive
pip install microlive

# Launch
microlive
```

### Using Conda Environment Files

Pre-configured environment files are available:

```bash
# macOS / CPU
conda env create -f installation/microlive.yml
conda activate microlive

# Windows/Linux with NVIDIA GPU
conda env create -f installation/microlive_cuda.yml
conda activate microlive
```

### Verify GPU Support

```python
from microlive.utils.device import check_gpu_status
check_gpu_status()
```

Expected output:

```
PyTorch version: 2.x.x
✅ CUDA available: NVIDIA GeForce RTX ...
   Memory: 8.0 GB
```

Or for Apple Silicon:

```
PyTorch version: 2.x.x
✅ MPS available: Apple Silicon GPU (MPS)
```

---

### Development Installation

For developers who want to modify the source code:

```bash
# Clone the repository
git clone --depth 1 https://github.com/ningzhaoAnschutz/microlive.git
cd microlive

# Create environment
conda create -n microlive python=3.10 -y
conda activate microlive

# Install in editable mode
pip install -e .

# Launch
microlive
```

---

### Troubleshooting Installation

If you encounter errors during installation (especially on Python 3.12):

```bash
# Option 1: Upgrade pip and pre-install numpy
pip install --upgrade pip setuptools wheel
pip install numpy
pip install microlive

# Option 2: Use Python 3.10 (most compatible)
conda create -n microlive python=3.10 -y
conda activate microlive
pip install microlive
```

**Common error**: `ModuleNotFoundError: No module named 'numpy'` during `pystackreg` build

- This occurs when pip tries to build older package versions from source
- Solution: Pre-install NumPy before installing MicroLive

---

## Usage

### GUI Application

```bash
conda activate microlive
microlive
```

### Programmatic API

```python
import microlive.microscopy as mi

# Load images from a Leica .lif file
reader = mi.ReadLif("experiment.lif")
(list_images, list_names, pixel_XY, pixel_Z,
 channel_names, num_channels, *_) = reader.read()

# Access image data
image = list_images[0]
print(f"Image shape: {image.shape}")  # (T, Z, Y, X, C)

# Run Cellpose segmentation
cellpose = mi.Cellpose(image=image[0, 0, :, :, 0])
masks = cellpose.calculate_masks()

# Spot detection
spots = mi.SpotDetection(
    image=image[0],  # Single time point [Z, Y, X, C]
    channels_spots=[0],
    channels_cytosol=[],
    channels_nucleus=[],
    masks_complete_cells=masks,
    yx_spot_size_in_px=5,
    threshold_for_spot_detection=100
)
```

---

## Project Structure

```text
microlive/
├── microlive/                    # Core package (pip installable)
│   ├── microscopy.py             # Main analysis classes
│   ├── imports.py                # Central import management
│   ├── ml_spot_detection.py      # ML-based spot detection
│   ├── gui/                      # GUI application
│   │   ├── app.py                # Main GUI window
│   │   └── main.py               # CLI entry point
│   ├── utils/                    # Utility modules
│   │   ├── device.py             # GPU detection
│   │   ├── model_downloader.py   # ML model provisioning
│   │   └── resources.py          # Resource paths
│   ├── data/                     # ML models and resources
│   └── pipelines/                # Analysis pipeline modules
├── simulations/                  # Spot simulation & validation (not in pip)
│   ├── spot_simulator.py         # Multi-cell simulation engine
│   ├── run_simulation.py         # CLI runner
│   ├── visualize_results.py      # Visualization tools
│   ├── config_*.yaml             # Configuration examples
│   └── tests/                    # 9-test validation suite (100% pass)
├── docs/                         # Documentation
│   ├── user_guide.md             # User manual
│   ├── tutorial.md               # Step-by-step tutorials
│   └── api_reference.md          # API documentation
├── modeling/                     # Research/development (not in pip package)
├── notebooks/                    # Example Jupyter notebooks
├── installation/                 # Environment files
│   ├── microlive.yml             # Conda env (macOS / CPU)
│   └── microlive_cuda.yml        # Conda env (NVIDIA GPU)
├── pyproject.toml                # Package configuration
└── LICENSE                       # GPL v3 License
```

---

## License

This project is licensed under the GNU General Public License v3 (GPLv3). See [LICENSE](LICENSE) for details.

---

## Citation

If you use MicroLive in your research, please cite:

> **Aguilera LU, Raymond WS, Sears RM, Nowling NL, Munsky B, Zhao N.** *MicroLive: An Image Processing Toolkit for Quantifying Live-cell Single-Molecule Microscopy.* Bioinformatics Advances, 2026; vbag095.
> DOI: [10.1093/bioadv/vbag095](https://doi.org/10.1093/bioadv/vbag095)

```bibtex
@article{aguilera2026microlive,
  title={MicroLive: An Image Processing Toolkit for Quantifying Live-cell Single-Molecule Microscopy},
  author={Aguilera, Luis U and Raymond, William S and Sears, Rhiannon M and Nowling, Nathan L and Munsky, Brian and Zhao, Ning},
  journal={Bioinformatics Advances},
  pages={vbag095},
  year={2026},
  doi={10.1093/bioadv/vbag095}
}
```

---

## Support

- **Documentation**: [User Guide](docs/user_guide.md) | [Tutorial](docs/tutorial.md) | [API Reference](docs/api_reference.md)
- **Issues & Contributions**: [GitHub](https://github.com/ningzhaoAnschutz/microlive/issues)
