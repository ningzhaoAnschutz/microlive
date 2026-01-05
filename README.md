# MicroLive <img src="docs/icons/icon_micro.png" alt="Micro Logo" width="150" align="right" />

**Authors:** Luis U. Aguilera, William S. Raymond, Rhiannon M. Sears, Nathan L. Nowling, Brian Munsky, Ning Zhao

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/user_guide.md)
[![Tutorial](https://img.shields.io/badge/tutorial-step--by--step-orange.svg)](docs/tutorial.md)
[![API Reference](https://img.shields.io/badge/API-reference-blue.svg)](docs/api_reference.md)

## About

**MicroLive** is a Python-based GUI application for live-cell microscopy image analysis and single-molecule measurements. It provides an end-to-end workflow from image loading through particle tracking, colocalization analysis, and statistical analysis.

<img src="docs/Microlive_video_720_fast.gif" alt="MicroLive Demo" width="900" />

---

## Features

- **Multi-format image loading** (.lif, .tif, .ome.tif) with automatic metadata extraction
- **Image registration** for drift correction across time-lapse acquisitions
- **Cell segmentation** via Cellpose (GPU-accelerated), watershed, or manual drawing
- **Particle tracking** with customizable spot detection, trajectory linking, and clustering
- **Diffusion analysis** (MSD) with per-cell diffusion coefficient calculation
- **Colocalization** using ML-based or intensity-based methods with manual verification
- **Statistical analysis** including distributions, time courses, and correlation (auto/cross)
- **Photobleaching correction** with exponential decay modeling
- **Export** to PNG, TIFF, CSV, MP4/GIF with comprehensive metadata logging

---

## GUI Tabs

| Tab | Description |
|-----|-------------|
| Import | Load images, manage files, adjust display settings |
| Registration | Correct drift and align time-lapse images |
| Segmentation | Cell segmentation (Watershed, Cellpose, Manual, Import) |
| Photobleaching | Correct for fluorescence decay over time |
| Tracking | Detect spots, link trajectories, analyze clusters |
| MSD | Calculate diffusion coefficients from trajectories |
| Distribution | Histogram analysis of particle properties |
| Time Course | Temporal analysis of spot intensities |
| Correlation | Auto- and cross-correlation analysis |
| Coloc / Coloc Edit | Colocalization analysis and manual verification |
| Visualization | Trajectory display and video export |
| Export | Batch export of all analysis results |

---

## Documentation

- **[User Guide](docs/user_guide.md)** — Complete guide to using MicroLive
- **[Tutorial](docs/tutorial.md)** — Step-by-step tutorials for all workflows
- **[API Reference](docs/api_reference.md)** — Technical documentation for developers

---

## Installation

We recommend using [Anaconda](https://www.anaconda.com) for environment management.

### 1. Clone the Repository

```bash
git clone https://github.com/ningzhaoAnschutz/microlive.git  
cd microlive 
```

### 2. Create Environment

**macOS (Apple Silicon):**

```bash
conda env create -f micro_mac.yml
conda activate micro_mac
```

**Windows (NVIDIA GPU):**

```bash
conda env create -f micro_windows.yml
conda activate micro_windows
```

### Alternative: Pip Installation

```bash
# macOS
pip install -r requirements_mac.txt

# Windows (with CUDA GPU support)
pip install -r requirements_windows.txt
```

---

## Launching MicroLive

**Platform launchers:**

- macOS: Double-click `gui/micro_mac.command`
- Windows: Double-click `gui/micro_windows.bat`

**Manual launch:**

```bash
conda activate micro_mac  # or micro_windows
cd gui
python micro.py
```

---

## Project Structure

```text
microlive/
├── src/                          # Core analysis library
│   ├── microscopy.py             # Main analysis classes
│   ├── imports.py                # Central import management
│   ├── ML_SpotDetection.py       # ML-based spot detection
│   └── pipelines/                # Analysis pipeline modules
├── gui/                          # GUI application
│   ├── micro.py                  # Main GUI application
│   ├── micro_mac.command         # macOS launcher
│   └── micro_windows.bat         # Windows launcher
├── docs/                         # Documentation
│   ├── user_guide.md             # User manual
│   ├── tutorial.md               # Step-by-step tutorials
│   └── api_reference.md          # API documentation
├── modeling/                     # Machine learning models
├── notebooks/                    # Example Jupyter notebooks
├── micro_mac.yml                 # Conda environment (macOS)
├── micro_windows.yml             # Conda environment (Windows GPU)
├── requirements_mac.txt          # Pip dependencies (macOS)
└── requirements_windows.txt      # Pip dependencies (Windows GPU)
```

---

## Contributing

We welcome contributions! Please submit issues and pull requests via [GitHub](https://github.com/ningzhaoAnschutz/microlive).

---

## License

This project is licensed under the GNU General Public License v3 (GPLv3). See [LICENSE](LICENSE) for details.

---

## Citation

If you use MicroLive in your research, please cite:

> **Aguilera LU, Raymond WS, Sears RM, Nowling NL, Munsky B, Zhao N.** *MicroLive: An Image Processing Toolkit for Quantifying Live-cell Single-Molecule Microscopy.* GitHub, 2025.  
> [https://github.com/ningzhaoAnschutz/microlive](https://github.com/ningzhaoAnschutz/microlive)

```bibtex
@misc{Aguilera2025MicroLive,
  author       = {Aguilera, Luis U. and Raymond, William S. and Sears, Rhiannon M. and Nowling, Nathan L. and Munsky, Brian and Zhao, Ning},
  title        = {MicroLive: An Image Processing Toolkit for Quantifying Live-cell Single-Molecule Microscopy},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/ningzhaoAnschutz/microlive}},
  note         = {Licensed under GPL v3}
}
```

---

## Support

- **Documentation**: [User Guide](docs/user_guide.md) | [Tutorial](docs/tutorial.md) | [API Reference](docs/api_reference.md)
- **Issues**: [GitHub Issues](https://github.com/ningzhaoAnschutz/microlive/issues)
