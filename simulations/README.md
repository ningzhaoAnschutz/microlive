# MicroLive Spot Simulation

Generate synthetic microscopy images with controlled spot dynamics for MicroLive validation.

## Overview

This simulation framework creates realistic 3D microscopy images with:

- Multi-channel fluorescent spots with known positions and intensities
- Realistic cell geometry (cytosol + nucleus compartments)
- Brownian diffusion with configurable coefficients
- Exponential photobleaching per channel
- Colocalization between channels at configurable rates

The simulation provides ground truth data that can be compared against MicroLive's analysis tools to validate spot detection, tracking, photobleaching correction, MSD analysis, and compartment assignment.

## Quick Start

```bash
# Activate MicroLive environment
conda activate microlive

# Run single-cell simulation (default)
python run_simulation.py

# Run multi-cell simulation
python run_simulation.py --config config_multicell.yaml --output results_multicell

# Generate visualizations
python visualize_results.py --sim-dir results

# Run validation tests against ground truth
cd tests
python run_test.py

# Run GUI output validation (after analyzing in MicroLive GUI)
python run_test_gui.py
```

## Features

### Simulation Capabilities

| Feature | Description |
| :--- | :--- |
| **3-Channel Images** | Lead channel (Ch0) + two colocalized channels (Ch1, Ch2) |
| **Dual Compartments** | Cytosol and nucleus with separate intensity baselines |
| **Brownian Motion** | Configurable diffusion coefficient (D in px²/frame) |
| **Photobleaching** | Per-channel exponential decay rates |
| **Colocalization** | Independent probabilities for Ch1 and Ch2 colocalization |
| **Multi-Cell** | Support for multiple cells with per-cell particle counts |
| **Ground Truth** | Complete trajectory data for validation |

### Validation Tests

The automated test suite validates MicroLive's recovery of simulation parameters:

| Test | What It Validates |
| :--- | :--- |
| **Ground Truth Quality** | Data integrity, <10% background particles |
| **Colocalization** | Ch1/Ch2 percentage recovery + distance verification |
| **Photobleaching** | Decay rate (k) recovery accuracy |
| **Spot Detection** | True positive rate using BigFISH detector |
| **Position Accuracy** | Sub-pixel position error via tracking |
| **Compartment Assignment** | Nucleus vs cytosol classification accuracy |
| **MSD Recovery** | Diffusion coefficient (D) from ParticleMotion |
| **Colocalization Recovery** | ML-based colocalization detection (CNN classifier) |
| **GUI Syntax** | Python syntax validation for GUI module |

## File Structure

```text
simulations/
├── spot_simulator.py           # Core simulator (SpotSimulator, Particle, CellRegion)
├── run_simulation.py           # CLI entry point
├── visualize_results.py        # Generate PNG visualizations
├── config_simple.yaml          # Single-cell configuration
├── config_multicell.yaml       # Multi-cell configuration (4 cells)
├── __init__.py                 # Package init
├── README.md                   # This file
├── tests/
│   ├── run_test.py             # Programmatic API validation (9 tests)
│   ├── run_test_gui.py         # GUI export validation (7 tests)
│   ├── helpers.py              # Shared utilities (config, cell matching, reporting)
│   ├── README.md               # Tests documentation
│   ├── report.md               # Generated API validation report
│   └── gui_report.md           # Generated GUI validation report
├── results_single_cell/        # Single-cell ground truth (default for run_test.py)
│   ├── simulated_spots.tif
│   ├── mask_cytosol.tif
│   ├── mask_cytosol_no_nuclei.tif
│   ├── mask_nucleus.tif
│   ├── ground_truth.csv
│   ├── simulation_metadata.txt
│   └── viz/                    # Visualization PNGs
├── results_single_cell_gui/    # Single-cell GUI export (default for run_test_gui.py)
│   ├── Metadata_simulated_spots.txt
│   ├── tracking_simulated_spots.csv
│   ├── cellpose_cytosol_simulated_spots.tif
│   ├── cellpose_nucleus_simulated_spots.tif
│   ├── colocalization_data_simulated_spots.csv
│   ├── msd_dataframe_simulated_spots.csv
│   └── *.png                   # Exported plots
├── results_multicell/          # Multi-cell ground truth
│   ├── simulated_spots.tif
│   ├── mask_cytosol.tif
│   ├── mask_nucleus.tif
│   └── ground_truth.csv
└── kk_results_simulated_spots/ # Multi-cell GUI export
    ├── Metadata_simulated_spots.txt
    ├── tracking_simulated_spots.csv
    └── ...
```

## Output Files

| File | Format | Description |
| :--- | :---: | :--- |
| `simulated_spots.tif` | TIFF [TCZYX] | Multi-channel 3D+T image stack |
| `mask_cytosol.tif` | TIFF [ZYX] | Labeled cell mask (pixel = cell_id) |
| `mask_nucleus.tif` | TIFF [ZYX] | Labeled nucleus mask (pixel = cell_id) |
| `mask_cytosol_no_nuclei.tif` | TIFF [ZYX] | Cytosol-only mask (excluding nucleus) |
| `ground_truth.csv` | CSV | All spot positions, intensities, compartments |
| `ground_truth.parquet` | Parquet | Same data in efficient binary format (conditional) |
| `simulation_metadata.txt` | Text | Configuration and timing log |

## Configuration

### Single-Cell (`config_simple.yaml`)

```yaml
# Image dimensions
image:
  size_yx: [512, 512]
  num_z_slices: 10
  voxel_size_yx_nm: 130.0
  voxel_size_z_nm: 300.0

# Simulation timing
simulation:
  total_time_seconds: 600    # 600s = 120 frames at 5s
  frame_rate_seconds: 5
  random_seed: 42

# Particle properties
particles:
  average_count: 50          # Total particles in cell

spot_properties:
  snr_mean: 3.5              # Signal-to-Noise Ratio
  size_mean: 1.5             # PSF sigma (pixels)

# Diffusion
motion:
  diffusion_coefficient: 0.05  # D in px²/frame

# Colocalization probabilities
colocalization:
  ch1_probability: 0.8       # 80% of Ch0 spots have Ch1 partner
  ch2_probability: 0.6       # 60% of Ch0 spots have Ch2 partner
  ch1_snr_multiplier: 1.0    # Same brightness as Ch0
  ch2_snr_multiplier: 1.0    # Same brightness as Ch0

# Photobleaching (exponential decay rates)
photobleaching:
  ch0_decay_rate: 0.00037    # ~20% loss over 600s
  ch1_decay_rate: 0.00060    # ~30% loss over 600s
  ch2_decay_rate: 0.000718   # ~35% loss over 600s

# Noise model (per-channel Gaussian noise)
noise:
  ch0_noise_std: 300.0   # Channel 0 noise std
  ch1_noise_std: 300.0   # Channel 1 noise std
  ch2_noise_std: 300.0   # Channel 2 noise std

# Intensity baselines
baseline:
  outside_cell: 1000
  cytosol: 1600
  nucleus: 1800
```

### Multi-Cell (`config_multicell.yaml`)

```yaml
# Per-cell particle counts
particles:
  per_cell_counts: [50, 35, 40, 25]  # Different counts per cell

# Cell geometry
cell_geometry:
  num_cells: 4                      # 4 cells in 2x2 grid
  layout: 'grid'
  grid_spacing_yx: [230, 230]
  cell_diameter_yx: [180, 180]
  nucleus_diameter_yx: [70, 60]

# Colocalization (same as single-cell)
colocalization:
  ch1_probability: 0.8
  ch2_probability: 0.6
  ch1_snr_multiplier: 1.0
  ch2_snr_multiplier: 1.0
```

## Running Simulations

### Basic Usage

```bash
# Default single-cell simulation
python run_simulation.py

# Custom configuration
python run_simulation.py --config my_config.yaml

# Custom output directory
python run_simulation.py --output my_results

# Multi-cell simulation
python run_simulation.py --config config_multicell.yaml --output results_multicell
```

### Programmatic Usage

```python
from spot_simulator import SpotSimulator

# Create simulator
sim = SpotSimulator('config_simple.yaml')

# Run simulation
image, ground_truth = sim.run()

# Save results
sim.save_results('results')
```

## Visualization

Generate PNG images for quick inspection:

```bash
# Single-cell results
python visualize_results.py

# Multi-cell results
python visualize_results.py --sim-dir results_multicell
```

### Generated Visualizations

| File | Description |
| :--- | :--- |
| `frame0_ch{ch}.png` | Start frame max projection per channel |
| `frame{t}_rgb.png` | RGB composite for start, middle, and end frames |
| `mask_cytosol.png` | Cell masks (color = cell ID) |
| `mask_nucleus.png` | Nucleus masks |
| `trajectories.png` | Particle movement paths |
| `compartment_distribution.png` | Pie chart of cytosol/nucleus |
| `colocalization_rates.png` | Bar chart of coloc percentages |
| `per_cell_stats.png` | Per-cell particle/coloc stats (multi-cell) |

## Validation Tests

### Programmatic API Tests (`run_test.py`)

Tests MicroLive's Python API directly against simulation ground truth:

```bash
cd tests

# Test single-cell simulation (default)
python run_test.py

# Test multi-cell simulation
python run_test.py --sim-dir ../results_multicell --config ../config_multicell.yaml

# View results
cat report.md
```

### Test Results Example

```text
============================================================
VALIDATION SUMMARY
============================================================

  ✅ Passed: 9
  ❌ Failed: 0
  ⚠️ Skipped: 0
  📊 Total:  9
    ✅ Ground Truth Quality
    ✅ Colocalization
    ✅ Photobleaching
    ✅ Spot Detection
    ✅ Position Accuracy
    ✅ Compartment Assignment
    ✅ MSD Recovery
    ✅ Colocalization Recovery
    ✅ GUI Syntax

  Overall: ✅ PASS
```

### Individual Test Details

| Test | Pass Criteria | Typical Results |
| :--- | :--- | :--- |
| Ground Truth | <10% in background | 0.0% ✅ |
| Colocalization | ≤25% error | <5% error |
| Photobleaching | ≤30% error on k | 22-25% error |
| Spot Detection | ≥80% true positive | 95% matched |
| Position | ≥50% recall or <5px error | 2.6px error |
| Compartment | ≥75% accuracy | 80-97% |
| MSD | ≤50% error on D | <5% error |
| Colocalization Recovery | ≤25% error (ML) | <22% error |
| GUI Syntax | Compiles without errors | ✅ |

## GUI Output Validation

After running analysis in the MicroLive GUI and exporting results, validate
that the GUI correctly recovered simulation parameters.

### Default: Single-Cell Validation

By default, `run_test_gui.py` validates the single-cell GUI export:

```bash
cd tests

# Uses defaults: --gui-dir ../results_single_cell_gui
#                --gt-dir  ../results_single_cell
#                --config  ../config_simple.yaml
python run_test_gui.py
```

### Multi-Cell Validation

To validate a multi-cell GUI export, specify the directories explicitly:

```bash
python run_test_gui.py \
    --gui-dir ../kk_results_simulated_spots \
    --gt-dir ../results_multicell \
    --config ../config_multicell.yaml
```

### GUI Validation Tests

| Test | What It Compares | Pass Criteria |
| :--- | :--- | :--- |
| **Segmentation** | GUI cell count vs config `num_cells` | Exact match |
| **Spot Count per Cell** | GUI particles vs ground truth (centroid-based cell matching) | ≤60% error per cell |
| **Compartment Assignment** | GUI nucleus/cytosol vs ground truth labels | ≥70% accuracy |
| **Photobleaching** | GUI decay rates vs config values | ≤30% error per channel |
| **MSD** | GUI D coefficient vs config (with voxel scaling) | ≤80% error |
| **Colocalization** | GUI POOLED % vs ground truth `has_ch1_partner` | ≤25% absolute error |
| **is_colocalized Tracking** | Per-particle `is_colocalized` vs ground truth `has_ch1_partner` | ≥75% accuracy |

> **Note:** The `is_colocalized Tracking` test requires that colocalization analysis
> is run in the GUI *before* exporting tracking data. If the column is absent, the
> test is skipped.

### Example GUI Validation Output (Single-Cell)

```text
============================================================
VALIDATION SUMMARY
============================================================

  ✅ Passed: 6
  ❌ Failed: 1
  ⚠️ Skipped: 0
  📊 Total:  7
    ✅ Segmentation (1/1 cells)
    ✅ Spot Count per Cell (32.0% error)
    ✅ Compartment Assignment (99.8% accuracy)
    ❌ Photobleaching (Ch2 at 36.2% > 30% threshold)
    ✅ MSD (14.0% error)
    ✅ Colocalization (9.3% error)
    ✅ is_colocalized Tracking (100.0% accuracy)

  Overall: ❌ FAIL
```

### Cell Matching Strategy

The GUI test uses **centroid-based cell matching** to handle the fact that cell labels
may differ between the GUI export and ground truth:

1. Load cytosol masks from both GUI and ground truth
2. Compute centroids for each cell label in each mask
3. Match by nearest centroid (within 50px)
4. Align GUI tracking cell IDs (0-indexed) to mask labels (1-indexed) via +1 offset

## Photobleaching Formula

Intensity decay follows: `I(t) = I₀ × exp(-k × t)`

To calculate decay rate for a target percentage loss:

```text
k = -ln(remaining_fraction) / t_seconds

Examples:
- 20% loss: k = -ln(0.80)/600 = 0.00037 s⁻¹
- 30% loss: k = -ln(0.70)/600 = 0.00060 s⁻¹
- 35% loss: k = -ln(0.65)/600 = 0.000718 s⁻¹
```

## Comparing with MicroLive Analysis

```python
import tifffile
import pandas as pd
import numpy as np
from microlive import microscopy as mi

# Load simulation output
image = tifffile.imread('results/simulated_spots.tif')
image_tzyxc = np.moveaxis(image, 1, -1)  # TCZYX -> TZYXC
df_gt = pd.read_csv('results/ground_truth.csv')

# Run MicroLive photobleaching analysis
pb = mi.Photobleaching(image_TZYXC=image_tzyxc, time_interval_seconds=5.0)
decay_params = pb.calculate_photobleaching()
# Returns [k0, I0_ch0, k1, I0_ch1, k2, I0_ch2]

# Compare to ground truth
print(f"Config k0: 0.00037, Measured k0: {decay_params[0]:.6f}")

# Run MicroLive MSD analysis
ch0 = df_gt[df_gt['spot_type'] == 0]
pm = mi.ParticleMotion(
    trackpy_dataframe=ch0[['frame', 'particle', 'x', 'y']],
    microns_per_pixel=0.130,
    step_size_in_sec=5.0,
    show_plot=False
)
D_measured, r2 = pm.calculate_msd()
print(f"Config D: 0.000169 µm²/s, Measured D: {D_measured:.6f} µm²/s")
```

## Ground Truth Schema

The `ground_truth.csv` contains:

| Column | Type | Description |
| :--- | :---: | :--- |
| `frame` | int | Time frame index |
| `z`, `y`, `x` | float | 3D position in pixels |
| `particle` | int | Unique particle ID |
| `cell_id` | int | Cell ID (for multi-cell) |
| `spot_type` | int | Channel (0=lead, 1=coloc1, 2=coloc2) |
| `is_nuc` | bool | True if in nucleus |
| `compartment` | str | 'cytosol' or 'nucleus' |
| `has_ch1_partner` | bool | Ch1 colocalization |
| `has_ch2_partner` | bool | Ch2 colocalization |
| `snr_ch_0` | float | Signal-to-noise ratio |
| `psf_amplitude_ch_0` | float | Peak intensity |
| `local_background` | float | Background at position |

## Dependencies

Uses existing MicroLive packages (no additional installation required):

- numpy, scipy, pandas
- tifffile, pyyaml
- scikit-image
- matplotlib

Optional:

- pyarrow (for parquet export)
