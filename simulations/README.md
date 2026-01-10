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

# Run validation tests
cd tests
python run_validation_test.py
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
| **Ground Truth Quality** | Data integrity, 0% background particles |
| **Colocalization** | Ch1/Ch2 percentage recovery + distance verification |
| **Photobleaching** | Decay rate (k) recovery accuracy |
| **Spot Detection** | True positive rate using BigFISH detector |
| **Position Accuracy** | Sub-pixel position error via tracking |
| **Compartment Assignment** | Nucleus vs cytosol classification accuracy |
| **MSD Recovery** | Diffusion coefficient (D) from ParticleMotion |

## File Structure

```
simulations/
├── run_simulation.py        # Main simulation script
├── spot_simulator.py        # Core simulator classes
├── visualize_results.py     # Generate PNG visualizations
├── config_simple.yaml       # Single-cell configuration
├── config_multicell.yaml    # Multi-cell configuration
├── IMPLEMENTATION_PLAN.md   # Detailed design document
├── README.md               # This file
├── tests/
│   ├── run_validation_test.py  # Automated test suite
│   └── test_report.md          # Generated test results
├── results/                 # Default single-cell output
│   ├── simulated_spots.tif
│   ├── mask_cytosol.tif
│   ├── mask_nucleus.tif
│   ├── ground_truth.csv
│   └── viz/                 # Visualization PNGs
└── results_multicell/       # Multi-cell output
```

## Output Files

| File | Format | Description |
| :--- | :---: | :--- |
| `simulated_spots.tif` | TIFF [TCZYX] | Multi-channel 3D+T image stack |
| `mask_cytosol.tif` | TIFF [ZYX] | Labeled cell mask (pixel = cell_id) |
| `mask_nucleus.tif` | TIFF [ZYX] | Labeled nucleus mask (pixel = cell_id) |
| `mask_cytosol_no_nuclei.tif` | TIFF [ZYX] | Cytosol-only mask (excluding nucleus) |
| `ground_truth.csv` | CSV | All spot positions, intensities, compartments |
| `ground_truth.parquet` | Parquet | Same data in efficient binary format |
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
  snr_mean: 3.0              # Signal-to-Noise Ratio
  size_mean: 1.5             # PSF sigma (pixels)

# Diffusion
particle_motion:
  diffusion_coefficient: 0.05  # D in px²/frame

# Colocalization probabilities
colocalization:
  ch1_probability: 0.7       # 70% of Ch0 spots have Ch1 partner
  ch2_probability: 0.3       # 30% of Ch0 spots have Ch2 partner

# Photobleaching (exponential decay rates)
photobleaching:
  ch0_decay_rate: 0.00037    # ~20% loss over 600s
  ch1_decay_rate: 0.00060    # ~30% loss over 600s
  ch2_decay_rate: 0.00085    # ~40% loss over 600s

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
  per_cell_counts: [20, 15, 10, 5]  # Different counts per cell

# Cell geometry
cell_geometry:
  num_cells: 4                      # 4 cells in 2x2 grid
  layout: 'grid'
  grid_spacing_yx: [230, 230]
  cell_diameter_yx: [180, 180]
  nucleus_diameter_yx: [70, 60]
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
| `frame0_rgb.png` | RGB composite (G=Ch0, R=Ch1, B=Ch2) |
| `frame0_ch0.png` | Channel 0 max projection |
| `mask_cytosol.png` | Cell masks (color = cell ID) |
| `mask_nucleus.png` | Nucleus masks |
| `trajectories.png` | Particle movement paths |
| `compartment_distribution.png` | Pie chart of cytosol/nucleus |
| `colocalization_rates.png` | Bar chart of coloc percentages |
| `per_cell_stats.png` | Per-cell particle/coloc stats (multi-cell) |

## Validation Tests

### Running Tests

```bash
cd tests

# Test single-cell simulation
python run_validation_test.py

# Test multi-cell simulation
python run_validation_test.py --sim-dir ../results_multicell --config ../config_multicell.yaml

# View results
cat test_report.md
```

### Test Results Example

```
============================================================
VALIDATION SUMMARY
============================================================

  ✅ Passed: 8
  ❌ Failed: 0
  📊 Total:  8
    ✅ Ground Truth Quality
    ✅ Colocalization
    ✅ Photobleaching
    ✅ Spot Detection
    ✅ Position Accuracy
    ✅ Compartment Assignment
    ✅ MSD Recovery
    ✅ GUI Pipeline

  Overall: ✅ PASS
```

### Individual Test Details

| Test | Pass Criteria | Typical Results |
| :--- | :--- | :--- |
| Ground Truth | 0% in background | 0.0% ✅ |
| Colocalization | ≤20% error | <5% error |
| Photobleaching | ≤30% error on k | 2-3% error |
| Spot Detection | ≥80% true positive | 100% matched |
| Position | ≥50% recall or <5px error | 0.17px error |
| Compartment | ≥85% accuracy | 88-100% |
| MSD | ≤50% error on D | 2-4% error |
| GUI Pipeline | ≥50% recall, ≥80% compartment | 92% recall |

## GUI Output Validation

After running analysis in the MicroLive GUI and exporting results, you can validate
that the GUI correctly recovered simulation parameters:

```bash
cd tests

# Validate GUI output against ground truth
python validate_gui_output.py \
    --gui-dir ../results_simulated_spots \
    --gt-dir ../results_multicell \
    --config ../config_multicell.yaml
```

### GUI Validation Tests

| Test | What It Compares |
| :--- | :--- |
| **Photobleaching** | GUI decay rates vs config values |
| **Colocalization** | GUI POOLED % vs config probability |
| **Tracking** | GUI spot positions vs ground truth |
| **MSD** | GUI D coefficient vs config (with voxel scaling) |

### Example GUI Validation Output

```
============================================================
VALIDATION SUMMARY
============================================================

  ✅ Passed: 4
  ❌ Failed: 0
    ✅ Photobleaching (1.5-1.9% error)
    ✅ Colocalization (0.2% error)
    ✅ Tracking (89% recall, 0.30 px error)
    ✅ MSD (10.8% error after voxel scaling)

  Overall: ✅ PASS
```

## Photobleaching Formula

Intensity decay follows: `I(t) = I₀ × exp(-k × t)`

To calculate decay rate for a target percentage loss:

```
k = -ln(remaining_fraction) / t_seconds

Examples:
- 20% loss: k = -ln(0.80)/600 = 0.00037 s⁻¹
- 30% loss: k = -ln(0.70)/600 = 0.00060 s⁻¹
- 40% loss: k = -ln(0.60)/600 = 0.00085 s⁻¹
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

## Documentation

See `IMPLEMENTATION_PLAN.md` for detailed design, phase roadmap, and validation checklist.
