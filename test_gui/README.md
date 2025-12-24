# MicroLive GUI Performance Test

## Overview

This test validates MicroLive GUI outputs against ground truth from VirtualCell simulation. It verifies that the GUI correctly processes simulated microscopy data through segmentation, photobleaching correction, and particle tracking.

## Test Workflow

1. **Manual GUI Processing** (user performs):
   - Load simulated image: `simulated_microscopy.tif`
   - Segment cytosol and nucleus with Cellpose
   - Apply photobleaching correction
   - Detect and track particles (Channel 1 = Mature RNA)
   - Export results to `test_gui/results_*` folder

2. **Automated Test** (script runs):
   - Compare GUI outputs against ground truth
   - Generate test report with PASS/FAIL status

## Test Cases

| Test | Description | Threshold |
|------|-------------|-----------|
| **Mask IoU** | Cellpose masks vs ground truth masks | IoU ≥ 0.80 |
| **Photobleaching Rates** | GUI-fitted decay rates vs simulation values | ≤20% error |
| **Spot Count** | Average mRNA spots/frame vs ground truth | ≤40% error |
| **Intensity** | Ch1 spot intensities comparison | Report only |

## Ground Truth Values

From `simulation_metadata.txt`:

| Parameter | Value |
|-----------|-------|
| Avg Mature RNA per Frame | 136.1 spots |
| RNA Diffusion Coefficient | 0.5 px²/s |
| Photobleaching Ch0 (TS) | 0.00016 s⁻¹ |
| Photobleaching Ch1 (RNA) | 0.00012 s⁻¹ |
| Photobleaching Ch2 (Nascent Prot) | 0.00016 s⁻¹ |
| Photobleaching Ch3 (Mature Prot) | 0.0002 s⁻¹ |

## File Paths

### Ground Truth

```
/Users/nzlab-la/Desktop/Github/virtual_cell/spatio_temporal_models/results_simulation/
├── simulated_microscopy.tif       # Input image
├── mask_cytosol.tif               # GT cytosol mask
├── mask_nucleus.tif               # GT nucleus mask
├── ground_truth_mature_rna.csv    # GT spot positions
└── simulation_metadata.txt        # GT parameters
```

### GUI Results

```
/Users/nzlab-la/Desktop/microlive/test_gui/
├── README.md                      # This file
├── run_gui_test.py                # Test script
├── test_report.md                 # Generated report
└── results_simulated_microscopy_simulated_microscopy/
    ├── cellpose_cytosol_*.tif     # GUI cytosol mask
    ├── cellpose_nucleus_*.tif     # GUI nucleus mask
    ├── tracking_*.csv             # GUI tracking results
    └── Metadata_*.txt             # GUI metadata
```

## Running the Test

```bash
cd /Users/nzlab-la/Desktop/microlive/test_gui
python run_gui_test.py
```

The test report will be saved to `test_report.md`.
