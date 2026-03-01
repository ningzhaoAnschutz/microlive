# MicroLive Simulation Validation Report

**Generated:** 2026-03-01 15:41:22  
**Status:** ✅ ALL PASS

---

## Summary

| Passed | Failed | Total |
| :---: | :---: | :---: |
| 9 | 0 | 9 |

---

## Simulation Parameters

### Image Configuration

| Parameter | Value |
| :--- | :---: |
| Dimensions | 3D |
| Size (YX) | [512, 512] |
| Z Slices | 10 |
| Voxel Size (YX) | 130.0 nm |
| Voxel Size (Z) | 300.0 nm |

### Simulation Timing

| Parameter | Value |
| :--- | :---: |
| Total time | 600 s |
| Frame rate | 5 s/frame |
| Total frames | 120 |

### Noise Configuration (Per-Channel)

| Channel | Noise Std |
| :---: | :---: |
| Ch 0 | 300.0 |
| Ch 1 | 300.0 |
| Ch 2 | 300.0 |

### Spot Properties

| Parameter | Value |
| :--- | :---: |
| SNR Mean | 3.5 |
| SNR Std | 1.0 |
| SNR Range | [0.5, 5.0] |
| Size Mean | 1.5 px |

### Motion

| Parameter | Value |
| :--- | :---: |
| Diffusion Coefficient | 0.05 px²/frame |
| Confinement | cell |

### Cell Geometry

| Parameter | Value |
| :--- | :---: |
| Number of Cells | 1 |
| Layout | single |
| Cell Diameter (YX) | [350, 350] px |
| Nucleus Enabled | Yes |

---

## Test Results

### Ground Truth Quality

**Status:** ✅ PASS

*Validates that the simulation correctly generates ground truth data. Checks that particles are assigned to cells (not background), all cells contain particles, and the measured SNR matches the configured value within tolerance.*

---

### Colocalization

**Status:** ✅ PASS

*Verifies that channel colocalization probabilities are correctly implemented. For each spot in Channel 0, checks that the fraction with co-localized signal in Channel 1 and Channel 2 matches the configured probabilities (±10% tolerance).*

- Ch1 coloc: 75.2% (config: 80.0%)
- Ch2 coloc: 67.5% (config: 60.0%)

---

### Photobleaching

**Status:** ✅ PASS

*Tests recovery of photobleaching decay rates from simulated images. Fits an exponential decay to mean intensity over time for each channel and compares the recovered decay constant (k) to the configured value (≤30% error threshold).*

| Channel | Config | Measured | Error |
| :---: | :---: | :---: | :---: |
| 0 | 0.000370 | 0.000300 | 18.9% |
| 1 | 0.000600 | 0.000476 | 20.6% |
| 2 | 0.000718 | 0.000514 | 28.4% |

---

### Spot Detection

**Status:** ✅ PASS

*Evaluates spot detection accuracy using BigFISH's automatic thresholding. Measures what fraction of ground truth spots are detected (recall ≥70%) and reports false positive rate. Uses a 5-pixel matching radius.*

---

### Position Accuracy

**Status:** ✅ PASS

*Assesses tracking position accuracy by matching tracked spots to ground truth. Reports mean and median position error in pixels for matched spots. Uses min_length_trajectory=15 and memory=0 for realistic tracking parameters.*

- Mean position error: 0.47 px
- Tracked spots: 19

---

### Compartment Assignment

**Status:** ✅ PASS

*Validates that detected spots are correctly assigned to nucleus or cytosol. Compares tracked spot compartment labels against ground truth. Requires ≥50% detection recall and ≥75% compartment accuracy.*

- Compartment accuracy: 100.0%
- Correct: 8/8

---

### MSD Recovery

**Status:** ✅ PASS

*Tests diffusion coefficient (D) recovery from Mean Squared Displacement analysis. Runs tracking on photobleaching-corrected images, then calculates MSD using trackpy. Uses unique particle IDs (cell_id + particle) for multi-cell scenarios. Compares measured D (µm²/s) to configured value (≤50% error threshold).*

- Tracking channel: Ch 0
- Config D: 0.000169 µm²/s
- Measured D: 0.000201 µm²/s
- Error: 18.7%
- Particles tracked: 1

---

### Colocalization Recovery

**Status:** ✅ PASS

*Tests ML-based colocalization detection between channels. For each trajectory-averaged Ch0 crop, uses the CNN classifier (threshold=0.51) to detect colocalized signal in Ch1 and Ch2. Compares measured colocalization percentages against target probabilities (ground truth when available, otherwise config; ≤25% absolute error threshold).*

- ML Threshold: 0.51
- Spots analyzed: 50
- Target source: ground_truth

| Channel | Target | ML Recovery | Error |
| :---: | :---: | :---: | :---: |
| Ch1 | 76.0% | 74.0% | 2.0% |
| Ch2 | 66.0% | 66.0% | 0.0% |

---

### GUI Syntax

**Status:** ✅ PASS

*Validates that the MicroLive GUI module compiles without syntax errors. Checks: (1) Python syntax validation using py_compile, (2) module spec validation using importlib, and (3) core package import verification. Reports file size and line count for the GUI module.*

- File: app.py
- Lines of code: 18,830
- File size: 901,537 bytes
- Syntax OK: ✅
- Package import OK: ✅

---
