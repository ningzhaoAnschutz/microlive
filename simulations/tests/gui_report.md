# MicroLive GUI Validation Report

**Generated:** 2026-07-06 14:26:24  
**Status:** ❌ 1 FAILED

---

## Summary

| Passed | Failed | Skipped | Total |
| :---: | :---: | :---: | :---: |
| 6 | 1 | 0 | 7 |

---

## Test Results

### Segmentation

**Status:** ✅ PASS

*Validates that the GUI correctly segmented the expected number of cells. Checks that Cellpose detected 4 cells as configured in the simulation.*

- Expected cells: 1
- GUI detected: 1

---

### Spot Count per Cell

**Status:** ✅ PASS

*Compares the number of tracked spots per cell between GUI and ground truth. Note: Particle labels may differ, so we compare counts rather than matching individual particles. Requires ≤60% relative error per cell and for total count.*

- Ground-truth particles: 50
- GUI particles: 34
- Total error: 32.0%
- Per-cell pass: Yes
- Cell matching complete: Yes

---

### Compartment Assignment

**Status:** ✅ PASS

*Validates that tracked spots are correctly assigned to nucleus or cytosol compartments by matching GUI positions to ground truth and comparing labels. Requires ≥70% compartment accuracy.*

- Matched spots: 2600
- Correct compartment: 2594
- Accuracy: 99.8%

---

### Photobleaching

**Status:** ❌ FAIL

*Compares photobleaching decay rates (k) recovered by the GUI against the configured ground truth values. Tests all 3 channels with ≤30% error threshold.*


| Channel | Config | GUI | Error |
| :---: | :---: | :---: | :---: |
| Ch0 | 0.000370 | 0.000268 | 27.6% |
| Ch1 | 0.000600 | 0.000425 | 29.2% |
| Ch2 | 0.000718 | 0.000458 | 36.2% |

---

### MSD

**Status:** ✅ PASS

*Compares the diffusion coefficient (D in µm²/s) recovered by the GUI's MSD analysis against the configured ground truth. Accounts for voxel calibration. Threshold: ≤80% error.*

- Config D: 0.000169 µm²/s
- GUI D: 0.000193 µm²/s
- Error: 14.0%

---

### Colocalization

**Status:** ✅ PASS

*Compares the Ch0 vs Ch1 colocalization percentage detected by the GUI against the ground truth from simulation data (has_ch1_partner column). Threshold: ≤25% absolute error.*


| Source | Ch0 vs Ch1 Coloc |
| :---: | :---: |
| Ground Truth | 76.0% |
| GUI Recovery | 85.3% |
| Error | 9.3% |

---

### is_colocalized Tracking

**Status:** ✅ PASS

*End-to-end validation of the per-particle is_colocalized column in the exported tracking CSV. Matches GUI-tracked particles to ground truth by position and compares is_colocalized vs has_ch1_partner per particle. Threshold: ≥75% accuracy.*

- GUI particles evaluated: 34
- Matched to GT: 28
- Agreement: 28/28 (100.0%)

| Metric | Value |
| :--- | :---: |
| True Positive | 24 |
| True Negative | 4 |
| False Positive | 0 |
| False Negative | 0 |
| Precision | 100.0% |
| Recall | 100.0% |
| F1 Score | 100.0% |

---
