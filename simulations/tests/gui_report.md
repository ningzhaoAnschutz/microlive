# MicroLive GUI Validation Report

**Generated:** 2026-01-14 07:09:57  
**Status:** ✅ ALL PASS

---

## Summary

| Passed | Failed | Total |
| :---: | :---: | :---: |
| 6 | 0 | 6 |

---

## Test Results

### Segmentation

**Status:** ✅ PASS

*Validates that the GUI correctly segmented the expected number of cells. Checks that Cellpose detected 4 cells as configured in the simulation.*

- Expected cells: 4
- GUI detected: 4

---

### Spot Count per Cell

**Status:** ✅ PASS

*Compares the number of tracked spots per cell between GUI and ground truth. Note: Particle labels may differ, so we compare counts rather than matching individual particles. Requires ≤50% relative error per cell.*


---

### Compartment Assignment

**Status:** ✅ PASS

*Validates that tracked spots are correctly assigned to nucleus or cytosol compartments by matching GUI positions to ground truth and comparing labels. Requires ≥70% compartment accuracy.*

- Matched spots: 1087
- Correct compartment: 1049
- Accuracy: 96.5%

---

### Photobleaching

**Status:** ✅ PASS

*Compares photobleaching decay rates (k) recovered by the GUI against the configured ground truth values. Tests all 3 channels with ≤30% error threshold.*


| Channel | Config | GUI | Error |
| :---: | :---: | :---: | :---: |
| Ch0 | 0.000370 | 0.000286 | 22.8% |
| Ch1 | 0.000600 | 0.000459 | 23.5% |
| Ch2 | 0.000850 | 0.000644 | 24.3% |

---

### MSD

**Status:** ✅ PASS

*Compares the diffusion coefficient (D in µm²/s) recovered by the GUI's MSD analysis against the configured ground truth. Accounts for voxel calibration. Threshold: ≤80% error.*

- Config D: 0.000169 µm²/s
- GUI D: 0.000270 µm²/s
- Error: 59.6%

---

### Colocalization

**Status:** ✅ PASS

*Compares the Ch0 vs Ch1 colocalization percentage detected by the GUI against the ground truth from simulation data (has_ch1_partner column). Threshold: ≤25% absolute error.*


| Source | Ch0 vs Ch1 Coloc |
| :---: | :---: |
| Ground Truth | 82.7% |
| GUI Recovery | 74.6% |
| Error | 8.0% |

---
