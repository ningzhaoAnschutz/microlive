# MicroLive GUI Test Report

**Generated:** 2026-01-07 09:18:02

## Summary

| Status | Count |
|--------|-------|
| ✅ PASS | 8 |
| ❌ FAIL | 1 |
| ⏭️ SKIP | 0 |

**Overall Result:** ❌ SOME TESTS FAILED

---

## Detailed Results

### Mask IoU Comparison

| Test | Status | Details |
|------|--------|---------|
| Cytosol Mask | ✅ PASS | IoU = 0.969 (threshold: 0.8) |
| Nucleus Mask | ✅ PASS | IoU = 0.966 (threshold: 0.8) |

### Photobleaching Decay Rates

| Test | Status | Details |
|------|--------|---------|
| Channel 0 | ✅ PASS | GT: 1.60e-04, GUI: 1.60e-04, Error: 0.0% |
| Channel 1 | ✅ PASS | GT: 1.20e-04, GUI: 1.18e-04, Error: 1.5% |
| Channel 2 | ✅ PASS | GT: 1.60e-04, GUI: 1.60e-04, Error: 0.0% |
| Channel 3 | ✅ PASS | GT: 2.00e-04, GUI: 1.79e-04, Error: 10.4% |

### Average Spot Count per Frame

| Test | Status | Details |
|------|--------|---------|
| Mature RNA (Ch1) | ❌ FAIL | GT: 102.1, GUI: 51.6, Error: 49.5% |

### Spot Intensity (PSF Amplitude Ch1)

| Test | Status | Details |
|------|--------|---------|
| PSF Amplitude Ch1 | ✅ PASS | GT: 4000.0, GUI: 3874.8, Error: 3.1% |

### Diffusion Coefficient (MSD)

| Test | Status | Details |
|------|--------|---------|
| RNA Diffusion (2D) | ✅ PASS | GT (corrected): 0.500, GUI: 0.468, Error: 6.5% |
| Correction Info | ❓ INFO | Raw GT D=0.500 px²/s. 2D mode: no Z-compression correction needed. Formula: (2 + 1/ratio²)/3 |

---

## Test Thresholds

- **Mask IoU:** ≥ 80%
- **Photobleaching Error:** ≤ 20%
- **Spot Count Error:** ≤ 40%
- **Diffusion Coefficient Error:** ≤ 25%
- **PSF Amplitude Error:** ≤ 20%
