#!/usr/bin/env python3
"""
MicroLive GUI Performance Test
==============================

Compares GUI outputs against ground truth from VirtualCell simulation.

Usage:
    python run_gui_test.py

Output:
    test_report.md - Markdown report with PASS/FAIL status
"""

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Optional imports
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("Warning: tifffile not installed. Mask comparison will be skipped.")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Ground Truth Paths
GT_BASE = Path("/Users/nzlab-la/Desktop/Github/virtual_cell/spatio_temporal_models/results_simulation")
GT_CYTOSOL_MASK = GT_BASE / "mask_cytosol.tif"
GT_NUCLEUS_MASK = GT_BASE / "mask_nucleus.tif"
GT_MATURE_RNA = GT_BASE / "ground_truth_mature_rna.csv"
GT_METADATA = GT_BASE / "simulation_metadata.txt"

# GUI Results Paths
GUI_BASE = Path("/Users/nzlab-la/Desktop/microlive/test_gui/results_simulated_microscopy_simulated_microscopy")
GUI_CYTOSOL_MASK = GUI_BASE / "cellpose_cytosol_simulated_microscopy_simulated_microscopy.tif"
GUI_NUCLEUS_MASK = GUI_BASE / "cellpose_nucleus_simulated_microscopy_simulated_microscopy.tif"
GUI_TRACKING = GUI_BASE / "tracking_simulated_microscopy_simulated_microscopy.csv"
GUI_METADATA = GUI_BASE / "Metadata_simulated_microscopy_simulated_microscopy.txt"

# Output
OUTPUT_DIR = Path("/Users/nzlab-la/Desktop/microlive/test_gui")
REPORT_FILE = OUTPUT_DIR / "test_report.md"

# Thresholds
THRESHOLD_MASK_IOU = 0.80      # Mask IoU must be >= 80%
THRESHOLD_PHOTOBLEACHING = 0.20  # Decay rate error must be <= 20%
THRESHOLD_SPOT_COUNT = 0.40     # Spot count error must be <= 40%
THRESHOLD_DIFFUSION = 0.25      # Diffusion coefficient error must be <= 25%
THRESHOLD_INTENSITY = 0.20      # PSF amplitude error must be <= 20%

# Spot Type Configuration
# Note: spot_type in MicroLive is the ACTUAL imaging channel number (e.g., 0, 1, 2, 3),
# NOT an index into a list. For the VirtualCell simulation:
#   Channel 1 = Mature RNA (the channel used for tracking)
MATURE_RNA_CHANNEL = 1



# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union for binary masks."""
    # Binarize masks (non-zero = foreground)
    m1 = (mask1 > 0).astype(bool)
    m2 = (mask2 > 0).astype(bool)
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return 0.0
    return intersection / union


def parse_gt_metadata(filepath: Path) -> dict:
    """Parse ground truth metadata file."""
    metadata = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse photobleaching rates - format: "Channel X (Name)......... 0.00016 s⁻¹"
    pb_pattern = r"Channel\s+(\d)\s+\([^)]+\)\.+\s*([\d.e+-]+)"
    for match in re.finditer(pb_pattern, content, re.IGNORECASE):
        ch = int(match.group(1))
        rate = float(match.group(2))
        metadata[f'photobleaching_ch{ch}'] = rate
    
    # Parse avg mature RNA per frame - format: "Avg Mature Rna Per Frame...... 136.1"
    rna_pattern = r"Avg Mature Rna Per Frame\.+\s*([\d.]+)"
    match = re.search(rna_pattern, content)
    if match:
        metadata['avg_mature_rna_per_frame'] = float(match.group(1))
    
    # Parse diffusion coefficient - format: "RNA diffusion coeff..... 0.5 px²/s"
    diff_pattern = r"RNA diffusion coeff\.+\s*([\d.]+)"
    match = re.search(diff_pattern, content)
    if match:
        metadata['diffusion_coeff'] = float(match.group(1))
    
    return metadata


def parse_gui_metadata(filepath: Path) -> dict:
    """Parse GUI-generated metadata file."""
    metadata = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse photobleaching decay rates
    # Format: "Channel X Decay Rate (k)................ 1.600371e-04"
    pb_pattern = r"Channel\s+(\d)\s+Decay Rate\s+\(k\)\.+\s*([\d.e+-]+)"
    for match in re.finditer(pb_pattern, content, re.IGNORECASE):
        ch = int(match.group(1))
        rate = float(match.group(2))
        metadata[f'photobleaching_ch{ch}'] = rate
    
    # Parse diffusion coefficient (px²/s)
    # Format: "Diffusion Coefficient (px²/s)....... 2.958580e+01"
    diff_pattern = r"Diffusion Coefficient\s+\(px.*?/s\)\.+\s*([\d.e+-]+)"
    match = re.search(diff_pattern, content, re.IGNORECASE)
    if match:
        metadata['diffusion_px2_s'] = float(match.group(1))
    
    # Parse MSD mode (2D or 3D)
    # Format: "MSD Mode....... 3D" or "MSD Mode....... 2D"
    mode_pattern = r"MSD Mode\.+\s*(2D|3D)"
    match = re.search(mode_pattern, content, re.IGNORECASE)
    if match:
        metadata['msd_mode'] = match.group(1).upper()
    
    return metadata





def compute_relative_error(measured: float, expected: float) -> float:
    """Compute relative error as |measured - expected| / expected."""
    if expected == 0:
        return float('inf') if measured != 0 else 0.0
    return abs(measured - expected) / abs(expected)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_mask_iou() -> dict:
    """Compare GUI masks against ground truth masks."""
    results = {'name': 'Mask IoU Comparison', 'tests': []}
    
    if not HAS_TIFFFILE:
        results['tests'].append({
            'name': 'Cytosol Mask',
            'status': 'SKIP',
            'message': 'tifffile not installed'
        })
        return results
    
    for mask_type, gt_path, gui_path in [
        ('Cytosol', GT_CYTOSOL_MASK, GUI_CYTOSOL_MASK),
        ('Nucleus', GT_NUCLEUS_MASK, GUI_NUCLEUS_MASK)
    ]:
        if not gt_path.exists():
            results['tests'].append({
                'name': f'{mask_type} Mask',
                'status': 'SKIP',
                'message': f'Ground truth not found: {gt_path}'
            })
            continue
        
        if not gui_path.exists():
            results['tests'].append({
                'name': f'{mask_type} Mask',
                'status': 'SKIP',
                'message': f'GUI result not found: {gui_path}'
            })
            continue
        
        gt_mask = tifffile.imread(gt_path)
        gui_mask = tifffile.imread(gui_path)
        
        # Handle shape differences (GT may be 2D, GUI may be 3D)
        if gt_mask.ndim > 2:
            gt_mask = gt_mask.max(axis=tuple(range(gt_mask.ndim - 2)))  # Max project to 2D
        if gui_mask.ndim > 2:
            gui_mask = gui_mask.max(axis=tuple(range(gui_mask.ndim - 2)))
        
        iou = compute_iou(gt_mask, gui_mask)
        passed = iou >= THRESHOLD_MASK_IOU
        
        results['tests'].append({
            'name': f'{mask_type} Mask',
            'status': 'PASS' if passed else 'FAIL',
            'iou': iou,
            'threshold': THRESHOLD_MASK_IOU,
            'message': f'IoU = {iou:.3f} (threshold: {THRESHOLD_MASK_IOU})'
        })
    
    return results


def test_photobleaching_rates() -> dict:
    """Compare GUI photobleaching rates against ground truth."""
    results = {'name': 'Photobleaching Decay Rates', 'tests': []}
    
    if not GT_METADATA.exists():
        results['tests'].append({
            'name': 'All Channels',
            'status': 'SKIP',
            'message': f'Ground truth metadata not found: {GT_METADATA}'
        })
        return results
    
    if not GUI_METADATA.exists():
        results['tests'].append({
            'name': 'All Channels',
            'status': 'SKIP',
            'message': f'GUI metadata not found: {GUI_METADATA}'
        })
        return results
    
    gt_meta = parse_gt_metadata(GT_METADATA)
    gui_meta = parse_gui_metadata(GUI_METADATA)
    
    for ch in range(4):
        key = f'photobleaching_ch{ch}'
        
        if key not in gt_meta:
            results['tests'].append({
                'name': f'Channel {ch}',
                'status': 'SKIP',
                'message': 'Ground truth rate not found'
            })
            continue
        
        if key not in gui_meta:
            results['tests'].append({
                'name': f'Channel {ch}',
                'status': 'SKIP',
                'message': 'GUI rate not found - did you run photobleaching?'
            })
            continue
        
        gt_rate = gt_meta[key]
        gui_rate = gui_meta[key]
        error = compute_relative_error(gui_rate, gt_rate)
        passed = error <= THRESHOLD_PHOTOBLEACHING
        
        results['tests'].append({
            'name': f'Channel {ch}',
            'status': 'PASS' if passed else 'FAIL',
            'gt_rate': gt_rate,
            'gui_rate': gui_rate,
            'error': error,
            'threshold': THRESHOLD_PHOTOBLEACHING,
            'message': f'GT: {gt_rate:.2e}, GUI: {gui_rate:.2e}, Error: {error*100:.1f}%'
        })
    
    return results


def test_spot_count() -> dict:
    """Compare average spot count per frame."""
    results = {'name': 'Average Spot Count per Frame', 'tests': []}
    
    if not GT_METADATA.exists():
        results['tests'].append({
            'name': 'Mature RNA (Ch1)',
            'status': 'SKIP',
            'message': f'Ground truth metadata not found'
        })
        return results
    
    if not GUI_TRACKING.exists():
        results['tests'].append({
            'name': 'Mature RNA (Ch1)',
            'status': 'SKIP',
            'message': f'GUI tracking CSV not found: {GUI_TRACKING}'
        })
        return results
    
    gt_meta = parse_gt_metadata(GT_METADATA)
    gt_avg = gt_meta.get('avg_mature_rna_per_frame', None)
    
    if gt_avg is None:
        results['tests'].append({
            'name': 'Mature RNA (Ch1)',
            'status': 'SKIP',
            'message': 'Ground truth avg not found in metadata'
        })
        return results
    
    # Load GUI tracking and compute avg spots per frame
    df = pd.read_csv(GUI_TRACKING)
    
    # Only filter by spot_type if multiple types exist
    # Note: spot_type is the actual imaging channel number, not an index
    if 'spot_type' in df.columns:
        unique_types = df['spot_type'].unique()
        if len(unique_types) > 1:
            # Multiple channels - filter to the Mature RNA channel
            df = df[df['spot_type'] == MATURE_RNA_CHANNEL]
        # If only one type, use all spots
    
    # Count spots per frame
    if 'frame' in df.columns and len(df) > 0:
        spots_per_frame = df.groupby('frame').size()
        gui_avg = spots_per_frame.mean()
    else:
        gui_avg = 0.0
    
    error = compute_relative_error(gui_avg, gt_avg)
    passed = error <= THRESHOLD_SPOT_COUNT
    
    results['tests'].append({
        'name': 'Mature RNA (Ch1)',
        'status': 'PASS' if passed else 'FAIL',
        'gt_avg': gt_avg,
        'gui_avg': gui_avg,
        'error': error,
        'threshold': THRESHOLD_SPOT_COUNT,
        'message': f'GT: {gt_avg:.1f}, GUI: {gui_avg:.1f}, Error: {error*100:.1f}%'
    })
    
    return results


def test_spot_intensity() -> dict:
    """Compare PSF amplitude (spot intensity) between GT and GUI."""
    results = {'name': 'Spot Intensity (PSF Amplitude Ch1)', 'tests': []}
    
    if not GT_MATURE_RNA.exists():
        results['tests'].append({
            'name': 'PSF Amplitude Ch1',
            'status': 'SKIP',
            'message': 'Ground truth CSV not found'
        })
        return results
    
    if not GUI_TRACKING.exists():
        results['tests'].append({
            'name': 'PSF Amplitude Ch1',
            'status': 'SKIP',
            'message': 'GUI tracking CSV not found'
        })
        return results
    
    # Load ground truth
    gt_df = pd.read_csv(GT_MATURE_RNA)
    
    # Load GUI tracking
    gui_df = pd.read_csv(GUI_TRACKING)
    
    # Only filter by spot_type if multiple types exist
    # Note: spot_type is the actual imaging channel number, not an index
    if 'spot_type' in gui_df.columns:
        unique_types = gui_df['spot_type'].unique()
        if len(unique_types) > 1:
            # Multiple channels - filter to the Mature RNA channel
            gui_df = gui_df[gui_df['spot_type'] == MATURE_RNA_CHANNEL]
    
    # Use psf_amplitude_ch_1 for comparison
    gt_col = 'psf_amplitude_ch_1'
    gui_col = 'psf_amplitude_ch_1'
    
    if gt_col not in gt_df.columns:
        results['tests'].append({
            'name': 'PSF Amplitude Ch1',
            'status': 'SKIP',
            'message': f'GT column {gt_col} not found. Cols: {list(gt_df.columns)[:8]}'
        })
        return results
    
    if gui_col not in gui_df.columns:
        results['tests'].append({
            'name': 'PSF Amplitude Ch1',
            'status': 'SKIP',
            'message': f'GUI column {gui_col} not found. Cols: {list(gui_df.columns)[:8]}'
        })
        return results
    
    gt_values = gt_df[gt_col].dropna()
    gui_values = gui_df[gui_col].dropna()
    
    if len(gt_values) == 0 or len(gui_values) == 0:
        results['tests'].append({
            'name': 'PSF Amplitude Ch1',
            'status': 'SKIP',
            'message': 'No data found in one or both columns'
        })
        return results
    
    gt_mean = gt_values.mean()
    gui_mean = gui_values.mean()
    
    error = compute_relative_error(gui_mean, gt_mean)
    passed = error <= THRESHOLD_INTENSITY
    
    results['tests'].append({
        'name': 'PSF Amplitude Ch1',
        'status': 'PASS' if passed else 'FAIL',
        'gt_mean': gt_mean,
        'gui_mean': gui_mean,
        'error': error,
        'threshold': THRESHOLD_INTENSITY,
        'message': f'GT: {gt_mean:.1f}, GUI: {gui_mean:.1f}, Error: {error*100:.1f}%'
    })
    
    return results


def test_diffusion_coefficient() -> dict:
    """
    Compare diffusion coefficient from MSD analysis.
    
    IMPORTANT: Coordinate System Correction
    ========================================
    
    The biological simulation runs in a 3D pixel space with dimensions:
        - Simulation volume: [512, 512, 100] (X, Y, Z)
        
    The microscopy image is generated with dimensions:
        - Image volume: [512, 512, 10] (X, Y, Z slices)
        
    This creates a Z-axis compression:
        - Z_image = Z_sim / 10  (10:1 compression)
        
    Because MSD measures squared displacements, the Z contribution is affected:
        - In simulation space: MSD_sim = Δx² + Δy² + Δz² = 6 * D_sim * t
        - In image space: MSD_image = Δx² + Δy² + (Δz/10)² = Δx² + Δy² + Δz²/100
        
    For isotropic diffusion where each axis contributes 2*D*t:
        - MSD_image = 2*D*t + 2*D*t + 2*D*t/100 = D*t*(4 + 0.02) ≈ 4.02*D*t
        
    When MicroLive fits MSD_image = 6 * D_recovered * t (assuming 3D diffusion):
        - D_recovered = D_sim * (4.02 / 6) ≈ D_sim * 0.67
        
    So we apply a correction factor:
        - D_expected_in_image_space = D_simulation * Z_COMPRESSION_FACTOR
        
    where Z_COMPRESSION_FACTOR = (2/3 + 1/(3 * compression_ratio²))
    For compression_ratio = 10: factor ≈ 0.67
    """
    results = {'name': 'Diffusion Coefficient (MSD)', 'tests': []}
    
    # ==========================================================================
    # Z-Compression Correction Parameters
    # ==========================================================================
    # These values come from the simulation config:
    #   - simulation_volume_size: [512, 512, 100]  (Z = 100 pixels)
    #   - num_z_slices: 10                         (Z = 10 slices in image)
    SIM_Z_PIXELS = 100
    IMAGE_Z_SLICES = 10
    Z_COMPRESSION_RATIO = SIM_Z_PIXELS / IMAGE_Z_SLICES  # = 10
    
    # For 3D isotropic diffusion with Z-compression:
    # MSD_image = 2D*t + 2D*t + 2D*t/(ratio²) = 2D*t * (2 + 1/ratio²)
    # MicroLive fits: MSD = 6 * D_recovered * t
    # So: D_recovered = D_sim * (2 + 1/ratio²) / 3
    Z_COMPRESSION_FACTOR = (2 + 1/(Z_COMPRESSION_RATIO**2)) / 3  # ≈ 0.67

    if not GT_METADATA.exists():
        results['tests'].append({
            'name': 'RNA Diffusion',
            'status': 'SKIP',
            'message': 'Ground truth metadata not found'
        })
        return results
    
    if not GUI_METADATA.exists():
        results['tests'].append({
            'name': 'RNA Diffusion',
            'status': 'SKIP',
            'message': 'GUI metadata not found'
        })
        return results
    
    gt_meta = parse_gt_metadata(GT_METADATA)
    gui_meta = parse_gui_metadata(GUI_METADATA)
    
    # GT diffusion is in simulation px²/s
    gt_diffusion_sim = gt_meta.get('diffusion_coeff', None)
    if gt_diffusion_sim is None:
        results['tests'].append({
            'name': 'RNA Diffusion',
            'status': 'SKIP',
            'message': 'GT diffusion coefficient not found in metadata'
        })
        return results
    
    # Detect MSD mode from GUI metadata (2D or 3D)
    msd_mode = gui_meta.get('msd_mode', '3D')  # Default to 3D if not specified
    
    # Apply appropriate correction based on tracking mode
    if msd_mode == '2D':
        # 2D tracking: Z is collapsed via max projection, no Z-compression issue
        # The simulation XY diffusion maps directly to image XY
        # For 2D: MSD = 4*D*t for XY only
        # Since simulation has 3D diffusion but we only track XY:
        # MSD_image_2D = 4*D*t (only XY contributes)
        # MicroLive fits: MSD = 4*D*t, so D_recovered ≈ D_sim
        correction_factor = 1.0
        correction_note = "2D mode: no Z-compression correction needed"
    else:
        # 3D tracking: Z is compressed from simulation to image space
        correction_factor = Z_COMPRESSION_FACTOR
        correction_note = f"3D mode: Z-compression factor={Z_COMPRESSION_FACTOR:.3f}"
    
    gt_diffusion_expected = gt_diffusion_sim * correction_factor
    
    # GUI diffusion is in image px²/s (already in image space)
    gui_diffusion = gui_meta.get('diffusion_px2_s', None)
    if gui_diffusion is None:
        results['tests'].append({
            'name': 'RNA Diffusion',
            'status': 'SKIP',
            'message': 'GUI diffusion coefficient not found - did you run tracking?'
        })
        return results
    
    error = compute_relative_error(gui_diffusion, gt_diffusion_expected)
    passed = error <= THRESHOLD_DIFFUSION
    
    results['tests'].append({
        'name': f'RNA Diffusion ({msd_mode})',
        'status': 'PASS' if passed else 'FAIL',
        'gt_value': gt_diffusion_expected,
        'gui_value': gui_diffusion,
        'error': error,
        'threshold': THRESHOLD_DIFFUSION,
        'message': f'GT (corrected): {gt_diffusion_expected:.3f}, GUI: {gui_diffusion:.3f}, Error: {error*100:.1f}%'
    })
    
    # Also add an informational note about the correction
    results['tests'].append({
        'name': 'Correction Info',
        'status': 'INFO',
        'message': f'Raw GT D={gt_diffusion_sim:.3f} px²/s. {correction_note}. Formula: (2 + 1/ratio²)/3'
    })
    
    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================


def generate_report(all_results: list) -> str:
    """Generate markdown report from test results."""
    lines = []
    lines.append("# MicroLive GUI Test Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary
    total_pass = 0
    total_fail = 0
    total_skip = 0
    
    for result_group in all_results:
        for test in result_group['tests']:
            if test['status'] == 'PASS':
                total_pass += 1
            elif test['status'] == 'FAIL':
                total_fail += 1
            elif test['status'] == 'SKIP':
                total_skip += 1
    
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Status | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| ✅ PASS | {total_pass} |")
    lines.append(f"| ❌ FAIL | {total_fail} |")
    lines.append(f"| ⏭️ SKIP | {total_skip} |")
    lines.append("")
    
    overall = "✅ ALL TESTS PASSED" if total_fail == 0 and total_pass > 0 else "❌ SOME TESTS FAILED"
    lines.append(f"**Overall Result:** {overall}")
    lines.append("")
    
    # Detailed results
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")
    
    for result_group in all_results:
        lines.append(f"### {result_group['name']}")
        lines.append("")
        lines.append("| Test | Status | Details |")
        lines.append("|------|--------|---------|")
        
        for test in result_group['tests']:
            status_icon = {'PASS': '✅', 'FAIL': '❌', 'SKIP': '⏭️', 'REPORT': '📊'}.get(test['status'], '❓')
            lines.append(f"| {test['name']} | {status_icon} {test['status']} | {test['message']} |")
        
        lines.append("")
    
    # Thresholds
    lines.append("---")
    lines.append("")
    lines.append("## Test Thresholds")
    lines.append("")
    lines.append(f"- **Mask IoU:** ≥ {THRESHOLD_MASK_IOU*100:.0f}%")
    lines.append(f"- **Photobleaching Error:** ≤ {THRESHOLD_PHOTOBLEACHING*100:.0f}%")
    lines.append(f"- **Spot Count Error:** ≤ {THRESHOLD_SPOT_COUNT*100:.0f}%")
    lines.append(f"- **Diffusion Coefficient Error:** ≤ {THRESHOLD_DIFFUSION*100:.0f}%")
    lines.append(f"- **PSF Amplitude Error:** ≤ {THRESHOLD_INTENSITY*100:.0f}%")
    lines.append("")

    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("MicroLive GUI Performance Test")
    print("=" * 60)
    print()
    
    # Run all tests
    all_results = []
    
    print("Running Mask IoU test...")
    all_results.append(test_mask_iou())
    
    print("Running Photobleaching test...")
    all_results.append(test_photobleaching_rates())
    
    print("Running Spot Count test...")
    all_results.append(test_spot_count())
    
    print("Running Intensity Comparison...")
    all_results.append(test_spot_intensity())
    
    print("Running Diffusion Coefficient test...")
    all_results.append(test_diffusion_coefficient())
    
    # Generate report
    print()
    print("Generating report...")
    report = generate_report(all_results)
    
    # Save report
    with open(REPORT_FILE, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {REPORT_FILE}")
    print()
    
    # Print summary
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
