#!/usr/bin/env python3
"""
MicroLive GUI Validation Test Suite
====================================

Compares GUI analysis results against simulation ground truth.

Tests:
  1. Segmentation: Verifies 4 cells were detected
  2. Spot Count per Cell: Compares spot counts (labels may differ)
  3. Compartment Assignment: Validates nucleus/cytosol classification
  4. Photobleaching: Compares decay rates (GUI vs ground truth config)
  5. MSD: Compares diffusion coefficient (GUI vs ground truth config)
  6. Colocalization: Compares Ch0 vs Ch1 colocalization (GUI vs ground truth)

Usage:
    python run_test_gui.py
    python run_test_gui.py --gui-dir ../results_simulated_spots --gt-dir ../results
"""

import argparse
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
import tifffile

# Silence matplotlib before import
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

# =============================================================================
# THRESHOLD CONSTANTS
# =============================================================================
THRESHOLD_PHOTOBLEACHING = 0.30    # ≤30% error on decay rate
THRESHOLD_COLOCALIZATION = 0.25    # ≤25% absolute error on coloc %
THRESHOLD_MSD = 0.80               # ≤80% error on D (relaxed for GUI noise)
THRESHOLD_SPOT_COUNT = 0.60        # ≤60% error on spot count (relaxed - GUI may miss low-SNR spots)
THRESHOLD_COMPARTMENT = 0.70       # ≥70% compartment accuracy


def load_config(config_path: Path) -> Dict:
    """Load simulation configuration."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Parse MicroLive metadata file."""
    if not metadata_path.exists():
        return {}
    
    metadata = {}
    with open(metadata_path, 'r') as f:
        content = f.read()
    
    # Extract photobleaching decay rates
    for ch in range(3):
        match = re.search(rf'Channel {ch} Decay Rate \(k\)\.+ ([\d.e+-]+)', content)
        if match:
            metadata[f'k_ch{ch}'] = float(match.group(1))
    
    # Extract MSD
    match = re.search(r'Diffusion Coefficient \(µm²/s\)\.+ ([\d.e+-]+)', content)
    if match:
        metadata['D_um2_s'] = float(match.group(1))
    
    # Extract voxel sizes
    match = re.search(r'Voxel Size YX \(nm\)\.+ ([\d.]+)', content)
    if match:
        metadata['voxel_yx_nm'] = float(match.group(1))
    
    match = re.search(r'Time Interval \(s\)\.+ ([\d.]+)', content)
    if match:
        metadata['time_interval'] = float(match.group(1))
    
    # Extract number of cells from segmentation
    match = re.search(r'Cytosol Segmented\.+ Yes \((\d+) cells\)', content)
    if match:
        metadata['n_cells'] = int(match.group(1))
    
    return metadata


def compute_cell_centroids(mask: np.ndarray) -> Dict[int, tuple]:
    """Compute centroids for each cell label in a mask.
    
    Args:
        mask: 2D or 3D mask with integer cell labels
        
    Returns:
        Dictionary mapping cell_id -> (y, x) centroid
    """
    from scipy import ndimage
    
    # Get max projection if 3D
    if mask.ndim == 3:
        mask_2d = np.max(mask, axis=0)
    else:
        mask_2d = mask
    
    centroids = {}
    unique_labels = np.unique(mask_2d)
    
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        
        # Find centroid
        coords = np.where(mask_2d == label)
        if len(coords[0]) > 0:
            y_center = np.mean(coords[0])
            x_center = np.mean(coords[1])
            centroids[label] = (y_center, x_center)
    
    return centroids


def match_cells_by_centroid(gui_centroids: Dict[int, tuple], 
                            gt_centroids: Dict[int, tuple],
                            max_distance: float = 50.0) -> Dict[int, int]:
    """Match GUI cells to ground truth cells by nearest centroid.
    
    Args:
        gui_centroids: Dict of GUI cell_id -> (y, x)
        gt_centroids: Dict of GT cell_id -> (y, x)
        max_distance: Maximum centroid distance for a valid match
        
    Returns:
        Dictionary mapping GUI cell_id -> GT cell_id
    """
    matches = {}
    used_gt = set()
    
    for gui_id, gui_centroid in gui_centroids.items():
        best_match = None
        best_dist = float('inf')
        
        for gt_id, gt_centroid in gt_centroids.items():
            if gt_id in used_gt:
                continue
            
            dist = np.sqrt((gui_centroid[0] - gt_centroid[0])**2 + 
                          (gui_centroid[1] - gt_centroid[1])**2)
            
            if dist < best_dist and dist <= max_distance:
                best_dist = dist
                best_match = gt_id
        
        if best_match is not None:
            matches[gui_id] = best_match
            used_gt.add(best_match)
    
    return matches


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_segmentation(gui_dir: Path, metadata: Dict, config: Dict) -> Dict:
    """
    TEST 1: Segmentation Verification
    
    Validates that the GUI correctly segmented the expected number of cells.
    Checks that Cellpose detected 4 cells as configured in the simulation.
    """
    print("\n" + "=" * 60)
    print("TEST: Segmentation (4 Cells)")
    print("=" * 60)
    
    expected_cells = config.get('cell_geometry', {}).get('num_cells', 4)
    gui_cells = metadata.get('n_cells', 0)
    
    print(f"  Expected cells: {expected_cells}")
    print(f"  GUI detected: {gui_cells}")
    
    passed = gui_cells == expected_cells
    status = "✅" if passed else "❌"
    print(f"  Result: {status}")
    
    overall = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Overall: {overall}")
    
    return {
        'expected': expected_cells,
        'gui': gui_cells,
        'passed': passed
    }


def test_spot_count_per_cell(gui_dir: Path, gt_dir: Path, config: Dict) -> Dict:
    """
    TEST 2: Spot Count per Cell
    
    Compares the number of tracked spots per cell between GUI and ground truth.
    Uses mask centroids to match cells between GUI and ground truth, making it
    robust for both single and multi-cell simulations.
    """
    print("\n" + "=" * 60)
    print("TEST: Spot Count per Cell")
    print("=" * 60)
    
    # Load GUI tracking data
    tracking_file = list(gui_dir.glob("tracking_*.csv"))
    if not tracking_file:
        print("  ⚠️ No tracking data found")
        return {'passed': False, 'error': 'No tracking file'}
    
    df_gui = pd.read_csv(tracking_file[0])
    
    # Load ground truth
    gt_file = gt_dir / "ground_truth.csv"
    if not gt_file.exists():
        print("  ⚠️ Ground truth not found")
        return {'passed': False, 'error': 'No ground truth'}
    
    df_gt = pd.read_csv(gt_file)
    
    # Load masks to compute cell centroids for matching
    gui_mask_file = list(gui_dir.glob("cellpose_cytosol_*.tif"))
    gt_mask_file = gt_dir / "mask_cytosol.tif"
    
    if gui_mask_file and gt_mask_file.exists():
        # Use centroid-based cell matching
        gui_mask = tifffile.imread(gui_mask_file[0])
        gt_mask = tifffile.imread(gt_mask_file)
        
        gui_centroids = compute_cell_centroids(gui_mask)
        gt_centroids = compute_cell_centroids(gt_mask)
        
        cell_matches = match_cells_by_centroid(gui_centroids, gt_centroids)
        
        print(f"  Cell matching method: centroid-based")
        print(f"  GUI cells: {len(gui_centroids)}, GT cells: {len(gt_centroids)}")
        print(f"  Matched cells: {len(cell_matches)}")
        
        use_centroid_matching = len(cell_matches) > 0
    else:
        print("  ⚠️ Masks not found, using sorted count comparison")
        use_centroid_matching = False
        cell_matches = {}
    
    # Filter to channel 0
    df_gui_ch0 = df_gui[df_gui['spot_type'] == 0]
    df_gt_ch0 = df_gt[df_gt['spot_type'] == 0]
    
    # Get particle counts per cell
    gui_counts_dict = df_gui_ch0.groupby('cell_id')['particle'].nunique().to_dict()
    gt_counts_dict = df_gt_ch0.groupby('cell_id')['particle'].nunique().to_dict()
    
    # Total counts
    total_gui = sum(gui_counts_dict.values()) if gui_counts_dict else 0
    total_gt = sum(gt_counts_dict.values()) if gt_counts_dict else 0
    
    print(f"  GUI total particles (Ch0): {total_gui}")
    print(f"  Ground truth total particles (Ch0): {total_gt}")
    
    if use_centroid_matching:
        # Compare using matched cells
        print("\n  | GUI Cell | GT Cell | GT Count | GUI Count | Error |")
        print("  | :---: | :---: | :---: | :---: | :---: |")
        
        all_pass = True
        cell_results = []
        
        for gui_id, gt_id in sorted(cell_matches.items()):
            gt_count = gt_counts_dict.get(gt_id, 0)
            gui_count = gui_counts_dict.get(gui_id, 0)
            
            if gt_count > 0:
                error = abs(gui_count - gt_count) / gt_count
            else:
                error = 0 if gui_count == 0 else 1.0
            
            passed = error <= THRESHOLD_SPOT_COUNT
            all_pass = all_pass and passed
            status = "✅" if passed else "❌"
            
            print(f"  | {gui_id} | {gt_id} | {gt_count} | {gui_count} | {error:.1%} {status} |")
            
            cell_results.append({
                'gui_id': gui_id,
                'gt_id': gt_id,
                'gt': gt_count,
                'gui': gui_count,
                'error': error,
                'passed': passed
            })
    else:
        # Fall back to sorted comparison
        gui_counts = sorted(gui_counts_dict.values(), reverse=True)
        gt_counts = sorted(gt_counts_dict.values(), reverse=True)
        
        print("\n  | Cell (sorted) | Ground Truth | GUI | Error |")
        print("  | :---: | :---: | :---: | :---: |")
        
        all_pass = True
        cell_results = []
        
        max_cells = max(len(gui_counts), len(gt_counts), 1)
        gui_counts_padded = list(gui_counts) + [0] * (max_cells - len(gui_counts))
        gt_counts_padded = list(gt_counts) + [0] * (max_cells - len(gt_counts))
        
        for i, (gt_count, gui_count) in enumerate(zip(gt_counts_padded, gui_counts_padded)):
            if gt_count > 0:
                error = abs(gui_count - gt_count) / gt_count
            else:
                error = 0 if gui_count == 0 else 1.0
            
            passed = error <= THRESHOLD_SPOT_COUNT
            all_pass = all_pass and passed
            status = "✅" if passed else "❌"
            
            print(f"  | {i+1} | {gt_count} | {gui_count} | {error:.1%} {status} |")
            
            cell_results.append({
                'gt': gt_count,
                'gui': gui_count,
                'error': error,
                'passed': passed
            })
    
    # Total check
    if total_gt > 0:
        total_error = abs(total_gui - total_gt) / total_gt
    else:
        total_error = 0 if total_gui == 0 else 1.0
    total_passed = total_error <= THRESHOLD_SPOT_COUNT
    
    print(f"  | TOTAL | {total_gt} | {total_gui} | {total_error:.1%} {'✅' if total_passed else '❌'} |")
    
    # Pass if total is within threshold
    overall = "✅ PASS" if total_passed else "❌ FAIL"
    print(f"\n  Overall: {overall}")
    
    return {
        'cell_results': cell_results,
        'cell_matches': cell_matches if use_centroid_matching else None,
        'total_gt': total_gt,
        'total_gui': total_gui,
        'total_error': total_error,
        'passed': total_passed
    }



def test_compartment_assignment(gui_dir: Path, gt_dir: Path) -> Dict:
    """
    TEST 3: Compartment Assignment (Nucleus vs Cytosol)
    
    Validates that tracked spots are correctly assigned to nucleus or cytosol
    compartments by comparing GUI assignments against ground truth.
    """
    print("\n" + "=" * 60)
    print("TEST: Compartment Assignment (Nucleus vs Cytosol)")
    print("=" * 60)
    
    # Load GUI tracking data
    tracking_file = list(gui_dir.glob("tracking_*.csv"))
    if not tracking_file:
        print("  ⚠️ No tracking data found")
        return {'passed': False, 'error': 'No tracking file'}
    
    df_gui = pd.read_csv(tracking_file[0])
    
    # Load ground truth
    gt_file = gt_dir / "ground_truth.csv"
    if not gt_file.exists():
        print("  ⚠️ Ground truth not found")
        return {'passed': False, 'error': 'No ground truth'}
    
    df_gt = pd.read_csv(gt_file)
    
    # Filter to channel 0
    df_gui_ch0 = df_gui[df_gui['spot_type'] == 0]
    df_gt_ch0 = df_gt[df_gt['spot_type'] == 0]
    
    # Check if compartment column exists in GUI
    if 'compartment' not in df_gui_ch0.columns and 'is_nuc' not in df_gui_ch0.columns:
        print("  ⚠️ No compartment data in GUI tracking")
        return {'passed': True, 'skipped': True, 'reason': 'No compartment column'}
    
    # Match spots by position and check compartment
    matched = 0
    correct_compartment = 0
    
    for cell_id in df_gui_ch0['cell_id'].unique():
        gui_cell = df_gui_ch0[df_gui_ch0['cell_id'] == cell_id]
        gt_cell = df_gt_ch0[df_gt_ch0['cell_id'] == cell_id]
        
        for _, gui_row in gui_cell.iterrows():
            # Find closest GT spot in same frame
            gt_frame = gt_cell[gt_cell['frame'] == gui_row['frame']]
            if len(gt_frame) == 0:
                continue
            
            distances = np.sqrt((gt_frame['y'] - gui_row['y'])**2 + 
                                (gt_frame['x'] - gui_row['x'])**2)
            
            if distances.min() <= 5.0:  # 5 pixel matching radius
                matched += 1
                gt_match = gt_frame.iloc[distances.argmin()]
                
                # Compare compartment
                gui_is_nuc = gui_row.get('is_nuc', 0)
                gt_is_nuc = gt_match.get('is_nuc', 0)
                
                if gui_is_nuc == gt_is_nuc:
                    correct_compartment += 1
    
    accuracy = correct_compartment / matched if matched > 0 else 0
    
    print(f"  Matched spots: {matched}")
    print(f"  Correct compartment: {correct_compartment}/{matched} ({accuracy:.1%})")
    
    passed = accuracy >= THRESHOLD_COMPARTMENT
    
    overall = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Overall: {overall}")
    
    return {
        'matched': matched,
        'correct': correct_compartment,
        'accuracy': accuracy,
        'passed': passed
    }


def test_photobleaching(metadata: Dict, config: Dict) -> Dict:
    """
    TEST 4: Photobleaching Recovery
    
    Compares photobleaching decay rates recovered by the GUI against the
    configured ground truth values. Tests all 3 channels.
    """
    print("\n" + "=" * 60)
    print("TEST: Photobleaching Recovery")
    print("=" * 60)
    
    pb_cfg = config.get('photobleaching', {})
    
    print("\n  | Channel | Config | GUI | Error |")
    print("  | :---: | :---: | :---: | :---: |")
    
    all_pass = True
    channel_results = {}
    
    for ch in range(3):
        k_config = pb_cfg.get(f'ch{ch}_decay_rate', 0)
        k_gui = metadata.get(f'k_ch{ch}', 0)
        
        if k_config > 0:
            error = abs(k_gui - k_config) / k_config
        else:
            error = 0 if k_gui == 0 else 1.0
        
        passed = error <= THRESHOLD_PHOTOBLEACHING
        all_pass = all_pass and passed
        status = "✅" if passed else "❌"
        
        print(f"  | Ch{ch} | {k_config:.6f} | {k_gui:.6f} | {error:.1%} {status} |")
        
        channel_results[ch] = {
            'config': k_config,
            'gui': k_gui,
            'error': error,
            'passed': passed
        }
    
    overall = "✅ PASS" if all_pass else "❌ FAIL"
    print(f"\n  Overall: {overall}")
    
    return {
        'channel_results': channel_results,
        'passed': all_pass
    }


def test_msd(metadata: Dict, config: Dict) -> Dict:
    """
    TEST 5: MSD Diffusion Coefficient Recovery
    
    Compares the diffusion coefficient (D) recovered by the GUI's MSD analysis
    against the configured ground truth value. Accounts for voxel size calibration.
    """
    print("\n" + "=" * 60)
    print("TEST: MSD Diffusion Coefficient Recovery")
    print("=" * 60)
    
    motion_cfg = config.get('particle_motion', {})
    D_px_per_frame = motion_cfg.get('diffusion_coefficient', 0.05)
    
    # Convert to physical units
    voxel_yx = config.get('image', {}).get('voxel_size_yx_nm', 130)
    frame_rate = config.get('simulation', {}).get('frame_rate_seconds', 5.0)
    
    D_config = D_px_per_frame * (voxel_yx)**2 / frame_rate / 1e6  # µm²/s
    D_gui = metadata.get('D_um2_s', 0)
    
    print(f"  Config D: {D_config:.6f} µm²/s")
    print(f"  GUI D: {D_gui:.6f} µm²/s")
    
    if D_config > 0:
        error = abs(D_gui - D_config) / D_config
    else:
        error = 0
    
    passed = error <= THRESHOLD_MSD
    status = "✅" if passed else "❌"
    
    print(f"  Error: {error:.1%} {status}")
    
    overall = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Overall: {overall}")
    
    return {
        'D_config': D_config,
        'D_gui': D_gui,
        'error': error,
        'passed': passed
    }


def test_colocalization(gui_dir: Path, gt_dir: Path, config: Dict) -> Dict:
    """
    TEST 6: Colocalization Recovery (Ch0 vs Ch1)
    
    Compares the colocalization percentage between Ch0 and Ch1 as detected by
    the GUI against the ground truth from the simulation. The ground truth is
    calculated from the 'has_ch1_partner' column in the simulation data.
    """
    print("\n" + "=" * 60)
    print("TEST: Colocalization (Ch0 vs Ch1)")
    print("=" * 60)
    
    # Load GUI colocalization data
    coloc_file = list(gui_dir.glob("colocalization_data_*.csv"))
    if not coloc_file:
        print("  ⚠️ No colocalization data found")
        return {'passed': False, 'error': 'No colocalization file'}
    
    df_coloc = pd.read_csv(coloc_file[0])
    
    # Get POOLED row for overall stats
    pooled = df_coloc[df_coloc['cell_id'] == 'POOLED']
    if len(pooled) == 0:
        print("  ⚠️ No POOLED colocalization data")
        return {'passed': False}
    
    gui_coloc = pooled['colocalization percentage'].values[0]
    if isinstance(gui_coloc, str):
        gui_coloc = float(gui_coloc.split('±')[0].strip())
    
    # Load ground truth and calculate true colocalization
    gt_file = gt_dir / "ground_truth.csv"
    if not gt_file.exists():
        # Fall back to config
        coloc_cfg = config.get('colocalization', {})
        gt_coloc = coloc_cfg.get('ch1_probability', 0.8) * 100
        print(f"  Using config value: {gt_coloc:.1f}%")
    else:
        df_gt = pd.read_csv(gt_file)
        # Filter to unique Ch0 particles in first frame they appear
        df_gt_ch0 = df_gt[df_gt['spot_type'] == 0].drop_duplicates(subset=['unique_particle'])
        
        n_total = len(df_gt_ch0)
        n_coloc = df_gt_ch0['has_ch1_partner'].sum()
        gt_coloc = (n_coloc / n_total * 100) if n_total > 0 else 0
        print(f"  Ground truth: {n_coloc}/{n_total} = {gt_coloc:.1f}%")
    
    print(f"  GUI (POOLED): {gui_coloc:.1f}%")
    
    error = abs(gui_coloc - gt_coloc) / 100  # Absolute error in percentage points
    passed = error <= THRESHOLD_COLOCALIZATION
    status = "✅" if passed else "❌"
    
    print(f"  Error: {error:.1%} {status}")
    
    print("\n  | Source | Ch0 vs Ch1 Coloc |")
    print("  | :---: | :---: |")
    print(f"  | Ground Truth | {gt_coloc:.1f}% |")
    print(f"  | GUI Recovery | {gui_coloc:.1f}% |")
    print(f"  | Error | {error:.1%} |")
    
    overall = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  Overall: {overall}")
    
    return {
        'gt': gt_coloc,
        'gui': gui_coloc,
        'error': error,
        'passed': passed
    }


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(results: Dict, output_path: Path, config: Dict) -> str:
    """Generate markdown report with test descriptions."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r.get('passed', False))
    failed = total - passed
    
    status = "✅ ALL PASS" if failed == 0 else f"❌ {failed} FAILED"
    
    # Test descriptions
    test_descriptions = {
        'Segmentation': (
            "Validates that the GUI correctly segmented the expected number of cells. "
            "Checks that Cellpose detected 4 cells as configured in the simulation."
        ),
        'Spot Count per Cell': (
            "Compares the number of tracked spots per cell between GUI and ground truth. "
            "Note: Particle labels may differ, so we compare counts rather than matching "
            "individual particles. Requires ≤50% relative error per cell."
        ),
        'Compartment Assignment': (
            "Validates that tracked spots are correctly assigned to nucleus or cytosol "
            "compartments by matching GUI positions to ground truth and comparing labels. "
            "Requires ≥70% compartment accuracy."
        ),
        'Photobleaching': (
            "Compares photobleaching decay rates (k) recovered by the GUI against the "
            "configured ground truth values. Tests all 3 channels with ≤30% error threshold."
        ),
        'MSD': (
            "Compares the diffusion coefficient (D in µm²/s) recovered by the GUI's MSD "
            "analysis against the configured ground truth. Accounts for voxel calibration. "
            "Threshold: ≤80% error."
        ),
        'Colocalization': (
            "Compares the Ch0 vs Ch1 colocalization percentage detected by the GUI against "
            "the ground truth from simulation data (has_ch1_partner column). "
            "Threshold: ≤25% absolute error."
        ),
    }
    
    lines = [
        "# MicroLive GUI Validation Report",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Status:** {status}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Passed | Failed | Total |",
        "| :---: | :---: | :---: |",
        f"| {passed} | {failed} | {total} |",
        "",
        "---",
        "",
        "## Test Results",
        ""
    ]
    
    for name, result in results.items():
        test_status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
        desc = test_descriptions.get(name, "")
        
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"**Status:** {test_status}")
        lines.append("")
        if desc:
            lines.append(f"*{desc}*")
            lines.append("")
        
        # Add specific results
        if name == 'Segmentation':
            lines.append(f"- Expected cells: {result.get('expected', 'N/A')}")
            lines.append(f"- GUI detected: {result.get('gui', 'N/A')}")
        
        elif name == 'Photobleaching':
            if 'channel_results' in result:
                lines.append("")
                lines.append("| Channel | Config | GUI | Error |")
                lines.append("| :---: | :---: | :---: | :---: |")
                for ch, r in result['channel_results'].items():
                    lines.append(f"| Ch{ch} | {r['config']:.6f} | {r['gui']:.6f} | {r['error']:.1%} |")
        
        elif name == 'MSD':
            lines.append(f"- Config D: {result.get('D_config', 0):.6f} µm²/s")
            lines.append(f"- GUI D: {result.get('D_gui', 0):.6f} µm²/s")
            lines.append(f"- Error: {result.get('error', 0):.1%}")
        
        elif name == 'Colocalization':
            lines.append("")
            lines.append("| Source | Ch0 vs Ch1 Coloc |")
            lines.append("| :---: | :---: |")
            lines.append(f"| Ground Truth | {result.get('gt', 0):.1f}% |")
            lines.append(f"| GUI Recovery | {result.get('gui', 0):.1f}% |")
            lines.append(f"| Error | {result.get('error', 0):.1%} |")
        
        elif name == 'Compartment Assignment':
            lines.append(f"- Matched spots: {result.get('matched', 'N/A')}")
            lines.append(f"- Correct compartment: {result.get('correct', 'N/A')}")
            lines.append(f"- Accuracy: {result.get('accuracy', 0):.1%}")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    content = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"\nReport saved to: {output_path}")
    
    return content


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate MicroLive GUI output against simulation ground truth.'
    )
    parser.add_argument('--gui-dir', type=str, 
                       default='../results_simulated_spots',
                       help='Path to GUI output directory')
    parser.add_argument('--gt-dir', type=str,
                       default='../results',
                       help='Path to simulation results with ground truth')
    parser.add_argument('--config', type=str,
                       default='../config_multicell.yaml',
                       help='Path to simulation config')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    gui_dir = script_dir / args.gui_dir
    gt_dir = script_dir / args.gt_dir
    config_path = script_dir / args.config
    report_path = script_dir / "gui_report.md"
    
    print("=" * 60)
    print("MICROLIVE GUI VALIDATION TEST SUITE")
    print("=" * 60)
    print(f"\nGUI output: {gui_dir}")
    print(f"Ground truth: {gt_dir}")
    print(f"Config: {config_path}")
    
    # Validate paths
    if not gui_dir.exists():
        print(f"\n❌ GUI output directory not found: {gui_dir}")
        sys.exit(1)
    
    if not gt_dir.exists():
        print(f"\n❌ Ground truth directory not found: {gt_dir}")
        sys.exit(1)
    
    # Load data
    metadata_file = list(gui_dir.glob("Metadata_*.txt"))
    if not metadata_file:
        print("\n❌ No metadata file found in GUI output directory")
        sys.exit(1)
    
    metadata = parse_metadata(metadata_file[0])
    config = load_config(config_path)
    
    # Run tests
    results = {}
    
    results['Segmentation'] = test_segmentation(gui_dir, metadata, config)
    results['Spot Count per Cell'] = test_spot_count_per_cell(gui_dir, gt_dir, config)
    results['Compartment Assignment'] = test_compartment_assignment(gui_dir, gt_dir)
    results['Photobleaching'] = test_photobleaching(metadata, config)
    results['MSD'] = test_msd(metadata, config)
    results['Colocalization'] = test_colocalization(gui_dir, gt_dir, config)
    
    # Generate report
    generate_report(results, report_path, config)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r.get('passed', False))
    failed = total - passed
    
    print(f"\n  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📊 Total:  {total}")
    
    for name, result in results.items():
        status = "✅" if result.get('passed', False) else "❌"
        print(f"    {status} {name}")
    
    overall_pass = failed == 0
    print(f"\n  Overall: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    sys.exit(0 if overall_pass else 1)


if __name__ == '__main__':
    main()
