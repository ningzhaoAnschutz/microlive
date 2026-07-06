#!/usr/bin/env python3
"""
MicroLive GUI Validation Test Suite
====================================

Compares GUI analysis results against simulation ground truth.
Validates that the MicroLive GUI correctly recovers known simulation
parameters when analyzing exported data.

Tests:
  1. Segmentation: Verifies expected number of cells were detected
  2. Spot Count per Cell: Compares spot counts (centroid-based cell matching)
  3. Compartment Assignment: Validates nucleus/cytosol classification
  4. Photobleaching: Compares final retained intensity (GUI vs ground truth config)
  5. MSD: Compares diffusion coefficient (GUI vs ground truth config)
  6. Colocalization: Compares Ch0 vs Ch1 colocalization (GUI vs ground truth)
  7. is_colocalized Tracking: Validates per-particle is_colocalized column

Usage:
    # Single-cell validation (defaults)
    python run_test_gui.py

    # Multi-cell validation
    python run_test_gui.py \\
        --gui-dir ../kk_results_simulated_spots \\
        --gt-dir ../results_multicell \\
        --config ../config_multicell.yaml
"""

import argparse
import sys
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

from helpers import (
    load_config,
    parse_metadata,
    compute_cell_centroids,
    match_cells_by_centroid,
    load_cell_matches,
    align_matches_to_tracking_ids,
    print_summary,
)

# =============================================================================
# THRESHOLD CONSTANTS
# =============================================================================
THRESHOLD_PHOTOBLEACHING = 0.10    # ≤0.10 absolute error in final retained intensity
THRESHOLD_COLOCALIZATION = 0.25    # ≤25% absolute error on coloc %
THRESHOLD_MSD = 0.80               # ≤80% error on D (relaxed for GUI noise)
THRESHOLD_SPOT_COUNT = 0.60        # ≤60% error on spot count (relaxed - GUI may miss low-SNR spots)
THRESHOLD_COMPARTMENT = 0.70       # ≥70% compartment accuracy
THRESHOLD_IS_COLOCALIZED = 0.75    # ≥75% per-particle agreement with ground truth



def photobleaching_duration_seconds(config: Dict) -> float:
    """Return the last-frame acquisition time used for final retained comparisons.

    The simulator creates ``int(total_time_seconds / frame_rate_seconds)``
    frames, and the photobleaching fitter uses frame times starting at 0.
    Therefore a 600 s movie sampled every 5 s has 120 frames with a last
    frame at 595 s.
    """
    sim_cfg = config.get('simulation', {})
    total_time = float(sim_cfg.get('total_time_seconds', 0) or 0)
    frame_rate = float(sim_cfg.get('frame_rate_seconds', 1) or 1)
    if total_time <= 0 or frame_rate <= 0:
        return 0.0
    n_frames = int(total_time / frame_rate)
    return max(0.0, (n_frames - 1) * frame_rate)




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
    
    cell_matches, use_centroid_matching, gui_centroids, gt_centroids = load_cell_matches(
        gui_dir, gt_dir
    )
    
    if use_centroid_matching:
        print(f"  Cell matching method: centroid-based")
        print(f"  GUI cells: {len(gui_centroids)}, GT cells: {len(gt_centroids)}")
        print(f"  Matched cells: {len(cell_matches)}")
    else:
        print("  ⚠️ Masks not found, using sorted count comparison")
    
    # Filter to channel 0
    df_gui_ch0 = df_gui[df_gui['spot_type'] == 0]
    df_gt_ch0 = df_gt[df_gt['spot_type'] == 0]
    
    # Get particle counts per cell
    gui_counts_dict = df_gui_ch0.groupby('cell_id')['particle'].nunique().to_dict()
    gt_counts_dict = df_gt_ch0.groupby('cell_id')['particle'].nunique().to_dict()

    if use_centroid_matching:
        cell_matches, aligned_tracking_ids = align_matches_to_tracking_ids(
            cell_matches, gui_counts_dict.keys()
        )
        if aligned_tracking_ids:
            print("  Cell matching: aligned GUI tracking cell_ids to mask labels")
    
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
    
    unmatched_gui = []
    unmatched_gt = []
    matching_complete = True
    if use_centroid_matching:
        unmatched_gui = sorted(set(gui_counts_dict) - set(cell_matches))
        unmatched_gt = sorted(set(gt_counts_dict) - set(cell_matches.values()))
        matching_complete = len(unmatched_gui) == 0 and len(unmatched_gt) == 0
        if unmatched_gui:
            print(f"  ⚠️ Unmatched GUI cells with particles: {unmatched_gui}")
        if unmatched_gt:
            print(f"  ⚠️ Unmatched GT cells with particles: {unmatched_gt}")

    passed = total_passed and all_pass and matching_complete

    overall = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  Overall: {overall}")
    
    return {
        'cell_results': cell_results,
        'cell_matches': cell_matches if use_centroid_matching else None,
        'total_gt': total_gt,
        'total_gui': total_gui,
        'total_error': total_error,
        'total_passed': total_passed,
        'per_cell_passed': all_pass,
        'matching_complete': matching_complete,
        'unmatched_gui': unmatched_gui,
        'unmatched_gt': unmatched_gt,
        'passed': passed
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
    
    cell_matches, use_centroid_matching, _, _ = load_cell_matches(gui_dir, gt_dir)
    if use_centroid_matching:
        cell_matches, aligned_tracking_ids = align_matches_to_tracking_ids(
            cell_matches, df_gui_ch0['cell_id'].unique()
        )
        if aligned_tracking_ids:
            print("  Cell matching: aligned GUI tracking cell_ids to mask labels")
    if use_centroid_matching:
        print(f"  Cell matching method: centroid-based ({len(cell_matches)} matched)")
    else:
        print("  ⚠️ Masks not found, matching cells by raw cell_id")

    # Match spots by position and check compartment
    matched = 0
    correct_compartment = 0
    unmatched_cells = []
    
    for cell_id in df_gui_ch0['cell_id'].unique():
        gui_cell = df_gui_ch0[df_gui_ch0['cell_id'] == cell_id]
        if use_centroid_matching:
            gt_cell_id = cell_matches.get(cell_id)
            if gt_cell_id is None:
                unmatched_cells.append(cell_id)
                continue
        else:
            gt_cell_id = cell_id
        gt_cell = df_gt_ch0[df_gt_ch0['cell_id'] == gt_cell_id]
        
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
    if unmatched_cells:
        print(f"  ⚠️ Unmatched GUI cells skipped: {sorted(unmatched_cells)}")
    
    passed = accuracy >= THRESHOLD_COMPARTMENT and not unmatched_cells
    
    overall = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Overall: {overall}")
    
    return {
        'matched': matched,
        'correct': correct_compartment,
        'accuracy': accuracy,
        'cell_matches': cell_matches if use_centroid_matching else None,
        'unmatched_cells': unmatched_cells,
        'passed': passed
    }


def test_photobleaching(metadata: Dict, config: Dict) -> Dict:
    """
    TEST 4: Photobleaching Recovery
    
    Compares photobleaching final retained intensity recovered by the GUI
    against the configured ground truth values. Tests all 3 channels.
    """
    print("\n" + "=" * 60)
    print("TEST: Photobleaching Recovery")
    print("=" * 60)
    
    pb_cfg = config.get('photobleaching', {})
    duration_seconds = photobleaching_duration_seconds(config)
    print(f"  Duration for final retained intensity comparison: {duration_seconds:.1f} s")
    
    print("\n  | Channel | Config k | GUI k | Config retained | GUI retained | Abs retained error |")
    print("  | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    all_pass = True
    channel_results = {}
    
    for ch in range(3):
        k_config = pb_cfg.get(f'ch{ch}_decay_rate', 0)
        k_gui = metadata.get(f'k_ch{ch}', 0)

        retained_config = np.exp(-k_config * duration_seconds)
        retained_gui = np.exp(-k_gui * duration_seconds)
        error = abs(retained_gui - retained_config)
        
        passed = error <= THRESHOLD_PHOTOBLEACHING
        all_pass = all_pass and passed
        status = "✅" if passed else "❌"
        
        print(
            f"  | Ch{ch} | {k_config:.6f} | {k_gui:.6f} | "
            f"{retained_config:.3f} | {retained_gui:.3f} | {error:.3f} {status} |"
        )
        
        channel_results[ch] = {
            'config': k_config,
            'gui': k_gui,
            'retained_config': retained_config,
            'retained_gui': retained_gui,
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
    
    motion_cfg = config.get('motion', {})
    if not motion_cfg:
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


def test_is_colocalized_tracking(gui_dir: Path, gt_dir: Path, config: Dict) -> Dict:
    """
    TEST 7: is_colocalized Tracking Column Validation
    
    Validates that the per-particle 'is_colocalized' column in the exported
    tracking CSV correctly reflects the ground truth 'has_ch1_partner' column.
    
    This is the end-to-end test for §5.1-§5.8: it verifies that the full
    pipeline (tracking → colocalization → CropArray → ML/Intensity → 
    _apply_colocalization_to_tracking → export) produces correct per-particle
    colocalization labels by comparing against simulation ground truth.
    
    Matching strategy:
    - For each GUI particle (unique_particle), take first-frame position
    - Find closest GT particle (same cell, same frame) within 5px
    - Compare is_colocalized vs has_ch1_partner
    """
    print("\n" + "=" * 60)
    print("TEST: is_colocalized Tracking Column Validation")
    print("=" * 60)
    
    # Load GUI tracking data
    tracking_file = list(gui_dir.glob("tracking_*.csv"))
    if not tracking_file:
        print("  ⚠️ No tracking data found")
        return {'passed': False, 'error': 'No tracking file'}
    
    df_gui = pd.read_csv(tracking_file[0])
    
    # Check if is_colocalized column exists
    if 'is_colocalized' not in df_gui.columns:
        print("  ⚠️ No 'is_colocalized' column in tracking CSV")
        print("  This column is created when colocalization is run before export.")
        return {'passed': True, 'skipped': True, 'reason': 'No is_colocalized column'}
    
    # Load ground truth
    gt_file = gt_dir / "ground_truth.csv"
    if not gt_file.exists():
        print("  ⚠️ Ground truth not found")
        return {'passed': False, 'error': 'No ground truth'}
    
    df_gt = pd.read_csv(gt_file)
    
    # Filter to channel 0 (the tracked/analyzed channel)
    df_gui_ch0 = df_gui[df_gui['spot_type'] == 0].copy()
    df_gt_ch0 = df_gt[df_gt['spot_type'] == 0].copy()
    
    # Drop rows where is_colocalized is NaN (non-analyzed particles)
    df_gui_evaluated = df_gui_ch0.dropna(subset=['is_colocalized'])
    n_total_rows = len(df_gui_ch0)
    n_evaluated = len(df_gui_evaluated)
    n_nan = n_total_rows - n_evaluated
    
    print(f"  GUI Ch0 rows: {n_total_rows}")
    print(f"  Evaluated (non-NaN): {n_evaluated}")
    print(f"  NaN (non-analyzed): {n_nan}")
    
    if n_evaluated == 0:
        print("  ⚠️ No evaluated particles in is_colocalized column")
        return {'passed': False, 'error': 'All is_colocalized values are NaN'}
    
    # Determine particle ID column
    pid_col = 'unique_particle' if 'unique_particle' in df_gui_evaluated.columns else 'particle'
    
    # Get one representative row per GUI particle (first frame appearance)
    gui_particles = df_gui_evaluated.sort_values('frame').drop_duplicates(subset=[pid_col])
    n_gui_particles = len(gui_particles)
    print(f"  GUI unique particles (evaluated): {n_gui_particles}")
    
    # Load cell matching if available
    cell_matches, use_centroid_matching, _, _ = load_cell_matches(gui_dir, gt_dir)
    if use_centroid_matching:
        cell_matches, _ = align_matches_to_tracking_ids(
            cell_matches, df_gui_ch0['cell_id'].unique()
        )
        print(f"  Cell matching: centroid-based ({len(cell_matches)} matched)")
    else:
        print("  Cell matching: raw cell_id")
    
    # Match GUI particles to GT particles by position
    match_radius = 5.0
    matched = 0
    correct = 0
    true_positive = 0   # GUI=True, GT=True
    true_negative = 0   # GUI=False, GT=False
    false_positive = 0  # GUI=True, GT=False
    false_negative = 0  # GUI=False, GT=True
    
    for _, gui_row in gui_particles.iterrows():
        gui_y, gui_x = gui_row['y'], gui_row['x']
        gui_frame = gui_row['frame']
        gui_coloc = bool(gui_row['is_colocalized'])
        gui_cell = gui_row.get('cell_id', 0)
        
        # Map GUI cell to GT cell
        if use_centroid_matching:
            gt_cell = cell_matches.get(gui_cell)
            if gt_cell is None:
                continue
        else:
            gt_cell = gui_cell
        
        # Find GT spots in same cell and frame
        gt_frame = df_gt_ch0[
            (df_gt_ch0['frame'] == gui_frame) & 
            (df_gt_ch0['cell_id'] == gt_cell)
        ]
        
        if len(gt_frame) == 0:
            continue
        
        # Find closest GT spot
        distances = np.sqrt(
            (gt_frame['y'].values - gui_y)**2 + 
            (gt_frame['x'].values - gui_x)**2
        )
        min_idx = np.argmin(distances)
        
        if distances[min_idx] <= match_radius:
            matched += 1
            gt_row = gt_frame.iloc[min_idx]
            gt_coloc = bool(gt_row['has_ch1_partner'])
            
            if gui_coloc == gt_coloc:
                correct += 1
            
            # Confusion matrix
            if gui_coloc and gt_coloc:
                true_positive += 1
            elif not gui_coloc and not gt_coloc:
                true_negative += 1
            elif gui_coloc and not gt_coloc:
                false_positive += 1
            else:
                false_negative += 1
    
    if matched == 0:
        print("  ⚠️ No particles matched to ground truth")
        return {'passed': False, 'error': 'No matches'}
    
    accuracy = correct / matched
    
    # Derived metrics
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Matched to GT: {matched}/{n_gui_particles}")
    print(f"  Agreement: {correct}/{matched} ({accuracy:.1%})")
    print(f"")
    print(f"  Confusion matrix:")
    print(f"    True Positive  (GUI=T, GT=T): {true_positive}")
    print(f"    True Negative  (GUI=F, GT=F): {true_negative}")
    print(f"    False Positive (GUI=T, GT=F): {false_positive}")
    print(f"    False Negative (GUI=F, GT=T): {false_negative}")
    print(f"")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.1%}")
    
    passed = accuracy >= THRESHOLD_IS_COLOCALIZED
    
    overall = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Overall: {overall} (threshold: ≥{THRESHOLD_IS_COLOCALIZED:.0%} accuracy)")
    
    return {
        'n_gui_particles': n_gui_particles,
        'n_matched': matched,
        'n_correct': correct,
        'accuracy': accuracy,
        'true_positive': true_positive,
        'true_negative': true_negative,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'passed': passed
    }


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(results: Dict, output_path: Path, config: Dict) -> str:
    """Generate markdown report with test descriptions."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    total = len(results)
    skipped = sum(1 for r in results.values() if r.get('skipped', False))
    passed = sum(
        1 for r in results.values()
        if r.get('passed', False) and not r.get('skipped', False)
    )
    failed = sum(1 for r in results.values() if not r.get('passed', False))
    
    if failed > 0:
        status = f"❌ {failed} FAILED"
    elif skipped > 0:
        status = f"✅ PASS ({skipped} SKIPPED)"
    else:
        status = "✅ ALL PASS"
    
    # Test descriptions
    test_descriptions = {
        'Segmentation': (
            "Validates that the GUI correctly segmented the expected number of cells. "
            "Checks that Cellpose detected 4 cells as configured in the simulation."
        ),
        'Spot Count per Cell': (
            "Compares the number of tracked spots per cell between GUI and ground truth. "
            "Note: Particle labels may differ, so we compare counts rather than matching "
            "individual particles. Requires ≤60% relative error per cell and for total count."
        ),
        'Compartment Assignment': (
            "Validates that tracked spots are correctly assigned to nucleus or cytosol "
            "compartments by matching GUI positions to ground truth and comparing labels. "
            "Requires ≥70% compartment accuracy."
        ),
        'Photobleaching': (
            "Compares photobleaching final retained intensity exp(-k*T) recovered by the GUI "
            "against the configured ground truth values. Tests all 3 channels with "
            f"≤{THRESHOLD_PHOTOBLEACHING:.2f} absolute retained-intensity error threshold."
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
        'is_colocalized Tracking': (
            "End-to-end validation of the per-particle is_colocalized column in the "
            "exported tracking CSV. Matches GUI-tracked particles to ground truth by "
            "position and compares is_colocalized vs has_ch1_partner per particle. "
            f"Threshold: ≥{THRESHOLD_IS_COLOCALIZED:.0%} accuracy."
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
        "| Passed | Failed | Skipped | Total |",
        "| :---: | :---: | :---: | :---: |",
        f"| {passed} | {failed} | {skipped} | {total} |",
        "",
        "---",
        "",
        "## Test Results",
        ""
    ]
    
    for name, result in results.items():
        if result.get('skipped', False):
            test_status = "⚠️ SKIPPED"
        else:
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

        elif name == 'Spot Count per Cell':
            lines.append(f"- Ground-truth particles: {result.get('total_gt', 'N/A')}")
            lines.append(f"- GUI particles: {result.get('total_gui', 'N/A')}")
            lines.append(f"- Total error: {result.get('total_error', 0):.1%}")
            lines.append(f"- Per-cell pass: {'Yes' if result.get('per_cell_passed') else 'No'}")
            lines.append(f"- Cell matching complete: {'Yes' if result.get('matching_complete') else 'No'}")
        
        elif name == 'Photobleaching':
            if 'channel_results' in result:
                lines.append("")
                lines.append("| Channel | Config k | GUI k | Config retained | GUI retained | Abs retained error |")
                lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
                for ch, r in result['channel_results'].items():
                    lines.append(
                        f"| Ch{ch} | {r['config']:.6f} | {r['gui']:.6f} | "
                        f"{r['retained_config']:.3f} | {r['retained_gui']:.3f} | {r['error']:.3f} |"
                    )
        
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
        
        elif name == 'is_colocalized Tracking':
            lines.append(f"- GUI particles evaluated: {result.get('n_gui_particles', 'N/A')}")
            lines.append(f"- Matched to GT: {result.get('n_matched', 'N/A')}")
            lines.append(f"- Agreement: {result.get('n_correct', 'N/A')}/{result.get('n_matched', 'N/A')} ({result.get('accuracy', 0):.1%})")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| :--- | :---: |")
            lines.append(f"| True Positive | {result.get('true_positive', 0)} |")
            lines.append(f"| True Negative | {result.get('true_negative', 0)} |")
            lines.append(f"| False Positive | {result.get('false_positive', 0)} |")
            lines.append(f"| False Negative | {result.get('false_negative', 0)} |")
            lines.append(f"| Precision | {result.get('precision', 0):.1%} |")
            lines.append(f"| Recall | {result.get('recall', 0):.1%} |")
            lines.append(f"| F1 Score | {result.get('f1', 0):.1%} |")
        
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
                       default='../results_single_cell_gui',
                       help='Path to GUI output directory')
    parser.add_argument('--gt-dir', type=str,
                       default='../results_single_cell',
                       help='Path to simulation results with ground truth')
    parser.add_argument('--config', type=str,
                       default='../config_simple.yaml',
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
    results['is_colocalized Tracking'] = test_is_colocalized_tracking(gui_dir, gt_dir, config)
    
    # Generate report
    generate_report(results, report_path, config)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    overall_pass = print_summary(results)

    sys.exit(0 if overall_pass else 1)


if __name__ == '__main__':
    main()
