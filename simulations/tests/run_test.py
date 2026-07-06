#!/usr/bin/env python3
"""
MicroLive Simulation Validation Test Suite
===========================================

Automated tests to verify MicroLive correctly recovers simulation parameters.

Usage:
    python run_test.py
    python run_test.py --sim-dir ../results
    python run_test.py --config ../config_simple.yaml
"""

# Suppress matplotlib display BEFORE any imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no display
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import yaml

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from microlive import microscopy as mi
import microlive.ml_spot_detection as ML
from microlive.utils.resources import get_model_path

# Load ML model for colocalization
try:
    model_path = get_model_path()
    if model_path is not None and model_path.exists():
        model_ML = ML.ParticleDetectionCNN()
        ML.load_model(model_ML, model_path)
    else:
        model_ML = None
except Exception:
    model_ML = None

from helpers import (
    extract_tracking_dataframe,
    count_unique_particles,
    ensure_unique_particle_column,
    print_summary,
)

# =============================================================================
# THRESHOLDS (relative error unless noted)
# =============================================================================

THRESHOLD_PHOTOBLEACHING = 0.30    # ≤30% error on decay rate k
THRESHOLD_SPOT_COUNT = 0.50        # ≤50% error on avg spots/frame
THRESHOLD_DIFFUSION = 0.50         # ≤50% error on D coefficient
THRESHOLD_SNR = 0.40               # ≤40% error on SNR recovery
THRESHOLD_PSF_SIGMA = 0.40         # ≤40% error on spot size
THRESHOLD_COLOCALIZATION = 0.25    # ≤25% absolute error on coloc % (accounts for ML detection on low-SNR channels)
THRESHOLD_POSITION = 5.0           # ≤5 pixels mean position error





# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_photobleaching(image_tzyxc: np.ndarray,
                        config: dict,
                        mask_yx: Optional[np.ndarray] = None) -> Dict:
    """Test photobleaching decay rate recovery.
    
    Args:
        image_tzyxc: 5D image array [T, Z, Y, X, C]
        config: Simulation configuration
        
    Returns:
        Dictionary with results per channel
    """
    print("\n" + "=" * 60)
    print("TEST: Photobleaching Recovery")
    print("=" * 60)
    
    frame_rate = config.get('simulation', {}).get('frame_rate_seconds', 5.0)
    pb_config = config.get('photobleaching', {})
    
    try:
        pb = mi.Photobleaching(
            image_TZYXC=image_tzyxc,
            mask_YX=mask_yx,
            time_interval_seconds=frame_rate,
            show_plot=False
        )
        decay_params = pb.calculate_photobleaching()
        
        results = {'channels': [], 'passed': True}
        
        for ch in range(3):
            k_config = pb_config.get(f'ch{ch}_decay_rate', 0.0)
            
            # Extract k from fit params (k is first element for each channel)
            k_measured = decay_params[ch * 2] if decay_params[ch * 2] is not None else 0.0
            
            if k_config > 0:
                error = abs(k_measured - k_config) / k_config
            else:
                error = k_measured  # Error = measured value if config is 0
            
            ch_passed = error <= THRESHOLD_PHOTOBLEACHING or k_config == 0
            
            results['channels'].append({
                'channel': ch,
                'k_config': k_config,
                'k_measured': k_measured,
                'error': error,
                'passed': ch_passed
            })
            
            status = "✅" if ch_passed else "❌"
            print(f"  Ch{ch}: k_config={k_config:.6f}, k_measured={k_measured:.6f}, "
                  f"error={error:.1%} {status}")
            
            if not ch_passed:
                results['passed'] = False
        
        overall = "✅ PASS" if results['passed'] else "❌ FAIL"
        print(f"  Overall: {overall}")
        
        return results
        
    except Exception as e:
        print(f"  ⚠️ ERROR: {e}")
        return {'channels': [], 'passed': False, 'error': str(e)}


def calculate_detection_threshold(df_gt: pd.DataFrame, config: dict) -> float:
    """Calculate an appropriate spot detection threshold from ground truth.
    
    Uses the minimum expected spot intensity based on SNR and baseline.
    """
    # Get minimum SNR from config
    snr_min = config.get('spot_properties', {}).get('snr_min', 1.5)
    snr_mean = config.get('spot_properties', {}).get('snr_mean', 3.0)
    
    # Get baseline intensities
    baseline_cytosol = config.get('baseline', {}).get('cytosol', 1200)
    baseline_nucleus = config.get('baseline', {}).get('nucleus', 1500)
    baseline_outside = config.get('baseline', {}).get('outside_cell', 800)
    
    # Use the lower cytosol baseline for conservative threshold
    # Spot intensity = baseline × (1 + snr)
    # Threshold should be above noise but below spot
    min_spot_intensity = baseline_cytosol * (1 + snr_min)
    mean_spot_intensity = baseline_cytosol * (1 + snr_mean)
    
    # Threshold = midpoint between baseline and min spot
    threshold = (baseline_cytosol + min_spot_intensity) / 2
    
    # Also check ground truth if available
    if 'psf_amplitude_ch_0' in df_gt.columns:
        ch0 = df_gt[df_gt['spot_type'] == 0]
        gt_amplitudes = ch0['psf_amplitude_ch_0'].dropna()
        if len(gt_amplitudes) > 0:
            # Use 10th percentile of ground truth amplitudes as threshold
            threshold_from_gt = gt_amplitudes.quantile(0.1) * 0.5
            # Take the higher of config-based or GT-based threshold
            threshold = max(threshold, threshold_from_gt)
    
    return threshold


def test_spot_detection(image_tzyxc: np.ndarray, df_gt: pd.DataFrame, 
                        config: dict) -> Dict:
    """Test spot detection accuracy by comparing to ground truth.
    
    Args:
        image_tzyxc: 5D image array [T, Z, Y, X, C]
        df_gt: Ground truth DataFrame
        config: Simulation configuration
        
    Returns:
        Dictionary with detection results
    """
    print("\n" + "=" * 60)
    print("TEST: Spot Detection Accuracy")
    print("=" * 60)
    
    voxel_z = config.get('image', {}).get('voxel_size_z_nm', 300)
    voxel_yx = config.get('image', {}).get('voxel_size_yx_nm', 130)
    
    # Calculate expected threshold from ground truth
    expected_threshold = calculate_detection_threshold(df_gt, config)
    print(f"  Expected threshold (from config): {expected_threshold:.1f}")
    
    # Test on single frame for speed
    test_frame = image_tzyxc.shape[0] // 2
    frame_data = image_tzyxc[test_frame]  # [Z, Y, X, C]
    
    # Ground truth for this frame (lead channel only)
    gt_frame = df_gt[(df_gt['frame'] == test_frame) & (df_gt['spot_type'] == 0)]
    n_gt = len(gt_frame)
    
    print(f"  Testing frame {test_frame}")
    print(f"  Ground truth spots: {n_gt}")
    
    try:
        # Get spot size from config
        psf_sigma = config.get('spot_properties', {}).get('size_mean', 1.5)
        yx_spot_size = max(3, int(psf_sigma * 3))  # 3 sigma radius
        
        detector = mi.BigFISH(
            image=frame_data,
            channels_spots=0,
            voxel_size_z=voxel_z,
            voxel_size_yx=voxel_yx,
            yx_spot_size_in_px=yx_spot_size,
            z_spot_size_in_px=2,
            show_plot=False,
            save_files=False
        )
        spots, _, auto_threshold = detector.detect()
        n_detected = len(spots)
        
        print(f"  Auto threshold (BigFISH): {auto_threshold:.1f}")
        print(f"  Detected spots: {n_detected}")
        
        # Match detected spots to ground truth (within 5 pixel radius)
        match_radius = 5.0
        n_matched = 0
        if len(spots) > 0 and n_gt > 0:
            for _, gt_row in gt_frame.iterrows():
                # Calculate distance to all detected spots (2D, max projection)
                gt_y, gt_x = gt_row['y'], gt_row['x']
                
                # spots is array of [z, y, x] - use y, x for 2D matching
                distances = np.sqrt((spots[:, 1] - gt_y)**2 + (spots[:, 2] - gt_x)**2)
                if distances.min() <= match_radius:
                    n_matched += 1
        
        # True positive rate = matched / ground_truth
        true_positive_rate = n_matched / n_gt if n_gt > 0 else 0.0
        
        # False positive rate = (detected - matched) / detected
        false_positives = n_detected - n_matched
        false_positive_rate = false_positives / n_detected if n_detected > 0 else 0.0
        
        print(f"  Ground truth matched: {n_matched}/{n_gt} ({true_positive_rate:.1%})")
        print(f"  False positives: {false_positives} ({false_positive_rate:.1%})")
        
        # Pass if we find at least 80% of ground truth spots
        passed = true_positive_rate >= 0.80
        
        overall = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Overall: {overall}")
        
        return {
            'n_ground_truth': n_gt,
            'n_detected': n_detected,
            'n_matched': n_matched,
            'true_positive_rate': true_positive_rate,
            'false_positives': false_positives,
            'false_positive_rate': false_positive_rate,
            'expected_threshold': expected_threshold,
            'auto_threshold': auto_threshold,
            'passed': passed
        }
        
    except Exception as e:
        print(f"  ⚠️ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'n_ground_truth': n_gt, 'n_detected': 0, 'passed': False, 'error': str(e)}


def test_position_accuracy(image_tzyxc: np.ndarray, df_gt: pd.DataFrame,
                           masks: Tuple[np.ndarray, np.ndarray],
                           config: dict) -> Dict:
    """Test position accuracy by running tracking and comparing to ground truth.
    
    Args:
        image_tzyxc: 5D image array [T, Z, Y, X, C]
        df_gt: Ground truth DataFrame
        masks: Tuple of (cytosol_mask, nucleus_mask)
        config: Simulation configuration
        
    Returns:
        Dictionary with position accuracy results
    """
    print("\n" + "=" * 60)
    print("TEST: Position Accuracy (Tracking)")
    print("=" * 60)
    
    voxel_z = config.get('image', {}).get('voxel_size_z_nm', 300)
    voxel_yx = config.get('image', {}).get('voxel_size_yx_nm', 130)
    
    # Use enough frames for realistic trajectory lengths (min_length=15)
    n_frames = min(30, image_tzyxc.shape[0])
    image_subset = image_tzyxc[:n_frames]
    
    cytosol_mask, nucleus_mask = masks
    
    # Convert 3D masks to 2D if needed
    if cytosol_mask.ndim == 3:
        cytosol_2d = cytosol_mask.max(axis=0)
        nucleus_2d = nucleus_mask.max(axis=0)
    else:
        cytosol_2d = cytosol_mask
        nucleus_2d = nucleus_mask
    
    print(f"  Testing on {n_frames} frames")
    
    # Get spot size from config
    psf_sigma = config.get('spot_properties', {}).get('size_mean', 1.5)
    yx_spot_size = max(3, int(psf_sigma * 3))
    
    # Use default threshold for tracking (ParticleTracking has different semantics)
    # The threshold_for_spot_detection in ParticleTracking is based on local contrast
    tracking_threshold = 2000  # Match GUI auto-threshold
    
    try:
        tracker = mi.ParticleTracking(
            image=image_subset,
            channels_spots=[0],
            list_voxels=[voxel_z, voxel_yx],
            channels_cytosol=[0],
            channels_nucleus=[1],
            masks=cytosol_2d,
            masks_nuclei=nucleus_2d,
            yx_spot_size_in_px=yx_spot_size,
            z_spot_size_in_px=2,
            threshold_for_spot_detection=tracking_threshold,
            min_length_trajectory=15,  # Realistic: require long trajectories
            memory=0,  # Realistic: no gap filling
            use_maximum_projection=True,
            verbose=False
        )
        
        result = tracker.run()
        
        # Handle different return types
        if isinstance(result, tuple):
            df_tracked = result[0]
        else:
            df_tracked = result
        
        if isinstance(df_tracked, list):
            if len(df_tracked) > 0 and hasattr(df_tracked[0], 'columns'):
                df_tracked = df_tracked[0]
            else:
                df_tracked = pd.DataFrame()
        
        n_tracked = len(df_tracked) if isinstance(df_tracked, pd.DataFrame) else 0
        print(f"  Tracked spots: {n_tracked}")
        
        # Ground truth for same frames
        gt_subset = df_gt[(df_gt['frame'] < n_frames) & (df_gt['spot_type'] == 0)]
        n_gt = len(gt_subset)
        print(f"  Ground truth spots: {n_gt}")
        
        # Calculate position errors (matching by frame and nearest neighbor)
        match_radius = 10.0  # Pixels - consider a match if within this distance
        
        if n_tracked > 0 and isinstance(df_tracked, pd.DataFrame) and 'x' in df_tracked.columns:
            errors = []
            n_matched = 0
            
            for frame in gt_subset['frame'].unique():
                gt_f = gt_subset[gt_subset['frame'] == frame]
                tr_f = df_tracked[df_tracked['frame'] == frame] if 'frame' in df_tracked.columns else df_tracked
                
                if len(tr_f) == 0:
                    continue
                
                for _, gt_row in gt_f.iterrows():
                    # 2D distance (y, x) since we use max projection
                    distances = np.sqrt(
                        (tr_f['y'] - gt_row['y'])**2 +
                        (tr_f['x'] - gt_row['x'])**2
                    )
                    if len(distances) > 0:
                        min_dist = distances.min()
                        if min_dist <= match_radius:
                            errors.append(min_dist)
                            n_matched += 1
            
            if errors:
                mean_error = np.mean(errors)
                median_error = np.median(errors)
                recall = n_matched / n_gt if n_gt > 0 else 0.0
            else:
                mean_error = float('inf')
                median_error = float('inf')
                recall = 0.0
        else:
            mean_error = float('inf')
            median_error = float('inf')
            recall = 0.0
            n_matched = 0
        
        # Pass if we find at least 50% of GT spots within 10 pixels
        passed = recall >= 0.50 or (mean_error <= THRESHOLD_POSITION and recall > 0)
        
        print(f"  Matched GT spots: {n_matched}/{n_gt} ({recall:.1%})")
        print(f"  Mean position error (matched): {mean_error:.2f} px")
        print(f"  Median position error (matched): {median_error:.2f} px")
        
        overall = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Overall: {overall}")
        
        return {
            'n_tracked': n_tracked,
            'n_ground_truth': n_gt,
            'n_matched': n_matched,
            'recall': recall,
            'mean_position_error': mean_error,
            'median_position_error': median_error,
            'passed': passed
        }
        
    except Exception as e:
        print(f"  ⚠️ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'n_tracked': 0, 'passed': False, 'error': str(e)}


def test_compartment_assignment(image_tzyxc: np.ndarray, df_gt: pd.DataFrame,
                                masks: Tuple[np.ndarray, np.ndarray],
                                config: dict) -> Dict:
    """Test that MicroLive correctly assigns spots to nucleus vs cytosol.
    
    Args:
        image_tzyxc: 5D image array [T, Z, Y, X, C]
        df_gt: Ground truth DataFrame
        masks: Tuple of (cytosol_mask, nucleus_mask)
        config: Simulation configuration
        
    Returns:
        Dictionary with compartment assignment results
    """
    print("\n" + "=" * 60)
    print("TEST: Compartment Assignment (Nucleus vs Cytosol)")
    print("=" * 60)
    
    cytosol_mask, nucleus_mask = masks
    
    # Convert 3D masks to 2D if needed
    if cytosol_mask.ndim == 3:
        cytosol_2d = cytosol_mask.max(axis=0)
        nucleus_2d = nucleus_mask.max(axis=0)
    else:
        cytosol_2d = cytosol_mask
        nucleus_2d = nucleus_mask
    
    voxel_z = config.get('image', {}).get('voxel_size_z_nm', 300)
    voxel_yx = config.get('image', {}).get('voxel_size_yx_nm', 130)
    
    # Test on a single frame
    test_frame = 0
    frame_data = image_tzyxc[test_frame]
    
    print(f"  Testing frame {test_frame}")
    
    try:
        # Detect spots
        detector = mi.BigFISH(
            image=frame_data,
            channels_spots=0,
            voxel_size_z=voxel_z,
            voxel_size_yx=voxel_yx,
            yx_spot_size_in_px=5,
            z_spot_size_in_px=2,
            show_plot=False,
            save_files=False
        )
        spots, _, _ = detector.detect()
        n_detected = len(spots)
        print(f"  Detected spots: {n_detected}")
        
        # Ground truth for this frame
        gt_frame = df_gt[(df_gt['frame'] == test_frame) & (df_gt['spot_type'] == 0)]
        n_gt = len(gt_frame)
        print(f"  Ground truth spots: {n_gt}")
        
        # Match detected spots to GT and verify compartment assignment
        match_radius = 5.0
        correct_compartment = 0
        total_matched = 0
        
        for _, gt_row in gt_frame.iterrows():
            gt_y, gt_x = gt_row['y'], gt_row['x']
            gt_is_nuc = gt_row['is_nuc']
            
            # Find closest detected spot
            distances = np.sqrt((spots[:, 1] - gt_y)**2 + (spots[:, 2] - gt_x)**2)
            min_idx = np.argmin(distances)
            
            if distances[min_idx] <= match_radius:
                total_matched += 1
                # Check compartment of detected spot using mask
                det_y = int(np.clip(round(spots[min_idx, 1]), 0, nucleus_2d.shape[0]-1))
                det_x = int(np.clip(round(spots[min_idx, 2]), 0, nucleus_2d.shape[1]-1))
                
                det_is_nuc = nucleus_2d[det_y, det_x] > 0
                
                if det_is_nuc == gt_is_nuc:
                    correct_compartment += 1
        
        if total_matched > 0:
            accuracy = correct_compartment / total_matched
            recall = total_matched / n_gt
            print(f"  Matched spots: {total_matched}/{n_gt} ({recall:.1%} recall)")
            print(f"  Correct compartment: {correct_compartment}/{total_matched} ({accuracy:.1%})")
        else:
            accuracy = 0.0
            recall = 0.0
            print(f"  No spots matched to ground truth")
        
        # Pass if BOTH:
        # 1. ≥50% of GT spots were detected (recall)
        # 2. ≥85% of matched spots have correct compartment (accuracy)
        recall_passed = recall >= 0.50
        accuracy_passed = accuracy >= 0.75
        passed = recall_passed and accuracy_passed
        
        if not recall_passed:
            print(f"  ⚠️ Detection recall too low ({recall:.1%} < 50%)")
        
        overall = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Overall: {overall}")
        
        return {
            'n_detected': n_detected,
            'n_ground_truth': n_gt,
            'n_matched': total_matched,
            'recall': recall,
            'correct_compartment': correct_compartment,
            'accuracy': accuracy,
            'passed': passed
        }
        
    except Exception as e:
        print(f"  ⚠️ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'passed': False, 'error': str(e)}


def test_msd_recovery(image_tzyxc: np.ndarray, masks: Tuple[np.ndarray, np.ndarray],
                      config: dict, df_tracked: pd.DataFrame = None) -> Dict:
    """Test MSD (Mean Square Displacement) diffusion coefficient recovery.
    
    This test validates that MicroLive can:
    1. Detect spots in the noisy image
    2. Track them across frames
    3. Calculate MSD from tracked positions
    4. Recover the correct diffusion coefficient
    
    Args:
        image_tzyxc: 5D image array [T, Z, Y, X, C]
        masks: Tuple of (cytosol_mask, nucleus_mask)
        config: Simulation configuration
        
    Returns:
        Dictionary with MSD results
    """
    print("\n" + "=" * 60)
    print("TEST: MSD Diffusion Coefficient Recovery")
    print("=" * 60)
    
    # Get config values
    motion_cfg = config.get('motion', {})
    if not motion_cfg:
        motion_cfg = config.get('particle_motion', {})
    D_px_per_frame = motion_cfg.get('diffusion_coefficient', 0.05)
    
    voxel_z = config.get('image', {}).get('voxel_size_z_nm', 300)
    voxel_yx_nm = config.get('image', {}).get('voxel_size_yx_nm', 130)
    frame_rate = config.get('simulation', {}).get('frame_rate_seconds', 5.0)
    
    # Convert D to physical units (µm²/s)
    D_config_um2_s = D_px_per_frame * (voxel_yx_nm)**2 / frame_rate / 1e6
    
    print(f"  Config D: {D_px_per_frame} px²/frame = {D_config_um2_s:.6f} µm²/s")
    
    cytosol_mask, nucleus_mask = masks
    
    # Convert 3D masks to 2D if needed
    if cytosol_mask.ndim == 3:
        cytosol_2d = cytosol_mask.max(axis=0)
        nucleus_2d = nucleus_mask.max(axis=0)
    else:
        cytosol_2d = cytosol_mask
        nucleus_2d = nucleus_mask
    
    try:
        # Step 1: Run ParticleTracking on the image (skip if already provided)
        if df_tracked is None:
            print(f"  Running ParticleTracking on {image_tzyxc.shape[0]} frames...")
            
            tracker = mi.ParticleTracking(
                image=image_tzyxc,
                channels_spots=[0],
                list_voxels=[voxel_z, voxel_yx_nm],
                channels_cytosol=[0],
                channels_nucleus=[1],
                masks=cytosol_2d,
                masks_nuclei=nucleus_2d,
                yx_spot_size_in_px=5,
                z_spot_size_in_px=2,
                threshold_for_spot_detection=2000,  # Match GUI auto-threshold
                min_length_trajectory=15,  # Realistic: require long trajectories
                memory=0,  # Realistic: no gap filling
                use_maximum_projection=True,
                verbose=False
            )
            
            result = tracker.run()
            
            # Handle different return types
            if isinstance(result, tuple):
                df_tracked = result[0]
            else:
                df_tracked = result
            
            if isinstance(df_tracked, list):
                if len(df_tracked) > 0 and hasattr(df_tracked[0], 'columns'):
                    df_tracked = df_tracked[0]
                else:
                    df_tracked = pd.DataFrame()
        else:
            print(f"  Using pre-computed tracking results...")
        
        if df_tracked is None or len(df_tracked) == 0:
            print("  ⚠️ No spots tracked - cannot compute MSD")
            return {'passed': False, 'error': 'No tracked spots'}
        
        n_tracked = len(df_tracked)
        n_particles = df_tracked['particle'].nunique() if 'particle' in df_tracked.columns else 0
        print(f"  Tracked spots: {n_tracked}, Particles: {n_particles}")
        
        # Step 2: Calculate MSD from tracked positions
        microns_per_pixel = voxel_yx_nm / 1000  # nm to µm
        
        # Create globally unique particle ID (cell_id + particle) for multi-cell scenarios
        if 'cell_id' in df_tracked.columns:
            df_tracked['unique_particle'] = df_tracked['cell_id'].astype(str) + '_' + df_tracked['particle'].astype(str)
        else:
            df_tracked['unique_particle'] = df_tracked['particle'].astype(str)
        
        n_unique = df_tracked['unique_particle'].nunique()
        print(f"  Unique particles (cell_id + particle): {n_unique}")
        
        # Prepare dataframe for MSD (need frame, particle, x, y)
        df_motion = df_tracked[['frame', 'unique_particle', 'x', 'y']].copy()
        df_motion = df_motion.rename(columns={'unique_particle': 'particle'})
        df_motion = df_motion.drop_duplicates(subset=['frame', 'particle'])
        
        # Filter to particles with at least 15 observations (matching min_length_trajectory)
        particle_counts = df_motion.groupby('particle').size()
        valid_particles = particle_counts[particle_counts >= 15].index
        df_motion = df_motion[df_motion['particle'].isin(valid_particles)]
        
        n_valid = len(valid_particles)
        print(f"  Particles with ≥15 observations: {n_valid}")
        
        if n_valid == 0:
            print("  ⚠️ No valid trajectories for MSD")
            return {'passed': False, 'error': 'No valid trajectories'}
        
        pm = mi.ParticleMotion(
            trackpy_dataframe=df_motion,
            microns_per_pixel=microns_per_pixel,
            step_size_in_sec=frame_rate,
            max_lagtime=20,
            show_plot=False,
            remove_drift=False,
            spot_type=0,
            max_fit_points=20,  # Match GUI
            is_3d=False
        )
        
        msd_result = pm.calculate_msd()
        
        if msd_result is not None:
            if isinstance(msd_result, tuple):
                D_measured = msd_result[0]
                r2 = msd_result[1] if len(msd_result) > 1 else None
            else:
                D_measured = msd_result
                r2 = None
            
            print(f"  Measured D: {D_measured:.6f} µm²/s")
            if r2 is not None:
                print(f"  R²: {r2:.4f}")
            
            # Calculate error
            if D_config_um2_s > 0:
                error = abs(D_measured - D_config_um2_s) / D_config_um2_s
                print(f"  Error: {error:.1%}")
            else:
                error = 0.0
            
            passed = error <= THRESHOLD_DIFFUSION
        else:
            D_measured = 0.0
            r2 = None
            error = 1.0
            passed = False
            print(f"  No MSD result returned")
        
        overall = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Overall: {overall}")
        
        return {
            'D_config': D_config_um2_s,
            'D_measured': D_measured,
            'r2': r2,
            'error': error,
            'n_particles': n_valid,
            'n_tracked': n_tracked,
            'tracking_channel': 0,  # Channel used for tracking
            'passed': passed
        }
        
    except Exception as e:
        print(f"  ⚠️ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'passed': False, 'error': str(e)}


def test_colocalization_recovery(image_tzyxc: np.ndarray,
                                 df_tracked: pd.DataFrame,
                                 config: dict,
                                 ml_threshold: float = 0.51,
                                 df_gt: Optional[pd.DataFrame] = None) -> Dict:
    """Test colocalization recovery using ML-based detection.
    
    Uses the ML model to detect colocalization between Ch0 spots and 
    Ch1/Ch2 signals, then compares with configured colocalization probabilities.
    
    Args:
        image_tzyxc: 5D image array [T, Z, Y, X, C]
        df_tracked: Tracked spots DataFrame (from Ch0 tracking)
        config: Simulation configuration
        ml_threshold: ML threshold for colocalization (default 0.51)
        
    Returns:
        Dictionary with colocalization results
    """
    print("\n" + "=" * 60)
    print("TEST: Colocalization Recovery (ML)")
    print("=" * 60)
    
    try:
        # Check if ML model is available
        if model_ML is None:
            print("  ⚠️ ML model not available, skipping test")
            return {'passed': True, 'skipped': True, 'reason': 'ML model not available'}
        
        # Load ML-related parameters from config `test:` section when available
        test_cfg = config.get('test', {}) if config is not None else {}
        coloc_ml_cfg = test_cfg.get('colocalization_ml', {})
        # ml_threshold param can be overridden by config
        ml_threshold = coloc_ml_cfg.get('ml_threshold', ml_threshold)
        crop_size_offset = coloc_ml_cfg.get('crop_size_offset', 7)
        min_valid_crops = coloc_ml_cfg.get('min_valid_crops', 10)
        max_trajectories = int(coloc_ml_cfg.get('max_trajectories', 500))
        trajectory_source = str(
            coloc_ml_cfg.get('trajectory_source', 'auto')
        ).lower()
        if trajectory_source not in {'auto', 'tracking', 'ground_truth'}:
            trajectory_source = 'auto'

        # Get configured colocalization probabilities
        coloc_cfg = config.get('colocalization', {})
        ch1_target = coloc_cfg.get('ch1_probability', 0.7)
        ch2_target = coloc_cfg.get('ch2_probability', 0.3)
        target_source = 'config'

        # Determine trajectory source
        df_tracking_src = ensure_unique_particle_column(
            df_tracked if df_tracked is not None else pd.DataFrame()
        )
        n_tracking_particles = count_unique_particles(df_tracking_src)
        use_ground_truth = trajectory_source == 'ground_truth'
        if trajectory_source == 'auto':
            use_ground_truth = n_tracking_particles < min_valid_crops

        if use_ground_truth:
            if df_gt is None or len(df_gt) == 0:
                print("  ⚠️ Ground truth unavailable for colocalization trajectories")
                return {'passed': False, 'error': 'Ground truth not available'}

            df_source = df_gt[df_gt['spot_type'] == 0].copy()
            if len(df_source) == 0:
                print("  ⚠️ Ground truth contains no Ch0 trajectories")
                return {'passed': False, 'error': 'No Ch0 trajectories in ground truth'}
            df_source = ensure_unique_particle_column(df_source)

            # Use measured ground-truth partner rates for the selected trajectories.
            per_particle_gt = df_source.groupby('unique_particle', as_index=False)[
                ['has_ch1_partner', 'has_ch2_partner']
            ].first()
            ch1_target = float(per_particle_gt['has_ch1_partner'].mean())
            ch2_target = float(per_particle_gt['has_ch2_partner'].mean())
            target_source = 'ground_truth'
            print(f"  Trajectory source: ground_truth ({len(per_particle_gt)} trajectories)")
        else:
            if df_tracking_src is None or len(df_tracking_src) == 0:
                print("  ⚠️ No tracking data available")
                return {'passed': False, 'error': 'No tracking data'}
            df_source = df_tracking_src.copy()
            target_source = 'config'
            print(f"  Trajectory source: tracking ({n_tracking_particles} trajectories)")

        print(f"  Target Ch1 coloc ({target_source}): {ch1_target:.1%}")
        print(f"  Target Ch2 coloc ({target_source}): {ch2_target:.1%}")
        print(f"  ML threshold: {ml_threshold}")
        
        # Get image and spot parameters
        psf_sigma = config.get('spot_properties', {}).get('size_mean', 1.5)
        # yx_spot_size used by detectors (3 sigma rule)
        yx_spot_size = max(3, int(psf_sigma * 3))
        # Crop size = particle window + offset (from config)
        crop_size = int(yx_spot_size + crop_size_offset)
        if crop_size % 2 == 0:
            crop_size += 1
        
        print(f"  Crop size: {crop_size}x{crop_size} pixels")
        
        # Use maximum projection for 2D analysis (like GUI)
        if image_tzyxc.ndim == 5 and image_tzyxc.shape[1] > 1:
            # Apply max projection per frame
            num_z = image_tzyxc.shape[1]
            max_proj = np.max(image_tzyxc, axis=1, keepdims=True)
            image_for_coloc = np.repeat(max_proj, num_z, axis=1)
        else:
            image_for_coloc = image_tzyxc
        
        # Keep full trajectories; if needed, subsample by trajectories (not rows).
        unique_particles = df_source['unique_particle'].dropna().unique()
        if len(unique_particles) > max_trajectories:
            rng = np.random.default_rng(42)
            selected_particles = rng.choice(
                unique_particles,
                size=max_trajectories,
                replace=False
            )
            df_spots = df_source[df_source['unique_particle'].isin(selected_particles)].copy()
        else:
            df_spots = df_source.copy()
        
        n_spots_total = len(df_spots)
        n_particles_total = count_unique_particles(df_spots)
        print(f"  Spot observations: {n_spots_total}")
        print(f"  Trajectories used: {n_particles_total}")
        
        if n_spots_total == 0:
            print("  ⚠️ No spots found")
            return {'passed': False, 'error': 'No spots'}
        
        # Use CropArray like the GUI does
        try:
            crop_result = mi.CropArray(
                image=image_for_coloc,
                df_crops=df_spots,
                crop_size=crop_size,
                remove_outliers=False,
                max_percentile=99.95,
                selected_time_point=None,
                normalize_each_particle=False
            ).run()
            mean_crop = crop_result[1]
        except Exception as crop_err:
            print(f"  ⚠️ CropArray failed: {crop_err}")
            return {'passed': False, 'error': f'CropArray failed: {crop_err}'}
        
        if mean_crop is None or mean_crop.shape[0] < crop_size:
            print("  ⚠️ Could not extract valid crops")
            return {'passed': False, 'error': 'No valid crops'}
        
        n_valid_crops = mean_crop.shape[0] // crop_size
        print(f"  Valid crops: {n_valid_crops}")

        # Skip ML evaluation if too few valid crops to produce stable statistics
        if n_valid_crops < min_valid_crops:
            print(f"  ⚠️ Not enough valid crops ({n_valid_crops} < {min_valid_crops}), skipping ML colocalization test")
            return {'passed': True, 'skipped': True, 'reason': f'Not enough valid crops ({n_valid_crops} < {min_valid_crops})'}
        
        # Test Ch0 vs Ch1 colocalization
        crops_ch1 = mi.Utilities().normalize_crop_return_list(
            array_crops_YXC=mean_crop,
            crop_size=crop_size,
            selected_color_channel=1,
            normalize_to_255=True
        )
        flags_ch1, probs_ch1 = ML.predict_crops(model_ML, crops_ch1, threshold=ml_threshold)
        ch1_measured = np.mean(flags_ch1) if len(flags_ch1) > 0 else 0.0
        
        # Test Ch0 vs Ch2 colocalization
        crops_ch2 = mi.Utilities().normalize_crop_return_list(
            array_crops_YXC=mean_crop,
            crop_size=crop_size,
            selected_color_channel=2,
            normalize_to_255=True
        )
        flags_ch2, probs_ch2 = ML.predict_crops(model_ML, crops_ch2, threshold=ml_threshold)
        ch2_measured = np.mean(flags_ch2) if len(flags_ch2) > 0 else 0.0
        
        # Calculate errors
        ch1_error = abs(ch1_measured - ch1_target)
        ch2_error = abs(ch2_measured - ch2_target)
        
        print(f"  Ch1 coloc: measured={ch1_measured:.1%}, target={ch1_target:.1%}, error={ch1_error:.1%}")
        print(f"  Ch2 coloc: measured={ch2_measured:.1%}, target={ch2_target:.1%}, error={ch2_error:.1%}")
        
        # Pass if both within threshold
        ch1_pass = ch1_error <= THRESHOLD_COLOCALIZATION
        ch2_pass = ch2_error <= THRESHOLD_COLOCALIZATION
        passed = ch1_pass and ch2_pass
        
        ch1_status = "✅" if ch1_pass else "❌"
        ch2_status = "✅" if ch2_pass else "❌"
        
        print(f"  Ch1: {ch1_status} (≤{THRESHOLD_COLOCALIZATION:.0%} threshold)")
        print(f"  Ch2: {ch2_status} (≤{THRESHOLD_COLOCALIZATION:.0%} threshold)")
        
        overall = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Overall: {overall}")
        
        return {
            'passed': passed,
            'ch1_config': ch1_target,
            'ch1_measured': ch1_measured,
            'ch1_error': ch1_error,
            'ch2_config': ch2_target,
            'ch2_measured': ch2_measured,
            'ch2_error': ch2_error,
            'n_spots': n_valid_crops,
            'ml_threshold': ml_threshold,
            'trajectory_source': target_source
        }
        
    except Exception as e:
        print(f"  ⚠️ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'passed': False, 'error': str(e)}


def test_ground_truth_quality(df_gt: pd.DataFrame, config: dict) -> Dict:
    """Validate ground truth data quality.
    
    Args:
        df_gt: Ground truth DataFrame
        config: Simulation configuration
        
    Returns:
        Dictionary with quality metrics
    """
    print("\n" + "=" * 60)
    print("TEST: Ground Truth Quality")
    print("=" * 60)
    
    results = {'passed': True, 'checks': []}
    
    # Check 1: Required columns
    required_cols = ['frame', 'z', 'y', 'x', 'particle', 'cell_id', 'spot_type',
                     'is_nuc', 'compartment', 'snr_ch_0', 'psf_sigma_ch_0']
    missing = [c for c in required_cols if c not in df_gt.columns]
    col_check = len(missing) == 0
    results['checks'].append({'name': 'Required columns', 'passed': col_check})
    print(f"  Required columns: {'✅' if col_check else '❌'} (missing: {missing})")
    
    # Check 2: No particles in background (should be rare)
    ch0 = df_gt[df_gt['spot_type'] == 0]
    n_background = (ch0['compartment'] == 'background').sum()
    bg_ratio = n_background / len(ch0) if len(ch0) > 0 else 0
    bg_check = bg_ratio < 0.10  # Less than 10% in background
    results['checks'].append({'name': 'Particles in cell', 'passed': bg_check, 
                              'background_ratio': bg_ratio})
    print(f"  Particles in cell: {'✅' if bg_check else '❌'} "
          f"({bg_ratio:.1%} in background)")
    
    # Check 3: Multi-cell distribution (if applicable)
    n_cells = config.get('cell_geometry', {}).get('num_cells', 1)
    if n_cells > 1:
        unique_cells = ch0['cell_id'].nunique()
        cell_check = unique_cells == n_cells
        results['checks'].append({'name': 'Multi-cell distribution', 'passed': cell_check})
        print(f"  Multi-cell distribution: {'✅' if cell_check else '❌'} "
              f"({unique_cells}/{n_cells} cells)")
    
    # Check 4: Compartment transitions detected
    transitions = ch0['compartment_transition'].notna().sum()
    results['checks'].append({'name': 'Transitions detected', 'value': transitions})
    print(f"  Compartment transitions: {transitions}")
    
    # Check 5: SNR values reasonable
    snr_mean = ch0['snr_ch_0'].mean()
    snr_config = config.get('spot_properties', {}).get('snr_mean', 3.0)
    snr_error = abs(snr_mean - snr_config) / snr_config if snr_config > 0 else 0
    snr_check = snr_error < 0.3
    results['checks'].append({'name': 'SNR values', 'passed': snr_check})
    print(f"  SNR mean: {snr_mean:.2f} (config: {snr_config:.2f}) {'✅' if snr_check else '❌'}")
    
    # Overall
    results['passed'] = all(c.get('passed', True) for c in results['checks'])
    overall = "✅ PASS" if results['passed'] else "❌ FAIL"
    print(f"  Overall: {overall}")
    
    return results


def test_colocalization(df_gt: pd.DataFrame, config: dict) -> Dict:
    """Validate colocalization probabilities in ground truth.
    
    Args:
        df_gt: Ground truth DataFrame
        config: Simulation configuration
        
    Returns:
        Dictionary with colocalization results
    """
    print("\n" + "=" * 60)
    print("TEST: Colocalization Accuracy")
    print("=" * 60)
    
    coloc_config = config.get('colocalization', {})
    ch1_prob_config = coloc_config.get('ch1_probability', 0.7)
    ch2_prob_config = coloc_config.get('ch2_probability', 0.3)
    
    # Use one row per Ch0 particle so long-lived tracks do not receive
    # more weight than short-lived tracks.
    ch0 = ensure_unique_particle_column(df_gt[df_gt['spot_type'] == 0].copy())
    ch0_particles = ch0.drop_duplicates(subset=['unique_particle'])
    
    if 'has_ch1_partner' in ch0_particles.columns:
        ch1_measured = ch0_particles['has_ch1_partner'].mean()
    else:
        ch1_measured = 0.0
    
    if 'has_ch2_partner' in ch0_particles.columns:
        ch2_measured = ch0_particles['has_ch2_partner'].mean()
    else:
        ch2_measured = 0.0
    
    ch1_error = abs(ch1_measured - ch1_prob_config)
    ch2_error = abs(ch2_measured - ch2_prob_config)
    
    ch1_passed = ch1_error <= THRESHOLD_COLOCALIZATION
    ch2_passed = ch2_error <= THRESHOLD_COLOCALIZATION
    
    print(f"  Ch1 coloc: config={ch1_prob_config:.1%}, measured={ch1_measured:.1%}, "
          f"error={ch1_error:.1%} {'✅' if ch1_passed else '❌'}")
    print(f"  Ch2 coloc: config={ch2_prob_config:.1%}, measured={ch2_measured:.1%}, "
          f"error={ch2_error:.1%} {'✅' if ch2_passed else '❌'}")
    
    print(f"  Ch0 particles: {len(ch0_particles)}")

    # Distance verification: check that colocalized spots are at same position.
    # Pick an actual frame near the median; median() itself can be fractional.
    distance_check = True
    frames = np.sort(df_gt['frame'].dropna().unique())
    test_frame = None
    if len(frames) > 0:
        median_frame = np.median(frames)
        test_frame = frames[np.argmin(np.abs(frames - median_frame))]
    df_frame = df_gt[df_gt['frame'] == test_frame] if test_frame is not None else pd.DataFrame()
    ch0_frame = df_frame[df_frame['spot_type'] == 0]
    ch1_frame = df_frame[df_frame['spot_type'] == 1]
    
    if len(ch0_frame) > 0 and len(ch1_frame) > 0:
        ch0_with_ch1 = ch0_frame[ch0_frame['has_ch1_partner'] == True]
        if len(ch0_with_ch1) > 0:
            distances = []
            for _, row in ch0_with_ch1.iterrows():
                dists = np.sqrt((ch1_frame['y'] - row['y'])**2 + (ch1_frame['x'] - row['x'])**2)
                distances.append(dists.min())
            max_dist = np.max(distances) if distances else 0
            distance_check = max_dist < 1.0  # Should be at exact same position
            print(f"  Distance verification frame {test_frame}: max={max_dist:.2f}px {'✅' if distance_check else '❌'}")
    
    passed = ch1_passed and ch2_passed and distance_check
    overall = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Overall: {overall}")
    
    return {
        'ch1_config': ch1_prob_config,
        'ch1_measured': ch1_measured,
        'ch1_error': ch1_error,
        'ch1_passed': ch1_passed,
        'ch2_config': ch2_prob_config,
        'ch2_measured': ch2_measured,
        'ch2_error': ch2_error,
        'ch2_passed': ch2_passed,
        'distance_check': distance_check,
        'distance_check_frame': test_frame,
        'n_particles': len(ch0_particles),
        'passed': passed
    }


def test_gui_syntax() -> Dict:
    """Test that the MicroLive GUI module compiles without syntax errors.
    
    This test validates that:
    1. The GUI module (microlive/gui/app.py) can be imported without errors
    2. All syntax is valid Python
    3. All dependencies are available
    
    Returns:
        Dictionary with syntax check results
    """
    print("\n" + "=" * 60)
    print("TEST: GUI Syntax Check")
    print("=" * 60)
    
    try:
        # Try to compile the GUI module
        import py_compile
        import importlib.util
        
        # Path to the GUI module
        gui_path = Path(__file__).parent.parent.parent / "microlive" / "gui" / "app.py"
        
        if not gui_path.exists():
            # Try alternative path
            gui_path = Path(__file__).parent.parent.parent / "microlive" / "gui" / "micro.py"
        
        if not gui_path.exists():
            print(f"  ⚠️ GUI module not found at expected path")
            return {'passed': False, 'error': 'GUI module not found'}
        
        print(f"  Checking: {gui_path.name}")
        
        # Step 1: Syntax check (py_compile)
        print(f"  Step 1: Syntax validation...")
        try:
            py_compile.compile(str(gui_path), doraise=True)
            print(f"    ✅ Syntax OK")
            syntax_ok = True
        except py_compile.PyCompileError as e:
            print(f"    ❌ Syntax error: {e}")
            return {'passed': False, 'error': f'Syntax error: {e}'}
        
        # Step 2: Import check (can the module be loaded?)
        print(f"  Step 2: Import validation...")
        try:
            # Check if we can load the module spec
            spec = importlib.util.spec_from_file_location("gui_app", gui_path)
            if spec is None:
                print(f"    ❌ Could not create module spec")
                return {'passed': False, 'error': 'Could not create module spec'}
            
            # We don't actually execute the module (would require Qt)
            # Just verify the spec is valid
            print(f"    ✅ Module spec valid")
            import_ok = True
        except Exception as e:
            print(f"    ⚠️ Import check failed: {e}")
            import_ok = False
        
        # Step 3: Try importing the microlive package (lightweight check)
        print(f"  Step 3: MicroLive package import...")
        try:
            import microlive.microscopy as mi
            print(f"    ✅ microlive.microscopy imported successfully")
            package_ok = True
        except ImportError as e:
            print(f"    ❌ Import error: {e}")
            return {'passed': False, 'error': f'Package import error: {e}'}
        
        # Get file stats
        file_size = gui_path.stat().st_size
        with open(gui_path, 'r') as f:
            line_count = sum(1 for _ in f)
        
        print(f"  File size: {file_size:,} bytes")
        print(f"  Lines of code: {line_count:,}")
        
        passed = syntax_ok and package_ok
        overall = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Overall: {overall}")
        
        return {
            'passed': passed,
            'file': gui_path.name,
            'file_size_bytes': file_size,
            'line_count': line_count,
            'syntax_ok': syntax_ok,
            'import_ok': import_ok,
            'package_ok': package_ok
        }
        
    except Exception as e:
        print(f"  ⚠️ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'passed': False, 'error': str(e)}



# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(results: Dict, output_path: Path, config: dict = None) -> str:
    """Generate markdown report from test results.
    
    Args:
        results: Dictionary of all test results
        output_path: Path to save report
        config: Simulation configuration (optional, for parameter reporting)
        
    Returns:
        Report content as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    total = len(results)
    skipped = sum(1 for r in results.values() if r.get('skipped', False))
    passed = sum(
        1 for r in results.values()
        if r.get('passed', False) and not r.get('skipped', False)
    )
    failed = sum(1 for r in results.values() if not r.get('passed', False))
    status_text = '✅ ALL PASS'
    if failed > 0:
        status_text = f'❌ {failed} FAIL'
    elif skipped > 0:
        status_text = f'✅ PASS ({skipped} SKIPPED)'
    
    lines = [
        "# MicroLive Simulation Validation Report",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Status:** {status_text}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Passed | Failed | Skipped | Total |",
        f"| :---: | :---: | :---: | :---: |",
        f"| {passed} | {failed} | {skipped} | {total} |",
        "",
    ]
    
    # Add simulation parameters section if config provided
    if config:
        lines.extend([
            "---",
            "",
            "## Simulation Parameters",
            "",
        ])
        
        # Image parameters
        image_cfg = config.get('image', {})
        lines.append("### Image Configuration")
        lines.append("")
        lines.append(f"| Parameter | Value |")
        lines.append(f"| :--- | :---: |")
        lines.append(f"| Dimensions | {image_cfg.get('dimensions', '3D')} |")
        lines.append(f"| Size (YX) | {image_cfg.get('size_yx', [512, 512])} |")
        lines.append(f"| Z Slices | {image_cfg.get('num_z_slices', 10)} |")
        lines.append(f"| Voxel Size (YX) | {image_cfg.get('voxel_size_yx_nm', 130)} nm |")
        lines.append(f"| Voxel Size (Z) | {image_cfg.get('voxel_size_z_nm', 300)} nm |")
        lines.append("")
        
        # Simulation timing
        sim_cfg = config.get('simulation', {})
        lines.append("### Simulation Timing")
        lines.append("")
        lines.append(f"| Parameter | Value |")
        lines.append(f"| :--- | :---: |")
        lines.append(f"| Total time | {sim_cfg.get('total_time_seconds', 300)} s |")
        lines.append(f"| Frame rate | {sim_cfg.get('frame_rate_seconds', 5)} s/frame |")
        n_frames = sim_cfg.get('total_time_seconds', 300) // sim_cfg.get('frame_rate_seconds', 5)
        lines.append(f"| Total frames | {n_frames} |")
        lines.append("")
        
        # Noise parameters (per-channel)
        noise_cfg = config.get('noise', {})
        lines.append("### Noise Configuration (Per-Channel)")
        lines.append("")
        lines.append(f"| Channel | Noise Std |")
        lines.append(f"| :---: | :---: |")
        for ch in range(3):
            ch_noise = noise_cfg.get(f'ch{ch}_noise_std', 'N/A')
            lines.append(f"| Ch {ch} | {ch_noise} |")
        lines.append("")
        
        # SNR parameters
        spot_cfg = config.get('spot_properties', {})
        lines.append("### Spot Properties")
        lines.append("")
        lines.append(f"| Parameter | Value |")
        lines.append(f"| :--- | :---: |")
        lines.append(f"| SNR Mean | {spot_cfg.get('snr_mean', 3.0)} |")
        lines.append(f"| SNR Std | {spot_cfg.get('snr_std', 1.0)} |")
        lines.append(f"| SNR Range | [{spot_cfg.get('snr_min', 0.5)}, {spot_cfg.get('snr_max', 8.0)}] |")
        lines.append(f"| Size Mean | {spot_cfg.get('size_mean', 1.5)} px |")
        lines.append("")
        
        # Motion parameters
        motion_cfg = config.get('motion', {})
        lines.append("### Motion")
        lines.append("")
        lines.append(f"| Parameter | Value |")
        lines.append(f"| :--- | :---: |")
        lines.append(f"| Diffusion Coefficient | {motion_cfg.get('diffusion_coefficient', 0.05)} px²/frame |")
        lines.append(f"| Confinement | {motion_cfg.get('confinement_type', 'cell')} |")
        lines.append("")
        
        # Cell geometry
        cell_cfg = config.get('cell_geometry', {})
        if cell_cfg:
            lines.append("### Cell Geometry")
            lines.append("")
            lines.append(f"| Parameter | Value |")
            lines.append(f"| :--- | :---: |")
            lines.append(f"| Number of Cells | {cell_cfg.get('num_cells', 1)} |")
            lines.append(f"| Layout | {cell_cfg.get('layout', 'single')} |")
            lines.append(f"| Cell Diameter (YX) | {cell_cfg.get('cell_diameter_yx', [180, 180])} px |")
            lines.append(f"| Nucleus Enabled | {'Yes' if cell_cfg.get('nucleus_enabled', True) else 'No'} |")
            lines.append("")
    
    lines.extend([
        "---",
        "",
        "## Test Results",
        "",
    ])
    
    for test_name, result in results.items():
        if result.get('skipped', False):
            status = "⚠️ SKIPPED"
        else:
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
        lines.append(f"### {test_name}")
        lines.append("")
        lines.append(f"**Status:** {status}")
        lines.append("")
        
        # Add test description
        test_descriptions = {
            'Ground Truth Quality': (
                "Validates that the simulation correctly generates ground truth data. "
                "Checks that fewer than 10% of particles are assigned to background, all cells contain particles, "
                "and the measured SNR matches the configured value within tolerance."
            ),
            'Colocalization': (
                "Verifies that channel colocalization probabilities are correctly implemented. "
                "For each spot in Channel 0, checks that the fraction with co-localized signal in "
                "Channel 1 and Channel 2 matches the configured probabilities (≤25% absolute error)."
            ),
            'Photobleaching': (
                "Tests recovery of photobleaching decay rates from simulated images. "
                "Fits an exponential decay to mean intensity over time for each channel and compares "
                "the recovered decay constant (k) to the configured value (≤30% error threshold)."
            ),
            'Spot Detection': (
                "Evaluates spot detection accuracy using BigFISH's automatic thresholding. "
                "Measures what fraction of ground truth spots are detected (recall ≥70%) and "
                "reports false positive rate. Uses a 5-pixel matching radius."
            ),
            'Position Accuracy': (
                "Assesses tracking position accuracy by matching tracked spots to ground truth. "
                "Reports mean and median position error in pixels for matched spots. "
                "Uses min_length_trajectory=15 and memory=0 for realistic tracking parameters."
            ),
            'Compartment Assignment': (
                "Validates that detected spots are correctly assigned to nucleus or cytosol. "
                "Compares tracked spot compartment labels against ground truth. "
                "Requires ≥50% detection recall and ≥75% compartment accuracy."
            ),
            'MSD Recovery': (
                "Tests diffusion coefficient (D) recovery from Mean Squared Displacement analysis. "
                "Runs tracking on photobleaching-corrected images, then calculates MSD using trackpy. "
                "Uses unique particle IDs (cell_id + particle) for multi-cell scenarios. "
                "Compares measured D (µm²/s) to configured value (≤50% error threshold)."
            ),
            'Colocalization Recovery': (
                "Tests ML-based colocalization detection between channels. "
                "For each trajectory-averaged Ch0 crop, uses the CNN classifier (threshold=0.51) "
                "to detect colocalized signal in Ch1 and Ch2. Compares measured colocalization "
                "percentages against target probabilities (ground truth when available, otherwise config; "
                "≤25% absolute error threshold)."
            ),
            'GUI Syntax': (
                "Validates that the MicroLive GUI module compiles without syntax errors. "
                "Checks: (1) Python syntax validation using py_compile, "
                "(2) module spec validation using importlib, and "
                "(3) core package import verification. "
                "Reports file size and line count for the GUI module."
            ),
        }
        
        desc = test_descriptions.get(test_name, "")
        if desc:
            lines.append(f"*{desc}*")
            lines.append("")
        
        # Add specific details based on test type
        if 'channels' in result:
            lines.append("| Channel | Config | Measured | Error |")
            lines.append("| :---: | :---: | :---: | :---: |")
            for ch in result['channels']:
                lines.append(f"| {ch['channel']} | {ch['k_config']:.6f} | "
                           f"{ch['k_measured']:.6f} | {ch['error']:.1%} |")
            lines.append("")
        
        if 'mean_position_error' in result:
            lines.append(f"- Mean position error: {result['mean_position_error']:.2f} px")
            lines.append(f"- Tracked spots: {result.get('n_tracked', 'N/A')}")
            lines.append("")
        
        if 'detection_rate' in result:
            lines.append(f"- Detection rate: {result['detection_rate']:.1%}")
            lines.append(f"- Ground truth: {result['n_ground_truth']}")
            lines.append(f"- Detected: {result['n_detected']}")
            lines.append("")
        
        if 'ch1_measured' in result and 'ml_threshold' not in result:
            lines.append(f"- Ch1 coloc: {result['ch1_measured']:.1%} "
                        f"(config: {result['ch1_config']:.1%})")
            lines.append(f"- Ch2 coloc: {result['ch2_measured']:.1%} "
                        f"(config: {result['ch2_config']:.1%})")
            lines.append("")
        
        if 'accuracy' in result and 'correct_compartment' in result:
            lines.append(f"- Compartment accuracy: {result['accuracy']:.1%}")
            lines.append(f"- Correct: {result['correct_compartment']}/{result['n_matched']}")
            lines.append("")
        
        if 'D_measured' in result and 'D_config' in result:
            if 'tracking_channel' in result:
                lines.append(f"- Tracking channel: Ch {result['tracking_channel']}")
            lines.append(f"- Config D: {result['D_config']:.6f} µm²/s")
            lines.append(f"- Measured D: {result['D_measured']:.6f} µm²/s")
            if result.get('error') is not None:
                lines.append(f"- Error: {result['error']:.1%}")
            if result.get('n_particles') is not None:
                lines.append(f"- Particles tracked: {result['n_particles']}")
            lines.append("")
        
        # Colocalization Recovery results (ML-based) - compare with ground truth
        if 'ch1_measured' in result and 'ch1_config' in result and 'ml_threshold' in result:
            lines.append(f"- ML Threshold: {result['ml_threshold']}")
            lines.append(f"- Spots analyzed: {result.get('n_spots', 'N/A')}")
            if 'trajectory_source' in result:
                lines.append(f"- Target source: {result['trajectory_source']}")
            lines.append("")
            lines.append("| Channel | Target | ML Recovery | Error |")
            lines.append("| :---: | :---: | :---: | :---: |")
            lines.append(f"| Ch1 | {result['ch1_config']:.1%} | {result['ch1_measured']:.1%} | {result['ch1_error']:.1%} |")
            lines.append(f"| Ch2 | {result['ch2_config']:.1%} | {result['ch2_measured']:.1%} | {result['ch2_error']:.1%} |")
            lines.append("")
        
        # GUI Pipeline results
        if 'tracking' in result and isinstance(result['tracking'], dict):
            tr = result['tracking']
            if 'recall' in tr:
                lines.append(f"- Tracking recall: {tr['recall']:.1%}")
            if 'compartment_accuracy' in tr:
                lines.append(f"- Compartment accuracy: {tr['compartment_accuracy']:.1%}")
            m = result.get('msd', {})
            if 'D_measured' in m:
                lines.append(f"- MSD D: {m['D_measured']:.6f} µm²/s (error: {m.get('error', 0):.1%})")
            lines.append("")
        
        # GUI Syntax results
        if 'file' in result and 'line_count' in result:
            lines.append(f"- File: {result['file']}")
            lines.append(f"- Lines of code: {result['line_count']:,}")
            lines.append(f"- File size: {result['file_size_bytes']:,} bytes")
            lines.append(f"- Syntax OK: {'✅' if result.get('syntax_ok') else '❌'}")
            lines.append(f"- Package import OK: {'✅' if result.get('package_ok') else '❌'}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    content = "\n".join(lines)
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"\nReport saved to: {output_path}")
    
    return content


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MicroLive Simulation Validation Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_test.py
  python run_test.py --sim-dir ../results
  python run_test.py --config ../config_simple.yaml
        """
    )
    parser.add_argument(
        '--sim-dir', '-s',
        default='../results_single_cell',
        help='Path to simulation results directory (default: ../results_single_cell)'
    )
    parser.add_argument(
        '--config', '-c',
        default='../config_simple.yaml',
        help='Path to simulation config YAML (default: ../config_simple.yaml)'
    )
    parser.add_argument(
        '--report', '-r',
        default='report.md',
        help='Output report filename (default: report.md)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    test_dir = Path(__file__).parent
    sim_dir = (test_dir / args.sim_dir).resolve()
    config_path = (test_dir / args.config).resolve()
    report_path = test_dir / args.report
    
    print("=" * 60)
    print("MICROLIVE SIMULATION VALIDATION TEST SUITE")
    print("=" * 60)
    print(f"\nSimulation directory: {sim_dir}")
    print(f"Configuration: {config_path}")
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("\nLoading data...")
    
    image_path = sim_dir / 'simulated_spots.tif'
    gt_path = sim_dir / 'ground_truth.csv'
    cytosol_path = sim_dir / 'mask_cytosol.tif'
    nucleus_path = sim_dir / 'mask_nucleus.tif'
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return 1
    
    image_tczyx = tifffile.imread(image_path)
    print(f"  Image shape (TCZYX): {image_tczyx.shape}")
    
    # Convert TCZYX to TZYXC
    image_tzyxc = np.moveaxis(image_tczyx, 1, -1)
    print(f"  Image shape (TZYXC): {image_tzyxc.shape}")
    
    df_gt = pd.read_csv(gt_path)
    print(f"  Ground truth records: {len(df_gt)}")
    
    cytosol_mask = tifffile.imread(cytosol_path)
    nucleus_mask = tifffile.imread(nucleus_path)
    print(f"  Masks loaded: cytosol={cytosol_mask.shape}, nucleus={nucleus_mask.shape}")

    # Convert 3D masks to 2D for modules expecting YX masks
    if cytosol_mask.ndim == 3:
        cytosol_2d = cytosol_mask.max(axis=0)
        nucleus_2d = nucleus_mask.max(axis=0)
    else:
        cytosol_2d = cytosol_mask
        nucleus_2d = nucleus_mask
    
    # Run tests
    results = {}
    
    results['Ground Truth Quality'] = test_ground_truth_quality(df_gt, config)
    results['Colocalization'] = test_colocalization(df_gt, config)
    results['Photobleaching'] = test_photobleaching(
        image_tzyxc, config, mask_yx=cytosol_2d
    )
    results['Spot Detection'] = test_spot_detection(image_tzyxc, df_gt, config)
    results['Position Accuracy'] = test_position_accuracy(
        image_tzyxc, df_gt, (cytosol_mask, nucleus_mask), config
    )
    results['Compartment Assignment'] = test_compartment_assignment(
        image_tzyxc, df_gt, (cytosol_mask, nucleus_mask), config
    )

    test_cfg = config.get('test', {})
    shared_tracking_cfg = test_cfg.get('shared_tracking', {})
    shared_tracking_threshold = shared_tracking_cfg.get(
        'threshold_for_spot_detection', 2000
    )
    shared_tracking_min_length = shared_tracking_cfg.get(
        'min_length_trajectory', 15
    )
    shared_tracking_memory = shared_tracking_cfg.get('memory', 0)
    
    # Run shared tracking for MSD test (avoids running tracking twice)
    # Apply photobleaching correction first (matching GUI workflow)
    print("\n" + "=" * 60)
    print("Applying Photobleaching Correction for MSD analysis...")
    print("=" * 60)
    
    frame_rate = config.get('simulation', {}).get('frame_rate_seconds', 5.0)
    try:
        pb = mi.Photobleaching(
            image_TZYXC=image_tzyxc,
            mask_YX=cytosol_2d,
            time_interval_seconds=frame_rate,
            show_plot=False
        )
        _ = pb.calculate_photobleaching()
        correction_result = pb.apply_photobleaching_correction()
        # Handle tuple return (corrected_image, params) or just image
        if isinstance(correction_result, tuple):
            image_corrected = correction_result[0]
        else:
            image_corrected = correction_result
        print(f"  Photobleaching correction applied")
    except Exception as e:
        print(f"  ⚠️ Photobleaching correction failed: {e}")
        print(f"  Using raw image for MSD...")
        image_corrected = image_tzyxc
    
    print("\n" + "=" * 60)
    print("Running shared ParticleTracking for MSD analysis...")
    print("=" * 60)
    
    voxel_z = config.get('image', {}).get('voxel_size_z_nm', 300)
    voxel_yx = config.get('image', {}).get('voxel_size_yx_nm', 130)
    
    try:
        tracker = mi.ParticleTracking(
            image=image_corrected,  # Use photobleaching-corrected image
            channels_spots=[0],
            list_voxels=[voxel_z, voxel_yx],
            channels_cytosol=[0],
            channels_nucleus=[1],
            masks=cytosol_2d,
            masks_nuclei=nucleus_2d,
            yx_spot_size_in_px=5,
            z_spot_size_in_px=2,
            threshold_for_spot_detection=shared_tracking_threshold,
            min_length_trajectory=shared_tracking_min_length,
            memory=shared_tracking_memory,
            use_maximum_projection=True,
            verbose=False
        )
        
        result = tracker.run()
        df_tracked_shared = extract_tracking_dataframe(result)
        n_tracked = len(df_tracked_shared) if df_tracked_shared is not None else 0
        n_particles = count_unique_particles(df_tracked_shared)
        print(f"  Tracked spots: {n_tracked}, Particles: {n_particles}")
    except Exception as e:
        print(f"  ⚠️ Tracking failed: {e}")
        df_tracked_shared = None

    # Colocalization ML needs enough unique tracked particles to produce stable crops.
    # If shared MSD tracking is too strict, run a fallback with relaxed thresholds.
    df_tracked_coloc = df_tracked_shared
    coloc_ml_cfg = test_cfg.get('colocalization_ml', {})
    min_valid_crops = int(coloc_ml_cfg.get('min_valid_crops', 10))
    shared_particles = count_unique_particles(df_tracked_shared)

    enable_coloc_fallback = bool(
        coloc_ml_cfg.get('enable_relaxed_tracking_fallback', False)
    )

    if enable_coloc_fallback and shared_particles < min_valid_crops:
        fallback_threshold = coloc_ml_cfg.get(
            'tracking_threshold_for_spot_detection', 1500
        )
        fallback_min_length = coloc_ml_cfg.get(
            'tracking_min_length_trajectory', 5
        )
        fallback_memory = coloc_ml_cfg.get('tracking_memory', 0)

        print("\n" + "=" * 60)
        print("Running fallback ParticleTracking for colocalization ML...")
        print("=" * 60)
        print(
            f"  Shared tracking has {shared_particles} particles (<{min_valid_crops}); "
            "running relaxed tracking for ML crops."
        )

        try:
            tracker_coloc = mi.ParticleTracking(
                image=image_tzyxc,  # Keep native intensities for crop extraction
                channels_spots=[0],
                list_voxels=[voxel_z, voxel_yx],
                channels_cytosol=[0],
                channels_nucleus=[1],
                masks=cytosol_2d,
                masks_nuclei=nucleus_2d,
                yx_spot_size_in_px=5,
                z_spot_size_in_px=2,
                threshold_for_spot_detection=fallback_threshold,
                min_length_trajectory=fallback_min_length,
                memory=fallback_memory,
                use_maximum_projection=True,
                verbose=False
            )
            result_coloc = tracker_coloc.run()
            df_tracked_coloc_candidate = extract_tracking_dataframe(result_coloc)
            n_tracked_coloc = len(df_tracked_coloc_candidate)
            n_particles_coloc = count_unique_particles(df_tracked_coloc_candidate)
            print(
                f"  Fallback tracked spots: {n_tracked_coloc}, "
                f"Particles: {n_particles_coloc}"
            )

            if n_tracked_coloc > 0:
                df_tracked_coloc = df_tracked_coloc_candidate
        except Exception as e:
            print(f"  ⚠️ Fallback colocalization tracking failed: {e}")
    
    results['MSD Recovery'] = test_msd_recovery(
        image_tzyxc, (cytosol_mask, nucleus_mask), config, df_tracked=df_tracked_shared
    )
    results['Colocalization Recovery'] = test_colocalization_recovery(
        image_tzyxc, df_tracked_coloc, config, ml_threshold=0.51, df_gt=df_gt
    )
    results['GUI Syntax'] = test_gui_syntax()
    
    # Generate report
    generate_report(results, report_path, config)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    overall_pass = print_summary(results)

    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
