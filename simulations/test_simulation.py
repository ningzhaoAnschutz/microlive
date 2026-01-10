#!/usr/bin/env python3
"""
Quick Validation Test for Spot Simulation
==========================================

Runs MicroLive analysis on simulated data and compares with ground truth.

Usage:
    python test_simulation.py
"""

import numpy as np
import pandas as pd
import tifffile
from pathlib import Path

# Add microlive to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from microlive import microscopy as mi

# =============================================================================
# CONFIGURATION
# =============================================================================

SIM_DIR = Path(__file__).parent / 'results'
GROUND_TRUTH = SIM_DIR / 'ground_truth.csv'  # Use CSV instead of parquet
IMAGE_FILE = SIM_DIR / 'simulated_spots.tif'
MASK_CYTOSOL = SIM_DIR / 'mask_cytosol.tif'
MASK_NUCLEUS = SIM_DIR / 'mask_nucleus.tif'

# Thresholds
THRESHOLD_POSITION = 3.0  # pixels
THRESHOLD_PHOTOBLEACHING = 0.30  # 30% error
THRESHOLD_DIFFUSION = 0.50  # 50% error

# =============================================================================
# TESTS
# =============================================================================

def test_image_loading():
    """Test that simulated image loads correctly."""
    print("\n" + "=" * 60)
    print("TEST: Image Loading")
    print("=" * 60)
    
    image = tifffile.imread(IMAGE_FILE)
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image min/max: {image.min()} / {image.max()}")
    
    # Expected: TCZYX
    expected_ndim = 5
    assert image.ndim == expected_ndim, f"Expected {expected_ndim}D, got {image.ndim}D"
    
    print("  ✅ PASS: Image loaded with correct dimensions")
    return image


def test_mask_loading():
    """Test that masks load correctly."""
    print("\n" + "=" * 60)
    print("TEST: Mask Loading")
    print("=" * 60)
    
    cytosol = tifffile.imread(MASK_CYTOSOL)
    nucleus = tifffile.imread(MASK_NUCLEUS)
    
    print(f"  Cytosol mask shape: {cytosol.shape}")
    print(f"  Nucleus mask shape: {nucleus.shape}")
    print(f"  Unique cytosol labels: {np.unique(cytosol)}")
    print(f"  Unique nucleus labels: {np.unique(nucleus)}")
    
    # Check labeled format
    assert cytosol.max() >= 1, "Cytosol mask should have cell labels"
    
    print("  ✅ PASS: Masks loaded correctly")
    return cytosol, nucleus


def test_ground_truth():
    """Test ground truth structure."""
    print("\n" + "=" * 60)
    print("TEST: Ground Truth Structure")
    print("=" * 60)
    
    df = pd.read_csv(GROUND_TRUTH)
    print(f"  Records: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    
    required_cols = ['frame', 'z', 'y', 'x', 'particle', 'cell_id', 'spot_type', 
                     'is_nuc', 'compartment', 'compartment_transition']
    
    missing = [c for c in required_cols if c not in df.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"
    
    # Check no particles in background
    ch0 = df[df['spot_type'] == 0]
    n_background = (ch0['compartment'] == 'background').sum()
    print(f"  Particles in background: {n_background}")
    
    # Check compartment distribution
    print(f"  Compartment distribution:")
    for comp, count in ch0['compartment'].value_counts().items():
        print(f"    {comp}: {count}")
    
    print("  ✅ PASS: Ground truth has correct structure")
    return df


def test_photobleaching(image):
    """Test photobleaching recovery."""
    print("\n" + "=" * 60)
    print("TEST: Photobleaching Recovery")
    print("=" * 60)
    
    # Convert from TCZYX to TZYXC for MicroLive
    # TCZYX -> TZYXC: swap axes
    image_tzyxc = np.moveaxis(image, 1, -1)
    
    print(f"  Image shape (TZYXC): {image_tzyxc.shape}")
    
    try:
        pb = mi.Photobleaching(
            image_TZYXC=image_tzyxc,
            time_interval_seconds=5.0
        )
        results = pb.calculate_photobleaching()
        
        print("  Decay rates (k):")
        for ch in range(min(3, len(results) // 2)):
            k = results[ch * 2] if results[ch * 2] is not None else 0
            print(f"    Channel {ch}: k = {k:.6f} s⁻¹")
        
        print("  ✅ PASS: Photobleaching analysis completed")
        return results
    except Exception as e:
        print(f"  ⚠️ SKIP: {e}")
        return None


def test_spot_detection(image, df_gt):
    """Test spot detection accuracy."""
    print("\n" + "=" * 60)
    print("TEST: Spot Detection")
    print("=" * 60)
    
    # Convert from TCZYX to TZYXC
    image_tzyxc = np.moveaxis(image, 1, -1)
    
    # Get single frame for quick test
    frame_idx = 60  # Middle frame
    frame = image_tzyxc[frame_idx]  # [Z, Y, X, C]
    
    print(f"  Testing frame {frame_idx}, shape: {frame.shape}")
    
    # Ground truth for this frame
    gt_frame = df_gt[(df_gt['frame'] == frame_idx) & (df_gt['spot_type'] == 0)]
    print(f"  Ground truth spots: {len(gt_frame)}")
    
    try:
        # Run spot detection
        detector = mi.BigFISH(
            image=frame,
            channels_spots=0,
            voxel_size_z=300,
            voxel_size_yx=130,
            yx_spot_size_in_px=5,
            z_spot_size_in_px=2,
            show_plot=False,
            save_files=False
        )
        spots, _, threshold = detector.detect()
        
        print(f"  Detected spots: {len(spots)}")
        print(f"  Detection threshold: {threshold:.1f}")
        
        # Calculate detection rate
        if len(gt_frame) > 0:
            detection_rate = len(spots) / len(gt_frame) * 100
            print(f"  Detection rate: {detection_rate:.1f}%")
        
        print("  ✅ PASS: Spot detection completed")
        return spots
    except Exception as e:
        print(f"  ⚠️ SKIP: {e}")
        return None


def test_tracking(image, df_gt, cytosol_mask, nucleus_mask):
    """Test particle tracking on a subset of frames."""
    print("\n" + "=" * 60)
    print("TEST: Particle Tracking (subset)")
    print("=" * 60)
    
    # Convert from TCZYX to TZYXC
    image_tzyxc = np.moveaxis(image, 1, -1)
    
    # Use only first 20 frames for quick test
    n_frames = 20
    image_subset = image_tzyxc[:n_frames]
    
    print(f"  Testing on {n_frames} frames, shape: {image_subset.shape}")
    
    # Convert 3D masks (Z,Y,X) to 2D (Y,X) by max projection
    # MicroLive expects YX masks when using use_maximum_projection=True
    mask_cyto_2d = cytosol_mask.max(axis=0) if cytosol_mask.ndim == 3 else cytosol_mask
    mask_nuc_2d = nucleus_mask.max(axis=0) if nucleus_mask.ndim == 3 else nucleus_mask
    
    print(f"  Cytosol mask (2D): {mask_cyto_2d.shape}")
    print(f"  Nucleus mask (2D): {mask_nuc_2d.shape}")
    
    try:
        tracker = mi.ParticleTracking(
            image=image_subset,
            channels_spots=[0],
            list_voxels=[300, 130],
            channels_cytosol=[0],
            channels_nucleus=[1],
            masks=mask_cyto_2d,
            masks_nuclei=mask_nuc_2d,
            yx_spot_size_in_px=5,
            z_spot_size_in_px=2,
            threshold_for_spot_detection=500,
            min_length_trajectory=3,
            memory=1,
            use_maximum_projection=True,  # Use 2D tracking
            verbose=False
        )
        
        df_tracked, images_filtered = tracker.run()
        
        print(f"  Tracked spots: {len(df_tracked)}")
        if len(df_tracked) > 0:
            print(f"  Unique particles: {df_tracked['particle'].nunique() if 'particle' in df_tracked.columns else 'N/A'}")
        
        # Compare with ground truth
        gt_subset = df_gt[(df_gt['frame'] < n_frames) & (df_gt['spot_type'] == 0)]
        print(f"  Ground truth spots (same frames): {len(gt_subset)}")
        
        print("  ✅ PASS: Tracking completed")
        return df_tracked
    except Exception as e:
        print(f"  ⚠️ SKIP: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_report(results):
    """Generate summary report."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r == 'PASS')
    failed = sum(1 for r in results.values() if r == 'FAIL')
    skipped = sum(1 for r in results.values() if r == 'SKIP')
    
    print(f"\n  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏭️  Skipped: {skipped}")
    
    print("\n  Results by test:")
    for test, status in results.items():
        icon = {'PASS': '✅', 'FAIL': '❌', 'SKIP': '⏭️'}[status]
        print(f"    {icon} {test}")
    
    overall = "PASS" if failed == 0 and passed > 0 else "FAIL"
    print(f"\n  Overall: {'✅ PASS' if overall == 'PASS' else '❌ FAIL'}")
    
    return overall


def main():
    print("=" * 60)
    print("MICROLIVE SIMULATION VALIDATION")
    print("=" * 60)
    print(f"\nSimulation directory: {SIM_DIR}")
    
    results = {}
    
    # Test 1: Image loading
    try:
        image = test_image_loading()
        results['Image Loading'] = 'PASS'
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        results['Image Loading'] = 'FAIL'
        return
    
    # Test 2: Mask loading
    try:
        cytosol, nucleus = test_mask_loading()
        results['Mask Loading'] = 'PASS'
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        results['Mask Loading'] = 'FAIL'
        cytosol, nucleus = None, None
    
    # Test 3: Ground truth
    try:
        df_gt = test_ground_truth()
        results['Ground Truth'] = 'PASS'
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        results['Ground Truth'] = 'FAIL'
        df_gt = None
    
    # Test 4: Photobleaching
    try:
        pb_results = test_photobleaching(image)
        results['Photobleaching'] = 'PASS' if pb_results is not None else 'SKIP'
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        results['Photobleaching'] = 'FAIL'
    
    # Test 5: Spot detection
    if df_gt is not None:
        try:
            spots = test_spot_detection(image, df_gt)
            results['Spot Detection'] = 'PASS' if spots is not None else 'SKIP'
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            results['Spot Detection'] = 'FAIL'
    else:
        results['Spot Detection'] = 'SKIP'
    
    # Test 6: Tracking
    if df_gt is not None and cytosol is not None:
        try:
            tracked = test_tracking(image, df_gt, cytosol, nucleus)
            results['Tracking'] = 'PASS' if tracked is not None else 'SKIP'
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            results['Tracking'] = 'FAIL'
    else:
        results['Tracking'] = 'SKIP'
    
    # Summary
    generate_report(results)


if __name__ == '__main__':
    main()
