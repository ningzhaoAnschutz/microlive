#!/usr/bin/env python3
"""
Generate visualization PNGs from simulation output.

Usage:
    python visualize_results.py                    # Uses default results/ folder
    python visualize_results.py --sim-dir results_multicell
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize image to 0-255 uint8."""
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255
    return img.astype(np.uint8)

def save_frame_png(frame_3d: np.ndarray, output_path: Path, title: str = ""):
    """Save a 3D frame as max projection PNG."""
    # Max projection across Z
    max_proj = frame_3d.max(axis=0)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(max_proj, cmap='gray')
    plt.colorbar(label='Intensity')
    plt.title(title)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_rgb_composite(frame_czyx: np.ndarray, output_path: Path, title: str = ""):
    """Save multi-channel frame as RGB composite PNG."""
    # Max projection for each channel
    n_channels = frame_czyx.shape[0]
    projections = [frame_czyx[c].max(axis=0) for c in range(n_channels)]
    
    # Normalize each channel
    normalized = [normalize_to_uint8(p) for p in projections]
    
    # Create RGB (ch0=green, ch1=red, ch2=blue)
    rgb = np.zeros((*projections[0].shape, 3), dtype=np.uint8)
    if len(normalized) > 0:
        rgb[:, :, 1] = normalized[0]  # Green = ch0 (lead)
    if len(normalized) > 1:
        rgb[:, :, 0] = normalized[1]  # Red = ch1
    if len(normalized) > 2:
        rgb[:, :, 2] = normalized[2]  # Blue = ch2
    
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title(title)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_mask_png(mask: np.ndarray, output_path: Path, title: str = ""):
    """Save mask as colored PNG."""
    # Max projection if 3D
    if mask.ndim == 3:
        mask_2d = mask.max(axis=0)
    else:
        mask_2d = mask
    
    # Create colored version
    n_labels = mask_2d.max()
    
    plt.figure(figsize=(8, 8))
    if n_labels > 0:
        cmap = plt.cm.get_cmap('tab10', n_labels + 1)
        plt.imshow(mask_2d, cmap=cmap, interpolation='nearest')
        plt.colorbar(label='Cell ID', ticks=range(n_labels + 1))
    else:
        plt.imshow(mask_2d, cmap='gray')
    plt.title(title)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_trajectory_plot(df: pd.DataFrame, output_path: Path, title: str = "", 
                         mask: np.ndarray = None):
    """Save particle trajectories as PNG with optional mask overlay."""
    ch0 = df[df['spot_type'] == 0]
    n_cells = ch0['cell_id'].nunique()
    
    plt.figure(figsize=(10, 10))
    
    # Show mask as background if provided
    if mask is not None:
        if mask.ndim == 3:
            mask_2d = mask.max(axis=0)
        else:
            mask_2d = mask
        plt.imshow(mask_2d > 0, cmap='gray', alpha=0.3)
    
    # Color by cell if multi-cell
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_cells)))
    
    # Plot each particle's trajectory
    for particle_id in ch0['particle'].unique():
        p_data = ch0[ch0['particle'] == particle_id].sort_values('frame')
        cell_id = p_data['cell_id'].iloc[0]
        color = colors[int(cell_id) % len(colors)]
        
        # Line style based on compartment
        linestyle = '--' if p_data['is_nuc'].iloc[0] else '-'
        
        plt.plot(p_data['x'], p_data['y'], linestyle, alpha=0.6, linewidth=0.8, 
                color=color)
        plt.scatter(p_data['x'].iloc[0], p_data['y'].iloc[0], s=30, 
                   color=color, marker='o', edgecolors='white', linewidths=0.5)
    
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    
    if n_cells > 1:
        plt.title(f'{title}\nColors=cells, Solid=cytosol, Dashed=nucleus')
    else:
        plt.title(f'{title}\nBlue=cytosol, Red=nucleus, Dots=start positions')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_compartment_pie(df: pd.DataFrame, output_path: Path):
    """Save compartment distribution as pie chart."""
    ch0 = df[df['spot_type'] == 0]
    comp_counts = ch0['compartment'].value_counts()
    
    plt.figure(figsize=(6, 6))
    colors = {'cytosol': '#4CAF50', 'nucleus': '#2196F3', 'background': '#9E9E9E'}
    plt.pie(comp_counts.values, labels=comp_counts.index, autopct='%1.1f%%',
            colors=[colors.get(c, '#888888') for c in comp_counts.index])
    plt.title('Particle Compartment Distribution')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_colocalization_bar(df: pd.DataFrame, output_path: Path):
    """Save colocalization rates as bar chart."""
    ch0 = df[df['spot_type'] == 0]
    
    ch1_rate = ch0['has_ch1_partner'].mean() * 100
    ch2_rate = ch0['has_ch2_partner'].mean() * 100
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['Ch0-Ch1', 'Ch0-Ch2'], [ch1_rate, ch2_rate], 
                   color=['#FF5722', '#3F51B5'])
    plt.ylabel('Colocalization %')
    plt.title('Colocalization Rates')
    plt.ylim(0, 100)
    
    # Add value labels
    for bar, val in zip(bars, [ch1_rate, ch2_rate]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_per_cell_stats(df: pd.DataFrame, output_path: Path):
    """Save per-cell particle statistics."""
    ch0 = df[df['spot_type'] == 0]
    
    cell_stats = ch0.groupby('cell_id').agg({
        'particle': 'nunique',
        'has_ch1_partner': 'mean',
        'has_ch2_partner': 'mean',
        'is_nuc': 'mean'
    }).rename(columns={
        'particle': 'Particles',
        'has_ch1_partner': 'Ch1 Coloc',
        'has_ch2_partner': 'Ch2 Coloc',
        'is_nuc': 'Nuclear %'
    })
    
    n_cells = len(cell_stats)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart 1: Particles per cell
    ax1 = axes[0]
    ax1.bar(cell_stats.index.astype(str), cell_stats['Particles'], color='steelblue')
    ax1.set_xlabel('Cell ID')
    ax1.set_ylabel('Number of Particles')
    ax1.set_title('Particles per Cell')
    
    # Add value labels
    for i, (idx, row) in enumerate(cell_stats.iterrows()):
        ax1.text(i, row['Particles'] + 0.5, str(int(row['Particles'])), 
                ha='center', fontsize=10)
    
    # Bar chart 2: Colocalization per cell
    ax2 = axes[1]
    x = np.arange(n_cells)
    width = 0.35
    ax2.bar(x - width/2, cell_stats['Ch1 Coloc'] * 100, width, label='Ch1', color='#FF5722')
    ax2.bar(x + width/2, cell_stats['Ch2 Coloc'] * 100, width, label='Ch2', color='#3F51B5')
    ax2.set_xlabel('Cell ID')
    ax2.set_ylabel('Colocalization %')
    ax2.set_title('Colocalization per Cell')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cell_stats.index.astype(str))
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Generate visualization PNGs from simulation output.'
    )
    parser.add_argument('--sim-dir', type=str, default='results',
                       help='Simulation results directory (default: results)')
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / args.sim_dir
    viz_dir = ensure_dir(results_dir / 'viz')
    
    print(f"Loading simulation data from: {results_dir}")
    print(f"Saving visualizations to: {viz_dir}")
    
    # Load data
    image = tifffile.imread(results_dir / 'simulated_spots.tif')
    df = pd.read_csv(results_dir / 'ground_truth.csv')
    mask_cytosol = tifffile.imread(results_dir / 'mask_cytosol.tif')
    mask_nucleus = tifffile.imread(results_dir / 'mask_nucleus.tif')
    
    n_cells = df['cell_id'].nunique()
    n_particles = df[df['spot_type'] == 0]['particle'].nunique()
    
    print(f"  Image shape: {image.shape} (TCZYX)")
    print(f"  Ground truth: {len(df)} records")
    print(f"  Cells: {n_cells}")
    print(f"  Particles: {n_particles}")
    
    # 1. First frame - each channel
    print("\nGenerating channel visualizations...")
    for ch in range(min(3, image.shape[1])):
        save_frame_png(
            image[0, ch],  # First frame, channel ch
            viz_dir / f'frame0_ch{ch}.png',
            f'Frame 0, Channel {ch} (Max Z-projection)'
        )
        print(f"  Saved frame0_ch{ch}.png")
    
    # 2. RGB composite
    save_rgb_composite(
        image[0],  # First frame [C, Z, Y, X]
        viz_dir / 'frame0_rgb.png',
        'Frame 0 RGB Composite (G=Ch0, R=Ch1, B=Ch2)'
    )
    print(f"  Saved frame0_rgb.png")
    
    # 3. Middle frame RGB
    mid_frame = image.shape[0] // 2
    save_rgb_composite(
        image[mid_frame],
        viz_dir / f'frame{mid_frame}_rgb.png',
        f'Frame {mid_frame} RGB Composite'
    )
    print(f"  Saved frame{mid_frame}_rgb.png")
    
    # 4. Last frame RGB (to show photobleaching)
    save_rgb_composite(
        image[-1],
        viz_dir / f'frame{image.shape[0]-1}_rgb.png',
        f'Frame {image.shape[0]-1} RGB Composite (Final)'
    )
    print(f"  Saved frame{image.shape[0]-1}_rgb.png")
    
    # 5. Masks
    print("\nGenerating mask visualizations...")
    save_mask_png(mask_cytosol, viz_dir / 'mask_cytosol.png', 
                 f'Cytosol Mask ({n_cells} cells)')
    save_mask_png(mask_nucleus, viz_dir / 'mask_nucleus.png', 
                 f'Nucleus Mask ({n_cells} cells)')
    print(f"  Saved mask_cytosol.png, mask_nucleus.png")
    
    # 6. Trajectories with mask overlay
    print("\nGenerating trajectory visualization...")
    save_trajectory_plot(df, viz_dir / 'trajectories.png', 
                        f'Particle Trajectories ({n_particles} particles)',
                        mask_cytosol)
    print(f"  Saved trajectories.png")
    
    # 7. Compartment distribution
    print("\nGenerating statistics...")
    save_compartment_pie(df, viz_dir / 'compartment_distribution.png')
    print(f"  Saved compartment_distribution.png")
    
    # 8. Colocalization rates
    save_colocalization_bar(df, viz_dir / 'colocalization_rates.png')
    print(f"  Saved colocalization_rates.png")
    
    # 9. Per-cell stats (multi-cell only)
    if n_cells > 1:
        save_per_cell_stats(df, viz_dir / 'per_cell_stats.png')
        print(f"  Saved per_cell_stats.png")
    
    print(f"\n✅ All visualizations saved to: {viz_dir}")
    print("\nGenerated files:")
    for f in sorted(viz_dir.glob('*.png')):
        print(f"  - {f.name}")

if __name__ == '__main__':
    main()
