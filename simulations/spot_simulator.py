"""
Simplified Spot Simulation for MicroLive Validation
====================================================

Generates synthetic 2D/3D microscopy images with:
- 3 color channels with colocalization
- Three spatial regions (background, cytosol, nucleus)
- Brownian particle motion
- Per-channel photobleaching
- MicroLive-compatible ground truth DataFrame

Author: MicroLive Development Team
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
import tifffile
import yaml
from scipy.ndimage import gaussian_filter
from scipy.stats import truncnorm


# =============================================================================
# PARTICLE DATACLASS
# =============================================================================

@dataclass
class Particle:
    """Represents a single tracked particle with absolute image coordinates.
    
    Attributes:
        particle_id: Unique ID within this cell
        cell_id: Which cell this particle belongs to (1, 2, 3, ...)
        birth_frame: Frame when particle appears
        death_frame: Frame when particle disappears
        initial_position: [z, y, x] in absolute image pixels
        current_position: [z, y, x] updated each frame
        position_history: Dict mapping frame -> position
        size: PSF sigma in pixels
        snr: Signal-to-Noise Ratio
        has_ch1_partner: Whether ch1 colocalization exists
        has_ch2_partner: Whether ch2 colocalization exists
    """
    particle_id: int
    cell_id: int
    birth_frame: int
    death_frame: int
    initial_position: np.ndarray
    current_position: np.ndarray
    size: float
    snr: float
    has_ch1_partner: bool
    has_ch2_partner: bool
    position_history: Dict[int, np.ndarray] = field(default_factory=dict)
    
    def is_alive(self, frame: int) -> bool:
        """Check if particle exists at given frame."""
        return self.birth_frame <= frame < self.death_frame
    
    def get_peak_intensity(self, local_background: float) -> float:
        """Calculate peak intensity: background × (1 + SNR)."""
        return local_background * (1.0 + self.snr)
    
    def get_unique_id(self, image_id: int = 0, spot_type: int = 0) -> str:
        """Generate globally unique particle identifier."""
        return f"{image_id}_{self.cell_id}_{spot_type}_{self.particle_id}"
    
    def record_position(self, frame: int) -> None:
        """Store current position in history for ground truth export."""
        self.position_history[frame] = self.current_position.copy()


# =============================================================================
# CELL REGION DATACLASS
# =============================================================================

@dataclass
class CellRegion:
    """Represents a single cell with cytosol and nucleus regions.
    
    Attributes:
        cell_id: Unique cell identifier (1, 2, 3, ...)
        center_yx: [y, x] center in absolute image pixels
        cell_diameter_yx: [dy, dx] cell diameter in pixels
        cell_height_z: Cell height in Z slices
        nucleus_diameter_yx: [dy, dx] nucleus diameter in pixels
        nucleus_height_z: Nucleus height in Z slices
        nucleus_offset_yx: [dy, dx] nucleus offset from cell center
    """
    cell_id: int
    center_yx: Tuple[float, float]
    cell_diameter_yx: Tuple[float, float]
    cell_height_z: float
    nucleus_diameter_yx: Tuple[float, float] = (100, 100)
    nucleus_height_z: float = 5.0
    nucleus_offset_yx: Tuple[float, float] = (0, 0)
    cytosol_mask: np.ndarray = field(default=None, repr=False)
    nucleus_mask: np.ndarray = field(default=None, repr=False)
    particles: List[Particle] = field(default_factory=list)


# =============================================================================
# SPOT SIMULATOR
# =============================================================================

class SpotSimulator:
    """Main simulation engine for generating synthetic microscopy data.
    
    Generates synthetic 2D/3D microscopy images with controlled spot dynamics
    for MicroLive validation and parameter recovery testing.
    
    Example:
        sim = SpotSimulator("config_simple.yaml")
        image_stack, df_gt = sim.run()
        sim.save_results("./results")
    """
    
    def __init__(self, config_path: str):
        """Initialize from YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self._load_config()
        self._init_rng()
        self._init_geometry()
        
        # State
        self.particles: List[Particle] = []
        self.cells: List[CellRegion] = []
        self.image_stack: Optional[np.ndarray] = None
        self.df_ground_truth: Optional[pd.DataFrame] = None
        
    def _load_config(self) -> None:
        """Load and validate YAML configuration."""
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Extract key parameters with defaults
        img = self.config.get('image', {})
        self.dimensions = img.get('dimensions', '3D')
        self.size_yx = tuple(img.get('size_yx', [512, 512]))
        self.num_z = img.get('num_z_slices', 10) if self.dimensions == '3D' else 1
        self.voxel_yx = img.get('voxel_size_yx_nm', 130.0)
        self.voxel_z = img.get('voxel_size_z_nm', 300.0)
        
        sim = self.config.get('simulation', {})
        self.total_time = sim.get('total_time_seconds', 600)
        self.frame_rate = sim.get('frame_rate_seconds', 5)
        self.total_frames = int(self.total_time / self.frame_rate)
        
        # Image shape: [Z, Y, X]
        self.shape = (self.num_z, self.size_yx[0], self.size_yx[1])
        
    def _init_rng(self) -> None:
        """Initialize random number generator with seed."""
        seed = self.config.get('simulation', {}).get('random_seed', None)
        self.rng = np.random.default_rng(seed)
        
    def _init_geometry(self) -> None:
        """Initialize cell geometry and masks."""
        geo = self.config.get('cell_geometry', {})
        baseline = self.config.get('baseline', {})
        
        # Baseline intensities
        self.baseline_outside = baseline.get('outside_cell', 800)
        self.baseline_cytosol = baseline.get('cytosol', 1200)
        self.baseline_nucleus = baseline.get('nucleus', 1500)
        
        # Multi-cell configuration
        self.num_cells = geo.get('num_cells', 1)
        self.layout = geo.get('layout', 'single')
        self.grid_spacing_yx = tuple(geo.get('grid_spacing_yx', [360, 360]))
        
        # Cell geometry
        self.geometry_type = geo.get('geometry_type', 'half_ellipsoid')
        self.cell_center_yx = tuple(geo.get('cell_center_yx', [256, 256]))
        self.cell_diameter_yx = tuple(geo.get('cell_diameter_yx', [350, 350]))
        self.cell_height_z = geo.get('cell_height_z', 8)
        
        # Nucleus geometry
        self.nucleus_enabled = geo.get('nucleus_enabled', True)
        self.nucleus_diameter_yx = tuple(geo.get('nucleus_diameter_yx', [100, 100]))
        self.nucleus_height_z = geo.get('nucleus_height_z', 5)
        self.nucleus_offset_yx = tuple(geo.get('nucleus_offset_yx', [0, 0]))
        
        # Particle compartment
        self.particle_compartment = geo.get('particle_compartment', 'both')
        self.confinement_mode = geo.get('confinement_mode', 'cell')
        
    # =========================================================================
    # MASK GENERATION
    # =========================================================================
    
    def _create_masks(self) -> None:
        """Generate cytosol and nucleus masks for one or more cells."""
        self.cells = []
        
        # Calculate cell positions
        cell_positions = self._calculate_cell_positions()
        
        # Initialize combined masks
        self.cytosol_mask = np.zeros(self.shape, dtype=bool)
        self.nucleus_mask = np.zeros(self.shape, dtype=bool)
        
        for cell_id, center_yx in enumerate(cell_positions, start=1):
            cell = CellRegion(
                cell_id=cell_id,
                center_yx=center_yx,
                cell_diameter_yx=self.cell_diameter_yx,
                cell_height_z=self.cell_height_z,
                nucleus_diameter_yx=self.nucleus_diameter_yx,
                nucleus_height_z=self.nucleus_height_z,
                nucleus_offset_yx=self.nucleus_offset_yx,
            )
            
            # Generate cell mask
            cell.cytosol_mask = self._generate_ellipsoid_mask(
                center_zyx=(self.num_z // 2, center_yx[0], center_yx[1]),
                radii_zyx=(self.cell_height_z / 2, self.cell_diameter_yx[0] / 2, self.cell_diameter_yx[1] / 2),
                half_ellipsoid=(self.geometry_type == 'half_ellipsoid')
            )
            
            if self.nucleus_enabled:
                nuc_center_y = center_yx[0] + self.nucleus_offset_yx[0]
                nuc_center_x = center_yx[1] + self.nucleus_offset_yx[1]
                cell.nucleus_mask = self._generate_ellipsoid_mask(
                    center_zyx=(self.num_z // 2, nuc_center_y, nuc_center_x),
                    radii_zyx=(self.nucleus_height_z / 2, self.nucleus_diameter_yx[0] / 2, self.nucleus_diameter_yx[1] / 2),
                    half_ellipsoid=(self.geometry_type == 'half_ellipsoid')
                )
            else:
                cell.nucleus_mask = np.zeros(self.shape, dtype=bool)
            
            self.cells.append(cell)
            
            # Add to combined masks
            self.cytosol_mask |= cell.cytosol_mask
            self.nucleus_mask |= cell.nucleus_mask
        
        self.cell_mask = self.cytosol_mask | self.nucleus_mask
        
        # Create baseline intensity map
        self._create_baseline_map()
        
        print(f"Created {len(self.cells)} cell(s)")
    
    def _calculate_cell_positions(self) -> List[Tuple[float, float]]:
        """Calculate center positions for all cells based on layout."""
        if self.num_cells == 1 or self.layout == 'single':
            return [self.cell_center_yx]
        
        positions = []
        
        if self.layout == 'grid':
            # Calculate grid dimensions
            n_cols = int(np.ceil(np.sqrt(self.num_cells)))
            n_rows = int(np.ceil(self.num_cells / n_cols))
            
            spacing_y, spacing_x = self.grid_spacing_yx
            
            # Calculate starting position (center grid in image)
            total_width = (n_cols - 1) * spacing_x
            total_height = (n_rows - 1) * spacing_y
            start_y = (self.size_yx[0] - total_height) / 2
            start_x = (self.size_yx[1] - total_width) / 2
            
            count = 0
            for row in range(n_rows):
                for col in range(n_cols):
                    if count >= self.num_cells:
                        break
                    y = start_y + row * spacing_y
                    x = start_x + col * spacing_x
                    positions.append((y, x))
                    count += 1
                    
        elif self.layout == 'random':
            # Random non-overlapping placement
            min_distance = max(self.cell_diameter_yx) * 0.8  # Minimum distance between centers
            margin = max(self.cell_diameter_yx) // 2 + 10   # Edge margin
            
            attempts = 0
            max_attempts = 1000
            
            while len(positions) < self.num_cells and attempts < max_attempts:
                y = self.rng.uniform(margin, self.size_yx[0] - margin)
                x = self.rng.uniform(margin, self.size_yx[1] - margin)
                
                # Check distance to existing cells
                valid = True
                for py, px in positions:
                    dist = np.sqrt((y - py)**2 + (x - px)**2)
                    if dist < min_distance:
                        valid = False
                        break
                
                if valid:
                    positions.append((y, x))
                
                attempts += 1
            
            if len(positions) < self.num_cells:
                print(f"Warning: Only placed {len(positions)}/{self.num_cells} cells (couldn't fit more)")
        
        return positions
        
    def _generate_ellipsoid_mask(
        self,
        center_zyx: Tuple[float, float, float],
        radii_zyx: Tuple[float, float, float],
        half_ellipsoid: bool = False
    ) -> np.ndarray:
        """Generate 3D ellipsoid mask.
        
        Args:
            center_zyx: Center coordinates (z, y, x)
            radii_zyx: Radii in each dimension (rz, ry, rx)
            half_ellipsoid: If True, only generate bottom half (dome)
            
        Returns:
            Boolean mask array [Z, Y, X]
        """
        z, y, x = np.ogrid[:self.shape[0], :self.shape[1], :self.shape[2]]
        
        cz, cy, cx = center_zyx
        rz, ry, rx = radii_zyx
        
        # Ellipsoid equation: ((z-cz)/rz)^2 + ((y-cy)/ry)^2 + ((x-cx)/rx)^2 <= 1
        dist = ((z - cz) / max(rz, 0.1))**2 + ((y - cy) / max(ry, 0.1))**2 + ((x - cx) / max(rx, 0.1))**2
        mask = dist <= 1.0
        
        if half_ellipsoid:
            # Only keep bottom half (z >= center)
            mask = mask & (z >= cz)
        
        return mask
    
    def _create_baseline_map(self) -> None:
        """Create baseline intensity map with compartment-specific values."""
        self.baseline_map = np.full(self.shape, self.baseline_outside, dtype=np.float32)
        self.baseline_map[self.cytosol_mask] = self.baseline_cytosol
        self.baseline_map[self.nucleus_mask] = self.baseline_nucleus
        
    # =========================================================================
    # PARTICLE MANAGEMENT
    # =========================================================================
    
    def _plan_trajectories(self) -> None:
        """Pre-plan particle birth/death and initial positions."""
        particles_cfg = self.config.get('particles', {})
        
        # Support per-cell particle counts
        per_cell_counts = particles_cfg.get('per_cell_counts', None)
        n_cells = len(self.cells)
        
        if per_cell_counts is not None:
            # Use explicit per-cell counts
            if len(per_cell_counts) < n_cells:
                # Extend with last value if not enough
                per_cell_counts = list(per_cell_counts) + [per_cell_counts[-1]] * (n_cells - len(per_cell_counts))
            cell_particle_counts = per_cell_counts[:n_cells]
            n_particles = sum(cell_particle_counts)
        else:
            # Use average_count distributed evenly
            n_particles = particles_cfg.get('average_count', 50)
            base_per_cell = n_particles // n_cells
            remainder = n_particles % n_cells
            cell_particle_counts = [base_per_cell + (1 if i < remainder else 0) for i in range(n_cells)]
        
        min_traj_frac = particles_cfg.get('min_trajectory_fraction', 0.5)
        max_traj_frac = particles_cfg.get('max_trajectory_fraction', 1.0)
        
        spot_cfg = self.config.get('spot_properties', {})
        size_mean = spot_cfg.get('size_mean', 1.5)
        size_std = spot_cfg.get('size_std', 0.3)
        size_min = spot_cfg.get('size_min', 0.8)
        size_max = spot_cfg.get('size_max', 3.0)
        
        snr_mean = spot_cfg.get('snr_mean', 3.0)
        snr_std = spot_cfg.get('snr_std', 0.5)
        snr_min = spot_cfg.get('snr_min', 1.5)
        snr_max = spot_cfg.get('snr_max', 8.0)
        
        coloc_cfg = self.config.get('colocalization', {})
        ch1_prob = coloc_cfg.get('ch1_probability', 0.7)
        ch2_prob = coloc_cfg.get('ch2_probability', 0.3)
        
        min_len = int(self.total_frames * min_traj_frac)
        max_len = int(self.total_frames * max_traj_frac)
        
        particle_id = 0
        for cell_idx, count in enumerate(cell_particle_counts):
            cell = self.cells[cell_idx]
            
            for _ in range(count):
                # Trajectory length and timing
                length = self.rng.integers(min_len, max_len + 1)
                birth = self.rng.integers(0, max(1, self.total_frames - length + 1))
                death = min(birth + length, self.total_frames)
                
                # Sample position inside this specific cell
                position = self._sample_position_inside_specific_cell(cell)
                
                # Sample spot properties from truncated normal
                size = self._sample_truncated_normal(size_mean, size_std, size_min, size_max)
                snr = self._sample_truncated_normal(snr_mean, snr_std, snr_min, snr_max)
                
                # Colocalization partners
                has_ch1 = self.rng.random() < ch1_prob
                has_ch2 = self.rng.random() < ch2_prob
                
                particle = Particle(
                    particle_id=particle_id,
                    cell_id=cell.cell_id,  # Assign to specific cell
                    birth_frame=birth,
                    death_frame=death,
                    initial_position=position.copy(),
                    current_position=position.copy(),
                    size=size,
                    snr=snr,
                    has_ch1_partner=has_ch1,
                    has_ch2_partner=has_ch2,
                )
                
                self.particles.append(particle)
                cell.particles.append(particle)
                particle_id += 1
            
            
    def _sample_position_inside_cell(self) -> np.ndarray:
        """Sample random position inside cytosol OR nucleus (never background).
        
        Uses rejection sampling: generate random positions until one is inside cell.
        
        Returns:
            Position array [z, y, x] in absolute image coordinates
        """
        # Determine which compartments are valid based on config
        if self.particle_compartment == 'cytosol':
            valid_mask = self.cytosol_mask & ~self.nucleus_mask
        elif self.particle_compartment == 'nucleus':
            valid_mask = self.nucleus_mask
        else:  # 'both'
            valid_mask = self.cell_mask
        
        # Get all valid positions
        valid_positions = np.argwhere(valid_mask)
        
        if len(valid_positions) == 0:
            raise ValueError("No valid positions inside cell. Check mask generation.")
        
        # Random selection with sub-pixel offset
        idx = self.rng.integers(0, len(valid_positions))
        z, y, x = valid_positions[idx]
        
        # Add sub-pixel offset for continuous positions
        z_offset = self.rng.uniform(-0.5, 0.5)
        y_offset = self.rng.uniform(-0.5, 0.5)
        x_offset = self.rng.uniform(-0.5, 0.5)
        
        return np.array([z + z_offset, y + y_offset, x + x_offset], dtype=np.float64)
    
    def _sample_position_inside_specific_cell(self, cell: CellRegion) -> np.ndarray:
        """Sample random position inside a specific cell's mask.
        
        Applies a 5-pixel buffer zone (erosion) to avoid spawning particles
        too close to the cell boundary.
        
        Args:
            cell: The CellRegion to sample within
            
        Returns:
            Position array [z, y, x] in absolute image coordinates
        """
        from scipy import ndimage
        
        # Buffer zone in pixels (erode mask to keep particles away from boundary)
        buffer_px = 5
        
        # Determine valid mask based on particle_compartment setting
        if self.particle_compartment == 'cytosol':
            valid_mask = cell.cytosol_mask & ~cell.nucleus_mask
        elif self.particle_compartment == 'nucleus':
            valid_mask = cell.nucleus_mask
        else:  # 'both'
            valid_mask = cell.cytosol_mask | cell.nucleus_mask
        
        # Apply erosion to create buffer zone (avoid boundary)
        # Create 2D structuring element for YX erosion
        struct_2d = ndimage.generate_binary_structure(2, 1)
        
        # Apply erosion slice by slice (3D mask)
        eroded_mask = np.zeros_like(valid_mask)
        for z in range(valid_mask.shape[0]):
            eroded_mask[z] = ndimage.binary_erosion(
                valid_mask[z], 
                structure=struct_2d, 
                iterations=buffer_px
            )
        
        # Fall back to original mask if erosion removes all pixels
        if not eroded_mask.any():
            eroded_mask = valid_mask
        
        # Get all valid positions
        valid_positions = np.argwhere(eroded_mask)
        
        if len(valid_positions) == 0:
            raise ValueError(f"No valid positions inside cell {cell.cell_id}. Check mask generation.")
        
        # Random selection with sub-pixel offset
        idx = self.rng.integers(0, len(valid_positions))
        z, y, x = valid_positions[idx]
        
        # Add sub-pixel offset for continuous positions
        z_offset = self.rng.uniform(-0.5, 0.5)
        y_offset = self.rng.uniform(-0.5, 0.5)
        x_offset = self.rng.uniform(-0.5, 0.5)
        
        return np.array([z + z_offset, y + y_offset, x + x_offset], dtype=np.float64)
    
    def _sample_truncated_normal(
        self, mean: float, std: float, min_val: float, max_val: float
    ) -> float:
        """Sample from a truncated normal distribution."""
        if std <= 0:
            return np.clip(mean, min_val, max_val)
        
        a = (min_val - mean) / std
        b = (max_val - mean) / std
        
        return truncnorm.rvs(a, b, loc=mean, scale=std, random_state=self.rng)
    
    # =========================================================================
    # BROWNIAN MOTION
    # =========================================================================
    
    def _update_positions(self, frame: int) -> None:
        """Apply Brownian motion and boundary reflection."""
        motion_cfg = self.config.get('motion', {})
        D = motion_cfg.get('diffusion_coefficient', 0.05)  # px²/frame
        sigma = np.sqrt(2 * D)
        
        for particle in self.particles:
            if not particle.is_alive(frame):
                continue
            
            # Generate displacement
            displacement = self.rng.normal(0, sigma, size=3)
            
            # Scale Z displacement if anisotropic
            z_scale = self.voxel_yx / self.voxel_z if self.voxel_z > 0 else 1.0
            displacement[0] *= z_scale
            
            # Apply displacement
            new_pos = particle.current_position + displacement
            
            # Reflect into cell (particle stays in its assigned cell)
            cell = self._get_cell_by_id(particle.cell_id)
            new_pos = self._reflect_into_specific_cell(new_pos, particle.current_position, cell)
            
            particle.current_position = new_pos
            particle.record_position(frame)
    
    def _get_cell_by_id(self, cell_id: int) -> Optional[CellRegion]:
        """Get cell region by its ID."""
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        return None
            
    def _reflect_into_cell(
        self, pos: np.ndarray, prev_pos: np.ndarray
    ) -> np.ndarray:
        """Ensure position stays inside any cell (for backwards compatibility)."""
        # Clamp to image bounds
        pos = np.clip(pos, [0, 0, 0], [s - 1 for s in self.shape])
        
        iz, iy, ix = int(pos[0]), int(pos[1]), int(pos[2])
        
        # Bounds check
        if iz < 0 or iz >= self.shape[0] or iy < 0 or iy >= self.shape[1] or ix < 0 or ix >= self.shape[2]:
            return prev_pos
        
        # Check if inside valid region based on confinement mode
        if self.confinement_mode == 'cell':
            valid = self.cell_mask[iz, iy, ix]
        elif self.confinement_mode == 'cytosol':
            valid = self.cytosol_mask[iz, iy, ix] and not self.nucleus_mask[iz, iy, ix]
        elif self.confinement_mode == 'nucleus':
            valid = self.nucleus_mask[iz, iy, ix]
        else:  # 'none'
            valid = True
        
        if valid:
            return pos
        else:
            return prev_pos
    
    def _reflect_into_specific_cell(
        self, pos: np.ndarray, prev_pos: np.ndarray, cell: CellRegion
    ) -> np.ndarray:
        """Ensure position stays inside a specific cell.
        
        Uses round() to match compartment detection logic.
        """
        # Clamp to image bounds
        pos = np.clip(pos, [0, 0, 0], [s - 1 for s in self.shape])
        
        # Use round() to match compartment detection
        iz = int(np.clip(round(pos[0]), 0, self.shape[0] - 1))
        iy = int(np.clip(round(pos[1]), 0, self.shape[1] - 1))
        ix = int(np.clip(round(pos[2]), 0, self.shape[2] - 1))
        
        # Check if inside this specific cell based on confinement mode
        cell_interior = cell.cytosol_mask | cell.nucleus_mask
        
        if self.confinement_mode == 'cell':
            valid = cell_interior[iz, iy, ix]
        elif self.confinement_mode == 'cytosol':
            valid = cell.cytosol_mask[iz, iy, ix] and not cell.nucleus_mask[iz, iy, ix]
        elif self.confinement_mode == 'nucleus':
            valid = cell.nucleus_mask[iz, iy, ix]
        else:  # 'none'
            valid = True
        
        if valid:
            return pos
        else:
            # Reject move, stay at previous position
            return prev_pos
    
    # =========================================================================
    # COMPARTMENT DETECTION
    # =========================================================================
    
    def _get_compartment_at_position(self, z: float, y: float, x: float) -> str:
        """Determine which compartment a position is in.
        
        Uses rounding and clamping to handle edge cases near boundaries.
        """
        # Use round() instead of int() for better edge handling
        iz = int(np.clip(round(z), 0, self.shape[0] - 1))
        iy = int(np.clip(round(y), 0, self.shape[1] - 1))
        ix = int(np.clip(round(x), 0, self.shape[2] - 1))
        
        if self.nucleus_mask[iz, iy, ix]:
            return 'nucleus'
        elif self.cytosol_mask[iz, iy, ix]:
            return 'cytosol'
        
        # Edge case: position is technically outside mask but particle
        # was spawned inside cell - check cell_mask as fallback
        if self.cell_mask[iz, iy, ix]:
            return 'cytosol'  # Default to cytosol if in cell but not nucleus
        
        return 'background'
    
    def _get_local_background(self, z: float, y: float, x: float) -> float:
        """Get baseline intensity at position."""
        compartment = self._get_compartment_at_position(z, y, x)
        return {
            'nucleus': self.baseline_nucleus,
            'cytosol': self.baseline_cytosol,
            'background': self.baseline_outside
        }[compartment]
    
    # =========================================================================
    # RENDERING
    # =========================================================================
    
    def _render_frame(self, frame: int) -> np.ndarray:
        """Render a single frame with all channels.
        
        Returns:
            Frame array [Z, Y, X, C] as float32
        """
        n_channels = 3
        frame_img = np.zeros((*self.shape, n_channels), dtype=np.float32)
        
        # Add baseline to all channels
        for c in range(n_channels):
            frame_img[:, :, :, c] = self.baseline_map.copy()
        
        # Render particles
        for particle in self.particles:
            if not particle.is_alive(frame):
                continue
            
            pos = particle.current_position
            local_bg = self._get_local_background(*pos)
            amplitude = particle.get_peak_intensity(local_bg)
            
            # Channel 0 (lead channel)
            self._render_gaussian_spot(frame_img[:, :, :, 0], pos, amplitude, particle.size)
            
            # Channel 1 (colocalized)
            if particle.has_ch1_partner:
                ch1_cfg = self.config.get('colocalization', {})
                ch1_mult = ch1_cfg.get('ch1_snr_multiplier', 0.8)
                ch1_amp = local_bg * (1.0 + particle.snr * ch1_mult)
                self._render_gaussian_spot(frame_img[:, :, :, 1], pos, ch1_amp, particle.size)
            
            # Channel 2 (colocalized)
            if particle.has_ch2_partner:
                ch2_cfg = self.config.get('colocalization', {})
                ch2_mult = ch2_cfg.get('ch2_snr_multiplier', 0.5)
                ch2_amp = local_bg * (1.0 + particle.snr * ch2_mult)
                self._render_gaussian_spot(frame_img[:, :, :, 2], pos, ch2_amp, particle.size)
        
        # Apply photobleaching
        frame_img = self._apply_photobleaching(frame_img, frame)
        
        # Add noise
        frame_img = self._apply_noise(frame_img)
        
        return frame_img
    
    def _render_gaussian_spot(
        self,
        image: np.ndarray,
        position: np.ndarray,
        amplitude: float,
        sigma_xy: float
    ) -> None:
        """Add a Gaussian PSF to the image at the given position.
        
        Args:
            image: Image array [Z, Y, X] to modify in place
            position: [z, y, x] position
            amplitude: Peak amplitude above baseline
            sigma_xy: XY sigma in pixels
        """
        z, y, x = position
        sigma_z = sigma_xy * (self.voxel_yx / self.voxel_z) if self.voxel_z > 0 else sigma_xy
        
        # Kernel size (4 sigma for 99.9% of energy)
        kz = int(4 * sigma_z) + 1
        ky = int(4 * sigma_xy) + 1
        kx = int(4 * sigma_xy) + 1
        
        # Calculate bounds
        z0 = max(0, int(z) - kz)
        z1 = min(self.shape[0], int(z) + kz + 1)
        y0 = max(0, int(y) - ky)
        y1 = min(self.shape[1], int(y) + ky + 1)
        x0 = max(0, int(x) - kx)
        x1 = min(self.shape[2], int(x) + kx + 1)
        
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return
        
        # Create coordinate grids
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        
        # Calculate Gaussian
        gauss = amplitude * np.exp(
            -((zz - z)**2 / (2 * sigma_z**2)) 
            -((yy - y)**2 / (2 * sigma_xy**2)) 
            -((xx - x)**2 / (2 * sigma_xy**2))
        )
        
        # Add to image
        image[z0:z1, y0:y1, x0:x1] += gauss
        
    def _apply_photobleaching(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Apply exponential photobleaching decay to each channel.
        
        In real microscopy, photobleaching affects all fluorescent signal,
        including autofluorescence (baseline). We apply decay to entire image.
        """
        pb_cfg = self.config.get('photobleaching', {})
        t = frame_idx * self.frame_rate
        
        decay_rates = [
            pb_cfg.get('ch0_decay_rate', 0.0),
            pb_cfg.get('ch1_decay_rate', 0.0),
            pb_cfg.get('ch2_decay_rate', 0.0),
        ]
        
        for ch, k in enumerate(decay_rates):
            if k > 0:
                decay_factor = np.exp(-k * t)
                # Apply decay to entire image (baseline + spots)
                # This matches real microscopy where autofluorescence also decays
                frame[:, :, :, ch] = frame[:, :, :, ch] * decay_factor
        
        return frame
    
    def _apply_noise(self, frame: np.ndarray) -> np.ndarray:
        """Apply noise model (read noise + shot noise)."""
        noise_cfg = self.config.get('noise', {})
        read_noise_std = noise_cfg.get('read_noise_std', 10.0)
        shot_noise = noise_cfg.get('shot_noise_enabled', True)
        
        # Ensure non-negative before Poisson
        frame = np.maximum(frame, 0)
        
        if shot_noise:
            # Poisson noise (shot noise)
            frame = self.rng.poisson(frame.astype(np.float64)).astype(np.float32)
        
        if read_noise_std > 0:
            # Gaussian read noise
            frame += self.rng.normal(0, read_noise_std, frame.shape).astype(np.float32)
        
        return frame
    
    # =========================================================================
    # GROUND TRUTH EXPORT
    # =========================================================================
    
    def _generate_ground_truth(self) -> pd.DataFrame:
        """Generate MicroLive-compatible ground truth DataFrame.
        
        Includes compartment tracking and transition detection.
        """
        records = []
        
        for particle in self.particles:
            prev_compartment = None
            
            for frame in sorted(particle.position_history.keys()):
                pos = particle.position_history[frame]
                z, y, x = pos
                
                # Determine current compartment
                compartment = self._get_compartment_at_position(z, y, x)
                is_nuc = 1 if compartment == 'nucleus' else 0
                local_bg = self._get_local_background(z, y, x)
                
                # Detect compartment transition
                transition = None
                if prev_compartment is not None and compartment != prev_compartment:
                    transition = f"{prev_compartment}_to_{compartment}"
                
                # Base record
                record = {
                    'frame': frame,
                    'z': z,
                    'y': y,
                    'x': x,
                    'particle': particle.particle_id,
                    'cell_id': particle.cell_id,
                    'image_id': 0,
                    'spot_type': 0,
                    'birth_frame': particle.birth_frame,
                    'death_frame': particle.death_frame,
                    'trajectory_length': particle.death_frame - particle.birth_frame,
                    
                    # Spot properties
                    'snr_ch_0': particle.snr,
                    'psf_sigma_ch_0': particle.size,
                    'psf_amplitude_ch_0': local_bg * (1 + particle.snr),
                    
                    # Compartment tracking
                    'is_nuc': is_nuc,
                    'compartment': compartment,
                    'local_background': local_bg,
                    'compartment_transition': transition,
                    
                    # Colocalization
                    'has_ch1_partner': particle.has_ch1_partner,
                    'has_ch2_partner': particle.has_ch2_partner,
                    
                    # For MicroLive compatibility
                    'is_cluster': 0,
                    'unique_particle': particle.get_unique_id(0, 0),
                }
                
                records.append(record)
                
                # Also add ch1/ch2 records if colocalized
                if particle.has_ch1_partner:
                    ch1_record = record.copy()
                    ch1_record['spot_type'] = 1
                    ch1_record['unique_particle'] = particle.get_unique_id(0, 1)
                    ch1_mult = self.config.get('colocalization', {}).get('ch1_snr_multiplier', 0.8)
                    ch1_record['snr_ch_1'] = particle.snr * ch1_mult
                    ch1_record['psf_amplitude_ch_1'] = local_bg * (1 + particle.snr * ch1_mult)
                    records.append(ch1_record)
                
                if particle.has_ch2_partner:
                    ch2_record = record.copy()
                    ch2_record['spot_type'] = 2
                    ch2_record['unique_particle'] = particle.get_unique_id(0, 2)
                    ch2_mult = self.config.get('colocalization', {}).get('ch2_snr_multiplier', 0.5)
                    ch2_record['snr_ch_2'] = particle.snr * ch2_mult
                    ch2_record['psf_amplitude_ch_2'] = local_bg * (1 + particle.snr * ch2_mult)
                    records.append(ch2_record)
                
                prev_compartment = compartment
        
        return pd.DataFrame(records)
    
    # =========================================================================
    # MAIN SIMULATION LOOP
    # =========================================================================
    
    def run(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Execute simulation.
        
        Returns:
            image_stack: 5D array [T, Z, Y, X, C] as uint16
            df_ground_truth: MicroLive-compatible DataFrame
        """
        print(f"Starting simulation: {self.total_frames} frames, {self.shape} shape")
        
        # Initialize
        self._create_masks()
        self._plan_trajectories()
        print(f"Created {len(self.particles)} particles")
        
        # Pre-allocate image stack
        n_channels = 3
        self.image_stack = np.zeros(
            (self.total_frames, *self.shape, n_channels), 
            dtype=np.float32
        )
        
        # Record initial positions
        for particle in self.particles:
            if particle.is_alive(0):
                particle.record_position(0)
        
        # Simulation loop
        for frame in range(self.total_frames):
            if frame % 20 == 0:
                print(f"  Frame {frame}/{self.total_frames}")
            
            # Update positions (except frame 0, already at initial)
            if frame > 0:
                self._update_positions(frame)
            
            # Render frame
            self.image_stack[frame] = self._render_frame(frame)
        
        # Generate ground truth
        self.df_ground_truth = self._generate_ground_truth()
        
        # Convert to uint16
        self.image_stack = np.clip(self.image_stack, 0, 65535).astype(np.uint16)
        
        print(f"Simulation complete. Ground truth: {len(self.df_ground_truth)} records")
        
        return self.image_stack, self.df_ground_truth
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    def save_results(self, output_dir: str = None) -> None:
        """Save TIFF, masks, and ground truth to disk.
        
        Args:
            output_dir: Output directory (default: results/)
        """
        if output_dir is None:
            output_dir = self.config_path.parent / 'results'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving results to {output_dir}")
        
        # Save image stack (TCZYX format) with OME-TIFF metadata
        if self.image_stack is not None:
            # Transpose from TZYXC to TCZYX
            img_tczyx = np.moveaxis(self.image_stack, -1, 1)
            
            # Get voxel sizes from config
            img_cfg = self.config.get('image', {})
            voxel_yx_nm = img_cfg.get('voxel_size_yx_nm', 130.0)
            voxel_z_nm = img_cfg.get('voxel_size_z_nm', 300.0)
            
            sim_cfg = self.config.get('simulation', {})
            frame_rate_sec = sim_cfg.get('frame_rate_seconds', 5.0)
            
            # Convert to micrometers for OME-TIFF
            voxel_yx_um = voxel_yx_nm / 1000.0
            voxel_z_um = voxel_z_nm / 1000.0
            
            # Create OME-XML metadata
            ome_metadata = {
                'axes': 'TCZYX',
                'PhysicalSizeX': voxel_yx_um,
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': voxel_yx_um,
                'PhysicalSizeYUnit': 'µm',
                'PhysicalSizeZ': voxel_z_um,
                'PhysicalSizeZUnit': 'µm',
                'TimeIncrement': frame_rate_sec,
                'TimeIncrementUnit': 's',
                'Channel': {'Name': [f'Ch{i}' for i in range(img_tczyx.shape[1])]},
            }
            
            tifffile.imwrite(
                output_dir / 'simulated_spots.tif',
                img_tczyx.astype(np.uint16),
                compression='zlib',
                compressionargs={'level': 6},
                metadata=ome_metadata,
                ome=True,  # Enable OME-TIFF format
                photometric='minisblack',
            )
            print(f"  Saved simulated_spots.tif: {img_tczyx.shape}"
                  f" (voxel: {voxel_yx_nm}x{voxel_yx_nm}x{voxel_z_nm} nm, dt={frame_rate_sec}s)")
        
        # Save masks
        self._save_masks(output_dir)
        
        # Save ground truth
        if self.df_ground_truth is not None:
            self.df_ground_truth.to_csv(output_dir / 'ground_truth.csv', index=False)
            try:
                self.df_ground_truth.to_parquet(output_dir / 'ground_truth.parquet', index=False)
            except ImportError:
                pass  # pyarrow not installed, skip parquet
            print(f"  Saved ground_truth: {len(self.df_ground_truth)} records")
        
        # Save metadata
        self._save_metadata(output_dir)
        
    def _save_masks(self, output_dir: Path) -> None:
        """Save segmentation masks in MicroLive-compatible labeled format."""
        # Cytosol mask (complete cell = cytosol + nucleus)
        cytosol_mask = np.zeros(self.shape, dtype=np.uint8)
        for cell in self.cells:
            cell_interior = cell.cytosol_mask | cell.nucleus_mask
            cytosol_mask[cell_interior] = cell.cell_id
        
        tifffile.imwrite(
            output_dir / 'mask_cytosol.tif',
            cytosol_mask,
            compression='zlib',
            compressionargs={'level': 9}
        )
        print(f"  Saved mask_cytosol.tif: {cytosol_mask.shape}")
        
        # Nucleus mask
        nucleus_mask = np.zeros(self.shape, dtype=np.uint8)
        for cell in self.cells:
            nucleus_mask[cell.nucleus_mask] = cell.cell_id
        
        tifffile.imwrite(
            output_dir / 'mask_nucleus.tif',
            nucleus_mask,
            compression='zlib',
            compressionargs={'level': 9}
        )
        print(f"  Saved mask_nucleus.tif: {nucleus_mask.shape}")
        
        # Cytosol-only mask (excluding nucleus)
        cytosol_only_mask = np.zeros(self.shape, dtype=np.uint8)
        for cell in self.cells:
            cytosol_only = cell.cytosol_mask & ~cell.nucleus_mask
            cytosol_only_mask[cytosol_only] = cell.cell_id
        
        tifffile.imwrite(
            output_dir / 'mask_cytosol_no_nuclei.tif',
            cytosol_only_mask,
            compression='zlib',
            compressionargs={'level': 9}
        )
        print(f"  Saved mask_cytosol_no_nuclei.tif: {cytosol_only_mask.shape}")
        
    def _save_metadata(self, output_dir: Path) -> None:
        """Save simulation metadata."""
        lines = [
            "Simulation Metadata",
            "=" * 50,
            f"Config: {self.config_path}",
            f"Total Frames: {self.total_frames}",
            f"Frame Rate: {self.frame_rate} s",
            f"Image Size: {self.shape}",
            f"Voxel Size (YX): {self.voxel_yx} nm",
            f"Voxel Size (Z): {self.voxel_z} nm",
            "",
            "Particle Statistics",
            "-" * 50,
            f"Total Particles: {len(self.particles)}",
            f"Avg Birth Frame: {np.mean([p.birth_frame for p in self.particles]):.1f}",
            f"Avg Trajectory Length: {np.mean([p.death_frame - p.birth_frame for p in self.particles]):.1f}",
            f"Ch1 Colocalized: {sum(p.has_ch1_partner for p in self.particles)} ({100*sum(p.has_ch1_partner for p in self.particles)/len(self.particles):.0f}%)",
            f"Ch2 Colocalized: {sum(p.has_ch2_partner for p in self.particles)} ({100*sum(p.has_ch2_partner for p in self.particles)/len(self.particles):.0f}%)",
            "",
            "Baseline Intensities",
            "-" * 50,
            f"Outside Cell: {self.baseline_outside}",
            f"Cytosol: {self.baseline_cytosol}",
            f"Nucleus: {self.baseline_nucleus}",
        ]
        
        with open(output_dir / 'simulation_metadata.txt', 'w') as f:
            f.write('\n'.join(lines))
        print("  Saved simulation_metadata.txt")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line interface for running simulations."""
    parser = argparse.ArgumentParser(
        description='MicroLive Spot Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python spot_simulator.py
  python spot_simulator.py --config my_config.yaml
  python spot_simulator.py --config my_config.yaml --output ./my_results
        """
    )
    parser.add_argument(
        '--config', '-c',
        default='config_simple.yaml',
        help='Path to configuration YAML file (default: config_simple.yaml)'
    )
    parser.add_argument(
        '--output', '-o',
        default='results',
        help='Output directory (default: results)'
    )
    
    args = parser.parse_args()
    
    # Run simulation
    sim = SpotSimulator(args.config)
    sim.run()
    sim.save_results(args.output)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
