"""MicroLive Spot Simulation Package.

Generates synthetic 3D+T multi-channel microscopy images with known
ground truth for validating MicroLive's analysis pipeline. Features
include Brownian diffusion, photobleaching, colocalization, and
multi-cell geometries.

Modules:
    spot_simulator      - Core simulator (SpotSimulator, Particle, CellRegion)
    run_simulation      - CLI entry point
    visualize_results   - Generate PNG visualizations from simulation output

Sub-packages:
    tests               - Validation test suites (API + GUI)
"""

from .spot_simulator import SpotSimulator

__all__ = ['SpotSimulator']
__version__ = '1.0.0'
