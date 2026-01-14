# -*- coding: utf-8 -*-
"""
MicroLive - Live-cell microscopy image analysis toolkit

A Python-based GUI application for live-cell microscopy image analysis
and single-molecule measurements.

Example usage:
    # Programmatic API
    from microlive import microscopy as mi
    # Or: import microlive.microscopy as mi
    
    # Load and analyze images
    reader = mi.ReadLif("experiment.lif")
    images = reader.get_images()
    
    # Run segmentation
    seg = mi.CellSegmentation(images[0])
    masks = seg.calculate_masks()

Authors:
    Luis U. Aguilera, William S. Raymond, Rhiannon M. Sears,
    Nathan L. Nowling, Brian Munsky, Ning Zhao
"""

__version__ = "1.0.14"
__author__ = "Luis U. Aguilera, William S. Raymond, Rhiannon M. Sears, Nathan L. Nowling, Brian Munsky, Ning Zhao"

# Package name (for backward compatibility)
name = 'microlive'

# Direct submodule access (import these directly, not through __getattr__)
# Users should use:
#   from microlive import microscopy as mi
#   import microlive.microscopy as mi
#   from microlive.microscopy import ReadLif
#
# Note: We intentionally do NOT import microscopy here to avoid
# circular imports with imports.py and slow startup times.

__all__ = [
    "__version__",
    "__author__",
    # Submodules (listed for pdoc documentation discovery)
    "microscopy",
    "imports",
    "pipelines",
    "utils",
    "gui",
]