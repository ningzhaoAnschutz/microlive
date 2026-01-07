"""Analysis pipeline modules for MicroLive.

These pipelines provide automated batch processing workflows
for common microscopy analysis tasks.

Available pipelines:
    - pipeline_particle_tracking: Full particle tracking workflow
    - pipeline_FRAP: FRAP (Fluorescence Recovery After Photobleaching) analysis
    - pipeline_folding_efficiency: Protein folding efficiency quantification
    - pipeline_spot_detection_no_tracking: Spot detection without linking
"""

# Note: Imports are intentionally lazy to avoid circular imports
# and speed up package loading. Import pipelines directly when needed:
#
#   from microlive.pipelines.pipeline_particle_tracking import pipeline_particle_tracking
#   from microlive.pipelines.pipeline_FRAP import run_frap_pipeline
