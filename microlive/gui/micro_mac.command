#!/bin/bash
# MicroLive GUI Launcher (macOS)

# Find the conda base directory
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "Conda does not appear to be installed or not in your PATH."
    exit 1
fi

# Source conda's shell functions
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the 'microlive' environment
conda activate microlive

# Launch MicroLive using the pip-installed entry point
microlive