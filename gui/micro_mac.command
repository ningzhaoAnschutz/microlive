#!/bin/bash
# launch_gui.command

# Find the conda base directory
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "Conda does not appear to be installed or not in your PATH."
    exit 1
fi

# Source conda’s shell functions
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the 'micro_mac' environment
conda activate micro_mac

# Change to the folder containing this script (which should also contain gui.py)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Launch the GUI script
python micro.py