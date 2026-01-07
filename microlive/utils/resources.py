"""Resource file utilities for MicroLive.

Handles finding bundled data files like icons and ML models,
whether running from source or installed as a package.
"""

from pathlib import Path
import importlib.resources


def get_package_data_dir():
    """Get the path to the package data directory.
    
    Returns:
        Path: Path to microlive/data directory.
    """
    try:
        # Python 3.9+ with importlib.resources.files
        return Path(importlib.resources.files("microlive.data"))
    except (TypeError, AttributeError):
        # Fallback for older Python or editable installs
        return Path(__file__).parent.parent / "data"


def get_icon_path():
    """Get the path to the application icon.
    
    Returns:
        Path or None: Path to icon_micro.png, or None if not found.
    """
    # Try package data first
    icon_path = get_package_data_dir() / "icons" / "icon_micro.png"
    if icon_path.exists():
        return icon_path
    
    # Fallback to docs/icons (for development)
    dev_path = Path(__file__).parent.parent.parent / "docs" / "icons" / "icon_micro.png"
    if dev_path.exists():
        return dev_path
    
    return None


def get_model_path():
    """Get the path to the ML spot detection model.
    
    Returns:
        Path or None: Path to spot_detection_cnn.pth, or None if not found.
    """
    # Try package data first
    model_path = get_package_data_dir() / "models" / "spot_detection_cnn.pth"
    if model_path.exists():
        return model_path
    
    # Fallback to modeling directory (for development)
    dev_path = Path(__file__).parent.parent.parent / "modeling" / "machine_learning" / "spot_detection_cnn.pth"
    if dev_path.exists():
        return dev_path
    
    return None
