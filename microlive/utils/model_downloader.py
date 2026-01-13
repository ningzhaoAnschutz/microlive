"""
Model download utilities for MicroLive.

This module provides functions to download and cache pretrained models from
the MicroLive GitHub repository. It follows the same patterns used by Cellpose
for robust model provisioning.

Models are downloaded on first use and cached locally in ~/.microlive/models/
to avoid repeated downloads.
"""

import os
import ssl
import shutil
import tempfile
import logging
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Base URL for raw GitHub content
_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/ningzhaoAnschutz/microlive/main"

# Model URLs - Add new models here
MODEL_URLS = {
    "frap_nuclei": f"{_GITHUB_RAW_BASE}/modeling/cellpose_models/cellpose_models/FRAP_nuclei_model/models/cellpose_1728581750.581418",
}

# Local cache directory (similar to Cellpose's ~/.cellpose/models/)
_MODEL_DIR_ENV = os.environ.get("MICROLIVE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = Path.home() / ".microlive" / "models"
MODEL_DIR = Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT


# =============================================================================
# Download Utilities (adapted from Cellpose)
# =============================================================================

def download_url_to_file(url: str, dst: str, progress: bool = True) -> None:
    """
    Download object at the given URL to a local path.
    
    Adapted from Cellpose/torch implementation for robustness.
    
    Args:
        url: URL of the object to download.
        dst: Full path where object will be saved.
        progress: Whether to display a progress bar. Default: True.
    
    Raises:
        HTTPError: If the server returns an error status.
        URLError: If the URL cannot be reached.
    """
    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
        progress = False
    
    file_size = None
    
    # Handle SSL certificate verification issues
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        u = urlopen(url, context=ssl_context)
    except URLError as e:
        raise URLError(f"Failed to connect to {url}: {e}")
    
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    
    # Save to temp file first, then move (atomic operation)
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    os.makedirs(dst_dir, exist_ok=True)
    
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        if HAS_TQDM and progress:
            with tqdm(total=file_size, disable=not progress, unit="B", 
                      unit_scale=True, unit_divisor=1024,
                      desc=f"Downloading {Path(dst).name}") as pbar:
                while True:
                    buffer = u.read(8192)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    pbar.update(len(buffer))
        else:
            # Simple download without progress bar
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
        
        f.close()
        shutil.move(f.name, dst)
        logger.info(f"Successfully downloaded model to {dst}")
        
    except Exception as e:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
        raise RuntimeError(f"Download failed: {e}")
    finally:
        if os.path.exists(f.name):
            try:
                os.remove(f.name)
            except OSError:
                pass


# =============================================================================
# Model Cache Functions
# =============================================================================

def get_model_path(model_name: str) -> Path:
    """
    Get the local cache path for a model.
    
    Args:
        model_name: Name of the model (e.g., "frap_nuclei").
        
    Returns:
        Path to the cached model file.
    """
    return MODEL_DIR / model_name


def is_model_cached(model_name: str) -> bool:
    """
    Check if a model is already cached locally.
    
    Args:
        model_name: Name of the model.
        
    Returns:
        True if the model exists locally, False otherwise.
    """
    return get_model_path(model_name).exists()


def cache_model(model_name: str, force_download: bool = False) -> str:
    """
    Ensure a model is cached locally, downloading if necessary.
    
    This function follows the Cellpose pattern:
    1. Check if model exists in local cache
    2. If not (or force_download=True), download from GitHub
    3. Return the local path
    
    Args:
        model_name: Name of the model (must be in MODEL_URLS).
        force_download: If True, re-download even if cached.
        
    Returns:
        String path to the cached model file.
        
    Raises:
        ValueError: If model_name is not recognized.
        RuntimeError: If download fails.
    """
    if model_name not in MODEL_URLS:
        available = ", ".join(MODEL_URLS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cached_file = get_model_path(model_name)
    
    if not cached_file.exists() or force_download:
        url = MODEL_URLS[model_name]
        logger.info(f"Downloading model '{model_name}' from {url}")
        print(f"Downloading MicroLive model '{model_name}' (first time only)...")
        
        try:
            download_url_to_file(url, str(cached_file), progress=True)
        except (HTTPError, URLError) as e:
            raise RuntimeError(
                f"Failed to download model '{model_name}' from GitHub. "
                f"Error: {e}\n\n"
                f"If this persists, you can manually download from:\n"
                f"  {url}\n"
                f"And place it at:\n"
                f"  {cached_file}"
            )
    else:
        logger.debug(f"Model '{model_name}' already cached at {cached_file}")
    
    return str(cached_file)


# =============================================================================
# Convenience Functions for Specific Models
# =============================================================================

def get_frap_nuclei_model_path() -> str:
    """
    Get the path to the FRAP nuclei segmentation model.
    
    Downloads the model from GitHub if not already cached locally.
    The model is stored in ~/.microlive/models/frap_nuclei
    
    Returns:
        String path to the FRAP nuclei model file.
        
    Example:
        >>> from microlive.utils.model_downloader import get_frap_nuclei_model_path
        >>> model_path = get_frap_nuclei_model_path()
        >>> # Use with Cellpose
        >>> from cellpose import models
        >>> model = models.CellposeModel(pretrained_model=model_path)
    """
    return cache_model("frap_nuclei")


# =============================================================================
# Verification and Diagnostics
# =============================================================================

def verify_model_integrity(model_name: str) -> bool:
    """
    Verify that a cached model file exists and has non-zero size.
    
    Args:
        model_name: Name of the model to verify.
        
    Returns:
        True if the model file exists and is valid.
    """
    model_path = get_model_path(model_name)
    if not model_path.exists():
        return False
    
    # Check file size (should be > 1MB for a real model)
    size_bytes = model_path.stat().st_size
    if size_bytes < 1_000_000:
        logger.warning(f"Model file seems too small ({size_bytes} bytes): {model_path}")
        return False
    
    return True


def list_cached_models() -> dict:
    """
    List all cached models and their status.
    
    Returns:
        Dictionary mapping model names to their cache status and size.
    """
    result = {}
    for name in MODEL_URLS:
        path = get_model_path(name)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            result[name] = {"cached": True, "size_mb": round(size_mb, 2), "path": str(path)}
        else:
            result[name] = {"cached": False, "size_mb": 0, "path": str(path)}
    return result


def clear_model_cache(model_name: str = None) -> None:
    """
    Clear cached models.
    
    Args:
        model_name: Specific model to clear, or None to clear all.
    """
    if model_name:
        path = get_model_path(model_name)
        if path.exists():
            path.unlink()
            logger.info(f"Cleared cached model: {model_name}")
    else:
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
            logger.info(f"Cleared all cached models from {MODEL_DIR}")
