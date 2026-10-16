"""
File operations and safe path utilities for AI Traffic Monitoring System.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Union

from config.settings import get_config


def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename to prevent directory traversal or invalid characters.

    Args:
        filename: Original user filename.

    Returns:
        Clean, safe filename preserving extension.
    """
    raw_name = Path(filename).name
    # Remove path traversal tokens and dangerous characters
    clean_name = re.sub(r"[^\w\-.]", "_", raw_name)
    # Strip leading dots or hyphens
    clean_name = clean_name.lstrip(".-")
    if not clean_name:
        clean_name = f"unnamed_{uuid.uuid4().hex[:8]}"
    return clean_name


def format_bytes(size_bytes: Union[int, float]) -> str:
    """
    Format byte count into human-readable string (KB, MB, GB).

    Args:
        size_bytes: File size in bytes.

    Returns:
        Formatted string (e.g., '14.2 MB').
    """
    if size_bytes < 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    size = float(size_bytes)
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.1f} {units[unit_idx]}"


def generate_unique_filepath(directory: Path, original_filename: str, prefix: str = "") -> Path:
    """
    Generate a non-colliding file path in the given directory.

    Args:
        directory: Target destination directory.
        original_filename: Original filename to preserve extension.
        prefix: Optional prefix for the output filename.

    Returns:
        Safe, unique Path in the target directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(original_filename)
    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix or ".mp4"
    unique_id = uuid.uuid4().hex[:8]

    if prefix:
        filename = f"{prefix}_{stem}_{unique_id}{suffix}"
    else:
        filename = f"{stem}_{unique_id}{suffix}"

    return directory / filename
