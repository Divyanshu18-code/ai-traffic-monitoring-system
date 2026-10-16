"""
Pytest configuration and environment initialization for Windows.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Prevent OpenMP runtime library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure PyTorch lib directory is explicitly registered in DLL search path
if sys.platform == "win32":
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

# Import torch first
import torch
