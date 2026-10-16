"""
Logging utility module for AI Traffic Monitoring System.

Configures file and console logging, and provides credential sanitization
utilities to prevent sensitive RTSP or HTTP camera credentials from leaking.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from config.settings import get_config


def sanitize_url(url: str) -> str:
    """
    Sanitize camera stream URLs to mask passwords and sensitive authentication info.

    Example:
        rtsp://admin:pass123@192.168.1.50:554/live -> rtsp://admin:***@192.168.1.50:554/live
    """
    if not url or not isinstance(url, str):
        return str(url)
    # Match scheme://user:pass@host pattern
    pattern = r"://([^:@\s]+):([^@\s]+)@"
    return re.sub(pattern, r"://\1:***@", url)


def setup_logger(
    name: str = "traffic_system",
    log_file: Optional[Path] = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Set up and configure a logger instance with console and file output.

    Args:
        name: Logger name.
        log_file: Path to destination log file. If None, uses config setting.
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        Configured logging.Logger instance.
    """
    config = get_config()
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    target_file = log_file or config.log_file
    try:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(target_file), encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning("Could not initialize file log handler: %s", e)

    return logger
