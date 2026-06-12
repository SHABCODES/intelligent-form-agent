"""Structured logging setup."""

from __future__ import annotations
import logging
import sys
from typing import Optional


_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a logger configured with a consistent format."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(level or logging.INFO)
    logger.propagate = False
    return logger


# Root platform logger
log = get_logger("docai")
