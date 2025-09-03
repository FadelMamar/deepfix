"""
Cursor CLI Integration

A Python wrapper around Cursor's CLI for non-interactive mode.
"""

from .cursor import Cursor
from .config import CursorConfig

__all__ = ["Cursor", "CursorConfig"]
