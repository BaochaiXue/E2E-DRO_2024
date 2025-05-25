"""Distributionally Robust End-to-End Portfolio Construction package."""

from importlib.metadata import version

__all__ = ["__version__"]

try:  # pragma: no cover - metadata might not be available during tests
    __version__: str = version("e2edro")
except Exception:  # fallback when package metadata is not present
    __version__ = "0.0.0"

