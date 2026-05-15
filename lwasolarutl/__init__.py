"""OVRO-LWA solar FITS image and spectrum processing utilities."""

from . import file, ndfits, visualization
from .file import (
    check_h5_fits_consistency,
    compress_fits_to_h5,
    recover_fits_from_h5,
)

__version__ = "0.1.0"

__all__ = [
    "file",
    "ndfits",
    "visualization",
    "compress_fits_to_h5",
    "recover_fits_from_h5",
    "check_h5_fits_consistency",
]
