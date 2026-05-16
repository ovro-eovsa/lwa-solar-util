"""Load OVRO-LWA dynamic spectrum FITS (Stokes I and V)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from astropy.io import fits
from astropy.time import Time


@dataclass
class LwaSpectrum:
    """Dynamic spectrum arrays and axes from a standard LWA spec FITS."""

    spec_I: np.ndarray
    spec_V: np.ndarray
    freqs_mhz: np.ndarray
    times: np.ndarray
    path: str


def load_spectrum_fits(path: str) -> LwaSpectrum:
    """
    Read an LWA dynamic-spectrum FITS file.

    Expected layout (as in `lwasolarview`):

    - HDU 0: ``data`` shape ``(2, 1, nfreq, ntime)`` — Stokes I and V
    - HDU 1: binary table with ``sfreq`` (GHz)
    - HDU 2: binary table with ``mjd`` and ``time`` (milliseconds within day)

    Parameters
    ----------
    path : str
        Path to the FITS file.

    Returns
    -------
    LwaSpectrum
        ``spec_I``, ``spec_V`` shaped ``(nfreq, ntime)``; ``freqs_mhz`` length
        ``nfreq``; ``times`` as ``astropy.time.Time`` length ``ntime``.
    """
    path = os.path.abspath(path)
    with fits.open(path) as f:
        data = np.asarray(f[0].data)
        freqs_ghz = np.asarray(f[1].data["sfreq"], dtype=float)
        ut = f[2].data
        mjd = ut["mjd"].astype(np.float64) + ut["time"].astype(np.float64) / 1000.0 / 86400.0

    if data.ndim != 4 or data.shape[0] < 2 or data.shape[1] != 1:
        raise ValueError(
            f"Expected primary data shape (2+, 1, nfreq, ntime); got {data.shape}"
        )

    spec_I = np.asarray(data[0, 0], dtype=float)
    spec_V = np.asarray(data[1, 0], dtype=float)
    freqs_mhz = freqs_ghz * 1e3
    times = Time(mjd, format="mjd")

    return LwaSpectrum(
        spec_I=spec_I,
        spec_V=spec_V,
        freqs_mhz=freqs_mhz,
        times=times,
        path=path,
    )


def vi_ratio(spec: LwaSpectrum, i_floor_percentile: float = 0.1) -> np.ndarray:
    """V / I with a floor on I based on a percentile of Stokes I (reduces noise blow-up)."""
    floor = max(float(np.nanpercentile(spec.spec_I, i_floor_percentile)), 1e-6)
    return spec.spec_V / np.where(spec.spec_I > floor, spec.spec_I, np.nan)


def robust_vmin_vmax(arr: np.ndarray, pct_lo: float = 0.5, pct_hi: float = 99.5) -> Tuple[float, float]:
    """Percentile limits ignoring NaNs."""
    return (
        float(np.nanpercentile(arr, pct_lo)),
        float(np.nanpercentile(arr, pct_hi)),
    )
