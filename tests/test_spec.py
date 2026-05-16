"""Tests for LWA spectrum FITS loading and plotting."""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC_FITS = ROOT / "demofile" / "20260513.fits"


@pytest.mark.skipif(not SPEC_FITS.is_file(), reason="demofile/20260513.fits missing")
def test_load_spectrum_fits():
    import lwasolarutl.spec as spec

    s = spec.load_spectrum_fits(str(SPEC_FITS))
    assert s.spec_I.shape == s.spec_V.shape
    assert s.spec_I.ndim == 2
    assert len(s.freqs_mhz) == s.spec_I.shape[0]
    assert len(s.times) == s.spec_I.shape[1]
    assert np.isfinite(s.freqs_mhz[0])


@pytest.mark.skipif(not SPEC_FITS.is_file(), reason="demofile/20260513.fits missing")
def test_plot_spec():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lwasolarutl.visualization import plot_spec

    fig, axes = plot_spec(str(SPEC_FITS))
    assert len(axes) == 2
    plt.close(fig)

    fig2, axes2 = plot_spec(
        str(SPEC_FITS),
        t_range_ratio=(0.2, 0.8),
        f_range_ratio=(0.25, 0.75),
    )
    assert len(axes2) == 2
    plt.close(fig2)
