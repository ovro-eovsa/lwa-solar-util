"""Visualization helpers for OVRO-LWA solar products.

Merged from `lwasolarproc.visualization` and the legacy ovro-lwa-solar plotter
(commit b55f56d5). No CASA or suncasa runtime dependencies.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Sequence

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.io import fits
from astropy.time import Time
from matplotlib import gridspec
from matplotlib.patches import Circle, Ellipse
from sunpy import map as smap

from dataclasses import replace
from matplotlib.dates import AutoDateFormatter, AutoDateLocator

from . import spec as spec_mod

try:  # Registers SunPy colormaps such as ``hinodexrt`` with Matplotlib.
    import sunpy.visualization.colormaps  # noqa: F401
except Exception:  # pragma: no cover - plotting still works with fallback cmap.
    pass


OVRO_LWA_LOCATION = EarthLocation(lat=37.23977727 * u.deg, lon=-118.2816667 * u.deg, height=1183 * u.m)
NJIT_LOGO_STR = "iVBORw0KGgoAAAANSUhEUgAAAHgAAAA3CAMAAADwtH5ZAAAAVFBMVEVHcEzuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCTuNCQLSl2nAAAAG3RSTlMA2xyijg4I9TgExewsRlFwFPqvz7x7IuVcZJtcMgeUAAADGklEQVRYw+XY2ZKzKhAAYJRFFgUFFJT3f8/TGJPRaP6aVGXIxemam8HlSwNiI0L/z5i7w78t5/DH2b6NQUObW3t02Xw4GfVGQ/M+eMfPrrBhL4uOBKV8PNyLRmgMSpr6CEesQlA20kNz16RG5Qhw8BYDFhdwqg6/h9nU6OezmE8pzaeLpyql5jkbMzhvMV7skJJbMMY+pOUqYzh8uHZMoT2dhlMaulMrlSmF6alxIb2AyJknkgU2YXINJ7eXzAu4uYar+umG4/To8rSBfDzDaycm2R7g6RLmv4IRRSeY1pdwc5Tn873egtEZvgomBxJA9m1p2CetVc55Kg+jOfe23+RYBoZr4ak1O1kPVV8A7l1eGMT4I+vBlYMR+5HLwncZ7tIVg8224mTZ1nBBKXhEe7nnjaQF4Lq6w5uMdePpGy+JD8Cb7IZSMEFHOZWHkbitJO/AF68ytC1EMG7iJQxFxL48WGXPfgnDzEzqFRzTP2Gu0mESZ9mKX87qKVw239/rlylsr4gu35FwcZCv4VNdIjhZH8B4OsAYpXUubYKe6mma6r7vKWVM7OpHM8fZ6P0PE7O5LJDk87LCtIkQs1nHQNC65TqakWBrvZTVkOdpU93DSentQoxuX3e/uJxFL2com3gciZVVaFbsEMPQqFA5Jy1eCBkhTy7eqPphMMPFWLIaSCyrAwgUSNIDNJqoO95CV1OWC8/3Ax6N50KxbyOxlfohB1U5j4mJmcrSB7Y5UOTj3dQXPTfYPUwQJSZzx6fPaPv9VYMfM2tFm81UlV9gq1TTD4NbvrMbb/mKKRKp7qYlkdfs77aTbFzWecVaY7dnJHgw+z80tzShHykfvbo9mHbspr82HxvhmzpktBZlUCRas6rKEz2VQmEOaxxyqnhuabmvFIwTN6TGEV2XSxWSjVZltetLfpIR7ejSUFqF1/QSUrXosiqi2jbKxqnwd69+lo0bOSvM1sYFW7qLV9aR4skCK6UpPbL5E6T/Qh9DuWlxV7yPoVDFXxha2E0QwkV5tjak+wLL9PiFsUWoNZp+gUU81ugrQcU31P8AFQ9Mc+zu8kIAAAAASUVORK5CYII="
CALTECH_LOGO = "iVBORw0KGgoAAAANSUhEUgAAALQAAABNCAMAAAA4jr7RAAAAS1BMVEVHcEz/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh7/bh5CVYR9AAAAGHRSTlMA2OBx/CBS9hhAtIcDyA7v5jQpYQimmEmEErTxAAAEtElEQVRo3u1Zi5KrKhAMoIAIIiDq/3/pBQZfiUnWbKr2VF26TtWuLGI7zPQ0ntutoKCgoKCgoKCgoKCgoKDgX8PATdMY272bx2nANMCFtPLqc+D+7guMpSUCacZ0r0b6ekHKqmpGHC6Iv0y6iffj+vec7djPK5iir5jQKszJpAkTH5AO9/e/J03RfIBuux+RJnq+TFp+h7QMz75D1Q7vSUsSHv9XkXZsoar1+it5T5qGydUfRdrkdEYtNaYhCq6wfUu6jfP+JtKdgOob0zLyxluIdvsj0n8UaVdBEm8PJ2nkqSj9A5HOgfZb4ck8RJcZtTHG1t0PIj1wa07ak0zD/J70UMdR+QHpJilHbw7BD6KHfBoamlFhHYDV2AwnkcY+oAGxJyJODe1p2msPdx71YRjl4Uy6c2k28nS4TDo9eR4PY/VIDMTFeraq4KxH/kg6/mF2cUNavLUnv5bx4NBuuF5IYyqqZWF/NVWGlAqMPmmU6ijfKYnuSc+JdC0OM1Heuq5l+2FlM2mGd6PiohHh6EXR5eTWSCGQxYo+kK5i7rgbF4vUA8lKpV2R7V3XCvwa9tDK3EXTkdio81edWNo9wztuoWtGe3QkrUKVhlwCcridgtKjTTIpy27GewTpQDbSGuVYzOJSWkvDXugWsOL7i+6etN8aVKUgJyBTUNg9Dumlpk5KThJDxDNpNhrObQs6YK+LZn7yI2mMez3ulQ7zc51Ov6Ll0RbnPU9TUyJnu8AEheYSfILcYsGaT0ifR7rjdZ3kdeANSeF7IA06DRElxy0KbzseqlyOPooePHIpPijLa6SNfpdTvHGjwDkPn0Q6rVKJcYGCVOrUsWC6Tafnpfbq/jppmxPtWb8MjWFX7NWTSE/V/AjMa3xiYiA99NLN0pyLpCEY2jzZB1HdEzmNtDvhPGtr9TFr9m3c/oK09PPJwtSl0Dd4NdpoHKvnOX1Kmhkg7V66vE9I38j8KNQh/IrUi2D1IhhtDinwJNJpiLXkCA651750eR+Rhkpk+3DI1BKQS6a1EgaqdHoR6cSjouftdpOmOlmuL5AG87FpbFwHItyOhxPMK50GduMmnDzdJcXxDNQy38hvkIZeHdvxyhleo7cp3dWw194npKEy9Bpq7vu2XnNvOV80IVv0WH+D9OCzMJB6SN9/srNrb34fJ9qfkwa1BI+B3QAeXKT06qA1zoykimmyUH2D9M0uhheH7uDV5tIgTsJ0AzfgEU5IV945YnKSMUGoIwLmxoMF+KgqDufPQe1X0iMkyPZ1aZVlbJYanbVSaPkwEj3sSprs/LTBD4oXzQW/M+Szqr9E+jahh242PZjharFjK2mjd4eA6Y61BkNk1MPZ4Euk96cq2E6T62lriCodQVh4GbrkyfZSbk3ZZYX1c6DdN9VU7d8iLQcq+mr9/uiWVtMRSHGtCOcKB5C4L+Fnctnhz5ox1qMJDrBKw+eIXji+sy8ChuMiycKn++vloB6XRebTb72u9UL4kTTdbSe4U2htzsS3CD61rnk8+cefMKUzzdSsnwE6Q2MnnOzRNA6Wroskn7vdH1nHq+H2OaQ8Mdby2grng/Ff+X+LgoKCgoKCgoKCgoKCgv8Z/gMgTqct1e1j7QAAAABJRU5ErkJggg=="


def _header_get(header, key: str, default=None):
    if hasattr(header, "get"):
        value = header.get(key)
        if value is not None:
            return value
        upper_value = header.get(key.upper())
        if upper_value is not None:
            return upper_value
        return default
    return default


def get_solar_altaz_multiple_times(times, location: EarthLocation = OVRO_LWA_LOCATION):
    """Return solar azimuth and altitude in degrees for one or more times."""
    obstime = Time(times)
    solar_altaz = get_sun(obstime).transform_to(AltAz(obstime=obstime, location=location))
    return solar_altaz.az.to_value(u.deg), solar_altaz.alt.to_value(u.deg)


def _safe_channel(meta: dict, name: str, idx: int, default: float) -> float:
    arr = meta.get(name)
    if arr is None:
        return default
    try:
        return float(arr[idx])
    except (IndexError, TypeError, ValueError):
        return default


def _extent_from_fov(fov: float) -> list[float]:
    half = fov / 2.0
    return [-half, half, -half, half]


def _set_solar_axes_fov(ax, fov: float, x_shift: float = 0.0, y_shift: float = 0.0) -> None:
    half = fov / 2.0
    ax.set_xlim([-half - x_shift, half - x_shift])
    ax.set_ylim([-half - y_shift, half - y_shift])


def _channel_image(data: np.ndarray, freq_idx: int, pol_idx: int = 0) -> np.ndarray:
    """Return a 2D ``(y, x)`` image from common slow-pipeline cube layouts."""
    arr = np.asarray(data)
    if arr.ndim == 2:
        return np.asarray(arr, dtype=float)
    if arr.ndim == 3:
        return np.asarray(arr[freq_idx, :, :], dtype=float)
    if arr.ndim == 4:
        return np.asarray(arr[pol_idx, freq_idx, :, :], dtype=float)
    raise ValueError(f"Expected a 2D, 3D, or 4D FITS image cube, got shape {arr.shape}")


def _channel_header(header, freq_hz: float | None = None):
    """Return a 2D HPLN/HPLT header suitable for a SunPy map channel slice."""
    channel_header = header.copy()
    channel_header["NAXIS"] = 2
    if "NAXIS1" in header:
        channel_header["NAXIS1"] = header["NAXIS1"]
    if "NAXIS2" in header:
        channel_header["NAXIS2"] = header["NAXIS2"]

    for axis in range(3, int(header.get("NAXIS", 2)) + 1):
        for prefix in ("NAXIS", "CTYPE", "CRVAL", "CDELT", "CRPIX", "CUNIT", "CROTA"):
            key = f"{prefix}{axis}"
            if key in channel_header:
                del channel_header[key]
    for key in list(channel_header):
        upper = key.upper()
        if upper.startswith("PC") or upper.startswith("CD"):
            suffix = upper[2:] if upper.startswith("PC") else upper[2:]
            parts = suffix.replace("_", " ").split()
            try:
                axes = [int(part) for part in parts[:2]]
            except ValueError:
                continue
            if len(axes) == 2 and max(axes) > 2:
                del channel_header[key]

    if freq_hz is not None:
        channel_header["RESTFRQ"] = float(freq_hz)
    return channel_header


def _axis_edges_arcsec(header, axis: int, n_pix: int) -> np.ndarray:
    crpix = float(_header_get(header, f"CRPIX{axis}", (n_pix + 1) / 2.0))
    crval = float(_header_get(header, f"CRVAL{axis}", 0.0))
    cdelt = float(_header_get(header, f"CDELT{axis}", 1.0))
    pixels = np.array([0.5, n_pix + 0.5], dtype=float)
    return (pixels - crpix) * cdelt + crval


def _image_extent_arcsec(header, image: np.ndarray) -> list[float]:
    ny, nx = image.shape
    x0, x1 = _axis_edges_arcsec(header, 1, nx)
    y0, y1 = _axis_edges_arcsec(header, 2, ny)
    return [float(x0), float(x1), float(y0), float(y1)]


def _plot_solar_limb(
    ax,
    solar_map,
    header,
    color: str = "w",
    alpha: float = 0.5,
    lw: float = 1.2,
    linestyle: str = "-",
) -> None:
    radius = _header_get(header, "RSUN_OBS")
    if radius is None:
        try:
            radius = solar_map.rsun_obs.to_value(u.arcsec)
        except Exception:
            radius = None
    if radius is not None and np.isfinite(float(radius)):
        ax.add_patch(Circle((0.0, 0.0), float(radius), fc="none", ec=color, lw=lw, alpha=alpha, ls=linestyle))


def _available_cmap(preferred: str, fallback: str = "afmhot"):
    try:
        return plt.get_cmap(preferred)
    except ValueError:
        return plt.get_cmap(fallback)


def _copy_cmap_with_bad(cmap_name: str, bad_color: str = "black"):
    cmap = plt.get_cmap(cmap_name)
    if hasattr(cmap, "copy"):
        cmap = cmap.copy()
    cmap.set_bad(bad_color)
    return cmap


def _logo_image(logo_string: str) -> np.ndarray:
    return mpimg.imread(io.BytesIO(base64.b64decode(logo_string)), format="png")


def _add_default_logos(fig) -> None:
    ax_logo1 = fig.add_axes([0.015, 0.035, 0.07, 0.07])
    ax_logo2 = fig.add_axes([0.005, -0.003, 0.09, 0.08])
    ax_logo1.imshow(_logo_image(NJIT_LOGO_STR))
    ax_logo2.imshow(_logo_image(CALTECH_LOGO))
    ax_logo1.axis("off")
    ax_logo2.axis("off")


def _hide_wcs_axis_labels(ax, panel_index: int) -> None:
    ax.coords[0].set_format_unit(u.arcsec)
    ax.coords[1].set_format_unit(u.arcsec)
    ax.coords[0].set_axislabel("")
    ax.coords[1].set_axislabel("")
    if panel_index not in [8, 9, 10, 11]:
        ax.coords[0].set_ticklabel_visible(False)
        ax.coords[0].set_ticks_visible(False)
    if panel_index not in [0, 4, 8]:
        ax.coords[1].set_ticklabel_visible(False)
        ax.coords[1].set_ticks_visible(False)


def _corner_noise_sigma(image: np.ndarray, corner_size: int = 56) -> float:
    size_y = min(int(corner_size), image.shape[0])
    size_x = min(int(corner_size), image.shape[1])
    corner = np.asarray(image[-size_y:, -size_x:], dtype=float)
    if corner.size == 0:
        return float("nan")
    return float(np.nanstd(corner))


def slow_pipeline_default_plot(
    fname,
    freqs_plt: Sequence[float] = (34.1, 38.7, 43.2, 47.8, 52.4, 57.0, 61.6, 66.2, 70.8, 75.4, 80.0, 84.5),
    fov: float = 14400,
    add_logo: bool = True,
    interpolation: str = "bicubic",
    apply_refraction_param: bool = False,
    spec_fits=None,
    spec_dur: float = 600.0,
    spec_cmap: str = "viridis",
    spec_vmin=None,
    spec_vmax=None,
    spec_norm: str = "log",
    spec_frange: Sequence[float] = (30.0, 88.0),
    apply_fiducial_primary_beam: bool = False,
    badants_arr=None,
):
    """
    Plot the default 12-panel slow-pipeline FITS product.

    ``spec_fits`` is intentionally ignored here. This copy assumes
    the spectrogram panel is disabled, so it does not import ``suncasa.dspec``.
    """
    from . import ndfits

    del spec_fits, spec_dur, spec_cmap, spec_vmin, spec_vmax, spec_norm, spec_frange

    meta, rdata = ndfits.read(fname)
    header = meta.get("header", {})

    if apply_fiducial_primary_beam:
        obstime = Time(_header_get(header, "DATE-OBS"))
        _, alt = get_solar_altaz_multiple_times(obstime)
        beam_gain = np.sin(np.deg2rad(alt)) ** 1.6
        if np.all(np.isfinite(beam_gain)) and np.all(beam_gain != 0):
            rdata[0, ...] /= beam_gain

    date_obs = _header_get(header, "DATE-OBS", "")
    rfrcor = bool(_header_get(header, "REFCOR", False) or _header_get(header, "rfrcor", False))

    fig = plt.figure(figsize=(8, 6.5))
    gs = gridspec.GridSpec(3, 4, left=0.07, right=0.98, top=0.94, bottom=0.10, wspace=0.02, hspace=0.02)

    freqs_mhz = np.asarray(meta["ref_cfreqs"], dtype=float) / 1e6
    do_badants_label = badants_arr is not None and len(badants_arr) == len(freqs_mhz)
    badants_arr = np.asarray(badants_arr) if do_badants_label else None

    image_cmap = _available_cmap("hinodexrt")
    axes = []
    for i in range(12):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor("black")
        freq_plt = freqs_plt[i]
        ax.text(0.02, 0.98, f"{freq_plt:.0f} MHz", color="w", ha="left", va="top", fontsize=11, transform=ax.transAxes)
        ax.set_xlabel("Solar X [arcsec]")
        ax.set_ylabel("Solar Y [arcsec]")
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, ha="center", va="center", rotation_mode="anchor")

        if np.min(np.abs(freqs_mhz - freqs_plt[i])) < 2.0:
            bd = int(np.argmin(np.abs(freqs_mhz - freqs_plt[i])))
            image = _channel_image(rdata, bd) / 1e6
            channel_header = _channel_header(header, freqs_mhz[bd] * 1e6)
            solar_map = smap.Map(image, channel_header)

            if apply_refraction_param and not rfrcor:
                x_shift = _safe_channel(meta, "refra_shift_x", bd, 0.0)
                y_shift = _safe_channel(meta, "refra_shift_y", bd, 0.0)
            else:
                x_shift = 0.0
                y_shift = 0.0

            vmaxplt = np.nanpercentile(image, 99.99)
            if not np.isfinite(vmaxplt):
                vmaxplt = None
            ax.imshow(
                solar_map.data,
                origin="lower",
                extent=_image_extent_arcsec(channel_header, image),
                cmap=image_cmap,
                vmin=0,
                vmax=vmaxplt,
                interpolation=interpolation,
            )
            _plot_solar_limb(ax, solar_map, channel_header, color="0.5", alpha=1.0, lw=1.0)
            _set_solar_axes_fov(ax, fov, x_shift=x_shift, y_shift=y_shift)

            bmaj = _safe_channel(meta, "bmaj", bd, 0.0)
            bmin = _safe_channel(meta, "bmin", bd, 0.0)
            bpa = _safe_channel(meta, "bpa", bd, 0.0)
            if bmaj > 0 and bmin > 0:
                beam = Ellipse(
                    (-fov * 0.375, -fov * 0.375),
                    bmaj * 3600,
                    bmin * 3600,
                    angle=-(90 - bpa),
                    fc="none",
                    lw=2,
                    ec="w",
                )
                ax.add_artist(beam)

            label_max = "nan" if vmaxplt is None else str(np.round(vmaxplt, 2))
            ax.text(
                0.99,
                0.02,
                r"$T_B^{\rm max}=$" + label_max + "MK",
                color="w",
                ha="right",
                va="bottom",
                fontsize=10,
                transform=ax.transAxes,
            )
            if do_badants_label:
                ax.text(
                    0.99,
                    0.98,
                    "$N_{ants}=$" + str(352 - badants_arr[bd]),
                    color="w",
                    ha="right",
                    va="top",
                    fontsize=10,
                    transform=ax.transAxes,
                )
        else:
            ax.text(0.5, 0.5, "No Data", color="w", ha="center", va="center", fontsize=18, transform=ax.transAxes)
            _set_solar_axes_fov(ax, fov)
        if i not in [8, 9, 10, 11]:
            ax.set_xlabel("")
            ax.get_xaxis().set_ticks([])
        if i not in [0, 4, 8]:
            ax.set_ylabel("")
            ax.get_yaxis().set_ticks([])
        axes.append(ax)

    suffix = "[refraction corrected]" if rfrcor else "[original]"
    fig.suptitle(f"OVRO-LWA {str(date_obs)[:19]} {suffix}", fontsize=12)
    if add_logo:
        _add_default_logos(fig)
    return fig, axes


def slow_pipeline_default_polarization_plot(
    i_fname,
    v_fname,
    freqs_plt: Sequence[float] = (34.1, 38.7, 43.2, 47.8, 52.4, 57.0, 61.6, 66.2, 70.8, 75.4, 80.0, 84.5),
    fov: float = 14400,
    add_logo: bool = True,
    interpolation: str = "bicubic",
    apply_refraction_param: bool = False,
    ratio_vmin: float = -0.6,
    ratio_vmax: float = 0.6,
    mask_sigma: float = 5.0,
    mask_corner_size: int = 56,
    badants_arr=None,
):
    """Plot a 12-panel Stokes ``V`` quicklook with per-panel ``max|V|`` labels."""
    from . import ndfits

    del ratio_vmin, ratio_vmax, mask_sigma, mask_corner_size

    meta_i, i_data = ndfits.read(i_fname)
    meta_v, v_data = ndfits.read(v_fname)

    header = meta_i.get("header", {})
    date_obs = _header_get(header, "DATE-OBS", "")
    rfrcor = bool(_header_get(header, "REFCOR", False) or _header_get(header, "rfrcor", False))

    freqs_i_mhz = np.asarray(meta_i["ref_cfreqs"], dtype=float) / 1e6
    freqs_v_mhz = np.asarray(meta_v["ref_cfreqs"], dtype=float) / 1e6
    if freqs_i_mhz.shape != freqs_v_mhz.shape or not np.allclose(freqs_i_mhz, freqs_v_mhz, atol=1e-3):
        raise ValueError("I and V FITS cubes must share the same frequency axis for Stokes V plotting")

    do_badants_label = badants_arr is not None and len(badants_arr) == len(freqs_i_mhz)
    badants_arr = np.asarray(badants_arr) if do_badants_label else None

    fig = plt.figure(figsize=(8, 6.5))
    gs = gridspec.GridSpec(3, 4, left=0.07, right=0.98, top=0.94, bottom=0.10, wspace=0.02, hspace=0.02)
    v_cmap = _copy_cmap_with_bad("RdBu_r", bad_color="white")

    axes = []
    for i in range(12):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor("white")
        freq_plt = freqs_plt[i]
        ax.text(0.02, 0.98, f"{freq_plt:.0f} MHz", color="k", ha="left", va="top", fontsize=11, transform=ax.transAxes)
        ax.set_xlabel("Solar X [arcsec]")
        ax.set_ylabel("Solar Y [arcsec]")
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, ha="center", va="center", rotation_mode="anchor")

        if np.min(np.abs(freqs_i_mhz - freq_plt)) < 2.0:
            bd = int(np.argmin(np.abs(freqs_i_mhz - freq_plt)))
            image_v = _channel_image(v_data, bd)
            v_plot = image_v / 1e6
            abs_vmax = np.nanpercentile(np.abs(v_plot), 99.99)
            if not np.isfinite(abs_vmax) or abs_vmax <= 0:
                abs_vmax = None
            max_abs_v = np.nanmax(np.abs(v_plot)) if np.any(np.isfinite(v_plot)) else np.nan

            channel_header = _channel_header(header, freqs_i_mhz[bd] * 1e6)
            solar_map = smap.Map(v_plot, channel_header)

            if apply_refraction_param and not rfrcor:
                x_shift = _safe_channel(meta_i, "refra_shift_x", bd, 0.0)
                y_shift = _safe_channel(meta_i, "refra_shift_y", bd, 0.0)
            else:
                x_shift = 0.0
                y_shift = 0.0

            ax.imshow(
                np.ma.masked_invalid(solar_map.data),
                origin="lower",
                extent=_image_extent_arcsec(channel_header, v_plot),
                cmap=v_cmap,
                vmin=-abs_vmax if abs_vmax is not None else None,
                vmax=abs_vmax,
                interpolation=interpolation,
            )
            _plot_solar_limb(ax, solar_map, channel_header, color="0.5", alpha=1.0, lw=1.0)
            _set_solar_axes_fov(ax, fov, x_shift=x_shift, y_shift=y_shift)

            bmaj = _safe_channel(meta_i, "bmaj", bd, 0.0)
            bmin = _safe_channel(meta_i, "bmin", bd, 0.0)
            bpa = _safe_channel(meta_i, "bpa", bd, 0.0)
            if bmaj > 0 and bmin > 0:
                beam = Ellipse(
                    (-fov * 0.375, -fov * 0.375),
                    bmaj * 3600,
                    bmin * 3600,
                    angle=-(90 - bpa),
                    fc="none",
                    lw=2,
                    ec="k",
                )
                ax.add_artist(beam)
            v_label = "nan" if not np.isfinite(max_abs_v) else f"{max_abs_v:.2f}MK"
            ax.text(
                0.99,
                0.02,
                r"$\max |V|=$" + v_label,
                color="k",
                ha="right",
                va="bottom",
                fontsize=9,
                transform=ax.transAxes,
            )
            if do_badants_label:
                ax.text(
                    0.99,
                    0.98,
                    "$N_{ants}=$" + str(352 - badants_arr[bd]),
                    color="k",
                    ha="right",
                    va="top",
                    fontsize=10,
                    transform=ax.transAxes,
                )
        else:
            ax.text(0.5, 0.5, "No Data", color="k", ha="center", va="center", fontsize=18, transform=ax.transAxes)
            _set_solar_axes_fov(ax, fov)

        if i not in [8, 9, 10, 11]:
            ax.set_xlabel("")
            ax.get_xaxis().set_ticks([])
        if i not in [0, 4, 8]:
            ax.set_ylabel("")
            ax.get_yaxis().set_ticks([])
        axes.append(ax)

    suffix = "[refraction corrected]" if rfrcor else "[original]"
    fig.suptitle(f"OVRO-LWA {str(date_obs)[:19]} Stokes V {suffix}", fontsize=12)
    if add_logo:
        _add_default_logos(fig)
    return fig, axes


def plot_spec(
    fits_path_or_spec,
    outpath=None,
    vmax_I=None,
    pct_hi_I=99,
    vi_range=(-0.3, 0.3),
    cmap_I="inferno",
    cmap_VI="RdBu_r",
    figsize=(14, 7),
    suptitle=None,
    t_range_ratio=(0.0, 1.0),
    f_range_ratio=(0.0, 1.0),
):
    """
    Two-panel dynamic spectrum: Stokes I (Jy) and V / I.

    FITS format matches `lwasolarview` (see `lwasolarutl.spec.load_spectrum_fits`).
    Based on `plot_spec_fits.py` from
    https://github.com/peijin94/lwasolarview/blob/main/plot_spec_fits.py

    Parameters
    ----------
    fits_path_or_spec : str or LwaSpectrum
        Path to an LWA spectrum FITS file, or a pre-loaded :class:`~lwasolarutl.spec.LwaSpectrum`.
    outpath : str, optional
        If set, save figure to this path (PNG recommended).
    vmax_I : float, optional
        Fixed colour maximum for Stokes I; if omitted, use ``pct_hi_I`` percentile.
    pct_hi_I : float
        Upper percentile for auto Stokes-I vmax (default 99).
    vi_range : (float, float)
        Colour limits for the V / I panel.
    cmap_I, cmap_VI : str
        Colormaps for the two panels.
    figsize : tuple
        Figure size in inches.
    suptitle : str, optional
        Override automatic suptitle.
    t_range_ratio : (float, float)
        Fraction of the **time** axis to plot, each in ``[0, 1]`` with ``lo < hi``
        (e.g. ``(0.25, 0.75)`` shows the middle half). Default ``(0, 1)`` is full range.
    f_range_ratio : (float, float)
        Same for the **frequency** axis (first index of the spectrum array).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray
        The two axes (Stokes I, then V / I).
    """
    if isinstance(fits_path_or_spec, str):
        sp = spec_mod.load_spectrum_fits(fits_path_or_spec)
    else:
        sp = fits_path_or_spec

    def _ratio_slice(n: int, r, name: str):
        if len(r) != 2:
            raise ValueError(f"{name} must be a length-2 sequence, got {r!r}")
        r0, r1 = float(r[0]), float(r[1])
        r0 = float(np.clip(r0, 0.0, 1.0))
        r1 = float(np.clip(r1, 0.0, 1.0))
        if r0 >= r1:
            raise ValueError(f"{name} must satisfy lo < hi after clipping; got {r!r}")
        lo = int(np.floor(r0 * n))
        hi = int(np.ceil(r1 * n))
        lo = max(0, min(lo, max(0, n - 1)))
        hi = max(lo + 1, min(hi, n))
        return slice(lo, hi)

    nfreq, ntime = sp.spec_I.shape
    f_sl = _ratio_slice(nfreq, f_range_ratio, "f_range_ratio")
    t_sl = _ratio_slice(ntime, t_range_ratio, "t_range_ratio")
    sp = replace(
        sp,
        spec_I=np.asarray(sp.spec_I[f_sl, t_sl], dtype=float),
        spec_V=np.asarray(sp.spec_V[f_sl, t_sl], dtype=float),
        freqs_mhz=np.asarray(sp.freqs_mhz[f_sl], dtype=float),
        times=sp.times[t_sl],
    )

    spec_VI = spec_mod.vi_ratio(sp)
    vmin_I = max(float(np.nanpercentile(sp.spec_I, 3)), 0.0)
    vmax_I_eff = (
        float(vmax_I) if vmax_I is not None else float(np.nanpercentile(sp.spec_I, pct_hi_I))
    )
    vi_lo, vi_hi = vi_range

    times_dt = sp.times.to_datetime()
    extent = [
        times_dt[0],
        times_dt[-1],
        float(sp.freqs_mhz[0]),
        float(sp.freqs_mhz[-1]),
    ]

    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"hspace": 0.08}
    )

    ax_I = axes[0]
    im_I = ax_I.imshow(
        sp.spec_I,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=vmin_I,
        vmax=vmax_I_eff,
        cmap=cmap_I,
        interpolation="nearest",
    )
    ax_I.set_ylabel("Frequency (MHz)", fontsize=11)
    ax_I.set_title("Stokes I", fontsize=11, loc="left")
    cb_I = fig.colorbar(im_I, ax=ax_I, pad=0.01, fraction=0.03)
    cb_I.set_label("Flux density (Jy)", fontsize=9)

    ax_VI = axes[1]
    im_VI = ax_VI.imshow(
        spec_VI,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=vi_lo,
        vmax=vi_hi,
        cmap=cmap_VI,
        interpolation="nearest",
    )
    ax_VI.set_ylabel("Frequency (MHz)", fontsize=11)
    ax_VI.set_xlabel("Time (UT)", fontsize=11)
    ax_VI.set_title("V / I", fontsize=11, loc="left")
    cb_VI = fig.colorbar(im_VI, ax=ax_VI, pad=0.01, fraction=0.03)
    cb_VI.set_label("V / I", fontsize=9)

    locator = AutoDateLocator(minticks=4, maxticks=10)
    formatter = AutoDateFormatter(locator)
    formatter.scaled[1 / 24] = "%H:%M"
    formatter.scaled[1 / (24 * 60)] = "%H:%M"
    ax_VI.xaxis.set_major_locator(locator)
    ax_VI.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=0, ha="center")

    if suptitle is None:
        suptitle = "OVRO-LWA  {}  {}–{} UT".format(
            sp.times[0].strftime("%Y-%m-%d"),
            sp.times[0].strftime("%H:%M:%S"),
            sp.times[-1].strftime("%H:%M:%S"),
        )
    fig.suptitle(suptitle, fontsize=12, y=1.01)

    if outpath is not None:
        os.makedirs(os.path.dirname(os.path.abspath(outpath)) or ".", exist_ok=True)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")

    return fig, axes


def make_allsky_image_plots(
    allsky_fitsfiles,
    vmaxs=(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5),
    vmins=(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    cmap="viridis",
):
    """Make a 12-panel plot from all-sky FITS files generated by the pipeline."""
    fig = plt.figure(figsize=(8, 6.3))
    gs = gridspec.GridSpec(3, 4, left=0.02, right=0.98, top=0.94, bottom=0.02, wspace=0.02, hspace=0.02)
    axes = []
    bands = ["32MHz", "36MHz", "41MHz", "46MHz", "50MHz", "55MHz", "59MHz", "64MHz", "69MHz", "73MHz", "78MHz", "82MHz"]
    plotted_fits = []
    allsky_fitsfiles = [str(path) for path in allsky_fitsfiles]

    for i, band in enumerate(bands):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor("black")
        found_band = False
        for fits_file in allsky_fitsfiles:
            if band in fits_file:
                with fits.open(fits_file) as hdu:
                    ax.imshow(hdu[0].data[0, 0], origin="lower", vmin=vmins[i], vmax=vmaxs[i], cmap=cmap)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.text(0.02, 0.98, f"{band[:2]} MHz", color="w", ha="left", va="top", fontsize=11, transform=ax.transAxes)
                plotted_fits.append(fits_file)
                found_band = True
                break

        if not found_band:
            ax.text(0.5, 0.5, "No Data", color="w", ha="center", va="center", fontsize=18, transform=ax.transAxes)
        axes.append(ax)

    if len(plotted_fits) > 1:
        basename_parts = os.path.basename(plotted_fits[0]).split(".")
        if len(basename_parts) > 2:
            timestr0 = basename_parts[2]
            timestr = timestr0[:13] + ":" + timestr0[13:15] + ":" + timestr0[15:17]
            fig.suptitle("OVRO-LWA All Sky Images " + timestr, fontsize=12)
        else:
            fig.suptitle("OVRO-LWA All Sky Images", fontsize=12)
        return fig, axes
    return -1
