"""
Refraction correction for OVRO-LWA multi-frequency solar FITS cubes.

Ported from ovro-lwa-solar ``refraction_correction.py`` (commit b55f56d5):
https://github.com/ovro-eovsa/ovro-lwa-solar/blob/b55f56d5/ovrolwasolar/refraction_correction.py

Uses ``lwasolarutl.ndfits`` instead of suncasa (no CASA / suncasa dependency).
"""

from __future__ import annotations

import os
import warnings
from shutil import copyfile

import numpy as np
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import binary_dilation, binary_erosion, center_of_mass
from skimage.morphology import convex_hull_image, remove_small_objects

from . import ndfits


def thresh_func(freq):
    """Return brightness threshold (Tb) for frequency ``freq`` in Hz."""
    return 1.1e6 * (1 - 1.8e4 * freq ** (-0.6))


def find_quite_sun_region(data, thresh, min_size, convex_hull=False):
    threshed_img = data > thresh
    threshed_img_1st = remove_small_objects(
        threshed_img, min_size=min_size, connectivity=1
    )
    threshed_img_2nd = binary_erosion(threshed_img_1st, iterations=3)
    threshed_img_3rd = remove_small_objects(
        threshed_img_2nd, min_size=min_size, connectivity=1
    )
    threshed_img_4th = binary_dilation(threshed_img_3rd, iterations=3)
    if convex_hull:
        threshed_img_4th = convex_hull_image(threshed_img_4th)
    return threshed_img_4th


def find_center_of_thresh(
    data_this, thresh, meta, index, min_size_50=1000, convex_hull=False
):
    """
    Find the center of the thresholded image.

    ``min_size_50`` is the smallest allowable object area in pixels at 50 MHz;
    ``min_size`` scales with ``1/(nu[MHz]/50 MHz)**2``.
    """
    meta_header = meta["header"]
    min_size = (
        min_size_50
        / (meta_header["CDELT1"] / 60.0) ** 2.0
        / (meta["ref_cfreqs"][index] / 50e6) ** 2.0
    )
    threshed_img_4th = find_quite_sun_region(
        data_this, thresh, min_size, convex_hull=convex_hull
    )

    com = center_of_mass(threshed_img_4th)

    x_arr = meta_header["CRVAL1"] + meta_header["CDELT1"] * (
        np.arange(meta_header["NAXIS1"]) - (meta_header["CRPIX1"] - 1)
    )
    y_arr = meta_header["CRVAL2"] + meta_header["CDELT2"] * (
        np.arange(meta_header["NAXIS2"]) - (meta_header["CRPIX2"] - 1)
    )

    com_x_arcsec = x_arr[0] + com[1] * (x_arr[-1] - x_arr[0]) / (len(x_arr) - 1)
    com_y_arcsec = y_arr[0] + com[0] * (y_arr[-1] - y_arr[0]) / (len(y_arr) - 1)

    x_arr_new = x_arr - com_x_arcsec
    y_arr_new = y_arr - com_y_arcsec

    return [com_x_arcsec, com_y_arcsec, com, x_arr_new, y_arr_new, threshed_img_4th]


def refraction_fit_param(
    fname,
    thresh_freq=45e6,
    overbright=2.0e6,
    min_freqfrac=0.3,
    return_full_data=False,
    return_record=False,
    convex_hull=False,
    background_factor=1 / 8,
    data=None,
    meta=None,
):
    """
    Fit refraction parameters for a multi-frequency FITS cube:

    ``x = px[0] * 1/freq**2 + px[1]``
    ``y = py[0] * 1/freq**2 + py[1]``

    Parameters
    ----------
    overbright : float or None
        Channels with peak Tb above this value are excluded from the fit (to drop
        flaring intervals). Default ``2e6`` K. Use ``None`` to disable this filter.
        If the fit returns NaN, peaks may exceed the default—raise ``overbright`` or
        pass ``return_full_data=True`` to inspect per-channel peaks.
    min_freqfrac : float
        Minimum fraction of channels above ``thresh_freq`` required for fitting.
        The absolute minimum number of channels is 3 (5 when more than 20 channels
        are above ``thresh_freq``).
    """
    if overbright is None:
        overbright = np.inf

    if data is None or meta is None:
        meta, data = ndfits.read(fname)

    freqs_arr = meta["ref_cfreqs"]

    com_x_arr = []
    com_y_arr = []
    peak_values_tmp = []
    area_collect_tmp = []
    for idx_img in range(freqs_arr.shape[0]):
        thresh = thresh_func(freqs_arr[idx_img]) * background_factor
        data_this = np.squeeze(data[0, idx_img, :, :])
        (
            com_x_arcsec,
            com_y_arcsec,
            com,
            x_arr_new,
            y_arr_new,
            threshed_img_4th,
        ) = find_center_of_thresh(
            data_this, thresh, meta, idx_img, convex_hull=convex_hull
        )
        peak_values_tmp.append(np.nanmax(data_this))
        com_x_arr.append(com_x_arcsec)
        com_y_arr.append(com_y_arcsec)
        area_collect_tmp.append(np.sum(threshed_img_4th > thresh))

    com_x_tmp = np.array(com_x_arr)
    com_y_tmp = np.array(com_y_arr)
    peak_values_tmp = np.array(peak_values_tmp)

    idx_for_gt_freqthresh = np.where(freqs_arr > thresh_freq)

    freq_for_fit = freqs_arr[idx_for_gt_freqthresh]
    com_x_for_fit = com_x_tmp[idx_for_gt_freqthresh]
    com_y_for_fit = com_y_tmp[idx_for_gt_freqthresh]
    peak_values_for_fit = peak_values_tmp[idx_for_gt_freqthresh]

    idx_not_too_bright = np.where(peak_values_for_fit < overbright)
    freq_for_fit_v1 = freq_for_fit[idx_not_too_bright]
    com_x_for_fit_v1 = com_x_for_fit[idx_not_too_bright]
    com_y_for_fit_v1 = com_y_for_fit[idx_not_too_bright]

    idx_nan = np.where(np.isnan(com_x_for_fit_v1) | np.isnan(com_y_for_fit_v1))
    freq_for_fit_v2 = np.delete(freq_for_fit_v1, idx_nan)
    com_x_for_fit_v2 = np.delete(com_x_for_fit_v1, idx_nan)
    com_y_for_fit_v2 = np.delete(com_y_for_fit_v1, idx_nan)

    n_above = len(idx_for_gt_freqthresh[0])
    min_points = max(int(n_above * min_freqfrac), 3)
    if n_above >= 20:
        min_points = max(min_points, 5)

    if freq_for_fit_v2.size >= min_points:
        px = np.polyfit(1 / freq_for_fit_v2 ** 2, com_x_for_fit_v2, 1)
        py = np.polyfit(1 / freq_for_fit_v2 ** 2, com_y_for_fit_v2, 1)
    else:
        px = [np.nan, np.nan]
        py = [np.nan, np.nan]
        warnings.warn(
            "refraction_fit_param: fit failed — {n} channel(s) passed filters "
            "(need > {minp}); {nabove} above {thresh_mhz:.0f} MHz, "
            "overbright={ob:.3g} K. Try overbright=None or a higher overbright, "
            "or return_full_data=True to inspect peak_values.".format(
                n=freq_for_fit_v2.size,
                minp=min_points,
                nabove=n_above,
                thresh_mhz=thresh_freq / 1e6,
                ob=overbright if np.isfinite(overbright) else float("inf"),
            ),
            stacklevel=2,
        )

    reftime = meta["header"]["date-obs"][:19]

    if return_full_data:
        return {
            "Time": reftime,
            "px0": px[0],
            "px1": px[1],
            "py0": py[0],
            "py1": py[1],
            "freqs": freqs_arr,
            "com_x": com_x_tmp,
            "com_y": com_y_tmp,
            "peak_values": peak_values_tmp,
            "area_collect": area_collect_tmp,
        }
    if return_record:
        return {
            "Time": reftime,
            "px0": px[0],
            "px1": px[1],
            "py0": py[0],
            "py1": py[1],
        }
    return px, py


def save_refraction_fit_param(fname_in, fname_out, px, py):
    """
    Copy FITS to ``fname_out`` and store refraction fit parameters (no data shift).
    """
    copyfile(fname_in, fname_out)
    meta, _data = ndfits.read(fname_in)
    freqs_arr = meta["ref_cfreqs"]
    com_x_fitted = px[0] * 1 / freqs_arr ** 2 + px[1]
    com_y_fitted = py[0] * 1 / freqs_arr ** 2 + py[1]

    col_add1 = fits.Column(name="refra_shift_x", format="E", array=com_x_fitted)
    col_add2 = fits.Column(name="refra_shift_y", format="E", array=com_y_fitted)
    new_table_columns = [col_add1, col_add2]

    new_header_entries = {
        "RFRPX0": px[0],
        "RFRPX1": px[1],
        "RFRPY0": py[0],
        "RFRPY1": py[1],
        "RFRCOR": False,
        "RFRVER": "1.0",
        "LVLNUM": "1.0",
        "HISTORY": (
            "Refraction corrections V1.0 calculated and saved to the header on "
            "{0:s}. No corrections applied to the data.".format(Time.now().isot[:19])
        ),
    }

    success = ndfits.update(
        fname_out,
        new_columns=new_table_columns,
        new_header_entries=new_header_entries,
    )
    if success:
        print("FITS file successfully updated.")
    else:
        print("Failed to update FITS file.")
    return True


def apply_refra_coeff(fname_in, px, py, fname_out=None, verbose=False):
    """
    Apply refraction coefficients to a level 1.0 FITS file; write level 1.5 FITS.
    """
    if fname_out is None:
        fname_out = "./" + os.path.basename(fname_in).replace("lev1", "lev1.5")
    copyfile(fname_in, fname_out)
    meta, data = ndfits.read(fname_in)
    freqs_arr = meta["ref_cfreqs"]
    com_x_fitted = px[0] * 1 / freqs_arr ** 2 + px[1]
    com_y_fitted = py[0] * 1 / freqs_arr ** 2 + py[1]

    datasize = data.shape
    new_data = np.zeros(datasize)
    old_crval1 = meta["header"]["CRVAL1"]
    old_crval2 = meta["header"]["CRVAL2"]
    delta_x = meta["header"]["CDELT1"]
    delta_y = meta["header"]["CDELT2"]
    nx = meta["header"]["NAXIS1"]
    ny = meta["header"]["NAXIS2"]

    for pol in range(datasize[0]):
        for chn in range(datasize[1]):
            datatmp = data[pol, chn, :, :]
            shift_x_tmp, shift_y_tmp = (
                com_x_fitted[chn] - old_crval1,
                com_y_fitted[chn] - old_crval2,
            )
            datatmp = np.roll(
                datatmp, -int(np.round(shift_y_tmp / delta_y)), axis=0
            )
            datatmp = np.roll(
                datatmp, -int(np.round(shift_x_tmp / delta_x)), axis=1
            )
            new_data[pol, chn, :, :] = datatmp

    new_header_entry = {
        "CRVAL1": 0,
        "CRVAL2": 0,
        "CRPIX1": nx // 2,
        "CRPIX2": ny // 2,
        "RFRPX0": px[0],
        "RFRPX1": px[1],
        "RFRPY0": py[0],
        "RFRPY1": py[1],
        "RFRCOR": True,
        "RFRVER": "1.0",
        "LVLNUM": "1.5",
        "HISTORY": "Refraction corrections V1.0 applied to data array on {0:s}".format(
            Time.now().isot[:19]
        ),
    }

    success = ndfits.update(
        fname_out, new_data=new_data, new_header_entries=new_header_entry
    )
    if success:
        if verbose:
            print("FITS file successfully updated.")
        return fname_out
    print("Failed to update FITS file.")
    return False


def apply_refra_record(
    fname_in, refra_record, fname_out=None, interp="linear", max_dt=600.0
):
    """
    Apply refraction correction from one record or interpolate over a time series.
    """
    from scipy import interpolate

    import pandas as pd

    if fname_out is None:
        fname_out = "./" + os.path.basename(fname_in).replace("lev1", "lev1.5")
    if isinstance(refra_record, dict):
        rec = refra_record
        if "px0" in rec and "px1" in rec and "py0" in rec and "py1" in rec:
            px = [rec["px0"], rec["px1"]]
            py = [rec["py0"], rec["py1"]]
            return apply_refra_coeff(fname_in, px, py, fname_out=fname_out)
        print("The input refraction record does not have all the required keys. Abort.")
        return False
    if isinstance(refra_record, list):
        rec_df = pd.DataFrame(refra_record)
    elif isinstance(refra_record, pd.DataFrame):
        rec_df = refra_record
    else:
        print(
            "Input refra_record needs to be a dictionary, list of dictionaries, "
            "or a pandas.DataFrame. Abort."
        )
        return False

    meta, _data = ndfits.read(fname_in)
    time_in = Time(meta["header"]["date-obs"]).mjd
    times = Time(list(rec_df["Time"].values)).mjd
    dt_minute = np.min(np.abs(times - time_in) * 24.0 * 60.0 * 60.0)
    if dt_minute < max_dt:
        px0s = rec_df["px0"].values
        px1s = rec_df["px1"].values
        py0s = rec_df["py0"].values
        py1s = rec_df["py1"].values
        fx0 = interpolate.interp1d(
            times, px0s, kind=interp, fill_value="extrapolate"
        )
        fx1 = interpolate.interp1d(
            times, px1s, kind=interp, fill_value="extrapolate"
        )
        fy0 = interpolate.interp1d(
            times, py0s, kind=interp, fill_value="extrapolate"
        )
        fy1 = interpolate.interp1d(
            times, py1s, kind=interp, fill_value="extrapolate"
        )
        px0 = fx0(time_in).item()
        px1 = fx1(time_in).item()
        py0 = fy0(time_in).item()
        py1 = fy1(time_in).item()
        return apply_refra_coeff(
            fname_in, [px0, px1], [py0, py1], fname_out=fname_out
        )
    print(
        "Time difference between the input fits file and the record is greater "
        "than the set maximum {0:.1f} s. Abort.".format(max_dt)
    )
    return False
