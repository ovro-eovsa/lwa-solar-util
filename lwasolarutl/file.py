"""FITS ↔ HDF5 compression, recovery, and consistency checks."""

from __future__ import annotations

import logging
import os

import h5py
import numpy as np
from astropy import units as u
from astropy.io import fits
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


def compress_fits_to_h5(
    fits_file,
    hdf5_file=None,
    beam_ratio=3.0,
    smaller_than_src=True,
    theoretical_beam_thresh=True,
    longest_baseline=3000,
    purge_corrupted=False,
    purge_thresh=1.5,
):
    """
    Compress an OVRO-LWA multi-channel FITS file to HDF5.

    Parameters
    ----------
    fits_file : str
        Input FITS path.
    hdf5_file : str, optional
        Output HDF5 path. Defaults to ``{basename}.hdf`` in the current directory.
    beam_ratio : float
        Beam size divisor used when choosing the downsample factor.
    smaller_than_src : bool
        If True, do not upscale pixels (downsize ratio is at least 1).
    theoretical_beam_thresh : bool
        If True, enforce a theoretical beam floor per channel.
    longest_baseline : float
        Longest baseline in metres for theoretical beam estimate.
    purge_corrupted : bool
        If True, skip channels where Stokes I looks corrupted.
    purge_thresh : float
        Corruption threshold: ``-min * purge_thresh > max`` on Stokes I.
    """
    if hdf5_file is None:
        hdf5_file = "./" + os.path.basename(fits_file).replace(".fits", ".hdf")

    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        header = hdul[0].header

        ch_vals = []
        for ch_val in hdul[1].data.dtype.names:
            ch_vals.append(hdul[1].data[ch_val])
        ch_vals = np.array(ch_vals)

        freqs = hdul[1].data["cfreqs"]
        head_tb = hdul[1].header
        bmin = head_tb["TTYPE4"]

        thresh_arr = np.copy(hdul[1].data[bmin] * 3600)
        if theoretical_beam_thresh:
            beam_size_thresh = (
                (3e8 / freqs) / longest_baseline / np.pi * 180 * 3600
            )
            for i in range(len(thresh_arr)):
                thresh_arr[i] = max(thresh_arr[i], beam_size_thresh[i])
                if not (thresh_arr[i] > 0):
                    thresh_arr[i] = beam_size_thresh[i]

        unit_angle = u.Unit(hdul[0].header["CUNIT2"])
        downsize_ratio = (thresh_arr) / beam_ratio / (
            hdul[0].header["CDELT2"] * unit_angle.to(u.arcsec)
        )

        if smaller_than_src:
            downsize_ratio[downsize_ratio < 1] = 1

        count_avail = 0
        with h5py.File(hdf5_file, "w") as f:
            for pol in range(data.shape[0]):
                for ch_idx in range(len(downsize_ratio)):
                    if (
                        purge_corrupted
                        and (
                            -np.min(data[0, ch_idx, :, :]) * purge_thresh
                            > np.max(data[0, ch_idx, :, :])
                        )
                    ):
                        logger.warning(
                            "Pol %s Ch %s is corrupted, skipped", pol, ch_idx
                        )
                        downsized_data = np.zeros((1, 1))
                    else:
                        count_avail += 1
                        downsized_data = zoom(
                            data[pol, ch_idx, :, :],
                            1 / downsize_ratio[ch_idx],
                            order=3,
                            prefilter=False,
                        )
                    f.create_dataset(
                        "FITS_pol"
                        + str(pol)
                        + "ch"
                        + str(ch_idx).rjust(4, "0"),
                        data=downsized_data,
                        compression="gzip",
                        compression_opts=9,
                    )

            dset = f.create_dataset("ch_vals", data=ch_vals)
            dset.attrs["arr_name"] = hdul[1].data.dtype.names
            dset.attrs["original_shape"] = data.shape
            for key, value in header.items():
                dset.attrs[key] = value

    if count_avail == 0:
        logger.warning("No available data in the fits file %s", fits_file)
        if os.path.isfile(hdf5_file):
            os.remove(hdf5_file)


def recover_fits_from_h5(
    hdf5_file,
    fits_out=None,
    return_data=False,
    return_meta_only=False,
):
    """
    Recover a FITS file from a compressed HDF5 archive.

    Parameters
    ----------
    hdf5_file : str
        Input HDF5 path.
    fits_out : str, optional
        Output FITS path when writing to disk.
    return_data : bool
        If True, return ``(meta, data)`` without writing a file.
    return_meta_only : bool
        If True, return metadata only.

    Returns
    -------
    dict, tuple, or None
        Depends on ``return_meta_only`` / ``return_data``. Otherwise writes FITS
        and returns None.
    """
    if fits_out is None and not return_data:
        fits_out = "./" + os.path.basename(hdf5_file).replace(".hdf", ".fits")

    with h5py.File(hdf5_file, "r") as f:
        header = dict(f["ch_vals"].attrs)
        datashape = header["original_shape"]
        header.pop("arr_name", None)
        header.pop("original_shape", None)
        header = fits.Header(header)
        ch_vals = {
            name: f["ch_vals"][i]
            for i, name in enumerate(f["ch_vals"].attrs["arr_name"])
        }
        attaching_columns = [
            fits.Column(name=key, format="E", array=ch_vals[key])
            for key in ch_vals
        ]
        meta = {"header": header, **{col.name: col.array for col in attaching_columns}}

        if return_meta_only:
            return meta

        recover_data = np.zeros(datashape)
        for pol in range(datashape[0]):
            for ch_idx in range(len(meta["cfreqs"])):
                key = f"FITS_pol{pol}ch{str(ch_idx).rjust(4, '0')}"
                tmp_small = f[key][:]
                if tmp_small.shape[0] == 1:
                    recover_data[pol, ch_idx, :, :] = tmp_small[0, 0]
                else:
                    recover_data[pol, ch_idx, :, :] = zoom(
                        tmp_small,
                        datashape[-1] / tmp_small.shape[-1],
                        order=3,
                        prefilter=False,
                    )

        if return_data:
            return meta, recover_data

        hdu_list = fits.HDUList(
            [
                fits.PrimaryHDU(recover_data, header),
                fits.BinTableHDU.from_columns(attaching_columns),
            ]
        )
        hdu_list.writeto(fits_out, overwrite=True)


def check_h5_fits_consistency(
    fits_file,
    hdf5_file=None,
    ignore_corrupted=False,
    work_dir="./",
    tolerance=1e-3,
    ignore_ratio=2,
    auto_tol=True,
):
    """
    Compare an original FITS file with data recovered from HDF5.

    Returns
    -------
    int
        0 if consistent; positive codes flag header/data mismatches; -1 on error.
    """
    hdf5_file = hdf5_file if hdf5_file is not None else fits_file.replace(
        ".fits", ".hdf"
    )

    pass_check = 0
    tmp_path = os.path.join(work_dir, "tmp.fits")
    try:
        recover_fits_from_h5(hdf5_file, fits_out=tmp_path)
        with fits.open(tmp_path) as hdu_tmp, fits.open(fits_file) as hdu:
            header_tmp = hdu_tmp[0].header
            header = hdu[0].header
            for key in header.keys():
                if key not in header_tmp.keys():
                    logger.warning("Key %s not in the recovered fits header", key)
                    pass_check = 1
                elif header[key] != header_tmp[key]:
                    logger.warning(
                        "Key %s not consistent in the recovered fits header", key
                    )
                    pass_check = 2

            data_tmp = hdu_tmp[0].data
            data = hdu[0].data
            checked_items = 0
            for pol in range(data.shape[0]):
                for ch_idx in range(data.shape[1]):
                    if ignore_corrupted and (
                        -np.min(data[0, ch_idx, :, :]) * ignore_ratio
                        > np.max(data[0, ch_idx, :, :])
                    ):
                        continue
                    checked_items += 1
                    ch_tol = tolerance
                    if auto_tol:
                        ch_tol = -np.min(data[0, ch_idx, :, :]) / np.max(
                            np.abs(data[0, ch_idx, :, :])
                        ) / 3
                    rel_diff = np.mean(
                        np.abs(data[pol, ch_idx, :, :] - data_tmp[pol, ch_idx, :, :])
                    ) / np.max(np.abs(data[pol, ch_idx, :, :]))
                    if rel_diff > ch_tol:
                        logger.warning(
                            "Pol %s Ch %s not consistent. Difference: %s for Tol: %s",
                            pol,
                            ch_idx,
                            rel_diff,
                            ch_tol,
                        )
                        pass_check = 4
                        break
            logger.info("Checked %s items in the fits file", checked_items)
    except Exception:
        pass_check = -1
        logger.exception(
            "Error checking consistency between %s and %s", fits_file, hdf5_file
        )
    finally:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)

    return pass_check
