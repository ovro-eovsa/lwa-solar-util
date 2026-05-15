"""Heliocentric map plotting helpers (adapted from suncasa plot_mapX Sunmap)."""

from __future__ import annotations

import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from sunpy import map as smap


class Sunmap:
    """Wrapper around a sunpy Map for arcsec-coordinate imshow and limb overlay."""

    def __init__(self, sunmap):
        self.sunmap = sunmap
        try:
            top_right_coord = self.sunmap.top_right_coord
            bottom_left_coord = self.sunmap.bottom_left_coord
            self.xrange = (
                np.array(
                    [
                        bottom_left_coord.Tx.to(u.arcsec).value,
                        top_right_coord.Tx.to(u.arcsec).value,
                    ]
                )
                * u.arcsec
            )
            self.yrange = (
                np.array(
                    [
                        bottom_left_coord.Ty.to(u.arcsec).value,
                        top_right_coord.Ty.to(u.arcsec).value,
                    ]
                )
                * u.arcsec
            )
        except Exception:
            self.xrange = self.sunmap.xrange
            self.yrange = self.sunmap.yrange

    def get_map_extent(self, sunpymap=None, rot=0, rangereverse=False):
        if sunpymap is None:
            sunpymap = self.sunmap
        rot = rot % 360
        if rot == 90:
            extent = np.array(
                self.yrange.to(u.arcsec).value[::-1].tolist()
                + self.xrange.to(u.arcsec).value.tolist()
            )
            extent = extent - np.array(
                [sunpymap.scale.axis2.value] * 2 + [sunpymap.scale.axis1.value] * 2
            ) / 2.0
        elif rot == 180:
            extent = np.array(
                self.xrange.to(u.arcsec).value[::-1].tolist()
                + self.yrange.to(u.arcsec).value[::-1].tolist()
            )
            extent = extent - np.array(
                [sunpymap.scale.axis1.value] * 2 + [sunpymap.scale.axis2.value] * 2
            ) / 2.0
        elif rot == 270:
            extent = np.array(
                self.yrange.to(u.arcsec).value.tolist()
                + self.xrange.to(u.arcsec).value[::-1].tolist()
            )
            extent = extent - np.array(
                [sunpymap.scale.axis1.value] * 2 + [sunpymap.scale.axis2.value] * 2
            ) / 2.0
        else:
            extent = np.array(
                self.xrange.to(u.arcsec).value.tolist()
                + self.yrange.to(u.arcsec).value.tolist()
            )
            extent = extent - np.array(
                [sunpymap.scale.axis1.value] * 2 + [sunpymap.scale.axis2.value] * 2
            ) / 2.0
        if rangereverse:
            x0, x1, y0, y1 = extent
            extent = -np.array([x1, x0, y1, y0])
        return extent

    def imshow(self, axes=None, rot=0, rangereverse=False, **kwargs):
        sunpymap = self.sunmap
        if axes is None:
            axes = plt.subplot()
        rot = rot % 360
        if rot == 0:
            imdata = sunpymap.data
        elif rot == 90:
            imdata = sunpymap.data.transpose()[:, ::-1]
        elif rot == 180:
            imdata = sunpymap.data[::-1, ::-1]
        elif rot == 270:
            imdata = sunpymap.data.transpose()[::-1, :]
        else:
            warnings.warn("rot must be an integer multiple of 90; using rot=0")
            imdata = sunpymap.data
        extent = self.get_map_extent(rot=rot, rangereverse=rangereverse)
        im = axes.imshow(imdata, extent=extent, origin="lower", **kwargs)
        if rot == 0:
            axes.set_xlabel("Solar X [arcsec]")
            axes.set_ylabel("Solar Y [arcsec]")
        return im

    def draw_limb(self, axes=None, **kwargs):
        if "c" not in kwargs and "color" not in kwargs:
            kwargs["c"] = "w"
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs["ls"] = "solid"
        if axes is None:
            axes = plt.gca()
        rsun = self.sunmap.rsun_obs
        phi = np.linspace(-180, 180, num=181) * u.deg
        x = np.cos(phi) * rsun
        y = np.sin(phi) * rsun
        axes.set_autoscale_on(False)
        return axes.plot(x, y, **kwargs)
