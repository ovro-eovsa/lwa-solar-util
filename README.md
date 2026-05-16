# lwasolarutl

Python utilities for OVRO-LWA solar radio imaging: FITS ↔ HDF5 compression, multi-dimensional FITS I/O, and pipeline visualization.

Import as:

```python
import lwasolarutl as lsu
```

**No CASA or suncasa** — dependencies are NumPy, Astropy, SciPy, h5py, Matplotlib, and SunPy only.

## Install

From GitHub:

```bash
pip install git+https://github.com/ovro-eovsa/lwa-solar-util.git
```

For development:

```bash
git clone https://github.com/ovro-eovsa/lwa-solar-util.git
cd lwa-solar-util
pip install -e ".[dev]"
```

Requires Python ≥ 3.8.

## Quick start

### FITS ↔ HDF5

Compress a multi-channel solar FITS cube to HDF5, then recover it:

```python
import lwasolarutl as lsu

lsu.file.compress_fits_to_h5("image.fits", "image.hdf")
lsu.file.recover_fits_from_h5("image.hdf", "image_recovered.fits")

# 0 = consistent; >0 = mismatch; -1 = error
lsu.file.check_h5_fits_consistency("image.fits", "image.hdf")
```

### Read FITS cubes

```python
meta, data = lsu.ndfits.read("image.fits")
# data: (npol, nchan, ny, nx); meta includes ref_cfreqs, bmaj, bmin, bpa, header, ...
```

### Default 12-panel plot

OVRO-LWA slow-pipeline layout (3×4 grid, 12 frequencies in solar coordinates):

```python
fig, axes = lsu.visualization.slow_pipeline_default_plot(
    "image.fits",
    add_logo=False,
)
```

Requires a FITS file readable by `ndfits` (not raw `.hdf` — recover to FITS first if needed).

### Dynamic spectrum FITS (Stokes I / V)

Load and plot two panels (I and V/I) in the format used by [lwasolarview](https://github.com/peijin94/lwasolarview/blob/main/plot_spec_fits.py):

```python
s = lsu.spec.load_spectrum_fits("demofile/20260513.fits")
fig, axes = lsu.visualization.plot_spec(s)  # or plot_spec("demofile/20260513.fits")

# Plot a fraction of the time or frequency axis (each in [0, 1], low < high):
fig, axes = lsu.visualization.plot_spec(
    s,
    t_range_ratio=(0.2, 0.8),   # middle 60% of times
    f_range_ratio=(0.0, 1.0),   # full band (default)
)
```

Optional keyword arguments include `outpath` (save PNG), `vmax_I` / `pct_hi_I` for Stokes-I scaling, `vi_range` for the V/I panel, and colormap names `cmap_I` / `cmap_VI`.

## Example notebook

**`notebook/image_plot_hdf.ipynb`** (image cube / HDF): recover, compress, consistency check, 12-panel image plot.

**`notebook/spec_plot.ipynb`** (dynamic spectrum): load `demofile/20260513.fits`, plot Stokes I and V/I.

Demo data: [`demofile/`](demofile/).

```bash
conda activate <your-env>
pip install -e ".[dev]"
jupyter notebook notebook/image_plot_hdf.ipynb
# or
jupyter notebook notebook/spec_plot.ipynb
```

## Package layout

| Module | Purpose |
|--------|---------|
| `lwasolarutl.file` | `compress_fits_to_h5`, `recover_fits_from_h5`, `check_h5_fits_consistency` |
| `lwasolarutl.ndfits` | Read/write/wrap multi-dimensional solar FITS ([lwasolarproc](https://github.com/peijin94/lwasolarproc)) |
| `lwasolarutl.spec` | `load_spectrum_fits`, `vi_ratio` for LWA dynamic-spectrum FITS |
| `lwasolarutl.plot_map` | `Sunmap` helper for heliocentric `imshow` / limb overlay |
| `lwasolarutl.visualization` | `slow_pipeline_default_plot`, `plot_spec`, `make_allsky_image_plots` |

Top-level shortcuts: `lsu.compress_fits_to_h5`, `lsu.recover_fits_from_h5`, `lsu.check_h5_fits_consistency`.

## Tests

```bash
pytest tests/
```

## Provenance

- HDF5 compression/recovery: [ovro-lwa-solar `utils.py`](https://github.com/ovro-eovsa/ovro-lwa-solar/blob/main/ovrolwasolar/utils.py)
- `ndfits.py`: [lwasolarproc](https://github.com/peijin94/lwasolarproc/blob/main/lwasolarproc/ndfits.py)
- 12-panel plot: [ovro-lwa-solar `visualization.py`](https://github.com/ovro-eovsa/ovro-lwa-solar/blob/b55f56d5f63f37e8168d374697af0ba3097b0dc6/ovrolwasolar/visualization.py)
- Dynamic spectrum plot: [lwasolarview `plot_spec_fits.py`](https://github.com/peijin94/lwasolarview/blob/main/plot_spec_fits.py)

## License

MIT
