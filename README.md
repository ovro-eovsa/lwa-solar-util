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

## Example notebook

See [`notebook/example.ipynb`](notebook/example.ipynb). It uses the demo HDF in [`demofile/`](demofile/):

1. Recover HDF5 → FITS  
2. Compress FITS → HDF5  
3. Consistency check  
4. 12-panel plot  

```bash
conda activate <your-env>
pip install -e ".[dev]"
jupyter notebook notebook/example.ipynb
```

## Package layout

| Module | Purpose |
|--------|---------|
| `lwasolarutl.file` | `compress_fits_to_h5`, `recover_fits_from_h5`, `check_h5_fits_consistency` |
| `lwasolarutl.ndfits` | Read/write/wrap multi-dimensional solar FITS ([lwasolarproc](https://github.com/peijin94/lwasolarproc)) |
| `lwasolarutl.plot_map` | `Sunmap` helper for heliocentric `imshow` / limb overlay |
| `lwasolarutl.visualization` | `slow_pipeline_default_plot`, `make_allsky_image_plots` |

Top-level shortcuts: `lsu.compress_fits_to_h5`, `lsu.recover_fits_from_h5`, `lsu.check_h5_fits_consistency`.

## Tests

```bash
pytest tests/
```

## Provenance

- HDF5 compression/recovery: [ovro-lwa-solar `utils.py`](https://github.com/ovro-eovsa/ovro-lwa-solar/blob/main/ovrolwasolar/utils.py)
- `ndfits.py`: [lwasolarproc](https://github.com/peijin94/lwasolarproc/blob/main/lwasolarproc/ndfits.py)
- 12-panel plot: [ovro-lwa-solar `visualization.py`](https://github.com/ovro-eovsa/ovro-lwa-solar/blob/b55f56d5f63f37e8168d374697af0ba3097b0dc6/ovrolwasolar/visualization.py)

## License

MIT
