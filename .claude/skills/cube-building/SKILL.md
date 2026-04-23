---
name: cube-building
description: >
  This skill should be used when the user asks to "simulate a cube", "build a
  datacube", "populate a cube", "run simulate_cube", "debug cube generation",
  "fix OOM", "understand zcalc_truncate", "precompute sky grids", "convolve
  a cube", "rebin spatially", or "set up instrument kernels".
---

# Cube Generation and Simulation

## Cube Generation Flow

```
ModelSet.simulate_cube()
    └─> _compute_3d_flux_vel_sigma()   [per component]
        └─> populate_cube_jax()         [full grid, GPU]
        └─> populate_cube_jax_ais()     [sparse, active pixels only]
        └─> populate_cube_active()      [active-only, CPU-friendly]
    └─> np.asarray()                    [convert JAX array to numpy]
```

## Three Populate Modes

### `populate_cube_jax` — Full grid

JIT-compiled, `jax.vmap` over spectral channels.  Evaluates the flux-weighted
Gaussian at every (z, y, x) position and sums along the line-of-sight.

```python
from dysmalpy.models.cube_processing import populate_cube_jax
cube = populate_cube_jax(flux, vel, sigma, vspec)  # (nspec, ny, nx)
```

**Memory:** Requires full 3D coordinate grids on GPU.  For a 201^3 grid in
float64, this is ~650 MB per grid array.  Multiple intermediates can push
GPU memory past 24 GB for large FOV or high oversample.

### `populate_cube_jax_ais` — Sparse (Active Index Set)

Uses `.at[].add()` scatter-add to only update spatial pixels where flux is
non-negligible.  The `ai` (active index) array specifies which (y, x) pixels
to propagate.

**Memory:** Same as full grid for the input arrays, but the scatter-add is
faster for sparse sources.

### `populate_cube_active` — Active-only (default for large cubes)

1-D propagation along the line-of-sight for active spatial pixels only.
Intermediate arrays stay on CPU (numpy), and only active z-slices are
propagated through JAX.

**Memory:** Peak GPU memory drops from ~18 GB to < 100 MB for 603^3 grids.

## `zcalc_truncate=True` (Default)

When enabled (default), only z-slices with non-negligible flux are evaluated.
This is controlled by `obs.mod_options.zcalc_truncate`.

- **Active path** uses `populate_cube_active` with numpy coordinate transforms
  (`_numpy_coord_transform`) instead of `Geometry.evaluate()` (which requires
  JAX tracers).
- **Sparse index array `ai`** is precomputed by `_make_cube_ai()` and determines
  which (y, x) pixels have active z-slices.

For JAX loss functions, `ai` is pre-computed with concrete parameter values and
passed into the JIT closure via `ai_precomputed`.

## `sky_grids_precomputed` for JAXNS Geometry Tracing

When geometry parameters (inc, pa, xshift, yshift, vel_shift) are traced by JAXNS:

1. Sky coordinate grids are pre-computed with concrete geometry values
2. `sky_grids_precomputed` dict is passed to `simulate_cube()`
3. `Geometry.evaluate()` is bypassed — pre-computed grids are used directly

This allows the full pipeline to be JIT-compiled while fitting geometry.

## `_numpy_coord_transform` vs `Geometry.evaluate()`

| Function | Backend | Use case |
|----------|---------|----------|
| `_numpy_coord_transform` | numpy | Active-only path, CPU-friendly, concrete values |
| `Geometry.evaluate()` | JAX | Full grid path, GPU, supports traced parameters |

## `_make_cube_ai()` and Sparse Index Array

`_make_cube_ai()` computes the active index arrays:
- `ai`: flat indices of active (y, x) pixels
- `ai_sky`: sky-frame active indices

Used by `populate_cube_jax_ais` and `populate_cube_active`.

## Observation Pipeline

The full observation simulation chain:

```
simulate_cube()     → intrinsic cube (nspec, ny, nx)
    ↓
_rebin_spatial()    → rebinned to instrument pixel scale
    ↓
convolve_cube_jax() → beam + LSF convolution
    ↓
_crop_cube()        → crop to observation FOV
    ↓
moment extraction   → mom0, mom1, mom2 maps
```

### JAX FFT Convolution

Located in `dysmalpy/convolution.py`:

- `_fft_convolve_3d(cube, kernel)` — JAX-traceable, replicates
  `scipy.signal.fftconvolve(mode='same')` to `rtol=1e-10`
- `convolve_cube_jax(cube, beam_kernel, lsf_kernel)` — sequential beam + LSF
- `get_jax_kernels(instrument)` — extracts pre-computed kernels from `Instrument`

### Spatial Rebin

`_rebin_spatial(cube, new_ny, new_nx)` — rebin via reshape+sum, matches
`dysmalpy.utils.rebin`.

## Instrument Setup

```python
from dysmalpy import Instrument
inst = Instrument(
    fov=(101, 101),           # pixels
    pixscale=0.15*u.arcsec,
    beam=[0.5]*u.arcsec,      # FWHM
    lsf=10.0*u.km/u.s,       # FWHM
    spectral_resolution=10.0*u.km/u.s,
)
inst.set_beam_kernel()        # pre-compute 2D Gaussian kernel
inst.set_lsf_kernel()         # pre-compute 1D Gaussian kernel
```

Kernels are numpy arrays stored on the instrument.  `get_jax_kernels()` converts
them to JAX arrays for use inside JIT-compiled loss functions.

## Memory Budget

| Path | Grid size | Peak GPU memory | Use case |
|------|-----------|-----------------|----------|
| Full (`populate_cube_jax`) | 201^3 | ~2 GB | Small cubes, GPU fitting |
| Sparse (`populate_cube_jax_ais`) | 201^3 | ~2 GB | Sparse sources |
| Active-only (`populate_cube_active`) | 603^3 | < 100 MB | Large cubes, default |
| With `oversample=3` full | 603^3 | ~18 GB | **OOM risk** — use active-only |

## Key Files

| File | Description |
|------|-------------|
| `dysmalpy/models/cube_processing.py` | `populate_cube_jax`, `populate_cube_active`, `_make_cube_ai`, `_numpy_coord_transform` |
| `dysmalpy/models/model_set.py` | `ModelSet.simulate_cube()`, active-only dispatch, `sky_grids_precomputed` |
| `dysmalpy/convolution.py` | `_fft_convolve_3d`, `convolve_cube_jax`, `_rebin_spatial`, `get_jax_kernels` |
| `dysmalpy/observation.py` | `Observation` — full simulate/rebin/convolve/crop pipeline |
| `dysmalpy/instrument.py` | `Instrument` — kernel setup, FOV, spectral config |
| `dysmalpy/fitting/jax_loss.py` | Loss closures that include the full cube pipeline |
