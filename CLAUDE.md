# dysmalpy — Galaxy Dynamical Modeling with JAX Acceleration

## Overview

dysmalpy builds and fits multi-component galaxy dynamical models to observed
kinematic data (rotation curves, velocity fields, IFU datacubes).  It supports
1D, 2D, and 3D fitting with multiple backends (MPFIT, MCMC, nested sampling,
JAX-Adam, JAXNS).  The `dev_jax` branch replaces the original Cython backend
with JAX-accelerated computation for GPU speedup.

## Branch

- **`dev_jax`** — active development branch; JAX-accelerated, no Cython dependency
- **`main`** — original release; Cython `cutils.pyx` backend, astropy.modeling

The two branches are **not pickle-compatible**.  Use JSON params for cross-branch
data transfer (see `dev/problem.md` §5).

## Module Map

| Module | Purpose |
|--------|---------|
| `dysmalpy/__init__.py` | Version (`2.0.0`), sets `JAX_ENABLE_X64=1` before any JAX import |
| `dysmalpy/parameters.py` | `DysmalParameter` descriptor with prior/constraint support |
| `dysmalpy/models/` | Galaxy model components (baryons, halos, geometry, kinematics) |
| `dysmalpy/models/base.py` | `_DysmalModel` base class, `_DysmalModelMeta` metaclass, physical constants |
| `dysmalpy/models/model_set.py` | `ModelSet` — orchestrates multi-component cube simulation |
| `dysmalpy/models/baryons.py` | Baryonic profiles (Sersic, DiskBulge, ExpDisk, BlackHole, GaussianRing) |
| `dysmalpy/models/halos.py` | Dark matter halos (NFW, TwoPowerHalo, Burkert, Einasto, DekelZhao) |
| `dysmalpy/models/geometry.py` | Sky-plane geometric transformations |
| `dysmalpy/models/kinematic_options.py` | Kinematic configuration (dispersion, asymmetric drift, pressure support) |
| `dysmalpy/models/cube_processing.py` | JAX cube population (`populate_cube_jax`, `populate_cube_active`) |
| `dysmalpy/models/higher_order_kinematics.py` | Outflows, radial flows, bar flows, spiral arms |
| `dysmalpy/models/light_distributions.py` | Light profile models |
| `dysmalpy/models/dispersion_profiles.py` | Velocity dispersion profiles |
| `dysmalpy/models/zheight.py` | Vertical (z-height) density profiles |
| `dysmalpy/models/extinction.py` | Dust extinction models |
| `dysmalpy/models/dimming.py` | Cosmological surface brightness dimming |
| `dysmalpy/convolution.py` | JAX FFT convolution (`_fft_convolve_3d`) and spatial rebin |
| `dysmalpy/observation.py` | `Observation` class — simulate/rebin/convolve/crop pipeline |
| `dysmalpy/instrument.py` | `Instrument` class — beam kernel, LSF kernel |
| `dysmalpy/galaxy.py` | `Galaxy` class — top-level container for a galaxy |
| `dysmalpy/fitting/` | Fitter classes and loss functions |
| `dysmalpy/fitting/base.py` | Base fitter, chi-squared metrics |
| `dysmalpy/fitting/jax_loss.py` | `make_jax_loss_function()`, `make_jax_log_prob_function()` |
| `dysmalpy/fitting/jax_optimize.py` | `JAXAdamFitter` (Adam optimizer) |
| `dysmalpy/fitting/jaxns.py` | `JAXNSFitter` (nested sampling via jaxns) |
| `dysmalpy/fitting/mcmc.py` | `MCMCFitter` (emcee-based) |
| `dysmalpy/fitting/mpfit.py` | `MPFITFitter` (Levenberg-Marquardt) |
| `dysmalpy/fitting/nested_sampling.py` | `NestedFitter` (Multinest-based) |
| `dysmalpy/fitting/utils.py` | Fitting utilities |
| `dysmalpy/fitting_wrappers/` | High-level fitting API |
| `dysmalpy/fitting_wrappers/dysmalpy_fit_single.py` | Main entry point: `dysmalpy_fit_single()` |
| `dysmalpy/fitting_wrappers/dysmalpy_fit_single_{1D,2D,3D}.py` | Dimension-specific fitting |
| `dysmalpy/fitting_wrappers/setup_gal_models.py` | Galaxy model setup utilities |
| `dysmalpy/fitting_wrappers/tied_functions.py` | Tied parameter functions |
| `dysmalpy/special/` | JAX-traceable special functions |
| `dysmalpy/special/gammaincinv.py` | Inverse incomplete gamma (Newton-Raphson + `jax.lax.scan`) |
| `dysmalpy/special/hyp2f1.py` | Gauss hypergeometric 2F1 (power series + LFT) |
| `dysmalpy/special/bessel.py` | Modified Bessel K0, K1 (`jax.pure_callback` wrapping scipy) |

## Environment

| Item | Value |
|------|-------|
| Conda env | `dysmalpy-jax` |
| GPU | NVIDIA 4090 (CUDA 12) |
| JAX backend | `JAX_PLATFORMS=cpu` for testing, GPU default for fitting |
| Float64 | `JAX_ENABLE_X64=1` set in `__init__.py` (must be before any JAX import) |

**Known working versions (as of 2026-04):**

```
jax==0.4.38, jaxlib==0.4.38, jax-cuda12-plugin==0.4.38
jaxns==2.4.13, tensorflow-probability==0.25.0
numpy>=2.0, astropy>=6.0
```

JAX, jaxlib, and downstream packages (jaxns, tensorflow-probability) must be
pinned together.  A version mismatch causes hard-to-diagnose import errors or
segfaults.

## Key Classes

| Class | Module | Description |
|-------|--------|-------------|
| `Galaxy` | `galaxy.py` | Top-level container: components + observations |
| `ModelSet` | `models/model_set.py` | Multi-component model: `simulate_cube()` |
| `Observation` | `observation.py` | Data + instrument: simulate → rebin → convolve → crop |
| `Instrument` | `instrument.py` | Beam/LSF kernels, FOV, spectral settings |
| `_DysmalModel` | `models/base.py` | Base class for all model components |
| `MassModel` | `models/base.py` | Mass profile base (halos, baryons) |
| `LightModel` | `models/base.py` | Light profile base |
| `Geometry` | `models/geometry.py` | Sky-plane transform (inc, PA, x/y shift) |
| `KinematicOptions` | `models/kinematic_options.py` | Dispersion, pressure support, kinematic config |
| `DysmalParameter` | `parameters.py` | Parameter descriptor with constraints |

## Critical Gotchas

See `dev/problem.md` for the full catalogue.  The top 5:

1. **JAX defaults to float32** — `JAX_ENABLE_X64=1` is set in `__init__.py` before
   any JAX import.  Do not use `jax.config.update()` — by the time `import jax`
   has run, the dtype is locked.  Verify with
   `JAX_PLATFORMS=cpu python -c "import jax; print(jax.config.read('jax_enable_x64'))"`.

2. **DysmalParameter descriptor pollution** — `__get__` returns the class-level
   descriptor.  Setting `.tied`/`.fixed`/`.prior` on an instance pollutes the
   class for all future instances.  Always use `comp._get_param(name)` to read
   constraint state.  The `__init__` anti-pollution reset restores originals via
   `copy.deepcopy`.

3. **`jnp.pad` incompatibility** — `jnp.pad` with `mode='constant'` has a `copy`
   keyword argument that causes `TypeError` in certain JAX/NumPy combinations.
   Use `jax.lax.pad` instead.

4. **Multiprocessing requires `forkserver`** — JAX initializes thread pools at
   import time.  Python's default `fork` copies this state, causing deadlocks.
   Use `multiprocess.get_context('forkserver').Pool(nCPUs)`.

5. **Pickle cross-branch incompatibility** — Pickles saved on `dev_jax` cannot be
   loaded on `main` and vice versa.  Use JSON params for cross-branch transfer.

Additional important gotchas:

- **`np.NaN` removed in NumPy 2.x** — use `np.nan` (19 files affected).
- **GPU OOM for large cubes** — `zcalc_truncate=True` (default) uses active-only
  evaluation, keeping intermediate arrays off GPU.  Peak GPU memory: ~18 GB → < 100 MB.
- **`SpectralCube.moment()` unreliable for CASA cubes** — compute moments manually
  with `numpy.nansum`.
- **MCMC 0% acceptance** — usually caused by `_model` back-references being `None`
  after pickle.  `ModelSet.__setstate__` rebinds them.

## Development Workflow

The `dev/` folder contains development notes.  Update them during each development session:

- **`dev/problem.md`** — catalogue of known problems and pitfalls.  Read before debugging.
- **`dev/develop_log.md`** — log of all changes, phase-by-phase.
- **`dev/prompt.md`** — development prompts (if present).

## Testing

Tests require the `dysmalpy-jax` conda environment:

```bash
conda activate dysmalpy-jax
JAX_PLATFORMS=cpu pytest tests/ -v
```

The `JAX_PLATFORMS=cpu` flag forces CPU execution for reproducibility.
`tests/conftest.py` also sets `jax_enable_x64 = True` as a safety net.

Test files:
- `tests/test_jax.py` — 74 JAX-specific unit tests (special functions, cube population,
  model computations, convolution, full pipeline integration)
- `tests/test_models.py` — 27 existing model tests (numpy 2.x compat fixes)

All 101 tests pass on the `dev_jax` branch.  JAX (dev_jax) and Cython (main) produce
numerically identical cubes (max diff = 8.6e-16).
