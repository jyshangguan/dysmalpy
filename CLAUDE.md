# dysmalpy — Galaxy Dynamical Modeling with JAX Acceleration

## Overview

dysmalpy builds and fits multi-component galaxy dynamical models to observed
kinematic data (rotation curves, velocity fields, IFU datacubes).  It supports
1D, 2D, and 3D fitting with multiple backends (MPFIT, MCMC, nested sampling,
JAX-Adam, JAXNS).  The current `main` branch uses JAX-accelerated computation
(replacing the original Cython backend) for GPU speedup.

## Branch

- **`main`** — current development branch; JAX-accelerated, no Cython dependency
- **`dysmalpy_origin`** — original upstream release; Cython `cutils.pyx` backend,
  astropy.modeling-based parameters.  Use `git show dysmalpy_origin:<path>` to
  inspect the original implementation of any file for reference.
- **`dev_jax`** — historical branch (superseded by `main` after merge)

The two main branches are **not pickle-compatible**.  Use JSON params for
cross-branch data transfer (see `dev/problem.md` §5).

### Referencing the original code

To compare with the original Cython/astropy implementation, checkout or inspect
the `dysmalpy_origin` branch without switching:

```bash
# View a file from the original branch
git show dysmalpy_origin:dysmalpy/models/base.py

# Diff a file between branches
git diff main..dysmalpy_origin -- dysmalpy/models/halos.py

# Checkout the original branch temporarily
git stash && git checkout dysmalpy_origin && git checkout main && git stash pop
```

Key differences from the original (`dysmalpy_origin`) to current (`main`):
- `parameters.py`: standalone `DysmalParameter` descriptor (replaces astropy.modeling.Parameter)
- `models/base.py`: `_DysmalModelMeta` metaclass (replaces astropy.modeling.Model)
- `models/cube_processing.py`: JAX cube population (replaces Cython `cutils.pyx`)
- `convolution.py`: JAX FFT convolution (new, no original equivalent)
- `special/`: JAX-traceable special functions (replaces scipy.special calls)
- No `cutils.pyx` / `cutils.c` files on `main`

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
| Conda env | `alma` |
| GPU | NVIDIA 4090 (CUDA 12.4) |
| JAX backend | **GPU by default** for JAXNS/Adam fitting (use `JAX_PLATFORMS=cpu` only for MCMC multiprocessing issues) |
| Float64 | `JAX_ENABLE_X64=1` set in `__init__.py` (must be before any JAX import) |

**Known working versions (as of 2026-04):**

```
jax==0.7.2, jaxlib==0.7.2
jaxns==2.6.9, tfp-nightly
numpy>=2.0, astropy>=6.0
```

**Note:** JAX 0.7.0+ requires `tfp-nightly` (not stable `tensorflow-probability`). 
JAX 0.7.2 uses `tfp-nightly` which is maintained to work with the latest JAX releases.
JAXNS 2.6.9 uses `NestedSampler` (dynamic nested sampling), not `DefaultNestedSampler`.

JAX, jaxlib, and downstream packages (jaxns, tfp-nightly) must be
pinned together.  A version mismatch causes hard-to-diagnose import errors or
segfaults.

---

## Installation and Setup

### Quick Start

```bash
# 1. Clone the repository
git clone <repository_url>
cd dysmalpy

# 2. Create conda environment
conda create -n alma python=3.11
conda activate alma

# 3. Install dependencies
pip install jax==0.7.2 jaxlib==0.7.2
pip install jaxns==2.6.9 tfp-nightly
pip install numpy>=2.0 astropy>=6.0 matplotlib scipy
pip install -e .

# 4. Find and set cuPTI library path (REQUIRED for GPU support)
# Common locations (adjust for your system):
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
# Or find it automatically:
# export LD_LIBRARY_PATH=$(find /usr/local/cuda* -name "libcupti.so" -printf "%h\n" 2>/dev/null | head -1):$LD_LIBRARY_PATH

# 5. Verify GPU support
python -c "import jax; print('Devices:', jax.devices()); print('X64:', jax.config.read('jax_enable_x64'))"
```

### Environment Variables (CRITICAL for GPU Support)

**For JAX GPU acceleration, you MUST set these environment variables BEFORE importing JAX:**

```bash
# Find cuPTI library (location varies by system)
find /usr/local/cuda* -name "libcupti.so" 2>/dev/null

# Set environment variables
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export JAX_ENABLE_X64=1
export CUDA_VISIBLE_DEVICES=0  # Optional: select specific GPU
```

**Common cuPTI locations by installation type:**

| Installation Type | cuPTI Location |
|-------------------|-----------------|
| CUDA 12.x from NVIDIA | `/usr/local/cuda-12.4/extras/CUPTI/lib64/` |
| conda-forge cuda | `$CONDA_PREFIX/lib/` |
| System packages | `/usr/lib/x86_64-linux-gnu/` |
| CUDA 11.x | `/usr/local/cuda-11.x/extras/CUPTI/lib64/` |

**Why this is needed:**
- `LD_LIBRARY_PATH` with cuPTI: JAX cannot find CUDA profiling tools without this
- `JAX_ENABLE_X64=1`: Prevents JAX from defaulting to float32 (must be set before `import jax`)

### Selecting a GPU

**For single-GPU usage (RECOMMENDED for JAXNS):**

```bash
# Check available GPUs and memory
nvidia-smi --query-gpu=index,memory.free --format=csv
# Look for GPUs with >4 GB free for c=300, or >2 GB for c=150

# Use GPU 5 (example - choose one with enough free memory)
export CUDA_VISIBLE_DEVICES=5

# Run your fitting
python demo/demo_2D_fitting_JAXNS.py
```

**Important:** JAXNS 2.6.9 does NOT support multi-GPU parallelization. It only uses ONE GPU at a time, regardless of how many GPUs are visible. Use `CUDA_VISIBLE_DEVICES` to select which GPU to use.

### Verification

**Test JAX GPU support:**
```bash
python -c "import jax; print('Devices:', jax.devices()); print('X64:', jax.config.read('jax_enable_x64'))"
# Should show: Devices: [CudaDevice(id=0)...], X64: True
```

**Test JAXNS:**
```bash
python -c "from jaxns import NestedSampler; print('JAXNS: OK')"
```

**Test full pipeline:**
```bash
python -c "
import dysmalpy
import jax
print('dysmalpy: OK')
print('JAX devices:', jax.devices())
print('JAX X64:', jax.config.read('jax_enable_x64'))
"
```

### Troubleshooting

**Error: "Unable to load cuPTI"**
```bash
# Find cuPTI on your system
find /usr/local/cuda* -name "libcupti.so" 2>/dev/null

# Set LD_LIBRARY_PATH to the directory containing libcupti.so
export LD_LIBRARY_PATH=/path/to/cupti/lib64:$LD_LIBRARY_PATH
```

**Error: "JAX falls back to cpu"**
- Check that `LD_LIBRARY_PATH` is set correctly
- Verify CUDA installation: `nvidia-smi`
- Check JAX version: `python -c "import jax; print(jax.__version__)"`
- Verify cuPTI path contains `libcupti.so`

**JAXNS uses wrong c value**
- Set BOTH `num_live_points` and `c` explicitly in parameter file:
  ```
  num_live_points, 300
  c,                300
  ```

**Git tip:** `activate_alma.sh` is NOT tracked in git as it contains machine-specific paths. Create your own local version if needed:

```bash
# Create local activation script (not in git)
cat > ~/activate_dysmalpy.sh << 'EOF'
#!/bin/bash
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export JAX_ENABLE_X64=1
conda activate alma
EOF

chmod +x ~/activate_dysmalpy.sh
source ~/activate_dysmalpy.sh
```

See `demo/JAXNS_RUN_REPORT.md` for detailed troubleshooting guide.

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
   `python -c "import jax; print(jax.config.read('jax_enable_x64'))"`.

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

5. **Pickle cross-branch incompatibility** — Pickles saved on `main` (JAX) cannot be
   loaded on `dysmalpy_origin` (Cython) and vice versa.  Use JSON params for
   cross-branch transfer.

Additional important gotchas:

- **`np.NaN` removed in NumPy 2.x** — use `np.nan` (19 files affected).
- **GPU OOM for large cubes** — `zcalc_truncate=True` (default) uses active-only
  evaluation, keeping intermediate arrays off GPU.  Peak GPU memory: ~18 GB → < 100 MB.
- **`SpectralCube.moment()` unreliable for CASA cubes** — compute moments manually
  with `numpy.nansum`.
- **MCMC 0% acceptance** — usually caused by `_model` back-references being `None`
  after pickle.  `ModelSet.__setstate__` rebinds them.

## Development Workflow

**Branch discipline:**
- **`dev_jax`** — develop new features and make changes here.  Only merge into
  `main` once the feature is tested and stable.
- **`main`** — stable versions only.  Do not commit work-in-progress directly.

The `dev/` folder contains development notes.  Update them during each development session:

- **`dev/problem.md`** — catalogue of known problems and pitfalls.  Read before debugging.
- **`dev/develop_log.md`** — log of all changes, phase-by-phase.
- **`dev/prompt.md`** — development prompts (if present).

**End-to-end comparison with the original code:**
When model fitting produces unexpected results (wrong chi-squared, bad parameter
recovery, numerical artefacts), use the original Cython implementation on the
`dysmalpy_origin` branch for a detailed comparison.  End-to-end comparison
(run both branches with identical parameters, then compare output cubes, moment
maps, and chi-squared values) is usually the most efficient way to isolate whether
the issue is in the model computation, the fitting pipeline, or the observation
simulation.

```bash
# Diff output cubes between branches (after running both)
git show dysmalpy_origin:path/to/reference_cube.py  # inspect original implementation
git diff main..dysmalpy_origin -- dysmalpy/models/halos.py  # compare a specific file
```

## Testing

Tests require the `dysmalpy-jax` conda environment:

```bash
# For testing reproducibility (CPU)
conda activate dysmalpy-jax
JAX_PLATFORMS=cpu pytest tests/ -v

# For GPU testing (faster, default for fitting)
conda activate dysmalpy-jax
pytest tests/ -v
```

**Note:** Use `JAX_PLATFORMS=cpu` only for reproducible unit testing. For all fitting work (JAXNS, Adam), GPU should be the default for performance. Use `JAX_PLATFORMS=cpu` with MCMC only when troubleshooting multiprocessing issues.

`tests/conftest.py` sets `jax_enable_x64 = True` as a safety net.

Test files:
- `tests/test_jax.py` — 74 JAX-specific unit tests (special functions, cube population,
  model computations, convolution, full pipeline integration)
- `tests/test_models.py` — 27 existing model tests (numpy 2.x compat fixes)

All 101 tests pass.  JAX (`main`) and Cython (`dysmalpy_origin`) produce
numerically identical cubes (max diff = 8.6e-16).
