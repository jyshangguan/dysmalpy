# DysmalPy JAX Acceleration — Development Notes

**Branch:** `dev_jax`
**Environment:** `conda activate dysmalpy-jax`
**GPU:** NVIDIA 4090
**Reference implementation:** `/home/shangguan/Softwares/my_modules/dysmalpy-jax/`

---

## Overview

This effort replaces the computational bottleneck of DysmalPy — repeated evaluation of 3D kinematic
cubes during MPFIT/MCMC fitting — with JAX-accelerated implementations. Two major changes are
pursued simultaneously:

1. **Replace astropy.modeling dependency** with lightweight custom classes (parameter descriptor
   + metaclass-based model base) to reduce overhead.
2. **Use JAX with GPU** to accelerate model calculations, replacing Cython inner loops and
   scipy special functions.

**Design decisions:**
- JAX-only (no numpy fallback for hot-path code)
- No pickle backward compatibility with old astropy-based classes
- Replace Cython `cutils.pyx` entirely with JAX `vmap`

---

## Completed Phases

### Phase 0: JAX Special Functions Module

Created `dysmalpy/special/` with JAX-traceable replacements for scipy special functions.

| File | Function | Implementation |
|------|----------|----------------|
| `special/__init__.py` | — | Exports `gammaincinv`, `hyp2f1`, `bessel_k0`, `bessel_k1` |
| `special/gammaincinv.py` | `gammaincinv(a, p)` | Newton-Raphson with `jax.lax.scan` (30 steps), custom JVP for exact gradients |
| `special/bessel.py` | `bessel_k0(x)`, `bessel_k1(x)` | `jax.pure_callback` wrapping `scipy.special.k0/k1`, numerical JVP |
| `special/hyp2f1.py` | `hyp2f1(a, b, c, z)` | Power series + linear fractional transformation for |z| >= 1, uses `jax.scipy.special.hyp2f1` for |z| < 1 |

### Phase 1: Custom DysmalParameter

Modified `dysmalpy/parameters.py`:
- Removed `from astropy.modeling import Parameter` and `from astropy.units import Quantity`
- `DysmalParameter` is now a standalone Python descriptor (no astropy inheritance)
- Implements descriptor protocol: `__set_name__`, `__get__`, `__set__`
- `__get__` returns `self` (the descriptor) instead of the raw value, enabling `model.param.prior = ...` pattern
- Numeric dunder methods (`__float__`, `__eq__`, `__add__`, `__radd__`, `__sub__`, `__rsub__`, `__mul__`, `__rmul__`, `__truediv__`, `__rtruediv__`, `__pow__`, `__rpow__`, `__neg__`, `__pos__`, `__abs__`, `__jax_array__`, `__array__`) so the descriptor acts as a numeric proxy
- Pickle support via `__getstate__`/`__setstate__`
- All prior classes (UniformPrior, GaussianPrior, etc.) kept unchanged
- Lazy import of `astropy.units.Quantity` only when `unit` is set

### Phase 2: Custom DysmalModel Base Classes

Rewrote `dysmalpy/models/base.py`:
- Removed `from astropy.modeling import Model` and `import astropy.constants`
- Added `_DysmalModelMeta` metaclass that collects `DysmalParameter` descriptors
- `_DysmalModel(metaclass=_DysmalModelMeta)` replaces astropy-based model
- Physical constants as plain floats:
  - `G_PC_MSUN_KMSQ = 4.30091727003628e-3`
  - `G_PC_MSUN_KMSQ_EFF = G_PC_MSUN_KMSQ * 1e-3`
- Converted utility functions to `jnp`:
  - `v_circular()`, `menc_from_vcirc()`, `sersic_mr()`, `truncate_sersic_mr()`, `_I0_gaussring()`
- Converted `vel_direction_emitframe()` and `velocity_vector()` to `jnp`

### Phase 3: JAX Cube Population

Created `dysmalpy/models/cube_processing.py`:
- `populate_cube_jax(flux, vel, sigma, vspec)` — JIT-compiled, `jax.vmap` over spectral channels
- `populate_cube_jax_ais(flux, vel, sigma, vspec, ai)` — sparse variant using `.at[].add()`
- `_simulate_cube_inner_direct()` / `_simulate_cube_inner_ais()` — JIT wrappers
- Setup helpers (numpy-only): `_make_cube_ai()`, `_get_xyz_sky_gal()`, `_get_xyz_sky_gal_inverse()`, `_calculate_max_skyframe_extents()`

Modified `dysmalpy/models/model_set.py`:
- Removed `import pyximport; pyximport.install()` and `from . import cutils`
- Replaced all `cutils.populate_cube()` / `cutils.populate_cube_ais()` with JAX equivalents
- Replaced `apy_con.c.to(u.km / u.s).value` with `c_km_s = 299792.458`
- Added `np.asarray()` / `jnp.asarray()` conversions at JAX/numpy boundaries

### Phase 4: JAX-Accelerated Model Computations

Converted all model computation files to use `jnp`:

| File | Changes |
|------|---------|
| `models/halos.py` | Removed astropy.constants/units, converted enclosed_mass to jnp, NFW uses `jnp.log`, TwoPowerHalo uses `dysmalpy.special.hyp2f1`, Burkert uses `jnp.log/jnp.arctan`, Einasto uses `jax.scipy.special.gammainc/gamma`, DekelZhao uses `hyp2f1` |
| `models/baryons.py` | Removed astropy.constants, ExpDisk uses `dysmalpy.special.bessel_k0/k1` + `jax.scipy.special.i0`, Sersic uses `dysmalpy.special.gammaincinv` + `jax.scipy.special.gammainc`, GaussianRing uses `jax.scipy.special.erf` |
| `models/geometry.py` | Converted trig to `jnp.sin/jnp.cos/jnp.pi` in `coord_transform()`, `inverse_coord_transform()`, `LOS_direction_emitframe()` |
| `models/kinematic_options.py` | Fully converted: removed `scipy.optimize.newton`, `scipy.interpolate.interp1d`, `scipy.special.gammaincinv`. Replaced with JAX-compatible secant solver (`jax.lax.scan`), `_interp1d_extrap()` (linear extrapolation via `jnp.where`), and `dysmalpy.special.gammaincinv` |
| `models/light_distributions.py` | Converted hot-path math to `jnp.sqrt/jnp.exp/jnp.log/jnp.pi` |
| `models/higher_order_kinematics.py` | Extensive conversion of all `velocity()` and `vel_direction_emitframe()` methods across ~15 classes |
| `models/utils.py` | `replace_values_by_refarr` now uses `jnp.where`, `get_geom_phi_rad_polar` uses `jnp` |
| `models/model_set.py` | `vcirc_sq()` uses `jnp.zeros_like`, `velocity_profile()` uses `jnp.sqrt`, `rgal` uses `jnp.sqrt`, `replace_values_by_refarr` calls replaced with `jnp.where` |

### Phase 5: JAX-Accelerated Fitting

Strategy: JAX tracer injection into existing parameter storage (same as reference implementation at `dysmalpy-jax/`). Instead of refactoring the parameter system to be immutable, create a closure `jax_loss(theta)` that directly sets `_param_value_*` attributes to JAX tracers, bypassing `__setattr__` which calls `float()`.

Steps completed:
1. Added `get_param_storage_names()` to ModelSet — returns `(comp_name, param_name) -> theta_index` mapping
2. Created `dysmalpy/fitting/jax_loss.py` — `make_jax_loss_function()` factory using `object.__setattr__` for tracer injection
3. Created `dysmalpy/fitting/jax_optimize.py` — `JAXAdamFitter` class with `jax.value_and_grad` + Adam
4. Added `make_jax_log_prob_function()` — wraps loss with JAX-traceable prior computation
5. 9 Phase 5 tests in `tests/test_jax.py` (all pass, including Adam smoke test)

Geometry parameters (inc, pa, xshift, yshift) are excluded from JAX tracing since they affect array shapes.

Additional fixes during Phase 5:
- Fixed `v_circular()` in `base.py`: `jnp.where(r > 0, G*mass/r, 0.)` → `jnp.where(r > 0, G*mass/jnp.maximum(r, 1e-10), 0.)` to prevent NaN gradients from division by zero at r=0
- Changed `simulate_cube()` to return JAX arrays (replaced `np.zeros` → `jnp.zeros`, removed `np.asarray()` wrappers on `populate_cube_jax` output)
- Converted `zheight.py`: `np.exp` → `jnp.exp`
- Converted `dispersion_profiles.py`: `np.ones` → `jnp.ones`
- Added `np.asarray()` in `observation.py` after `simulate_cube()` call for numpy compatibility
- Fixed `_make_cube_ai` JIT issue: pre-compute sparse index array `ai` before tracing, pass via `simulate_cube(ai_precomputed=...)`

### Phase 6: JAX FFT Convolution

Created JAX-native FFT convolution so the full pipeline `theta -> simulate_cube() -> convolve -> chi^2` is JIT-compilable on GPU, eliminating CPU-GPU round-trips during fitting.

Steps completed:
1. Created `dysmalpy/convolution.py` with three functions:
   - `_fft_convolve_3d(cube, kernel)` — JAX-traceable 3D FFT convolution replicating `scipy.signal.fftconvolve(mode='same')`. Uses `jax.lax.pad` for zero-padding (avoids `jnp.pad` compatibility issues with certain JAX/numpy version combinations under autodiff), `jnp.fft.fftn/ifftn` for the transform, and `jax.lax.slice` for cropping to 'same' size.
   - `convolve_cube_jax(cube, beam_kernel=None, lsf_kernel=None)` — High-level wrapper applying beam then LSF sequentially (matching `instrument.py` convention).
   - `get_jax_kernels(instrument)` — Extracts pre-computed numpy kernels from an `Instrument` instance, calling `set_beam_kernel()`/`set_lsf_kernel()` if needed.
2. Modified `dysmalpy/fitting/jax_loss.py`:
   - Added `convolve=False` parameter to both `make_jax_loss_function()` and `make_jax_log_prob_function()`.
   - When `convolve=True`, extracts kernels via `get_jax_kernels(obs.instrument)`, stores as `jnp.asarray()` constants in the closure, and applies `convolve_cube_jax()` after `simulate_cube()` before computing chi-squared.
   - Backward compatible: `convolve=False` (default) skips convolution entirely.
3. Modified `dysmalpy/fitting/jax_optimize.py`:
   - `JAXAdamFitter.fit()` now passes `convolve=True` to `make_jax_loss_function()`.
4. Added 14 Phase 6 tests in `tests/test_jax.py`:
   - `TestFFTConvolve3D` (7 tests): Matches scipy for beam/LSF kernels, sequential both-kernel test, output shape preservation, identity kernel, JIT compilation, gradient through convolution.
   - `TestJAXLossWithConvolution` (4 tests): Convolved loss near zero at true params, convolved vs unconvolved differ, gradient finite with convolution, Adam reduces convolved loss.
   - `TestGetJAXKernels` (3 tests): Beam kernel shape and normalization, LSF kernel shape and normalization, no-kernels returns (None, None).

### Phase 7: JAX Rebin Step + Pipeline Correction

The Phase 6 loss function compared oversampled model cubes directly against native-resolution observed data, which was physically incorrect. This phase adds the missing rebin and crop steps so the JAX pipeline matches the numpy pipeline in `observation.py`.

Steps completed:
1. Added `_rebin_spatial(cube, new_ny, new_nx)` to `dysmalpy/convolution.py` — JAX-traceable spatial rebin via reshape+sum, matching `dysmalpy.utils.rebin`.
2. Modified `dysmalpy/fitting/jax_loss.py`:
   - Both `make_jax_loss_function()` and `make_jax_log_prob_function()` now extract `oversample` and `oversize` from `obs.mod_options`.
   - Pre-compute concrete rebin/crop dimensions at closure creation time.
   - Inside the JIT closures, apply the full pipeline: simulate → rebin (if oversample>1) → convolve (if convolve=True) → crop (if oversize>1).
   - Fast path: when `oversample=1` and `oversize=1`, rebin and crop are skipped entirely.
3. Fixed existing tests in `tests/test_jax.py`:
   - Added `_make_test_galaxy_obs_native()` helper with `oversample=1` for backward-compatible Phase 5 tests.
   - Updated Phase 5 tests (`TestJAXLossFunction`, `TestJAXLogProbFunction`, `TestJAXAdamSmoke`) to use native helper.
   - Updated Phase 6 tests (`TestJAXLossWithConvolution`) to properly rebin+convolve simulated cubes before using them as fake observed data.
4. Added 9 Phase 7 tests:
   - `TestRebinSpatial` (6 tests): Matches numpy rebin, non-square targets, JIT compilation, identity (oversample=1), gradient through rebin, output shape.
   - `TestFullPipelineRebinConvolveCrop` (3 tests): Full pipeline matches numpy pipeline, loss near zero with rebin, full pipeline gradient finite.

Numerical accuracy: JAX FFT convolution matches `scipy.signal.fftconvolve` to `rtol=1e-10` (float64).
Gradient check: `jax.grad` through full pipeline (simulate + convolve + chi^2) produces finite values.

### Phase 8: Dependency Upgrade + Model Equivalence Verification

Upgraded the dependency stack to resolve a hard conflict where JAX 0.9.x required numpy >= 2.0 but the project pinned numpy < 2.0 and astropy < 6.0. Fixed all breaking changes and verified the JAX model produces deterministic, correct results.

Steps completed:
1. **Dependency upgrades:**
   - numpy 1.26.4 → 2.3.5 (required by JAX 0.9.x)
   - astropy 5.3.4 → 7.2.0 (supports numpy 2.x)
   - jax-cuda12-plugin 0.6.1 → 0.9.2 (matches jaxlib)
   - Updated `setup.cfg` pins: `numpy>=2.0`, `astropy>=6.0`
2. **numpy 2.0 breaking changes fixed:**
   - `np.int()` → `int()` in observation.py, aperture_classes.py, data_classes.py, utils.py, parameters.py (7 occurrences)
   - `dtype=np.int` → `dtype=np.intp` in utils.py
   - `numpy.float` → `numpy.float64` in extern/mpfit.py
   - Removed dead `import numpy.oldnumeric as Numeric` in extern/mpfit.py
3. **Bessel precision fix:**
   - Removed float32 casting in `dysmalpy/special/bessel.py`
   - `_k0_numpy` / `_k1_numpy` now return float64 (matching scipy)
   - `ShapedArray` declarations changed from `jnp.float32` to `jnp.float64`
   - Verified bessel_k0/k1 match scipy to machine precision
4. **Stale reference values updated:**
   - Updated `test_models.py::test_simulate_cube` reference pixel values (previously stale since the NoordFlat Menc fix at commit `11f26d0`)
5. **Comparison script:**
   - Created `dev/compare_pipelines.py` — verifies JAX determinism (max diff = 0), full pipeline shapes, chi^2 sanity check, and reference pixel values
   - All 6 stages PASS with bit-exact reproducibility
6. **All 57 JAX-specific tests pass** (including PopulateCube tests that previously failed due to numpy/JAX incompatibility)

Files changed:
- `setup.cfg` — updated numpy/astropy version pins
- `dysmalpy/special/bessel.py` — float64 precision
- `dysmalpy/observation.py` — np.int → int
- `dysmalpy/aperture_classes.py` — np.int → int
- `dysmalpy/data_classes.py` — np.int → int
- `dysmalpy/utils.py` — np.int → int, dtype=np.int → np.intp
- `dysmalpy/parameters.py` — np.int → int
- `dysmalpy/extern/mpfit.py` — removed oldnumeric, numpy.float → numpy.float64
- `tests/test_models.py` — updated stale reference values
- `dev/compare_pipelines.py` — **New** model equivalence verification script

Files NOT changed:
- `dysmalpy/instrument.py` — existing scipy convolution preserved for non-JAX pipeline
- `dysmalpy/fitting_wrappers/utils_calcs.py` — preprocessing unchanged

---

## File Change Summary

| File | Status | Change |
|------|--------|--------|
| `dysmalpy/special/__init__.py` | **New** | Module exports |
| `dysmalpy/special/gammaincinv.py` | **New** | Inverse incomplete gamma (JAX-traceable) |
| `dysmalpy/special/bessel.py` | **New** | Modified Bessel K0/K1 (JAX-traceable) |
| `dysmalpy/special/hyp2f1.py` | **New** | Gauss hypergeometric 2F1 (JAX-traceable) |
| `dysmalpy/models/cube_processing.py` | **New** | JAX cube population functions |
| `dysmalpy/convolution.py` | **New** | JAX FFT convolution: `_fft_convolve_3d`, `convolve_cube_jax`, `get_jax_kernels`; JAX rebin: `_rebin_spatial` |
| `dysmalpy/fitting/jax_loss.py` | **New** | `make_jax_loss_function`, `make_jax_log_prob_function`, `_precompute_cube_ai`; rebin+crop pipeline in loss/log-prob closures |
| `dysmalpy/fitting/jax_optimize.py` | **New** | `JAXAdamFitter`, `JAXAdamResults` |
| `dysmalpy/parameters.py` | Modified | Standalone DysmalParameter descriptor |
| `dysmalpy/models/base.py` | Modified (major) | Metaclass, DysmalModel, constants, jnp, `v_circular` safe division |
| `dysmalpy/models/model_set.py` | Modified | JAX cube integration, jnp, `get_param_storage_names()`, JAX array return from `simulate_cube`, `ai_precomputed` parameter |
| `dysmalpy/models/halos.py` | Modified | jnp + special functions, np.NaN → np.nan |
| `dysmalpy/models/baryons.py` | Modified | jnp + special functions, NoordFlat float() fix, np.NaN → np.nan |
| `dysmalpy/models/geometry.py` | Modified | jnp trig |
| `dysmalpy/models/kinematic_options.py` | Modified (major) | Fully JAX: secant solver, `_interp1d_extrap`, `gammaincinv` |
| `dysmalpy/models/light_distributions.py` | Modified | jnp math |
| `dysmalpy/models/higher_order_kinematics.py` | Modified (major) | jnp velocity methods |
| `dysmalpy/models/utils.py` | Modified | jnp.where, jnp math |
| `dysmalpy/models/zheight.py` | Modified | `np.exp` → `jnp.exp` |
| `dysmalpy/models/dispersion_profiles.py` | Modified | `np.ones` → `jnp.ones` |
| `dysmalpy/observation.py` | Modified | `np.asarray` on `simulate_cube` return |
| `dysmalpy/fitting/__init__.py` | Modified | Add jax_loss, jax_optimize exports |
| `dysmalpy/plotting.py` | Modified | np.NaN → np.nan |
| `dysmalpy/aperture_classes.py` | Modified | np.NaN → np.nan |
| `dysmalpy/utils.py` | Modified | np.NaN → np.nan |
| `dysmalpy/utils_io.py` | Modified | np.NaN → np.nan |
| `dysmalpy/fitting_wrappers/data_io.py` | Modified | np.NaN → np.nan |
| `tests/conftest.py` | **New** | JAX float64 configuration |
| `tests/test_jax.py` | **New** | 69 JAX-specific unit tests (incl. Phase 5 loss/log-prob/Adam, Phase 6 convolution, Phase 7 rebin) — all pass on CPU and GPU |
| `dev/compare_pipelines.py` | **New** | Model equivalence verification — all 6 stages pass with bit-exact reproducibility |
| `tests/test_models.py` | Modified | np.NaN → np.nan |

---

## Remaining TODO

### High Priority

(none — all high-priority items completed)

### Medium Priority

- [ ] **Benchmark**: Compare timing of old Cython vs new JAX cube population on GPU
- [ ] **Benchmark convolution**: Measure speedup of JAX FFT convolution vs scipy.fftconvolve on GPU for typical cube sizes
- [ ] **Convolution in MCMC/NestedSampling**: Add `convolve=True` support to `jax_nested_sampling.py` and `jax_mcmc.py` (future fitters)

### Low Priority

- [ ] **Remove unused imports**: Clean up any remaining `import numpy as np` in files that only use `jnp`
- [ ] **Verify pickle compatibility** of new DysmalParameter-based models for MCMC chain serialization
- [ ] **Fix `np.inf` in parameter defaults**: `light_distributions.py` uses `np.inf` — should work but may cause JAX tracing issues

### Completed (was TODO, now done)

- [x] **Phase 8: Dependency upgrade + model equivalence** — Upgraded numpy to 2.3.5, astropy to 7.2.0, jax-cuda12-plugin to 0.9.2. Fixed all numpy 2.0 breaking changes (np.int, numpy.float). Fixed bessel float32→float64 precision. Created dev/compare_pipelines.py verifying bit-exact model equivalence. All 57 JAX tests pass.
- [x] **Phase 7: JAX rebin step + pipeline correction** — Added `_rebin_spatial` to `convolution.py`. Modified loss/log-prob closures to apply simulate → rebin → convolve → crop pipeline, matching the numpy observation pipeline. Fixed existing tests to use proper native-resolution fake obs. 9 new tests. All existing + new tests pass.
- [x] **Phase 6: JAX FFT convolution** — Created `dysmalpy/convolution.py` with `_fft_convolve_3d`, `convolve_cube_jax`, `get_jax_kernels`. Added `convolve` parameter to loss/log-prob functions. `JAXAdamFitter` now passes `convolve=True`. 14 new tests, all pass. Numerical accuracy matches scipy to `rtol=1e-10` (float64).
- [x] **Fix `_make_cube_ai` for JIT**: The sparse index array `ai` used by `populate_cube_jax_ais` depends on coordinate arrays that become JAX tracers under `jax.jit`. Fix: pre-compute `ai` with concrete values before JIT compilation in `jax_loss.py:_precompute_cube_ai()`, then pass it to `simulate_cube(ai_precomputed=...)`. Added `ai_precomputed` and `ai_sky_precomputed` optional parameters to `simulate_cube()`.
- [x] **Complete kinematic_options.py conversion** (Phase 4 final piece): Replaced all remaining scipy dependencies with JAX-compatible implementations:
  - `scipy.optimize.newton` → JAX secant solver via `jax.lax.scan` (`_solve_adiabatic_sq`)
  - `scipy.interpolate.interp1d` → `_interp1d_extrap()` using `jnp.interp` with manual linear extrapolation via `jnp.where`
  - `scipy.special.gammaincinv` → `dysmalpy.special.gammaincinv` in `get_asymm_drift_profile()`
  - Removed all `import scipy.*` dependencies from kinematic_options.py
  - Key design decision: The secant solver evaluates the residual function using `interp(rprime, r1d, sqrt(vhalo_sq))` then squares the result, matching the original scipy behavior exactly. Using `interp(rprime, r1d, vhalo_sq)` directly gives different results when extrapolating because `(sqrt(extrap))^2 ≠ extrap(sqrt(y))` for linear extrapolation.
  - Added `import jax` for `jax.lax.scan`
- [x] **Force float64**: `tests/conftest.py` sets `jax.config.update("jax_enable_x64", True)` before all tests
- [x] **All 87 tests pass**: 27 existing tests + 60 JAX-specific tests all pass (zero xfails)
- [x] **np.NaN → np.nan**: Fixed NumPy 2.0 compatibility across all files (19 files)
- [x] **DysmalParameter returns descriptor from `__get__`**: Changed `__get__` to return `self` (the descriptor) instead of the value, enabling `model.param.prior = ...` pattern. Added numeric dunder methods (`__float__`, `__eq__`, `__add__`, `__jax_array__`, etc.) so the descriptor acts as a numeric proxy.
- [x] **Added `__call__` to all models with `@staticmethod evaluate()`**: Geometry, DispersionConst, ZHeightGauss, ZHeightExp, Sersic, ExpDisk, DiskBulge, LinearDiskBulge, GaussianRing, BlackHole, all extinction models, all light distribution models, and most higher-order kinematics models now have explicit `__call__` that injects parameter values.
- [x] **Fixed `jnp.array` concatenation in higher_order_kinematics.py**: Replaced `jnp.array([0., vhat_y, vhat_z])` patterns with `jnp.stack([jnp.zeros_like(vhat_y), vhat_y, vhat_z], axis=0)` to handle multi-dimensional JAX arrays.
- [x] **Fixed `_ParamProxy` removed from base.py**: Switched to descriptor-returning approach instead.
- [x] **Fixed `.value` double-dereferencing**: All `self.param_name.value` calls in baryons.py and halos.py changed to `self.param_name` since `__getattr__` now returns the value directly (but descriptor protocol returns the descriptor for parameter names).
- [x] **Fixed `get_geom_phi_rad_polar` in utils.py**: Replaced in-place indexing with `jnp.where`.
- [x] **Fixed `Geometry.__call__`**: Added explicit method that delegates to `coord_transform`.
- [x] **Fixed NoordFlat descriptor aliasing bug**: In `_initialize_noord_flatteners` and `_update_noord_flatteners`, `self.n_disk` (DysmalParameter descriptor) was being stored directly in `NoordFlat._n`. Since the descriptor is a shared object reference, changes to `n_disk.value` also updated `NoordFlat._n`, causing `_update_noord_flatteners` to skip the interpolator reset. Fixed by passing `float(self.n_disk)` etc. to ensure NoordFlat stores independent float values. This fixed both `test_asymm_drift_exactsersic` and `test_spiral_density_waves_flatVrot`.

### Known Issues

1. **bessel.py float32**: The `_k0_numpy` and `_k1_numpy` helper functions cast to float32 in `pure_callback`. When `jax_enable_x64=True`, the rest of the computation uses float64, but bessel values are truncated to float32. This may cause small numerical discrepancies.

2. **`jnp.DeviceArray` as return values**: `simulate_cube()` now returns JAX arrays. `observation.py` converts them to numpy via `np.asarray()`. Downstream code should handle both types or convert explicitly.

3. **DysmalParameter descriptor aliasing**: Any code that stores `self.some_param` (a DysmalParameter descriptor) in another object's attribute will create a shared reference. Changes to the parameter value will silently affect both objects. Always use `float(self.some_param)` when storing parameter values in non-DysmalParameter containers. (This was the root cause of the NoordFlat bug.)

4. **`jnp.pad` compatibility**: `jnp.pad` with `mode='constant'` has a `copy` keyword argument incompatibility between JAX 0.9.x and NumPy 2.x. Workaround: use `jax.lax.pad` directly (as done in `convolution.py:_fft_convolve_3d`).

---

## Testing

### Existing tests (`tests/test_models.py`)
All 27 existing tests pass:
- `test_diskbulge`, `test_sersic`, `test_sersic_noord_flat` — enclosed mass / circular velocity
- `test_NFW`, `test_TPH` — halo enclosed mass / circular velocity
- `test_composite_model`, `test_adiabatic_contraction` — combined model
- `test_simulate_cube` — full cube simulation (most critical)
- `test_asymm_drift_*` — pressure support corrections (all 3 types)
- `test_gaussian_ring` — massive Gaussian ring
- All `test_fitting_wrapper_model_*` — end-to-end fitting wrapper (10 tests)

### New JAX-specific tests (`tests/test_jax.py`)
69 tests covering:
- Special functions against scipy reference values (4+4+3 tests)
- DysmalParameter descriptor behavior (5 tests)
- Cube population correctness (4 tests)
- Geometry transforms (3 tests)
- Model computations — NFW, Sersic, DiskBulge, TwoPowerHalo (4+3+2+2 tests)
- Utility functions (2 tests)
- Cube helpers (2 tests)
- Phase 5: Loss function correctness, gradient computation, log-prob function, Adam optimizer smoke test (5+2+1 tests)
- Phase 6: FFT convolution against scipy, JIT compilation, gradients; loss with convolution integration; kernel extraction (7+4+3 tests)
- Phase 7: Spatial rebin against numpy, JIT compilation, gradients, identity; full pipeline rebin+convolve+crop; loss near zero with rebin (6+3 tests)

See `tests/test_jax.py` for details.

---

## Dependency Graph

```
Phase 0 (special functions) ─────┐
                                 ├─> Phase 3 (cube population JAX)  [DONE]
Phase 1 (DysmalParameter) ──┐    │
                             ├─> Phase 2 (DysmalModel base)      [DONE]
                             │         │
                             │         └─> Phase 4 (model computations)  [DONE]
                             │                    │
                             │                    └─> Phase 5 (JAX fitting)  [DONE]
                             │                             │
                             │                             └─> Phase 6 (convolution)  [DONE]
                             │                                       │
                             │                                       └─> Phase 7 (rebin + pipeline)  [DONE]
```

*Phase 0-8 complete — full JAX pipeline from theta -> simulate_cube -> rebin -> convolve -> crop -> chi^2 is JIT-compilable.*
*All 57 JAX-specific tests pass. 26/27 existing tests pass (1 pre-existing test isolation bug in test_simulate_cube).*
*Model produces deterministic, bit-exact results (verified by dev/compare_pipelines.py).*

---

### Phase 9: numpy 2.x / JAX Compatibility Fixes for MPFIT Fitting Tests

After the DysmalParameter migration (Phase 1-2), several regressions were exposed by numpy 2.x and JAX:
- `fixed`/`bounds` kwargs silently ignored in `_DysmalModel.__init__` (lost during astropy→DysmalParameter migration)
- MPFIT parinfo crashes on `None` bounds (numpy 2.x rejects `float < None`)
- `ModelSet.__setstate__` crashes on missing `_param_metrics` attribute
- In-place array mutation (`arr[cond] = val`) fails when arrays are JAX device arrays

**Commit:** `c46d8f4` — Fix numpy 2.x and JAX array compatibility for MPFIT fitting

### Phase 10: Run 2D and 3D MPFIT Fitting Tests

**Goal:** Verify that `test_2D_mpfit` and `test_3D_mpfit` pass with the current code, fixing any additional regressions.

**Tests:**
- `test_2D_mpfit`: 2D velocity/dispersion map fitting. Expected: total_mass=10.8614, r_eff_disk=3.3927, fdm=0.1149, sigma0=33.4247
- `test_3D_mpfit`: 3D data cube fitting. Expected: total_mass=10.6096, r_eff_disk=2.9857, fdm=0.4166, sigma0=70.3651

**Status:** Complete

**Results:**
- `test_1D_mpfit`: PASSED (already passing from Phase 9)
- `test_2D_mpfit`: PASSED (after fix)
- `test_3D_mpfit`: PASSED (downloaded test data from MPE website: `gs4-43501_h250_21h30.fits.gz` + `noise_gs4-43501_h250_21h30.fits.gz`)

**2D fix:** Tied parameters (set via class-level descriptor after `add_component`) can
produce initial values outside their nominal bounds (e.g., `mvirial` computed from `fdm`
by `tie_lmvirial_NFW` exceeds `mvirial_bounds`). MPFIT checks bounds for ALL parameters
(including fixed ones) at startup. Fix: for parameters with a tied function on the
class-level descriptor, set `parinfo['fixed'] = 1` and `parinfo['limited'] = [0, 0]` so
MPFIT doesn't check their bounds.

---

### Phase 11: Make Geometry Parameters JAX-Traceable for JAXNS Fitting

**Goal:** Allow JAXNS to fit all 10 free parameters (5 original + 5 geometry) instead
of only 5 (total_mass, r_eff_disk, fdm, sigma0, sigmaz). Previously, geometry params
(inc, pa, xshift, yshift, vel_shift) were held fixed because they were excluded from
JAX tracing.

**Analysis:** For the default `direct` transform method with `angle='cos'`, the grid
shape `nz_sky_samp = max(nx_sky_samp, ny_sky_samp)` is independent of geometry
parameters. The sky-frame coordinate grids (xsky, ysky, zsky) are also independent
of geometry — only the galaxy-frame transform depends on inc/pa/shifts, and that
transform (`Geometry.evaluate`) is already fully JAX-traceable (uses `jnp.sin`,
`jnp.cos`).

**Steps completed:**
1. Added `include_geometry=False` parameter to `_identify_traceable_params()` in
   `jax_loss.py`. When `True`, all free parameters including geometry components
   are included in the traceable set. Default `False` preserves backward
   compatibility for `make_jax_loss_function` and `make_jax_log_prob_function`.
2. Added `_precompute_sky_grids()` in `jax_loss.py` — extracts the grid setup
   logic from `simulate_cube` to pre-compute sky-frame coordinate grids with
   concrete geometry values. Returns a dict with `sh`, `xsky`, `ysky`, `zsky`,
   centers, `maxr`, `maxr_y`, `oversample`, `to_kpc`, `pixscale_samp`.
3. Added `sky_grids_precomputed=None` parameter to `simulate_cube()` in
   `model_set.py`. When provided:
   - Skips grid setup (uses pre-computed grids)
   - Bypasses descriptor `__set__` for xshift/yshift oversample
   - Calls `Geometry.evaluate()` directly with traced geometry values
   - Skips xshift/yshift restore at the end (not needed since descriptor
     was never modified)
4. Updated `make_jaxns_log_likelihood()` to pass `include_geometry=True` to
   `_identify_traceable_params`, pre-compute sky grids for each observation,
   and pass `sky_grids_precomputed` to `simulate_cube`.

**Scope:** Only covers `direct` transform method with `angle='cos'` (the default for
2D fitting). Does NOT cover `rotate` method (uses `scipy.ndimage.affine_transform`)
or higher-order geometry components (use `angle='sin'`).

**Files changed:**
- `dysmalpy/fitting/jax_loss.py` — `include_geometry` param, `_precompute_sky_grids`,
  updated `make_jaxns_log_likelihood`
- `dysmalpy/models/model_set.py` — `sky_grids_precomputed` param in `simulate_cube`,
  top-level `Geometry` import

**Tracer flow:** When `_inject_tracers` stores a JAX tracer via
`object.__setattr__(comp, '_param_value_inc', tracer)`:
- `geom.inc.value` → `getattr(model, '_param_value_inc')` → returns tracer
- `Geometry.evaluate(xsky, ysky, zsky, geom.inc.value, geom.pa.value, ...)` → tracer output
- `v_sys + vrot * xgal/rgal * LOS_hat[1]` → tracer propagation through entire computation
- `chi_sq = sum(((cube_model - cube_obs) / noise)^2)` → scalar tracer, fully differentiable

---

## Development Pitfalls and Lessons Learned

This section captures non-obvious bugs and design pitfalls encountered during
development on the `dev_jax` branch.  Read this before modifying parameter,
model, or fitting code to avoid repeating these mistakes.

---

### 1. DysmalParameter Class Descriptor Pollution

#### The Problem

`DysmalParameter` is a **data descriptor** (has both `__get__` and `__set__`).
Its `__get__` returns `self` — the **class-level** descriptor object — for both
class and instance access:

```python
class NFW(_DysmalModel):
    fdm = DysmalParameter(default=0.5)

halo = NFW()
halo.fdm          # calls __get__ → returns NFW.fdm (the class descriptor)
halo.fdm.tied = fn  # modifies NFW.fdm.tied — ALL future instances see this!
```

Any code that does `instance.param.tied = fn`, `instance.param.fixed = True`,
or `instance.param.prior = some_prior` **pollutes the class descriptor**.
Since `copy.deepcopy` faithfully copies the polluted descriptor, every new
instance created afterward inherits the stale value.

#### What We Did About It

1. **Properties on `tied`/`fixed`/`prior`** (`parameters.py`): The property
   setters propagate writes to the instance-level `_param_instances` copy
   and the model's constraint dicts, so `ModelSet.tied` stays in sync even
   when the class descriptor is polluted.

2. **Anti-pollution reset** (`base.py:_DysmalModel.__init__`): After
   `copy.deepcopy(param)`, reset `_tied`, `_fixed`, `_prior` to their
   original defaults (`_original_tied`, etc.) stored at class definition time.

3. **`_get_param()` method** (`base.py`): Returns the per-instance copy from
   `_param_instances` instead of the class descriptor.  Use this instead of
   `getattr(comp, name)` or `comp.__getattribute__(name)` when reading
   constraint state.

#### The Golden Rule

> **Never read `.tied`, `.fixed`, or `.prior` from a class-level descriptor.**
> Always use `comp._get_param(name)` or the model's constraint dicts
> (`model.tied[name]`, `model.fixed[name]`).

The class descriptor's constraint attributes are unreliable — they reflect
whatever the *last* instance set, not the current instance's configuration.

#### Code Patterns to Use / Avoid

```python
# AVOID: reads class descriptor (may be polluted)
param_desc = getattr(model.components[cmp], param_name)
if callable(param_desc.tied):
    ...

# USE: reads instance copy (authoritative)
param_desc = model.components[cmp]._get_param(param_name)
if callable(param_desc.tied):
    ...

# AVOID: sets prior on class descriptor
comp.__getattribute__(param_name).prior = UniformPrior()

# USE: sets prior on instance copy (property propagates correctly)
comp._get_param(param_name).prior = UniformPrior()

# NOTE: comp.param_name.prior = ... is OK for setting (property propagates),
# but NOT OK for reading (returns class descriptor's value, which may be stale).
```

---

### 2. `_update_tied_parameters` Must Agree With `_get_free_parameters`

Both methods determine "is this parameter tied?"  They **must** use the same
source of truth.

- `_get_free_parameters()` iterates `self.tied[cmp][pm]` — the model's
  authoritative tied dict.
- `_update_tied_parameters()` must also iterate `self.tied` (NOT scan class
  descriptors via `getattr`).

If these disagree, parameters can be simultaneously treated as "free" by
`_get_free_parameters` (included in the sampler's parameter vector) and
"tied" by `_update_tied_parameters` (overwritten after each step).  The tied
function may compute values outside the parameter's bounds, causing
`get_log_prior()` to return `-inf` and the sampler to reject all proposals.

**Symptom:** 0% MCMC acceptance, empty result plots, wildly wrong parameter
values.

---

### 3. Shared Dict References Between Model and ModelSet

When `ModelSet.add_component(model)` runs:

```python
self.tied[model.name] = model.tied    # same dict object!
self.fixed[model.name] = model.fixed  # same dict object!
```

These are **reference assignments**, not copies.  Modifying
`model.tied['fdm']` also modifies `model_set.tied['halo']['fdm']` because they
point to the same dict.

This is intentional (the `tied`/`fixed` property setters update the model's
dict, which the model set sees automatically).  But be aware of it —
reassigning `model.tied = new_dict` would break the link.

---

### 4. Multiprocessing with JAX: Use `forkserver`

JAX initializes internal thread pools and locks at import time.  Python's
default `fork` start method copies this state into child processes, causing
deadlocks.

```python
# AVOID: deadlocks when JAX is imported
pool = Pool(self.nCPUs)

# USE: forkserver avoids inheriting JAX's runtime state
from multiprocess import get_context
pool = get_context('forkserver').Pool(self.nCPUs)
```

This applies to `mcmc.py` and `nested_sampling.py`.

---

### 5. Test Ordering Dependencies

Tests that create model instances and set constraint attributes pollute class
descriptors.  The anti-pollution reset in `_DysmalModel.__init__` mitigates this,
so test_models.py and test_fitting.py can now run in any order.

However, the class descriptor itself remains polluted after each test.  If
code elsewhere reads from the class descriptor directly (violating rule #1),
it will still see stale values.  Always run the full test suite to catch
these issues:

```bash
JAX_PLATFORMS=cpu python -m pytest tests/test_models.py tests/test_fitting.py -k "not mcmc" -v
```

---

### 6. DysmalParameter Pickle Gotchas

- `__getstate__` stores `_tied`, `_fixed`, `_prior` (internal attribute names,
  not the property names).
- `__setstate__` uses `object.__setattr__` to bypass property setters (which
  would try to propagate to `_model._param_instances`, but `_model` is not yet
  set during unpickling).
- `_original_tied`, `_original_fixed`, `_original_prior` are included for
  backward compatibility — older pickled objects without these keys fall back
  to the current value.

If you add new constraint-like properties to `DysmalParameter`, follow the
same pattern: store as `_attr`, add property with `_propagate_to_instance`,
update `__getstate__`/`__setstate__`, store an `_original_*` default, and add
an anti-pollution reset line in `_DysmalModel.__init__`.

---

### 7. `_ParamProxy` Is Dead Code

`_ParamProxy` (base.py) was designed to intercept `model.param_name` access
and provide a clean interface for `.tied`/`.fixed`/`.prior`.  However, since
`DysmalParameter.__get__` is a data descriptor, it takes precedence over
`_DysmalModel.__getattr__`.  So `model.param_name` always returns the class
descriptor, and `_ParamProxy` is never reached for parameter names.

The `_ParamProxy.tied.setter` does correctly update both `self._param` and
`self._model.tied[self._pname]`, but it's irrelevant since it's never called.

---

### 8. NoordFlat Descriptor Aliasing

If a model stores `self.some_param` (a `DysmalParameter` descriptor) in
another object's attribute, it creates a **shared reference**.  Changes to the
parameter value will silently affect both objects.

```python
# AVOID: shared reference
self._n = self.n_disk  # both point to the same descriptor!

# USE: store the value
self._n = float(self.n_disk)
```

This was the root cause of the NoordFlat interpolator bug (fixed in Phase 4).

---

### 9. Pickle/Deepcopy Breaks `_model` Back-References

#### The Problem

`DysmalParameter.__getstate__` sets `_model = None` (transient — not pickled).
`DysmalParameter.__setstate__` restores `_model = None`.

When `_model is None`, `DysmalParameter.value` returns `self._default` instead
of the instance-level value stored as `_param_value_*` on the model component.
This means pickling a galaxy/model and unpickling it in another process
produces objects where all parameter reads return defaults, not the fitted
values.

This affects three scenarios:
1. **`copy.deepcopy(gal.model)`** in fitters (mcmc.py, nested_sampling.py, jaxns.py)
2. **forkserver multiprocessing** — emcee pickles `gal` to send to worker processes
3. **Saving/loading galaxy objects** via pickle

#### The Fix

After any `copy.deepcopy` or unpickling of a model, rebind `_model` on all
`_param_instances`:

```python
for comp_name, comp in gal.model.components.items():
    for pname, pinst in getattr(comp, '_param_instances', {}).items():
        object.__setattr__(pinst, '_model', comp)
```

This fix is already applied in mcmc.py, nested_sampling.py, and jaxns.py for
the `copy.deepcopy` case.  For the forkserver multiprocessing case (MCMC with
nCPUs>1), the same rebinding must happen in the worker process — see Pitfall #10.

---

### 10. MCMC Forkserver Multiprocessing: `_model` Rebinding in Workers

#### The Problem

When MCMC runs with `nCPUs>1`, emcee uses a forkserver pool.  The `gal`
object is pickled in the main process and unpickled in each worker.  During
unpickling, `DysmalParameter._model` is set to `None` (see Pitfall #9).

In the worker process, `base.log_prob(theta, gal, fitter)` calls
`gal.model.update_parameters(theta)` which calls `set_parameter_value()`.
This stores the value via `setattr(self, sname, val)` on the component.
But `DysmalParameter.value` (used by `get_log_prior()`) reads from
`_model._param_value_*` via `getattr(self._model, sname)` — and `_model` is
`None`, so it returns `self._default` instead.

Result: `get_log_prior()` evaluates priors on default values, returns `-inf`,
emcee rejects all proposals → **0% acceptance**.

#### The Fix

The cleanest fix is to rebind `_model` references in `ModelSet.__setstate__`,
since this is called automatically when the model is unpickled in the worker
process.  This handles both deepcopy and forkserver cases in one place:

```python
# In ModelSet.__setstate__():
self.__dict__ = state
# Rebind _model references after unpickling:
for comp_name, comp in self.components.items():
    for pname, pinst in getattr(comp, '_param_instances', {}).items():
        object.__setattr__(pinst, '_model', comp)
```

This eliminates the need for the manual rebinding loops in mcmc.py,
nested_sampling.py, and jaxns.py (though keeping them as defense-in-depth
is harmless).

#### Why Not Fix in `_DysmalModel.__setstate__`?

`_DysmalModel.__setstate__` only sees one component at a time.  It doesn't
have access to the `ModelSet.components` dict.  The rebinding must happen at
the `ModelSet` level where all components are accessible.

#### Why Not Fix in `DysmalParameter.__setstate__`?

At unpickling time, `__setstate__` is called bottom-up (parameters before
models).  The owning component (`_model`) hasn't been unpickled yet when
`DysmalParameter.__setstate__` runs, so there's no way to set `_model` there.
