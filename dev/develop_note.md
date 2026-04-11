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

---

## File Change Summary

| File | Status | Change |
|------|--------|--------|
| `dysmalpy/special/__init__.py` | **New** | Module exports |
| `dysmalpy/special/gammaincinv.py` | **New** | Inverse incomplete gamma (JAX-traceable) |
| `dysmalpy/special/bessel.py` | **New** | Modified Bessel K0/K1 (JAX-traceable) |
| `dysmalpy/special/hyp2f1.py` | **New** | Gauss hypergeometric 2F1 (JAX-traceable) |
| `dysmalpy/models/cube_processing.py` | **New** | JAX cube population functions |
| `dysmalpy/fitting/jax_loss.py` | **New** | `make_jax_loss_function`, `make_jax_log_prob_function` |
| `dysmalpy/fitting/jax_optimize.py` | **New** | `JAXAdamFitter`, `JAXAdamResults` |
| `dysmalpy/parameters.py` | Modified | Standalone DysmalParameter descriptor |
| `dysmalpy/models/base.py` | Modified (major) | Metaclass, DysmalModel, constants, jnp, `v_circular` safe division |
| `dysmalpy/models/model_set.py` | Modified | JAX cube integration, jnp, `get_param_storage_names()`, JAX array return from `simulate_cube` |
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
| `tests/test_jax.py` | **New** | 46 JAX-specific unit tests (incl. Phase 5 loss/log-prob/Adam) |
| `tests/test_models.py` | Modified | np.NaN → np.nan |

---

## Remaining TODO

### High Priority

(None)

### Medium Priority

- [ ] **Fix `_make_cube_ai` for JIT**: Replace `np.vstack`/`np.atleast_2d` with `jnp.stack` in `_make_cube_ai()` (model_set.py:70) so the Adam optimizer can trace through the full `simulate_cube` path. This is the only remaining blocker for full JIT-compiled fitting.
- [ ] **Benchmark**: Compare timing of old Cython vs new JAX cube population on GPU

### Phase 5: JAX-Accelerated Fitting [DONE]

Strategy: JAX tracer injection into existing parameter storage (same as reference implementation at `dysmalpy-jax/`). Instead of refactoring the parameter system to be immutable, create a closure `jax_loss(theta)` that directly sets `_param_value_*` attributes to JAX tracers, bypassing `__setattr__` which calls `float()`.

Steps completed:
1. Added `get_param_storage_names()` to ModelSet — returns `(comp_name, param_name) -> theta_index` mapping
2. Created `dysmalpy/fitting/jax_loss.py` — `make_jax_loss_function()` factory using `object.__setattr__` for tracer injection
3. Created `dysmalpy/fitting/jax_optimize.py` — `JAXAdamFitter` class with `jax.value_and_grad` + Adam
4. Added `make_jax_log_prob_function()` — wraps loss with JAX-traceable prior computation
5. 9 Phase 5 tests in `tests/test_jax.py` (8 pass, 1 xfail for Adam smoke test)

Geometry parameters (inc, pa, xshift, yshift) are excluded from JAX tracing since they affect array shapes.

Additional fixes during Phase 5:
- Fixed `v_circular()` in `base.py`: `jnp.where(r > 0, G*mass/r, 0.)` → `jnp.where(r > 0, G*mass/jnp.maximum(r, 1e-10), 0.)` to prevent NaN gradients from division by zero at r=0
- Changed `simulate_cube()` to return JAX arrays (replaced `np.zeros` → `jnp.zeros`, removed `np.asarray()` wrappers on `populate_cube_jax` output)
- Converted `zheight.py`: `np.exp` → `jnp.exp`
- Converted `dispersion_profiles.py`: `np.ones` → `jnp.ones`
- Added `np.asarray()` in `observation.py` after `simulate_cube()` call for numpy compatibility

### Low Priority

- [ ] **Phase 6 (Optional): JAX FFT convolution** — Replace `astropy.convolution` with `jnp.fft`-based PSF/LSF convolution
- [ ] **Remove unused imports**: Clean up any remaining `import numpy as np` in files that only use `jnp`
- [ ] **Verify pickle compatibility** of new DysmalParameter-based models for MCMC chain serialization
- [ ] **Fix `np.inf` in parameter defaults**: `light_distributions.py` uses `np.inf` — should work but may cause JAX tracing issues

### Completed (was TODO, now done)

- [x] **Complete kinematic_options.py conversion** (Phase 4 final piece): Replaced all remaining scipy dependencies with JAX-compatible implementations:
  - `scipy.optimize.newton` → JAX secant solver via `jax.lax.scan` (`_solve_adiabatic_sq`)
  - `scipy.interpolate.interp1d` → `_interp1d_extrap()` using `jnp.interp` with manual linear extrapolation via `jnp.where`
  - `scipy.special.gammaincinv` → `dysmalpy.special.gammaincinv` in `get_asymm_drift_profile()`
  - Removed all `import scipy.*` dependencies from kinematic_options.py
  - Key design decision: The secant solver evaluates the residual function using `interp(rprime, r1d, sqrt(vhalo_sq))` then squares the result, matching the original scipy behavior exactly. Using `interp(rprime, r1d, vhalo_sq)` directly gives different results when extrapolating because `(sqrt(extrap))^2 ≠ extrap(sqrt(y))` for linear extrapolation.
  - Added `import jax` for `jax.lax.scan`
- [x] **Force float64**: `tests/conftest.py` sets `jax.config.update("jax_enable_x64", True)` before all tests
- [x] **All 72 tests pass**: 27 existing tests + 45 JAX-specific tests (1 xfail) all pass
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

4. **`_make_cube_ai` uses `np.vstack` on JAX arrays**: The cube setup function `_make_cube_ai()` (model_set.py:70) uses `np.vstack` which internally calls `np.atleast_2d` on the coordinate arrays. Since `Geometry.evaluate()` returns JAX arrays (using `jnp.sin/jnp.cos`), these are JAX arrays even when geometry params are concrete. When traced under `jax.jit`, this causes a `TracerArrayConversionError`. This blocks the full Adam optimizer smoke test. Fix: convert `np.vstack` to `jnp.stack` or similar JAX-compatible operation.

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
46 tests covering:
- Special functions against scipy reference values
- DysmalParameter descriptor behavior
- Cube population correctness
- Geometry transforms
- Model computations (NFW, Sersic, DiskBulge, TwoPowerHalo)
- Utility functions
- Cube helpers
- Phase 5: Loss function correctness, gradient computation, log-prob function, Adam optimizer smoke test (xfail due to kinematic_options.py limitation)

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
                             │                             └─> Phase 6 (convolution)  [TODO]
```

*Phase 4 complete — all scipy dependencies removed from kinematic_options.py
*Phase 5 complete — loss function, log-prob, Adam optimizer all working. Full Adam smoke test blocked by `_make_cube_ai` numpy issue (separate fix needed)
