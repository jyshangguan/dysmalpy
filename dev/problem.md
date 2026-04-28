# Common Problems and Pitfalls

A catalogue of issues encountered during development on the `dev_jax` branch.
Read this before debugging to avoid repeating known mistakes.

---

## 1. JAX Float32 by Default

JAX defaults to **float32** for all array operations, even when the input is
numpy float64.  `jnp.asarray(np.float64(1.0))` silently produces a float32
array.

**Symptom:** Cube data declared as float64 but actually float32.  Moment maps
computed manually show garbage values (velocity ~40,000 km/s, dispersion ~27,000
km/s at edge pixels).  SpectralCube.moment() masks some of this internally,
so it may go unnoticed in the fitting pipeline.

**Fix:** Set `JAX_ENABLE_X64=1` **before any JAX import**:

```python
import os
os.environ.setdefault('JAX_ENABLE_X64', '1')
```

This is done in `dysmalpy/__init__.py` (before `import dysmalpy.models` which
triggers the first JAX import).  **Do not** use `jax.config.update()` — by the
time `import jax` has executed, the dtype is already locked.

**Verification:**

```bash
JAX_PLATFORMS=cpu python -c "
import dysmalpy
import jax
print(jax.config.read('jax_enable_x64'))  # must be True
"
```

---

## 2. JAX/TFP/JAXNS Version Lockstep

JAX, jaxlib, and downstream packages (jaxns, tensorflow-probability) must be
installed in compatible versions.  A version mismatch causes hard-to-diagnose
import errors or silent numerical issues.

**Symptom:** `ImportError`, `RuntimeError: module compiled against API version
X but this version is Y`, or segfaults during JIT compilation.

**Known working combination (as of 2026-04):**

```
jax==0.4.38
jaxlib==0.4.38
jax-cuda12-plugin==0.4.38
jaxns==2.4.13
tensorflow-probability==0.25.0
```

**Rule:** When upgrading any one of these, check the others.  Pin all of them
together in the environment file.

---

## 3. `np.NaN` Removed in NumPy 2.x

`np.NaN` was removed.  Use `np.nan` instead.

**Symptom:** `AttributeError: module 'numpy' has no attribute 'NaN'` when
running with numpy >= 2.0.

**Fix:** `np.NaN` -> `np.nan` everywhere (19 files affected in this repo).

---

## 4. `np.int()` and `numpy.float` Removed in NumPy 2.x

`np.int` (alias for Python `int`) and `numpy.float` were removed.

**Symptom:** `AttributeError: module 'numpy' has no attribute 'int'`.

**Fix:** Replace `np.int(x)` with `int(x)`, `numpy.float` with `numpy.float64`.

---

## 5. Pickle Deserialization Fails Across Branches

Model pickles saved with one branch's code cannot be loaded on another branch
if `__setstate__` methods differ.  The pickle stores class-level state that
must match the current code exactly.

**Symptom:** `KeyError: '_input_units_strict'` (or similar) when loading a
pickle saved on `dev_jax` but loaded on `main`, or vice versa.

**Workaround:** Extract model parameters to a portable format (JSON) on the
branch where loading works, then reconstruct the model from scratch on the
other branch.  See `dev/debug_save_cube_moments.py --from-pickle` and
`--from-params` for an example.

**General rule:** Never rely on pickles for cross-branch data transfer.

---

## 6. `SpectralCube.moment()` Is Unreliable for CASA FITS Cubes

`SpectralCube.moment()` applies unit conversions that depend on the spectral
axis units (Hz vs km/s) and beam handling.  For frequency-axis cubes it returns
values inflated by the channel width in Hz (~3.3e6).

**Symptom:** Moment-0 or moment-2 values many orders of magnitude too large.

**Fix:** Compute moments manually from raw FITS data using `numpy.nansum`:

```python
mom0 = np.nansum(cube_data, axis=0)
mom1 = np.nansum(cube_data * spec_axis[None, :, None], axis=0) / mom0
```

---

## 7. Active-vs-Full Cube Path Chi2 Difference Is Not a Bug

The active-only path (`zcalc_truncate=True`) and the full path
(`zcalc_truncate=False`) can produce slightly different chi2 values because
truncation excludes marginal z-slices with negligible flux.  Both chi2 values
should independently match the MPFIT bestfit chisq.

**Symptom:** `chi2_active != chi2_full` by a small amount (< 1).

**Not a bug.**  Both paths are correct.  The active path is the default because
it uses much less memory.

---

## 8. `jnp.pad` Compatibility Issue

`jnp.pad` with `mode='constant'` has a `copy` keyword argument incompatibility
between certain JAX and NumPy version combinations.

**Symptom:** `TypeError: pad() got an unexpected keyword argument 'copy'`.

**Fix:** Use `jax.lax.pad` directly instead of `jnp.pad`.

---

## 9. JAX Multiprocessing Requires `forkserver`

JAX initializes internal thread pools at import time.  Python's default `fork`
copies this state into child processes, causing deadlocks.

**Symptom:** Hangs or deadlocks when using `multiprocess.Pool` or `emcee` with
`nCPUs > 1`.

**Fix:** Use `forkserver` start method:

```python
from multiprocess import get_context
pool = get_context('forkserver').Pool(self.nCPUs)
```

---

## 10. DysmalParameter Descriptor Pollution

`DysmalParameter.__get__` returns the class-level descriptor, not a per-instance
copy.  Setting `.tied`, `.fixed`, or `.prior` on an instance pollutes the class
descriptor for all future instances.

**Symptom:** Parameter constraints appear to "leak" between unrelated model
instances or test cases.

**Fix:** Always use `comp._get_param(name)` to read constraint state.  See
`development_note.md` section "DysmalParameter Class Descriptor Pollution" for
full details.

---

## 11. Pickle/Deepcopy Breaks `_model` Back-References

After unpickling, `DysmalParameter._model` is `None`, so `.value` returns the
default instead of the fitted value.

**Symptom:** MCMC acceptance rate = 0, or fitted parameters revert to defaults
after loading a saved model.

**Fix:** `ModelSet.__setstate__` rebinds `_model` on all parameter instances.
This is already handled in the current code.

---

## 12. MCMC Fitting Finds Different Minimum Than MPFIT

For high-z kinematic data, the chi2 landscape is multimodal.  MPFIT finds a
local minimum, MCMC explores the posterior.  They may converge to different
solutions (e.g., different inclination / effective radius combinations).

**Symptom:** MCMC and MPFIT best-fit parameters differ significantly.

**Not a bug.**  This is a feature of the data — the parameters are degenerate.
PA and velocity zero-point are usually well-constrained; inclination and
effective radius are often degenerate.

---

## 13. GPU OOM for Large Cubes

Cubes with `oversample=3` and large FOV produce 603^3 coordinate grids (~1.75 GB
each in float64 on GPU).  Multiple intermediate arrays push GPU memory past 24 GB.

**Symptom:** `RuntimeError: Out of memory` during `simulate_cube()`.

**Fix:** `zcalc_truncate=True` (default) uses the active-only path, which keeps
intermediate arrays on CPU (numpy) and only propagates active z-slices through
JAX.  Peak GPU memory drops from ~18 GB to < 100 MB.

---

## 14. JAXNS Version Differences (Static vs Dynamic Nested Sampling)

JAXNS 2.4.13 (our version) uses `DefaultNestedSampler` with **static** nested sampling
(fixed number of live points), while JAXNS 2.6.7+ uses `NestedSampler` with
**dynamic** nested sampling (adjusts live points during sampling).

**Symptom:** Missing "zigzag nlive" pattern in first row of diagnostic plots.

**Not a bug.**  Static nested sampling (JAXNS 2.4.13) produces a flat line for
`num_live_points` because the number is fixed.  Dynamic nested sampling (JAXNS
2.6.7+) shows the characteristic zigzag pattern as live points are adjusted
during sampling.

**Implication:** Diagnostic plots from JAXNS 2.4.13 will look different from
documentation examples (which use 2.6.7+).  Both are correct for their respective
versions.

**Solution:** Either accept the 2.4.13 behavior (static is fine for most use cases)
or upgrade to JAXNS 2.6.7+ if dynamic sampling features are needed.

---

## 15. Wide Prior Ranges Cause Inefficient JAXNS Sampling

JAXNS explores the full prior volume by design.  Wide prior ranges that include
unphysical parameter combinations cause JAXNS to waste computation exploring
regions with extremely poor likelihood values.

**Symptom:** Large percentage (20-30%) of JAXNS samples have very bad likelihood
(log_L < -200), with worst cases like log_L = -60,000 (chi-squared ≈ 120,000).

**Example:**
```python
# Too wide - causes 23% bad samples
total_mass_prior_bounds: 10.0 13.0  # 3 dex range
inc_prior_bounds: 42.0 82.0         # 40 degree range

# Better - reduces bad samples to <5%
total_mass_prior_bounds: 11.5 12.5  # 1 dex range
inc_prior_bounds: 55.0 70.0         # 15 degree range
```

**Not a bug.**  JAXNS is working correctly - it's exploring the full prior volume
as designed.  The bad samples don't affect the final results (best-fit parameters
are still good).

**Impact:**
- Wasted computation time
- Slower convergence
- Abnormal diagnostic plots (weight evolution in third row)

**Fix:** Use narrower prior bounds based on:
1. Domain knowledge / physical constraints
2. MPFIT results ± uncertainty margin
3. Literature values for similar systems

**Recommendation:** After an initial MPFIT run, update the JAXNS priors to be
centered on the MPFIT best-fit values with ±20-50% margins.  This dramatically
improves efficiency without excluding true posterior support.


---

## 16. JAX >=0.7.0 Requires tfp-nightly

**Problem:** The stable `tensorflow-probability` package (0.25.0) is incompatible with
JAX >=0.7.0 due to deprecated API removal (`jax.interpreters.xla.pytype_aval_mappings`).

**Symptom:**
```
AttributeError: jax.interpreters.xla.pytype_aval_mappings was deprecated in JAX v0.5.0
and removed in JAX v0.7.0. jax.core.pytype_aval_mappings can be used as a replacement
in most cases.
```

**Root Cause:**
- JAX 0.7.0 removed deprecated APIs
- `tensorflow-probability` 0.25.0 still uses the deprecated APIs
- No new stable `tensorflow-probability` release fixes this

**Fix:** Use `tfp-nightly` instead of `tensorflow-probability` for JAX >=0.7.0

**Why tfp-nightly works:**
- tfp-nightly is actively maintained to support the latest JAX releases
- Confirmed by TFP maintainers: "TFP nightly is intended to work with the latest JAX release"
- JAXNS 2.6.9+ requires tfp-nightly for JAX >=0.6.0
- Successfully tested with JAX 0.7.0 + tfp-nightly + JAXNS 2.6.9

**Known Working Combination (as of 2026-04-28):**
```
jax==0.7.2
jaxlib==0.7.2
jaxns==2.6.9
tfp-nightly
```

**Migration:**
1. Uninstall `tensorflow-probability`
2. Install `tfp-nightly`
3. Update `setup.cfg` to use `tfp-nightly` instead of `tensorflow-probability==0.25.0`

**Notes:**
- tfp-nightly is a development build, but actively maintained
- Recommended by TensorFlow Probability team for use with latest JAX
- All dysmalpy tests pass with tfp-nightly
- JAXNS API changed: `DefaultNestedSampler` → `NestedSampler` (internal only)

**References:**
- tensorflow/probability#1994 - TFP incompatibility with JAX 0.7.0
- Joshuaalbert/jaxns#235 - JAXNS confirms tfp-nightly works with JAX 0.7.0


---

## 17. JAX 0.7.2 GPU Support Requires CUPTI Library

**Problem:** JAX 0.7.2 with `jax-cuda12-plugin` fails to initialize GPU backend with error
"Unable to load cuPTI" even when CUDA is properly installed.

**Symptom:**
```
RuntimeError: Unable to load cuPTI. Is it installed?
RuntimeError: jaxlib/cuda/versions_helpers.cc:48: operation cuptiGetVersion(&version) failed:
Unknown CUPTI error 999. This probably means that JAX was unable to load cupti.
```

**Root Cause:**
- CUPTI (CUDA Profiling Tools Interface) is required by JAX 0.7.2 for GPU support
- CUPTI is installed in `/usr/local/cuda-12.4/extras/CUPTI/lib64/` but not in the default library path
- JAX cannot find `libcupti.so` without `LD_LIBRARY_PATH` pointing to the CUPTI directory

**Fix:** Add CUPTI library to `LD_LIBRARY_PATH` before running Python

**Temporary fix (current session only):**
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

**Permanent fix (add to `~/.zshrc`):**
```bash
# CUDA CUPTI library path for JAX 0.7.2 GPU support
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

**Verification:**
```python
import jax
print(f'JAX backend: {jax.devices()[0].platform}')
# Should print: "gpu" instead of "cpu"
```

**Known Working Setup (as of 2026-04-29):**
```
jax==0.7.2
jaxlib==0.7.2
jax-cuda12-plugin==0.7.2
CUDA 12.4 with CUPTI in /usr/local/cuda-12.4/extras/CUPTI/lib64/
LD_LIBRARY_PATH includes /usr/local/cuda-12.4/extras/CUPTI/lib64/
```

**Notes:**
- This is specific to JAX 0.7.2 + CUDA 12.4
- CUPTI is part of the CUDA toolkit but in a non-standard location
- Without this fix, JAX falls back to CPU (still works but slower)
- Multiple GPU devices will be available once CUPTI is loaded (e.g., 8 CUDA devices)


