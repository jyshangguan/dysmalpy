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




---

## 18. JAXNS 2.6.9 Import Path Changes

**Problem:** JAXNS 2.6.9 changed the import path for `TerminationCondition`, causing
`ModuleNotFoundError: No module named 'jaxns.nested_sampler'`.

**Symptom:**
```
ModuleNotFoundError: No module named 'jaxns.nested_sampler'
```

**Root Cause:**
- JAXNS 2.6.9 refactored module structure
- `TerminationCondition` moved from `jaxns.nested_sampler` to top-level `jaxns` module
- Old import: `from jaxns.nested_sampler import TerminationCondition`
- New import: `from jaxns import TerminationCondition`

**Fix:** Update import statement in `dysmalpy/fitting/jaxns.py`

**Change:**
```python
# OLD (JAXNS 2.4.13):
from jaxns import NestedSampler, Model, Prior
from jaxns.nested_sampler import TerminationCondition

# NEW (JAXNS 2.6.9):
from jaxns import NestedSampler, Model, Prior, TerminationCondition
```

**Verification:**
```python
import jaxns
print('TerminationCondition' in dir(jaxns))  # True for 2.6.9+
```

**Notes:**
- `NestedSampler`, `Model`, `Prior` remain in top-level `jaxns` module
- `TerminationCondition` now also in top-level (previously in `jaxns.nested_sampler`)
- Other imports (summary, plot_diagnostics, plot_cornerplot) unchanged


---

## 19. JAXNS 2.6.9 Multi-GPU Performance Requires Explicit Device Configuration

**Problem:** After upgrading from JAXNS 2.4.13 to 2.6.9, GPU memory usage dropped from ~8GB to <500MB
and performance degraded significantly.

**Symptom:**
- GPU memory: <500MB (was ~8GB with JAXNS 2.4.13)
- GPU utilization: 0-12% (very low)
- Slow performance compared to JAXNS 2.4.13

**Root Cause:**
JAXNS 2.6.9 `NestedSampler` requires explicit `devices` parameter to distribute work across GPUs.
While `devices=None` (default) technically uses all devices, the parallelization efficiency is lower
than JAXNS 2.4.13's `DefaultNestedSampler`.

**Fix:**
Added `num_parallel_workers` parameter to `JAXNSFitter` which maps to the `devices` parameter:

```python
# In dysmalpy/fitting/jaxns.py:
if self.num_parallel_workers is not None and self.num_parallel_workers > 0:
    # User specified number of workers
    import jax
    all_devices = jax.devices()
    num_to_use = min(self.num_parallel_workers, len(all_devices))
    ns_kwargs['devices'] = all_devices[:num_to_use]
else:
    # Auto-detect: use all available devices
    import jax
    num_devices = len(jax.devices())
    logger.info(f"JAXNS: Using all {num_devices} available GPUs")
```

**Usage:**
```python
# In parameter file or demo script:
num_parallel_workers, 8  # Use 8 GPUs (1 worker per GPU)
num_parallel_workers, 16  # Use 16 workers on 8 GPUs (2 per GPU, if memory allows)
```

**Results:**
- GPU memory: 2-5GB per GPU (restored to expected levels)
- GPU utilization: ~38-40% on active GPUs
- Log message: "JAXNS: Using all 8 available GPUs"

**Notes:**
- `num_parallel_workers=None` (default) auto-detects and uses all available GPUs
- Allows >1 worker per GPU if memory permits
- JAXNS 2.6.9 uses `NestedSampler` instead of `DefaultNestedSampler` (API change)


---

## 20. JAXNS Parallelization: c (Markov Chains) vs devices (Multi-GPU)

**Understanding JAXNS 2.6.9 Parallelization:**

JAXNS has TWO levels of parallelization that work together:

### 1. `c` Parameter - Parallel Markov Chains (PER GPU)

**What it controls:**
- Number of parallel Markov chains running on EACH GPU
- This is the primary source of parallel sampling speedup
- Default: `c = 20 * D` (where D = number of dimensions/parameters)

**Example:**
- 10 parameters → default `c = 20 * 10 = 200` parallel chains
- Each GPU runs 200 parallel sampling chains simultaneously
- This is what gave fast performance on 1 GPU with JAXNS 2.4.13

**How to set in param file:**
```python
c, 150  # 150 parallel chains per GPU
```

**Recommendation:** Set `c` to match or exceed `num_live_points` for optimal efficiency.

### 2. `devices` Parameter - Multi-GPU Selection

**What it controls:**
- WHICH GPUs to use for computation
- Does NOT control parallelization speed (that's `c`)
- Default: all available GPUs

**How to control:**
- Via `num_parallel_workers` parameter (maps to `devices`)
- Or via `CUDA_VISIBLE_DEVICES` environment variable

**Examples:**
```python
# Use only 1 GPU
num_parallel_workers, 1  # Or CUDA_VISIBLE_DEVICES=0

# Use 4 GPUs
num_parallel_workers, 4  # Or CUDA_VISIBLE_DEVICES=0,1,2,3

# Use all 8 GPUs
num_parallel_workers, 8  # Or don't set (default = all)
```

### Combined Parallelization

**Maximum parallelism (8 GPUs × 150 chains = 1200 total):**
```python
c, 150                  # 150 parallel chains per GPU
num_parallel_workers, 8  # Use all 8 GPUs
```

**Single GPU with high parallelism (like JAXNS 2.4.13):**
```python
c, 150  # 150 parallel chains on 1 GPU
# Run with: CUDA_VISIBLE_DEVICES=0
```

### Performance Comparison

| Configuration | Parallel Chains | GPUs | Total Capacity |
|--------------|-----------------|------|----------------|
| Before (JAXNS 2.4.13) | 200 (default) | 1 | 200 chains |
| After fix (c=150, 1 GPU) | 150 | 1 | 150 chains |
| After fix (c=150, 8 GPUs) | 150 | 8 | 1200 chains |

**Key Insight:** The `c` parameter (parallel Markov chains) is what provides the sampling parallelization,
not the number of GPUs. GPUs just provide more hardware for running those chains.

### Migration from JAXNS 2.4.13

**Before (JAXNS 2.4.13 with DefaultNestedSampler):**
```python
# You probably didn't set c explicitly
# Default c = 20 * 10 = 200 chains
# Result: Fast performance from 200 parallel chains
```

**After (JAXNS 2.6.9 with NestedSampler):**
```python
# Set c explicitly to match desired parallelism
c, 150  # 150 parallel chains (matching num_live_points)
# Result: Similar fast performance
```

**Note:** If `c` is not set, JAXNS 2.6.9 uses the same default (20 * D), so performance should be similar.
However, setting `c = num_live_points` is recommended for optimal efficiency.


---

## 21. JAXNS 2.6.9 Does NOT Support Multi-GPU Parallelization

**Critical Finding:** JAXNS 2.6.9's `NestedSampler` does **NOT** actually parallelize across multiple GPUs.

**Symptoms:**
- Memory allocated on all 8 GPUs (~100GB total, ~12GB per GPU)
- BUT only 1 GPU actually computes (0-4% utilization on 1 GPU, 0% on others)
- Slow performance despite high memory allocation
- Log says "Using all 7/8 available GPUs" but only 1 GPU computes

**Root Cause:**
The `devices` parameter in `NestedSampler` exists but does NOT distribute computation across devices.
It only controls where memory is allocated. JAXNS still runs on a single device at a time.

**Testing:**
```
# Memory allocation:
GPU 0: 24 GB allocated, 0% utilization
GPU 1: 23 GB allocated, 0% utilization
GPU 2: 21 GB allocated, 0% utilization
GPU 3: 6 GB allocated, 0% utilization
GPU 4: 4 GB allocated, 0% utilization
GPU 5: 7 GB allocated, 0% utilization
GPU 6: 6 GB allocated, 0% utilization
GPU 7: 5 GB allocated, 4% utilization ← ONLY THIS GPU COMPUTES
```

**Solution:**
Use **single GPU** with **high `c` value** (parallel Markov chains) for performance.

```python
# CORRECT: Single GPU with parallel chains
c, 200  # 200 parallel Markov chains on 1 GPU
CUDA_VISIBLE_DEVICES=0 python demo/demo_2D_fitting_JAXNS.py

# WRONG: Multi-GPU (doesn't actually parallelize)
# Even with devices parameter, only 1 GPU computes
```

**Comparison with JAXNS 2.4.13:**
- JAXNS 2.4.13: `DefaultNestedSampler` with default `c=200`
  - Single GPU, 200 parallel Markov chains
  - Fast performance from parallel chains
  - ~8GB GPU memory usage

- JAXNS 2.6.9: `NestedSampler` with `c=150`
  - Single GPU, 150 parallel Markov chains (if only 1 GPU visible)
  - Same performance model as 2.4.13, but with different sampler
  - Multi-GPU option does NOT improve performance

**Key Insight:**
- **`c` parameter** = parallel Markov chains (actual parallelization)
- **`devices` parameter** = just memory allocation, NOT computation distribution
- Performance comes from `c`, not from number of GPUs
- Use `CUDA_VISIBLE_DEVICES` to select which single GPU to use

**Recommendation:**
Set `c` to match or exceed `num_live_points` for optimal efficiency:
```python
c, 150  # Match num_live_points=150
c, 200  # Default (20 * 10 parameters)
```

# Problem #22: JAXNS 2.6.9 parameter file comment parsing

**Date:** 2026-04-29

**Issue:** Inline comments in parameter files break the parser

**Symptoms:**
```python
# WRONG - parser reads "300      # comment" as string
c, 300      # 300 parallel chains
```

**Root Cause:** The parameter file parser splits on the first comma and takes everything after as the value, including inline comments.

**Fix:** Put comments on separate lines
```python
# CORRECT
c, 300
# c=300 gives 300 parallel Markov chains (30*n_dim for 10 params)
```

**Files Affected:** 
- `demo/demo_2D_fitting_JAXNS.py`

**Impact:** If using inline comments, the value becomes a string instead of integer, causing JAXNS to ignore it or crash.

---

# Problem #23: JAXNS 2.6.9 c parameter override behavior

**Date:** 2026-04-29

**Issue:** Setting only `c` parameter doesn't work as expected in JAXNS 2.6.9

**Symptoms:**
- Setting `c=300` results in JAXNS using `c=150`
- Setting `num_live_points=150` results in `c=150`
- Log shows "Number of Markov-chains set to: 150" instead of 300

**Root Cause:** JAXNS 2.6.9 `NestedSampler` calculates one from the other:
- If `num_live_points=X` is set, JAXNS calculates `c = X / (k + 1)` where `k=0` by default
- If `c=X` is set (and `num_live_points=None`), JAXNS calculates `num_live_points = X * (k + 1)`
- If BOTH are set, JAXNS uses the explicit values

**Fix:** Always set both `num_live_points` and `c` explicitly when you want a specific value:
```python
num_live_points, 300
c,                300
```

**Difference from JAXNS 2.4.13:**
- Old: `DefaultNestedSampler` defaulted to `c = 30 * n_dim`
- New: `NestedSampler` defaults to `c * (k + 1)` for num_live_points

**Files Affected:**
- `dysmalpy/fitting/jaxns.py`
- `demo/demo_2D_fitting_JAXNS.py`

---

# Problem #24: cuPTI library not found error

**Date:** 2026-04-29

**Issue:** JAX fails to initialize GPU with error:
```
RuntimeError: Unable to load cuPTI. Is it installed?
```

**Symptoms:**
- JAX falls back to CPU mode
- GPU memory not allocated
- Nested sampling extremely slow

**Root Cause:** cuPTI (CUDA Profiling Tools Interface) library path not in `LD_LIBRARY_PATH`

**Fix:** Add cuPTI library path before importing JAX:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

**Location:** Added to `activate_alma.sh` to be set automatically on environment activation

**Files Affected:**
- `activate_alma.sh`

**Impact:** Without this fix, JAX cannot use GPU even if CUDA is installed correctly.

---

# Problem #25: JAXNS 2.6.9 progress output buffering

**Date:** 2026-04-29

**Issue:** JAXNS progress output doesn't appear when running in background

**Symptoms:**
- Demo appears to hang at "Running nested sampling..."
- No progress output for hours
- Output only appears at completion

**Root Cause:** Python stdout buffering when running in background with `&`

**Fix:** Use `python -u` flag for unbuffered output:
```bash
python -u demo/demo_2D_fitting_JAXNS.py
```

**Alternative:** Run in foreground or use `| tee` to capture output while also seeing it

**Files Affected:**
- `demo/JAXNS_RUN_REPORT.md` (documentation)

**Impact:** Without `-u` flag, users cannot monitor progress and may think the job is stuck.

---

# Problem #26: JAXNS log file missing progress output

**Date:** 2026-04-29

**Issue:** JAXNS log file (`GS4_43501_jaxns.log`) was missing all progress information

**Symptoms:**
- Log file only contained initial setup messages
- Missing progress updates (samples, efficiency, log(Z) estimates)
- No completion information
- Console showed progress but log file didn't

**Root Cause:** JAXNS has **two** output streams:
1. **DysmalPy logger messages** → Captured by FileHandler ✅
2. **JAXNS progress output** → Goes to stdout/JAXNS logger, NOT captured ❌

JAXNS uses:
- `jax.debug.print()` for sampling progress (goes to stdout)
- `logging.getLogger('jaxns')` for setup messages (separate logger)

**Fix:** Two-part solution in `dysmalpy/fitting/jaxns.py`:

1. **Add JAXNS logger handler** (line 397-401):
```python
# Also capture JAXNS logger output
jaxns_logger = logging.getLogger('jaxns')
jaxns_handler = logging.FileHandler(output_options.f_log)
jaxns_handler.setLevel(logging.INFO)
jaxns_logger.addHandler(jaxns_handler)
```

2. **Redirect stdout during ns() call** (line 476-497):
```python
# Capture stdout/stderr to log file during ns() call
# JAXNS uses jax.debug.print() for progress, which requires stdout redirection
if output_options.f_log is not None:
    # Open file with line buffering to ensure jax.debug.print() output is captured promptly
    with open(output_options.f_log, 'a', buffering=1) as f:
        # Redirect both stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = f
        sys.stderr = f

        try:
            # Run with output captured
            termination_reason, state = ns(
                jax.random.PRNGKey(42),
                term_cond=term_cond
            )
        finally:
            # Restore stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
```

**Result:** Log file now contains:
- "Number of Markov-chains set to: 300"
- "Creating initial state with 300 live points."
- "Running uniform sampling down to efficiency threshold of 0.1."
- Multiple progress updates with Num samples, Efficiency, log(Z) estimates, etc.

**Files Affected:**
- `dysmalpy/fitting/jaxns.py` (added sys import, JAXNS logger handler, stdout redirection)

**Impact:** Users can now review complete fitting progress in log file, essential for:
- Debugging failed runs
- Performance analysis
- Audit trail of fitting process
- Monitoring progress without console access

---

# Problem #27: pip install -e . fails with Cython compilation errors

**Date:** 2026-04-29

**Issue:** Installation with `pip install -e .` fails for JAX version when Cython not installed

**Symptoms:**
```
Building editable for dysmalpy (pyproject.toml) ... error
exit code: 1
INFO:root:building 'dysmalpy.models.cutils' extension
gcc: command failed with exit code 1
error: Failed building editable for dysmalpy
```

**Root Cause:** Three separate issues preventing JAX-only installation:

1. **pyproject.toml requires Cython in build-system**
   ```toml
   requires = ["setuptools", 'wheel', 'Cython']  # Cython mandatory!
   ```

2. **setup.py has mandatory Cython import**
   ```python
   from Cython.Build import cythonize  # Fails if Cython not installed!
   ```

3. **cutils extension not marked as optional**
   ```python
   Extension("dysmalpy.models.cutils",
           sources=["dysmalpy/models/cutils.pyx"],
           # No optional=True!
           )
   ```

**Fix:** Make Cython optional throughout build system:

**1. pyproject.toml - Remove Cython requirement:**
```toml
# Before
requires = ["setuptools", 'wheel', 'Cython']

# After
requires = ["setuptools", 'wheel']  # Cython optional
```

**2. setup.py - Make Cython import optional:**
```python
# Before
from Cython.Build import cythonize

# After
try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    print("Note: Cython not installed. JAX-only installation...")
```

**3. setup.py - Make extensions optional and skip compilation:**
```python
# Add optional=True to extension
Extension("dysmalpy.models.cutils",
        sources=["dysmalpy/models/cutils.pyx"],
        optional=True,  # Make optional!
        )

# Skip all extensions when Cython unavailable
if HAS_CYTHON and os.path.exists("dysmalpy/models/cutils.pyx"):
    ext_modules = cythonize(original_ext_modules, annotate=True)
else:
    print("Installing without Cython extensions (JAX-only mode)")
    ext_modules = []
```

**Result:** Installation now works with simple:
```bash
conda create -n dysmalpy python=3.11
conda activate dysmalpy
pip install -e .  # Works without Cython!
```

**Files Affected:**
- `pyproject.toml` (removed Cython from build-system requires)
- `setup.py` (made Cython import optional, added optional=True to extensions)
- `setup.cfg` (updated dependency notes)
- `README.rst` (added Quick Start installation section)

**Impact:** Users can now install JAX version with standard `pip install -e .` workflow,
no manual Cython installation or complex dependency management required.

**Testing:**
- ✅ Fresh conda environment created
- ✅ `pip install -e .` succeeded without Cython
- ✅ All core imports work (Galaxy, ModelSet, Observation)
- ✅ JAX integration verified (JAX 0.7.2, JAXNS 2.6.9)

**Key Insight:** The JAX version doesn't need Cython (uses JAX instead of cutils.pyx),
but the build system was still configured for the old Cython-based version. Making Cython
optional throughout allows simple `pip install -e .` installation.

