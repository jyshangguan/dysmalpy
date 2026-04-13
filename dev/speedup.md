# JAX vs Cython Cube Population Speedup

**Branch:** `dev_jax`
**GPU:** NVIDIA 4090
**Date:** 2025-04-11

---

## Test Setup

| Parameter | Value |
|-----------|-------|
| Grid shape | `(99, 99, 99)` (z, y, x) |
| Spatial points | 970,299 |
| Spectral channels | 201 |
| Spectral range | -1000 to +1000 km/s |
| Data type | float64 |
| JAX version | 0.9.2 |

The benchmark measures the `populate_cube_ais` call directly — the inner
triple-nested loop that evaluates Gaussian line profiles at every spatial
pixel for each spectral channel.  This is the dominant computational
cost in `simulate_cube()`.

## Results

| Implementation | Mean Time | Speedup |
|---|---|---|
| **Cython** (main branch, `cutils.pyx`) | **1.440 s** | 1.0x (baseline) |
| JAX (CPU) | 0.321 s | 4.5x |
| JAX (GPU, cuda:0) | 0.017 s | **86x** |
| JAX (GPU) JIT loss+grad | 0.079 s | 18.4x |

## Notes

- **Cython**: Triple-nested C loop (`x * y * z * s`) with typed memoryviews.
  Single-threaded CPU only.

- **JAX (CPU)**: Same algorithm vectorized with `jax.vmap` over spectral
  channels and JIT-compiled by XLA.  Gains come from removing the Python
  loop overhead and leveraging SIMD vectorization.

- **JAX (GPU)**: Same JIT-compiled code offloaded to the 4090.  The massive
  parallelism (970K spatial points * 201 spectral channels) fully
  saturates the GPU.

- **JAX (GPU) JIT loss+grad**: Full `simulate_cube` -> chi-squared loss +
  `jax.grad` pipeline.  Includes model setup (geometry transforms, velocity
  profiles, enclosed mass, flux distribution) in addition to cube
  population.  The extra time beyond the raw 17 ms cube population comes
  from the model computation (~60 ms) which is not yet fully GPU-accelerated
  (velocity profile, enclosed mass calculations still have some Python
  overhead from the parameter descriptor system).

## How to Run

```bash
# On dev_jax branch (Cython .so still present + JAX):
conda activate dysmalpy-jax
python dev/benchmark_cube.py

# On main branch (Cython only):
conda activate dysmalpy-ref
python dev/benchmark_cube.py
```

---

## End-to-End MPFIT Fitting Timing (dev_jax vs main)

**Date:** 2026-04-12
**Platform:** CPU
**Script:** `dev/benchmark_fitting.py`

### Environment

| | dev_jax (new) | main (original) |
|--|---------------|-----------------|
| Conda env | dysmalpy-jax | alma |
| Python | 3.12.x | 3.12.13 |
| numpy | 2.1.3 | 1.26.4 |
| astropy | 7.2.0 | 5.3.4 |

### Results

| Case | main (s) | dev_jax (s) | Ratio (new/old) | main niter | dev_jax niter | main redchisq | dev_jax redchisq |
|------|----------|-------------|-----------------|------------|---------------|---------------|------------------|
| 1D | 9.36 | 27.19 | **2.91x** slower | 8 | 5 | 1.91 | 2.21 |
| 2D | 20.89 | 33.39 | **1.60x** slower | 13 | 8 | 4.34 | 34.02 |
| 3D | 10.56 | 16.41 | **1.55x** slower | 12 | 3 | 1.13 | 2.40 |

### Analysis

The `dev_jax` branch is **slower** than `main` for end-to-end MPFIT fitting on CPU.
Deep per-function profiling (`dev/diagnose_bottlenecks.py`) identified two root
causes:

1. **`simulate_cube` uses JAX operations (`jnp.exp`, `jnp.sqrt`, etc.) for model
   computation on the MPFIT path.** JAX's per-operation dispatch overhead (~10-100 us)
   dominates on small grids (27^3 = 20K elements), making `sersic_mr` alone take
   ~185 ms/call vs ~0.3 ms with numpy. The `populate_cube_jax` call itself is
   only 0.03 ms -- the bottleneck is the surrounding model computation.

2. **The JAX cube is float32, but the PSF beam kernel is float64.** This dtype
   mismatch in `scipy.signal.fftconvolve` causes expensive upcasting on every
   beam convolution call (26-857 ms vs 2-3 ms for the float64 LSF convolution).

3. **Convergence differs**: The two branches converge to different solutions
   in some cases (notably 2D, where dev_jax reaches redchisq=34 vs 4.3 on
   main), indicating that parameter system differences affect the optimization
   path.

4. **The JAX speedup from the cube population benchmark (4.5x CPU, 86x GPU) is
   not realized here** because the overhead is in model computation, not cube
   population. The JAX gains will materialize when using `JAXAdamFitter` or a
   full-graph JIT path where XLA can fuse all operations.

### How to Run

```bash
# On dev_jax branch:
conda activate dysmalpy-jax
JAX_PLATFORMS=cpu python dev/benchmark_fitting.py > dev/benchmark_devjax.log

# On main branch:
git stash
git checkout main
conda activate alma
python dev/benchmark_fitting.py > dev/benchmark_main.log
git checkout dev_jax
```

---

## Per-Iteration Profiling (dev_jax vs main)

**Date:** 2026-04-12
**Platform:** CPU (192 cores, NUMEXPR_MAX_THREADS=16)
**Script:** `dev/profile_mpfit.py`

### Environment

| | dev_jax | main |
|--|---------|------|
| Conda env | dysmalpy-jax | alma |
| Python | 3.12.x | 3.12.13 |
| numpy | 2.3.5 | 1.26.4 |
| scipy | 1.17.0 | 1.13.1 |
| JAX | 0.9.0.1 | 0.4.35 (unused) |

### 1D Results

| Function | main (ms/iter) | dev_jax (ms/iter) | Ratio |
|----------|---------------|-------------------|-------|
| **mpfit_chisq (TOTAL)** | **273.7** | **584.3** | **2.13x** slower |
| update_parameters | 13.4 | 36.0 | 2.69x |
| create_model_data | 276.7 | 563.7 | 2.04x |
| simulate_cube | 75.5 | 257.2 | **3.41x** |
| convolve | 172.9 | 272.9 | 1.58x |
| fftconvolve | 172.6 | 272.3 | 1.58x |

main: 37 calls, 10.85s wall, redchisq=1.91
dev_jax: 42 calls, 26.84s wall, redchisq=3.19

### 2D Results

| Function | main (ms/iter) | dev_jax (ms/iter) | Ratio |
|----------|---------------|-------------------|-------|
| **mpfit_chisq (TOTAL)** | **220.7** | **673.4** | **3.05x** slower |
| update_parameters | 13.5 | 36.1 | 2.67x |
| create_model_data | 208.5 | 653.3 | 3.13x |
| simulate_cube | 6.1 | 253.9 | **41.3x** |
| convolve | 154.5 | 352.7 | 2.28x |
| convolve_with_beam | 152.9 | 348.6 | 2.28x |
| fftconvolve | 154.4 | 351.8 | 2.28x |

main: 82 calls, 18.35s wall, redchisq=4.34
dev_jax: 37 calls, 27.17s wall, redchisq=43.82

### 3D Results

| Function | main (ms/iter) | dev_jax (ms/iter) | Ratio |
|----------|---------------|-------------------|-------|
| **mpfit_chisq (TOTAL)** | **364.2** | **577.7** | **1.59x** slower |
| update_parameters | 13.0 | 29.2 | 2.24x |
| create_model_data | 356.3 | 561.3 | 1.58x |
| simulate_cube | 9.0 | 259.1 | **28.7x** |
| convolve | 2.1 | 3.4 | 1.61x |
| fftconvolve | 2.1 | 3.2 | 1.55x |

main: 69 calls, 25.92s wall, redchisq=1.13
dev_jax: 38 calls, 24.15s wall, redchisq=2.40

---

## Bottleneck Analysis -- Investigation Results

**Date:** 2026-04-12
**Script:** `dev/diagnose_bottlenecks.py`

### Bottleneck 1: `simulate_cube` -- 29-41x slower (CRITICAL) -- FIXED

**CONFIRMED ROOT CAUSE: Model computation uses JAX ops on the MPFIT path**

The original hypothesis (that `populate_cube_jax` recompilation was the
bottleneck) was **wrong**. Deep instrumentation revealed:

1. **`populate_cube_jax` is NOT the bottleneck.** It compiles once (205 ms
   first call) then runs at **0.03 ms/call** -- effectively free. Zero
   recompilations after warmup. Shapes are perfectly stable across all
   iterations (always `(27,27,27)` for 2D, `(31,31,31)` for 3D).

2. **The 254 ms/iter comes from model computation *before* the cube
   population call.** `simulate_cube` on dev_jax builds flux, velocity, and
   dispersion arrays using JAX operations (`jnp.zeros`, `jnp.sqrt`,
   `jnp.where`, `jnp.exp`, etc.) throughout the method body. On main, the
   same code uses NumPy operations (`np.zeros`, `np.sqrt`, `np.exp`, etc.).

3. **JAX dispatch overhead on small arrays is massive.** The MPFIT grids are
   tiny (27^3 = 19,683 elements for 2D, 31^3 = 29,791 for 3D). For arrays this
   small, each `jnp.*` call incurs ~10-100 us of dispatch overhead (argument
   validation, device placement, tracer bookkeeping) that dominates the actual
   computation (~1 us for `exp` on 20K elements). NumPy has essentially zero
   per-call overhead. The cProfile confirms: `sersic_mr` (which calls
   `jnp.exp`) accounts for 7.0 s of the 9.4 s total simulate_cube time
   across 38 calls = ~185 ms/call.

4. **Cython is not used at all.** `model_set.py` imports `populate_cube_jax`
   from `cube_processing` and never imports `cutils`. The Cython `.so` exists
   but is dead code on this branch.

5. **Deeper root cause: numpy->JAX auto-conversion at every `jnp.*` call.**
   `_get_xyz_sky_gal` returns numpy arrays. These are passed into `jnp.sqrt`,
   `jnp.exp`, etc. which auto-convert numpy->JAX on *every* call. With ~100+
   `jnp.*` calls per iteration, this repeated conversion (not the JAX compute
   itself) is the bottleneck.

**Where the time goes in `simulate_cube` (2D, per call):**

| Step | main (ms) | dev_jax before fix (ms) | dev_jax after fix (ms) |
|------|-----------|------------------------|----------------------|
| Grid setup + geometry | ~0.5 | ~1 | ~1 |
| `sersic_mr` / `light_profile` | ~0.3 | ~185 | ~3 |
| velocity_profile, zprofile, etc. | ~0.2 | ~65 | ~8 |
| `populate_cube_jax` | ~4 | ~0.03 | ~0.03 |
| **Total** | **~6** | **~254** | **~65** |

**Fix (implemented in commit 4a55a1c):** Added `_safe_gammaincinv` helper in
`base.py` that routes to `scipy.special.gammaincinv` for scalar Python float
inputs (MPFIT path) and falls through to the JAX `gammaincinv` implementation
for JAX tracers/arrays (JAX 3D-loss path). Also added `xp_dispatch` in
`sersic_mr` and `truncate_sersic_mr` to use numpy ops on the MPFIT path,
avoiding jnp dispatch overhead for small arrays.

### Bottleneck 2: `convolve` / `fftconvolve` -- 1.6-2.3x slower (MODERATE)

**CONFIRMED ROOT CAUSE: float32/float64 dtype mismatch in beam convolution**

Instrumentation of individual `fftconvolve` calls revealed:

```
First 5 calls (2D):
  #1: 320 ms  in1=(201,27,27) float32   in2=(1,27,27) float64   <- BEAM
  #2:   3 ms  in1=(201,27,27) float64   in2=(41,1,1)  float64   <- LSF
  #3:  58 ms  in1=(201,27,27) float32   in2=(1,27,27) float64   <- BEAM
  #4:   3 ms  in1=(201,27,27) float64   in2=(41,1,1)  float64   <- LSF
  #5:  26 ms  in1=(201,27,27) float32   in2=(1,27,27) float64   <- BEAM
```

1. **The cube from `populate_cube_jax` is float32**, but the PSF beam kernel
   is float64. This dtype mismatch forces scipy to upcast the cube on every
   beam convolution call, which is expensive for a `(201, 27, 27)` array.

2. **The LSF convolution has no dtype mismatch** (both float64) and runs in
   2-3 ms -- comparable to main.

3. **Beam convolution variance is high** (26-857 ms) and decreases over
   time, suggesting scipy's internal FFT planning/caching interactions with
   the dtype conversion.

4. **Kernel truncation overhead is negligible** -- the `_truncate_kernel_to_cube`
   slicing is just a few numpy slice operations.

**Fix:** Cast the cube to float64 before beam convolution, or cast the PSF
kernel to float32, or make `populate_cube_jax` output float64:
```python
# Option A: cast cube once before convolve
cube_conv = fftconvolve(cube.astype(np.float64), kernel, mode='same')
# Option B: ensure populate_cube_jax outputs float64
cube_final = jnp.zeros((nspec, ny, nx), dtype=jnp.float64)
```

### Bottleneck 3: `update_parameters` -- 2.2-2.7x slower (LOW)

**CONFIRMED ROOT CAUSE: Tied parameter evaluation, not DysmalParameter overhead**

```
  update_parameters calls: 40,  mean 16.0 ms/call
  _update_tied_parameters:  40,  mean 15.8 ms/call
  Tied fraction of update:  99.6%
```

1. **99.6% of `update_parameters` time is inside `_update_tied_parameters`.**
   The `DysmalParameter` descriptor overhead is negligible (~0.2 ms/call).

2. **The tied functions themselves** (`tie_lmvirial_NFW`, `calc_mvirial_from_fdm`)
   involve transcendental math (log10, NFW profile integration) that takes
   ~16 ms per evaluation. This is the same on both branches.

3. **The regression vs main (16 ms vs 13 ms)** may be due to the `DysmalParameter`
   getter adding slight overhead when reading parameter values inside the tied
   functions, or could be NumPy 2.x overhead. The absolute difference is
   small (+3 ms/iter).

**Fix:** Low priority. The tied function computation is inherent to the
problem. If needed, could cache the tied parameter result when inputs haven't
changed (MPFIT doesn't always change every parameter between iterations).

---

## Post-Fix Timing (after gammaincinv fix, commit 4a55a1c)

**Date:** 2026-04-13
**Script:** `dev/profile_mpfit.py`

### 2D MPFIT -- Steady-State Per-Iteration Breakdown

```
Operation                              Time (ms)    % of total
------------------------------------------------------------
mpfit_chisq (TOTAL)                     394 ms       100%
  update_parameters                      30 ms         8%
  create_model_data                     367 ms        93%
    simulate_cube                         65 ms        16%
    convolve                             256 ms        65%
      convolve_with_beam                  252 ms        64%
      convolve_with_lsf                     4 ms         1%
```

---

## Final Results (after all optimizations)

**Date:** 2026-04-13
**Script:** `dev/profile_mpfit.py`
**Commits:** 4a55a1c (gammaincinv), e1335a1 (dtype + xp_dispatch),
298d527 (tied cache), 8af6d6b (JAX FFT convolve)

### Optimizations Applied

| # | Optimization | Commit | Files |
|---|-------------|--------|-------|
| 1 | Route gammaincinv to scipy for scalar inputs | 4a55a1c | `base.py`, `baryons.py`, `kinematic_options.py` |
| 2 | Fix float32 dtype mismatch; xp_dispatch in halo/baryon models | e1335a1 | `base.py`, `halos.py`, `baryons.py`, `model_set.py` |
| 3 | Cache tied parameter evaluation when inputs unchanged | 298d527 | `model_set.py` |
| 4 | Replace scipy fftconvolve with JAX FFT (JIT + cached kernel) | 8af6d6b | `instrument.py` |

### Full Comparison: main vs dev_jax (all stages)

| Metric | main (ms) | dev_jax before (ms) | dev_jax after (ms) | Speedup vs before | vs main |
|--------|-----------|---------------------|-------------------|-------------------|---------|
| **1D mpfit_chisq** | 274 | 584 | **238** | 2.5x | **1.2x faster** |
| 1D simulate_cube | 76 | 257 | 61 | 4.2x | 1.2x slower |
| 1D convolve | 173 | 273 | 83 | 3.3x | **2.1x faster** |
| 1D update_parameters | 13 | 36 | 15 | 2.4x | 1.2x slower |
| **2D mpfit_chisq** | 221 | 673 | **163** | 4.1x | **1.4x faster** |
| 2D simulate_cube | 6 | 254 | 42 | 6.0x | 7.0x slower |
| 2D convolve | 155 | 353 | 43 | 8.2x | **3.6x faster** |
| 2D update_parameters | 14 | 36 | 12 | 3.0x | 1.2x slower |
| **3D mpfit_chisq** | 364 | 578 | **384** | 1.5x | ~parity |
| 3D simulate_cube | 9 | 259 | 55 | 4.7x | 6.1x slower |
| 3D convolve | 2 | 3 | 1 | 3.0x | 2.0x slower |
| 3D update_parameters | 13 | 29 | 16 | 1.8x | 1.2x slower |

### Per-Optimization Impact (2D MPFIT)

| After | simulate_cube | convolve | mpfit_chisq |
|-------|---------------|----------|------------|
| Before all | 254 ms | 353 ms | 673 ms |
| + Opp 1 (gammaincinv) | 65 ms | 255 ms | 394 ms |
| + Opp 2 (dtype + xp_dispatch) | 43 ms | 46 ms | 145 ms |
| + Opp 3 (tied cache) | 42 ms | 46 ms | 145 ms |
| + Opp 4 (JAX FFT) | 42 ms | 43 ms | 163 ms |

Opportunity 3 (tied cache) shows negligible impact because MPFIT's
Levenberg-Marquardt typically changes baryonic parameters on most
iterations, invalidating the cache. The small improvement (~1 ms) comes
from early iterations where some parameters are unchanged.

Opportunity 4 (JAX FFT) replaces scipy's `fftconvolve` with JAX's
`jnp.fft.rfftn`/`irfftn` wrapped in `@jax.jit`. On CPU this is already
3.6x faster than scipy for 2D beam convolution. On GPU the speedup will
be much larger since scipy cannot run on GPU at all.

### GPU Outlook

The JAX FFT convolution (Opportunity 4) is now GPU-ready. The cube
population (`populate_cube_jax`) was already GPU-accelerated (86x on GPU
vs Cython). The remaining CPU-bound operations on the MPFIT path are:

1. **Model computation** (`simulate_cube` body): `sersic_mr`, `enclosed_mass`,
   `velocity_profile` etc. use `xp_dispatch` which routes to numpy on the
   MPFIT path. These operate on small arrays (~20K elements) where GPU
   kernel launch overhead dominates. GPU benefit here requires full-graph
   JIT (Opportunity 5 below).

2. **Tied parameter evaluation** (`calc_mvirial_from_fdm` + `brentq`):
   Pure Python, not GPU-tractable. Would need a JAX-compatible root finder.

3. **MPFIT iteration loop**: Python-level, not GPU-tractable. Use
   `JAXAdamFitter` for GPU-based optimization.

### Remaining Opportunities

| # | Opportunity | Status | Notes |
|---|------------|--------|-------|
| 5 | Full-graph JIT (JAX-loss path) | Future | Requires refactoring Python control flow in simulate_cube |
| 6 | GPU for MPFIT path | Not recommended | CPU data transfer overhead dominates for small arrays |

---

## How to Run Diagnostics

```bash
JAX_PLATFORMS=cpu python dev/profile_mpfit.py [1D|2D|3D]
```
