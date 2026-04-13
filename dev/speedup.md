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

### Comparison: main vs dev_jax (after fix) vs dev_jax (before fix)

| Function | main (ms) | dev_jax after fix | dev_jax before fix |
|----------|-----------|-------------------|-------------------|
| simulate_cube | 6 | **65** (11x) | 254 (42x) |
| convolve | 155 | **255** (1.6x) | 353 (2.3x) |
| mpfit_chisq | 221 | **394** (1.8x) | 673 (3.0x) |

The gammaincinv fix eliminated ~190 ms/iter from `simulate_cube`. The remaining
gap vs main (65 ms vs 6 ms) is from residual JAX dispatch overhead in
`velocity_profile`, `light_profile`, and other model computation functions
that still use `jnp.*` ops. The convolve gap (255 ms vs 155 ms) is the float32
dtype mismatch.

---

## Plan: Further MPFIT Speedup Opportunities

**Date:** 2026-04-13

### Architecture Boundary

The key constraint is that `simulate_cube` must remain fully JAX-compatible
because it is called under `jax.jit` on the 3D JAX-loss path (where parameters
are JAX tracers). Any fix on the MPFIT path must not break the JAX-loss path.

```
+---------------------------------------------------+
|  FITTING LAYER (numpy/Python)                     |
|  - MPFIT iteration, parameter updates            |
|  - JAX Adam optimizer                            |
|  - I/O, plotting, instrument setup               |
+---------------------------------------------------+
|  BRIDGE: DysmalParameter.value                   |
|  - Python float on MPFIT path                   |
|  - JAX tracer on JAX 3D-loss path               |
|  - Python float on JAX 1D-loss path             |
+---------------------------------------------------+
|  MODEL COMPUTATION (JAX, must be traceable)      |
|  - simulate_cube and everything it calls        |
|  - sersic_mr, velocity_profile, enclosed_mass   |
|  - populate_cube_jax, convolve_cube_jax         |
|  - gammaincinv (JAX @custom_jvp + lax.scan)     |
+---------------------------------------------------+
```

### Opportunity 1: Eliminate float32 dtype mismatch in convolve

**Impact:** ~100 ms/iter (2D), bringing convolve from 255 ms to ~155 ms (main parity)
**Risk:** LOW
**Complexity:** LOW

The `cube_final` in `simulate_cube` is created as:
```python
cube_final = jnp.zeros((nspec, ny_sky_samp, nx_sky_samp))  # default float32!
```
`populate_cube_jax` / `populate_cube_jax_ais` then adds float32 results into it.
The beam kernel is float64 from `astropy`'s Gaussian kernel. When scipy
`fftconvolve` receives float32 cube + float64 kernel, it upcasts to float64
internally on every call.

**Fix:** Change `cube_final` to float64 in `simulate_cube`:
```python
cube_final = jnp.zeros((nspec, ny_sky_samp, nx_sky_samp), dtype=jnp.float64)
```
This is safe because:
- On the JAX-loss path, the rest of the computation already produces float64
  intermediates (parameter values are float64).
- `populate_cube_jax` / `populate_cube_jax_ais` will automatically upcast
  their float32 computations to match the accumulator dtype.
- The chi-squared computation downstream expects float64 precision.

**Files to modify:** `dysmalpy/models/model_set.py` line 1400

**Alternative:** Cast to float64 in `convolve_with_beam` only:
```python
cube_conv = fftconvolve(np.asarray(cube, dtype=np.float64), kernel.copy(), mode='same')
```
This is even safer (only affects convolve, not model computation) but adds a
copy per iteration.

### Opportunity 2: Use `xp_dispatch` in `velocity_profile` and halo model methods

**Impact:** ~50 ms/iter (2D), bringing simulate_cube from 65 ms to ~15 ms
**Risk:** MEDIUM (must not break JAX-loss path)
**Complexity:** MEDIUM

The profile agent found that in steady state, `velocity_profile` + `vcirc_sq`
takes ~7 ms/call on the MPFIT path. This calls `jnp.sqrt`, `jnp.where`,
`jnp.log`, `jnp.abs` in NFW/Burkert/Einasto `enclosed_mass` methods. These
are called with numpy arrays (since coordinates from `_get_xyz_sky_gal` are
numpy) but use `jnp.*` which has dispatch overhead.

**Safe pattern (already proven in `sersic_mr` / `truncate_sersic_mr`):**
```python
def enclosed_mass(self, r):
    xp = xp_dispatch(r)
    rvirial = self.calc_rvir()
    rs = rvirial / self.conc
    aa = 4. * xp.pi * rho0 * rvirial**3 / self.conc**3
    bb = xp.abs(xp.log((rs + r) / rs) - r / (rs + r))
    return aa * bb
```

**Constraint:** This is safe ONLY when all inputs are guaranteed to be numpy
arrays on the MPFIT path and JAX tracers on the JAX-loss path. The `r`
parameter (radius array) satisfies this: it comes from `rgal * to_kpc` where
`rgal = jnp.sqrt(xgal**2 + ygal**2)` -- on the JAX-loss path `xgal`/`ygal`
are JAX tracers, so `rgal` is a tracer. On the MPFIT path they're numpy, so
`rgal` is numpy. The `xp_dispatch(r)` check is a safe discriminator.

**Problem cases:** Methods that read parameter values like `self.conc`,
`self.mvirial` via the descriptor. On the JAX-loss path these are plain Python
floats (they're NOT tracers -- only the JAX loss function parameters are
tracers, not the model attributes). So `rvirial / self.conc` produces a
Python float on both paths. The `xp.pi`, `xp.abs` etc. calls then just need
to handle the output type correctly.

**Files to modify:**
- `dysmalpy/models/halos.py`: `NFW.enclosed_mass`, `Burkert.enclosed_mass`,
  `Einasto.enclosed_mass`, `DekelZhao.enclosed_mass`, `TwoPowerHalo.enclosed_mass`
- `dysmalpy/models/base.py`: `v_circular` (already uses `jnp` only, small impact)
- `dysmalpy/models/baryons.py`: `Sersic.light_profile`, `ExpDisk.light_profile`

**Estimated impact per function (2D, numpy array ~20K elements):**

| Function | jnp dispatch overhead | With xp_dispatch |
|----------|----------------------|-----------------|
| NFW.enclosed_mass | ~2 ms | ~0.05 ms |
| Burkert.enclosed_mass | ~2 ms | ~0.05 ms |
| Sersic.light_profile | ~1 ms | ~0.03 ms |
| velocity_profile (jnp.sqrt) | ~1 ms | ~0.03 ms |
| Total | ~7 ms | ~0.2 ms |

### Opportunity 3: Pre-compute and cache tied parameters when inputs unchanged

**Impact:** ~22 ms/iter (2D), bringing update_parameters from 30 ms to ~8 ms
**Risk:** LOW
**Complexity:** LOW

The `_update_tied_parameters` call evaluates `calc_mvirial_from_fdm` every
iteration. This function calls `scipy.optimize.brentq` (which internally
calls `halo.vcirc_sq` ~15 times with temporary halo copies). Each call
involves `halo.copy()` (deepcopy), `halo.__setattr__`, and `halo.vcirc_sq`.

MPFIT's Levenberg-Marquardt often changes only 1-2 parameters per iteration.
If the baryonic parameters haven't changed, the tied NFW virial mass is
unchanged and the brentq solve can be skipped.

**Fix:** Cache the input parameters of the tied function and skip evaluation
if they're unchanged:
```python
def _update_tied_parameters(self):
    for cmp in self.components:
        comp = self.components[cmp]
        for pp in list(getattr(comp, 'param_names', [])):
            param = getattr(comp, pp, None)
            if param is None:
                continue
            tied_fn = getattr(param, 'tied', False)
            if callable(tied_fn):
                # Check cache key (input params)
                cache_key = _tied_cache_key(self, cmp, tied_fn)
                if cache_key == self._tied_cache.get((cmp, pp)):
                    continue
                new_value = tied_fn(self)
                self.set_parameter_value(cmp, pp, new_value,
                                         skip_updated_tied=True)
                self._tied_cache[(cmp, pp)] = cache_key
```

**Files to modify:** `dysmalpy/models/model_set.py`

### Opportunity 4: JIT-compile the entire simulate_cube + convolve pipeline

**Impact:** Potentially 3-5x speedup on GPU, but uncertain on CPU
**Risk:** HIGH (requires careful design to handle dynamic Python control flow)
**Complexity:** HIGH

The ultimate optimization is to wrap the entire `simulate_cube` body (geometry,
model computation, cube population, convolution) in a single `@jax.jit` function.
This would allow XLA to fuse all operations and eliminate Python dispatch overhead
entirely.

**Why this is hard:**
- `simulate_cube` has extensive Python control flow (if/else branches for
  `transform_method`, `zcalc_truncate`, component types, higher-order
  components, extinction, dimming)
- Geometry objects have Python attributes that are read at runtime
- The `for cmp in tracer_lcomps` loop iterates over a Python dict
- Some model methods (like `calc_mvirial_from_fdm`) use `scipy.optimize.brentq`
  which is not JAX-traceable

**Practical approach for the MPFIT path:** Don't try to JIT the whole thing.
Instead, ensure each individual operation is fast by using numpy (Opportunities
1-3 above). The MPFIT path is inherently Python-loop-based and will never
benefit from full-graph JIT.

**For the JAX-loss path (JAXAdamFitter):** The full-graph JIT is already the
goal. The current JAX-loss path passes parameters as flat JAX arrays and
reconstructs the model inside the loss function. The bottleneck there is the
`halo` model computation (NFW enclosed_mass uses `jnp.log`, `jnp.abs` which
are already JAX-compatible). GPU offloading of the full pipeline is where the
real gains are.

### Opportunity 5: Replace scipy fftconvolve with numpy FFT on CPU

**Impact:** ~50-100 ms/iter (2D), modest
**Risk:** LOW
**Complexity:** LOW

scipy's `fftconvolve` has overhead from input validation, padding calculation,
and internal bookkeeping. For the specific case of a 3D cube convolved with a
separable kernel (beam in spatial dims, LSF in spectral dim), we can use numpy's
FFT directly with pre-computed plans.

**Fix:** Pre-compute the padded FFT of the kernel once (not every iteration,
since the kernel doesn't change):
```python
# In set_beam_kernel(), after computing kernel:
self._beam_kernel_fft = np.fft.rfftn(kernel_padded)
# In convolve_with_beam():
cube_fft = np.fft.rfftn(cube_padded)
cube_conv = np.fft.irfftn(cube_fft * self._beam_kernel_fft)
```

This eliminates scipy's per-call overhead and the repeated kernel FFT
computation. The cube FFT is still needed each iteration, but that's the
unavoidable cost.

**Files to modify:** `dysmalpy/instrument.py`

### Opportunity 6: GPU acceleration for MPFIT path

**Impact:** Potentially 10-50x speedup for cube population + convolution
**Risk:** HIGH (requires GPU memory management, data transfer overhead)
**Complexity:** HIGH

The benchmark shows JAX GPU is 86x faster than Cython for cube population.
If the entire `simulate_cube` + `convolve` pipeline ran on GPU:
- Cube population: ~0.03 ms (GPU) vs ~4 ms (CPU)
- FFT convolution: ~1-2 ms (GPU) vs ~155 ms (CPU, after dtype fix)
- Model computation: would need to be GPU-compatible

**Blockers:**
- Data transfer between CPU (MPFIT, parameter updates) and GPU (model
  computation) adds latency (~1-5 ms per transfer for small arrays)
- The model computation (sersic_mr, enclosed_mass) operates on small arrays
  where GPU kernel launch overhead dominates
- scipy operations (brentq, fftconvolve) can't run on GPU without replacement

**Practical approach:** This is only worthwhile for the JAX-loss path
(JAXAdamFitter), not the MPFIT path. The MPFIT path should target main-branch
parity on CPU (~200 ms/iter) through Opportunities 1-3.

---

## Priority Summary (post gammaincinv fix)

| # | Opportunity | Severity | Impact (2D) | Risk | Status |
|---|------------|----------|-------------|------|--------|
| 1 | Fix float32 dtype mismatch | HIGH | -100 ms/iter | LOW | TODO |
| 2 | xp_dispatch in halo/baryon models | HIGH | -50 ms/iter | MEDIUM | TODO |
| 3 | Cache tied parameters | MEDIUM | -22 ms/iter | LOW | TODO |
| 4 | Pre-compute kernel FFT | LOW | -50 ms/iter | LOW | TODO |
| 5 | Full-graph JIT (JAX-loss only) | HIGH | GPU speedup | HIGH | Future |
| 6 | GPU for MPFIT path | LOW | Uncertain | HIGH | Not recommended |

**Target:** With Opportunities 1-3 implemented, dev_jax 2D MPFIT should reach
~220 ms/iter, matching main branch parity. Opportunities 4-5 provide additional
headroom beyond main.

### How to Run Diagnostics

```bash
JAX_PLATFORMS=cpu python dev/profile_mpfit.py [1D|2D|3D]
```
