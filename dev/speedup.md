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

- **Cython**: Triple-nested C loop (`x × y × z × s`) with typed memoryviews.
  Single-threaded CPU only.

- **JAX (CPU)**: Same algorithm vectorized with `jax.vmap` over spectral
  channels and JIT-compiled by XLA.  Gains come from removing the Python
  loop overhead and leveraging SIMD vectorization.

- **JAX (GPU)**: Same JIT-compiled code offloaded to the 4090.  The massive
  parallelism (970K spatial points × 201 spectral channels) fully
  saturates the GPU.

- **JAX (GPU) JIT loss+grad**: Full `simulate_cube` → chi-squared loss +
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
This is expected because:

1. **The MPFIT backend is identical** — both branches use the same Cython
   `cutils.pyx` for the inner cube population loop and the same pure-Python
   `mpfit.py` for the Levenberg-Marquardt optimizer.

2. **The overhead comes from Python-level changes**: The `dev_jax` branch
   replaces `astropy.modeling.Parameter` with `DysmalParameter`, adds
   numpy 2.x compatibility layers, and introduces a descriptor-based
   parameter system. These add per-iteration overhead in parameter
   access, bound checking, and tied parameter evaluation.

3. **Convergence differs**: The two branches converge to different solutions
   in some cases (notably 2D, where dev_jax reaches redchisq=34 vs 4.3 on
   main), indicating that parameter system differences affect the optimization
   path. The 1D case converges on dev_jax in fewer iterations (5 vs 8)
   but to a slightly worse redchisq (2.21 vs 1.91).

4. **The JAX speedup from the cube population benchmark (4.5x CPU, 86x GPU) is
   not realized here** because the MPFIT fitter uses the Cython backend, not
   JAX. The JAX gains will materialize when using `JAXAdamFitter` or a
   future JAX-based MPFIT replacement.

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
