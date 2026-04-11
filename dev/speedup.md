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
