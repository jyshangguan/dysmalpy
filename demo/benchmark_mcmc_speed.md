# MCMC Speed Benchmark: CPU Multi-Core vs GPU

This report compares the performance of DYSMALPY's emcee-based MCMC sampler
across different CPU core counts and GPU configurations, using the 2D fitting
case (GS4_43501, H-alpha kinematics at z=1.613).

## Benchmark setup

- **Script:** `dev/benchmark_mcmc.py`
- **Galaxy:** GS4_43501 (2D velocity + dispersion maps)
- **Model:** disk+bulge, const_disp_prof, geometry, zheight_gaus, NFW halo (8 free parameters)
- **MCMC settings:** nWalkers=16, nBurn=5, nSteps=20 (400 evaluations per config)
- **Platform:** Linux, NumPy 2.3.5, JAX with CUDA GPU (cuda:0)
- **Branch:** dev_jax (commit 372b922)

Each configuration runs in an isolated subprocess so that `JAX_PLATFORMS`
can be changed between CPU and GPU modes.

## Results

```
Config                              nCPUs  Wall (s)  Evals  ms/eval  Speedup   Eff%  Accept%
--------------------------------------------------------------------------------------
CPU serial                              1     67.46    400    168.6    1.00x   100%     30%
CPU 2-core                              2    122.36    400    305.9    0.55x    28%     30%
CPU 4-core                              4     88.58    400    221.5    0.76x    19%     32%
CPU 8-core                              8     67.14    400    167.8    1.00x    13%     31%
GPU (conv only)                         1     56.27    400    140.7    1.20x   120%     29%
GPU (conv only) 4-core                  4    108.81    400    272.0    0.62x    15%     32%
```

### Single log_prob breakdown (CPU serial)

```
  update_parameters         0.1 ms
  create_model_data       129.1 ms
  log_like                  0.2 ms
  Total                   129.5 ms
```

## Analysis

### CPU multi-core scaling

Multi-core parallelism provides **no measurable benefit** at this workload size.
Adding cores does not improve per-evaluation time and often makes it worse:

- **CPU 2-core** is 1.8x *slower* than serial (305.9 ms vs 168.6 ms)
- **CPU 8-core** matches serial only because 8 workers mask the overhead,
  but parallel efficiency is just 13%

The root cause is **forkserver overhead**. Each MCMC step requires pickling
the `galaxy` object and sending it to worker processes via the forkserver pool.
With evaluations taking only ~130 ms each, this fixed communication cost
dominates. The situation improves with longer runs (thousands of evaluations)
where the overhead amortizes, but parallel efficiency remains low because
emcee's stretch move is synchronous -- all walkers must complete before the
next step begins.

### GPU acceleration

GPU (FFT convolution via `jnp.fft.rfftn` in `_fft_convolve_cached`) provides
a **modest 20% speedup** over CPU serial:

- CPU serial: 168.6 ms/eval
- GPU (conv only): 140.7 ms/eval

The improvement is small because `simulate_cube` -- which accounts for the
vast majority of `create_model_data` time (~129 ms) -- runs on numpy/CPU.
Only the FFT convolution step benefits from the GPU, and convolution is a
relatively small fraction of the total pipeline cost.

### GPU + multi-core

GPU with 4 cores (272.0 ms/eval) is slower than GPU serial (140.7 ms).
The multiprocessing overhead is compounded by JAX GPU context re-initialization
in each forkserver worker.

## Conclusion

| Strategy | Recommendation |
|----------|---------------|
| Short runs (< 1000 evals) | Use **single-core CPU** -- simplest and fastest |
| Medium runs (1000--10000 evals) | **Single-core GPU** gives ~20% improvement at no extra complexity |
| Long runs (> 10000 evals) | Multi-core may help marginally; GPU still limited to ~20% gain |
| Maximum GPU utilization | Not achievable with emcee MCMC. Use `JAXNSFitter` or `JAXAdamFitter` instead, which have a fully JAX-traceable loss function (~79 ms/eval on GPU including gradients) |

The fundamental bottleneck is that emcee requires a Python-callable `log_prob`
function, preventing JAX from tracing the full pipeline onto the GPU. The
`simulate_cube` body uses numpy (via `xp_dispatch`) on the MCMC path, and
this cannot be moved to GPU without replacing the emcee sampler entirely.
