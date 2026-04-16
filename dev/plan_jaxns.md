# Plan: Make JAXNS Fitting Work Correctly and Fast

## Context

The JAXNS fitter (`dysmalpy/fitting/jaxns.py`) exists but is broken and extremely slow:

1. **Performance bottleneck (CRITICAL)**: `_make_log_likelihood()` wraps numpy's `gal.create_model_data()` in `jax.pure_callback`, so every likelihood evaluation exits JIT and runs on CPU via Python/numpy. This makes JAXNS ~250x slower than MPFIT (43 min vs 10 s for the 2D case).

2. **A fully JAX-traceable pipeline already exists**: `dysmalpy/fitting/jax_loss.py` provides `make_jax_loss_function()` and `make_jax_log_prob_function()` that are JIT-compilable. `JAXAdamFitter` uses these successfully. The full pipeline `theta → simulate_cube → rebin → convolve → crop → chi²` is JAX-traceable (Phases 5-7 complete per `dev/development_note.md`).

3. **Bug: `plot_results()` wrong signature** (`jaxns.py:463-465`): passes `output_options=` kwarg which `BayesianFitResults.plot_results()` doesn't accept. No plots are produced.

4. **Bug: posterior analysis fails ~64% of runs**: `analyze_posterior_dist()` fails because KDE covariance matrix is singular.

5. **Tied parameter issue**: The 2D case (GS4_43501) uses `mvirial_tied=True` (mvirial computed from fdm via `scipy.optimize.brentq`) and `zheight_tied=True`. These tied functions are NOT JAX-traceable. Inside JIT, `_update_tied_parameters()` is skipped due to its cache, leaving stale tied values.

## Approach

Replace the `jax.pure_callback`-based log-likelihood with the existing JAX-traceable pipeline from `jax_loss.py`. Handle tied parameters by converting them to free parameters with constraint enforcement via jaxns priors. Fix the plotting and posterior analysis bugs.

## Files to Modify

| File | Change |
|------|--------|
| `dysmalpy/fitting/jax_loss.py` | Add `make_jaxns_log_likelihood()` factory |
| `dysmalpy/fitting/jaxns.py` | Replace `_make_log_likelihood` with JAX-traceable path; fix `plot_results`; fix posterior analysis; handle tied→free conversion |
| `dev/test_jaxns.py` | New test script: verify JAXNS works on CPU and GPU |

## Implementation Steps

### Step 1: Fix `plot_results()` call

**File**: `dysmalpy/fitting/jaxns.py`, lines 462-467

Replace:
```python
jaxnsResults.plot_results(gal, output_options=output_options,
                          overwrite=output_options.overwrite)
```
With the correct `BayesianFitResults.plot_results()` signature (`base.py:394`):
```python
jaxnsResults.plot_results(
    gal,
    f_plot_param_corner=output_options.f_plot_param_corner,
    f_plot_bestfit=output_options.f_plot_bestfit,
    f_plot_trace=output_options.f_plot_trace,
    f_plot_run=output_options.f_plot_run,
    overwrite=output_options.overwrite,
    only_if_fname_set=True)
```

### Step 2: Fix missing `ai_precomputed` in `make_jax_log_prob_function`

**File**: `dysmalpy/fitting/jax_loss.py`, around line 450

`make_jax_log_prob_function()` calls `model_set.simulate_cube(obs, dscale)` without `ai_precomputed`, while `make_jax_loss_function()` correctly passes it. Add the ai pre-computation block (matching `make_jax_loss_function` lines 228-236) and pass `ai_precomputed=` to the `simulate_cube` call.

### Step 3: Create `make_jaxns_log_likelihood()` factory

**File**: `dysmalpy/fitting/jax_loss.py`

New factory function that returns a jaxns-compatible `log_likelihood(*theta_tuple)`:

```
make_jaxns_log_likelihood(gal, fitter)
  → (log_likelihood, traceable_param_info, set_all_theta)
```

Key design:
- Uses the same pipeline as `make_jax_loss_function`: `_inject_tracers` → `simulate_cube` → `_rebin_spatial` → `convolve_cube_jax` → crop → chi²
- Returns **log-likelihood only** (no prior) — jaxns handles priors via its `Prior` objects
- Accepts `*theta_tuple` signature (each arg is a scalar) matching jaxns's prior model output
- Handles multi-observation galaxies (iterate over `gal.observations`)
- Pre-computes: `ai_precomputed`, convolution kernels, rebin/crop dimensions (all as concrete values outside JIT)
- Calls `_identify_traceable_params(gal.model)` to get the traceable subset

Implementation: largely copied from `make_jax_loss_function` (lines 178-317) with these changes:
- Signature: `log_likelihood(*theta_tuple)` instead of `jax_chi2(theta_traceable)`
- Convert tuple to flat array: `theta_traceable = jnp.array([jnp.atleast_1d(v)[0] for v in theta_tuple])`
- Skip prior computation (jaxns handles priors)
- Return `-0.5 * chi_sq` (log-likelihood, not half chi-squared)

### Step 4: Handle tied parameters for JAXNS

**File**: `dysmalpy/fitting/jaxns.py`, in `fit()` method

The 2D case has `mvirial_tied=True` and `zheight_tied=True`. These tied functions use `scipy.optimize.brentq`, `np.argmin`, dictionary lookups — none are JAX-traceable.

**Solution**: Before running JAXNS, convert tied parameters to free parameters:

1. For each tied parameter:
   - Evaluate the tied function once with current parameter values to get the initial value
   - Set `model.tied[comp][param] = False` (mark as no longer tied)
   - Set `model.fixed[comp][param] = False` (mark as free)
   - The parameter is now free and will be sampled by jaxns

2. For `mvirial_tied=True` specifically:
   - mvirial becomes a free parameter sampled by jaxns
   - fdm remains free (already traceable)
   - The NFW mass profile uses mvirial directly (no tied function needed)
   - jaxns's prior on mvirial is set from `mvirial_bounds` (Uniform)

3. For `zheight_tied=True`:
   - sigmaz becomes a free parameter (already free in the param file, just untie it)
   - The zheight relation is no longer enforced — sigmaz is sampled independently

4. After JAXNS completes, restore tied status for result reporting.

**Alternative considered**: Make tied functions JAX-traceable. Rejected because `calc_mvirial_from_fdm` uses `scipy.optimize.brentq` (root-finding) and `np.argmin` (lookup table), which require fundamentally different approaches in JAX. Converting to free parameters is simpler and correct — jaxns's Bayesian framework naturally handles parameter constraints through priors.

### Step 5: Rewrite `_build_jaxns_prior_model()` for traceable-only params

**File**: `dysmalpy/fitting/jaxns.py`

Current: iterates ALL free parameters. Fix: accept `traceable_param_info` from Step 3, only build priors for traceable parameters. Same prior-to-TFP-distribution mapping, just filtered.

### Step 6: Rewrite `fit()` to use JAX-traceable path

**File**: `dysmalpy/fitting/jaxns.py`, `fit()` method (line 255)

Replace lines ~317-325 (building jaxns model with `pure_callback`) with:

```python
from dysmalpy.fitting.jax_loss import make_jaxns_log_likelihood
(log_likelihood, traceable_param_info, set_all_theta) = make_jaxns_log_likelihood(gal, self)
set_all_theta()  # set geometry params to current values
prior_model, param_names = _build_jaxns_prior_model(gal, traceable_param_info)
jaxns_model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
```

Add sanity check: verify `log_likelihood(*theta_init)` is finite before running jaxns.

### Step 7: Fix posterior analysis

**File**: `dysmalpy/fitting/jaxns.py`

In `_setup_samples_blobs()`: improve the resampling logic to handle low-ESS cases. Use jaxns's `resample` utility with proper error handling. Fall back to weighted random choice if resample fails.

In `fit()` posterior analysis fallback (lines 436-443): replace `best_fit.mean(axis=0)` (unweighted mean of nested sampling points) with `np.median(samples_eq, axis=0)` (median of resampled posterior).

### Step 8: Create test script

**File**: `dev/test_jaxns.py`

Tests:
1. `test_jaxns_log_likelihood_finite` — verify log-likelihood is finite at initial params, JIT-compilable
2. `test_jaxns_sampling_cpu` — run short JAXNS on CPU, verify finite evidence and reasonable chi-squared
3. `test_jaxns_sampling_gpu` — same on GPU (skip if no GPU)
4. `test_jaxns_tied_params` — verify tied→free conversion works for the 2D GS4_43501 case

Uses the 2D case from `tests/test_data/fitting_2D_mpfit.params` (same as the MCMC demo).

### Step 9: Benchmark JAXNS speed

Compare old (`pure_callback`) vs new (JAX-traceable) on CPU and GPU. Report per-evaluation timing.

Expected:
- **Old (pure_callback)**: ~163 ms/eval (numpy `create_model_data`), no GPU benefit
- **New (JAX-traceable, CPU)**: ~50-80 ms/eval (JIT-compiled simulate_cube + convolve)
- **New (JAX-traceable, GPU)**: ~5-15 ms/eval (full pipeline on GPU)

## Verification

1. Run existing JAX tests: `JAX_PLATFORMS=cpu python -m pytest tests/test_jax.py -v`
2. Run JAXNS test: `JAX_PLATFORMS=cpu python dev/test_jaxns.py`
3. Run JAXNS test on GPU: `python dev/test_jaxns.py`
4. Run the JAXNS demo: `JAX_PLATFORMS=cpu python demo/demo_2D_fitting_JAXNS.py`
5. Verify: finite evidence, reasonable chi-squared (< 20), finite parameter uncertainties
6. Verify: plots are produced (the `plot_results` fix)
