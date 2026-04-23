---
name: fitting
description: >
  This skill should be used when the user asks to "run a fit", "fit kinematics",
  "debug fitting", "fix MCMC acceptance", "set up tied parameters", "run JAXNS",
  "use dysmalpy_fit_single", "reload a fit", "make a loss function", or
  "configure MPFIT/MCMC/JAXNS/Adam fitter".
---

# Running and Debugging Fits

## Fitter Classes

| Fitter | Module | Backend | Use case |
|--------|--------|---------|----------|
| `MPFITFitter` | `fitting/mpfit.py` | Levenberg-Marquardt | Fast local optimization, good initial guess |
| `MCMCFitter` | `fitting/mcmc.py` | emcee | Posterior sampling, handles multimodal landscapes |
| `NestedFitter` | `fitting/nested_sampling.py` | MultiNest | Bayesian evidence computation |
| `JAXNSFitter` | `fitting/jaxns.py` | jaxns | JAX-native nested sampling, GPU-accelerated |
| `JAXAdamFitter` | `fitting/jax_optimize.py` | Adam | JAX-native gradient descent |

## High-Level API

### `dysmalpy_fit_single()`

Main entry point in `fitting_wrappers/dysmalpy_fit_single.py`.  Takes a parameter
file path and runs the full pipeline: load data → setup model → configure fitter → fit → save.

```python
from dysmalpy.fitting_wrappers import dysmalpy_fit_single
dysmalpy_fit_single(param_filename='params.ini')
```

Dimension-specific variants dispatch based on data type:
- `dysmalpy_fit_single_1D()` — rotation curves
- `dysmalpy_fit_single_2D()` — velocity fields
- `dysmalpy_fit_single_3D()` — IFU datacubes

### `reload_all_fitting()`

Restores a saved fit from pickle files.  Returns `(galaxy, fitter, results)`.

```python
from dysmalpy.fitting import reload_all_fitting
gal, fitter, results = reload_all_fitting('output_dir')
```

## JAX Fitting Path

### `make_jax_loss_function()`

Creates a JIT-compiled closure mapping parameter vector `theta` → half chi-squared.
Located in `fitting/jax_loss.py`.

The closure uses `object.__setattr__` to inject JAX tracers directly into model
component parameter storage, bypassing the `float()` conversion in
`_DysmalModel.__setattr__`.  This makes the entire computation graph
(velocity profile → cube population → chi-squared) traceable.

```python
from dysmalpy.fitting import make_jax_loss_function
loss_fn = make_jax_loss_function(model_set, obs_list, include_convolution=True)
chi2_half = loss_fn(theta)
```

### `make_jax_log_prob_function()`

Returns `theta → log_posterior` for JAXNS/Adam samplers (chi-squared + log-prior).

### `make_jaxns_log_likelihood()` with geometry tracing

For JAXNS, geometry parameters (inc, pa, xshift, yshift, vel_shift) are traced
through JAX by pre-computing sky grids with concrete values and passing
`sky_grids_precomputed` to `simulate_cube()`.

## Tied Parameters

### Class-level descriptor setup

Tied parameters are set via class-level `DysmalParameter` descriptors:

```python
class MyComponent(MassModel):
    r_eff = DysmalParameter(default=5.0)
    r_eff_bulge = DysmalParameter(
        default=2.0,
        tied=lambda model: model.r_eff * 0.4
    )
```

### MPFIT parinfo bounds for tied parameters

Tied parameters **must** have `parinfo['limited'] = [0, 0]` so MPFIT does not
check bounds on them (their value is determined by the tying function):

```python
# In MPFIT setup:
if tied:
    parinfo['limited'] = [0, 0]  # no bounds checking for tied params
```

### `_update_tied_parameters` / `_get_free_parameters` agreement rule

The free parameter list and tied parameter update function must agree: every
parameter that is not fixed and not tied must appear in the free parameter list,
and vice versa.  A mismatch causes silent chi-squared evaluation errors.

## Pickle / Deepcopy Issues

### `ModelSet.__setstate__` — `_model` rebinding

After unpickling, `DysmalParameter._model` is `None`, so `.value` returns the
default instead of the fitted value.  `ModelSet.__setstate__` rebinds `_model`
on all parameter instances in all components.  This is already handled.

**Symptom:** MCMC acceptance rate = 0%, or fitted parameters revert to defaults
after loading a saved model.

### `forkserver` requirement for MCMC

JAX initializes internal thread pools at import time.  Python's default `fork`
copies this state into child processes, causing deadlocks.

```python
from multiprocess import get_context
pool = get_context('forkserver').Pool(self.nCPUs)
```

### MCMC 0% acceptance diagnosis

1. Check `_model` back-references (pickle issue above)
2. Check that `_get_free_parameters` and `_update_tied_parameters` agree
3. Check that initial walker positions satisfy all prior/bound constraints
4. Verify `JAX_PLATFORMS=cpu` is set (GPU multiprocessing is problematic)
5. Check that priors are not accidentally zero-probability at walker positions

## Fitting Workflow

1. **Setup**: `setup_gal_models.py` constructs `Galaxy`, `Observation`, `ModelSet`
2. **Initial fit**: Run MPFIT first for fast local optimization
3. **Posterior sampling**: Use MCMC or JAXNS from the MPFIT solution as starting point
4. **Diagnose**: Check reduced chi-squared, parameter uncertainties, acceptance rate

## Key Files

| File | Description |
|------|-------------|
| `dysmalpy/fitting/base.py` | Base fitter class, chi-squared metrics |
| `dysmalpy/fitting/jax_loss.py` | `make_jax_loss_function`, `make_jax_log_prob_function` |
| `dysmalpy/fitting/jax_optimize.py` | `JAXAdamFitter` |
| `dysmalpy/fitting/jaxns.py` | `JAXNSFitter` |
| `dysmalpy/fitting/mcmc.py` | `MCMCFitter` |
| `dysmalpy/fitting/mpfit.py` | `MPFITFitter` |
| `dysmalpy/fitting/utils.py` | `_update_tied_parameters`, `_get_free_parameters` |
| `dysmalpy/fitting_wrappers/dysmalpy_fit_single.py` | High-level entry point |
| `dysmalpy/fitting_wrappers/setup_gal_models.py` | Model setup from parameter files |
| `dysmalpy/fitting_wrappers/tied_functions.py` | Tied parameter functions |
