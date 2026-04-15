# Debug Plan: MCMC and Dynesty Fitting on `dev_jax`

## Problem Statement

MCMC and dynesty (nested) fitting hang indefinitely when `nCPUs > 1` on the `dev_jax`
branch. MPFIT fitting works correctly. MCMC with `nCPUs=1` also works.

## Root Cause #1: JAX Fork Deadlock (Confirmed)

**JAX is imported as a side-effect of importing `dysmalpy`, and JAX's internal
thread pools / locks deadlock when Python `fork()`s worker processes after JAX
has been initialized.**

### Evidence chain

1. `import dysmalpy.fitting_wrappers.setup_gal_models` pulls in ~200 JAX/jaxlib
   submodules (verified by checking `sys.modules` before/after).
2. MCMC (`mcmc.py:607`) and dynesty (`nested_sampling.py:183`) create a
   `Pool(nCPUs)` using `multiprocess.Pool` (or `multiprocessing.Pool` as
   fallback), which defaults to the **fork** start method on Linux.
3. The fork happens **after** JAX has initialized its XLA runtime, thread pools,
   and internal locks. Forking a process that holds POSIX locks leads to
   deadlocks in the child processes.
4. A minimal test confirms:
   - `Pool(2)` + `base.log_prob` + galaxy **without** loaded data = **works**
   - `Pool(2)` + `base.log_prob` + galaxy **with** loaded data = **hangs**
   - `Pool(1)` (serial) + full pipeline = **works**
   - `pickle.dumps(gal)` (with data) = **works** (not a serialization issue)
   - The deadlock is in the fork itself, not in data transfer.

### Why the galaxy-with-data case is different

The galaxy model alone (no data) uses only `setup_gal_model_base`, which imports
JAX transitively but doesn't call `_set_instrument_kernels` or
`setup_oversampled_chisq`. The fork likely deadlocks because the child processes
inherit JAX's XLA runtime state, and when emcee tries to evaluate `log_prob` in
the worker, JAX re-initializes or accesses the (now-corrupted) shared state.

The galaxy-with-data path adds kernel pre-computation (`_set_instrument_kernels`)
and oversampling setup (`setup_oversampled_chisq`), which may trigger additional
JAX compilation or device state that makes the fork deadlock more likely.

## Root Cause #2: Tied Parameter Desynchronization (Confirmed)

**Even with `nCPUs=1`, MCMC produces 0% acceptance because
`_update_tied_parameters()` and `_get_free_parameters()` disagree on which
parameters are tied.**

### The desynchronization mechanism

Two code paths determine "is this parameter tied?":

- **Path A** — `_get_free_parameters()` (model_set.py:609-618): Uses
  `self.tied[cmp][pm]` — the model's authoritative tied dict, populated at
  `add_component()` time from `_param_instances`.

- **Path B** — `_update_tied_parameters()` (model_set.py:576-589, **old code**):
  Scans every component's parameter descriptors via `getattr(comp, pp)`, then
  checks `getattr(param, 'tied', False)` on the resulting object.  Because
  `DysmalParameter` is a data descriptor, `getattr(comp, pp)` returns the
  **class-level** descriptor (not the per-instance copy stored in
  `_param_instances`).

These two paths disagree when `.tied` is set on the class descriptor **after**
the component is added to the model. This happens for `sigmaz` in
`setup_gal_models.py`:

```python
zheight_prof = models.ZHeightGauss(sigmaz=sigmaz, ...)  # __init__ copies class descriptors
zheight_prof.sigmaz.tied = tied_sigmaz_func              # sets tied on CLASS descriptor
mod_set.add_component(zheight_prof)                       # self.tied['sigmaz'] = False
```

At step 1, `__init__` (base.py:282-286) deepcopies class descriptors into
`_param_instances`. At this point `sigmaz.tied = False` (class default). The
model's `self.tied['sigmaz']` is set to `False` at base.py:315.

At step 2, the CLASS-LEVEL descriptor gets `.tied = tied_sigmaz_func`. The
`_param_instances` copy is NOT updated.

**Result:**
- `_get_free_parameters()` sees `sigmaz` as **FREE** (from `self.tied`)
- `_update_tied_parameters()` OVERWRITES it after every `update_parameters()`
  call (from class descriptor's `.tied`)
- The tied function computes `sigz = 2.0 * r_eff_disk / invq / 2.35482`, which
  can exceed the `sigmaz` bounds `[0.1, 1.0]` when `r_eff_disk > ~1.2 kpc`
  (with typical `invq ~ 5`)
- `get_log_prior()` returns `-inf` for the out-of-bounds value
- emcee rejects all proposals → 0% acceptance

### Observable symptoms

- 0% acceptance fraction across all walkers
- Empty result plots (colorbars only, no data)
- FIXED parameters show wildly wrong values (e.g., `sigma0=80714` instead of
  `39.0`) — likely a secondary effect of the descriptor's `_model` reference
  being rebound during deep copy or tied parameter updates
- Free parameters have extreme values outside their bounds (e.g.,
  `xshift=-39845`, bounds `[-1.5, 1.5]`)

### The fix

Changed `_update_tied_parameters()` to iterate over `self.tied` dict instead of
scanning descriptors. This ensures both code paths use the same authoritative
configuration:

```python
def _update_tied_parameters(self):
    # ... (cache check stays the same) ...
    for cmp in self.tied:
        comp = self.components[cmp]
        for pp in self.tied[cmp]:
            tied_fn = self.tied[cmp][pp]
            if callable(tied_fn):
                try:
                    new_value = tied_fn(self)
                    self.set_parameter_value(cmp, pp, new_value,
                                             skip_updated_tied=True)
                except Exception:
                    pass
```

### Diagnostic script

`dev/diagnose_mcmc.py` can be used to verify the fix. It:
1. Sets up the galaxy/model identically to the MCMC demo
2. Checks tied parameter synchronization between `self.tied` and descriptors
3. Samples random parameter sets and tests `log_prob()` for each
4. Runs a 1-step MCMC to check acceptance fraction

## Impact

| Fitter  | nCPUs=1 | nCPUs>1 |
|---------|---------|---------|
| MPFIT   | Works   | N/A (single-threaded) |
| MCMC    | Works (after tied fix) | **Deadlock** (fork + JAX) |
| Dynesty | Works (after tied fix) | **Deadlock** (fork + JAX) |

## Fix Options for Fork Deadlock

### Option A: Defer JAX import until actually needed (Recommended)

Lazy-import JAX only inside the JAX-specific code paths (`jax_loss.py`,
`jax_optimize.py`, `cube_processing.py`), so that MCMC/nested fitting never
triggers JAX initialization.

**Pros:** Clean fix, no behavioral change for MCMC/nested users.
**Cons:** Requires auditing all JAX imports across the codebase. If the model
evaluation path (`simulate_cube` -> `populate_cube_jax`) uses JAX unconditionally,
this won't work without also providing a numpy fallback for that code.

**Feasibility check needed:** Does `simulate_cube` have a non-JAX fallback?
Currently `cube_final = jnp.zeros(...)` and `populate_cube_jax(...)` are
unconditional. This would need a `numpy` code path for the cube population.

### Option B: Use `forkserver` or `spawn` start method

Override the multiprocessing start method before creating the Pool:

```python
import multiprocessing
multiprocessing.set_start_method('forkserver', force=True)
```

**Pros:** Simple, avoids the fork-after-JAX deadlock.
**Cons:**
- `multiprocess` (used by emcee/dynesty) may not support `forkserver`.
- `spawn` requires all arguments to be picklable (already confirmed OK).
- `forkserver` adds a server process overhead.
- May break on some platforms or Python versions.

**Feasibility check needed:** Test if `multiprocess.Pool` works with
`forkserver` when JAX is already imported.

### Option C: Initialize the Pool BEFORE importing JAX

Create the worker pool in the fitter **before** any JAX code is triggered, then
pass the pool to emcee/dynesty. Workers forked before JAX init won't have the
deadlock.

**Pros:** Technically sound.
**Cons:** JAX is imported at module level in many places; restructuring would be
invasive and fragile.

### Option D: Disable multiprocessing when JAX is detected

Add a guard in the MCMC/nested fitters:

```python
import jax
if jax.config.values.get('jax_platforms', '') != '':
    logger.warning("JAX detected: forcing nCPUs=1 to avoid fork deadlock")
    self.nCPUs = 1
    pool = None
```

**Pros:** Minimal code change, guaranteed to work.
**Cons:** Loses parallelism for MCMC/nested when JAX is installed. Not a real
fix, just a workaround.

## Recommended Approach

**Option A + D as immediate mitigation:**

1. **Short term (Option D):** Add a guard in `_fit_emcee_3` and
   `NestedFitter.fit` that detects JAX has been imported and forces
   `nCPUs=1` with a warning. This unblocks users immediately.

2. **Medium term (Option A):** Audit and lazy-import JAX so that the MCMC/nested
   path never triggers JAX initialization. This restores parallelism.

## Test Plan

### Step 1: Reproduce and confirm the fix scope

```bash
# 1D MCMC, nCPUs=2 (currently hangs)
JAX_PLATFORMS=cpu python -m pytest tests/test_fitting.py::TestFittingWrappers::test_1D_mcmc -xvs

# 1D nested, nCPUs=2 (presumably also hangs)
JAX_PLATFORMS=cpu python -m pytest tests/test_fitting.py::TestFittingWrappers::test_1D_nested -xvs
```

### Step 2: Implement Option D (workaround)

- File: `dysmalpy/fitting/mcmc.py` (~line 606)
- File: `dysmalpy/fitting/nested_sampling.py` (~line 182)
- Add: if `jax` is in `sys.modules`, force `nCPUs=1` and log a warning.
- Re-run Step 1 tests to confirm they pass.

### Step 3: Implement Option A (proper fix)

- Audit all `import jax` / `import jax.numpy` / `from jax` in:
  - `dysmalpy/models/cube_processing.py`
  - `dysmalpy/models/model_set.py`
  - `dysmalpy/models/kinematic_options.py`
  - `dysmalpy/fitting/jax_loss.py`
  - `dysmalpy/fitting/jax_optimize.py`
- Wrap in `try/except ImportError` or use lazy imports.
- Provide numpy fallback for `populate_cube_jax` (restore the original Cython
  or numpy implementation).
- Re-run Step 1 tests with `nCPUs=2` to confirm parallelism is restored.

### Step 4: End-to-end validation

```bash
# 1D MCMC with nCPUs=2
# 2D MCMC with nCPUs=2 (fast settings: oversample=1, nWalkers=24, nBurn=2, nSteps=5)
# 1D nested with nCPUs=2
# MPFIT should still work unchanged
```

## Key Files

| File | Role |
|------|------|
| `dysmalpy/fitting/mcmc.py:606` | Pool creation for emcee |
| `dysmalpy/fitting/nested_sampling.py:182` | Pool creation for dynesty |
| `dysmalpy/fitting/base.py:729` | `log_prob()` — called in every worker |
| `dysmalpy/fitting/base.py:775` | `log_like()` — model evaluation |
| `dysmalpy/models/model_set.py:545-589` | `_update_tied_parameters()` — **tied bug fix** |
| `dysmalpy/models/model_set.py:609-618` | `_get_free_parameters()` — free param logic |
| `dysmalpy/models/model_set.py:1302` | `simulate_cube()` — uses `jnp.zeros`, `populate_cube_jax` |
| `dysmalpy/models/cube_processing.py:30` | `populate_cube_jax()` — JAX-based cube population |
| `dysmalpy/observation.py:314` | `np.asarray(sim_cube)` — converts JAX -> numpy after simulation |
| `tests/test_fitting.py:159` | 1D MCMC test |
| `tests/test_fitting.py:191` | 1D nested test |
| `dev/diagnose_mcmc.py` | Diagnostic script for tied parameter bug |
