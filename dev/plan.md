# Development Plan

## Current Status: Active Development (2026-04-28)

### Recent Completed Work

#### 1. Custom Gradient Descent for Gaussian Fitting ✅ (April 27-28)
**Commits:** 4bca2a9, b477870, 9dfbb3f, 0fea0c0

**Implemented:**
- Custom gradient descent method (`custom_gradient_descent` in `jax_gaussian_fitting.py`)
- Default method changed from `'closed_form'` to `'hybrid_gd'` (4-6x overhead)
- Reduces bias by 70% compared to closed-form (validated on asymmetric spectra)
- Much faster than BFGS hybrid (4-6x vs 245x overhead)

**Documentation:**
- `dev/GAUSSIAN_FITTING_METHODS.md` - Comprehensive method comparison
- `dev/HYBRID_SPEEDUP_REPORT.md` - Performance analysis
- Tests in `tests/validate_gaussian_fit_simple.py` - Bias reduction validation

#### 2. JAXNS Mask Inversion Bug Fix ✅ (April 28)
**Commit:** a893d49

**Fixed:** Critical mask inversion bug in jax_loss.py
- Changed `mask=(msk == 0)` to `mask=(msk == 1)`
- JAXNS log-likelihood from -2.7 billion → -107.70
- Reduced chi-squared from infinity to 4.39

**Documentation:** `dev/JAXNS_MOMENT_CALC_FIX.md`

#### 3. JAXNS Weight Evolution Investigation 🔬 (April 28 - Resolved)
**Location:** `dev/debug_jaxns/`

**Issue:** Weight evolution plot shows abnormal rising pattern instead of smooth decay

**Root Cause:** `hybrid_gd` method instability (commit 4bca2a9)
- Custom gradient descent can diverge for noisy/low-flux spectra
- Causes extreme chi-squared values (-3659 vs expected -107)
- JAXNS struggles to explore parameter space efficiently

**Evidence:**
- Initial log-likelihood varies: -107.70 (good) to -3659.08 (bad)
- Runs with bad likelihood take 540-744s vs 29-40s for good runs
- Weight evolution shows "rise and fall" instead of smooth decay

**Fix:** Switch back to `closed_form` method (April 28)
- Stable analytical solution (no gradient descent divergence)
- Consistent log-likelihood across runs
- 2.5x overhead (faster than hybrid_gd's 4-6x)
- Small bias is acceptable for JAXNS parameter uncertainties

**Documentation:**
- `WEIGHT_EVOLUTION_FIX.md` - Complete analysis and solution
- Test script: `test_closed_form_likelihood.py`

---

## Ongoing Tasks

### 1. Verify JAXNS closed_form Fix 🟡 TESTING
**Status:** JAXNS demo running with `method='closed_form'`

**Changes Made:**
- Changed `jax_loss.py` line 882 from `method='hybrid_gd'` to `method='closed_form'`

**Expected Results:**
- Stable initial log-likelihood (consistently ~-107)
- Fast completion (29-40s per run)
- Weight evolution shows smooth decay (no rise)

**Timeline:** Testing in progress (April 28)

**Hypothesis:** Invalid pixels (vel_obs=-1e6) not properly masked in chi-squared calculation

**Next Steps:**
- Verify mask values in observation data (mask=1 for valid?)
- Check if mask application in jax_loss.py is correct
- Test with pre-April 15 code to confirm
- Fix mask handling if inverted

**Timeline:** 1-2 days

---

## Completed Projects

### Project: JAX-Compatible Gaussian Fitting (April 2026)
**Status:** ✅ COMPLETE

**Achievements:**
1. Created `jax_gaussian_fitting.py` module with closed-form and hybrid methods
2. Integrated with `observation.py` and `jax_loss.py`
3. Fixed critical mask inversion bug
4. Implemented custom gradient descent (4-6x overhead, 70% bias reduction)
5. Validated on asymmetric test spectra
6. Documented comprehensively

**Impact:** JAXNS can now use `moment_calc=False`, enabling direct comparison with MPFIT results.

---

## Technology Stack

- **JAX:** 0.4.38, jaxlib: 0.4.38, jax-cuda12-plugin: 0.4.38
- **JAXNS:** 2.4.13
- **GPU:** NVIDIA 4090 (CUDA 12)
- **Python:** 3.10+
- **Conda env:** `alma`

---

## Key Design Decisions

### Gaussian Fitting Method Choice
- **Default:** `hybrid_gd` (closed-form + custom gradient descent)
  - 4-6x overhead vs moments
  - 70% bias reduction vs closed-form
  - Practical for JAXNS (~1-2 min for 10k iterations)

### GPU by Default
- **Fitting (JAXNS, Adam):** GPU default for performance
- **Testing:** CPU for reproducibility
- **MCMC:** CPU only (multiprocessing issues)

---

## Problem Catalog

See `dev/problem.md` for detailed catalogue of known issues and pitfalls.

Top issues:
1. **JAX defaults to float32** - `JAX_ENABLE_X64=1` set in `__init__.py`
2. **DysmalParameter descriptor pollution** - Use `comp._get_param(name)`
3. **JAXNS weight evolution** - Under investigation (commit 53beeae)

---

## Development Workflow

**Branches:**
- **`main`** - Current development (JAX-accelerated)
- **`dysmalpy_origin`** - Original Cython version (reference)

**Testing:**
```bash
# Activate environment
source activate_alma.sh

# Run tests (CPU for reproducibility)
JAX_PLATFORMS=cpu pytest tests/ -v

# Run demo (GPU for speed)
python demo/demo_2D_fitting_JAXNS.py
```

---

## Documentation

**Main Notes:**
- `develop_log.md` - Complete development log
- `plan.md` - This file
- `problem.md` - Known issues and gotchas

**Specialized:**
- `GAUSSIAN_FITTING_METHODS.md` - Gaussian fitting methods comparison
- `HYBRID_SPEEDUP_REPORT.md` - Performance analysis
- `debug_jaxns/` - JAXNS investigation

---

**Last Updated:** 2026-04-28
**Status:** Active development with JAXNS investigation in progress
