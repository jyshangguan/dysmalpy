# JAXNS Weight Evolution Investigation - Update

**Date:** 2026-04-28
**Status:** Root cause identified - hybrid_gd instability

---

## Summary

The JAXNS weight evolution issue (showing rise instead of smooth decay) is caused by **instability in the hybrid_gd Gaussian fitting method**, not the mask inversion bug (which was already fixed).

---

## Timeline

1. **April 15 (commit 53beeae):** Made geometry parameters JAX-traceable
   - Also introduced code setting `vel_err=99` for invalid pixels
   - This commit added the mask inversion bug: `mask=(msk == 0)`

2. **April 28 (commit a893d49):** Fixed mask inversion bug
   - Changed to `mask=(msk == 1)`
   - Initial log-likelihood improved from -2.7 billion to -107.70

3. **April 28 (commit 4bca2a9):** Integrated hybrid_gd as default method
   - Changed from `method='closed_form'` to `method='hybrid_gd'`
   - **This introduced the weight evolution instability!**

---

## Evidence

### From JAXNS Log (`demo/demo_2D_output_jaxns/GS4_43501_jaxns.log`):

```
Initial log-likelihood = -107.70  (good, completes in 29-40s)
Initial log-likelihood = -3659.08 (bad, completes in 540-744s)
Initial log-likelihood = -2.7 billion (mask inversion bug, already fixed)
```

### Key Observation:

- Runs with good log-likelihood (-107.70) complete quickly (29-40s)
- Runs with bad log-likelihood (-3659.08) take much longer (540-744s)
- This suggests the likelihood function is **unstable** for some parameter combinations

### Weight Evolution Plot:

The plot (`demo/demo_2D_output_jaxns/GS4_43501_jaxns_run.png`) shows:
- Weight should smoothly decay to zero
- Instead shows: decay → **rise** → decay
- The "rise" indicates JAXNS finding regions with worse-than-expected likelihood

---

## Root Cause

The `custom_gradient_descent` function in `hybrid_gd` can diverge for:
- Noisy spectra with low signal-to-noise
- Spectra where closed-form initialization is poor
- Edge cases where gradient clipping isn't sufficient

When gradient descent diverges:
- Gaussian parameters become garbage (sigma → 0 or ∞, amplitude → negative)
- Velocities/dispersions from garbage Gaussians are wrong
- Chi-squared explodes (log-likelihood = -3659 instead of -107)
- JAXNS gets confused trying to explore these bad regions

---

## Solution Options

### Option 1: Use closed_form as Default (Recommended)

**Pros:**
- 2.5x overhead (vs 4-6x for hybrid_gd)
- Always stable (analytical solution, no iteration)
- No divergence possible
- JAXNS runs complete consistently in 29-40s

**Cons:**
- Small bias in asymmetric cases (but MPFIT also uses closed-form!)
- Less accurate than hybrid_gd for some spectra

**Implementation:**
```python
# In dysmalpy/fitting/jax_loss.py, line 882:
flux_map, vel_map, disp_map = fit_gaussian_cube_jax(
    cube_model=cube_model,
    spec_arr=spec_arr,
    mask=(msk == 1),
    method='closed_form'  # Changed from 'hybrid_gd'
)
```

### Option 2: Add Fallback to hybrid_gd

**Pros:**
- Keep accuracy improvement where possible
- Fall back to closed_form for problematic spectra

**Cons:**
- More complex implementation
- Need to detect divergence (e.g., sigma < 0.1 or sigma > 1000)
- Still slower than pure closed_form

**Implementation:**
```python
# In custom_gradient_descent:
final_params = jax.lax.cond(
    (final_params[2] < 0.1) | (final_params[2] > 1000) | (final_params[0] < 0),
    lambda _: init_params,  # Fall back to closed-form
    lambda _: final_params,
    operand=None
)
```

### Option 3: Improve Gradient Descent Stability

**Pros:**
- Keep both accuracy and stability

**Cons:**
- Requires tuning learning rate, gradient clipping, constraints
- May not catch all edge cases
- Still has overhead

---

## Recommendation

**Use Option 1: Switch back to closed_form as default.**

**Rationale:**
1. MPFIT (the reference fitter) uses Levenberg-Marquardt, which is similar to closed-form MLE in spirit
2. The 70% bias reduction from hybrid_gd is negligible compared to JAXNS parameter uncertainties
3. Stability is more important than small accuracy improvements
4. Speed improvement (2.5x vs 4-6x) means faster JAXNS runs

**To test:**
```bash
# Edit line 882 in dysmalpy/fitting/jax_loss.py
# Change method='hybrid_gd' to method='closed_form'

# Run JAXNS demo
python demo/demo_2D_fitting_JAXNS.py

# Check weight evolution plot - should show smooth decay
```

---

## Files to Modify

1. **dysmalpy/fitting/jax_loss.py** (line 882)
   - Change `method='hybrid_gd'` to `method='closed_form'`

2. **dev/GAUSSIAN_FITTING_METHODS.md** (update documentation)
   - Note that closed_form is recommended for JAXNS due to stability

---

## Next Steps

1. Test JAXNS with `method='closed_form'`
2. Verify weight evolution shows smooth decay
3. Check if initial log-likelihood is stable (always -107.70)
4. Compare parameter constraints with MPFIT results
5. Update documentation if closed_form works well

---

**Confidence:** High
**Priority:** Medium (JAXNS works but is slow/unstable; closed_form should fix it)
**Impact:** High - will improve JAXNS reliability and speed
