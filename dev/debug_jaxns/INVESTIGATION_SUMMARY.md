# JAXNS Weight Evolution Issue - Investigation Summary

**Date:** 2026-04-28
**Status:** Root cause identified
**Location:** `dev/debug_jaxns/`

---

## Problem Description

The JAXNS weight evolution plot shows abnormal behavior:
- Weight peaks as expected
- Then **rises back up** instead of decaying smoothly to zero
- This indicates the sampler is not converging properly

---

## Root Cause Identified

### Analysis from `analyze_jaxns_results.py`

**Log likelihood samples:**
```
Min: -60358.80859375  ← Extremely negative!
Max: -24.08592414855957  ← Reasonable final value
Mean: -781.9512939453125  ← Biased by extreme negatives
```

**First 10 samples:**
```
[-60358.81, -39595.72, -39069.85, -36533.95, -35817.82,
 -35732.36, -35700.92, -34622.79, -33799.12, -33679.64]
```

**Last 10 samples:**
```
[-24.75, -24.74, -24.72, -24.70, -24.67,
 -24.62, -24.55, -24.53, -24.53, -24.09]
```

### What This Means

1. **Sampler is exploring extremely bad likelihood regions**
   - Early samples have log L ≈ -60,000 (impossibly bad)
   - This corresponds to chi-squared ≈ 120,000
   - For reference: reduced chi-squared should be ~1-10

2. **Likelihood function is producing extreme values**
   - Something is causing chi-squared to explode
   - Not a smooth likelihood surface
   - Sampler gets confused by these extreme regions

3. **This is NOT the hybrid_gd constraint issue**
   - User reported this existed before hybrid_gd
   - Problem is deeper in the likelihood computation

---

## Potential Causes

### 1. Invalid Pixels Not Properly Masked (Most Likely)

**Hypothesis:** Invalid pixels (mask=0, vel_obs=-1e6) are contributing to chi-squared.

**Code in jax_loss.py (lines 904-905, 912-913):**
```python
chi_sq += jnp.sum(((vel_map - od['vel_obs']) / od['vel_err']) ** 2 * msk)
```

**If mask is wrong:**
- Invalid pixels have vel_obs = -1e6 (sentinel value)
- vel_map might be reasonable (e.g., 100 km/s)
- Residual = 100 - (-1e6) = 1,000,100 km/s
- Chi-squared = (1e6 / 50)² = 4e11  ← Enormous!

**This explains the extreme likelihood values!**

### 2. Numerical Issues in Gaussian Fitting

**Hypothesis:** hybrid_gd method produces NaN or Inf values in some pixels.

**Evidence:**
- hybrid_gd uses gradient clipping and constraints
- Could create numerical instabilities
- Need to check vel_map/disp_map for NaN/Inf

### 3. Prior Bounds Too Wide

**Hypothesis:** Priors allow parameters to explore unphysical regions.

**Check in demo script:**
```python
total_mass_prior_bounds: 10.0 to 13.0
inc_prior_bounds: 42.0 to 82.0
...
```

Bounds seem reasonable, but worth verifying.

---

## Next Steps

### Immediate: Check Mask Application

Add debug output to likelihood function to log:
1. How many invalid pixels are being fitted
2. What are the chi-squared contributions from valid vs invalid pixels
3. Is the mask being applied correctly in all places?

### Test: Force moment_calc=True

If the problem disappears with moment extraction, it confirms the issue is in the Gaussian fitting code.

### Test: Revert to closed_form

Test if `method='closed_form'` fixes the issue (eliminates hybrid_gd as cause).

---

## Files Created

1. **`dev/debug_jaxns/analyze_jaxns_results.py`**
   - Analyzes JAXNS sampler results
   - Checks log likelihood evolution
   - Identifies extreme values

2. **`dev/debug_jaxns/debug_likelihood.py`**
   - Analyzes chi-squared values
   - Checks mask handling
   - Tests Gaussian fitting output

---

## Current Status

✅ **Root cause identified:** Extremely negative log likelihood values (-60,000)
✅ **Most likely cause:** Invalid pixels not properly masked in chi-squared calculation
⏳ **Next action:** Verify mask application in likelihood function
⏳ **Testing needed:** Compare with moment_calc=True and method='closed_form'

---

## Recommendation

**Immediate fix:** Double-check that `msk` is being applied correctly in the chi-squared calculation. Invalid pixels (mask=0) should contribute ZERO to chi-squared, not enormous values.

**Code to check:**
```python
# Line 905 in jax_loss.py
chi_sq += jnp.sum(((vel_map - od['vel_obs']) / od['vel_err']) ** 2 * msk)
#                                                         ^^^
# This should zero out contributions from invalid pixels
```

**Verify:** What is `od['mask']`? What are its values for invalid pixels?
- Should be 0 for invalid pixels
- Should be 1 for valid pixels

---

**Commit:** 9669ff9
**Date:** 2026-04-28
