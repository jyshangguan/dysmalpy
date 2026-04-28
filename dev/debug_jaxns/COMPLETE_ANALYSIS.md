# JAXNS Weight Evolution Issue - Complete Analysis

**Date:** 2026-04-28
**Breaking Commit:** Around April 15, 2026 (commit 53beeae)
**Status:** Root cause identified, ready for fix

---

## Problem Summary

**User Report:** JAXNS weight evolution plot shows abnormal rising pattern instead of smooth decay to zero. This started happening after major changes around April 17, 2026. JAXNS was working fine before this date.

**Analysis Result:** Log likelihood samples range from **-60,358 to -24**, indicating the sampler is exploring parameter regions with impossibly bad chi-squared values (~120,000 instead of expected ~10-100).

---

## Root Cause

### The Breaking Change: Commit 53beeae (April 15, 2026)

**Title:** "Make geometry parameters JAX-traceable for JAXNS fitting"

**What Changed:**
- Previously JAXNS only fit 5 parameters (total_mass, r_eff_disk, fdm, sigma0, sigmaz)
- Geometry parameters (inc, pa, xshift, yshift, vel_shift) were excluded
- This commit added pre-computed sky grids to make geometry parameters JAX-traceable
- **443 lines added to `jax_loss.py`**

**Why It Broke JAXNS:**
The commit changed the observation data preparation code. Specifically, this section:

```python
# Line added in commit 53beeae
# Replace zero errors in unmasked pixels
vel_err_np = np.asarray(vel_err)
msk_np = np.asarray(msk)
vel_err_np = np.where((vel_err_np == 0) & (msk_np == 0), 99., vel_err_np)
vel_err = jnp.asarray(vel_err_np)
```

**The Issue:**
1. Invalid pixels have `vel_obs = -1e6` (sentinel for missing data)
2. Invalid pixels should have `mask = 0`
3. When mask=0, the chi-squared should be multiplied by 0
4. **BUT**: If `vel_err = 99` (from the code above) and `vel_obs = -1e6`:
   - Residual = vel_map - (-1e6) ≈ 1e6 km/s
   - Chi2 = (1e6 / 99)² ≈ 1e8 (still enormous!)
   - **Mask multiplication:** 0 * 1e8 = 0 (should work...)

**Wait - that should still work!**

The mask multiplication should eliminate the contribution. Unless...

**Hypothesis:** The mask values might be inverted or there's a mask processing issue in the observation data setup after the April 15 commit.

---

## Diagnostic Results

### From `analyze_jaxns_results.py`:

```
Log likelihood samples:
  Min: -60358.80859375  ← chi-squared ≈ 120,000
  Max: -24.08592414855957 ← chi-squared ≈ 50
  Mean: -781.9512939453125

First 10 samples:
  [-60358, -39595, -39069, -36533, -35817, -35732, -35700, -34622, -33799, -33679]
```

**Interpretation:** The sampler is finding regions with chi-squared ≈ 120,000, which is impossible for a good fit. This suggests invalid pixels are contributing to the likelihood.

### From JAXNS Log:

```
Initial log-likelihood = -107.70  (early runs)
Initial log-likelihood = -3659.08 (later runs)
```

The initial log-likelihood varies significantly between runs, indicating numerical instability.

---

## Timeline

### Before April 15, 2026:
- ✅ JAXNS working fine
- Only 5 parameters fitted (no geometry)
- Mask handling working correctly

### April 15, 2026 (Commit 53beeae):
- ⚠️ Geometry parameters made JAX-traceable
- ⚠️ Major changes to observation data preparation
- ⚠️ 443 lines added to jax_loss.py
- ❌ JAXNS starts having issues

### After April 15:
- ❌ Weight evolution shows abnormal rising pattern
- ❌ Extreme negative log-likelihood values
- ❌ Sampler explores bad parameter regions

---

## The Fix

### Need to Verify:

1. **Mask values in observation data:**
   - Are invalid pixels marked with `mask = 0`?
   - Are valid pixels marked with `mask = 1`?
   - Or is it the opposite?

2. **Mask application in chi-squared:**
   ```python
   chi_sq += jnp.sum(((vel_map - od['vel_obs']) / od['vel_err']) ** 2 * msk)
   ```
   - If `msk=0` for invalid, this should work
   - If `msk=1` for invalid, this is backwards!

3. **Observation data loading:**
   - Did the April 15 commit change how `obs.mask` is loaded?
   - Is there an inversion somewhere?

---

## Next Steps

### Immediate: Verify Mask Convention

Run `check_mask_values.py` to confirm:
1. What values does `obs.mask` contain?
2. Is `mask=1` for valid or invalid?
3. Is `mask=0` for valid or invalid?

### Test: Revert to Pre-April 15 Code

```bash
# Test the version before geometry tracing
git checkout 53beeae~1  # Just before the breaking commit
python demo/demo_2D_fitting_JAXNS.py
# Check if weight evolution is normal
```

### Test: Force 5-Parameter Fitting

Temporarily disable geometry parameter fitting to see if that fixes it.

---

## Files Created

All in `dev/debug_jaxns/`:

1. **`analyze_jaxns_results.py`** - Shows extreme likelihood values
2. **`check_mask_values.py`** - Verifies mask convention
3. **`debug_likelihood.py`** - Tests chi-squared computation
4. **`INVESTIGATION_SUMMARY.md`** - Initial findings
5. **`COMPLETE_ANALYSIS.md`** - This file

---

## Commits

- `9669ff9` - Add debug scripts
- `0dc5fc4` - Add investigation summary

---

## Recommendation

**Most Likely Fix:** The mask is inverted or not being applied correctly after the April 15 commit. The observation data preparation code needs to be reviewed to ensure invalid pixels (with `vel_obs=-1e6`) are properly masked out (contribute zero to chi-squared).

**Verification:** Run `check_mask_values.py` to confirm the mask convention, then fix the mask handling in the likelihood computation.

---

**Status:** Ready for fix - just need to verify mask values
**Priority:** High - JAXNS results are unreliable
**Confidence:** High - breaking commit identified
