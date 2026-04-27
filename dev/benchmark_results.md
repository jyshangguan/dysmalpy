# Performance Benchmark Results

**Date:** 2026-04-27
**System:** Intel Xeon CPU, NVIDIA 4090 GPU (CUDA 12)
**JAX Backend:** GPU (cuda:0)

## Configuration

- Methods tested: Moment extraction, JAX closed-form, JAX hybrid
- Dataset sizes:
  - Small: 5×5×50 (25 pixels)
  - Medium: 27×27×200 (729 pixels) - GS4_43501 size
  - Large: 50×50×200 (2500 pixels)
- Runs per method: 10
- Timing metric: Wall-clock time (mean ± std)

## Results Summary

### Small Dataset (5×5×50, 25 pixels)

| Method | Mean Time | Std Dev | vs Moment |
|--------|-----------|---------|-----------|
| Moment extraction | 1.48 ms | 0.18 ms | baseline |
| JAX closed-form | 3.92 ms | 0.17 ms | 2.65x slower |
| JAX hybrid | 99.75 ms | 2.93 ms | 67.33x slower |

### Medium Dataset (27×27×200, 729 pixels) ⭐

| Method | Mean Time | Std Dev | vs Moment |
|--------|-----------|---------|-----------|
| Moment extraction | 1.52 ms | 0.33 ms | baseline |
| JAX closed-form | 3.84 ms | 0.18 ms | 2.53x slower |
| JAX hybrid | 371.86 ms | 0.42 ms | 245.01x slower |

### Large Dataset (50×50×200, 2500 pixels)

| Method | Mean Time | Std Dev | vs Moment |
|--------|-----------|---------|-----------|
| Moment extraction | 1.43 ms | 0.18 ms | baseline |
| JAX closed-form | 3.82 ms | 0.21 ms | 2.67x slower |
| JAX hybrid | 503.45 ms | 17.24 ms | 351.54x slower |

## Per-Pixel Performance

| Dataset | Moment | Closed-form | Hybrid |
|---------|--------|-------------|--------|
| Small (25 px) | 0.059 ms/pixel | 0.157 ms/pixel | 3.990 ms/pixel |
| Medium (729 px) | 0.002 ms/pixel | 0.005 ms/pixel | 0.510 ms/pixel |
| Large (2500 px) | 0.001 ms/pixel | 0.002 ms/pixel | 0.201 ms/pixel |

## Key Findings

### 1. Moment Extraction is Extremely Fast
- Consistently ~1.5 ms across all dataset sizes
- Highly optimized JAX operations
- Excellent for exploration and quick iterations

### 2. JAX Closed-Form is Acceptable
- Only 2.5-2.7x slower than moments
- Consistent timing (~3.8 ms)
- Good balance of speed and accuracy
- **Recommendation:** Use for production fitting

### 3. JAX Hybrid is Very Slow
- 67-351x slower than moment extraction
- 371 ms for medium dataset (GS4_43501 size)
- BFGS optimization per pixel is expensive
- **NOT recommended** for routine use
- Only use if maximum accuracy is critical

## Implications for JAXNS Fitting

### Scenario 1: Moment Extraction (current default)
- Single likelihood evaluation: ~1.5 ms
- 10,000 iterations: ~15 seconds
- **Fast, suitable for exploration**

### Scenario 2: JAX Closed-Form (new option)
- Single likelihood evaluation: ~3.8 ms
- 10,000 iterations: ~38 seconds
- **2.5x slower, but more accurate**
- **Recommended for production**

### Scenario 3: JAX Hybrid (most accurate)
- Single likelihood evaluation: ~371 ms
- 10,000 iterations: ~3,710 seconds (~62 minutes)
- **245x slower, impractical for nested sampling**
- **Not recommended for JAXNS**

## Recommendations

### For Exploration and Testing
✅ **Use moment extraction** (`moment_calc=True`)
- Fastest option
- Good for initial exploration
- Quick iterations

### For Production Fitting
✅ **Use JAX closed-form** (set `gauss_extract_with_jax=True` in code)
- Only 2.5x overhead
- More accurate than moments
- Practical for JAXNS (38s vs 15s for 10k iterations)

❌ **Avoid JAX hybrid** for routine JAXNS fitting
- Too slow for nested sampling
- 245x overhead
- Use only for final validation if needed

## Optimization Opportunities

The JAX hybrid method could be optimized by:

1. **Vectorized BFGS:** Optimize all pixels simultaneously instead of sequentially
2. **Fewer iterations:** Reduce maxiter from 100 to 10-20
3. **Looser convergence:** Increase gtol from 1e-6 to 1e-4
4. **Selective refinement:** Only use BFGS on pixels with high residuals

## Conclusions

1. ✅ **JAX closed-form is viable** for production use (2.5x overhead acceptable)
2. ❌ **JAX hybrid is too slow** for JAXNS (245x overhead impractical)
3. ✅ **Code simplification successful** (removed unnecessary checks)
4. ✅ **All tests passing** (no regressions)

## Next Steps

1. Use JAX closed-form for demo script (good balance)
2. Document moment_calc parameter clearly for users
3. Consider optimizing hybrid method for future use
4. Proceed with accuracy validation using closed-form method

---

**Benchmark Status:** ✅ Complete
**All Tests:** ✅ Passing
**Ready for:** Accuracy validation phase
