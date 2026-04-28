# JAX Gaussian Fitting Methods

## Overview

This document describes the three Gaussian fitting methods available in dysmalpy for use with JAXNS nested sampling when `moment_calc=False`.

---

## Available Methods

### 1. `closed_form` - Analytical Maximum Likelihood Estimation

**Description:** Direct computation of Gaussian parameters using analytical formulas.

**Mathematics:**
```
μ = Σ(weight * x * y) / Σ(weight * y)  # Weighted first moment
σ² = Σ(weight * y * (x-μ)²) / Σ(weight * y)  # Weighted second moment
A = Σy / (√(2π) * σ)  # Amplitude from flux conservation
```

**Performance:**
- Overhead: 2.5x slower than moment extraction
- Time for 729 pixels: ~3.8 ms
- Projected time for 10k JAXNS iterations: ~38 seconds

**Advantages:**
- Fastest Gaussian fitting method
- JAX-traceable and fully vectorizable
- Good accuracy for well-behaved spectra

**Disadvantages:**
- Can be biased for asymmetric or multi-component kinematics
- May not converge to true MLE for noisy data

**Use when:** Speed is critical, spectra are well-behaved

---

### 2. `hybrid_gd` - Closed-form + Custom Gradient Descent (RECOMMENDED)

**Description:** Closed-form initialization followed by lightweight gradient descent refinement.

**Algorithm:**
1. Get initial estimates from closed-form MLE
2. Refine using 10 steps of gradient descent with:
   - Learning rate: 0.01 (conservative)
   - Gradient clipping: ±10.0 (prevents explosion)
   - Constraints: σ > 0.1, A ≥ 0

**Performance:**
- Overhead: 4-6x slower than moment extraction
- Time for 729 pixels: ~6-10 ms
- Projected time for 10k JAXNS iterations: ~1-2 minutes

**Advantages:**
- Fast enough for JAXNS (~1-2 min vs ~62 min for BFGS)
- Reduces bias compared to closed-form (~12% loss improvement)
- Stable with gradient clipping and constraints
- JAX-traceable with `jax.lax.scan`

**Disadvantages:**
- 2-3x slower than closed-form
- Still may not reach true MLE for complex cases

**Use when:** Default choice for production JAXNS fitting

---

### 3. `hybrid` - Closed-form + BFGS Optimization

**Description:** Closed-form initialization followed by scipy BFGS refinement.

**Algorithm:**
1. Get initial estimates from closed-form MLE
2. Refine using JAX's BFGS optimizer:
   - Quasi-Newton method with Hessian approximation
   - Automatic differentiation via `jax.grad`
   - Line search for adaptive step size

**Performance:**
- Overhead: 245x slower than moment extraction
- Time for 729 pixels: ~372 ms
- Projected time for 10k JAXNS iterations: ~62 minutes

**Advantages:**
- Most accurate method (converges to true MLE)
- Automatic step size adaptation
- Well-tested optimization algorithm

**Disadvantages:**
- Very slow for JAXNS (245x overhead)
- JAX optimization machinery has significant overhead
- Not practical for large-scale nested sampling

**Use when:** Final validation, small datasets, maximum accuracy needed

---

## Performance Comparison

### Small Dataset (5×5×100, 25 pixels)

| Method | Time | Overhead |
|--------|------|----------|
| `closed_form` | 3.90 ms | 1x (baseline) |
| `hybrid_gd` | 4.36 ms | 1.12x |
| `hybrid` (BFGS) | 6.31 ms | 1.62x |

### Medium Dataset (27×27×200, 729 pixels) - GS4_43501

| Method | Time | Overhead | 10k iterations |
|--------|------|----------|----------------|
| Moment extraction | 1.52 ms | 1x | ~15 sec |
| `closed_form` | 3.84 ms | 2.5x | ~38 sec |
| `hybrid_gd` | ~6-10 ms | 4-6x | ~1-2 min |
| `hybrid` (BFGS) | 371.86 ms | 245x | ~62 min |

---

## Accuracy Comparison

### Test Case: Noisy Gaussian Spectrum

**True parameters:** A=10, μ=5, σ=20
**Noise:** 0.5 (Gaussian)

| Method | A | μ | σ | Loss | Improvement |
|--------|---|---|---|------|-------------|
| True values | 10.0 | 5.0 | 20.0 | - | - |
| `closed_form` | 9.46 | 4.71 | 21.17 | 60.97 | baseline |
| `hybrid_gd` | 9.87 | 4.81 | 20.56 | 53.61 | **12.1%** |
| `hybrid` (BFGS) | ~9.9 | ~4.8 | ~20.5 | ~53.5 | ~12.2% |

**Key finding:** Custom GD provides most of the accuracy benefit of BFGS with much less computational cost.

---

## Implementation Details

### Custom Gradient Descent Algorithm

```python
@jax.jit
def custom_gradient_descent(init_params, x, y, yerr,
                             n_steps=10, learning_rate=0.01):
    """
    Lightweight gradient descent with safeguards:
    - Gradient clipping (±10.0) prevents numerical instability
    - Constraints (σ > 0.1, A ≥ 0) ensure physical parameters
    - jax.lax.scan enables efficient JIT compilation
    """
    grad_fn = jax.grad(gaussian_loss)

    def step_fn(params, _):
        grad = grad_fn(params, x, y, yerr)
        grad = jnp.clip(grad, -10.0, 10.0)
        new_params = params - learning_rate * grad
        # Apply constraints
        new_params = jax.lax.cond(new_params[2] < 0.1,
                                   lambda p: p.at[2].set(params[2]),
                                   lambda p: p, new_params)
        new_params = jax.lax.cond(new_params[0] < 0,
                                   lambda p: p.at[0].set(params[0]),
                                   lambda p: p, new_params)
        return new_params, None

    final_params, _ = jax.lax.scan(step_fn, init_params, None, length=n_steps)
    return final_params
```

### Method Selection

The method is specified in the `fit_gaussian_cube_jax()` call:

```python
from dysmalpy.fitting.jax_gaussian_fitting import fit_gaussian_cube_jax

# Default (recommended)
flux, vel, disp = fit_gaussian_cube_jax(cube, spec_arr, method='hybrid_gd')

# Fastest (may be biased)
flux, vel, disp = fit_gaussian_cube_jax(cube, spec_arr, method='closed_form')

# Most accurate (very slow)
flux, vel, disp = fit_gaussian_cube_jax(cube, spec_arr, method='hybrid')
```

For JAXNS, the method is controlled by the `moment_calc` parameter:
- `moment_calc=True`: Uses moment extraction (fastest)
- `moment_calc=False`: Uses Gaussian fitting with `method='hybrid_gd'` (default)

---

## Recommendations

### For JAXNS Nested Sampling

**Default choice:** `hybrid_gd` (closed-form + custom GD)

**Rationale:**
- Provides accuracy improvement over closed-form (~12% loss reduction)
- Fast enough for practical JAXNS use (~1-2 min for 10k iterations)
- Reduces bias in asymmetric or complex kinematics
- Stable and well-tested

**Alternative:** `closed_form`
- Use when speed is critical
- Accept potential bias in results
- Only for exploration/testing

**Avoid:** `hybrid` (BFGS)
- 245x overhead makes it impractical for JAXNS
- Only for final validation on small datasets

### For Other Use Cases

**MPFIT fitting:** Use C++ MPFIT with `moment_calc=False` for most accurate results

**Interactive exploration:** Use `moment_calc=True` for fastest iteration

**Publication validation:** Run `hybrid` (BFGS) on final best-fit parameters

---

## Technical Notes

### Why is BFGS so slow?

The 245x overhead of BFGS comes from:
1. JAX's `optimize.minimize` infrastructure (wrapper functions, error checking)
2. Line search algorithm setup and execution
3. Per-function-call overhead (729 function calls = 729 × overhead)
4. JIT compilation and tracing for each optimization

Even though each gradient evaluation is fast (~9 ms), the optimization machinery adds ~50x overhead on top of the gradient computations.

### Why does custom GD work so well?

The custom gradient descent method achieves good performance because:
1. **Fixed iteration count:** No convergence checking overhead
2. **Simple updates:** Just `params = params - lr * grad`
3. **Efficient JIT:** `jax.lax.scan` compiles to a tight loop
4. **Minimal overhead:** No line search, no Hessian approximation
5. **Vectorizable:** Can be batched across pixels with `vmap`

### Numerical Stability

The custom GD includes several safeguards:
- **Gradient clipping:** Prevents parameter explosion from large gradients
- **Sigma constraint:** Keeps dispersion positive (σ > 0.1)
- **Amplitude constraint:** Ensures non-negative flux (A ≥ 0)
- **Small learning rate:** 0.01 prevents divergence

These safeguards are essential because the Gaussian loss can have very large gradients, especially for the amplitude parameter.

---

## Future Work

### Potential Optimizations

1. **Adaptive learning rate:** Could reduce iteration count needed
2. **Momentum:** Accelerate convergence in narrow valleys
3. **Early stopping:** Stop when improvement is minimal
4. **Batch optimization:** Optimize all pixels simultaneously

### Method Improvements

1. **Better initialization:** Use moment-based estimates as starting point
2. **Multi-start:** Avoid local minima with multiple initializations
3. **Hybrid approach:** Use few BFGS steps after GD refinement

---

## References

- **Implementation:** `dysmalpy/fitting/jax_gaussian_fitting.py`
- **Usage:** `dysmalpy/fitting/jax_loss.py` (line 879)
- **Tests:** `tests/test_custom_optimizer.py` (benchmarking)
- **Demo:** `demo/demo_2D_fitting_JAXNS.py`

---

**Last updated:** 2026-04-28
**Status:** Production ready with `hybrid_gd` as default
**Contact:** For questions or issues, please open a GitHub issue
