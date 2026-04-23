---
name: special-functions
description: >
  This skill should be used when the user asks to "add a special function",
  "make a function JAX-traceable", "implement gammaincinv", "add a bessel
  function", "fix hyp2f1 precision", "wrap scipy in jax.pure_callback",
  "add a custom JVP rule", or "implement an iterative solver with jax.lax.scan".
---

# JAX-Traceable Special Functions

## Module Overview

The `dysmalpy/special/` package provides JAX-traceable replacements for scipy
special functions used in galaxy profile evaluations.

| Function | File | Method | Used by |
|----------|------|--------|---------|
| `gammaincinv(a, p)` | `gammaincinv.py` | Newton-Raphson + `jax.lax.scan` | Sersic bn parameter |
| `hyp2f1(a, b, c, z)` | `hyp2f1.py` | Power series + linear fractional transform | TwoPowerHalo enclosed mass |
| `bessel_k0(y)` | `bessel.py` | `jax.pure_callback` wrapping scipy | Exponential disk vcirc |
| `bessel_k1(y)` | `bessel.py` | `jax.pure_callback` wrapping scipy | Exponential disk vcirc |

All functions are imported via `from dysmalpy.special import gammaincinv, hyp2f1, bessel_k0, bessel_k1`.

## JAX Traceability Requirements

Functions called inside `jax.jit` must satisfy:

1. **No Python control flow** — no `if/elif/else` on traced values.  Use
   `jnp.where()` or `jax.lax.cond()`.
2. **No scipy** — scipy functions are not JAX-traceable.  Wrap them with
   `jax.pure_callback` if needed (as done for bessel).
3. **No Python `for` loops with traced bounds** — use `jax.lax.scan()`.
4. **Fixed iteration counts** — `jax.lax.scan` requires a known-length loop.

## Function Implementations

### `gammaincinv` — Newton-Raphson with `jax.lax.scan`

Computes `gammaincinv(a, p)` such that `gammainc(a, x) = p`.

```python
@jax.custom_jvp
def gammaincinv(a, p):
    # Initial guess: x ~ a * p^(1/a)
    x = a * jnp.power(p, 1.0 / a)

    def body(x, _):
        fx = gammainc(a, x) - p
        fpx = jnp.exp(-x + (a - 1) * jnp.log(x)) / gamma_a
        x_new = x - fx / fpx
        x_new = jnp.maximum(x_new, 1e-15)
        return x_new, None

    x_final, _ = jax.lax.scan(body, x, None, length=30)
    return x_final
```

A custom JVP rule provides exact analytical gradients (inverse of the forward
Jacobian: `dx/da`, `dx/dp`).

**Key points:**
- 30 Newton-Raphson steps is sufficient for `rtol=1e-6` across the parameter
  range relevant to galaxy modeling (a = 0.5 to 10, p = 0.01 to 0.99).
- `jnp.maximum(x, 1e-15)` prevents log of zero.
- Uses `jax.scipy.special.gammainc` and `jax.scipy.special.gamma` (both JAX-traceable).

### `hyp2f1` — Power Series + Linear Fractional Transform

Computes `2F1(a, b; c; z)` using two strategies depending on `|z|`:

- **`|z| < 1`**: JAX's built-in `jax.scipy.special.hyp2f1` (accurate).
- **`|z| >= 1`**: Custom power series via `jax.lax.scan` with 200 terms,
  applying the linear fractional transformation:
  `2F1(a,b;c;z) = (1-z)^{-a} * 2F1(a, c-b; c; z/(z-1))`
  to convert to `|z/(z-1)| < 1`.

A custom JVP rule provides numerical gradients via finite differences.

**Key points:**
- The LFT maps `z >= 1` to `z' = z/(z-1)` where `|z'| < 1`, ensuring convergence.
- 200 series terms is sufficient for the halo profile parameter range.
- Used by `TwoPowerHalo.menc()` and `TwoPowerHalo.vcirc()`.

### `bessel_k0` / `bessel_k1` — `jax.pure_callback` wrapping scipy

Uses `jax.pure_callback` to call scipy's highly accurate Bessel function
implementations from within a JAX-traced computation:

```python
@jax.custom_jvp
def bessel_k0(y):
    y_safe = jnp.maximum(y, 1e-30)
    return jax.pure_callback(
        _k0_numpy,                       # host function (scipy)
        ShapedArray(jnp.shape(y_safe), jnp.float64),  # output spec
        y_safe,
    )
```

Custom JVP rules provide numerical gradients via finite differences
(`eps = 1e-7` central differences).

**Key points:**
- `pure_callback` is a JAX mechanism for calling arbitrary Python/NumPy/SciPy
  functions from within traced code.  The callback runs on CPU.
- The output shape and dtype must be declared via `ShapedArray`.
- `jnp.maximum(y, 1e-30)` prevents K0/K1 singularity at y=0.
- Used by `vcirc_exp_disk()` and `menc_exp_disk()`.

## How to Add a New Special Function

### Step 1 — Create the JAX implementation

Create a new file in `dysmalpy/special/` (e.g., `my_func.py`):

```python
import jax
import jax.numpy as jnp

@jax.custom_jvp
def my_func(x, *args):
    """JAX-traceable implementation."""
    # ... your computation using jnp, jax.lax.scan, etc. ...
    return result

@my_func.defjvp
def my_func_jvp(primals, tangents):
    x, *rest = primals
    x_dot, *rest_dot = tangents
    # Analytical or numerical gradient
    eps = 1e-7
    f_plus = my_func(x + eps, *rest)
    f_minus = my_func(x - eps, *rest)
    return my_func(x, *rest), (f_plus - f_minus) / (2 * eps) * x_dot
```

### Step 2 — Register in `special/__init__.py`

```python
from .my_func import my_func
```

And add to `__all__` if needed.

### Step 3 — Add tests in `tests/test_jax.py`

```python
class TestMyFunc:
    def test_known_values(self):
        # Compare against scipy or known analytical results
        result = float(my_func(1.0, 2.0))
        expected = scipy.special.my_func(1.0, 2.0)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_jit(self):
        f_jit = jax.jit(my_func)
        result = float(f_jit(1.0, 2.0))
        assert not jnp.isnan(result)

    def test_grad(self):
        f_jit = jax.jit(my_func)
        grad_fn = jax.grad(lambda x: f_jit(x, 2.0))
        grad_val = float(grad_fn(1.0))
        assert not jnp.isnan(grad_val) and not jnp.isinf(grad_val)
```

## Choosing an Implementation Strategy

| Strategy | When to use | Pros | Cons |
|----------|-------------|------|------|
| Pure JAX math | Simple functions, iterative solvers | Fully traceable, GPU-compatible | Must avoid scipy |
| `jax.pure_callback` | Complex scipy functions | Exact accuracy, easy to implement | Not GPU-accelerated, no auto-grad |
| `jax.scipy.special` | Functions already in JAX | Built-in, fast, auto-grad | Limited set of functions |
| Power series + LFT | Hypergeometric-type functions | Convergent for all arguments | May need many terms |

## Dependency on `jax_enable_x64`

All special functions require float64 for accuracy comparable to scipy.
`JAX_ENABLE_X64=1` is set in `dysmalpy/__init__.py` before any JAX import.
If this is missing, functions will silently compute in float32 with reduced
precision (~7 decimal digits vs ~15).

## Key Files

| File | Description |
|------|-------------|
| `dysmalpy/special/gammaincinv.py` | Inverse incomplete gamma, Newton-Raphson |
| `dysmalpy/special/hyp2f1.py` | Gauss hypergeometric 2F1, power series + LFT |
| `dysmalpy/special/bessel.py` | Modified Bessel K0/K1, pure_callback |
| `dysmalpy/special/__init__.py` | Public exports |
| `tests/test_jax.py` | Unit tests for all special functions |
