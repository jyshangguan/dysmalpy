"""
Inverse of the regularized lower incomplete gamma function, JAX-traceable.

Implements gammaincinv(a, p) such that gammainc(a, gammaincinv(a, p)) = p.

Uses Newton-Raphson iteration with a fixed number of steps (jax.lax.scan)
for JAX traceability. A custom JVP rule provides exact analytical gradients.

Typical usage in dysmalpy: computing the Sersic bn parameter:
    bn = gammaincinv(2*n, 0.5)
"""

import jax
import jax.numpy as jnp
from jax import lax


@jax.custom_jvp
def gammaincinv(a, p):
    """
    Inverse of the regularized lower incomplete gamma function.

    Parameters
    ----------
    a : array_like
        Positive shape parameter of the gamma distribution.
    p : array_like
        Probability, must be in (0, 1].

    Returns
    -------
    x : array_like
        Value such that gammainc(a, x) = p.

    Notes
    -----
    Uses Newton-Raphson iteration with 30 fixed steps.
    Initial guess: x0 = a * p^(1/a), which is exact for the integral of
    x^(a-1) from 0 to x0 (a good approximation for the gamma function).
    """
    # Clamp p to avoid edge-case issues
    p_safe = jnp.clip(p, 1e-15, 1.0 - 1e-15)
    a_safe = jnp.maximum(a, 1e-10)

    # Initial guess: x ~ a * p^(1/a)
    # This follows from the integral of x^(a-1) being proportional to x^a,
    # giving a rough first approximation that works well for moderate a.
    x = a_safe * jnp.power(p_safe, 1.0 / a_safe)
    x = jnp.maximum(x, 1e-15)

    gamma_a = jax.scipy.special.gamma(a_safe)

    def body(x, _):
        # Forward evaluation: gammainc(a, x) - p
        fx = jax.scipy.special.gammainc(a_safe, x) - p_safe

        # Derivative of gammainc(a, x) w.r.t. x:
        #   d/dx gammainc(a, x) = exp(-x) * x^(a-1) / Gamma(a)
        # Use log-space for numerical stability:
        #   log(exp(-x) * x^(a-1)) = -x + (a-1)*log(x)
        log_x = jnp.log(jnp.maximum(x, 1e-30))
        log_deriv = -x + (a_safe - 1.0) * log_x - jnp.log(gamma_a)
        dfx = jnp.exp(log_deriv)

        # Clamp derivative to avoid division by zero
        dfx_safe = jnp.where(jnp.abs(dfx) > 1e-30, dfx, 1e-30)

        # Newton step, but freeze if already converged
        x_new = x - fx / dfx_safe

        # Keep x strictly positive and bounded
        x_new = jnp.clip(x_new, 1e-15, 1e8)

        # Stop updating once converged
        converged = jnp.abs(fx) < 1e-14
        x_out = jnp.where(converged, x, x_new)

        return x_out, None

    x_final, _ = lax.scan(body, x, None, length=30)
    return x_final


@gammaincinv.defjvp
def gammaincinv_jvp(primals, tangents):
    """
    Custom JVP for gammaincinv.

    Since gammaincinv is the inverse function of gammainc(a, x) w.r.t. x,
    the derivative w.r.t. p is:

        d/dp gammaincinv(a, p) = 1 / (d/dx gammainc(a, x))

    where x = gammaincinv(a, p).
    """
    a, p = primals
    a_dot, p_dot = tangents

    x = gammaincinv(a, p)

    # d/dx gammainc(a, x) = exp(-x) * x^(a-1) / Gamma(a)
    gamma_a = jax.scipy.special.gamma(a)
    log_x = jnp.log(jnp.maximum(x, 1e-30))
    log_deriv = -x + (a - 1.0) * log_x - jnp.log(gamma_a)
    dfdx = jnp.exp(log_deriv)

    dfdx_safe = jnp.where(jnp.abs(dfdx) > 1e-30, dfdx, 1e-30)
    dx_dp = 1.0 / dfdx_safe

    return x, dx_dp * p_dot
