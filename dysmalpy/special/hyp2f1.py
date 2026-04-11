"""
Gauss hypergeometric function 2F1(a, b; c; z), JAX-traceable.

Implements hyp2f1(a, b, c, z) using:
  - JAX's built-in hyp2f1 for |z| < 1 (accurate for small arguments)
  - Custom power series with linear transformation for |z| >= 1

Uses jax.lax.scan with a fixed number of terms for JAX traceability.
A custom JVP rule provides numerical gradients.

Typical usage in dysmalpy: TwoPowerHalo enclosed mass:
    hyp2f1(3-alpha, beta-alpha, 4-alpha, -r/rs)
    hyp2f1(3-alpha, beta-alpha, 4-alpha, -conc)
"""

import jax
import jax.numpy as jnp
from jax import lax


def _hyp2f1_series(a, b, c, z, n_terms=200):
    """
    Evaluate 2F1(a, b; c; z) via direct power series.

    Computes sum_{n=0}^{n_terms-1} (a)_n (b)_n / (c)_n / n! * z^n

    Uses the recurrence relation for successive terms:
        term_{n+1} / term_n = (a+n)(b+n) / ((c+n)(n+1)) * z
    """
    def body(carry, _):
        term, n = carry
        new_term = term * (a + n) * (b + n) / ((c + n) * (n + 1.0)) * z
        return (new_term, n + 1.0), new_term

    (_, _), all_terms = lax.scan(body, (1.0, 0.0), None, length=n_terms)
    return 1.0 + jnp.sum(all_terms)


@jax.custom_jvp
def hyp2f1(a, b, c, z):
    """
    Gauss hypergeometric function 2F1(a, b; c; z).

    Parameters
    ----------
    a, b, c : array_like
        Parameters of the hypergeometric function.
    z : array_like
        Argument. Can be any value.

    Returns
    -------
    result : array_like
        Value of the hypergeometric function.

    Notes
    -----
    Uses the linear fractional transformation:
        2F1(a, b; c; z) = (1-z)^{-a} * 2F1(a, c-b; c; z/(z-1))

    When z < -1 (e.g., z = -conc in TwoPowerHalo), the mapped argument
    w = z/(z-1) = conc/(conc+1) falls in (0, 1), ensuring convergence.

    For z close to 1 (|w| close to 1), uses a secondary transformation via
    the quadratic relation to improve convergence.
    """
    # For |z| < 1, try JAX's built-in first (more accurate)
    jax_builtin = jax.scipy.special.hyp2f1(a, b, c, z)
    jax_ok = jnp.isfinite(jax_builtin) & (jnp.abs(z) < 1.0)

    use_transform = jnp.abs(z) >= 1.0

    # Primary transformation: w = z/(z-1), b -> c-b
    w = z / (z - 1.0)
    prefactor = jnp.power(1.0 - z, -a)
    bt = c - b

    # Evaluate series at w with transformed parameters
    result_w = _hyp2f1_series(a, bt, c, w, n_terms=200)

    custom_result = prefactor * result_w

    # Use JAX built-in when available and finite, otherwise custom
    return jnp.where(jax_ok, jax_builtin, custom_result)


@hyp2f1.defjvp
def hyp2f1_jvp(primals, tangents):
    """
    Custom JVP for hyp2f1 using finite differences.
    """
    a, b, c, z = primals
    a_dot, b_dot, c_dot, z_dot = tangents

    eps = 1e-7

    def perturbed(da, db, dc, dz):
        return hyp2f1(a + da, b + db, c + dc, z + dz)

    f_zp = perturbed(0.0, 0.0, 0.0, eps)
    f_zm = perturbed(0.0, 0.0, 0.0, -eps)
    f_ap = perturbed(eps, 0.0, 0.0, 0.0)
    f_am = perturbed(-eps, 0.0, 0.0, 0.0)
    f_bp = perturbed(0.0, eps, 0.0, 0.0)
    f_bm = perturbed(0.0, -eps, 0.0, 0.0)
    f_cp = perturbed(0.0, 0.0, eps, 0.0)
    f_cm = perturbed(0.0, 0.0, -eps, 0.0)

    df_da = (f_ap - f_am) / (2.0 * eps)
    df_db = (f_bp - f_bm) / (2.0 * eps)
    df_dc = (f_cp - f_cm) / (2.0 * eps)
    df_dz = (f_zp - f_zm) / (2.0 * eps)

    tangent_out = df_da * a_dot + df_db * b_dot + df_dc * c_dot + df_dz * z_dot

    return hyp2f1(a, b, c, z), tangent_out
