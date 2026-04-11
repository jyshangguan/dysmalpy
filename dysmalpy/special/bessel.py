"""
Modified Bessel functions of the second kind K0(y) and K1(y), JAX-traceable.

Uses scipy.special.k0/k1 via jax.pure_callback for exact accuracy,
with a custom JVP rule providing numerical gradients.

Typical usage in dysmalpy: exponential disk circular velocity:
    y = r / (2 * r_d)
    expdisk = y^2 * (I0(y)*K0(y) - I1(y)*K1(y))
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as _sp

from jax.core import ShapedArray


def _k0_numpy(y_arr):
    """K0 via scipy, handles both scalars and arrays."""
    y_np = np.asarray(y_arr, dtype=np.float64)
    return _sp.k0(y_np)


def _k1_numpy(y_arr):
    """K1 via scipy, handles both scalars and arrays."""
    y_np = np.asarray(y_arr, dtype=np.float64)
    return _sp.k1(y_np)


@jax.custom_jvp
def bessel_k0(y):
    """
    Modified Bessel function of the second kind, order 0: K0(y).

    Parameters
    ----------
    y : array_like
        Argument. Must be > 0.

    Returns
    -------
    K0 : array_like
        Value of the modified Bessel function K0(y).
    """
    y_safe = jnp.maximum(y, 1e-30)
    return jax.pure_callback(
        _k0_numpy,
        ShapedArray(jnp.shape(y_safe), jnp.float64),
        y_safe,
    )


@bessel_k0.defjvp
def bessel_k0_jvp(primals, tangents):
    y, = primals
    y_dot, = tangents
    eps = 1e-7
    f = bessel_k0(y)
    fp = bessel_k0(y + eps)
    fm = bessel_k0(y - eps)
    df = (fp - fm) / (2.0 * eps)
    return f, df * y_dot


@jax.custom_jvp
def bessel_k1(y):
    """
    Modified Bessel function of the second kind, order 1: K1(y).

    Parameters
    ----------
    y : array_like
        Argument. Must be > 0.

    Returns
    -------
    K1 : array_like
        Value of the modified Bessel function K1(y).
    """
    y_safe = jnp.maximum(y, 1e-30)
    return jax.pure_callback(
        _k1_numpy,
        ShapedArray(jnp.shape(y_safe), jnp.float64),
        y_safe,
    )


@bessel_k1.defjvp
def bessel_k1_jvp(primals, tangents):
    y, = primals
    y_dot, = tangents
    eps = 1e-7
    f = bessel_k1(y)
    fp = bessel_k1(y + eps)
    fm = bessel_k1(y - eps)
    df = (fp - fm) / (2.0 * eps)
    return f, df * y_dot
