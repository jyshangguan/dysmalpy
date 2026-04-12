# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information.
#
# JAX-native FFT convolution utilities.
#
# Provides a JAX-traceable 3D FFT convolution that replicates
# ``scipy.signal.fftconvolve(mode='same')``, enabling the full
# pipeline (theta -> simulate_cube -> convolve -> chi-squared) to be
# JIT-compiled on GPU.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ['_rebin_spatial', '_fft_convolve_3d', 'convolve_cube_jax', 'get_jax_kernels']


def _rebin_spatial(cube, new_ny, new_nx):
    """Rebin 3D cube along spatial axes via reshape+sum.

    Matches ``dysmalpy.utils.rebin``.  The spectral axis (axis 0) is
    preserved; only the y (axis 1) and x (axis 2) axes are rebinned.

    Parameters
    ----------
    cube : jnp.ndarray
        3D input array (nz, ny_in, nx_in).
    new_ny : int
        Target size along the y axis.
    new_nx : int
        Target size along the x axis.

    Returns
    -------
    jnp.ndarray
        Rebinned array with shape ``(nz, new_ny, new_nx)``.
    """
    ny_in = cube.shape[1]
    nx_in = cube.shape[2]
    reshape_shape = (cube.shape[0], new_ny, ny_in // new_ny,
                     new_nx, nx_in // new_nx)
    return cube.reshape(reshape_shape).sum(-1).sum(-2)


def _fft_convolve_3d(cube, kernel):
    """JAX-traceable 3D FFT convolution replicating ``scipy.signal.fftconvolve(mode='same')``.

    Parameters
    ----------
    cube : jnp.ndarray
        3D input array (nz, ny, nx).
    kernel : jnp.ndarray
        3D kernel array (nz, ny, nx).

    Returns
    -------
    jnp.ndarray
        Convolved array with the same shape as *cube*.
    """
    # Compute padded shape for full convolution
    full_shape = tuple(
        cube.shape[d] + kernel.shape[d] - 1 for d in range(3)
    )

    # Zero-pad both arrays using jax.lax.pad (avoids jnp.pad compatibility
    # issues with certain JAX/numpy version combinations under autodiff)
    padding_config = tuple(
        (0, full_shape[d] - cube.shape[d], 0) for d in range(3)
    )
    cube_padded = jax.lax.pad(cube, 0.0, padding_config)

    padding_config_k = tuple(
        (0, full_shape[d] - kernel.shape[d], 0) for d in range(3)
    )
    kernel_padded = jax.lax.pad(kernel, 0.0, padding_config_k)

    # FFT both, multiply, inverse FFT
    fft_cube = jnp.fft.fftn(cube_padded)
    fft_kernel = jnp.fft.fftn(kernel_padded)
    result = jnp.fft.ifftn(fft_cube * fft_kernel)

    # Take real part (imaginary part is numerical noise)
    result = jnp.real(result)

    # Crop to 'same' size
    starts = tuple((full_shape[d] - cube.shape[d]) // 2 for d in range(3))
    result = jax.lax.slice(
        result,
        start_indices=starts,
        limit_indices=tuple(starts[d] + cube.shape[d] for d in range(3)),
    )

    return result


def _crop_kernel(kernel, cube):
    """Crop kernel center to match cube spatial dimensions.

    When the kernel (e.g. PSF) is larger than the cube in any spatial
    dimension, this crops the kernel symmetrically about its center so
    that it matches the cube size.  Applied before the JAX-traced
    convolution so that padding shapes stay small.
    """
    slices = [slice(None)]  # preserve spectral dim
    for d in range(1, kernel.ndim):  # spatial dims
        if kernel.shape[d] > cube.shape[d]:
            start = (kernel.shape[d] - cube.shape[d]) // 2
            slices.append(slice(start, start + cube.shape[d]))
        else:
            slices.append(slice(None))
    return kernel[tuple(slices)]


def convolve_cube_jax(cube, beam_kernel=None, lsf_kernel=None):
    """Apply PSF (beam) and/or LSF convolution to a 3D model cube.

    Matches the convention in ``Instrument.convolve()``: beam first, then LSF.

    Parameters
    ----------
    cube : array-like
        3D model cube (nz, ny, nx).
    beam_kernel : array-like or None
        Beam kernel with shape ``(1, ky, kx)``.
    lsf_kernel : array-like or None
        LSF kernel with shape ``(nk, 1, 1)``.

    Returns
    -------
    jnp.ndarray
        Convolved cube with the same shape as the input.
    """
    cube = jnp.asarray(cube)

    if beam_kernel is not None:
        beam_kernel = _crop_kernel(np.asarray(beam_kernel), np.asarray(cube))
        cube = _fft_convolve_3d(cube, jnp.asarray(beam_kernel))

    if lsf_kernel is not None:
        lsf_kernel = _crop_kernel(np.asarray(lsf_kernel), np.asarray(cube))
        cube = _fft_convolve_3d(cube, jnp.asarray(lsf_kernel))

    return cube


def get_jax_kernels(instrument):
    """Extract pre-computed numpy kernels from an Instrument instance.

    Ensures kernels are computed (calling ``set_beam_kernel()`` /
    ``set_lsf_kernel()`` if needed) and returns them as numpy arrays
    suitable for conversion to JAX constants.

    Parameters
    ----------
    instrument : Instrument
        An Instrument instance with optional beam and/or LSF.

    Returns
    -------
    beam_kernel : np.ndarray or None
        Shape ``(1, ky, kx)`` if beam is set, else None.
    lsf_kernel : np.ndarray or None
        Shape ``(nk, 1, 1)`` if LSF is set, else None.
    """
    beam_kernel = None
    lsf_kernel = None

    if instrument.beam is not None:
        if instrument._beam_kernel is None:
            instrument.set_beam_kernel()
        beam_kernel = np.asarray(instrument._beam_kernel)

    if instrument.lsf is not None:
        if instrument._lsf_kernel is None:
            instrument.set_lsf_kernel(
                spec_center=instrument.line_center
            )
        lsf_kernel = np.asarray(instrument._lsf_kernel)

    return beam_kernel, lsf_kernel
