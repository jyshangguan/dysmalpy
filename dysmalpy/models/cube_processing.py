# coding=utf8
# JAX-accelerated cube population functions.
#
# Provides:
#   - populate_cube_jax: Replaces Cython cutils.populate_cube
#   - populate_cube_jax_ais: Sparse variant for truncated LOS
#   - _simulate_cube_inner_direct / _simulate_cube_inner_ais: JIT wrappers
#   - Helper functions for coordinate grid generation

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import jax
import jax.numpy as jnp

from collections import OrderedDict

__all__ = ['populate_cube_jax', 'populate_cube_jax_ais',
           '_simulate_cube_inner_direct', '_simulate_cube_inner_ais',
           '_make_cube_ai', '_get_xyz_sky_gal', '_get_xyz_sky_gal_inverse',
           '_calculate_max_skyframe_extents']


# ===================================================================
# Helper: populate_cube_jax  (replaces Cython cutils.populate_cube)
# ===================================================================

@jax.jit
def populate_cube_jax(flux, vel, sigma, vspec):
    """JAX-vectorised cube population replacing the Cython implementation.

    For every spectral channel the flux-weighted Gaussian at each spatial
    pixel is evaluated and summed along the line-of-sight (first axis).

    Parameters
    ----------
    flux : jnp.ndarray, shape ``(nz, ny, nx)``
        Flux at each spatial position per z-slice.
    vel : jnp.ndarray, shape ``(nz, ny, nx)``
        Line-of-sight velocity at each spatial position per z-slice.
    sigma : jnp.ndarray, shape ``(nz, ny, nx)``
        Velocity dispersion at each spatial position per z-slice.
    vspec : jnp.ndarray, shape ``(nspec,)``
        Spectral channel velocities (km/s).

    Returns
    -------
    cube : jnp.ndarray, shape ``(nspec, ny, nx)``
        Simulated data cube.
    """
    nspec = vspec.shape[0]
    ny, nx = flux.shape[1], flux.shape[2]

    # Pre-compute the amplitude of each Gaussian slab.
    # flux/sigma gives the integrated flux per slab; dividing by
    # sqrt(2*pi*sigma^2) normalises the Gaussian to unit integral.
    amp = flux / jnp.sqrt(2.0 * jnp.pi * sigma)

    def _body(vs):
        """Evaluate one spectral channel (vmap-friendly)."""
        gaussian_3d = amp * jnp.exp(-0.5 * ((vs - vel) / sigma) ** 2)
        cube_slice = jnp.sum(gaussian_3d, axis=0)  # sum over z
        return cube_slice

    cube_final = jax.vmap(_body)(vspec)
    return cube_final


# ===================================================================
# Helper: populate_cube_jax_ais  (sparse variant)
# ===================================================================

@jax.jit
def populate_cube_jax_ais(flux, vel, sigma, vspec, ai):
    """Sparse cube population using an active-pixel index array.

    This is the JAX equivalent of ``cutils.populate_cube_ais``.  Only the
    pixels whose indices are listed in *ai* are propagated, which can save
    significant memory when ``zcalc_truncate=True``.

    Parameters
    ----------
    flux, vel, sigma : jnp.ndarray, shape ``(nz, ny, nx)``
        Same meaning as :func:`populate_cube_jax`.
    vspec : jnp.ndarray, shape ``(nspec,)``
        Spectral channel velocities (km/s).
    ai : jnp.ndarray, shape ``(3, n_active)``
        Index array.  ``ai[0]`` = x indices, ``ai[1]`` = y indices,
        ``ai[2]`` = z indices of the active pixels.

    Returns
    -------
    cube : jnp.ndarray, shape ``(nspec, ny, nx)``
        Simulated data cube.
    """
    nspec = vspec.shape[0]

    # Extract active-pixel coordinates
    xi = ai[0]  # x indices
    yi = ai[1]  # y indices
    zi = ai[2]  # z indices

    # Gather the relevant quantities at the active pixels
    f_active = flux[zi, yi, xi]
    v_active = vel[zi, yi, xi]
    s_active = sigma[zi, yi, xi]

    amp = f_active / jnp.sqrt(2.0 * jnp.pi * s_active)

    def _body(vs):
        """Evaluate one spectral channel and scatter into cube slice."""
        gaussian = amp * jnp.exp(-0.5 * ((vs - v_active) / s_active) ** 2)
        cube_slice = jnp.zeros((flux.shape[1], flux.shape[2]))
        cube_slice = cube_slice.at[yi, xi].add(gaussian)
        return cube_slice

    cube_final = jax.vmap(_body)(vspec)
    return cube_final


# ===================================================================
# JIT-compiled cube simulation helper (parameter-dependent inner loop)
# ===================================================================

@jax.jit
def _simulate_cube_inner_direct(flux_mass, vobs_mass, sigmar, vx_jax):
    """JIT-compiled inner loop for direct-transform cube simulation.

    Parameters
    ----------
    flux_mass : jnp.ndarray, shape ``(nz, ny, nx)``
        Flux at each spatial position per z-slice.
    vobs_mass : jnp.ndarray, shape ``(nz, ny, nx)``
        Observed LOS velocity at each spatial position per z-slice.
    sigmar : jnp.ndarray, shape ``(nz, ny, nx)``
        Velocity dispersion at each spatial position per z-slice.
    vx_jax : jnp.ndarray, shape ``(nspec,)``
        Spectral channel velocities (km/s).

    Returns
    -------
    cube : jnp.ndarray, shape ``(nspec, ny, nx)``
        Simulated data cube.
    """
    return populate_cube_jax(flux_mass, vobs_mass, sigmar, vx_jax)


@jax.jit
def _simulate_cube_inner_ais(flux_mass, vobs_mass, sigmar, vx_jax, ai_jax):
    """JIT-compiled inner loop for truncated (ais) cube simulation.

    Parameters
    ----------
    flux_mass : jnp.ndarray, shape ``(nz, ny, nx)``
    vobs_mass : jnp.ndarray, shape ``(nz, ny, nx)``
    sigmar : jnp.ndarray, shape ``(nz, ny, nx)``
    vx_jax : jnp.ndarray, shape ``(nspec,)``
    ai_jax : jnp.ndarray, shape ``(3, n_active)``
        Active pixel index array.

    Returns
    -------
    cube : jnp.ndarray, shape ``(nspec, ny, nx)``
    """
    return populate_cube_jax_ais(flux_mass, vobs_mass, sigmar, vx_jax, ai_jax)


# ===================================================================
# Cube construction helpers (initialisation-only, numpy-based)
# ===================================================================

def _make_cube_ai(model, xgal, ygal, zgal, n_wholepix_z_min=3,
                  pixscale=None, oversample=None, dscale=None,
                  maxr=None, maxr_y=None):
    """Identify active pixels for truncated LOS propagation.

    This is a *numpy-only* helper used during initialisation / setup --
    it is **not** traced by JAX.

    Parameters
    ----------
    model : ModelSet
        The model set (needed for the z-profile scale length).
    xgal, ygal, zgal : np.ndarray
        Galaxy-frame coordinate grids (pixel units).
    n_wholepix_z_min : int
        Minimum number of whole z-pixels to sample.
    pixscale : float
        Pixel scale in arcsec/pixel (after oversampling).
    oversample : int
        Oversampling factor.
    dscale : float
        Conversion arcsec/kpc.
    maxr, maxr_y : float
        Maximum x/y extents in pixel units.

    Returns
    -------
    ai : np.ndarray, shape ``(3, n_active)``
        Index array of active pixels ``(x_idx, y_idx, z_idx)``.
    """
    oversize = 1.5  # Padding factor for x trimming

    thick = model.zprofile.z_scalelength
    if not np.isfinite(thick):
        thick = 0.0

    xsize = int(np.floor(2.0 * (maxr * oversize) + 0.5))
    ysize = int(np.floor(2.0 * maxr_y + 0.5))

    zsize = np.max([
        n_wholepix_z_min * oversample,
        int(np.floor(4.0 * thick / pixscale * dscale + 0.5)),
    ])

    if (xsize % 2) < 0.5:
        xsize += 1
    if (ysize % 2) < 0.5:
        ysize += 1
    if (zsize % 2) < 0.5:
        zsize += 1

    zi, yi, xi = np.indices(xgal.shape)
    full_ai = np.vstack([xi.flatten(), yi.flatten(), zi.flatten()])

    origpos = np.vstack([
        xgal.flatten() - np.mean(xgal.flatten()) + xsize / 2.0,
        ygal.flatten() - np.mean(ygal.flatten()) + ysize / 2.0,
        zgal.flatten() - np.mean(zgal.flatten()) + zsize / 2.0,
    ])

    validpts = np.where(
        (origpos[0, :] >= 0.0) & (origpos[0, :] <= xsize) &
        (origpos[1, :] >= 0.0) & (origpos[1, :] <= ysize) &
        (origpos[2, :] >= 0.0) & (origpos[2, :] <= zsize)
    )[0]

    ai = full_ai[:, validpts]
    return ai


def _get_xyz_sky_gal(geom, sh, xc_samp, yc_samp, zc_samp):
    """Get sky/galaxy coordinate grids, regular in sky frame.

    Parameters
    ----------
    geom : Geometry
        Geometry model (must be callable: ``geom(xsky, ysky, zsky)``).
    sh : tuple
        Shape ``(nz, ny, nx)`` of the sampling grid.
    xc_samp, yc_samp, zc_samp : float
        Centre pixel positions along each axis.

    Returns
    -------
    xgal, ygal, zgal : np.ndarray
        Galaxy-frame coordinates.
    xsky, ysky, zsky : np.ndarray
        Sky-frame coordinates (centred).
    """
    zsky, ysky, xsky = np.indices(sh)
    zsky = zsky - zc_samp
    ysky = ysky - yc_samp
    xsky = xsky - xc_samp
    xgal, ygal, zgal = geom(xsky, ysky, zsky)
    return xgal, ygal, zgal, xsky, ysky, zsky


def _get_xyz_sky_gal_inverse(geom, sh, xc_samp, yc_samp, zc_samp):
    """Get sky/galaxy coordinate grids, regular in galaxy frame.

    Parameters
    ----------
    geom : Geometry
        Geometry model (must support ``geom.inverse_coord_transform(...)``).
    sh : tuple
        Shape ``(nz, ny, nx)``.
    xc_samp, yc_samp, zc_samp : float
        Centre pixel positions.

    Returns
    -------
    xgal, ygal, zgal : np.ndarray
        Galaxy-frame coordinates.
    xsky, ysky, zsky : np.ndarray
        Sky-frame coordinates.
    """
    zgal, ygal, xgal = np.indices(sh)
    zgal = zgal - zc_samp
    ygal = ygal - yc_samp
    xgal = xgal - xc_samp
    xsky, ysky, zsky = geom.inverse_coord_transform(xgal, ygal, zgal)
    return xgal, ygal, zgal, xsky, ysky, zsky


def _calculate_max_skyframe_extents(geom, nx_sky_samp, ny_sky_samp,
                                    transform_method, angle='cos'):
    """Calculate max z sky-frame extent given geometry.

    Parameters
    ----------
    geom : Geometry
        Geometry model (provides ``geom.inc`` in degrees).
    nx_sky_samp, ny_sky_samp : float
        Sampled FOV extents in pixels.
    transform_method : str
        ``'direct'`` or ``'rotate'``.
    angle : str
        ``'cos'`` or ``'sin'`` -- which trig factor to use.

    Returns
    -------
    nz_sky_samp : int
        Number of z-sky pixels (forced to be odd).
    maxr : float
        Maximum radial extent in pixels.
    maxr_y : float
        Maximum y-extent in pixels.
    """
    maxr = np.sqrt(nx_sky_samp ** 2 + ny_sky_samp ** 2)

    if transform_method.lower().strip() == 'direct':
        if angle.lower().strip() == 'cos':
            cos_inc = np.cos(float(geom.inc) * np.pi / 180.0)
            geom_fac = cos_inc
        elif angle.lower().strip() == 'sin':
            sin_inc = np.sin(float(geom.inc) * np.pi / 180.0)
            geom_fac = sin_inc
        else:
            raise ValueError("angle must be 'cos' or 'sin'")
        maxr_y = np.max(np.array([
            maxr * 1.5,
            np.min(np.hstack([maxr * 1.5 / geom_fac, maxr * 5.0])),
        ]))
    else:
        maxr_y = maxr * 5.0

    if angle.lower().strip() == 'cos':
        nz_sky_samp = int(np.max([nx_sky_samp, ny_sky_samp]))
    elif angle.lower().strip() == 'sin':
        nz_sky_samp = int(np.max([nx_sky_samp, ny_sky_samp, maxr_y]))

    # Ensure odd
    if np.mod(nz_sky_samp, 2) < 0.5:
        nz_sky_samp += 1

    return nz_sky_samp, maxr, maxr_y
