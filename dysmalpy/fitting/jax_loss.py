# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information.
#
# JAX-accelerated loss functions for fitting DYSMALPY models.
#
# Provides factories that return JIT-compiled callables mapping a parameter
# vector ``theta`` to a scalar loss (half chi-squared or negative
# log-posterior).  JAX tracers are injected directly into the model
# component parameter storage, bypassing the ``float()`` conversion in
# ``_DysmalModel.__setattr__`` so that the entire computation graph
# (velocity profile -> cube population -> chi-squared) is traceable.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import jax
import jax.numpy as jnp
import numpy as np


__all__ = ['make_jax_loss_function', 'make_jax_log_prob_function']


def _precompute_cube_ai(model_set, obs, dscale):
    """Pre-compute sparse index arrays (ai) for z-truncation.

    Runs the grid setup from ``simulate_cube`` with concrete (non-traced)
    values to produce ``ai`` and ``ai_sky``.  These are then passed into
    ``simulate_cube(ai_precomputed=...)`` so that the JIT-compiled loss
    function never calls ``_make_cube_ai`` (which would require dynamic
    boolean indexing on traced arrays).

    Returns
    -------
    dict
        ``{'ai': array or None, 'ai_sky': array or None}``
    """
    from astropy import units as u

    from dysmalpy.models.model_set import (
        _calculate_max_skyframe_extents,
        _get_xyz_sky_gal,
        _get_xyz_sky_gal_inverse,
        _make_cube_ai,
    )

    if not obs.mod_options.zcalc_truncate:
        return {'ai': None, 'ai_sky': None}

    result = {'ai': None, 'ai_sky': None}

    nx_sky = obs.instrument.fov[0]
    ny_sky = obs.instrument.fov[1]
    pixscale = obs.instrument.pixscale.to(u.arcsec).value
    oversample = obs.mod_options.oversample
    oversize = obs.mod_options.oversize
    xcenter = obs.mod_options.xcenter
    ycenter = obs.mod_options.ycenter
    transform_method = obs.mod_options.transform_method
    n_wholepix_z_min = obs.mod_options.n_wholepix_z_min

    nx_sky_samp = nx_sky * oversample * oversize
    ny_sky_samp = ny_sky * oversample * oversize
    pixscale_samp = pixscale / oversample

    if (np.mod(nx_sky, 2) == 1) & (np.mod(oversize, 2) == 0) & (oversize > 1):
        nx_sky_samp = nx_sky_samp + 1
    if (np.mod(ny_sky, 2) == 1) & (np.mod(oversize, 2) == 0) & (oversize > 1):
        ny_sky_samp = ny_sky_samp + 1

    if xcenter is None:
        xcenter_samp = (nx_sky_samp - 1) / 2.
    else:
        xcenter_samp = (xcenter + 0.5) * oversample - 0.5
    if ycenter is None:
        ycenter_samp = (ny_sky_samp - 1) / 2.
    else:
        ycenter_samp = (ycenter + 0.5) * oversample - 0.5

    geom = model_set.geometries.get(obs.name)
    if geom is None:
        return result

    nz_sky_samp, maxr, maxr_y = _calculate_max_skyframe_extents(
        geom, nx_sky_samp, ny_sky_samp, transform_method)

    sh = (nz_sky_samp, ny_sky_samp, nx_sky_samp)

    # Temporarily apply oversample to geometry shifts
    orig_xshift = geom.xshift.value
    orig_yshift = geom.yshift.value
    geom.xshift = orig_xshift * oversample
    geom.yshift = orig_yshift * oversample

    try:
        if transform_method.lower().strip() == 'direct':
            xgal, ygal, zgal, _, _, _ = _get_xyz_sky_gal(
                geom, sh, xcenter_samp, ycenter_samp, (nz_sky_samp - 1) / 2.)
            ai = _make_cube_ai(model_set, xgal, ygal, zgal,
                               n_wholepix_z_min=n_wholepix_z_min,
                               pixscale=pixscale_samp, oversample=oversample,
                               dscale=dscale, maxr=maxr / 2., maxr_y=maxr_y / 2.)
            result['ai'] = np.asarray(ai)

        elif transform_method.lower().strip() == 'rotate':
            xgal, ygal, zgal, _, _, _ = _get_xyz_sky_gal_inverse(
                geom, sh, xcenter_samp, ycenter_samp, (nz_sky_samp - 1) / 2.)

            # For the rotate path, also compute ai_sky
            xgal_final, ygal_final, zgal_final, _, _, _ = _get_xyz_sky_gal_inverse(
                geom, sh, xcenter_samp, ycenter_samp, (nz_sky_samp - 1) / 2.)

            _, _, maxr_y_final = _calculate_max_skyframe_extents(
                geom, nx_sky_samp, ny_sky_samp, 'direct', angle='cos')

            ai_sky = _make_cube_ai(model_set, xgal_final, ygal_final, zgal_final,
                                    n_wholepix_z_min=n_wholepix_z_min,
                                    pixscale=pixscale_samp, oversample=oversample,
                                    dscale=dscale,
                                    maxr=maxr / 2., maxr_y=maxr_y_final / 2.)
            result['ai_sky'] = np.asarray(ai_sky)
    finally:
        # Restore original geometry shifts
        geom.xshift = orig_xshift
        geom.yshift = orig_yshift

    return result


def _storage_name(name):
    """Return the instance-attribute name used to store *name*'s value."""
    return '_param_value_{}'.format(name)


def _identify_traceable_params(model_set):
    """Identify free parameters that can be JAX-traced.

    Geometry parameters (inc, pa, xshift, yshift) are excluded because they
    affect array shapes in grid computation (numpy requires concrete values).

    Parameters
    ----------
    model_set : ModelSet

    Returns
    -------
    reindexed : list of ((cmp_name, param_name), new_theta_idx)
        Re-indexed mapping where *new_theta_idx* is contiguous starting from 0.
    n_traceable : int
        Number of traceable parameters.
    orig_theta_indices : list of int
        The original theta indices corresponding to each entry in *reindexed*.
    """
    param_map = model_set.get_param_storage_names()

    # Identify geometry components
    geom_component_names = set()
    for cmp_name, comp in model_set.components.items():
        if getattr(comp, '_type', None) == 'geometry':
            geom_component_names.add(cmp_name)

    # Filter out geometry params
    traceable_entries = sorted(
        [(k, v) for k, v in param_map.items()
         if k[0] not in geom_component_names],
        key=lambda x: x[1],
    )

    reindexed = [(k, new_idx)
                 for new_idx, (k, _) in enumerate(traceable_entries)]
    orig_theta_indices = [orig_idx for _, orig_idx in traceable_entries]
    n_traceable = len(reindexed)

    return reindexed, n_traceable, orig_theta_indices


def make_jax_loss_function(model_set, obs, dscale, cube_obs, noise,
                           mask=None, weight=1.0, convolve=False):
    """Return a JAX-traceable chi-squared loss function.

    The returned function sets model parameter storage attributes directly
    to JAX tracer values (bypassing ``update_parameters()``) so that the
    DysmalParameter descriptors propagate tracers through
    ``simulate_cube()`` and all downstream operations.

    Parameters
    ----------
    model_set : ModelSet
        The model set whose free parameters define the loss landscape.
    obs : Observation
        Observation descriptor (used by ``simulate_cube``).
    dscale : float
        Arcsec-to-kpc conversion factor.
    cube_obs : array-like
        Observed data cube.
    noise : array-like
        Noise cube (same shape as *cube_obs*).
    mask : array-like or None
        Boolean mask. Only unmasked pixels contribute to chi-squared.
    weight : float
        Observation weight.
    convolve : bool
        If True, apply PSF/LSF convolution to the model cube before
        computing chi-squared.  Requires ``obs.instrument`` to have
        beam and/or LSF kernels available.

    Returns
    -------
    callable
        ``jax_chi2(theta_traceable) -> float``  (half chi-squared, scalar).
    get_traceable_theta : callable
        ``get_traceable_theta() -> ndarray`` extracts current traceable
        parameter values from the model set.
    set_all_theta : callable
        ``set_all_theta(theta_full)`` sets all free parameters (including
        non-traceable geometry params) from a full-length theta vector.
    """
    cube_obs_jax = jnp.asarray(cube_obs)
    noise_jax = jnp.asarray(noise)
    if mask is not None:
        mask_jax = jnp.asarray(mask)
    else:
        mask_jax = None

    reindexed, n_traceable, orig_theta_indices = _identify_traceable_params(model_set)

    # Pre-compute the sparse index array (ai) for z-truncation using
    # concrete (non-traced) values.  This avoids dynamic boolean indexing
    # inside the JIT-compiled loss function.
    ai_precomputed = None
    ai_sky_precomputed = None
    if obs.mod_options.zcalc_truncate:
        _ai_precomputed = _precompute_cube_ai(model_set, obs, dscale)
        ai_precomputed = _ai_precomputed.get('ai')
        ai_sky_precomputed = _ai_precomputed.get('ai_sky')

    # Extract convolution kernels if requested
    _beam_kernel_jax = None
    _lsf_kernel_jax = None
    if convolve and obs.instrument is not None:
        from dysmalpy.convolution import get_jax_kernels, convolve_cube_jax
        beam_np, lsf_np = get_jax_kernels(obs.instrument)
        if beam_np is not None:
            _beam_kernel_jax = jnp.asarray(beam_np)
        if lsf_np is not None:
            _lsf_kernel_jax = jnp.asarray(lsf_np)

    _do_convolve = (_beam_kernel_jax is not None or _lsf_kernel_jax is not None)

    # Pre-compute rebin/crop dimensions (concrete Python ints, known at
    # closure creation time from oversample and oversize).
    oversample = obs.mod_options.oversample
    oversize = obs.mod_options.oversize
    _do_rebin = (oversample > 1)
    _do_crop = (oversize > 1)

    if _do_rebin or _do_crop:
        from dysmalpy.convolution import _rebin_spatial
        nx_sky = obs.instrument.fov[0]
        ny_sky = obs.instrument.fov[1]
        rebin_ny = ny_sky * oversize
        rebin_nx = nx_sky * oversize
        # Crop indices (concrete Python ints)
        crop_y_start = (rebin_ny - ny_sky) // 2
        crop_y_end = crop_y_start + ny_sky
        crop_x_start = (rebin_nx - nx_sky) // 2
        crop_x_end = crop_x_start + nx_sky

    def _inject_tracers(theta_traceable):
        """Inject JAX tracer values into model parameter storage."""
        for (cmp_name, param_name), theta_idx in reindexed:
            comp = model_set.components[cmp_name]
            sname = _storage_name(param_name)
            object.__setattr__(comp, sname, theta_traceable[theta_idx])

    def jax_chi2(theta_traceable):
        _inject_tracers(theta_traceable)

        cube_model, _ = model_set.simulate_cube(obs, dscale,
                                                 ai_precomputed=ai_precomputed,
                                                 ai_sky_precomputed=ai_sky_precomputed)

        # Rebin from oversampled to native pixel scale
        if _do_rebin:
            cube_model = _rebin_spatial(cube_model, rebin_ny, rebin_nx)

        # Convolve (beam then LSF)
        if _do_convolve:
            cube_model = convolve_cube_jax(cube_model,
                                           beam_kernel=_beam_kernel_jax,
                                           lsf_kernel=_lsf_kernel_jax)

        # Crop to native FOV if oversize > 1
        if _do_crop:
            cube_model = cube_model[:, crop_y_start:crop_y_end,
                                     crop_x_start:crop_x_end]

        if mask_jax is not None:
            chi_sq = jnp.sum(
                ((cube_model - cube_obs_jax) / noise_jax) ** 2 * mask_jax
            )
        else:
            chi_sq = jnp.sum(((cube_model - cube_obs_jax) / noise_jax) ** 2)

        return 0.5 * chi_sq * weight

    def get_traceable_theta():
        """Extract current traceable parameter values as a numpy array."""
        pfree = model_set.get_free_parameters_values()
        return np.array([pfree[i] for i in orig_theta_indices])

    def set_all_theta(theta_full):
        """Set all free parameters from a full-length theta vector."""
        model_set.update_parameters(theta_full)

    return jax_chi2, get_traceable_theta, set_all_theta


def _jax_log_prior(theta_traceable, model_set, reindexed):
    """Compute log-prior using JAX-traceable operations.

    Supports UniformPrior, UniformLinearPrior, GaussianPrior, and
    BoundedGaussianPrior.  Returns ``-jnp.inf`` for out-of-bounds parameters.
    """
    lp = jnp.array(0.0)
    for (cmp_name, param_name), theta_idx in reindexed:
        comp = model_set.components[cmp_name]
        param = comp._param_instances[param_name]
        val = theta_traceable[theta_idx]
        prior = param.prior
        bounds = param.bounds  # (lo, hi) or (None, None)

        prior_type = type(prior).__name__

        if prior_type == 'UniformPrior':
            pmin = bounds[0] if bounds[0] is not None else -jnp.inf
            pmax = bounds[1] if bounds[1] is not None else jnp.inf
            in_bounds = (val >= pmin) & (val <= pmax)
            lp += jnp.where(in_bounds, 0.0, -jnp.inf)

        elif prior_type == 'UniformLinearPrior':
            # Bounds are in linear space; parameter value is log10(linear).
            # So 10^val should be within bounds.
            linear_val = jnp.power(10., val)
            pmin = bounds[0] if bounds[0] is not None else -jnp.inf
            pmax = bounds[1] if bounds[1] is not None else jnp.inf
            in_bounds = (linear_val >= pmin) & (linear_val <= pmax)
            lp += jnp.where(in_bounds, 0.0, -jnp.inf)

        elif prior_type == 'GaussianPrior':
            mu = prior.center
            sigma = prior.stddev
            lp += -0.5 * jnp.log(2.0 * jnp.pi * sigma ** 2) \
                  - 0.5 * ((val - mu) / sigma) ** 2

        elif prior_type == 'BoundedGaussianPrior':
            pmin = bounds[0] if bounds[0] is not None else -jnp.inf
            pmax = bounds[1] if bounds[1] is not None else jnp.inf
            in_bounds = (val >= pmin) & (val <= pmax)
            mu = prior.center
            sigma = prior.stddev
            gauss_lp = -0.5 * jnp.log(2.0 * jnp.pi * sigma ** 2) \
                       - 0.5 * ((val - mu) / sigma) ** 2
            lp += jnp.where(in_bounds, gauss_lp, -jnp.inf)

        else:
            # Fallback: assume flat prior for unknown types
            pass

    return lp


def make_jax_log_prob_function(model_set, obs, dscale, cube_obs, noise,
                               mask=None, weight=1.0, convolve=False):
    """Return a JAX-traceable log-posterior function.

    Parameters
    ----------
    model_set : ModelSet
    obs : Observation
    dscale : float
    cube_obs : array-like
    noise : array-like
    mask : array-like or None
    weight : float
    convolve : bool
        If True, apply PSF/LSF convolution to the model cube before
        computing chi-squared.

    Returns
    -------
    log_prob_fn : callable
        ``log_prob_fn(theta_traceable) -> float``
    get_traceable_theta : callable
    set_all_theta : callable
    """
    cube_obs_jax = jnp.asarray(cube_obs)
    noise_jax = jnp.asarray(noise)
    if mask is not None:
        mask_jax = jnp.asarray(mask)
    else:
        mask_jax = None

    reindexed, n_traceable, orig_theta_indices = _identify_traceable_params(model_set)

    # Extract convolution kernels if requested
    _beam_kernel_jax = None
    _lsf_kernel_jax = None
    if convolve and obs.instrument is not None:
        from dysmalpy.convolution import get_jax_kernels, convolve_cube_jax
        beam_np, lsf_np = get_jax_kernels(obs.instrument)
        if beam_np is not None:
            _beam_kernel_jax = jnp.asarray(beam_np)
        if lsf_np is not None:
            _lsf_kernel_jax = jnp.asarray(lsf_np)

    _do_convolve = (_beam_kernel_jax is not None or _lsf_kernel_jax is not None)

    # Pre-compute rebin/crop dimensions (concrete Python ints)
    oversample = obs.mod_options.oversample
    oversize = obs.mod_options.oversize
    _do_rebin = (oversample > 1)
    _do_crop = (oversize > 1)

    if _do_rebin or _do_crop:
        from dysmalpy.convolution import _rebin_spatial
        nx_sky = obs.instrument.fov[0]
        ny_sky = obs.instrument.fov[1]
        rebin_ny = ny_sky * oversize
        rebin_nx = nx_sky * oversize
        crop_y_start = (rebin_ny - ny_sky) // 2
        crop_y_end = crop_y_start + ny_sky
        crop_x_start = (rebin_nx - nx_sky) // 2
        crop_x_end = crop_x_start + nx_sky

    def _inject_tracers(theta_traceable):
        for (cmp_name, param_name), theta_idx in reindexed:
            comp = model_set.components[cmp_name]
            sname = _storage_name(param_name)
            object.__setattr__(comp, sname, theta_traceable[theta_idx])

    def log_prob_fn(theta_traceable):
        # Prior
        lp = _jax_log_prior(theta_traceable, model_set, reindexed)

        # Likelihood (always compute; JAX will optimize away if lp is -inf
        # via jnp.where at the end)
        _inject_tracers(theta_traceable)
        cube_model, _ = model_set.simulate_cube(obs, dscale)

        # Rebin from oversampled to native pixel scale
        if _do_rebin:
            cube_model = _rebin_spatial(cube_model, rebin_ny, rebin_nx)

        # Convolve (beam then LSF)
        if _do_convolve:
            cube_model = convolve_cube_jax(cube_model,
                                           beam_kernel=_beam_kernel_jax,
                                           lsf_kernel=_lsf_kernel_jax)

        # Crop to native FOV if oversize > 1
        if _do_crop:
            cube_model = cube_model[:, crop_y_start:crop_y_end,
                                     crop_x_start:crop_x_end]

        if mask_jax is not None:
            chi_sq = jnp.sum(
                ((cube_model - cube_obs_jax) / noise_jax) ** 2 * mask_jax
            )
        else:
            chi_sq = jnp.sum(((cube_model - cube_obs_jax) / noise_jax) ** 2)

        llike = -0.5 * chi_sq * weight
        lprob = lp + llike

        return jnp.where(jnp.isfinite(lprob), lprob, -jnp.inf)

    def get_traceable_theta():
        pfree = model_set.get_free_parameters_values()
        return np.array([pfree[i] for i in orig_theta_indices])

    def set_all_theta(theta_full):
        model_set.update_parameters(theta_full)

    return log_prob_fn, get_traceable_theta, set_all_theta
