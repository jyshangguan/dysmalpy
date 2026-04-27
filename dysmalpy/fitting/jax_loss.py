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


__all__ = ['make_jax_loss_function', 'make_jax_log_prob_function',
           'make_jax_loss_function_1d', 'make_jaxns_log_likelihood']


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
        _get_xyz_sky_gal_inverse,
        _make_cube_ai,
    )
    from dysmalpy.models.cube_processing import _numpy_coord_transform

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
            xgal, ygal, zgal = _numpy_coord_transform(
                float(geom.inc), float(geom.pa),
                float(geom.xshift.value),
                float(geom.yshift.value),
                nx_sky_samp, ny_sky_samp, nz_sky_samp,
                xcenter_samp, ycenter_samp, (nz_sky_samp - 1) / 2.)
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


def _precompute_sky_grids(model_set, obs, dscale):
    """Pre-compute sky-frame coordinate grids with concrete geometry values.

    Extracts the grid setup logic from ``simulate_cube`` so that the JIT-
    compiled loss function never calls shape-dependent grid computation on
    traced geometry parameters.  The returned dict can be passed as
    ``sky_grids_precomputed`` to ``simulate_cube``.

    Only supports ``transform_method='direct'`` with the default ``angle='cos'``,
    where ``nz_sky_samp = max(nx_sky_samp, ny_sky_samp)`` is independent of
    the geometry parameters.

    Returns
    -------
    dict or None
        ``None`` if transform_method is not ``'direct'``.  Otherwise a dict
        with keys: ``'sh'``, ``'xsky'``, ``'ysky'``, ``'zsky'``,
        ``'xcenter_samp'``, ``'ycenter_samp'``, ``'zc_samp'``, ``'maxr'``,
        ``'maxr_y'``, ``'nx_sky_samp'``, ``'ny_sky_samp'``, ``'oversample'``,
        ``'to_kpc'``, ``'pixscale_samp'``.
    """
    from astropy import units as u

    from dysmalpy.models.model_set import (
        _calculate_max_skyframe_extents,
    )

    transform_method = obs.mod_options.transform_method.lower().strip()
    if transform_method != 'direct':
        return None

    nx_sky = obs.instrument.fov[0]
    ny_sky = obs.instrument.fov[1]
    pixscale = obs.instrument.pixscale.to(u.arcsec).value
    oversample = obs.mod_options.oversample
    oversize = obs.mod_options.oversize
    xcenter = obs.mod_options.xcenter
    ycenter = obs.mod_options.ycenter

    nx_sky_samp = nx_sky * oversample * oversize
    ny_sky_samp = ny_sky * oversample * oversize
    pixscale_samp = pixscale / oversample
    to_kpc = pixscale_samp / dscale

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
        return None

    nz_sky_samp, maxr, maxr_y = _calculate_max_skyframe_extents(
        geom, nx_sky_samp, ny_sky_samp, transform_method, angle='cos')

    sh = (nz_sky_samp, ny_sky_samp, nx_sky_samp)
    zc_samp = (nz_sky_samp - 1) / 2.

    # Sky-frame grids (no geometry dependency)
    zsky, ysky, xsky = np.indices(sh)
    zsky = zsky - zc_samp
    ysky = ysky - ycenter_samp
    xsky = xsky - xcenter_samp

    return {
        'sh': sh,
        'xsky': xsky,
        'ysky': ysky,
        'zsky': zsky,
        'xcenter_samp': xcenter_samp,
        'ycenter_samp': ycenter_samp,
        'zc_samp': zc_samp,
        'maxr': maxr,
        'maxr_y': maxr_y,
        'nx_sky_samp': nx_sky_samp,
        'ny_sky_samp': ny_sky_samp,
        'oversample': oversample,
        'to_kpc': to_kpc,
        'pixscale_samp': pixscale_samp,
    }


def _storage_name(name):
    """Return the instance-attribute name used to store *name*'s value."""
    return '_param_value_{}'.format(name)


def _identify_traceable_params(model_set, include_geometry=False):
    """Identify free parameters that can be JAX-traced.

    By default, geometry parameters (inc, pa, xshift, yshift) are excluded
    because they can affect array shapes in grid computation (numpy requires
    concrete values).  When *include_geometry* is True, all free parameters
    are included — this is safe when using ``sky_grids_precomputed`` to fix
    the grid shapes with concrete values.

    Parameters
    ----------
    model_set : ModelSet
    include_geometry : bool
        If True, include geometry component parameters in the traceable set.

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

    if include_geometry:
        # Include all free parameters
        traceable_entries = sorted(param_map.items(), key=lambda x: x[1])
    else:
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

        llike = -0.5 * chi_sq * weight
        lprob = lp + llike

        return jnp.where(jnp.isfinite(lprob), lprob, -jnp.inf)

    def get_traceable_theta():
        pfree = model_set.get_free_parameters_values()
        return np.array([pfree[i] for i in orig_theta_indices])

    def set_all_theta(theta_full):
        model_set.update_parameters(theta_full)

    return log_prob_fn, get_traceable_theta, set_all_theta


def make_jaxns_log_likelihood(gal, fitter):
    """Return a JAX-traceable log-likelihood function for jaxns.

    Unlike ``make_jax_loss_function`` which returns half chi-squared, this
    returns log-likelihood only (-0.5 * chi_sq) since jaxns handles priors
    via its own ``Prior`` objects.  The function signature is
    ``log_likelihood(*theta_tuple)`` where each argument is a scalar,
    matching jaxns's prior model output.

    Supports both 3D (cube) and 2D (map) observations.  For 2D maps,
    velocity and dispersion maps are extracted from the simulated 3D cube
    using JAX-traceable moment extraction.

    Parameters
    ----------
    gal : Galaxy
        Galaxy instance with model and observations.
    fitter : Fitter
        Fitter instance (used for oversampled_chisq setting).

    Returns
    -------
    log_likelihood : callable
        ``log_likelihood(*theta_tuple) -> float``
    traceable_param_info : dict
        ``{'reindexed': [...], 'n_traceable': int, 'orig_theta_indices': [...]}``
    set_all_theta : callable
        ``set_all_theta(theta_full) -> None``
    """
    model_set = gal.model
    reindexed, n_traceable, orig_theta_indices = _identify_traceable_params(
        model_set, include_geometry=True)

    traceable_param_info = {
        'reindexed': reindexed,
        'n_traceable': n_traceable,
        'orig_theta_indices': orig_theta_indices,
    }

    # Pre-compute data, kernels, and dimensions for each observation
    _obs_data = []
    for obs_name in gal.observations:
        obs = gal.observations[obs_name]

        if not obs.fit_options.fit:
            continue

        ndim = obs.data.ndim
        dscale = gal.dscale

        # Pre-compute ai for z-truncation
        ai_precomputed = None
        ai_sky_precomputed = None
        if obs.mod_options.zcalc_truncate:
            _ai = _precompute_cube_ai(model_set, obs, dscale)
            ai_precomputed = _ai.get('ai')
            ai_sky_precomputed = _ai.get('ai_sky')

        # Pre-compute sky grids (allows geometry params to be JAX-traced)
        sky_grids = None
        if obs.mod_options.transform_method.lower().strip() == 'direct':
            sky_grids = _precompute_sky_grids(model_set, obs, dscale)

        # Extract convolution kernels
        _beam_kernel_jax = None
        _lsf_kernel_jax = None
        if obs.instrument is not None:
            from dysmalpy.convolution import get_jax_kernels
            beam_np, lsf_np = get_jax_kernels(obs.instrument)
            if beam_np is not None:
                _beam_kernel_jax = jnp.asarray(beam_np)
            if lsf_np is not None:
                _lsf_kernel_jax = jnp.asarray(lsf_np)

        _do_convolve = (_beam_kernel_jax is not None or _lsf_kernel_jax is not None)

        # Pre-compute rebin/crop dimensions
        oversample = obs.mod_options.oversample
        oversize = obs.mod_options.oversize
        _do_rebin = (oversample > 1)
        _do_crop = (oversize > 1)

        _rebin_ny = _rebin_nx = None
        _crop_y_start = _crop_y_end = _crop_x_start = _crop_x_end = None

        if _do_rebin or _do_crop:
            from dysmalpy.convolution import _rebin_spatial
            nx_sky = obs.instrument.fov[0]
            ny_sky = obs.instrument.fov[1]
            _rebin_ny = ny_sky * oversize
            _rebin_nx = nx_sky * oversize
            _crop_y_start = (_rebin_ny - ny_sky) // 2
            _crop_y_end = _crop_y_start + ny_sky
            _crop_x_start = (_rebin_nx - nx_sky) // 2
            _crop_x_end = _crop_x_start + nx_sky

        weight = float(obs.weight)

        # Oversampled chi-squared factor
        invnu = 1.0
        if fitter.oversampled_chisq:
            invnu = 1.0 / getattr(obs.data, 'oversample_factor_chisq', 1.0)

        obs_entry = {
            'obs': obs,
            'ndim': ndim,
            'dscale': dscale,
            'ai_precomputed': ai_precomputed,
            'ai_sky_precomputed': ai_sky_precomputed,
            'sky_grids': sky_grids,
            'beam_kernel': _beam_kernel_jax,
            'lsf_kernel': _lsf_kernel_jax,
            'do_convolve': _do_convolve,
            'do_rebin': _do_rebin,
            'do_crop': _do_crop,
            'rebin_ny': _rebin_ny,
            'rebin_nx': _rebin_nx,
            'crop_y_start': _crop_y_start,
            'crop_y_end': _crop_y_end,
            'crop_x_start': _crop_x_start,
            'crop_x_end': _crop_x_end,
            'weight': weight,
            'invnu': invnu,
        }

        if ndim == 3:
            # 3D cube fitting: compare model cube to observed cube
            cube_obs = jnp.asarray(np.asarray(
                obs.data.data.unmasked_data[:].value, dtype=np.float64))
            noise = jnp.asarray(np.asarray(
                obs.data.error.unmasked_data[:].value, dtype=np.float64))
            msk = obs.data.mask

            # Replace zero errors in unmasked pixels with large values
            noise_np = np.asarray(noise)
            msk_np = np.asarray(msk)
            noise_np = np.where((noise_np == 0) & (msk_np == 0), 99., noise_np)
            noise = jnp.asarray(noise_np)

            obs_entry['cube_obs'] = cube_obs
            obs_entry['noise'] = noise
            obs_entry['mask'] = jnp.asarray(msk)

        elif ndim == 2:
            # 2D map fitting: extract velocity/dispersion maps from cube
            # using moment extraction or Gaussian fitting (JAX-traceable)
            vel_obs = jnp.asarray(np.asarray(
                obs.data.data['velocity'], dtype=np.float64))
            vel_err = jnp.asarray(np.asarray(
                obs.data.error['velocity'], dtype=np.float64))
            msk = obs.data.mask

            # Replace zero errors in unmasked pixels
            vel_err_np = np.asarray(vel_err)
            msk_np = np.asarray(msk)
            vel_err_np = np.where((vel_err_np == 0) & (msk_np == 0), 99., vel_err_np)
            vel_err = jnp.asarray(vel_err_np)

            obs_entry['vel_obs'] = vel_obs
            obs_entry['vel_err'] = vel_err
            obs_entry['mask'] = jnp.asarray(msk)
            obs_entry['fit_velocity'] = obs.fit_options.fit_velocity

            # Check if we should use moment extraction or Gaussian fitting
            # moment=True: use moments, moment=False: use Gaussian fitting
            moment_calc = True
            if hasattr(obs.instrument, 'moment'):
                moment_calc = obs.instrument.moment
            obs_entry['moment_calc'] = moment_calc

            if obs.fit_options.fit_dispersion:
                disp_obs = jnp.asarray(np.asarray(
                    obs.data.data['dispersion'], dtype=np.float64))
                disp_err = jnp.asarray(np.asarray(
                    obs.data.error['dispersion'], dtype=np.float64))
                disp_err_np = np.asarray(disp_err)
                disp_err_np = np.where((disp_err_np == 0) & (msk_np == 0), 99., disp_err_np)
                disp_err = jnp.asarray(disp_err_np)

                obs_entry['disp_obs'] = disp_obs
                obs_entry['disp_err'] = disp_err
                obs_entry['fit_dispersion'] = True
            else:
                obs_entry['fit_dispersion'] = False

            # Instrument correction for dispersion
            inst_corr = obs.data.data.get('inst_corr', False)
            lsf_disp2 = 0.0
            if inst_corr and obs.instrument.lsf is not None:
                lsf_disp_val = obs.instrument.lsf.dispersion
                if hasattr(lsf_disp_val, 'unit'):
                    import astropy.units as au
                    lsf_disp2 = float(lsf_disp_val.to(au.km / au.s).value) ** 2
                else:
                    lsf_disp2 = float(lsf_disp_val) ** 2
            obs_entry['inst_corr'] = inst_corr
            obs_entry['lsf_disp2'] = lsf_disp2

            # Spectral axis for moment extraction
            nspec = obs.instrument.nspec
            spec_start = float(obs.instrument.spec_start.value)
            spec_step = float(obs.instrument.spec_step.value)
            spec_arr = np.asarray(
                spec_start + np.arange(nspec) * spec_step, dtype=np.float64)
            delspec = float(np.mean(spec_arr[1:] - spec_arr[:-1]))
            obs_entry['spec_arr'] = jnp.asarray(spec_arr)
            obs_entry['delspec'] = delspec

        else:
            raise NotImplementedError(
                f"make_jaxns_log_likelihood does not support ndim={ndim}")

        _obs_data.append(obs_entry)

    def _inject_tracers(theta_traceable):
        """Inject JAX tracer values into model parameter storage."""
        for (cmp_name, param_name), theta_idx in reindexed:
            comp = model_set.components[cmp_name]
            sname = _storage_name(param_name)
            object.__setattr__(comp, sname, theta_traceable[theta_idx])

    def log_likelihood(*theta_tuple):
        # Convert tuple of scalars to flat array
        theta_traceable = jnp.array([jnp.atleast_1d(v)[0] for v in theta_tuple])
        _inject_tracers(theta_traceable)

        total_llike = jnp.array(0.0)

        for od in _obs_data:
            cube_model, _ = model_set.simulate_cube(
                od['obs'], od['dscale'],
                ai_precomputed=od['ai_precomputed'],
                ai_sky_precomputed=od['ai_sky_precomputed'],
                sky_grids_precomputed=od['sky_grids'])

            if od['do_rebin']:
                from dysmalpy.convolution import _rebin_spatial
                cube_model = _rebin_spatial(cube_model, od['rebin_ny'], od['rebin_nx'])

            if od['do_convolve']:
                from dysmalpy.convolution import convolve_cube_jax
                cube_model = convolve_cube_jax(cube_model,
                                               beam_kernel=od['beam_kernel'],
                                               lsf_kernel=od['lsf_kernel'])

            if od['do_crop']:
                cube_model = cube_model[:,
                                       od['crop_y_start']:od['crop_y_end'],
                                       od['crop_x_start']:od['crop_x_end']]

            if od['ndim'] == 3:
                # Cube-based chi-squared
                chi_sq = jnp.sum(
                    ((cube_model - od['cube_obs']) / od['noise']) ** 2 * od['mask'])
                total_llike += -0.5 * chi_sq * od['invnu'] * od['weight']

            elif od['ndim'] == 2:
                # Extract 2D velocity/dispersion maps from cube
                # Use moment extraction or Gaussian fitting depending on moment_calc parameter
                spec_arr = od['spec_arr']
                delspec = od['delspec']
                msk = od['mask']

                # Check if we should use Gaussian fitting (moment_calc=False)
                if not od.get('moment_calc', True):
                    # Import here to avoid circular import at module level
                    from dysmalpy.fitting.jax_gaussian_fitting import fit_gaussian_cube_jax
                    # Use JAX Gaussian fitting for more accurate parameter extraction
                    flux_map, vel_map, disp_map = fit_gaussian_cube_jax(
                        cube_model=cube_model,
                        spec_arr=spec_arr,
                        mask=(msk == 0),  # JAX uses True for valid pixels
                        method='hybrid'
                    )
                else:
                    # Use moment extraction (faster, JAX-traceable)
                    # flux map: (ny, nx)
                    flux_map = jnp.nansum(cube_model, axis=0) * delspec

                    # velocity map: (ny, nx)
                    vel_map = (jnp.nansum(
                        cube_model * spec_arr[:, None, None], axis=0) * delspec
                        / jnp.where(flux_map != 0, flux_map, 1.))

                    # dispersion map: (ny, nx)
                    disp_map = jnp.sqrt(jnp.abs(
                        jnp.nansum(
                            cube_model * (spec_arr[:, None, None] - vel_map[None, :, :]) ** 2,
                            axis=0) * delspec
                        / jnp.where(flux_map != 0, flux_map, 1.)))

                chi_sq = jnp.array(0.0)

                if od['fit_velocity']:
                    chi_sq += jnp.sum(
                        ((vel_map - od['vel_obs']) / od['vel_err']) ** 2 * msk)

                if od['fit_dispersion']:
                    disp_mod = disp_map
                    if od['inst_corr']:
                        disp_mod = jnp.sqrt(
                            jnp.maximum(disp_mod ** 2 - od['lsf_disp2'], 0.))
                    chi_sq += jnp.sum(
                        ((disp_mod - od['disp_obs']) / od['disp_err']) ** 2 * msk)

                total_llike += -0.5 * chi_sq * od['invnu'] * od['weight']

        return total_llike

    def set_all_theta(theta_full=None):
        """Set all free parameters. If theta_full is None, use current values."""
        if theta_full is not None:
            model_set.update_parameters(theta_full)

    return log_likelihood, traceable_param_info, set_all_theta


def extract_1d_moments_jax(cube, spec_arr, aperture_masks):
    """Extract 1D kinematic profiles from a cube using moment method.

    This is a JAX-traceable equivalent of
    ``Aperture.extract_aper_kin()`` with ``moment=True``.

    Parameters
    ----------
    cube : jnp.ndarray
        3D data cube (nspec, ny, nx).
    spec_arr : jnp.ndarray
        1D spectral axis array (nspec,).
    aperture_masks : list of jnp.ndarray
        List of 2D aperture masks (ny, nx).

    Returns
    -------
    flux1d : jnp.ndarray
        Integrated flux per aperture (n_apertures,).
    vel1d : jnp.ndarray
        Velocity (first moment) per aperture (n_apertures,).
    disp1d : jnp.ndarray
        Dispersion (second moment) per aperture (n_apertures,).
    """
    delspec = jnp.mean(spec_arr[1:] - spec_arr[:-1])

    flux1d = []
    vel1d = []
    disp1d = []

    for mask_2d in aperture_masks:
        # Broadcast mask to 3D and extract spectrum
        mask_3d = mask_2d[jnp.newaxis, :, :]  # (1, ny, nx)
        spec = jnp.nansum(cube * mask_3d, axis=(1, 2))  # (nspec,)

        mom0 = jnp.sum(spec) * delspec
        mom1 = jnp.sum(spec * spec_arr) * delspec / jnp.where(mom0 != 0, mom0, 1.)
        mom2 = jnp.sum(spec * (spec_arr - mom1) ** 2) * delspec / jnp.where(mom0 != 0, mom0, 1.)

        flux1d.append(mom0)
        vel1d.append(mom1)
        disp1d.append(jnp.sqrt(jnp.abs(mom2)))

    return jnp.stack(flux1d), jnp.stack(vel1d), jnp.stack(disp1d)


def make_jax_loss_function_1d(model_set, obs, dscale, vel_obs, vel_err, disp_obs,
                              disp_err, mask_vel=None, mask_disp=None,
                              inst_corr=False, lsf_dispersion=0.0,
                              fit_velocity=True, fit_dispersion=True,
                              weight=1.0, convolve=True):
    """Return a JAX-traceable chi-squared loss function for 1D fitting.

    The loss function:
    1. Injects tracers into model parameter storage
    2. Calls ``simulate_cube()`` (rebin + convolve + crop)
    3. Calls ``extract_1d_moments_jax()`` on the model cube
    4. Computes chi-squared on velocity/dispersion profiles
    5. Applies inst dispersion correction if data is inst-corrected

    Parameters
    ----------
    model_set : ModelSet
        The model set whose free parameters define the loss landscape.
    obs : Observation
        Observation descriptor (used by ``simulate_cube``).
    dscale : float
        Arcsec-to-kpc conversion factor.
    vel_obs : array-like
        Observed velocity profile.
    vel_err : array-like
        Velocity errors.
    disp_obs : array-like
        Observed dispersion profile.
    disp_err : array-like
        Dispersion errors.
    mask_vel : array-like or None
        Boolean mask for velocity data.
    mask_disp : array-like or None
        Boolean mask for dispersion data.
    inst_corr : bool
        If True, data dispersion is instrument-corrected; model dispersion
        must be corrected accordingly.
    lsf_dispersion : float
        LSF dispersion in km/s (used for inst correction).
    fit_velocity : bool
        Whether to include velocity in chi-squared.
    fit_dispersion : bool
        Whether to include dispersion in chi-squared.
    weight : float
        Observation weight.
    convolve : bool
        If True, apply PSF/LSF convolution to the model cube.

    Returns
    -------
    callable
        ``jax_chi2(theta_traceable) -> float`` (half chi-squared, scalar).
    get_traceable_theta : callable
        ``get_traceable_theta() -> ndarray``
    set_all_theta : callable
        ``set_all_theta(theta_full) -> None``
    """
    # Convert observed data to numpy constants
    vel_obs_np = np.asarray(vel_obs, dtype=np.float64)
    vel_err_np = np.asarray(vel_err, dtype=np.float64)
    disp_obs_np = np.asarray(disp_obs, dtype=np.float64)
    disp_err_np = np.asarray(disp_err, dtype=np.float64)

    if mask_vel is not None:
        mask_vel_np = np.asarray(mask_vel)
    else:
        mask_vel_np = None

    if mask_disp is not None:
        mask_disp_np = np.asarray(mask_disp)
    else:
        mask_disp_np = None

    _inst_corr = inst_corr
    _lsf_disp2 = float(lsf_dispersion) ** 2

    # Identify traceable parameters
    reindexed, n_traceable, orig_theta_indices = _identify_traceable_params(model_set)

    # Pre-compute aperture masks from obs.instrument.apertures
    aperture_masks = []
    for aper in obs.instrument.apertures.apertures:
        mask_np = np.asarray(aper._mask_ap, dtype=np.float64)
        # Replace negative values (from partial weight calc) with 0
        mask_np = np.where(mask_np < 0, 0., mask_np)
        aperture_masks.append(mask_np)

    # Construct spectral axis
    nspec = obs.instrument.nspec
    spec_start = float(obs.instrument.spec_start.value)
    spec_step = float(obs.instrument.spec_step.value)
    spec_arr = np.asarray(spec_start + np.arange(nspec) * spec_step, dtype=np.float64)

    # Pre-compute AI arrays for z-truncation
    ai_precomputed = None
    ai_sky_precomputed = None
    if obs.mod_options.zcalc_truncate:
        _ai_precomputed = _precompute_cube_ai(model_set, obs, dscale)
        ai_precomputed = _ai_precomputed.get('ai')
        ai_sky_precomputed = _ai_precomputed.get('ai_sky')

    # Extract convolution kernels
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

    # Pre-compute rebin/crop dimensions
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
            # Use plain float (not JAX tracer) since simulate_cube is not
            # JAX-traceable
            val = float(theta_traceable[theta_idx]) if hasattr(theta_traceable[theta_idx], '__float__') else float(theta_traceable[theta_idx])
            object.__setattr__(comp, sname, val)
            # Also update the DysmalParameter descriptor's default
            if param_name in comp._param_instances:
                comp._param_instances[param_name]._default = val

    def jax_chi2(theta_traceable):
        _inject_tracers(theta_traceable)

        cube_model, _ = model_set.simulate_cube(obs, dscale,
                                                 ai_precomputed=ai_precomputed,
                                                 ai_sky_precomputed=ai_sky_precomputed)
        cube_model = np.asarray(cube_model, dtype=np.float64)

        # Rebin
        if _do_rebin:
            from dysmalpy.utils import rebin
            cube_model = np.asarray(rebin(cube_model, (rebin_ny, rebin_nx)))

        # Convolve
        if _do_convolve:
            cube_model = np.asarray(
                obs.instrument.convolve(cube_model, spec_center=obs.instrument.line_center))

        # Crop
        if _do_crop:
            cube_model = cube_model[:, crop_y_start:crop_y_end,
                                     crop_x_start:crop_x_end]

        # Extract 1D profiles via moment method (numpy version)
        flux1d, vel_mod, disp_mod = extract_1d_moments_jax(
            cube_model, spec_arr, aperture_masks)

        # Inst dispersion correction
        if _inst_corr:
            disp_mod = np.sqrt(np.maximum(disp_mod ** 2 - _lsf_disp2, 0.))

        # Chi-squared
        chi_sq = 0.0

        if fit_velocity:
            if mask_vel_np is not None:
                chi_sq += np.sum(
                    ((vel_obs_np - vel_mod) / vel_err_np) ** 2 * mask_vel_np)
            else:
                chi_sq += np.sum(((vel_obs_np - vel_mod) / vel_err_np) ** 2)

        if fit_dispersion:
            if mask_disp_np is not None:
                chi_sq += np.sum(
                    ((disp_obs_np - disp_mod) / disp_err_np) ** 2 * mask_disp_np)
            else:
                chi_sq += np.sum(((disp_obs_np - disp_mod) / disp_err_np) ** 2)

        return 0.5 * chi_sq * weight

    def get_traceable_theta():
        pfree = model_set.get_free_parameters_values()
        return np.array([pfree[i] for i in orig_theta_indices])

    def set_all_theta(theta_full):
        model_set.update_parameters(theta_full)

    return jax_chi2, get_traceable_theta, set_all_theta
