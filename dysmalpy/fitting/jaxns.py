# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information.
#
# Classes and functions for fitting DYSMALPY kinematic models
#   to the observed data using JAXNS nested sampling:
#   https://jaxns.readthedocs.io/en/latest/
#
# JAXNS uses JAX-native parallelism instead of Python multiprocessing,
# so it avoids the fork-deadlock issues that can arise with JAX + multiprocessing.
#
# The log-likelihood is fully JAX-traceable (simulate_cube -> rebin ->
# convolve -> crop -> chi-squared), so JAXNS runs with full JIT compilation
# on both CPU and GPU.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

## Standard library
import logging

# DYSMALPY code
from dysmalpy.data_io import load_pickle, dump_pickle
from dysmalpy import plotting
from dysmalpy import galaxy
from dysmalpy import utils as dpy_utils
from dysmalpy.fitting import base
from dysmalpy.fitting import utils as fit_utils

# Third party imports
import os
import numpy as np
from collections import OrderedDict
import astropy.units as u
import copy

import time, datetime


__all__ = ['JAXNSFitter', 'JAXNSResults']

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

try:
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    tfpd = tfp.distributions
    try:
        from jaxns.prior import Prior
    except ImportError:
        from jaxns import Prior
    _jaxns_loaded = True
except Exception as e:
    _jaxns_loaded = False
    _jaxns_import_error = str(e)
    logger.warning("JAX/TFP not found (needed for JAXNS): %s", _jaxns_import_error)


def _build_jaxns_prior_model(gal, traceable_param_info=None):
    """Build a jaxns prior_model generator from dysmalpy galaxy priors.

    Iterates over free parameters in order and yields jaxns Prior objects
    with TFP distributions matching the dysmalpy prior types.

    When *traceable_param_info* is provided, only those parameters
    are included.  Otherwise all free parameters are used
    (backwards compatible).

    Parameters
    ----------
    gal : Galaxy
    traceable_param_info : dict or None
        If provided, must contain ``'reindexed'`` — a list of
        ``((cmp_name, param_name), theta_idx)`` tuples. Only these
        parameters will get priors.

    Returns
    -------
    prior_model : callable
        Generator function suitable for jaxns.Model(prior_model=...)
    param_names : list of str
        Ordered list of parameter names (comp.param format)
    """
    pfree_dict = gal.model.get_free_parameter_keys()
    param_names = []

    # Determine which parameters to include
    if traceable_param_info is not None:
        reindexed = traceable_param_info['reindexed']
        # Build a set of (comp, param) pairs that are traceable
        traceable_set = {(cmp, param) for (cmp, param), _ in reindexed}
    else:
        traceable_set = None

    # Store prior info for each param
    _prior_info = []
    for compn in pfree_dict:
        comp = gal.model.components[compn]
        for paramn in pfree_dict[compn]:
            if pfree_dict[compn][paramn] >= 0:
                # Skip non-traceable params if filtering is requested
                if traceable_set is not None and (compn, paramn) not in traceable_set:
                    continue
                param = comp._get_param(paramn)
                prior = param.prior
                bounds = param.bounds
                pname = f"{compn}.{paramn}"
                param_names.append(pname)
                _prior_info.append((pname, prior, bounds))

    def prior_model():
        xs = []
        for pname, prior, bounds in _prior_info:
            prior_type = type(prior).__name__

            if prior_type == 'UniformPrior':
                pmin = float(bounds[0]) if bounds[0] is not None else -1e10
                pmax = float(bounds[1]) if bounds[1] is not None else 1e10
                x = yield Prior(tfpd.Uniform(low=pmin, high=pmax),
                                name=pname)
            elif prior_type == 'UniformLinearPrior':
                # Parameter is in log10 space; bounds are in linear space
                pmin = float(bounds[0]) if bounds[0] is not None else 1e-13
                pmax = float(bounds[1]) if bounds[1] is not None else 1e13
                log_min = np.log10(pmin)
                log_max = np.log10(pmax)
                x = yield Prior(tfpd.Uniform(low=log_min, high=log_max),
                                name=pname)
            elif prior_type == 'GaussianPrior':
                x = yield Prior(tfpd.Normal(loc=float(prior.center),
                                             scale=float(prior.stddev)),
                                name=pname)
            elif prior_type == 'BoundedGaussianPrior':
                pmin = float(bounds[0]) if bounds[0] is not None else -1e10
                pmax = float(bounds[1]) if bounds[1] is not None else 1e10
                x = yield Prior(
                    tfpd.TruncatedNormal(loc=float(prior.center),
                                         scale=float(prior.stddev),
                                         low=pmin, high=pmax),
                    name=pname)
            else:
                raise ValueError(f"Unsupported prior type: {prior_type}")
            xs.append(x)

        return tuple(xs)

    return prior_model, param_names


def _untie_parameters_for_jax(gal):
    """Convert tied parameters to free parameters for JAX-traceable fitting.

    Tied parameter functions (e.g. mvirial from fdm via brentq, sigmaz from
    r_eff_disk) use scipy/numpy operations that are not JAX-traceable.
    This function evaluates each tied function once to get the initial value,
    then marks the parameter as free (no longer tied).

    Parameters
    ----------
    gal : Galaxy

    Returns
    -------
    untied_info : list of dict
        Each entry records what was untied, for later restoration.
    """
    untied_info = []
    model = gal.model

    # Update tied functions first so current values are computed
    model._update_tied_parameters()

    for cmp_name in list(model.tied.keys()):
        comp = model.components[cmp_name]
        for param_name in list(model.tied[cmp_name].keys()):
            tied_fn = model.tied[cmp_name][param_name]
            if not callable(tied_fn):
                continue

            # Record the original state
            orig_tied = model.tied[cmp_name][param_name]
            orig_fixed = model.fixed[cmp_name].get(param_name, False)
            param_inst = comp._param_instances[param_name]
            orig_descriptor_tied = param_inst.tied

            # Evaluate the tied function to get current value
            try:
                current_value = float(tied_fn(model))
            except Exception as e:
                logger.warning("Could not evaluate tied function for "
                               "%s.%s: %s — skipping", cmp_name, param_name, e)
                continue

            # Set the parameter value explicitly
            model.set_parameter_value(cmp_name, param_name, current_value,
                                      skip_updated_tied=True)

            # Mark as no longer tied
            model.tied[cmp_name][param_name] = False
            model.fixed[cmp_name][param_name] = False
            param_inst.tied = False
            param_inst.fixed = False

            # Set a flat prior if the current prior is None or a tied-function prior.
            # UniformPrior() takes no arguments; it uses the parameter's bounds.
            if param_inst.prior is None:
                from dysmalpy.parameters import UniformPrior
                param_inst.prior = UniformPrior()

            untied_info.append({
                'comp': cmp_name,
                'param': param_name,
                'orig_tied': orig_tied,
                'orig_fixed': orig_fixed,
                'orig_descriptor_tied': orig_descriptor_tied,
                'init_value': current_value,
            })

    # Recount free parameters
    model.nparams_free = sum(
        1 for c in model.components
        for p in model.fixed[c]
        if not model.fixed[c][p] and not model.tied[c][p]
    )

    if untied_info:
        logger.info("JAXNS: Untied %d parameters for JAX-traceable fitting: %s",
                     len(untied_info),
                     [f"{u['comp']}.{u['param']}" for u in untied_info])

    return untied_info


def _retie_parameters(untied_info, gal):
    """Restore tied parameter status after JAXNS fitting.

    Parameters
    ----------
    untied_info : list of dict
        Output from ``_untie_parameters_for_jax``.
    gal : Galaxy
    """
    model = gal.model

    for info in untied_info:
        cmp_name = info['comp']
        param_name = info['param']
        comp = model.components[cmp_name]
        param_inst = comp._param_instances[param_name]

        # Restore tied status
        model.tied[cmp_name][param_name] = info['orig_tied']
        model.fixed[cmp_name][param_name] = info['orig_fixed']
        param_inst.tied = info['orig_descriptor_tied']

        # Restore fixed status
        if info['orig_fixed']:
            param_inst.fixed = True

    # Recount free parameters
    model.nparams_free = sum(
        1 for c in model.components
        for p in model.fixed[c]
        if not model.fixed[c][p] and not model.tied[c][p]
    )

    if untied_info:
        logger.info("JAXNS: Restored tied status for %d parameters",
                     len(untied_info))


class JAXNSFitter(base.Fitter):
    """
    Class to hold the JAXNS nested sampling fitter attributes + methods.

    JAXNS uses JAX-native parallelism (no Python multiprocessing), so it
    avoids fork-deadlock issues that can arise with JAX + multiprocessing
    (as experienced with dynesty/emcee on the dev_jax branch).

    The log-likelihood is fully JAX-traceable: theta -> simulate_cube ->
    rebin -> convolve -> crop -> chi-squared.  This enables full JIT
    compilation and GPU acceleration.

    Notes
    -----
    Requires ``jaxns`` and ``tensorflow-probability`` to be installed.

    Tied parameters (e.g. mvirial computed from fdm, sigmaz from r_eff_disk)
    are automatically converted to free parameters before fitting, since the
    tied functions use scipy/numpy operations that are not JAX-traceable.
    Their prior is set from the parameter bounds. After fitting, the original
    tied status is restored.
    """

    def __init__(self, **kwargs):
        if not _jaxns_loaded:
            raise ValueError(
                f"JAX/TFP not loaded (needed for JAXNS)! Error: {_jaxns_import_error}")

        self._set_defaults()
        super(JAXNSFitter, self).__init__(fit_method='JAXNS', **kwargs)

    def _set_defaults(self):
        # JAXNS-specific defaults
        self.max_samples = None  # None = auto (100 shrinkages)
        self.num_live_points = None  # None = auto
        self.s = None  # slices per dimension (default: 5)
        self.k = None  # phantom samples (default: 0)
        self.c = None  # parallel Markov chains (default: 30*D)
        self.difficult_model = False
        self.parameter_estimation = False
        self.shell_fraction = 0.5
        self.gradient_guided = False
        self.init_efficiency_threshold = 0.1

        # Termination conditions
        self.dlogZ = 1e-3  # evidence tolerance
        self.max_num_likelihood_evaluations = None

        self.oversampled_chisq = True

        self.nPostBins = 50
        self.linked_posterior_names = None

        self.verbose = True

    def fit(self, gal, output_options):
        """
        Fit observed kinematics using JAXNS nested sampling and a DYSMALPY model set.

        Parameters
        ----------
        gal : `Galaxy` instance
            observed galaxy, including kinematics.
        output_options : `config.OutputOptions` instance
            instance holding output options.

        Returns
        -------
        jaxnsResults : `JAXNSResults` instance
        """

        # Check option validity
        dpy_utils._check_data_inst_FOV_compatibility(gal)
        gal = dpy_utils._set_instrument_kernels(gal)

        # Lazy import jaxns (heavy imports)
        try:
            from jaxns import DefaultNestedSampler, Model, Prior
            from jaxns.nested_sampler import TerminationCondition
            from jaxns import summary as jaxns_summary
            from jaxns import plot_diagnostics as jaxns_plot_diagnostics
            from jaxns import plot_cornerplot as jaxns_plot_cornerplot
        except ImportError as e:
            raise ValueError(f"jaxns not available: {e}")

        # Deep copy the model
        mod_in = copy.deepcopy(gal.model)
        gal.model = mod_in

        # Rebind _model references in _param_instances after deepcopy.
        # copy.deepcopy breaks the back-references from parameter copies
        # to their owning component, which causes DysmalParameter.value
        # to return stale/default values instead of the model's stored
        # values, leading to -inf priors and 0% MCMC acceptance.
        for comp_name, comp in gal.model.components.items():
            for pname, pinst in getattr(comp, '_param_instances', {}).items():
                object.__setattr__(pinst, '_model', comp)

        # Convert tied parameters to free for JAX-traceable fitting
        untied_info = _untie_parameters_for_jax(gal)

        # Setup for oversampled_chisq
        if self.oversampled_chisq:
            gal = fit_utils.setup_oversampled_chisq(gal)

        # Set output options
        output_options.set_output_options(gal, self)

        # Check existing files
        fit_utils._check_existing_files_overwrite(output_options,
                                                  fit_type='jaxns',
                                                  fitter=self)

        # Setup file redirect logging
        if output_options.f_log is not None:
            loggerfile = logging.FileHandler(output_options.f_log)
            loggerfile.setLevel(logging.INFO)
            logger.addHandler(loggerfile)

        # ++++++++++++++++++++++++++++++++
        # Build JAX-traceable log-likelihood
        from dysmalpy.fitting.jax_loss import make_jaxns_log_likelihood

        (log_likelihood, traceable_param_info,
         set_all_theta) = make_jaxns_log_likelihood(gal, self)
        set_all_theta()  # ensure geometry params are set to current values

        # Build prior model for traceable parameters only
        prior_model, param_names = _build_jaxns_prior_model(
            gal, traceable_param_info=traceable_param_info)

        jaxns_model = Model(prior_model=prior_model,
                            log_likelihood=log_likelihood)

        ndim_traceable = traceable_param_info['n_traceable']
        ndim_total = gal.model.nparams_free
        logger.info(f"JAXNS: Fitting {ndim_traceable} traceable parameters "
                     f"(of {ndim_total} total free)")
        logger.info(f"JAXNS: Parameters: {param_names}")
        logger.info(f"JAXNS: Model U_ndims={jaxns_model.U_ndims}, "
                     f"num_params={jaxns_model.num_params}")

        # Sanity check: verify log-likelihood is finite at initial params
        pfree = gal.model.get_free_parameters_values()
        from dysmalpy.fitting.jax_loss import _identify_traceable_params
        _, _, orig_theta_indices = _identify_traceable_params(gal.model,
                                                                include_geometry=True)
        theta_init = jnp.array([pfree[i] for i in orig_theta_indices],
                               dtype=jnp.float64)
        try:
            llike_init = float(log_likelihood(*theta_init))
            logger.info(f"JAXNS: Initial log-likelihood = {llike_init:.2f}")
            if not np.isfinite(llike_init):
                raise ValueError(
                    f"Initial log-likelihood is not finite: {llike_init}")
        except Exception as e:
            logger.error(f"JAXNS: Log-likelihood sanity check failed: {e}")
            raise

        # Configure termination conditions
        term_cond = TerminationCondition(
            dlogZ=jnp.asarray(np.log(1. + self.dlogZ)),
        )
        if self.max_num_likelihood_evaluations is not None:
            term_cond = term_cond._replace(
                max_num_likelihood_evaluations=jnp.asarray(
                    self.max_num_likelihood_evaluations))

        # Create nested sampler
        ns_kwargs = dict(
            model=jaxns_model,
            max_samples=self.max_samples if self.max_samples is not None else 1_000_000,
            difficult_model=self.difficult_model,
            parameter_estimation=self.parameter_estimation,
            init_efficiency_threshold=self.init_efficiency_threshold,
            verbose=self.verbose,
        )
        if self.max_samples is not None:
            ns_kwargs['max_samples'] = self.max_samples
        if self.num_live_points is not None:
            ns_kwargs['num_live_points'] = self.num_live_points
        if self.s is not None:
            ns_kwargs['s'] = self.s
        if self.k is not None:
            ns_kwargs['k'] = self.k
        if self.c is not None:
            ns_kwargs['c'] = self.c

        ns = DefaultNestedSampler(**ns_kwargs)

        logger.info(f"JAXNS: Running nested sampling...")
        t0 = time.time()

        # Run
        termination_reason, state = ns(
            jax.random.PRNGKey(42),
            term_cond=term_cond
        )

        # Convert to results
        results = ns.to_results(termination_reason=termination_reason,
                                state=state,
                                trim=True)

        elapsed = time.time() - t0
        logger.info(f"JAXNS: Completed in {elapsed:.1f}s")

        # Print summary
        if self.verbose:
            jaxns_summary(results)

        # Save raw jaxns results
        if output_options.f_sampler_results is not None:
            if output_options.f_sampler_results.endswith('.json'):
                from jaxns import save_results as jaxns_save_results
                jaxns_save_results(results, output_options.f_sampler_results)
            else:
                dump_pickle(results, filename=output_options.f_sampler_results,
                            overwrite=output_options.overwrite)

        # Save diagnostics plot
        if output_options.f_plot_run is not None:
            try:
                jaxns_plot_diagnostics(results,
                                       save_name=output_options.f_plot_run)
            except Exception as e:
                logger.warning(f"Could not save diagnostics plot: {e}")

        # Save corner plot
        if output_options.f_plot_param_corner is not None:
            try:
                jaxns_plot_cornerplot(results,
                                      save_name=output_options.f_plot_param_corner)
            except Exception as e:
                logger.warning(f"Could not save corner plot: {e}")

        # ++++++++++++++++++++++++++++++++
        # Bundle results into JAXNSResults
        jaxnsResults = JAXNSResults(
            model=gal.model,
            jaxns_results=results,
            param_names=param_names,
            linked_posterior_names=self.linked_posterior_names,
            blob_name=self.blob_name,
            nPostBins=self.nPostBins,
        )

        if self.oversampled_chisq:
            jaxnsResults.oversample_factor_chisq = OrderedDict()
            for obs_name in gal.observations:
                obs = gal.observations[obs_name]
                jaxnsResults.oversample_factor_chisq[obs_name] = obs.data.oversample_factor_chisq

        # Store traceable param info for mapping posterior to full theta
        jaxnsResults._traceable_param_info = traceable_param_info

        # Do all analysis, plotting, saving
        jaxnsResults._setup_samples_blobs()

        # Save chain ascii (always possible)
        if output_options.f_chain_ascii is not None:
            try:
                jaxnsResults.save_chain_ascii(filename=output_options.f_chain_ascii,
                                              overwrite=output_options.overwrite)
            except Exception as e:
                logger.warning("JAXNS chain save failed: %s", e)

        # Posterior analysis may fail with too few samples (KDE needs variance)
        try:
            jaxnsResults.analyze_posterior_dist(gal=gal)
        except Exception as e:
            logger.warning("JAXNS posterior analysis failed: %s", e)
            # Fallback: use median of equally-weighted posterior samples
            if jaxnsResults.sampler is not None and jaxnsResults.sampler.samples is not None:
                jaxnsResults.bestfit_parameters = np.median(
                    jaxnsResults.sampler.samples, axis=0)

        # Map traceable posterior best-fit to full theta vector.
        # This must be done BEFORE _retie_parameters since the model currently
        # has nparams_free=10 (8 original + 2 untied).
        if jaxnsResults.bestfit_parameters is not None:
            full_theta = gal.model.get_free_parameters_values()
            reindexed = traceable_param_info['reindexed']
            for (cmp_name, param_name), new_idx in reindexed:
                orig_idx = traceable_param_info['orig_theta_indices'][new_idx]
                full_theta[orig_idx] = jaxnsResults.bestfit_parameters[new_idx]
            jaxnsResults.bestfit_parameters = full_theta

        # Update model to best-fit BEFORE restoring tied status,
        # since full_theta has the right length for the current (un-untied) model.
        gal.model.update_parameters(jaxnsResults.bestfit_parameters)
        gal.create_model_data()
        from dysmalpy.fitting.base import chisq_red
        jaxnsResults.bestfit_redchisq = chisq_red(gal)

        # Now restore tied parameter status (for result reporting)
        _retie_parameters(untied_info, gal)

        # Re-extract bestfit_parameters from the model so its length matches
        # the (now-retied) nparams_free=8.  The 10-element vector from the
        # un-untied model is no longer valid for update_parameters().
        jaxnsResults.bestfit_parameters = gal.model.get_free_parameters_values()

        # Save results
        if output_options.f_results is not None:
            jaxnsResults.save_results(filename=output_options.f_results,
                                      overwrite=output_options.overwrite)

        # Save galaxy model pickle (needed for downstream plotting)
        if output_options.f_model is not None:
            dump_pickle(gal, filename=output_options.f_model,
                        overwrite=output_options.overwrite)

        # Plotting (best-effort)
        if output_options.do_plotting:
            try:
                jaxnsResults.plot_results(
                    gal,
                    f_plot_param_corner=output_options.f_plot_param_corner,
                    f_plot_bestfit=output_options.f_plot_bestfit,
                    f_plot_trace=output_options.f_plot_trace,
                    f_plot_run=output_options.f_plot_run,
                    overwrite=output_options.overwrite,
                    only_if_fname_set=True)
            except Exception as e:
                logger.warning("JAXNS plotting failed: %s", e)

        # Clean up logger
        if output_options.f_log is not None:
            logger.removeHandler(loggerfile)
            loggerfile.close()

        return jaxnsResults


class JAXNSResults(base.BayesianFitResults, base.FitResults):
    """
    Class to hold results of JAXNS nested sampling fitting.

    Parameters
    ----------
    model : ModelSet
        The model that was fit.
    jaxns_results : jaxns NestedSamplerResults
        Raw results from jaxns.
    param_names : list of str
        Ordered parameter names.
    """

    def __init__(self, model=None, jaxns_results=None, param_names=None,
                 linked_posterior_names=None, blob_name=None, nPostBins=50):

        self._jaxns_results = jaxns_results
        self._param_names = param_names
        self._traceable_param_info = None

        # Initialize parent with placeholder, then set up properly
        super(JAXNSResults, self).__init__(
            model=model,
            blob_name=blob_name,
            fit_method='JAXNS',
            linked_posterior_names=linked_posterior_names,
            sampler_results=None,
            nPostBins=nPostBins)

    def _setup_samples_blobs(self):
        """Extract samples from jaxns results into BayesianSampler format."""
        from jaxns import resample

        results = self._jaxns_results
        num_samples = int(np.array(results.total_num_samples).item())

        # jaxns results.samples is a dict of {param_name: array[num_samples]}
        samples_dict = results.samples
        log_dp = results.log_dp_mean[:num_samples]

        # Resample to get equally-weighted samples
        log_weights = log_dp
        weights = jnp.exp(log_weights - jnp.max(log_weights))
        weights = np.array(weights / jnp.sum(weights))

        # Build flat samples array
        param_names = self._param_names
        ndim = len(param_names)
        samples = np.zeros((num_samples, ndim))
        for i, pname in enumerate(param_names):
            samples[:, i] = np.array(samples_dict[pname][:num_samples])

        # Resample to get equally weighted posterior samples
        ess = int(results.ESS) if results.ESS is not None else num_samples
        n_eq = max(100, min(num_samples, ess))
        try:
            eq_samples = np.array(resample(
                jax.random.PRNGKey(123),
                jax.tree.map(lambda x: x[:num_samples], samples_dict),
                log_dp,
                S=n_eq,
                replace=True
            ))
            samples_eq = np.zeros((n_eq, ndim))
            for i, pname in enumerate(param_names):
                samples_eq[:, i] = np.array(eq_samples[pname])
        except Exception:
            # Fallback: use weighted random choice
            indices = np.random.choice(num_samples, size=n_eq,
                                       p=weights, replace=True)
            samples_eq = samples[indices]

        # Blobs: not directly supported by jaxns, set to None
        blobs = None

        self.chain_param_names = param_names

        self.sampler = base.BayesianSampler(
            samples=samples_eq,
            blobs=blobs,
            weights=np.ones(len(samples_eq)),
            samples_unweighted=samples,
            blobs_unweighted=blobs,
        )

        # Store the jaxns results as sampler_results for compatibility
        self._sampler_results = _JAXNSResultWrapper(results, param_names)

    def plot_corner(self, gal=None, fileout=None, overwrite=False):
        """Plot/replot the corner plot for JAXNS posterior."""
        from jaxns import plot_cornerplot as jaxns_plot_cornerplot

        if self._jaxns_results is None:
            raise ValueError("No JAXNS results available for corner plot")

        if fileout is not None and not overwrite and os.path.isfile(fileout):
            logger.warning("overwrite=False & file already exists: %s", fileout)
            return

        jaxns_plot_cornerplot(self._jaxns_results, save_name=fileout)

    def plot_run(self, fileout=None, overwrite=False):
        """Plot/replot the run diagnostics for JAXNS."""
        from jaxns import plot_diagnostics as jaxns_plot_diagnostics

        if self._jaxns_results is None:
            raise ValueError("No JAXNS results available for run plot")

        if fileout is not None and not overwrite and os.path.isfile(fileout):
            logger.warning("overwrite=False & file already exists: %s", fileout)
            return

        jaxns_plot_diagnostics(self._jaxns_results, save_name=fileout)

    def reload_sampler_results(self, filename=None):
        """Reload the JAXNS results saved earlier."""
        if filename is None:
            raise ValueError("filename must be provided")

        if filename.endswith('.json'):
            from jaxns import load_results
            self._jaxns_results = load_results(filename)
        else:
            self._jaxns_results = load_pickle(filename)

        self._setup_samples_blobs()

    def get_evidence(self):
        """Return the log-evidence and its uncertainty."""
        if self._jaxns_results is not None:
            return (float(self._jaxns_results.log_Z_mean),
                    float(self._jaxns_results.log_Z_uncert))
        return None, None


class _JAXNSResultWrapper:
    """Thin wrapper to make jaxns results compatible with BayesianFitResults.

    Provides a dict-like interface for accessing samples and weights.
    """
    def __init__(self, results, param_names):
        self._results = results
        self._param_names = param_names

    @property
    def samples(self):
        num_samples = int(self._results.total_num_samples)
        ndim = len(self._param_names)
        samples = np.zeros((num_samples, ndim))
        for i, pname in enumerate(self._param_names):
            samples[:, i] = np.array(self._results.samples[pname][:num_samples])
        return samples

    def importance_weights(self):
        log_dp = self._results.log_dp_mean[:int(self._results.total_num_samples)]
        return np.array(jnp.exp(log_dp - jnp.max(log_dp)))


def _reload_all_fitting_jaxns(filename_galmodel=None, filename_results=None):
    gal = galaxy.load_galaxy_object(filename=filename_galmodel)
    results = JAXNSResults()
    results.reload_results(filename=filename_results)
    return gal, results
