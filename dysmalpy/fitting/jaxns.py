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
# **Current Status (dev_jax branch):**
# JAXNS requires a fully JAX-traceable log-likelihood function.  The current
# DYSMALPY model evaluation path (``gal.create_model_data()`` -> numpy) is
# not yet JAX-traceable, so JAXNS will be extremely slow (it falls back to
# ``jax.pure_callback`` or ``jax.disable_jit`` for each likelihood call).
#
# This module provides the infrastructure so that once the model evaluation
# becomes fully JAX-traceable (a goal of the dev_jax branch), JAXNS can be
# used as a drop-in replacement for dynesty with significant speedups from
# JAX-native parallelism.
#
# For now, users should use the dynesty NestedFitter (with forkserver) or
# MPFITFitter for actual fitting work.

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


def _build_jaxns_prior_model(gal):
    """Build a jaxns prior_model generator from dysmalpy galaxy priors.

    Iterates over free parameters in order and yields jaxns Prior objects
    with TFP distributions matching the dysmalpy prior types.

    Returns
    -------
    prior_model : callable
        Generator function suitable for jaxns.Model(prior_model=...)
    param_names : list of str
        Ordered list of parameter names (comp.param format)
    """
    pfree_dict = gal.model.get_free_parameter_keys()
    param_names = []

    # Store prior info for each param
    _prior_info = []
    for compn in pfree_dict:
        comp = gal.model.components[compn]
        for paramn in pfree_dict[compn]:
            if pfree_dict[compn][paramn] >= 0:
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


def _make_log_likelihood(gal, fitter):
    """Create a log-likelihood function for jaxns.

    Since dysmalpy's model evaluation uses numpy/Python (not JAX-traceable),
    we wrap the computation with ``jax.pure_callback`` so that jaxns can
    JIT-compile the sampler while calling our numpy likelihood as a
    side-effect.

    Parameters
    ----------
    gal : Galaxy
        Galaxy instance with model and data.
    fitter : JAXNSFitter
        Fitter instance with settings.

    Returns
    -------
    log_likelihood : callable
        JAX-compatible function that takes parameter values (as positional
        args) and returns a scalar log-likelihood.
    """
    # Capture the galaxy and fitter in the closure
    _gal = gal
    _fitter = fitter

    # Build ordered list of parameter names
    _param_names_ordered = []
    pfree_dict = gal.model.get_free_parameter_keys()
    for compn in pfree_dict:
        for paramn in pfree_dict[compn]:
            if pfree_dict[compn][paramn] >= 0:
                _param_names_ordered.append(f"{compn}.{paramn}")

    ndim = len(_param_names_ordered)

    def _numpy_log_likelihood(theta_flat):
        """Pure numpy log-likelihood computation."""
        theta_np = np.asarray(theta_flat).flatten()
        _gal.model.update_parameters(theta_np)
        _gal.create_model_data()
        llike = base.log_like(_gal, fitter=_fitter)
        # log_like returns (llike, blobvals) when blob_name is set;
        # jaxns expects a scalar log-likelihood only.
        if isinstance(llike, tuple):
            return np.float64(llike[0])
        return np.float64(llike)

    def log_likelihood(*theta_tuple):
        # theta_tuple contains one JAX array per parameter
        theta_flat = jnp.concatenate([jnp.atleast_1d(v) for v in theta_tuple])

        # Use pure_callback to call numpy code from within JIT
        result_shape = jax.ShapeDtypeStruct((), jnp.float64)
        llike = jax.pure_callback(
            _numpy_log_likelihood,
            result_shape,
            theta_flat
        )
        return llike

    return log_likelihood


class JAXNSFitter(base.Fitter):
    """
    Class to hold the JAXNS nested sampling fitter attributes + methods.

    JAXNS uses JAX-native parallelism (no Python multiprocessing), so it
    avoids fork-deadlock issues that can arise with JAX + multiprocessing
    (as experienced with dynesty/emcee on the dev_jax branch).

    Notes
    -----
    Requires ``jaxns`` and ``tensorflow-probability`` to be installed.

    **Performance warning:** The current DYSMALPY model evaluation is not
    fully JAX-traceable, so JAXNS falls back to ``jax.pure_callback`` for
    each likelihood evaluation. This is much slower than using dynesty with
    forkserver. Use the ``NestedFitter`` for production fitting and reserve
    ``JAXNSFitter`` for testing or when the model becomes JAX-traceable.
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
            from jaxns import NestedSampler, Model, Prior
            from jaxns.nested_samplers.common.types import TerminationCondition
            from jaxns.utils import summary as jaxns_summary
            from jaxns.plotting import plot_diagnostics as jaxns_plot_diagnostics
            from jaxns.plotting import plot_cornerplot as jaxns_plot_cornerplot
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
        # Build jaxns model
        ndim = gal.model.nparams_free
        logger.info(f"JAXNS: Fitting {ndim} free parameters")

        prior_model, param_names = _build_jaxns_prior_model(gal)
        log_likelihood = _make_log_likelihood(gal, self)

        jaxns_model = Model(prior_model=prior_model,
                            log_likelihood=log_likelihood)

        logger.info(f"JAXNS: Model U_ndims={jaxns_model.U_ndims}, "
                     f"num_params={jaxns_model.num_params}")

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
            difficult_model=self.difficult_model,
            parameter_estimation=self.parameter_estimation,
            shell_fraction=self.shell_fraction,
            gradient_guided=self.gradient_guided,
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

        ns = NestedSampler(**ns_kwargs)

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
                ns.save_results(results, output_options.f_sampler_results)
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
            logger.warning("JAXNS posterior analysis failed (likely too few samples): %s", e)
            # Set bestfit_parameters from MAP estimate as fallback
            best_fit = np.array([np.array(results.samples[pname]).flatten()
                                  for pname in param_names]).T
            jaxnsResults.bestfit_parameters = best_fit.mean(axis=0)

        # Update model to best-fit and compute chi-squared
        gal.model.update_parameters(jaxnsResults.bestfit_parameters)
        gal.create_model_data()
        from dysmalpy.fitting.base import chisq_red
        jaxnsResults.bestfit_redchisq = chisq_red(gal)

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
                jaxnsResults.plot_results(gal, output_options=output_options,
                                          overwrite=output_options.overwrite)
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

        # Convert jaxns results to the format expected by BayesianFitResults
        # BayesianFitResults expects:
        #   - sampler_results: object with .samples, .blob, .importance_weights()
        #   - chain_param_names: list of param names
        #   - sampler: BayesianSampler with .samples, .blobs, .weights

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
        from jaxns.utils import resample

        results = self._jaxns_results
        num_samples = int(np.array(results.total_num_samples).item())

        # jaxns results.samples is a dict of {param_name: array[num_samples]}
        # We need to convert to a flat array [num_samples, ndim]
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
        try:
            eq_samples = np.array(resample(
                jax.random.PRNGKey(123),
                jax.tree.map(lambda x: x[:num_samples], samples_dict),
                log_dp,
                S=max(100, min(num_samples, int(results.ESS))),
                replace=True
            ))
            samples_eq = np.zeros((eq_samples.shape[1], ndim))
            for i, pname in enumerate(param_names):
                samples_eq[:, i] = np.array(eq_samples[pname])
        except Exception:
            # Fallback: use weighted samples directly
            indices = np.random.choice(num_samples, size=num_samples,
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
        # Use direct attribute assignment to avoid triggering the property
        # setter which would recurse back into _setup_samples_blobs()
        self._sampler_results = _JAXNSResultWrapper(results, param_names)

    def plot_run(self, fileout=None, overwrite=False):
        """Plot/replot the trace for the fitting."""
        plotting.plot_run(self, fileout=fileout, overwrite=overwrite)

    def reload_sampler_results(self, filename=None):
        """Reload the JAXNS results saved earlier."""
        if filename is None:
            raise ValueError("filename must be provided")

        if filename.endswith('.json'):
            from jaxns.utils import load_results
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
