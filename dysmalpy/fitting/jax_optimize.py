# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information.
#
# JAX-native Adam optimizer for DYSMALPY model fitting.
#
# Uses JAX autodiff gradients with the Adam optimizer to fit models,
# providing a fully JIT-compiled alternative to MPFIT.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

## Standard library
import logging

# DYSMALPY code
from dysmalpy import galaxy
from dysmalpy import utils as dpy_utils
from dysmalpy.fitting import base
from dysmalpy.fitting import utils as fit_utils
from dysmalpy.fitting.jax_loss import make_jax_loss_function, make_jax_loss_function_1d

# Third party imports
import numpy as np
import time, datetime

import jax
import jax.numpy as jnp


__all__ = ['JAXAdamFitter', 'JAXAdamResults']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)


class JAXAdamResults(base.FitResults):
    """Results of JAX Adam optimization."""

    def __init__(self, model=None, blob_name=None):
        super(JAXAdamResults, self).__init__(model=model,
                                              fit_method='JAXAdam',
                                              blob_name=blob_name)
        self.loss_history = None
        self.n_traceable = None

    def analyze_plot_save_results(self, gal, output_options=None):
        """Save results after fitting."""
        # Update theta to best-fit
        gal.model.update_parameters(self.bestfit_parameters)
        gal.create_model_data()

        self.bestfit_redchisq = base.chisq_red(gal)
        self.bestfit_chisq = base.chisq_eval(gal)

        if output_options.save_results and output_options.f_results is not None:
            self.save_results(filename=output_options.f_results,
                              overwrite=output_options.overwrite)

        if output_options.save_model and output_options.f_model is not None:
            gal.preserve_self(filename=output_options.f_model,
                              save_data=output_options.save_data,
                              overwrite=output_options.overwrite)

    def plot_results(self, gal, f_plot_bestfit=None, overwrite=False):
        """Plot the bestfit."""
        if f_plot_bestfit is not None:
            self.plot_bestfit(gal, fileout=f_plot_bestfit, overwrite=overwrite)


class JAXAdamFitter(base.Fitter):
    """Fitter using JAX Adam optimizer with autodiff gradients.

    This fitter uses ``jax.jit(jax.value_and_grad)`` to compile both the
    chi-squared loss and its gradient, then optimizes with Adam.

    Only non-geometry free parameters are optimized via autodiff. Geometry
    parameters (inc, pa, xshift, yshift) are excluded because they affect
    grid shapes and cannot be JAX-traced.
    """

    def __init__(self, **kwargs):
        self._set_defaults()
        super(JAXAdamFitter, self).__init__(fit_method='JAXAdam', **kwargs)

    def _set_defaults(self):
        self.n_steps = 1000
        self.learning_rate = 1e-3
        self.maxiter = self.n_steps  # alias for interface compatibility

    def fit(self, gal, output_options):
        """Fit observed kinematics using JAX Adam optimizer.

        Parameters
        ----------
        gal : Galaxy instance
        output_options : config.OutputOptions instance

        Returns
        -------
        JAXAdamResults
        """
        # Check the FOV is large enough to cover the data output:
        dpy_utils._check_data_inst_FOV_compatibility(gal)

        # Pre-calculate instrument kernels:
        gal = dpy_utils._set_instrument_kernels(gal)

        # Set output options
        output_options.set_output_options(gal, self)
        fit_utils._check_existing_files_overwrite(output_options, fit_type='jax_adam')

        # Setup logging
        if output_options.f_log is not None:
            loggerfile = logging.FileHandler(output_options.f_log)
            loggerfile.setLevel(logging.INFO)
            logger.addHandler(loggerfile)

        logger.info("*************************************")
        logger.info(" Fitting: {} using JAX Adam".format(gal.name))

        start = time.time()

        # ---- Build loss function ----
        # Get the first observation for loss computation
        obs_name = list(gal.observations.keys())[0]
        obs = gal.observations[obs_name]

        # Get data cube, noise, mask from observation
        if obs.data.ndim == 3:
            cube_obs = obs.data.data.unmasked_data[:].value
            noise = obs.data.error.unmasked_data[:].value
            msk = obs.data.mask

            # Replace zero errors in unmasked pixels with large values
            noise = np.where((noise == 0) & (msk == 0), 99., noise)

            weight = float(obs.weight)

        elif obs.data.ndim == 1:
            # 1D profile fitting: extract observed profiles and build
            # loss function with aperture-based moment extraction
            if not obs.instrument.moment:
                raise ValueError(
                    "JAXAdamFitter requires obs.instrument.moment=True "
                    "for 1D fitting (JAX-traceable moment extraction)."
                )

            vel_obs = obs.data.data['velocity']
            vel_err = obs.data.error['velocity']
            disp_obs = obs.data.data['dispersion']
            disp_err = obs.data.error['dispersion']
            weight = float(obs.weight) if obs.weight is not None else 1.0

            mask_vel = obs.data.mask_velocity if obs.data.mask_velocity is not None else obs.data.mask
            mask_disp = obs.data.mask_vel_disp if obs.data.mask_vel_disp is not None else obs.data.mask

            inst_corr = obs.data.data.get('inst_corr', False)
            lsf_dispersion = 0.0
            if inst_corr and obs.instrument.lsf is not None:
                # Extract LSF dispersion value (always in km/s)
                lsf_disp = obs.instrument.lsf.dispersion
                if hasattr(lsf_disp, 'unit'):
                    import astropy.units as astropy_u
                    lsf_dispersion = lsf_disp.to(astropy_u.km / astropy_u.s).value
                else:
                    lsf_dispersion = float(lsf_disp)

            fit_velocity = obs.fit_options.fit_velocity
            fit_dispersion = obs.fit_options.fit_dispersion

        else:
            raise NotImplementedError(
                "JAXAdamFitter currently only supports 1D and 3D observations."
            )

        # Get dscale (arcsec per kpc proper)
        dscale = gal.dscale

        # Build loss function
        if obs.data.ndim == 3:
            jax_loss, get_traceable_theta, set_all_theta = make_jax_loss_function(
                gal.model, obs, dscale, cube_obs, noise,
                mask=msk, weight=weight, convolve=True,
            )
        elif obs.data.ndim == 1:
            # Ensure model_set constraint dicts are up-to-date
            # (component-level fixed/tied may have been modified after
            # add_component was called)
            for cmp_name in gal.model.components:
                comp = gal.model.components[cmp_name]
                gal.model.fixed[cmp_name] = dict(comp.fixed)
                gal.model.tied[cmp_name] = dict(comp.tied)
            gal.model.nparams_free = sum(
                1 for c in gal.model.components
                for p in gal.model.fixed[c]
                if not gal.model.fixed[c][p] and not gal.model.tied[c][p]
            )

            jax_loss, get_traceable_theta, set_all_theta = make_jax_loss_function_1d(
                gal.model, obs, dscale,
                vel_obs=vel_obs, vel_err=vel_err,
                disp_obs=disp_obs, disp_err=disp_err,
                mask_vel=mask_vel, mask_disp=mask_disp,
                inst_corr=inst_corr, lsf_dispersion=lsf_dispersion,
                fit_velocity=fit_velocity, fit_dispersion=fit_dispersion,
                weight=weight, convolve=True,
            )

        # JIT-compile loss + gradient
        if obs.data.ndim == 3:
            loss_grad_fn = jax.jit(jax.value_and_grad(jax_loss))
        else:
            # 1D/2D: simulate_cube uses scipy.interpolate which is not
            # JAX-traceable, so we cannot JIT or autodiff the full pipeline.
            # Use numerical (finite-difference) gradients instead.
            # Disable JAX JIT to avoid repeated LLVM compilation of
            # gammaincinv inside simulate_cube's sersic profile.
            loss_grad_fn = None
            jax.config.update("jax_disable_jit", True)

        # Get initial traceable theta
        theta = jnp.array(get_traceable_theta(), dtype=jnp.float64)
        n_traceable = int(theta.shape[0])
        logger.info("    nFreeParams: {}".format(gal.model.nparams_free))

        logger.info("    nTraceableParams: {}".format(n_traceable))
        logger.info("    nSteps: {}".format(self.n_steps))
        logger.info("    learning_rate: {}".format(self.learning_rate))

        # ---- Manual Adam optimization ----
        # (Avoid optax dependency — implement Adam inline)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = self.learning_rate

        m = jnp.zeros_like(theta)  # first moment
        v = jnp.zeros_like(theta)  # second moment

        best_loss = jnp.inf
        best_theta = np.array(theta)
        loss_history = []

        # Get full theta for non-traceable (geometry) params
        pfree_full = gal.model.get_free_parameters_values()

        for step_i in range(self.n_steps):
            if loss_grad_fn is not None:
                loss_val, grads = loss_grad_fn(theta)
            else:
                # Numerical gradient via finite differences
                loss_val = float(jax_loss(np.array(theta)))
                theta_np = np.array(theta)
                grads_np = np.zeros_like(theta_np)
                eps_fd = 1e-5
                for i in range(len(theta)):
                    theta_p = theta_np.copy(); theta_p[i] += eps_fd
                    theta_m = theta_np.copy(); theta_m[i] -= eps_fd
                    grads_np[i] = (float(jax_loss(theta_p)) - float(jax_loss(theta_m))) / (2 * eps_fd)
                theta = jnp.array(theta_np, dtype=jnp.float64)
                loss_val = jnp.float64(loss_val)
                grads = jnp.array(grads_np, dtype=jnp.float64)

            # Adam update
            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * grads ** 2
            m_hat = m / (1 - beta1 ** (step_i + 1))
            v_hat = v / (1 - beta2 ** (step_i + 1))
            theta = theta - lr * m_hat / (jnp.sqrt(v_hat) + eps)

            loss_history.append(float(loss_val))

            if float(loss_val) < float(best_loss):
                best_loss = loss_val
                best_theta = np.array(theta)

            if step_i % 50 == 0 or step_i == self.n_steps - 1:
                logger.info("  step={}, loss={:.6f}".format(
                    step_i, float(loss_val)))

        # ---- Restore best parameters ----
        # Use model_set.update_parameters() which correctly updates
        # instance storage, component arrays, model_set.parameters,
        # and evaluates tied functions.
        if n_traceable == gal.model.nparams_free:
            # All free params are traceable — use full theta vector
            gal.model.update_parameters(best_theta)
        else:
            # Some free params are not traceable (geometry) — restore
            # traceable params only, leave others unchanged
            from dysmalpy.fitting.jax_loss import _identify_traceable_params
            reindexed, _, orig_theta_indices = _identify_traceable_params(gal.model)
            for (cmp_name, param_name), new_idx in reindexed:
                comp = gal.model.components[cmp_name]
                sname = '_param_value_{}'.format(param_name)
                object.__setattr__(comp, sname, float(best_theta[new_idx]))
                param_desc = comp._param_instances[param_name]
                param_desc._default = float(best_theta[new_idx])
                # Keep model_set.parameters in sync
                key_idx = gal.model._param_keys.get(cmp_name, {}).get(param_name)
                if key_idx is not None:
                    gal.model.parameters[key_idx] = float(best_theta[new_idx])

        # Build the full best-fit parameter vector
        all_theta_best = gal.model.get_free_parameters_values()

        end = time.time()
        elapsed = end - start
        logger.info("Finished JAX Adam fitting in {:.2f} sec".format(elapsed))
        logger.info("  Best loss: {:.6f}".format(float(best_loss)))

        # Clean up logger
        if output_options.f_log is not None:
            logger.removeHandler(loggerfile)
            loggerfile.close()

        # Re-enable JIT if it was disabled for 1D/2D fitting
        if loss_grad_fn is None:
            jax.config.update("jax_disable_jit", False)

        # ---- Package results ----
        results = JAXAdamResults(model=gal.model, blob_name=self.blob_name)
        results.bestfit_parameters = all_theta_best
        results.loss_history = np.array(loss_history)
        results.n_traceable = n_traceable
        results.niter = self.n_steps

        results.analyze_plot_save_results(gal, output_options=output_options)

        return results
