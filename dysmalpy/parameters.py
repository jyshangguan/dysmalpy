# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# This module contains our new Parameter class which applies a prior function
# to a parameter

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

# Standard library
import abc
import operator
import functools
import copy

# Third party
import numpy as np
from scipy.stats import norm, truncnorm
import six

__all__ = ['DysmalParameter', 'Prior', 'UniformPrior', 'GaussianPrior',
           'BoundedGaussianPrior', 'BoundedGaussianLinearPrior',
           'BoundedSineGaussianPrior', 'UniformLinearPrior',
           'ConditionalUniformPrior', 'ConditionalEmpiricalUniformPrior']


def _binary_comparison_operation(op):
    @functools.wraps(op)
    def wrapper(self, val):

        if self._model is None:
            # Unbound descriptor -- support ordered protocol for __set_name__
            if op is operator.lt:
                return object.__lt__(self, val)
            else:
                return NotImplemented

        if self.unit is not None:
            from astropy.units import Quantity  # optional dep
            self_value = Quantity(self.value, self.unit)
        else:
            self_value = self.value

        return op(self_value, val)

    return wrapper


def _storage_name(name):
    """Return the instance-attribute name used to store *name*'s value."""
    return '_param_value_{}'.format(name)



def _f_cond_boundaries(param, modelset, f_bounds):
    """
    Default function f_cond(param, modelset, f_bounds) that takes the parameter and model set as input,
        as well as the boundary function for the conditions.
        It must return True/False if the value does / does not satisfy the conditional requirements.
    """

    # Get boundaries for current param / model values:
    pmin, pmax = f_bounds(param, modelset)

    if (param.value >= pmin) & (param.value <= pmax):
        return True
    else:
        return False

def _f_boundaries_from_cond(param, modelset, f_cond):
    """
    Default function f_bounds(param, modelset, f_cond) that takes the parameter and model set as input,
        as well as the conditional function.
        It will return pmin, pmax that are empirically determined based on conditions
            (up to some rough lower/upper limits).
    This should only be called to initialize walkers, so hopefully it should be robust against
        slight limit inaccuracies.
    """
    if (param.bounds[0] is None) or (param.bounds[0] == -np.inf):
        pmin_pb = -1.e2     # Need to default to a finite value
    else:
        pmin_pb = param.bounds[0]

    if (param.bounds[1] is None) or (param.bounds[1] == np.inf):
        pmax_pb = 1.e2     # Need to default to a finite value
    else:
        pmax_pb = param.bounds[1]


    origstepsize = 0.01
    Nsteps_max = 1001
    Nsteps = np.min([int(np.round((pmax_pb-pmin_pb)/origstepsize)), Nsteps_max])
    stepsize = (pmax_pb-pmin_pb)/(1.*Nsteps)
    parr = np.arange(pmin_pb, pmax_pb+stepsize, stepsize)
    condarr = np.zeros(len(parr), dtype=bool)

    rarr = np.arange(0., 15.1, 0.1)

    mod = copy.deepcopy(modelset)

    for i, p in enumerate(parr):
        # Update the param values:
        mod.set_parameter_value(param._model._name, param._name, p)
        condarr[i] = np.all(np.isfinite(mod.circular_velocity(rarr)))

    # Defaults:
    pmin = pmin_pb
    pmax = pmax_pb
    if not np.all(condarr):
        condarr_int = np.array(condarr, dtype=int)
        condarr_delts = condarr_int[1:]-condarr_int[:-1]
        wh_pos = np.where(condarr_delts > 0)[0]
        wh_neg = np.where(condarr_delts < 0)[0]
        if len(wh_pos) > 0:
            ind = wh_pos[-1]+1
            pmin = parr[ind]
        if len(wh_neg) > 0:
            ind = wh_neg[-1]-1
            if ind >= 0:
                pmax = parr[ind]
            else:
                # Probably suspect.... but default to minimum....
                pmax = parr[0]
                pmin -= 1.

    return pmin, pmax

# ******* PRIORS ************
# Base class for priors
@six.add_metaclass(abc.ABCMeta)
class Prior:
    """
    Base class for priors
    """

    @abc.abstractmethod
    def log_prior(self, *args, **kwargs):
        """Returns the log value of the prior given the parameter value"""

    @abc.abstractmethod
    def prior_unit_transform(self, *args, **kwargs):
        """Map a uniform random variable drawn from [0.,1.] to the prior of interest"""

    @abc.abstractmethod
    def sample_prior(self, *args, **kwargs):
        """Returns a random sample of parameter values distributed according to the prior"""


class UniformPrior(Prior):
    """
    Object for flat priors
    """
    def __init__(self):
        pass

    @staticmethod
    def log_prior(param, **kwargs):
        """
        Returns the log value of the prior given the parameter value

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        Returns
        -------
        lprior : `0` or `-np.inf`
            Log prior value. 0 if the parameter value is within the bounds otherwise `-np.inf`
        """

        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (param.value >= pmin) & (param.value <= pmax):
            return 0.
        else:
            return -np.inf


    def prior_unit_transform(self, param, u, **kwargs):
        """
        Transforms a uniform random variable Uniform[0.,1.] to the prior distribution

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        u : float or list-like
            Random uniform variable(s) drawn from Uniform[0.,1.]

        Returns
        -------
        v : float or list-like
            Transformation of the random uniform variable u to random value(s) 
            drawn from the prior distribution.

        """
        if param.bounds[0] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmin = param.bounds[0]
        if param.bounds[1] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmax = param.bounds[1]

        # Scale and shift the unit [0., 1.] to the bounds:
        # v = range * u + min
        v = (pmax-pmin) * u + pmin

        return v

    @staticmethod
    def sample_prior(param, N=1, **kwargs):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
        N : int, optional
            Size of random sample. Default is 1.

        Returns
        -------
        rsamp : float or array
            Random sample of parameter values

        """
        if param.bounds[0] is None:
            pmin = -1.e5  # Need to default to a finite value for the rand dist.
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = 1.e5  # Need to default to a finite value for the rand dist.
        else:
            pmax = param.bounds[1]

        return np.random.rand(N)*(pmax-pmin) + pmin


# CAN THIS? BC NEED TO USE LinearDiskBulge / etc, bc of walker jumps ?????
class UniformLinearPrior(Prior):
    # Note: must bounds input as LINEAR BOUNDS

    def __init__(self):
        pass

    @staticmethod
    def log_prior(param, **kwargs):

        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (np.power(10., param.value) >= pmin) & (np.power(10., param.value) <= pmax):
            return 0.
        else:
            return -np.inf

    def prior_unit_transform(self, param, u, **kwargs):
        """
        Transforms a uniform random variable Uniform[0.,1.] to the prior distribution

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        u : float or list-like
            Random uniform variable(s) drawn from Uniform[0.,1.]

        Returns
        -------
        v : float or list-like
            Transformation of the random uniform variable u to random value(s) 
            drawn from the prior distribution.

        """

        if param.bounds[0] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmin = param.bounds[0]
        if param.bounds[1] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmax = param.bounds[1]

        # Scale and shift the unit [0., 1.] to the bounds:
        # v = range * u + min
        v = (pmax-pmin) * u + pmin

        return v

    @staticmethod
    def sample_prior(param, N=1, **kwargs):
        if param.bounds[0] is None:
            pmin = -1.e13  # Need to default to a finite value for the rand dist.
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = 1.e13  # Need to default to a finite value for the rand dist.
        else:
            pmax = param.bounds[1]

        rvs = []
        while len(rvs) < N:
            test_v = np.random.rand(1)[0] * (pmax-pmin) + pmin

            if (test_v >= pmin) & (test_v <= pmax):
                rvs.append(np.log10(test_v))

        return rvs


class GaussianPrior(Prior):
    """
    Object for gaussian priors

    Parameters
    ----------
    center : float
        Mean of the Gaussian prior

    stddev : float
        Standard deviation of the Gaussian prior
    """
    def __init__(self, center=0, stddev=1.0):

        self.center = center
        self.stddev = stddev

    def log_prior(self, param, **kwargs):
        """
        Returns the log value of the prior given the parameter value

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        Returns
        -------
        lprior : float
            Log prior value calculated using `~scipy.stats.norm.pdf`
        """
        return np.log(norm.pdf(param.value, loc=self.center,
                        scale=self.stddev))


    def prior_unit_transform(self, param, u, **kwargs):
        """
        Transforms a uniform random variable Uniform[0.,1.] to the prior distribution

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        u : float or list-like
            Random uniform variable(s) drawn from Uniform[0.,1.]

        Returns
        -------
        v : float or list-like
            Transformation of the random uniform variable u to random value(s) 
            drawn from the prior distribution.

        """

        v  = norm.ppf(u, loc=self.center, scale=self.stddev)

        return v

    def sample_prior(self, param, N=1, **kwargs):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
        N : int, optional
            Size of random sample. Default is 1.

        Returns
        -------
        rsamp : float or array
            Random sample of parameter values

        """
        return np.random.normal(loc=self.center,
                                scale=self.stddev, size=N)


class BoundedGaussianPrior(Prior):
    """
    Object for Gaussian priors that only extend to a minimum and maximum value

    Parameters
    ----------
    center : float
        Mean of the Gaussian prior

    stddev : float
        Standard deviation of the Gaussian prior
    """
    def __init__(self, center=0, stddev=1.0):

        self.center = center
        self.stddev = stddev

    def log_prior(self, param, **kwargs):
        """
        Returns the log value of the prior given the parameter value

        The parameter value is first checked to see if its within `param.bounds`.
        If so then the standard Gaussian distribution is used to calculate the prior.

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        Returns
        -------
        lprior : float
            Log prior value calculated using `~scipy.stats.norm.pdf` if `param.value` is within
            `param.bounds`
        """

        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (param.value >= pmin) & (param.value <= pmax):
            return np.log(norm.pdf(param.value, loc=self.center, scale=self.stddev))
        else:
            return -np.inf

    def prior_unit_transform(self, param, u, **kwargs):
        """
        Transforms a uniform random variable Uniform[0.,1.] to the prior distribution

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        u : float or list-like
            Random uniform variable(s) drawn from Uniform[0.,1.]

        Returns
        -------
        v : float or list-like
            Transformation of the random uniform variable u to random value(s) 
            drawn from the prior distribution.

        """

        if param.bounds[0] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:            
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmax = param.bounds[1]

        a = (pmin - self.center) / self.stddev
        b = (pmax - self.center) / self.stddev
        v = truncnorm.ppf(u, a, b, loc=self.center, scale=self.stddev)

        return v

    def sample_prior(self, param, N=1, **kwargs):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
        N : int, optional
            Size of random sample. Default is 1.

        Returns
        -------
        rsamp : float or array
            Random sample of parameter values

        """
        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        rvs = []
        while len(rvs) < N:

            test_v = np.random.normal(loc=self.center, scale=self.stddev,
                                      size=1)[0]
            if (test_v >= pmin) & (test_v <= pmax):

                rvs.append(test_v)

        return rvs


class BoundedGaussianLinearPrior(Prior):

    def __init__(self, center=0, stddev=1.0):

        self.center = center
        self.stddev = stddev

    def log_prior(self, param, **kwargs):

        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (np.power(10., param.value) >= pmin) & (np.power(10., param.value) <= pmax):
            return np.log(norm.pdf(np.power(10., param.value), loc=self.center, scale=self.stddev))
        else:
            return -np.inf


    def prior_unit_transform(self, param, u, **kwargs):

        if param.bounds[0] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:            
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmax = param.bounds[1]

        a = (pmin - self.center) / self.stddev
        b = (pmax - self.center) / self.stddev
        v = truncnorm.ppf(u, a, b, loc=self.center, scale=self.stddev)

        return v

    def sample_prior(self, param, N=1, **kwargs):

        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        rvs = []
        while len(rvs) < N:

            test_v = np.random.normal(loc=self.center, scale=self.stddev,
                                      size=1)[0]
            if (test_v >= pmin) & (test_v <= pmax):

                rvs.append(np.log10(test_v))

        return rvs

class BoundedSineGaussianPrior(Prior):
    """
    Object for priors described by a bounded sine Gaussian distribution

    Parameters
    ----------
    center : float
        Central value of the Gaussian prior BEFORE applying sine function

    stddev : float
        Standard deviation of the Gaussian prior AFTER applying sine function
    """

    def __init__(self, center=0, stddev=1.0):
        # Center, bounds in INC
        # Stddev in SINE INC

        self.center = center
        self.stddev = stddev

        self.center_sine = np.sin(self.center*np.pi/180.)

    def log_prior(self, param, **kwargs):
        """
        Returns the log value of the prior given the parameter value

        The parameter value is first checked to see if its within `param.bounds`.
        If so then a Gaussian distribution on the sine of the parameter is used to
        calculate the prior.

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        Returns
        -------
        lprior : float
            Log prior value
        """
        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (param.value >= pmin) & (param.value <= pmax):
            return np.log(norm.pdf(np.sin(param.value*np.pi/180.), loc=self.center_sine, scale=self.stddev))
        else:
            return -np.inf

    def prior_unit_transform(self, param, u, **kwargs):
        """
        Transforms a uniform random variable Uniform[0.,1.] to the prior distribution

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        u : float or list-like
            Random uniform variable(s) drawn from Uniform[0.,1.]

        Returns
        -------
        v : float or list-like
            Transformation of the random uniform variable u to random value(s) 
            drawn from the prior distribution.

        """

        if param.bounds[0] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:            
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmax = param.bounds[1]

        a = (pmin - self.center) / self.stddev
        b = (pmax - self.center) / self.stddev
        v = truncnorm.ppf(np.sin(u*np.pi/180.), a, b, loc=self.center, scale=self.stddev)

        return v

    def sample_prior(self, param, N=1, **kwargs):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
        N : int, optional
            Size of random sample. Default is 1.

        Returns
        -------
        rsamp : float or array
            Random sample of parameter values

        """
        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        rvs = []
        while len(rvs) < N:

            test_v_sine = np.random.normal(loc=self.center_sine, scale=self.stddev,
                                      size=1)[0]
            test_v = np.abs(np.arcsin(test_v_sine))*180./np.pi
            if (test_v >= pmin) & (test_v <= pmax):

                rvs.append(test_v)

        return rvs


class ConditionalUniformPrior(Prior):
    """
    Object for flat priors, but with boundaries that are conditional on other model parameters.
    """

    def __init__(self, f_bounds=None, f_cond=_f_cond_boundaries):
        """
        Initialize `ConditionalUniformPrior` instance.

        Parameters
        ----------
        f_bounds : function
            Function `f_bounds(param, modelset)` that takes the parameter and model set as input.
            It must return a 2-element array with the lower, upper bounds for the parameter,
            for the given other model parameter settings.
            These will then be used to uniformly sample the parameter within these bounds.

            Note this will not be perfect, given the other parameters will be perturbed within their priors (and thus some of the sampled value tuples may end up in the bad region), but hopefully the MCMC walkers will still be able to work with this.

        f_cond : function, optional
            Function `f_cond(param, modelset, self.f_bounds)` that takes the parameter and model set as input.
            It must return True/False if the value does / does not satisfy the conditional requirements.
            If `True`, then the log prior will be 0., if `False`, then it will be `-np.inf`

            If not set, it will default to a conditional based on the boundary function.

        """
        self.f_bounds = f_bounds
        self.f_cond = f_cond

    def log_prior(self, param, modelset=None, **kwargs):
        """
        Returns the log value of the prior given the parameter value

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
        
        modelset : `~dysmalpy.models.ModelSet`
            Current `~dysmalpy.models.ModelSet`, of which param is a part

        Returns
        -------
        lprior : `0` or `-np.inf`
            Log prior value. 0 if the parameter value is within the bounds otherwise `-np.inf`
        """
        if modelset is None:
            raise ValueError("Must pass `modelset` when calling log_prior() for ConditionalUniformPrior!")

        if (self.f_cond(param, modelset, self.f_bounds)):
            return 0.
        else:
            return -np.inf

    
    def prior_unit_transform(self, param, u, modelset=None, **kwargs):

        raise NotImplementedError("Need to implement in a way that uses something similar to self.f_cond()")

        return v

    def sample_prior(self, param, N=1, modelset=None, **kwargs):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
        N : int, optional
            Size of random sample. Default is 1.

        Returns
        -------
        rsamp : float or array
            Random sample of parameter values

        """
        if modelset is None:
            raise ValueError("Must pass `modelset` when calling sample_prior() for ConditionalUniformPrior!")

        pmin, pmax = self.f_bounds(param, modelset)

        # Catch infs -- move to small/large, but finite, values:
        if (pmin is None) or (pmin == -np.inf):
            pmin = -1.e20  # Need to default to a finite value for the rand dist.

        if (pmax is None) or (pmax == np.inf):
            pmax = 1.e20  # Need to default to a finite value for the rand dist.

        return np.random.rand(N)*(pmax-pmin) + pmin



class ConditionalEmpiricalUniformPrior(Prior):
    """
    Object for flat priors, but with boundaries that are conditional on other model parameters.
    Determined through empirical testing of f_cond, and bounds are then inferred based on
    iterating f_cond.
    """

    def __init__(self, f_cond=None, f_bounds=_f_boundaries_from_cond):
        """
        Initialize ConditionalEmpiricalUniformPrior instance.

        Parameters
        ----------

        f_cond : function
            Function `f_cond(param, modelset, self.f_bounds)` that takes the parameter and model set as input.
            It must return True/False if the value does / does not satisfy the conditional requirements.
            If `True`, then the log prior will be 0., if `False`, then it will be `-np.inf`

        f_bounds : function, optional
            Function `f_bounds(param, modelset, self.f_cond)` that takes the parameter and model set as input.
            It must return a 2-element array with the lower, upper bounds for the parameter,
            for the given other model parameter settings.
            These will then be used to uniformly sample the parameter within these bounds.

            Note this will not be perfect, given the other parameters will be perturbed within their priors (and thus some of the sampled value tuples may end up in the bad region), but hopefully the MCMC walkers will still be able to work with this.

            If not set, it will default to a function that iterates to find boundaries based on `self.f_cond`.

        """
        self.f_cond = f_cond
        self.f_bounds = f_bounds

    def log_prior(self, param, modelset=None, **kwargs):
        """
        Returns the log value of the prior given the parameter value

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
        
        modelset : `~dysmalpy.models.ModelSet`
            Current `~dysmalpy.models.ModelSet`, of which param is a part

        Returns
        -------
        lprior : `0` or `-np.inf`
            Log prior value. 0 if the parameter value is within the bounds otherwise `-np.inf`
        """
        if modelset is None:
            raise ValueError("Must pass `modelset` when calling log_prior() for ConditionalUniformPrior!")

        if (self.f_cond(param, modelset)):
            return 0.
        else:
            return -np.inf


    def prior_unit_transform(self, param, u, modelset=None, **kwargs):

        raise NotImplementedError("Need to implement in a way that uses something similar to self.f_cond()")

        return v
        
    def sample_prior(self, param, N=1, modelset=None, **kwargs):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
        N : int, optional
            Size of random sample. Default is 1.

        Returns
        -------
        rsamp : float or array
            Random sample of parameter values

        """
        if modelset is None:
            raise ValueError("Must pass `modelset` when calling sample_prior() for ConditionalUniformPrior!")

        pmin, pmax = self.f_bounds(param, modelset, self.f_cond)

        # Catch infs -- move to small/large, but finite, values:
        if (pmin is None) or (pmin == -np.inf):
            pmin = -1.e20  # Need to default to a finite value for the rand dist.

        if (pmax is None) or (pmax == np.inf):
            pmax = 1.e20  # Need to default to a finite value for the rand dist.

        return np.random.rand(N)*(pmax-pmin) + pmin



def _propagate_to_instance(descriptor, attr_name, value):
    """Propagate a constraint attribute from a class-level descriptor to
    the owning model's instance-level ``_param_instances`` copy.

    When ``model.param_name.tied = fn`` is executed, ``__get__`` returns
    the *class-level* descriptor.  The property setter on the descriptor
    calls this helper so that ``model._param_instances['param_name']``
    (used by ModelSet to build its authoritative constraint dicts) stays
    in sync.
    """
    model = getattr(descriptor, '_model', None)
    name = getattr(descriptor, '_name', None)
    if model is not None and name is not None:
        if hasattr(model, '_param_instances') and name in model._param_instances:
            inst_copy = model._param_instances[name]
            if inst_copy is not descriptor:
                object.__setattr__(inst_copy, attr_name, value)


class DysmalParameter:
    """
    A lightweight, JAX-compatible replacement for
    ``astropy.modeling.Parameter``.

    It is a Python **descriptor** with the same constraint model as the
    original ``DysmalParameter``: ``('fixed', 'tied', 'bounds', 'prior')``.

    Parameters
    ----------
    name : str
        Parameter name.  Normally set automatically by ``__set_name__``.
    description : str
        Human-readable description.
    default : float or None
        Default value.  If *None* the parameter is initialised to ``0.0``.
    unit : astropy.units.Unit or None
        Physical unit associated with the parameter.
    getter : callable or None
        Custom getter function ``getter(self)``.
    setter : callable or None
        Custom setter function ``setter(self, val)``.
    fixed : bool
        If ``True`` the parameter value is held constant during fitting.
    tied : bool or callable
        If callable, the parameter value is derived from other parameters
        via ``tied(model_instance)``.
    min : float or None
        Deprecated -- use *bounds*.
    max : float or None
        Deprecated -- use *bounds*.
    bounds : tuple of (float, float) or None
        ``(lower, upper)`` bounds.  Use ``None`` for an unbounded side.
    prior : Prior instance or None
        Prior distribution.  Defaults to :class:`UniformPrior`.
    """

    constraints = ('fixed', 'tied', 'bounds', 'prior')

    def __init__(self, name='', description='', default=None, unit=None,
                 getter=None, setter=None, fixed=False, tied=False, min=None,
                 max=None, bounds=None, prior=None):

        if prior is None:
            prior = UniformPrior()

        self.name = name
        self.description = description
        self.unit = unit
        self.getter = getter
        self.setter = setter
        object.__setattr__(self, '_fixed', fixed)
        object.__setattr__(self, '_tied', tied)
        object.__setattr__(self, '_prior', prior)

        # Store original defaults so that _DysmalModel.__init__ can
        # reset class-level pollution (plain attributes set on the
        # class descriptor by earlier instances/tests).
        self._original_fixed = fixed
        self._original_tied = tied
        self._original_prior = copy.deepcopy(prior)

        # Resolve bounds from the various ways they can be specified
        if bounds is not None:
            self.bounds = tuple(bounds)
        elif min is not None or max is not None:
            self.bounds = (min, max)
        else:
            self.bounds = (None, None)

        # Internal default -- used when the owning model instance is first
        # created and no explicit value has been set yet.
        if default is None:
            self._default = 0.0
        else:
            self._default = default

        # Will be set by __set_name__ / __get__ when the descriptor is
        # bound to a model instance.
        self._model = None
        self._name = None          # attribute name on the owning class

    # ------------------------------------------------------------------
    # .tied / .fixed / .prior properties (propagate writes to instance copies)
    # ------------------------------------------------------------------

    @property
    def tied(self):
        """Tied-to-other-parameters function, or ``False``."""
        return self._tied

    @tied.setter
    def tied(self, value):
        object.__setattr__(self, '_tied', value)
        _propagate_to_instance(self, '_tied', value)
        # Also update the model's own tied dict
        model = getattr(self, '_model', None)
        name = getattr(self, '_name', None)
        if model is not None and name is not None:
            if hasattr(model, 'tied') and isinstance(model.tied, dict) and name in model.tied:
                model.tied[name] = value

    @property
    def fixed(self):
        """Whether the parameter is held fixed during fitting."""
        return self._fixed

    @fixed.setter
    def fixed(self, value):
        object.__setattr__(self, '_fixed', value)
        _propagate_to_instance(self, '_fixed', value)
        # Also update the model's own fixed dict
        model = getattr(self, '_model', None)
        name = getattr(self, '_name', None)
        if model is not None and name is not None:
            if hasattr(model, 'fixed') and isinstance(model.fixed, dict) and name in model.fixed:
                model.fixed[name] = value

    @property
    def prior(self):
        """Prior distribution for Bayesian fitting."""
        return self._prior

    @prior.setter
    def prior(self, value):
        object.__setattr__(self, '_prior', value)
        _propagate_to_instance(self, '_prior', value)

    # ------------------------------------------------------------------
    # Descriptor protocol
    # ------------------------------------------------------------------

    def __set_name__(self, owner, name):
        """Called automatically when the descriptor is assigned to a class
        attribute (Python 3.6+)."""
        self._name = name
        if not self.name:
            self.name = name

    def __get__(self, obj, objtype=None):
        """
        Return the descriptor itself (for both class and instance access).

        This allows ``model.param_name.prior = ...`` and
        ``model.param_name.fixed = True`` to work, while the numeric
        value is available via ``model.param_name.value``.
        """
        if obj is None:
            return self

        # Bind to the owning model instance the first time we see it
        if self._model is not obj:
            self._model = obj

        # Initialise the instance storage if it does not exist yet
        sname = _storage_name(self._name)
        if not hasattr(obj, sname):
            setattr(obj, sname, copy.deepcopy(self._default))

        if self.getter is not None:
            # Custom getter: update our cached value and return self
            self._default = self.getter(obj)
            setattr(obj, sname, self._default)

        return self

    def __set__(self, obj, value):
        """
        Set the parameter value on the owning model instance.

        If a custom *setter* is defined it receives ``(obj, value)``.
        Also keeps ``obj.parameters`` (numpy array) in sync.
        """
        if self._model is not obj:
            self._model = obj

        value = float(value)

        if self.setter is not None:
            self.setter(obj, value)
        else:
            setattr(obj, _storage_name(self._name), value)

        # Keep the model's numpy parameter array in sync
        if hasattr(obj, 'parameters') and hasattr(obj, 'param_names'):
            try:
                idx = list(obj.param_names).index(self._name)
                obj.parameters[idx] = value
            except (ValueError, AttributeError):
                pass

    # ------------------------------------------------------------------
    # .value property
    # ------------------------------------------------------------------

    @property
    def value(self):
        """Return the current parameter value.

        This property works both when the descriptor is bound to a model
        instance (returns the instance-level value) and when it is unbound
        (returns the default).
        """
        if self._model is not None:
            sname = _storage_name(self._name)
            if hasattr(self._model, sname):
                val = getattr(self._model, sname)
                if self.getter is not None:
                    return self.getter(self._model)
                return val
            # Storage not yet initialised -- fall through to default
        return copy.deepcopy(self._default)

    @value.setter
    def value(self, val):
        """Set the parameter value."""
        if self._model is not None:
            if self.setter is not None:
                self.setter(self._model, val)
            else:
                setattr(self._model, _storage_name(self._name), val)
        else:
            # Not yet bound -- update the default so that the first
            # __get__ on an instance will pick it up.
            self._default = val

    # ------------------------------------------------------------------
    # Constraint syncing
    # ------------------------------------------------------------------

    def __setattr__(self, name, value):
        # Set the attribute normally
        super().__setattr__(name, value)

        # If 'bounds' is set and we are bound to a model, also update the
        # _param_instances copy so that model.bounds stays in sync.
        # NOTE: We do NOT sync 'fixed' or 'tied' through this mechanism.
        # - 'fixed': setup_gal_models sets halomvirial.fixed = False on the
        #   class-level descriptor AFTER add_component. The model_set.fixed
        #   dict (shared ref with comp.fixed) should NOT be updated, because
        #   _get_free_parameters() uses it. The original astropy-based code
        #   had separate storage for the Parameter.fixed flag vs the
        #   model.fixed dict.
        # - 'tied': tied functions set after add_component are intentionally
        #   left out of model_set.tied. _update_tied_parameters() scans
        #   class-level descriptors directly.
        if name == 'bounds' and not self.__dict__.get('_syncing'):
            model = self.__dict__.get('_model')
            pname = self.__dict__.get('_name')
            if model is not None and pname is not None:
                instances = model.__dict__.get('_param_instances', {})
                if pname in instances:
                    target = instances[pname]
                    target.__dict__['_syncing'] = True
                    try:
                        setattr(target, name, value)
                    finally:
                        target.__dict__['_syncing'] = False
                    if 'bounds' in model.__dict__:
                        model.bounds[pname] = value

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------

    def __getstate__(self):
        return {
            'name': self.name,
            'description': self.description,
            'unit': self.unit,
            'getter': self.getter,
            'setter': self.setter,
            'fixed': self._fixed,
            'tied': self._tied,
            'bounds': self.bounds,
            'prior': self._prior,
            '_default': self._default,
            '_model': None,       # transient -- not pickled
            '_name': self._name,
            '_original_fixed': getattr(self, '_original_fixed', self._fixed),
            '_original_tied': getattr(self, '_original_tied', self._tied),
            '_original_prior': getattr(self, '_original_prior', self._prior),
        }

    def __setstate__(self, state):
        self.name = state['name']
        self.description = state['description']
        self.unit = state['unit']
        self.getter = state['getter']
        self.setter = state['setter']
        object.__setattr__(self, '_fixed', state['fixed'])
        object.__setattr__(self, '_tied', state['tied'])
        self.bounds = state['bounds']
        object.__setattr__(self, '_prior', state['prior'])
        self._default = state['_default']
        self._model = None
        self._name = state['_name']
        self._original_fixed = state.get('_original_fixed', state['fixed'])
        self._original_tied = state.get('_original_tied', state['tied'])
        self._original_prior = state.get('_original_prior', state['prior'])

    # ------------------------------------------------------------------
    # Rich comparison operators
    # ------------------------------------------------------------------

    __eq__ = _binary_comparison_operation(operator.eq)
    __ne__ = _binary_comparison_operation(operator.ne)
    __lt__ = _binary_comparison_operation(operator.lt)
    __gt__ = _binary_comparison_operation(operator.gt)
    __le__ = _binary_comparison_operation(operator.le)
    __ge__ = _binary_comparison_operation(operator.ge)

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            "DysmalParameter(name={!r}, value={!r}, default={!r}, "
            "bounds={!r}, fixed={!r}, prior={!r})"
            .format(self.name, self.value, self._default,
                    self.bounds, self.fixed, self.prior)
        )

    def __str__(self):
        return ("{}={}".format(self.name, self.value))

    def __hash__(self):
        return id(self)

    # ------------------------------------------------------------------
    # Numeric dunder methods -- act as the parameter value
    # ------------------------------------------------------------------

    def __float__(self):
        return float(self.value)

    def __index__(self):
        return int(self.value)

    def __bool__(self):
        return bool(self.value)

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __pow__(self, other):
        return self.value ** other

    def __rpow__(self, other):
        return other ** self.value

    def __neg__(self):
        return -self.value

    def __pos__(self):
        return self.value

    def __abs__(self):
        return abs(self.value)

    # JAX / numpy integration
    def __jax_array__(self):
        import jax.numpy as jnp
        return jnp.asarray(self.value)

    def __array__(self, dtype=None):
        import numpy as np
        return np.asarray(self.value, dtype=dtype)
