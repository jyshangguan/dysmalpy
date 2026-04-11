# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information.
#
# File containing base classes for DysmalPy models

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import abc
import copy
import logging

from collections import OrderedDict

# Third party imports
import numpy as np
import jax.numpy as jnp

from jax.scipy.special import erf, gamma, gammainc

# Local imports
from dysmalpy.parameters import DysmalParameter, UniformPrior

try:
    from dysmalpy.models import utils
except:
   from . import utils

from dysmalpy.special.gammaincinv import gammaincinv

__all__ = ['MassModel', 'LightModel',
           'HigherOrderKinematicsSeparate', 'HigherOrderKinematicsPerturbation',
           'v_circular', 'menc_from_vcirc', 'sersic_mr', 'truncate_sersic_mr',
           '_I0_gaussring']


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# G in pc * Msun^-1 * (km/s)^2
# Derived from: G.cgs * Msun.cgs / (pc.cgs * 1e13) where pc.cgs is in cm,
# r is in kpc so r*1000 gives meters, then /1e5 converts cm/s to km/s.
# Equivalently: astropy G.to('pc / Msun * (km/s)^2').value * 1e-3
# NOTE: This is NOT the same as G.to('pc/Msun/(km/s)^2') which gives 4.301e-3.
# The original dysmalpy base.py v_circular uses CGS internally which yields 4.301e-6.
G_PC_MSUN_KMSQ = 4.30091727003628e-3  # G.to('pc / Msun * (km/s)^2').value
G_PC_MSUN_KMSQ_EFF = G_PC_MSUN_KMSQ * 1e-3  # Effective constant used in v_circular and enclosed_mass


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)


# ===========================================================================
# Metaclass: collects DysmalParameter descriptors at class-creation time
# ===========================================================================

class _DysmalModelMeta(type):
    """Metaclass that collects DysmalParameter instances from class
    attributes.

    Every _DysmalModel subclass has ``_params`` -- an OrderedDict mapping
    parameter names to their class-level DysmalParameter descriptors.
    """

    def __new__(mcs, name, bases, namespace):
        # Skip patching the base class itself
        params = OrderedDict()

        # Collect parameters from parent classes (most-base first so that
        # subclass parameters override parent ones with the same name).
        for base in reversed(bases):
            if hasattr(base, '_params'):
                params.update(base._params)

        # Collect from this class's own namespace
        for key, value in list(namespace.items()):
            if isinstance(value, DysmalParameter):
                # Ensure the descriptor knows its name
                if value._name is None:
                    value._name = key
                if not value.name:
                    value.name = key
                params[key] = value

        namespace['_params'] = params
        cls = super().__new__(mcs, name, bases, namespace)

        # Also run __set_name__ on any descriptors that were copied from a
        # parent and still reference the parent name -- this mirrors what
        # the default type.__new__ does for descriptors defined *directly*
        # in the namespace.
        for pname, param in cls._params.items():
            param.__set_name__(cls, pname)

        return cls


# ===========================================================================
# Standalone helper used by DysmalModel
# ===========================================================================

def _storage_name(name):
    """Return the instance-attribute name used to store *name*'s value."""
    return '_param_value_{}'.format(name)


class _ParamProxy:
    """Proxy returned by ``_DysmalModel.__getattr__`` for parameter access.

    Acts as the parameter value for arithmetic / comparison, but exposes
    ``.prior``, ``.tied``, ``.fixed``, ``.bounds`` so that existing code
    like ``model.r_eff.prior = ...`` continues to work.
    """
    __slots__ = ('_value', '_param', '_model', '_pname')

    def __init__(self, value, param, model, pname):
        object.__setattr__(self, '_value', value)
        object.__setattr__(self, '_param', param)
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_pname', pname)

    # -- numeric dunder methods (forward to _value) -----------------------

    def __float__(self):
        return float(self._value)

    def __repr__(self):
        return repr(self._value)

    def __str__(self):
        return str(self._value)

    def __eq__(self, other):
        return self._value == other

    def __ne__(self, other):
        return self._value != other

    def __lt__(self, other):
        return self._value < other

    def __le__(self, other):
        return self._value <= other

    def __gt__(self, other):
        return self._value > other

    def __ge__(self, other):
        return self._value >= other

    def __add__(self, other):
        return self._value + other

    def __radd__(self, other):
        return other + self._value

    def __sub__(self, other):
        return self._value - other

    def __rsub__(self, other):
        return other - self._value

    def __mul__(self, other):
        return self._value * other

    def __rmul__(self, other):
        return other * self._value

    def __truediv__(self, other):
        return self._value / other

    def __rtruediv__(self, other):
        return other / self._value

    def __pow__(self, other):
        return self._value ** other

    def __rpow__(self, other):
        return other ** self._value

    def __neg__(self):
        return -self._value

    def __pos__(self):
        return self._value

    def __abs__(self):
        return abs(self._value)

    def __hash__(self):
        return hash(float(self._value))

    def __index__(self):
        return int(self._value)

    def __bool__(self):
        return bool(self._value)

    # -- JAX / numpy integration -----------------------------------------

    def __jax_array__(self):
        return jnp.asarray(self._value)

    def __array__(self, dtype=None):
        return np.asarray(self._value, dtype=dtype)

    # -- constraint attribute access -------------------------------------

    @property
    def prior(self):
        return self._param.prior

    @prior.setter
    def prior(self, value):
        self._param.prior = value

    @property
    def fixed(self):
        return self._param.fixed

    @fixed.setter
    def fixed(self, value):
        self._param.fixed = value
        self._model.fixed[self._pname] = value

    @property
    def tied(self):
        return self._param.tied

    @tied.setter
    def tied(self, value):
        self._param.tied = value
        self._model.tied[self._pname] = value

    @property
    def bounds(self):
        return self._param.bounds

    @bounds.setter
    def bounds(self, value):
        self._param.bounds = value
        self._model.bounds[self._pname] = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._param.value = float(v)
        object.__setattr__(self, '_value', float(v))


# ===========================================================================
# _DysmalModel -- the abstract base for ALL models
# ===========================================================================

class _DysmalModel(metaclass=_DysmalModelMeta):
    """
    Base abstract `dysmalpy` model component class.

    Replaces ``astropy.modeling.Model`` with a lightweight,
    JAX-compatible implementation.  Parameter management is handled via
    :class:`~dysmalpy.parameters.DysmalParameter` descriptors
    collected by the :class:`_DysmalModelMeta` metaclass.
    """

    parameter_constraints = DysmalParameter.constraints

    def __init__(self, name=None, **kwargs):
        self.name = name

        # _params is set on the class by the metaclass.  Build an
        # instance-level dict of *deep-copied* descriptors so each model
        # instance owns its own parameter state.
        self._param_instances = OrderedDict()
        for pname, param in self._params.items():
            p = copy.deepcopy(param)
            p._model = self
            self._param_instances[pname] = p

        # Ordered tuple of parameter names
        self.param_names = tuple(self._param_instances.keys())

        # Apply user-supplied keyword values
        for pname, value in kwargs.items():
            if pname in self._param_instances:
                self._param_instances[pname].value = float(value)

        # Build numpy array of current parameter values (host-side, for
        # optimizer / sampler access -- NOT used inside JAX-traced code).
        self.parameters = np.array(
            [self._param_instances[p].value for p in self.param_names]
        )

        # Constraint dicts (for compatibility with fitting backends)
        self.fixed = {p: self._param_instances[p].fixed
                      for p in self.param_names}
        self.tied = {p: self._param_instances[p].tied
                     for p in self.param_names}
        self.bounds = {p: self._param_instances[p].bounds
                       for p in self.param_names}

        # Register instance-storage attributes so the original class-level
        # descriptors (which may still be on the class) also see the right
        # values.  We shadow them with instance attributes.
        for pname, param in self._param_instances.items():
            sname = _storage_name(pname)
            setattr(self, sname, param.value)

    # ------------------------------------------------------------------
    # Attribute access -- transparent parameter proxy
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        # Only intercept non-private names
        if name.startswith('_'):
            raise AttributeError(
                "'{0}' has no attribute '{1}'".format(type(self).__name__, name)
            )
        params = self.__dict__.get('_param_instances', {})
        if name in params:
            return params[name].value
        raise AttributeError(
            "'{0}' has no attribute '{1}'".format(type(self).__name__, name)
        )

    def __setattr__(self, name, value):
        # Always allow internal attributes through
        if (name.startswith('_')
                or name in ('name', 'parameters', 'param_names',
                            'fixed', 'tied', 'bounds',
                            '_param_instances', '_type',
                            'n_inputs', 'n_outputs',
                            'linear', 'fit_deriv', 'col_fit_deriv', 'fittable')):
            super().__setattr__(name, value)
            return

        params = self.__dict__.get('_param_instances', {})
        if name in params:
            params[name].value = float(value)
            # Keep the numpy array in sync
            idx = list(params.keys()).index(name)
            self.parameters[idx] = float(value)
            # Also update the instance storage that the original descriptor
            # may reference
            sname = _storage_name(name)
            setattr(self, sname, float(value))
        else:
            super().__setattr__(name, value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __setstate__(self, state):
        """Compatibility for pickle loading."""
        self.__dict__ = state

    def set_parameter_value(self, param_name, value):
        """Set a parameter by name and keep ``self.parameters`` in sync.

        Parameters
        ----------
        param_name : str
        value : float
        """
        if param_name not in self._param_instances:
            raise KeyError(
                "Parameter '{0}' not found in '{1}'".format(
                    param_name, type(self).__name__)
            )
        self._param_instances[param_name].value = float(value)
        idx = list(self._param_instances.keys()).index(param_name)
        self.parameters[idx] = float(value)
        sname = _storage_name(param_name)
        setattr(self, sname, float(value))

    def get_parameter_value(self, param_name):
        """Return the current value of *param_name*.

        Parameters
        ----------
        param_name : str

        Returns
        -------
        float
        """
        if param_name not in self._param_instances:
            raise KeyError(
                "Parameter '{0}' not found in '{1}'".format(
                    param_name, type(self).__name__)
            )
        return self._param_instances[param_name].value

    def copy(self):
        """Return a deep copy of the model instance."""
        return copy.deepcopy(self)

    def __call__(self, *args, **kwargs):
        """Shortcut for ``self.evaluate(...)``."""
        return self.evaluate(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """Evaluate the model.  Must be overridden by subclasses."""
        raise NotImplementedError(
            "{0}.evaluate() is not implemented".format(type(self).__name__)
        )

    def __repr__(self):
        if self.name is not None:
            return "<{0}(name={1!r})>".format(type(self).__name__, self.name)
        return "<{0}>".format(type(self).__name__)

    def __str__(self):
        if self.name is not None:
            return "{0}(name={1})".format(type(self).__name__, self.name)
        return type(self).__name__


class _DysmalFittable1DModel(_DysmalModel):
    """
    Base class for 1D model components
    """

    linear = False
    fit_deriv = None
    col_fit_deriv = True
    fittable = True

    n_inputs = 1
    n_outputs = 1


class _DysmalFittable3DModel(_DysmalModel):
    """
        Base class for 3D model components
    """

    linear = False
    fit_deriv = None
    col_fit_deriv = True
    fittable = True

    n_inputs = 3


# ***** Mass Component Model Classes ******

class MassModel(_DysmalFittable1DModel):
    """
    Base model for components that exert a gravitational influence
    """

    _type = 'mass'
    _axisymmetric = True
    _multicoord_velocity = False
    _native_geometry = 'cylindrical'  ## possibility for further vel direction abstraction

    @property
    @abc.abstractmethod
    def _subtype(self):
        pass

    @abc.abstractmethod
    def enclosed_mass(self, *args, **kwargs):
        """Evaluate the enclosed mass as a function of radius"""
        pass

    def circular_velocity(self, r):
        r"""
        Default method to evaluate the circular velocity

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate circular velocity in kpc

        Returns
        -------
        vcirc : float or array
            Circular velocity at `r`

        Notes
        -----
        Calculates the circular velocity as a function of radius
        using the standard equation :math:`v(r) = \sqrt(GM(r)/r)`.
        This is only valid for a spherical mass distribution.
        """
        mass_enc = self.enclosed_mass(r)
        vcirc = v_circular(mass_enc, r)

        return vcirc

    def vcirc_sq(self, r):
        r"""
        Default method to evaluate the square of the circular velocity

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate circular velocity in kpc

        Returns
        -------
        vcirc_sq : float or array
            Square of circular velocity at `r`

        Notes
        -----
        Calculates the circular velocity as a function of radius
        as just the square of self.circular_velocity().

        This can be overwritten for inheriting classes with negative potential gradients.
        """
        return self.circular_velocity(r)**2

    def potential_gradient(self, r):
        r"""
        Default method to evaluate the gradient of the potential, :math:`\del\Phi(r)/\del r`.

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate circular velocity in kpc

        Returns
        -------
        dPhidr : float or array
            Gradient of the potential at `r`

        Notes
        -----
        Calculates the gradient of the potential from the circular velocity
        using :math:`\del\Phi(r)/\del r = v_{c}^2(r)/r`.
        An alternative should be written for components where the
        potential gradient is ever *negative* (i.e., rings).

        """
        vcirc = self.circular_velocity(r)
        dPhidr = vcirc ** 2 / r

        return dPhidr


    def vel_direction_emitframe(self, xgal, ygal, zgal, _save_memory=False):
        r"""
        Default method to return the velocity direction in the galaxy Cartesian frame.

        Parameters
        ----------
        xgal, ygal, zgal : float or array
            xyz position in the galaxy reference frame.

        _save_memory : bool, optional
            Option to save memory by only calculating the relevant matrices (eg during fitting).
            Default: False

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyzgal).

            As this is the base mass model, assumes the velocity direction
            is the phi direction in cylindrical coordinates, (R,phi,z).
        """
        rgal = jnp.sqrt(xgal ** 2 + ygal ** 2)

        vhat_y = jnp.where(rgal > 0, xgal / rgal, 0.)

        if not _save_memory:
            vhat_x = jnp.where(rgal > 0, -ygal / rgal, 0.)
            vhat_z = jnp.zeros_like(zgal)

            vel_dir_unit_vector = jnp.array([vhat_x, vhat_y, vhat_z])
        else:
            # Only calculate y values
            vel_dir_unit_vector = [0., vhat_y, 0.]

        return vel_dir_unit_vector


    def velocity_vector(self, xgal, ygal, zgal, vel=None, _save_memory=False):
        """ Return the relevant velocity -- if not specified, call self.circular_velocity() --
            as a vector in the the reference Cartesian frame coordinates. """
        if vel is None:
            vel = self.circular_velocity(jnp.sqrt(xgal**2 + ygal**2))

        vel_hat = self.vel_direction_emitframe(xgal, ygal, zgal, _save_memory=_save_memory)

        if not _save_memory:
            vel_cartesian = vel * vel_hat
        else:
            # Only calculated y direction, as this is cylindrical only
            if self._native_geometry == 'cylindrical':
                vel_cartesian = [0., vel*vel_hat[1], 0.]
            else:
                raise ValueError("all mass models assumed to be cylindrical for memory saving!")

        return vel_cartesian


class LightModel(_DysmalModel):
    """
    Base model for components that emit light, but are treated separately from any gravitational influence
    """

    _type = 'light'
    _axisymmetric = True

    @abc.abstractmethod
    def light_profile(self, *args, **kwargs):
        """Evaluate the enclosed mass as a function of radius"""


class _LightMassModel(_DysmalModel):
    """
    Abstract model for mass model that also emits light
    """

    mass_to_light = DysmalParameter(
        default=1., fixed=True, bounds=(0., 10.)
    )

    def __setstate__(self, state):
        if 'mass_to_light' not in state.keys():
            state['mass_to_light'] = DysmalParameter(
                default=1., fixed=True, bounds=(0., 10.),
                prior=UniformPrior()
            )

        self.__dict__ = state

        # Ensure _param_instances is rebuilt if missing
        if '_param_instances' not in state:
            self._param_instances = OrderedDict()
            if hasattr(self, '_params'):
                for pname, param in self._params.items():
                    p = copy.deepcopy(param)
                    p._model = self
                    self._param_instances[pname] = p
            self.param_names = tuple(self._param_instances.keys())
            self.parameters = np.array(
                [self._param_instances[p].value for p in self.param_names]
            )


class HigherOrderKinematics(_DysmalModel):
    """
    Base model for higher-order kinematic components
    """

    _type = 'higher_order'

    @property
    @abc.abstractmethod
    def _native_geometry(self):
        pass

    @property
    @abc.abstractmethod
    def _higher_order_type(self):
        pass

    @property
    @abc.abstractmethod
    def _separate_light_profile(self):
        pass

    @property
    @abc.abstractmethod
    def _spatial_type(self):
        pass

    @property
    @abc.abstractmethod
    def _multicoord_velocity(self):
        pass

    @abc.abstractmethod
    def velocity(self, *args, **kwargs):
        """Method to return the velocity amplitude (in the output geometry Cartesian frame,
           if self._multicoord_velocity==True)."""
        pass

    @abc.abstractmethod
    def vel_direction_emitframe(self, *args, **kwargs):
        """Method to return the velocity direction in the output geometry Cartesian frame."""
        pass


    def velocity_vector(self, x, y, z, vel=None, _save_memory=False):
        """ Return the velocity -- calling self.velocity() if vel is None -- of the higher order
            component as a vector in the the reference Cartesian frame coordinates. """
        if vel is None:
            vel = self.velocity(x, y, z)

        # Dot product of vel_hat, zsky_unit_vector
        if self._multicoord_velocity:
            # Matrix multiply the velocity direction matrix with the
            #   oritinal velocity tuple, then dot product with the zsky unit vector
            vel_dir_matrix = self.vel_direction_emitframe(x, y, z, _save_memory=_save_memory)

            # Need to explicity work this out, as this is a 3x3 matrix multiplication
            #   with a 3-element vector, where the elements themselves are arrays...
            vel_cartesian = [vel[0]*0., vel[1]*0., vel[2]*0.]
            for row in range(vel_dir_matrix.shape[0]):
                for col in range(vel_dir_matrix.shape[1]):
                    vel_cartesian[row] += vel_dir_matrix[row, col] * vel[col]

        else:
            # Simply apply magnitude to velhat
            vel_hat = self.vel_direction_emitframe(x, y, z, _save_memory=_save_memory)
            if not _save_memory:
                vel_cartesian = vel * vel_hat
            else:
                # Only calculated y,z directions
                vel_cartesian = [0., vel*vel_hat[1], vel*vel_hat[2]]

        return vel_cartesian


class HigherOrderKinematicsSeparate(HigherOrderKinematics):
    """
    Base model for higher-order kinematic components that are separate from the galaxy.
    Have separate light profiles, and can have separate geometry/dispersion components.
    """

    _higher_order_type = 'separate'
    _separate_light_profile = True

    @abc.abstractmethod
    def light_profile(self, *args, **kwargs):
        """Evaluate the light distribution associated with this component."""
        pass


class HigherOrderKinematicsPerturbation(HigherOrderKinematics):
    """
    Base model for higher-order kinematic components that are perturbations to the galaxy.
    Cannot have a separate light/geometry/dispersion components.
    However, they can have light that is then *added* to the galaxy,
        by adding to the ModelSet with light=True.
    """

    _higher_order_type = 'perturbation'
    _separate_light_profile = False
    _axisymmetric = False


#########################################

def v_circular(mass_enc, r):
    r"""
    Circular velocity given an enclosed mass and radius

    .. math::
        v(r) = \sqrt{(GM(r)/r)}

    Parameters
    ----------
    mass_enc : float
        Enclosed mass in solar units

    r : float or array
        Radius at which to calculate the circular velocity in kpc

    Returns
    -------
    vcirc : float or array
        Circular velocity in km/s as a function of radius
    """
    return jnp.sqrt(jnp.where(r > 0, G_PC_MSUN_KMSQ_EFF * mass_enc / r, 0.))


def menc_from_vcirc(vcirc, r):
    """
    Enclosed mass given a circular velocity and radius

    Parameters
    ----------
    vcirc : float or array
        Circular velocity in km/s

    r : float or array
        Radius at which to calculate the enclosed mass in kpc

    Returns
    -------
    menc : float or array
        Enclosed mass in solar units
    """
    return vcirc ** 2 * r / G_PC_MSUN_KMSQ_EFF


def sersic_mr(r, mass, n, r_eff):
    """
    Radial surface mass density function for a generic sersic model

    Parameters
    ----------
    r : float or array
        Radius or radii at which to calculate the surface mass density

    mass : float
        Total mass of the Sersic component

    n : float
        Sersic index

    r_eff : float
        Effective radius

    Returns
    -------
    mr : float or array
        Surface mass density as a function of `r`
    """
    bn = gammaincinv(2. * n, 0.5)

    two_n = 2.0 * n
    I0 = (mass * bn ** two_n
          / (2.0 * jnp.pi * r_eff ** 2 * n * gamma(two_n)))

    x = (r / r_eff) ** (1.0 / n)
    mr = I0 * jnp.exp(-bn * x)

    return mr


def truncate_sersic_mr(r, mass, n, r_eff, r_inner, r_outer):
    """
    Radial surface mass density function for a truncated sersic model

    Parameters
    ----------
    r : float or array
        Radius or radii at which to calculate the surface mass density

    mass : float
        Total mass of the Sersic component

    n : float
        Sersic index

    r_eff : float
        Effective radius

    r_inner: float
        Inner truncation radius

    r_outer: float
        Outer truncation radius

    Returns
    -------
    mr : float or array
        Surface mass density as a function of `r`
    """
    mr = sersic_mr(r, mass, n, r_eff)

    mr = jnp.where((r >= r_inner) & (r <= r_outer), mr, 0.)

    return mr


def _I0_gaussring(r_peak, sigma_r, L_tot):
    """
    Normalisation constant for a Gaussian-ring intensity profile.

    I(r) = I0 * exp(-(r - r_peak)^2 / (2 * sigma_r^2))

    I0 = L_tot / (2*pi*sigma_r^2 * Ih)
    where Ih = sqrt(pi)*x*(1 + erf(x)) + exp(-x^2), x = r_peak/(sigma_r*sqrt(2))
    """
    x = r_peak / (sigma_r * jnp.sqrt(2.0))
    Ih = jnp.sqrt(jnp.pi) * x * (1.0 + erf(x)) + jnp.exp(-x ** 2)
    I0 = L_tot / (2.0 * jnp.pi * sigma_r ** 2 * Ih)
    return I0
