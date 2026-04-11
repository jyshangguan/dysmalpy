# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information.
#
# Higher-order kinematics models for DysmalPy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np
import jax.numpy as jnp
# import scipy.misc as scp_misc
# import scipy.integrate as scp_integrate

# Local imports
from .base import _DysmalFittable3DModel, HigherOrderKinematicsSeparate, \
                  HigherOrderKinematicsPerturbation
from dysmalpy.parameters import DysmalParameter

try:
    from dysmalpy.models import utils
except:
   from . import utils

__all__ = ['BiconicalOutflow', 'UnresolvedOutflow',
           'UniformRadialFlow', 'PlanarUniformRadialFlow',
           'AzimuthalPlanarRadialFlow',
           'UniformBarFlow', 'VariableXBarFlow',
           'UniformWedgeFlow',
           'SpiralDensityWave']

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")

class BiconicalOutflow(HigherOrderKinematicsSeparate, _DysmalFittable3DModel):
    r"""
    Model for a biconical outflow

    Parameters
    ----------
    n : float
        Power law index of the outflow velocity profile

    vmax : float
        Maximum velocity of the outflow in km/s

    rturn : float
        Turn-over radius in kpc of the velocty profile

    thetain : float
        Half inner opening angle in degrees. Measured from the bicone
        axis

    dtheta : float
        Difference between inner and outer opening angle in degrees

    rend : float
        Maximum radius of the outflow in kpc

    norm_flux : float
        Log flux amplitude of the outflow at r = 0

    tau_flux : float
        Exponential decay rate of the flux

    profile_type : {'both', 'increase', 'decrease', 'constant'}
        Type of velocity profile

    Notes
    -----
    This biconical outflow model is based on the model presented in Bae et al. (2016) [1]_.
    It consists of two symmetric cones joined at their apexes. `thetain` and `dtheta` control
    how hollow the cones are. `thetain` = 0 therefore would produce a fully filled cone.

    Within the cone, the velocity radial profile of the gas follows a power law with index `n`.
    Four different profile types can be selected. The simplest is 'constant' in which case
    the velocity of the gas is `vmax` at all radii.

    For a `profile_type` = 'increase':

        .. math::

            v = v_{\rm max}\left(\frac{r}{r_{\rm end}}\right)^n

    For a `profile_type` = 'decrease':

        .. math::

            v = v_{\rm max}\left(1 - \left(\frac{r}{r_{\rm end}}\right)^n\right)

    For a `profile_type` = 'both' the velocity first increases up to the turnover radius, `rturn`,
    then decreases back to 0 at 2 `rturn`.

    For :math:`r < r_{\rm turn}`:

        .. math::

            v =  v_{\rm max}\left(\frac{r}{r_{\rm turn}}\right)^n

    For :math:`r > r_{\rm turn}`:

        .. math::

            v = v_{\rm max}\left(2 - \frac{r}{r_{\rm turn}}\right)^n

    The flux radial profile of the outflow is described by a decreasing exponential:

        .. math::

            F = A\exp\left\{\frac{-\tau r}{r_{\rm end}}\right\}

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/2016ApJ...828...97B/abstract

    """

    n = DysmalParameter(default=0.5, fixed=True)
    vmax = DysmalParameter(min=0)
    rturn = DysmalParameter(default=0.5, min=0)
    thetain = DysmalParameter(bounds=(0, 90))
    dtheta = DysmalParameter(default=20.0, bounds=(0, 90))
    rend = DysmalParameter(default=1.0, min=0)
    norm_flux = DysmalParameter(default=0.0, fixed=True)
    tau_flux = DysmalParameter(default=5.0, fixed=True)

    _spatial_type = 'resolved'
    _multicoord_velocity = False
    _native_geometry = 'spherical'
    outputs = ('vout',)

    def __init__(self, profile_type='both', **kwargs):

        valid_profiles = ['increase', 'decrease', 'both', 'constant']

        if profile_type in valid_profiles:
            self.profile_type = profile_type
        else:
            logger.error("Invalid profile type. Must be one of 'increase',"
                         "'decrease', 'constant', or 'both.'")

        super(BiconicalOutflow, self).__init__(**kwargs)

    def __setstate__(self, state):
        # Compatibility hack, to handle the change to generalized
        #    higher order components in ModelSet.simulate_cube().
        super(BiconicalOutflow, self).__setstate__(state)

        # Change '_type' from 'outflow' to 'higher_order':
        if self._type == 'outflow':
            self._type = 'higher_order'


    def evaluate(self, x, y, z, n, vmax, rturn, thetain, dtheta, rend, norm_flux, tau_flux):
        """Evaluate the outflow velocity as a function of position x, y, z"""

        r = jnp.sqrt(x**2 + y**2 + z**2)
        theta = jnp.arccos(jnp.where(r > 0, jnp.abs(z) / r, 1.)) * 180. / jnp.pi
        vel = jnp.zeros_like(r)

        if self.profile_type == 'increase':

            amp = vmax/rend**n
            vel = jnp.where(r <= rend, amp * r ** n, vel)
            vel = jnp.where(r == 0, 0., vel)

        elif self.profile_type == 'decrease':

            amp = -vmax/rend**n
            vel = jnp.where(r <= rend, vmax + amp * r ** n, vel)

        elif self.profile_type == 'both':

            vel_inc = jnp.where(r <= rturn, vmax * (r / rturn) ** n, 0.)
            ind_outer = (r > rturn) & (r <= 2 * rturn)
            vel_dec = jnp.where(ind_outer, vmax * (2 - r / rturn) ** n, 0.)
            vel = vel_inc + vel_dec

        elif self.profile_type == 'constant':

            vel = jnp.where(r <= rend, vmax, vel)

        thetaout = jnp.minimum(thetain + dtheta, 90.)
        ind_zero = (theta < thetain) | (theta > thetaout) | (vel < 0)
        vel = jnp.where(ind_zero, 0., vel)

        return vel

    def velocity(self, x, y, z, *args):
        """Return the velocity as a function of x, y, z"""
        return self.evaluate(x, y, z, self.n, self.vmax, self.rturn, self.thetain,
                             self.dtheta, self.rend, self.norm_flux, self.tau_flux)

    def light_profile(self, x, y, z):
        """Evaluate the outflow line flux as a function of position x, y, z"""

        r = jnp.sqrt(x**2 + y**2 + z**2)
        theta = jnp.arccos(jnp.where(r > 0, jnp.abs(z) / r, 1.)) * 180. / jnp.pi
        flux = 10**self.norm_flux * jnp.exp(-self.tau_flux * (r / self.rend))
        thetaout = jnp.minimum(self.thetain + self.dtheta, 90.)
        ind_zero = ((theta < self.thetain) |
                    (theta > thetaout) |
                    (r > self.rend))
        flux = jnp.where(ind_zero, 1e-16, flux)       # To avoid NaNs

        return flux

    def vel_direction_emitframe(self, x, y, z, _save_memory=False):
        r"""
        Method to return the velocity direction in the outflow Cartesian frame.

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the outflow reference frame.

        _save_memory : bool, optional
            Option to save memory by only calculating the relevant matrices (eg during fitting).
            Default: False

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyz).

            For biconical outflows, this is the +rhat direction, in spherical coordinates
            (r,phi,theta).
        """
        r = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)

        vhat_y = jnp.where(r > 0, y / r, 0.)
        vhat_z = jnp.where(r > 0, z / r, 0.)

        if not _save_memory:
            vhat_x = jnp.where(r > 0, x / r, 0.)

            vel_dir_unit_vector = jnp.stack([vhat_x, vhat_y, vhat_z], axis=0)
        else:
            # Only calculate y,z directions
            vel_dir_unit_vector = jnp.stack([jnp.zeros_like(vhat_y), vhat_y, vhat_z], axis=0)

        return vel_dir_unit_vector



class UnresolvedOutflow(HigherOrderKinematicsSeparate, _DysmalFittable3DModel):
    """
    Model for an unresolved outflow component described by a Gaussian

    Parameters
    ----------
    vcenter : float
        Central velocity of the Gaussian in km/s

    fwhm : float
        FWHM of the Gaussian in km/s

    amplitude : float
        Amplitude of the Gaussian

    Notes
    -----
    This model simply produces a broad Gaussian spectrum that will be placed in the
    central spectrum of the galaxy.
    """

    vcenter = DysmalParameter(default=0)
    fwhm = DysmalParameter(default=1000.0, bounds=(0, None))
    amplitude = DysmalParameter(default=1.0, bounds=(0, None))

    _spatial_type = 'unresolved'
    _multicoord_velocity = False
    _native_geometry = 'cartesian'
    outputs = ('vout',)

    def __setstate__(self, state):
        # Compatibility hack, to handle the change to generalized
        #    higher order components in ModelSet.simulate_cube().
        super(UnresolvedOutflow, self).__setstate__(state)

        # Change '_type' from 'outflow' to 'higher_order':
        if self._type == 'outflow':
            self._type = 'higher_order'

    @staticmethod
    def evaluate(x, y, z, vcenter, fwhm, amplitude):
        return jnp.ones_like(x) * vcenter

    def __call__(self, x, y, z):
        return self.evaluate(x, y, z, self.vcenter, self.fwhm, self.amplitude)

    def velocity(self, x, y, z, *args):
        """Return the velocity as a function of x, y, z"""
        return self.evaluate(x, y, z, self.vcenter, self.fwhm, self.amplitude)

    def dispersion_profile(self, x, y, z, fwhm=None):
        """Dispersion profile for the outflow"""
        if fwhm is None: fwhm = self.fwhm
        return jnp.ones_like(x) * (fwhm / 2.35482)

    def light_profile(self, x, y, z):
        """Evaluate the outflow line flux as a function of position x, y, z
           All the light will be deposited at the center pixel."""

        r = jnp.sqrt(x**2 + y**2 + z**2)
        r_min = jnp.min(r)
        flux = self.amplitude * jnp.exp(-0.5 * ((r - r_min) / 0.01) ** 2)

        return flux

    def vel_direction_emitframe(self, x, y, z, _save_memory=False):
        r"""Method to return the velocity direction in the output geometry Cartesian frame.
        Undefined for `UnresolvedOutflow`"""
        return None



class UniformRadialFlow(HigherOrderKinematicsPerturbation, _DysmalFittable3DModel):
    """
    Model for a uniform radial flow.

    Parameters
    ----------
    vr : float
        Radial velocity in km/s. vr > 0 for outflow, vr < 0 for inflow

    Notes
    -----
    This model simply adds a constant radial velocity component
    to all of the positions in the galaxy.
    """
    vr = DysmalParameter(default=30.)

    _spatial_type = 'resolved'
    _multicoord_velocity = False
    _native_geometry = 'spherical'
    outputs = ('vrad',)

    def __init__(self, **kwargs):

        super(UniformRadialFlow, self).__init__(**kwargs)

    def __setstate__(self, state):
        # Compatibility hack, to handle the change to generalized
        #    higher order components in ModelSet.simulate_cube().
        super(UniformRadialFlow, self).__setstate__(state)

        # Change '_type' from 'flow' to 'higher_order':
        if self._type == 'flow':
            self._type = 'higher_order'

    @staticmethod
    def evaluate(x, y, z, vr):
        """Evaluate the radial velocity as a function of position x, y, z"""

        vel = jnp.ones_like(x) * (vr)

        return vel

    def __call__(self, x, y, z):
        return self.evaluate(x, y, z, self.vr)

    def velocity(self, x, y, z, *args):
        """Return the velocity as a function of x, y, z"""
        return self.evaluate(x, y, z, self.vr)


    def vel_direction_emitframe(self, x, y, z, _save_memory=False):
        r"""
        Method to return the velocity direction in the galaxy Cartesian frame.

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the radial flow reference frame.

        _save_memory : bool, optional
            Option to save memory by only calculating the relevant matrices (eg during fitting).
            Default: False

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyz).

            For a uniform radial flow, this is the +rhat direction, in spherical coordinates
            (r,phi,theta).
        """
        r = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)

        vhat_y = jnp.where(r > 0, y / r, 0.)
        vhat_z = jnp.where(r > 0, z / r, 0.)

        if not _save_memory:
            vhat_x = jnp.where(r > 0, x / r, 0.)

            vel_dir_unit_vector = jnp.stack([vhat_x, vhat_y, vhat_z], axis=0)
        else:
            # Only y,z directions
            vel_dir_unit_vector = jnp.stack([jnp.zeros_like(vhat_y), vhat_y, vhat_z], axis=0)

        return vel_dir_unit_vector

class PlanarUniformRadialFlow(HigherOrderKinematicsPerturbation, _DysmalFittable3DModel):
    """
    Model for a planar uniform radial flow, with radial flow only in the plane of the galaxy.

    Parameters
    ----------
    vr : float
        Radial velocity in km/s. vr > 0 for outflow, vr < 0 for inflow

    Notes
    -----
    This model simply adds a constant radial velocity component
    to all of the positions in the galaxy.
    """
    vr = DysmalParameter(default=30.)

    _spatial_type = 'resolved'
    _multicoord_velocity = False
    _native_geometry = 'cylindrical'
    outputs = ('vrad',)

    def __init__(self, **kwargs):

        super(PlanarUniformRadialFlow, self).__init__(**kwargs)

    @staticmethod
    def evaluate(x, y, z, vr):
        """Evaluate the radial velocity as a function of position x, y, z"""

        vel = jnp.ones_like(x) * (vr)

        return vel

    def __call__(self, x, y, z):
        return self.evaluate(x, y, z, self.vr)

    def velocity(self, x, y, z, *args):
        """Return the velocity as a function of x, y, z"""
        return self.evaluate(x, y, z, self.vr)


    def vel_direction_emitframe(self, x, y, z, _save_memory=False):
        r"""
        Method to return the velocity direction in the galaxy Cartesian frame.

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the radial flow reference frame.

        _save_memory : bool, optional
            Option to save memory by only calculating the relevant matrices (eg during fitting).
            Default: False

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyz).

            For a planar uniform radial flow, this is the +Rhat direction, in cylindrical coordinates
            (R,phi,z).
        """

        R = jnp.sqrt(x ** 2 + y ** 2)

        vhat_y = jnp.where(R > 0, y / R, 0.)

        if not _save_memory:
            vhat_x = jnp.where(R > 0, x / R, 0.)
            vhat_z = jnp.zeros_like(z)

            vel_dir_unit_vector = jnp.stack([vhat_x, vhat_y, vhat_z], axis=0)
        else:
            # Only y direction: z is by definition 0.
            vel_dir_unit_vector = jnp.stack([jnp.zeros_like(vhat_y), vhat_y, jnp.zeros_like(vhat_y)], axis=0)

        return vel_dir_unit_vector



class AzimuthalPlanarRadialFlow(HigherOrderKinematicsPerturbation, _DysmalFittable3DModel):
    """
    Model for a planar radial flow, with radial flow only in the plane of the galaxy,
        but that can vary azimuthally with angle phi relative to the galaxy major axis.

    Parameters
    ----------

    m : int
        Number of modes in the azimuthal pattern. m=0 gives a purely radial profile.

    phi0 : float
        Angle offset relative to the galaxy angle, so the azimuthal variation goes as
        cos(m(phi_gal - phi0))

    Notes
    -----
    The following functions must be specified, which take the galaxy radius R and model_set:
        vr(R):  Radial velocity in km/s. vr > 0 for outflow, vr < 0 for inflow
    """
    m = DysmalParameter(default=2.)
    phi0 = DysmalParameter(default=0., bounds=(0.,360.))

    _spatial_type = 'resolved'
    _multicoord_velocity = False
    _native_geometry = 'cylindrical'
    outputs = ('vrad',)

    def __init__(self, vr=None, **kwargs):

        self.vr = vr

        super(AzimuthalPlanarRadialFlow, self).__init__(**kwargs)

    def _get_geom_phi_rad_polar(self, x, y):
        """JAX-compatible polar angle phi from (x, y).

        Returns phi in [-pi, pi] computed as arcsin(y/R) with correction
        for x < 0.
        """
        R = jnp.sqrt(x ** 2 + y ** 2)
        phi = jnp.arcsin(jnp.where(R > 0, y / R, 0.))
        # For x < 0, the correct angle is pi - arcsin(y/R)
        phi = jnp.where(x < 0, jnp.pi - phi, phi)
        # At R=0, phi should be 0
        phi = jnp.where(R == 0, 0., phi)
        return phi

    def evaluate(self, x, y, z, m, phi0, model_set):
        """Evaluate the radial velocity as a function of position x, y, z"""
        phi0_rad = phi0 * jnp.pi / 180.
        phi_gal_rad = self._get_geom_phi_rad_polar(x, y)
        R = jnp.sqrt(x**2 + y**2)
        vel = self.vr(R, model_set) * jnp.cos(m * (phi_gal_rad - phi0_rad))

        return vel

    def velocity(self, x, y, z, model_set, *args):
        """Return the velocity as a function of x, y, z"""
        return self.evaluate(x, y, z, self.m, self.phi0, model_set)


    def vel_direction_emitframe(self, x, y, z, _save_memory=False):
        r"""
        Method to return the velocity direction in the galaxy Cartesian frame.

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the radial flow reference frame.

        _save_memory : bool, optional
            Option to save memory by only calculating the relevant matrices (eg during fitting).
            Default: False

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyz).

            For a planar uniform radial flow, this is the +Rhat direction, in cylindrical coordinates
            (R,phi,z).
        """

        R = jnp.sqrt(x ** 2 + y ** 2)

        vhat_y = jnp.where(R > 0, y / R, 0.)

        if not _save_memory:
            vhat_x = jnp.where(R > 0, x / R, 0.)
            vhat_z = jnp.zeros_like(z)

            vel_dir_unit_vector = jnp.stack([vhat_x, vhat_y, vhat_z], axis=0)
        else:
            # Only y direction: z is by definition 0.
            vel_dir_unit_vector = jnp.stack([jnp.zeros_like(vhat_y), vhat_y, jnp.zeros_like(vhat_y)], axis=0)

        return vel_dir_unit_vector


class UniformBarFlow(HigherOrderKinematicsPerturbation, _DysmalFittable3DModel):
    """
    Model for a uniform flow along a bar, along some axis in the galaxy midplane.

    Parameters
    ----------
    vbar : float
        Bar streaming velocity in km/s. vbar > 0 for outflow, vbar < 0 for inflow

    phi : float
        Azimuthal angle of bar, counter-clockwise from blue major axis. In degrees.
        Default is 90 (eg, along galaxy minor axis)

    bar_width : float
        Width of the bar, in kpc.
        So bar velocity only is nonzero between -bar_width/2 < ygal < bar_width/2.

    """

    vbar = DysmalParameter(default=30.)
    phi = DysmalParameter(default=90., bounds=(0.,360.))
    bar_width = DysmalParameter(default=2., bounds=(0., None))

    _spatial_type = 'resolved'
    _multicoord_velocity = False
    _native_geometry = 'cartesian'
    outputs = ('vflow',)

    def __init__(self, **kwargs):
        super(UniformBarFlow, self).__init__(**kwargs)

    @staticmethod
    def evaluate(x, y, z, vbar, phi, bar_width):
        """Evaluate the bar velocity amplitude as a function of position x, y, z"""

        phi_rad = phi * jnp.pi / 180.
        # Rotate by phi_rad:
        ybar = - jnp.sin(phi_rad) * x + jnp.cos(phi_rad) * y

        vel = jnp.ones_like(ybar) * (vbar)
        vel = jnp.where(jnp.abs(ybar) > 0.5 * bar_width, 0., vel)

        return vel

    def __call__(self, x, y, z):
        return self.evaluate(x, y, z, self.vbar, self.phi, self.bar_width)

    def velocity(self, x, y, z, *args):
        """Return the velocity as a function of x, y, z"""
        return self.evaluate(x, y, z, self.vbar, self.phi, self.bar_width)

    def vel_direction_emitframe(self, x, y, z, _save_memory=False):
        r"""
        Method to return the velocity direction in the galaxy Cartesian frame.

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the galaxy/bar reference frame.

        _save_memory : bool, optional
            Option to save memory by only calculating the relevant matrices (eg during fitting).
            Default: False

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyz).

            For a bar uniform flow, this is the sign(xbar) direction, in cartesian coordinates.
        """
        # Rotate by phi_rad:
        phi_rad = self.phi * jnp.pi / 180.
        xbar = jnp.cos(phi_rad) * x + jnp.sin(phi_rad) * y
        if not _save_memory:
            vel_dir_unit_vector = jnp.stack([jnp.cos(phi_rad) * jnp.sign(xbar),
                                             jnp.sin(phi_rad) * jnp.sign(xbar),
                                             jnp.zeros_like(z)], axis=0)
        else:
            # Only return y direction
            vel_dir_unit_vector = jnp.stack([jnp.zeros_like(xbar), jnp.sin(phi_rad) * jnp.sign(xbar), jnp.zeros_like(xbar)], axis=0)

        return vel_dir_unit_vector


class VariableXBarFlow(HigherOrderKinematicsPerturbation, _DysmalFittable3DModel):
    """
    Model for flow along a bar, along some axis in the galaxy midplane,
        with flow velocity that varies with bar distance abs(xbar) (under reflection symmetry).

    Parameters
    ----------

    phi : float
        Azimuthal angle of bar, counter-clockwise from blue major axis. In degrees.
        Default is 90 (eg, along galaxy minor axis)

    bar_width : float
        Width of the bar, in kpc.
        So bar velocity only is nonzero between -bar_width/2 < ygal < bar_width/2.

    Notes
    -----
    The following function must also be passed when setting up the model,
    which takes the bar coordinate xbar as an input:

    vbar(xbar, model_set)   [Amplidute of flow velocity as a function of bar coordinate abs(xbar).  vbar > 0 for outflow, vbar < 0 for inflow.]

    """

    phi = DysmalParameter(default=90., bounds=(0.,360.))
    bar_width = DysmalParameter(default=2., bounds=(0., None))

    _spatial_type = 'resolved'
    _multicoord_velocity = False
    _native_geometry = 'cartesian'
    outputs = ('vflow',)

    def __init__(self, vbar=None, **kwargs):

        # Set function: MUST TAKE |xbar|, model_set as input!
        self.vbar = vbar

        super(VariableXBarFlow, self).__init__(**kwargs)

    def evaluate(self, x, y, z, phi, bar_width, model_set):
        """Evaluate the bar velocity amplitude as a function of position x, y, z"""

        phi_rad = phi * jnp.pi / 180.
        # Rotate by phi_rad:
        xbar =   jnp.cos(phi_rad) * x + jnp.sin(phi_rad) * y
        ybar = - jnp.sin(phi_rad) * x + jnp.cos(phi_rad) * y

        vel = self.vbar(jnp.abs(xbar), model_set)
        vel = jnp.where(jnp.abs(ybar) > 0.5 * bar_width, 0., vel)

        return vel

    def velocity(self, x, y, z, model_set, *args):
        """Return the velocity as a function of x, y, z"""
        return self.evaluate(x, y, z, self.phi, self.bar_width, model_set)

    def vel_direction_emitframe(self, x, y, z, _save_memory=False):
        r"""
        Method to return the velocity direction in the galaxy Cartesian frame.

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the galaxy/bar reference frame.

        _save_memory : bool, optional
            Option to save memory by only calculating the relevant matrices (eg during fitting).
            Default: False

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyz).

            For a bar uniform flow, this is the sign(xbar) direction, in cartesian coordinates.
        """
        # Rotate by phi_rad:
        phi_rad = self.phi * jnp.pi / 180.
        xbar = jnp.cos(phi_rad) * x + jnp.sin(phi_rad) * y
        if not _save_memory:
            vel_dir_unit_vector = jnp.stack([jnp.cos(phi_rad) * jnp.sign(xbar),
                                             jnp.sin(phi_rad) * jnp.sign(xbar),
                                             jnp.zeros_like(z)], axis=0)
        else:
            # Only return y direction
            vel_dir_unit_vector = jnp.stack([jnp.zeros_like(xbar), jnp.sin(phi_rad) * jnp.sign(xbar), jnp.zeros_like(xbar)], axis=0)

        return vel_dir_unit_vector


class UniformWedgeFlow(HigherOrderKinematicsPerturbation, _DysmalFittable3DModel):
    """
    Model for a uniform planar radial flow (only in the plane of the galaxy),
        but that only has flow within two wedges of a given opening angle,
        centered at an angle phi relative to the galaxy major axis.

    Parameters
    ----------

    vr : float
        Radial velocity in km/s. vr > 0 for outflow, vr < 0 for inflow

    theta: float
        Opening angle of wedge (the full angular span)

    phi : float
        Angle offset relative to the galaxy angle, so the wedge center is at phi.
        (phi_gal - phi)

    """

    vr = DysmalParameter(default=30.)
    theta = DysmalParameter(default=60., bounds=(0.,180.))
    phi = DysmalParameter(default=0., bounds=(0.,360.))

    _spatial_type = 'resolved'
    _multicoord_velocity = False
    _native_geometry = 'cylindrical'
    outputs = ('vrad',)

    def __init__(self, **kwargs):
        super(UniformWedgeFlow, self).__init__(**kwargs)

    def _get_geom_phi_rad_polar(self, x, y):
        """JAX-compatible polar angle phi from (x, y).

        Returns phi in [-pi, pi] computed as arcsin(y/R) with correction
        for x < 0.
        """
        R = jnp.sqrt(x ** 2 + y ** 2)
        phi = jnp.arcsin(jnp.where(R > 0, y / R, 0.))
        # For x < 0, the correct angle is pi - arcsin(y/R)
        phi = jnp.where(x < 0, jnp.pi - phi, phi)
        # At R=0, phi should be 0
        phi = jnp.where(R == 0, 0., phi)
        return phi

    def evaluate(self, x, y, z, vr, theta, phi):
        """Evaluate the radial velocity as a function of position x, y, z"""
        phi_rad = phi * jnp.pi / 180.
        theta_rad = theta * jnp.pi / 180.
        phi_gal_rad = self._get_geom_phi_rad_polar(x, y)

        vel = jnp.ones_like(x) * (vr)
        vel = jnp.where(
            jnp.abs(jnp.cos(phi_gal_rad - phi_rad)) < jnp.abs(jnp.cos(0.5 * theta_rad)),
            0.,
            vel
        )

        return vel

    def velocity(self, x, y, z, *args):
        """Return the velocity as a function of x, y, z"""
        return self.evaluate(x, y, z, self.vr, self.theta, self.phi)


    def vel_direction_emitframe(self, x, y, z, _save_memory=False):
        r"""
        Method to return the velocity direction in the galaxy Cartesian frame.

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the radial flow reference frame.

        _save_memory : bool, optional
            Option to save memory by only calculating the relevant matrices (eg during fitting).
            Default: False

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyz).

            For a planar uniform radial flow, this is the +Rhat direction, in cylindrical coordinates
            (R,phi,z).
        """

        R = jnp.sqrt(x ** 2 + y ** 2)

        vhat_y = jnp.where(R > 0, y / R, 0.)

        if not _save_memory:
            vhat_x = jnp.where(R > 0, x / R, 0.)
            vhat_z = jnp.zeros_like(z)

            vel_dir_unit_vector = jnp.stack([vhat_x, vhat_y, vhat_z], axis=0)
        else:
            # Only y direction: z is by definition 0.
            vel_dir_unit_vector = jnp.stack([jnp.zeros_like(vhat_y), vhat_y, jnp.zeros_like(vhat_y)], axis=0)

        return vel_dir_unit_vector

class SpiralDensityWave(HigherOrderKinematicsPerturbation, _DysmalFittable3DModel):
    """
    Model for a spiral density wave, assumed in the galaxy midplane.

    Parameters
    ----------
    m : int
        Number of photometric/density spiral arms.

    cs : float
        Sound speed of medium, in km/s.

    epsilon : float
        Density contrast of perturbation (unitless).

    Om_p : float
        Angular speed of the driving force, :math:`\Omega_p`.

    phi0 : float, optional
        Angle offset of the arm winding, in degreed. Default: 0.

    Notes
    -----
    This model is implemented following the derivation given in the Appendix of
    Davies et al. 2009, ApJ, 702, 114 [1]_.

    Functions for the following must also be passed when setting up the model,
    which take the midplane galaxy radius R as an input:

    * Vrot(R)      [Unperturbed rotation velocity of the galaxy]
    * dVrot_dR(R)  [Derivative of Vrot(R) -- ideally evaluated analytically, otherwise very slow]
    * rho0(R)      [Unperturbed midplane density profile of the galaxy]
    * f(R, m, cs, Om_p, Vrot) [Function describing the spiral shape, :math:`m\phi = f(R)`, with :math:`k \equiv df/dR`]
    * k(R, m, cs, Om_p, Vrot) [Function for the radial wavenumber]

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/2009ApJ...702..114D/abstract
    """

    m = DysmalParameter(default=2., bounds=(0, None), fixed=True)       # Number of photometric arms
    phi0 = DysmalParameter(default=0., bounds=(0, 360))     # Angle offset of arm winding, in degrees
    cs = DysmalParameter(default=50., bounds=(0, None))     # Speed of sound, in km/s
    epsilon = DysmalParameter(default=0.1, bounds=(0,None)) # Density contrast of perturbation
    Om_p = DysmalParameter(default=0.)                      # Driving angular speed


    _spatial_type = 'resolved'
    _multicoord_velocity = True
    _native_geometry = 'cylindrical'
    outputs = ('vr','vphi','vz',)

    def __init__(self, Vrot=None, rho0=None,
                 f=None, k=None, dVrot_dR=None,
                 **kwargs):

        # Set functions: MUST TAKE R as input!
        self.Vrot = Vrot
        self.rho0 = rho0
        self._f_func = f
        self._k_func = k
        self.dVrot_dR = dVrot_dR

        # Functions of R (midplane): rho0, f, k, Vrot,
        # -> k and f are actually derived....
        #       but separately calculate, very slow to always do numerically!!!

        super(SpiralDensityWave, self).__init__(**kwargs)

    def evaluate(self, x, y, z, m, phi, cs, epsilon, Om_p, Vrot, rho0):
        """Evaluate the spiral density wave at x,y,z"""

        vr1 = self.vr_perturb(x, y, z)
        vphi1 = self.vphi_perturb(x, y, z)

        # Tuple in the native cylindrical coordinates (really polar, ignoring z)
        return (vr1, vphi1, jnp.zeros_like(z))

    def _get_geom_phi_rad_polar(self, x, y):
        """JAX-compatible polar angle phi from (x, y).

        Returns phi in [-pi, pi] computed as arcsin(y/R) with correction
        for x < 0.
        """
        R = jnp.sqrt(x ** 2 + y ** 2)
        phi = jnp.arcsin(jnp.where(R > 0, y / R, 0.))
        # For x < 0, the correct angle is pi - arcsin(y/R)
        phi = jnp.where(x < 0, jnp.pi - phi, phi)
        # At R=0, phi should be 0
        phi = jnp.where(R == 0, 0., phi)
        return phi

    def ep_freq_sq(self, R):
        """ Return kappa^2, square of the epicyclic frequency """
        V = self.Vrot(R)
        Om = V / R
        dV_dR = self.dVrot_dR(R)

        kappasq = 2. * Om * (R * (1. / R * dV_dR - V / R ** 2) + 2. * Om)

        return kappasq

    def k(self, R):
        """Calculate wavenumber"""

        return self._k_func(R, self.m,  self.cs,  self.Om_p,  self.Vrot)

    def f(self, R):
        """Shape of spiral arms, with f=m*phi = Int_0^R(k dR) """

        return self._f_func(R, self.m,  self.cs,  self.Om_p,  self.Vrot)

    def rho_perturb(self, x, y, z):
        """
        Return the density perturbation -- consider only the midplane
        Given by Eq. A9, Davies et al. 2009, ApJ, 702, 114
        """

        R = jnp.sqrt(x**2 + y**2)
        phi0_rad = self.phi0 * jnp.pi / 180.
        phi = self._get_geom_phi_rad_polar(x, y) - phi0_rad

        fvals = self.f(R)
        rho0vals = self.rho0(R)

        rho1 = self.epsilon * rho0vals * jnp.cos(self.m * phi - fvals)

        # Excise R=0 values
        rho1 = jnp.where(R == 0, 0., rho1)

        return rho1

    def vr_perturb(self, x, y, z):
        """
        Return the radial velocity perturbation
        Given by Eq. A10, Davies et al. 2009, ApJ, 702, 114

        Here inflow is NEGATIVE, outflow is POSITIVE.
        """

        R = jnp.sqrt(x**2 + y**2)
        Om = self.Vrot(R) / R
        phi0_rad = self.phi0 * jnp.pi / 180.
        phi = self._get_geom_phi_rad_polar(x, y) - phi0_rad

        fvals = self.f(R)
        kvals = self.k(R)

        vr1 = -self.epsilon * self.m * (Om - self.Om_p) / kvals * jnp.cos(self.m * phi - fvals)

        # Excise R=0 values
        vr1 = jnp.where(R == 0, 0., vr1)

        return vr1


    def vphi_perturb(self, x, y, z):
        """
        Return the phi-direction velocity perturbation
        Given by Eq. A11, Davies et al. 2009, ApJ, 702, 114
        """

        R = jnp.sqrt(x**2 + y**2)
        Om = self.Vrot(R) / R
        phi0_rad = self.phi0 * jnp.pi / 180.
        phi = self._get_geom_phi_rad_polar(x, y) - phi0_rad

        fvals = self.f(R)
        kvals = self.k(R)

        kappasq = self.ep_freq_sq(R)

        vphi1 = self.epsilon * kappasq / (2. * kvals * Om) * jnp.sin(self.m * phi - fvals)

        # Excise R=0 values
        vphi1 = jnp.where(R == 0, 0., vphi1)

        return vphi1

    def vLOS_perturb(self, x, y, z):
        """
        Return the projected LOS velocity combining both radial and phi components.
        For use in visualizing.
        Handled via matrix multiplication & projection for model_set.simulate_cube()

        Uses NEGATIVE for inflow, POSITIVE for outflow
        """

        R = jnp.sqrt(x ** 2 + y ** 2)
        vr1 = self.vr_perturb(x, y, z)
        vphi1 = self.vphi_perturb(x, y, z)

        vLOS1 = jnp.where(R > 0,
                          vr1 * (y / R) + vphi1 * (x / R),
                          0.)

        return vLOS1


    def velocity(self, x, y, z, *args):
        """
        Evaluate the spiral density velocity amplitude as a function of position x, y, z,
        for the different radial and phi components.
        Return a tuple of (vr, vphi, vz).
        """

        vr1 = self.vr_perturb(x, y, z)
        vphi1 = self.vphi_perturb(x, y, z)

        # Tuple in the native cylindrical coordinates (really polar, ignoring z)
        return (vr1, vphi1, jnp.zeros_like(z))


    def light_profile(self, x, y, z):

        return self.rho_perturb(x, y, z)

    def vel_direction_emitframe(self, x, y, z, _save_memory=False):
        r"""
        Method to return the velocity matrix in the output Cartesian frame.

        As the native geometry is cylindrical (ignoring z dir), and the velocity is multicoordinate,
        we need a more complex output than a single vector for a dotproduct.

        vel = (R, phi, z).
        Need matmul(vel_dir_matrix, vel) = vel in (x,y,z). So:

        .. code-block:: text

            vel_dir_matrix = [[Rtox, phitox, ztox],
                            [Rtoy, phitoy, ztoy],
                            [Rtoz, phitoz, ztoz]]

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the output reference frame.

        _save_memory : bool, optional
            Option to save memory by only calculating the relevant matrices (eg during fitting).
            Default: False

        Returns
        -------
        vel_dir_matrix : 3x3-element matrix
            Transform of the velocity from the native coordinates to the output cartesian frame.

        """
        R = jnp.sqrt(x ** 2 + y ** 2)

        vhat_Rtoy = jnp.where(R > 0, y / R, 0.)
        vhat_phitoy = jnp.where(R > 0, x / R, 0.)

        if not _save_memory:
            vhat_Rtox = jnp.where(R > 0, x / R, 0.)
            vhat_phitox = jnp.where(R > 0, -y / R, 0.)

            vhat_ztox = jnp.zeros_like(z)
            vhat_ztoy = jnp.zeros_like(z)
            vhat_Rtoz = jnp.zeros_like(z)
            vhat_phitoz = jnp.zeros_like(z)
            vhat_ztoz = jnp.zeros_like(z)

            row0 = jnp.stack([vhat_Rtox, vhat_phitox, vhat_ztox], axis=0)
            row1 = jnp.stack([vhat_Rtoy, vhat_phitoy, vhat_ztoy], axis=0)
            row2 = jnp.stack([vhat_Rtoz, vhat_phitoz, vhat_ztoz], axis=0)
            vel_dir_matrix = jnp.stack([row0, row1, row2], axis=0)

        else:
            # Only return y directions (z dir are all 0):
            row0 = jnp.stack([jnp.zeros_like(R), jnp.zeros_like(R), jnp.zeros_like(R)], axis=0)
            row1 = jnp.stack([vhat_Rtoy, vhat_phitoy, jnp.zeros_like(R)], axis=0)
            row2 = jnp.stack([jnp.zeros_like(R), jnp.zeros_like(R), jnp.zeros_like(R)], axis=0)
            vel_dir_matrix = jnp.stack([row0, row1, row2], axis=0)

        return vel_dir_matrix
