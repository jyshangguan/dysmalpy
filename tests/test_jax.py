# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information.
#
# Unit tests for JAX-accelerated DysmalPy functions.
#
# Tests individual JAX components in isolation:
#   1. Special functions (gammaincinv, hyp2f1, bessel_k0, bessel_k1)
#   2. DysmalParameter descriptor
#   3. Cube population (populate_cube_jax, populate_cube_jax_ais)
#   4. Model computations (enclosed_mass, circular_velocity, geometry)
#   5. Integration: full model pipeline

import math

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import scipy.special as scp_spec
from scipy.signal import fftconvolve

from dysmalpy.special import gammaincinv, hyp2f1, bessel_k0, bessel_k1
from dysmalpy import parameters, models
from dysmalpy.models.cube_processing import (
    populate_cube_jax, populate_cube_jax_ais,
    _simulate_cube_inner_direct, _simulate_cube_inner_ais,
    _make_cube_ai, _get_xyz_sky_gal, _get_xyz_sky_gal_inverse,
    _calculate_max_skyframe_extents,
)
from dysmalpy.convolution import _fft_convolve_3d, convolve_cube_jax, get_jax_kernels, _rebin_spatial


# ===================================================================
# 1. Special function tests
# ===================================================================

class TestSpecialGammaincinv:
    """Test inverse regularized lower incomplete gamma function."""

    def test_scalar_identity(self):
        """gammainc(a, gammaincinv(a, p)) == p for various a, p."""
        rtol = 1e-6
        for a in [0.5, 1.0, 2.0, 4.0, 8.0, 10.0]:
            for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
                x = float(gammaincinv(a, p))
                recovered = scp_spec.gammainc(a, x)
                assert math.isclose(recovered, p, rel_tol=rtol), \
                    f"gammaincinv({a}, {p}) = {x}, gammainc({a}, {x}) = {recovered} != {p}"

    def test_sersic_bn(self):
        """Test bn = gammaincinv(2*n, 0.5) against scipy for standard Sersic indices."""
        rtol = 1e-6
        for n in [0.5, 1.0, 2.0, 4.0, 8.0]:
            bn_jax = float(gammaincinv(2.0 * n, 0.5))
            bn_scp = scp_spec.gammaincinv(2.0 * n, 0.5)
            assert math.isclose(bn_jax, bn_scp, rel_tol=rtol), \
                f"n={n}: bn_jax={bn_jax}, bn_scp={bn_scp}"

    def test_array_input(self):
        """Test with array inputs."""
        a_arr = jnp.array([1.0, 2.0, 4.0, 8.0])
        p_arr = jnp.array([0.5, 0.5, 0.5, 0.5])
        result = gammaincinv(a_arr, p_arr)
        assert result.shape == (4,)
        for i, a in enumerate([1.0, 2.0, 4.0, 8.0]):
            expected = scp_spec.gammaincinv(a, 0.5)
            assert math.isclose(float(result[i]), expected, rel_tol=1e-6)

    def test_grad_exists(self):
        """Verify gradient computation works (doesn't raise)."""
        def f(p):
            return gammaincinv(2.0, p)
        grad_f = jax.grad(f)
        result = grad_f(0.5)
        assert jnp.isfinite(result)


class TestSpecialHyp2f1:
    """Test Gauss hypergeometric function 2F1."""

    def test_small_z(self):
        """Compare with scipy for |z| < 1."""
        rtol = 1e-4
        a, b, c = 1.5, 2.5, 3.5
        for z in [0.0, 0.1, 0.5, 0.9, -0.5, -0.9]:
            result = float(hyp2f1(a, b, c, z))
            expected = scp_spec.hyp2f1(a, b, c, z)
            assert math.isclose(result, expected, rel_tol=rtol), \
                f"hyp2f1({a},{b},{c},{z}): got {result}, expected {expected}"

    def test_large_negative_z(self):
        """Test for z < -1 (linear fractional transformation path)."""
        rtol = 1e-4
        a, b, c = 2.0, 5.0, 3.0
        for z in [-2.0, -5.0, -10.0]:
            result = float(hyp2f1(a, b, c, z))
            expected = scp_spec.hyp2f1(a, b, c, z)
            assert math.isclose(result, expected, rel_tol=rtol), \
                f"hyp2f1({a},{b},{c},{z}): got {result}, expected {expected}"

    def test_two_power_halo_args(self):
        """Test with typical TwoPowerHalo arguments: hyp2f1(3-alpha, beta-alpha, 4-alpha, z)."""
        rtol = 1e-4
        alpha, beta = 1.0, 3.0
        a = 3.0 - alpha  # 2.0
        b = beta - alpha  # 2.0
        c = 4.0 - alpha  # 3.0
        for conc in [5.0, 10.0, 20.0]:
            z = -conc
            result = float(hyp2f1(a, b, c, z))
            expected = scp_spec.hyp2f1(a, b, c, z)
            assert math.isclose(result, expected, rel_tol=rtol), \
                f"hyp2f1({a},{b},{c},{z}): got {result}, expected {expected}"

    def test_grad_exists(self):
        """Verify gradient computation works."""
        def f(z):
            return hyp2f1(2.0, 5.0, 3.0, z)
        grad_f = jax.grad(f)
        result = grad_f(-5.0)
        assert jnp.isfinite(result)


class TestSpecialBessel:
    """Test modified Bessel functions K0, K1."""

    def test_bessel_k0(self):
        """Compare K0 with scipy for various arguments."""
        rtol = 1e-4
        for y in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            result = float(bessel_k0(y))
            expected = scp_spec.k0(y)
            assert math.isclose(result, expected, rel_tol=rtol), \
                f"k0({y}): got {result}, expected {expected}"

    def test_bessel_k1(self):
        """Compare K1 with scipy for various arguments."""
        rtol = 1e-4
        for y in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            result = float(bessel_k1(y))
            expected = scp_spec.k1(y)
            assert math.isclose(result, expected, rel_tol=rtol), \
                f"k1({y}): got {result}, expected {expected}"

    def test_array_input(self):
        """Test with array inputs."""
        y_arr = jnp.array([0.5, 1.0, 2.0, 5.0])
        result_k0 = bessel_k0(y_arr)
        result_k1 = bessel_k1(y_arr)
        assert result_k0.shape == (4,)
        assert result_k1.shape == (4,)
        for i, y in enumerate([0.5, 1.0, 2.0, 5.0]):
            assert math.isclose(float(result_k0[i]), scp_spec.k0(y), rel_tol=1e-4)
            assert math.isclose(float(result_k1[i]), scp_spec.k1(y), rel_tol=1e-4)

    def test_grad_exists(self):
        """Verify gradient computation works."""
        grad_k0 = jax.grad(bessel_k0)
        grad_k1 = jax.grad(bessel_k1)
        assert jnp.isfinite(grad_k0(1.0))
        assert jnp.isfinite(grad_k1(1.0))


# ===================================================================
# 2. DysmalParameter descriptor tests
# ===================================================================

class TestDysmalParameter:
    """Test standalone DysmalParameter descriptor."""

    def test_basic_descriptor(self):
        """Parameter stores and retrieves values correctly."""
        class FakeModel(models.base._DysmalModel):
            x = parameters.DysmalParameter(default=1.0, name='x')

        m = FakeModel()
        assert m.x == 1.0
        m.x = 5.0
        assert m.x == 5.0

    def test_bounds(self):
        """Parameter bounds accessible via _param_instances and bounds dict."""
        class FakeModel(models.base._DysmalModel):
            x = parameters.DysmalParameter(default=1.0, bounds=(0.0, 10.0), name='x')

        m = FakeModel()
        # Access via _param_instances to get the descriptor
        assert m._param_instances['x'].bounds == (0.0, 10.0)
        # Access via bounds dict on the model
        assert m.bounds['x'] == (0.0, 10.0)

    def test_fixed(self):
        """Parameter fixed attribute accessible via _param_instances and fixed dict."""
        class FakeModel(models.base._DysmalModel):
            x = parameters.DysmalParameter(default=1.0, fixed=True, name='x')

        m = FakeModel()
        assert m._param_instances['x'].fixed == True
        assert m.fixed['x'] == True

    def test_prior(self):
        """Parameter can hold a prior."""
        class FakeModel(models.base._DysmalModel):
            x = parameters.DysmalParameter(default=1.0, name='x')

        m = FakeModel()
        m._param_instances['x'].prior = parameters.GaussianPrior(center=1.0, stddev=0.5)
        assert isinstance(m._param_instances['x'].prior, parameters.GaussianPrior)

    def test_copy(self):
        """Model with parameters can be copied."""
        class FakeModel(models.base._DysmalModel):
            x = parameters.DysmalParameter(default=1.0, name='x')

        m = FakeModel()
        m.x = 5.0
        m2 = m.copy()
        assert m2.x == 5.0
        m2.x = 10.0
        assert m.x == 5.0  # original unchanged


# ===================================================================
# 3. Cube population tests
# ===================================================================

class TestPopulateCube:
    """Test JAX cube population functions."""

    def test_populate_cube_basic(self):
        """populate_cube_jax produces a cube of correct shape."""
        flux = jnp.ones((5, 10, 10))
        vel = jnp.zeros((5, 10, 10))
        sigma = jnp.ones((5, 10, 10))
        vspec = jnp.array([-2., -1., 0., 1., 2.])

        cube = populate_cube_jax(flux, vel, sigma, vspec)
        assert cube.shape == (5, 10, 10)
        # With zero velocity, the Gaussian is centered at 0.
        # The center channel (index 2, vspec=0) should have the most flux.
        center_sum = float(jnp.sum(cube[2, :, :]))
        edge_sum = float(jnp.sum(cube[0, :, :]))
        assert center_sum > edge_sum

    def test_populate_cube_shifted_velocity(self):
        """Shifted velocity shifts the peak in the cube."""
        flux = jnp.ones((3, 5, 5))
        vel = jnp.full((3, 5, 5), 50.0)  # velocity at 50 km/s
        sigma = jnp.ones((3, 5, 5))
        vspec = jnp.array([0., 50., 100.])

        cube = populate_cube_jax(flux, vel, sigma, vspec)
        assert cube.shape == (3, 5, 5)
        # Peak should be at channel 1 (vspec=50)
        center_sum = float(jnp.sum(cube[1, :, :]))
        assert center_sum > 0

    def test_populate_cube_ais_matches_direct(self):
        """AIS (sparse) variant matches direct variant for all active pixels."""
        np.random.seed(42)
        flux = jnp.array(np.random.rand(5, 8, 8))
        vel = jnp.array(np.random.rand(5, 8, 8) * 200 - 100)
        sigma = jnp.array(np.random.rand(5, 8, 8) * 50 + 10)
        vspec = jnp.linspace(-300, 300, 61)

        # Use all pixels as active
        zi, yi, xi = np.indices((5, 8, 8))
        ai = jnp.array(np.vstack([xi.flatten(), yi.flatten(), zi.flatten()]))

        cube_direct = populate_cube_jax(flux, vel, sigma, vspec)
        cube_ais = populate_cube_jax_ais(flux, vel, sigma, vspec, ai)

        # AIS with all active pixels should match direct
        np.testing.assert_allclose(
            np.array(cube_direct), np.array(cube_ais), rtol=1e-5, atol=1e-8
        )

    def test_populate_cube_ais_subset(self):
        """AIS with subset of pixels zeros out inactive pixels."""
        flux = jnp.ones((3, 4, 4))
        vel = jnp.zeros((3, 4, 4))
        sigma = jnp.ones((3, 4, 4))
        vspec = jnp.array([-1., 0., 1.])

        # Only activate the central 2x2 pixels in the first z-slice
        ai = jnp.array([[1, 2, 1, 2],
                        [1, 1, 2, 2],
                        [0, 0, 0, 0]])

        cube = populate_cube_jax_ais(flux, vel, sigma, vspec, ai)
        assert cube.shape == (3, 4, 4)
        # Pixel (0, 0) should be zero (not in active set)
        assert float(cube[1, 0, 0]) == 0.0
        # Active pixels should have non-zero flux
        assert float(cube[1, 1, 1]) > 0.0


# ===================================================================
# 4. Coordinate transform tests
# ===================================================================

class TestGeometry:
    """Test geometry coordinate transforms."""

    def test_face_on(self):
        """Face-on (inc=0) preserves radial distances in x, y plane."""
        geom = models.Geometry(inc=0., pa=0., xshift=0., yshift=0., obs_name='test')
        xsky = jnp.array([1., 2., 3.])
        ysky = jnp.array([0., 0., 0.])
        zsky = jnp.array([0., 0., 0.])
        xgal, ygal, zgal = geom(xsky, ysky, zsky)
        # With inc=0, zgal should be ~0
        np.testing.assert_allclose(np.array(zgal), [0., 0., 0.], atol=1e-10)
        # Radial distance in (x,y) should be preserved
        r_in = np.sqrt([1., 4., 9.])
        r_out = np.sqrt(np.array(xgal)**2 + np.array(ygal)**2)
        np.testing.assert_allclose(r_in, r_out, atol=1e-10)

    def test_edge_on(self):
        """Edge-on (inc=90) maps y_sky into z_gal, preserving total distance."""
        geom = models.Geometry(inc=90., pa=0., xshift=0., yshift=0., obs_name='test')
        xsky = jnp.array([0.])
        ysky = jnp.array([1.])
        zsky = jnp.array([0.])
        xgal, ygal, zgal = geom(xsky, ysky, zsky)
        # With inc=90, the y_sky contribution goes into z_gal
        r_in = np.sqrt(float(xsky[0])**2 + float(ysky[0])**2)
        r_out = np.sqrt(float(xgal[0])**2 + float(ygal[0])**2 + float(zgal[0])**2)
        np.testing.assert_allclose(r_in, r_out, atol=1e-10)
        assert abs(float(zgal[0])) > 0.

    def test_inverse_roundtrip(self):
        """Inverse transform roundtrips."""
        geom = models.Geometry(inc=30., pa=45., xshift=0., yshift=0., obs_name='test')
        xgal = jnp.array([1., 2., -3.])
        ygal = jnp.array([0.5, -1., 2.])
        zgal = jnp.array([0., 0., 0.])
        xsky, ysky, zsky = geom.inverse_coord_transform(xgal, ygal, zgal)
        xgal2, ygal2, zgal2 = geom(xsky, ysky, zsky)
        np.testing.assert_allclose(np.array(xgal), np.array(xgal2), atol=1e-10)
        np.testing.assert_allclose(np.array(ygal), np.array(ygal2), atol=1e-10)
        np.testing.assert_allclose(np.array(zgal), np.array(zgal2), atol=1e-10)


# ===================================================================
# 5. Model computation tests
# ===================================================================

class TestNFWHalo:
    """Test NFW halo enclosed mass and circular velocity."""

    def setup_method(self):
        self.halo = models.NFW(mvirial=12.0, conc=5.0, fdm=0.5, z=1.613, name='halo')

    def test_rvir(self):
        """Rvir matches expected value."""
        rtol = 1e-9
        assert math.isclose(self.halo.calc_rvir(), 113.19184480200144, rel_tol=rtol)

    def test_enclosed_mass(self):
        """Enclosed mass matches expected values."""
        rtol = 1e-6
        rarr = [0., 2.5, 5., 7.5, 10., 50.]
        expected = [0.0, 5529423277.0931425, 19459875132.71848,
                    38918647245.552315, 62033461205.42702, 498218492834.53705]
        for r, exp in zip(rarr, expected):
            result = float(self.halo.enclosed_mass(r))
            assert math.isclose(result, exp, rel_tol=rtol), \
                f"NFW enclosed_mass({r}): got {result}, expected {exp}"

    def test_circular_velocity(self):
        """Circular velocity matches expected values."""
        rtol = 1e-6
        rarr = [0., 2.5, 5., 7.5, 10., 50.]
        expected = [0.0, 97.53274745638375, 129.37952931721014,
                    149.39249515561673, 163.34037609257453, 207.0167394246318]
        for r, exp in zip(rarr, expected):
            result = float(self.halo.circular_velocity(r))
            assert math.isclose(result, exp, rel_tol=rtol), \
                f"NFW circular_velocity({r}): got {result}, expected {exp}"


class TestSersicBaryon:
    """Test Sersic baryonic component."""

    def setup_method(self):
        self.sersic = models.Sersic(
            total_mass=11.0, r_eff=5.0, n=1.0, invq=5.0,
            noord_flat=False, name='sersic'
        )

    def test_enclosed_mass(self):
        """Enclosed mass matches expected values."""
        rtol = 1e-6
        rarr = [0., 2.5, 5., 7.5, 10.]
        expected = [0.0, 20535293937.195515, 50000000000.0,
                    71627906617.42969, 84816797558.70425]
        for r, exp in zip(rarr, expected):
            result = float(self.sersic.enclosed_mass(r))
            assert math.isclose(result, exp, rel_tol=rtol), \
                f"Sersic enclosed_mass({r}): got {result}, expected {exp}"

    def test_circular_velocity(self):
        """Circular velocity matches expected values."""
        rtol = 1e-6
        rarr = [0., 2.5, 5., 7.5, 10.]
        expected = [0.0, 187.95808079510437, 207.38652969925448,
                    202.6707348023267, 190.9947720259013]
        for r, exp in zip(rarr, expected):
            result = float(self.sersic.circular_velocity(r))
            assert math.isclose(result, exp, rel_tol=rtol), \
                f"Sersic circular_velocity({r}): got {result}, expected {exp}"


class TestDiskBulge:
    """Test combined Disk+Bulge component."""

    def setup_method(self):
        self.bary = models.DiskBulge(
            total_mass=11.0, bt=0.3, r_eff_disk=5.0, n_disk=1.0,
            invq_disk=5.0, r_eff_bulge=1.0, n_bulge=4.0, invq_bulge=1.0,
            noord_flat=True, name='disk+bulge', gas_component='total'
        )

    def test_circular_velocity(self):
        """Circular velocity matches expected values."""
        rtol = 1e-6
        rarr = [0., 2.5, 5., 7.5, 10.]
        expected = [0., 234.5537625461696, 232.5422746300449,
                    222.81686610765897, 207.87067673318512]
        for r, exp in zip(rarr, expected):
            result = float(self.bary.circular_velocity(r))
            assert math.isclose(result, exp, rel_tol=rtol), \
                f"DiskBulge circular_velocity({r}): got {result}, expected {exp}"

    def test_enclosed_mass(self):
        """Enclosed mass matches expected values."""
        rtol = 1e-6
        rarr = [0., 2.5, 5., 7.5, 10.]
        expected = [0.0, 33608286555.18022, 59533241580.68471,
                    76951223276.7583, 87374973250.0867]
        for r, exp in zip(rarr, expected):
            result = float(self.bary.enclosed_mass(r))
            assert math.isclose(result, exp, rel_tol=rtol), \
                f"DiskBulge enclosed_mass({r}): got {result}, expected {exp}"


class TestTwoPowerHalo:
    """Test TwoPowerHalo enclosed mass and circular velocity."""

    def setup_method(self):
        self.halo = models.TwoPowerHalo(
            mvirial=12.0, conc=5.0, alpha=0., beta=3., fdm=0.5, z=1.613, name='halo'
        )

    def test_enclosed_mass(self):
        """Enclosed mass matches expected values."""
        rtol = 1e-6
        rarr = [0., 2.5, 5., 7.5, 10., 50.]
        expected = [0.0, 579896865.582416, 3741818525.6905646,
                    10367955836.483051, 20480448580.59536, 393647104816.9644]
        for r, exp in zip(rarr, expected):
            result = float(self.halo.enclosed_mass(r))
            assert math.isclose(result, exp, rel_tol=rtol), \
                f"TPH enclosed_mass({r}): got {result}, expected {exp}"

    def test_circular_velocity(self):
        """Circular velocity matches expected values."""
        rtol = 1e-6
        rarr = [0., 2.5, 5., 7.5, 10., 50.]
        expected = [0.0, 31.585366510730005, 56.73315065922273,
                    77.10747504831834, 93.85345758098778, 184.0132403616831]
        for r, exp in zip(rarr, expected):
            result = float(self.halo.circular_velocity(r))
            assert math.isclose(result, exp, rel_tol=rtol), \
                f"TPH circular_velocity({r}): got {result}, expected {exp}"


# ===================================================================
# 6. Utility function tests
# ===================================================================

class TestUtils:
    """Test utility functions."""

    def test_replace_values_by_refarr(self):
        """replace_values_by_refarr replaces arr where ref==excise_val, keeps arr otherwise."""
        from dysmalpy.models import utils as model_utils
        arr = jnp.array([0., 1., 2., 3., 4., 5.])
        ref = jnp.array([0., 0., 0., 3., 3., 3.])
        # Where ref==0: replace arr with 99. Where ref!=0: keep original arr value.
        result = model_utils.replace_values_by_refarr(arr, ref, 0., 99.)
        np.testing.assert_allclose(
            np.array(result), [99., 99., 99., 3., 4., 5.], atol=1e-10
        )

    def test_get_geom_phi_rad_polar(self):
        """get_geom_phi_rad_polar computes correct angles."""
        from dysmalpy.models import utils as model_utils
        x = jnp.array([1., 0., -1., 0.])
        y = jnp.array([0., 1., 0., -1.])
        result = model_utils.get_geom_phi_rad_polar(x, y)
        np.testing.assert_allclose(
            np.array(result), [0., np.pi/2, np.pi, -np.pi/2], atol=1e-10
        )


# ===================================================================
# 7. Helper function tests (numpy-only, setup phase)
# ===================================================================

class TestCubeHelpers:
    """Test numpy-only cube setup helpers."""

    def test_get_xyz_sky_gal(self):
        """Coordinate grids have correct shape."""
        geom = models.Geometry(inc=30., pa=0., xshift=0., yshift=0., obs_name='test')
        sh = (3, 5, 7)
        xgal, ygal, zgal, xsky, ysky, zsky = _get_xyz_sky_gal(geom, sh, 1., 2., 1.5)
        assert xgal.shape == sh
        assert ygal.shape == sh
        assert zgal.shape == sh

    def test_calculate_max_skyframe_extents(self):
        """Max sky frame extents return correct types and odd nz."""
        geom = models.Geometry(inc=45., pa=0., xshift=0., yshift=0., obs_name='test')
        nz, maxr, maxr_y = _calculate_max_skyframe_extents(geom, 33., 33., 'direct', 'cos')
        assert isinstance(nz, (int, np.integer))
        assert nz % 2 == 1
        assert maxr > 0
        assert maxr_y > 0


# ===================================================================
# 8. Phase 5: JAX loss function tests
# ===================================================================

def _make_test_galaxy_obs():
    """Build a test galaxy + observation for loss function testing.

    Uses the same setup as test_models.TestModels.test_simulate_cube.
    """
    from dysmalpy import galaxy, observation, instrument

    z = 0.5
    name = 'test_gal'

    gal = galaxy.Galaxy(z=z, name=name)
    obs = observation.Observation(name='halpha_1D', tracer='halpha')
    obs.mod_options.oversample = 3
    obs.mod_options.zcalc_truncate = True

    mod_set = models.ModelSet()

    # Geometry
    geom = models.Geometry(inc=30., pa=0., xshift=0., yshift=0.,
                           name='geom', obs_name='halpha_1D')

    # Disk+bulge
    bary = models.DiskBulge(
        name='disk+bulge',
        total_mass=11., r_d=3.0, bt=0.0,
        n_bulge=1.0, invq_disk=0.2, r_eff_bulge=1.0,
        mass_to_light=1.0, f_gas=0.1,
    )

    # NFW halo
    halo = models.NFW(name='halo', mvirial=12., fdm=0.5, conc=10.)

    # Dispersion
    disp_prof = models.DispersionConst(name='dispersion_halpha',
                                        sigma=50., tracer='halpha')

    # z-height
    zheight_prof = models.ZHeightExp(name='zheight', z0=1.0)

    mod_set.add_component(bary, light=True)
    mod_set.add_component(halo)
    mod_set.add_component(disp_prof)
    mod_set.add_component(zheight_prof)
    mod_set.add_component(geom)

    gal.model = mod_set

    # Instrument
    import astropy.units as u
    inst = instrument.Instrument()
    inst.pixscale = 0.125 * u.arcsec
    inst.fov = [33, 33]
    beamsize = 0.55 * u.arcsec
    inst.beam = instrument.GaussianBeam(major=beamsize)
    inst.lsf = instrument.LSF(45. * u.km / u.s)
    inst.spec_type = 'velocity'
    inst.spec_start = -1000 * u.km / u.s
    inst.spec_step = 10 * u.km / u.s
    inst.nspec = 201
    inst.ndim = 3
    inst.moment = False
    inst.set_beam_kernel()
    inst.set_lsf_kernel()
    obs.instrument = inst

    gal.add_observation(obs)

    return gal, obs


def _make_test_galaxy_obs_native():
    """Build a test galaxy + observation with oversample=1.

    This produces simulate_cube output at native resolution, matching the
    observed data shape directly (no rebin needed).  Used by Phase 5
    tests that don't test the rebin step.
    """
    gal, obs = _make_test_galaxy_obs()
    obs.mod_options.oversample = 1
    return gal, obs


class TestJAXLossFunction:
    """Test make_jax_loss_function and related utilities."""

    def test_param_storage_names(self):
        """ModelSet.get_param_storage_names returns correct mapping."""
        gal, obs = _make_test_galaxy_obs_native()
        pmap = gal.model.get_param_storage_names()

        assert len(pmap) > 0
        # All values should be non-negative integers
        for key, idx in pmap.items():
            assert isinstance(key, tuple) and len(key) == 2
            assert isinstance(idx, (int, np.integer))
            assert idx >= 0

    def test_identify_traceable_params(self):
        """Geometry params are excluded from traceable set."""
        from dysmalpy.fitting.jax_loss import _identify_traceable_params

        gal, obs = _make_test_galaxy_obs_native()
        reindexed, n_traceable, orig_idx = _identify_traceable_params(gal.model)

        # Geometry params (inc, pa, xshift, yshift) should be excluded
        geom_params = {'inc', 'pa', 'xshift', 'yshift'}
        for (cmp_name, param_name), _ in reindexed:
            if cmp_name == 'geometry_halpha_1D':
                assert param_name not in geom_params

        assert n_traceable > 0
        assert n_traceable == len(reindexed)

    def test_loss_matches_simulate_cube(self):
        """JAX loss function produces same chi-squared as manual computation."""
        from dysmalpy.fitting.jax_loss import make_jax_loss_function

        gal, obs = _make_test_galaxy_obs_native()

        # Generate model cube using simulate_cube directly
        cube_model, _ = gal.model.simulate_cube(obs, gal.dscale)
        cube_model = np.asarray(cube_model)

        # Use the model cube itself as "data" so we expect chi-squared = 0
        noise = np.ones_like(cube_model)
        msk = np.ones(cube_model.shape, dtype=bool)

        dscale = gal.dscale

        jax_loss, get_traceable, set_all = make_jax_loss_function(
            gal.model, obs, dscale, cube_model, noise, mask=msk
        )

        theta = get_traceable()
        loss_val = float(jax_loss(jnp.array(theta)))
        # chi-squared should be ~0 when data = model
        assert abs(loss_val) < 1e-6, \
            "Loss should be ~0 when data = model, got {}".format(loss_val)

    def test_loss_perturbed_params(self):
        """Perturbed parameters produce higher loss than true parameters."""
        from dysmalpy.fitting.jax_loss import make_jax_loss_function

        gal, obs = _make_test_galaxy_obs_native()

        cube_model, _ = gal.model.simulate_cube(obs, gal.dscale)
        cube_model = np.asarray(cube_model)
        noise = np.ones_like(cube_model)
        msk = np.ones(cube_model.shape, dtype=bool)

        dscale = gal.dscale

        jax_loss, get_traceable, set_all = make_jax_loss_function(
            gal.model, obs, dscale, cube_model, noise, mask=msk
        )

        theta_orig = get_traceable()
        loss_orig = float(jax_loss(jnp.array(theta_orig)))

        # Perturb parameters
        theta_pert = theta_orig.copy()
        theta_pert[0] += 0.5  # Shift first parameter significantly
        loss_pert = float(jax_loss(jnp.array(theta_pert)))

        assert loss_pert > loss_orig, \
            "Perturbed loss ({}) should be > original loss ({})".format(
                loss_pert, loss_orig)

    def test_loss_gradients_are_finite(self):
        """jax.grad of the loss function produces finite gradients."""
        from dysmalpy.fitting.jax_loss import make_jax_loss_function

        gal, obs = _make_test_galaxy_obs_native()

        cube_model, _ = gal.model.simulate_cube(obs, gal.dscale)
        cube_model = np.asarray(cube_model)
        noise = np.ones_like(cube_model)
        msk = np.ones(cube_model.shape, dtype=bool)

        dscale = gal.dscale

        jax_loss, get_traceable, set_all = make_jax_loss_function(
            gal.model, obs, dscale, cube_model, noise, mask=msk
        )

        theta = jnp.array(get_traceable())
        # Perturb slightly from exact solution to avoid degenerate gradients
        # (e.g., bt=0 causes NaN gradient for bulge mass)
        theta = theta + 0.05
        loss_fn = lambda th: jax_loss(th)
        grads = jax.grad(loss_fn)(theta)

        assert grads.shape == theta.shape
        assert jnp.all(jnp.isfinite(grads)), \
            "All gradients should be finite, got: {}".format(grads)

    def test_loss_with_noise(self):
        """Adding noise to data increases loss above zero."""
        from dysmalpy.fitting.jax_loss import make_jax_loss_function

        gal, obs = _make_test_galaxy_obs_native()

        cube_model, _ = gal.model.simulate_cube(obs, gal.dscale)
        cube_model = np.asarray(cube_model)

        # Add noise to "data"
        rng = np.random.RandomState(42)
        cube_obs = cube_model + rng.normal(0, 0.001, cube_model.shape)
        noise = np.ones_like(cube_model)
        msk = np.ones(cube_model.shape, dtype=bool)

        dscale = gal.dscale

        jax_loss, get_traceable, set_all = make_jax_loss_function(
            gal.model, obs, dscale, cube_obs, noise, mask=msk
        )

        theta = jnp.array(get_traceable())
        loss_val = float(jax_loss(theta))

        # Loss should be positive but small (noise std = 0.001)
        assert loss_val > 0, "Loss with noise should be > 0, got {}".format(loss_val)


class TestJAXLogProbFunction:
    """Test make_jax_log_prob_function."""

    def test_log_prob_at_true_params(self):
        """Log-prob is finite at true parameters (data = model)."""
        from dysmalpy.fitting.jax_loss import make_jax_log_prob_function

        gal, obs = _make_test_galaxy_obs_native()

        cube_model, _ = gal.model.simulate_cube(obs, gal.dscale)
        cube_model = np.asarray(cube_model)
        noise = np.ones_like(cube_model)
        msk = np.ones(cube_model.shape, dtype=bool)

        dscale = gal.dscale

        log_prob_fn, get_traceable, set_all = make_jax_log_prob_function(
            gal.model, obs, dscale, cube_model, noise, mask=msk
        )

        theta = jnp.array(get_traceable())
        lp = float(log_prob_fn(theta))
        assert np.isfinite(lp), "Log-prob should be finite at true params, got {}".format(lp)

    def test_log_prob_out_of_bounds(self):
        """Log-prob returns -inf for out-of-bounds parameters."""
        from dysmalpy.fitting.jax_loss import make_jax_log_prob_function

        gal, obs = _make_test_galaxy_obs_native()

        cube_model, _ = gal.model.simulate_cube(obs, gal.dscale)
        cube_model = np.asarray(cube_model)
        noise = np.ones_like(cube_model)
        msk = np.ones(cube_model.shape, dtype=bool)

        dscale = gal.dscale

        log_prob_fn, get_traceable, set_all = make_jax_log_prob_function(
            gal.model, obs, dscale, cube_model, noise, mask=msk
        )

        theta = jnp.array(get_traceable())
        # Shift far out of bounds
        theta_bad = theta.at[0].set(1e10)
        lp = float(log_prob_fn(theta_bad))
        assert lp == -np.inf or lp < -1e30, \
            "Log-prob should be -inf for out-of-bounds params, got {}".format(lp)


class TestJAXAdamSmoke:
    """Smoke tests for JAX Adam optimizer."""

    def test_adam_reduces_loss(self):
        """Adam optimizer reduces loss from initial value in a few steps."""
        from dysmalpy.fitting.jax_loss import make_jax_loss_function

        gal, obs = _make_test_galaxy_obs_native()

        cube_model, _ = gal.model.simulate_cube(obs, gal.dscale)
        cube_model = np.asarray(cube_model)
        rng = np.random.RandomState(123)
        cube_obs = cube_model + rng.normal(0, 0.01, cube_model.shape)
        noise = np.ones_like(cube_model)
        msk = np.ones(cube_model.shape, dtype=bool)

        dscale = gal.dscale

        jax_loss, get_traceable, set_all = make_jax_loss_function(
            gal.model, obs, dscale, cube_obs, noise, mask=msk
        )

        loss_grad_fn = jax.jit(jax.value_and_grad(jax_loss))

        theta = jnp.array(get_traceable(), dtype=jnp.float64)
        # Start from slightly perturbed initial values
        theta = theta + 0.1

        loss_init = float(jax_loss(theta))

        # Run a few Adam steps
        beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 1e-2
        m = jnp.zeros_like(theta)
        v = jnp.zeros_like(theta)
        n_steps = 10

        for i in range(n_steps):
            loss_val, grads = loss_grad_fn(theta)
            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * grads ** 2
            m_hat = m / (1 - beta1 ** (i + 1))
            v_hat = v / (1 - beta2 ** (i + 1))
            theta = theta - lr * m_hat / (jnp.sqrt(v_hat) + eps)

        loss_final = float(jax_loss(theta))

        assert loss_final < loss_init, \
            "Loss should decrease after Adam steps: init={}, final={}".format(
                loss_init, loss_final)


# ===================================================================
# 9. Phase 6: JAX FFT convolution tests
# ===================================================================

class TestFFTConvolve3D:
    """Test JAX FFT convolution correctness against scipy."""

    def test_matches_scipy_beam(self):
        """JAX convolution matches scipy for a beam-like kernel (1, ky, kx)."""
        np.random.seed(42)
        cube = np.random.rand(11, 15, 15)
        kernel = np.random.rand(1, 5, 5)
        kernel /= kernel.sum()

        expected = fftconvolve(cube, kernel, mode='same')
        result = np.array(_fft_convolve_3d(jnp.array(cube), jnp.array(kernel)))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_matches_scipy_lsf(self):
        """JAX convolution matches scipy for an LSF-like kernel (nk, 1, 1)."""
        np.random.seed(42)
        cube = np.random.rand(21, 10, 10)
        kernel = np.random.rand(7, 1, 1)
        kernel /= kernel.sum()

        expected = fftconvolve(cube, kernel, mode='same')
        result = np.array(_fft_convolve_3d(jnp.array(cube), jnp.array(kernel)))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_both_kernels(self):
        """Sequential beam+LSF convolution matches sequential scipy."""
        np.random.seed(42)
        cube = np.random.rand(21, 15, 15)
        beam = np.random.rand(1, 5, 5)
        beam /= beam.sum()
        lsf = np.random.rand(7, 1, 1)
        lsf /= lsf.sum()

        # Sequential scipy
        expected = fftconvolve(cube, beam, mode='same')
        expected = fftconvolve(expected, lsf, mode='same')

        # Sequential JAX
        result = np.array(convolve_cube_jax(jnp.array(cube),
                                            beam_kernel=jnp.array(beam),
                                            lsf_kernel=jnp.array(lsf)))

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_output_shape_preserved(self):
        """Output shape equals input shape for various kernel sizes."""
        for shape in [(11, 15, 15), (5, 7, 9), (3, 3, 3)]:
            for kshape in [(1, 3, 3), (5, 1, 1), (3, 5, 7)]:
                cube = np.random.rand(*shape)
                kernel = np.random.rand(*kshape)
                result = _fft_convolve_3d(jnp.array(cube), jnp.array(kernel))
                assert result.shape == shape, \
                    f"shape {shape} with kernel {kshape} gave {result.shape}"

    def test_identity_kernel(self):
        """Single-pixel kernel returns input unchanged."""
        cube = np.random.rand(10, 12, 14)
        kernel = np.ones((1, 1, 1))
        result = np.array(_fft_convolve_3d(jnp.array(cube), jnp.array(kernel)))
        np.testing.assert_allclose(result, cube, rtol=1e-10)

    def test_jit_compiles(self):
        """jax.jit(_fft_convolve_3d) works and matches unjitted."""
        cube = jnp.array(np.random.rand(7, 9, 11))
        kernel = jnp.array(np.random.rand(1, 3, 3))

        result_unjitted = _fft_convolve_3d(cube, kernel)
        result_jitted = jax.jit(_fft_convolve_3d)(cube, kernel)
        np.testing.assert_allclose(
            np.array(result_jitted), np.array(result_unjitted), rtol=1e-10)

    def test_gradient(self):
        """jax.grad through convolution produces finite values."""
        cube = jnp.array(np.random.rand(5, 7, 9))
        kernel = jnp.array(np.random.rand(1, 3, 3))

        def fn(c):
            return jnp.sum(_fft_convolve_3d(c, kernel))

        grad_fn = jax.grad(fn)
        grads = grad_fn(cube)
        assert jnp.all(jnp.isfinite(grads)), \
            "Gradients through convolution should be finite"


class TestJAXLossWithConvolution:
    """Integration tests for loss function with convolution."""

    def test_loss_convolved_near_zero(self):
        """Model cube rebin+convolved to create obs, loss near zero at true params."""
        from dysmalpy.fitting.jax_loss import make_jax_loss_function

        gal, obs = _make_test_galaxy_obs()

        # Generate model cube (oversampled)
        cube_model, _ = gal.model.simulate_cube(obs, gal.dscale)
        cube_model = np.asarray(cube_model)

        # Rebin then convolve to create "observed" data at native resolution
        oversample = obs.mod_options.oversample
        oversize = obs.mod_options.oversize
        nx_sky = obs.instrument.fov[0]
        ny_sky = obs.instrument.fov[1]

        if oversample > 1:
            cube_rebinned = np.asarray(_rebin_spatial(
                jnp.array(cube_model), ny_sky * oversize, nx_sky * oversize))
        else:
            cube_rebinned = cube_model

        beam_np, lsf_np = get_jax_kernels(obs.instrument)
        cube_obs = np.asarray(convolve_cube_jax(
            jnp.array(cube_rebinned),
            beam_kernel=jnp.array(beam_np) if beam_np is not None else None,
            lsf_kernel=jnp.array(lsf_np) if lsf_np is not None else None,
        ))

        # Crop if oversize > 1
        if oversize > 1:
            y_start = (cube_obs.shape[1] - ny_sky) // 2
            x_start = (cube_obs.shape[2] - nx_sky) // 2
            cube_obs = cube_obs[:, y_start:y_start+ny_sky,
                                 x_start:x_start+nx_sky]

        noise = np.ones_like(cube_obs)
        msk = np.ones(cube_obs.shape, dtype=bool)

        dscale = gal.dscale

        jax_loss, get_traceable, set_all = make_jax_loss_function(
            gal.model, obs, dscale, cube_obs, noise, mask=msk, convolve=True
        )

        theta = jnp.array(get_traceable())
        loss_val = float(jax_loss(theta))
        assert abs(loss_val) < 1e-3, \
            "Convolved loss should be ~0 at true params, got {}".format(loss_val)

    def test_loss_convolved_vs_unconvolved(self):
        """Convolved loss differs from unconvolved loss."""
        from dysmalpy.fitting.jax_loss import make_jax_loss_function

        gal, obs = _make_test_galaxy_obs()

        cube_model, _ = gal.model.simulate_cube(obs, gal.dscale)
        cube_model = np.asarray(cube_model)

        # Rebin to native resolution for fake obs (matching what the loss
        # function does internally)
        oversample = obs.mod_options.oversample
        oversize = obs.mod_options.oversize
        nx_sky = obs.instrument.fov[0]
        ny_sky = obs.instrument.fov[1]
        if oversample > 1:
            cube_native = np.asarray(_rebin_spatial(
                jnp.array(cube_model), ny_sky * oversize, nx_sky * oversize))
        else:
            cube_native = cube_model

        noise = np.ones_like(cube_native)
        msk = np.ones(cube_native.shape, dtype=bool)
        dscale = gal.dscale

        loss_conv, _, _ = make_jax_loss_function(
            gal.model, obs, dscale, cube_native, noise, mask=msk, convolve=True
        )
        loss_unconv, _, _ = make_jax_loss_function(
            gal.model, obs, dscale, cube_native, noise, mask=msk, convolve=False
        )

        # Use the traceable extractor
        from dysmalpy.fitting.jax_loss import _identify_traceable_params
        reindexed, _, orig_idx = _identify_traceable_params(gal.model)
        theta_traceable = jnp.array([gal.model.get_free_parameters_values()[i]
                                     for i in orig_idx])

        loss_c = float(loss_conv(theta_traceable))
        loss_u = float(loss_unconv(theta_traceable))

        # They should differ because convolution changes the model cube
        assert abs(loss_c - loss_u) > 1e-6, \
            "Convolved ({}) and unconvolved ({}) losses should differ".format(
                loss_c, loss_u)

    def test_loss_gradient_finite_with_convolution(self):
        """Full pipeline gradients are finite with convolution enabled."""
        from dysmalpy.fitting.jax_loss import make_jax_loss_function

        gal, obs = _make_test_galaxy_obs()

        cube_model, _ = gal.model.simulate_cube(obs, gal.dscale)
        cube_model = np.asarray(cube_model)

        # Rebin to native resolution for fake obs
        oversample = obs.mod_options.oversample
        oversize = obs.mod_options.oversize
        nx_sky = obs.instrument.fov[0]
        ny_sky = obs.instrument.fov[1]
        if oversample > 1:
            cube_native = np.asarray(_rebin_spatial(
                jnp.array(cube_model), ny_sky * oversize, nx_sky * oversize))
        else:
            cube_native = cube_model

        noise = np.ones_like(cube_native)
        msk = np.ones(cube_native.shape, dtype=bool)
        dscale = gal.dscale

        jax_loss, get_traceable, set_all = make_jax_loss_function(
            gal.model, obs, dscale, cube_native, noise, mask=msk, convolve=True
        )

        theta = jnp.array(get_traceable())
        theta = theta + 0.05  # Slight perturbation to avoid degenerate gradients

        grads = jax.grad(lambda th: jax_loss(th))(theta)
        assert jnp.all(jnp.isfinite(grads)), \
            "All gradients should be finite with convolution, got: {}".format(grads)

    def test_adam_reduces_loss_convolved(self):
        """Adam steps reduce convolved loss."""
        from dysmalpy.fitting.jax_loss import make_jax_loss_function

        gal, obs = _make_test_galaxy_obs()

        cube_model, _ = gal.model.simulate_cube(obs, gal.dscale)
        cube_model = np.asarray(cube_model)

        # Rebin then convolve to create observed data
        oversample = obs.mod_options.oversample
        oversize = obs.mod_options.oversize
        nx_sky = obs.instrument.fov[0]
        ny_sky = obs.instrument.fov[1]

        if oversample > 1:
            cube_rebinned = np.asarray(_rebin_spatial(
                jnp.array(cube_model), ny_sky * oversize, nx_sky * oversize))
        else:
            cube_rebinned = cube_model

        beam_np, lsf_np = get_jax_kernels(obs.instrument)
        cube_obs = np.asarray(convolve_cube_jax(
            jnp.array(cube_rebinned),
            beam_kernel=jnp.array(beam_np) if beam_np is not None else None,
            lsf_kernel=jnp.array(lsf_np) if lsf_np is not None else None,
        ))

        if oversize > 1:
            y_start = (cube_obs.shape[1] - ny_sky) // 2
            x_start = (cube_obs.shape[2] - nx_sky) // 2
            cube_obs = cube_obs[:, y_start:y_start+ny_sky,
                                 x_start:x_start+nx_sky]

        rng = np.random.RandomState(123)
        cube_obs = cube_obs + rng.normal(0, 0.01, cube_obs.shape)
        noise = np.ones_like(cube_obs)
        msk = np.ones(cube_obs.shape, dtype=bool)
        dscale = gal.dscale

        jax_loss, get_traceable, set_all = make_jax_loss_function(
            gal.model, obs, dscale, cube_obs, noise, mask=msk, convolve=True
        )

        loss_grad_fn = jax.jit(jax.value_and_grad(jax_loss))

        theta = jnp.array(get_traceable(), dtype=jnp.float64)
        theta = theta + 0.1  # Perturb from true values

        loss_init = float(jax_loss(theta))

        beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 1e-2
        m = jnp.zeros_like(theta)
        v = jnp.zeros_like(theta)

        for i in range(10):
            loss_val, grads = loss_grad_fn(theta)
            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * grads ** 2
            m_hat = m / (1 - beta1 ** (i + 1))
            v_hat = v / (1 - beta2 ** (i + 1))
            theta = theta - lr * m_hat / (jnp.sqrt(v_hat) + eps)

        loss_final = float(jax_loss(theta))

        assert loss_final < loss_init, \
            "Convolved loss should decrease: init={}, final={}".format(
                loss_init, loss_final)


class TestGetJAXKernels:
    """Test kernel extraction from Instrument instances."""

    def test_beam_kernel_shape(self):
        """Extracted beam kernel has shape (1, ky, kx) and sums to ~1."""
        gal, obs = _make_test_galaxy_obs_native()

        beam_np, lsf_np = get_jax_kernels(obs.instrument)

        assert beam_np is not None
        assert beam_np.shape[0] == 1  # spectral dimension
        assert beam_np.ndim == 3
        assert np.isclose(np.sum(beam_np), 1.0, atol=1e-5)

    def test_lsf_kernel_shape(self):
        """Extracted LSF kernel has shape (nk, 1, 1) and sums to ~1."""
        gal, obs = _make_test_galaxy_obs_native()

        beam_np, lsf_np = get_jax_kernels(obs.instrument)

        assert lsf_np is not None
        assert lsf_np.shape[1] == 1  # y spatial
        assert lsf_np.shape[2] == 1  # x spatial
        assert lsf_np.ndim == 3
        # Discretized Gaussian may sum slightly less than 1
        assert np.isclose(np.sum(lsf_np), 1.0, atol=1e-3)

    def test_no_kernels(self):
        """Instrument with no beam/LSF returns (None, None)."""
        from dysmalpy import instrument

        inst = instrument.Instrument()
        beam_np, lsf_np = get_jax_kernels(inst)

        assert beam_np is None
        assert lsf_np is None


# ===================================================================
# 10. Phase 7: JAX rebin tests
# ===================================================================

class TestRebinSpatial:
    """Test JAX spatial rebin against numpy reference."""

    def test_rebin_matches_numpy(self):
        """_rebin_spatial matches dysmalpy.utils.rebin to machine precision."""
        from dysmalpy.utils import rebin as np_rebin

        np.random.seed(42)
        cube = np.random.rand(11, 15, 15)
        new_ny, new_nx = 5, 5

        expected = np_rebin(cube, (new_ny, new_nx))
        result = np.array(_rebin_spatial(jnp.array(cube), new_ny, new_nx))

        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_rebin_matches_numpy_nonsquare(self):
        """Rebin with non-square target matches numpy."""
        from dysmalpy.utils import rebin as np_rebin

        np.random.seed(99)
        cube = np.random.rand(7, 12, 18)
        new_ny, new_nx = 4, 6

        expected = np_rebin(cube, (new_ny, new_nx))
        result = np.array(_rebin_spatial(jnp.array(cube), new_ny, new_nx))

        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_rebin_jit_compiles(self):
        """jax.jit(_rebin_spatial) works and matches unjitted."""
        np.random.seed(7)
        cube = jnp.array(np.random.rand(5, 9, 12))

        result_unjitted = _rebin_spatial(cube, 3, 4)
        result_jitted = jax.jit(lambda c: _rebin_spatial(c, 3, 4))(cube)
        np.testing.assert_allclose(
            np.array(result_jitted), np.array(result_unjitted), rtol=1e-12)

    def test_rebin_identity(self):
        """oversample=1 (new_ny=ny_in, new_nx=nx_in) returns same cube."""
        np.random.seed(3)
        cube = np.random.rand(5, 8, 8)
        result = np.array(_rebin_spatial(jnp.array(cube), 8, 8))
        np.testing.assert_allclose(result, cube, rtol=1e-12)

    def test_rebin_gradient(self):
        """jax.grad through rebin produces finite values."""
        cube = jnp.array(np.random.rand(3, 6, 6))

        def fn(c):
            return jnp.sum(_rebin_spatial(c, 2, 2))

        grads = jax.grad(fn)(cube)
        assert jnp.all(jnp.isfinite(grads)), \
            "Gradients through rebin should be finite"

    def test_rebin_output_shape(self):
        """Output shape is (nz, new_ny, new_nx)."""
        cube = jnp.array(np.random.rand(10, 15, 21))
        result = _rebin_spatial(cube, 5, 7)
        assert result.shape == (10, 5, 7)


class TestFullPipelineRebinConvolveCrop:
    """Test the full simulate -> rebin -> convolve -> crop pipeline."""

    def test_rebin_convolve_matches_numpy_pipeline(self):
        """Rebin+convolve+crop in JAX matches the numpy observation pipeline."""
        from dysmalpy.utils import rebin as np_rebin

        gal, obs = _make_test_galaxy_obs()
        dscale = gal.dscale

        # Simulate at oversampled resolution
        cube_model, _ = gal.model.simulate_cube(obs, dscale)
        cube_model = np.asarray(cube_model)

        oversample = obs.mod_options.oversample
        oversize = obs.mod_options.oversize
        nx_sky = obs.instrument.fov[0]
        ny_sky = obs.instrument.fov[1]

        # --- Numpy pipeline ---
        if oversample > 1:
            cube_rebinned_np = np_rebin(cube_model, (ny_sky * oversize,
                                                      nx_sky * oversize))
        else:
            cube_rebinned_np = cube_model

        # Convolve using scipy (via instrument.convolve)
        convolved_np = obs.instrument.convolve(cube=cube_rebinned_np,
                                               spec_center=obs.instrument.line_center)

        if oversize > 1:
            ny_os = convolved_np.shape[1]
            nx_os = convolved_np.shape[2]
            y_start = int(ny_os / 2 - ny_sky / 2)
            y_end = int(ny_os / 2 + ny_sky / 2)
            x_start = int(nx_os / 2 - nx_sky / 2)
            x_end = int(nx_os / 2 + nx_sky / 2)
            final_np = convolved_np[:, y_start:y_end, x_start:x_end]
        else:
            final_np = convolved_np

        # --- JAX pipeline ---
        cube_jax = jnp.array(cube_model)
        if oversample > 1:
            cube_jax = _rebin_spatial(cube_jax, ny_sky * oversize,
                                       nx_sky * oversize)

        beam_np, lsf_np = get_jax_kernels(obs.instrument)
        cube_jax = convolve_cube_jax(
            cube_jax,
            beam_kernel=jnp.array(beam_np) if beam_np is not None else None,
            lsf_kernel=jnp.array(lsf_np) if lsf_np is not None else None,
        )

        if oversize > 1:
            ny_os_j = cube_jax.shape[1]
            nx_os_j = cube_jax.shape[2]
            y_s = (ny_os_j - ny_sky) // 2
            x_s = (nx_os_j - nx_sky) // 2
            cube_jax = cube_jax[:, y_s:y_s+ny_sky, x_s:x_s+nx_sky]

        final_jax = np.array(cube_jax)

        # Compare — numpy uses scipy fftconvolve, JAX uses our own.
        # Allow small rtol for floating-point differences.
        np.testing.assert_allclose(final_jax, final_np, rtol=1e-8, atol=1e-10)

    def test_loss_near_zero_with_rebin(self):
        """Full pipeline (rebin+convolve) gives near-zero loss at true params."""
        from dysmalpy.fitting.jax_loss import make_jax_loss_function

        gal, obs = _make_test_galaxy_obs()
        dscale = gal.dscale

        # Simulate, rebin, convolve to create fake obs at native resolution
        cube_model, _ = gal.model.simulate_cube(obs, dscale)
        cube_model = np.asarray(cube_model)

        oversample = obs.mod_options.oversample
        oversize = obs.mod_options.oversize
        nx_sky = obs.instrument.fov[0]
        ny_sky = obs.instrument.fov[1]

        if oversample > 1:
            cube_rebinned = np.asarray(_rebin_spatial(
                jnp.array(cube_model), ny_sky * oversize, nx_sky * oversize))
        else:
            cube_rebinned = cube_model

        beam_np, lsf_np = get_jax_kernels(obs.instrument)
        cube_obs = np.asarray(convolve_cube_jax(
            jnp.array(cube_rebinned),
            beam_kernel=jnp.array(beam_np) if beam_np is not None else None,
            lsf_kernel=jnp.array(lsf_np) if lsf_np is not None else None,
        ))

        if oversize > 1:
            y_s = (cube_obs.shape[1] - ny_sky) // 2
            x_s = (cube_obs.shape[2] - nx_sky) // 2
            cube_obs = cube_obs[:, y_s:y_s+ny_sky, x_s:x_s+nx_sky]

        noise = np.ones_like(cube_obs)
        msk = np.ones(cube_obs.shape, dtype=bool)

        jax_loss, get_traceable, set_all = make_jax_loss_function(
            gal.model, obs, dscale, cube_obs, noise, mask=msk, convolve=True
        )

        theta = jnp.array(get_traceable())
        loss_val = float(jax_loss(theta))
        assert abs(loss_val) < 1e-3, \
            "Full pipeline loss should be ~0 at true params, got {}".format(
                loss_val)

    def test_full_pipeline_gradient_finite(self):
        """jax.grad through simulate+rebin+convolve+chi2 produces finite values."""
        from dysmalpy.fitting.jax_loss import make_jax_loss_function

        gal, obs = _make_test_galaxy_obs()
        dscale = gal.dscale

        cube_model, _ = gal.model.simulate_cube(obs, dscale)
        cube_model = np.asarray(cube_model)

        oversample = obs.mod_options.oversample
        oversize = obs.mod_options.oversize
        nx_sky = obs.instrument.fov[0]
        ny_sky = obs.instrument.fov[1]

        if oversample > 1:
            cube_native = np.asarray(_rebin_spatial(
                jnp.array(cube_model), ny_sky * oversize, nx_sky * oversize))
        else:
            cube_native = cube_model

        noise = np.ones_like(cube_native)
        msk = np.ones(cube_native.shape, dtype=bool)

        jax_loss, get_traceable, set_all = make_jax_loss_function(
            gal.model, obs, dscale, cube_native, noise, mask=msk, convolve=True
        )

        theta = jnp.array(get_traceable()) + 0.05
        grads = jax.grad(lambda th: jax_loss(th))(theta)
        assert jnp.all(jnp.isfinite(grads)), \
            "Full pipeline gradients should be finite, got: {}".format(grads)
