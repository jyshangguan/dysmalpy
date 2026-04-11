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
import jax
import jax.numpy as jnp
import scipy.special as scp_spec

from dysmalpy.special import gammaincinv, hyp2f1, bessel_k0, bessel_k1
from dysmalpy import parameters, models
from dysmalpy.models.cube_processing import (
    populate_cube_jax, populate_cube_jax_ais,
    _simulate_cube_inner_direct, _simulate_cube_inner_ais,
    _make_cube_ai, _get_xyz_sky_gal, _get_xyz_sky_gal_inverse,
    _calculate_max_skyframe_extents,
)


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
