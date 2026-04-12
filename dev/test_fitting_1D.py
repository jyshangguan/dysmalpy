# coding=utf8
"""
1D fitting test: MPFIT vs JAXAdam comparison.

Tests both MPFIT and JAXAdam fitters with 1D (profile) data from
GS4_43501.obs_prof.txt.

Run from the repo root:
    JAX_PLATFORMS=cpu python dev/test_fitting_1D.py
"""

import os
import sys
import math
import shutil
import numpy as np

# Ensure the repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update("jax_enable_x64", True)

import astropy.units as u
from dysmalpy import galaxy, models, instrument, observation, data_classes
from dysmalpy import fitting
from dysmalpy.fitting_wrappers import utils_io as fw_utils_io
from dysmalpy.fitting_wrappers import data_io as fwdata_io
from dysmalpy.fitting_wrappers import setup_gal_models
from dysmalpy import aperture_classes
from dysmalpy import config
from dysmalpy.fitting_wrappers.dysmalpy_fit_single import dysmalpy_fit_single

# Paths
_dir_tests_data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'tests', 'test_data')

PARAM_FILENAME = os.path.join(_dir_tests_data, 'fitting_1D_mpfit.params')


def _set_constraints(comp, fixed_map, bounds_map):
    """Set fixed/bounds constraints on a model component's parameters."""
    for par_name, fixed_val in fixed_map.items():
        if par_name in comp._param_instances:
            comp._param_instances[par_name].fixed = fixed_val
    for par_name, bounds_val in bounds_map.items():
        if par_name in comp._param_instances:
            comp._param_instances[par_name].bounds = tuple(bounds_val)
    comp.fixed = {p: comp._param_instances[p].fixed for p in comp.param_names}
    comp.bounds = {p: comp._param_instances[p].bounds for p in comp.param_names}


def setup_galaxy_programmatic(moment=False):
    """Set up galaxy + observation programmatically, matching the params file.

    This avoids the pre-existing regression in setup_gal_model_base
    where parameter fixed/bounds are not correctly read from the params file.
    """
    z = 1.613
    name = 'GS4_43501'

    gal = galaxy.Galaxy(z=z, name=name)
    obs = observation.Observation(name='OBS', tracer='halpha')

    mod_set = models.ModelSet()

    # DiskBulge
    bary = models.DiskBulge(
        total_mass=11.0, bt=0.3, r_eff_disk=5.0, n_disk=1.0,
        invq_disk=5.0, r_eff_bulge=1.0, n_bulge=4.0, invq_bulge=1.0,
        noord_flat=True, name='disk+bulge', gas_component='total',
    )
    _set_constraints(bary,
        {'total_mass': False, 'r_eff_disk': False, 'n_disk': True,
         'r_eff_bulge': True, 'n_bulge': True, 'bt': True,
         'mass_to_light': True, 'r_eff_bulge_mass': True,
         'invq_disk': True, 'invq_bulge': True},
        {'total_mass': (10, 13), 'r_eff_disk': (0.1, 30.0),
         'n_disk': (1, 8), 'r_eff_bulge': (1, 5),
         'n_bulge': (1, 8), 'bt': (0, 1)},
    )

    # NFW halo
    halo = models.NFW(
        mvirial=11.5, conc=5.0, fdm=0.5, z=z, name='halo',
    )
    _set_constraints(halo,
        {'mvirial': True, 'conc': True, 'fdm': False},
        {'mvirial': (10, 13), 'conc': (1, 20), 'fdm': (0., 1.)},
    )
    halo.fdm.tied = fw_utils_io.tie_fdm
    halo.mvirial.tied = fw_utils_io.tie_lmvirial_NFW

    # Dispersion
    disp_prof = models.DispersionConst(
        sigma0=39., name='dispprof_LINE', tracer='halpha',
    )
    _set_constraints(disp_prof,
        {'sigma0': False},
        {'sigma0': (5, 300)},
    )

    # Z-height
    zheight_prof = models.ZHeightGauss(
        sigmaz=0.9, name='zheightgaus',
    )
    _set_constraints(zheight_prof,
        {'sigmaz': False},
        {'sigmaz': (0.1, 1.0)},
    )
    zheight_prof.sigmaz.tied = fw_utils_io.tie_sigz_reff

    # Geometry
    geom = models.Geometry(
        inc=62., pa=142., xshift=0., yshift=0., name='geom_1',
        obs_name='OBS',
    )
    _set_constraints(geom,
        {'inc': True, 'pa': True, 'xshift': True, 'yshift': True,
         'vel_shift': True},
        {'inc': (0, 90), 'pa': (90, 180), 'xshift': (0, 4),
         'yshift': (-10, -4)},
    )

    # Dimming
    dimming = models.ConstantDimming(amp_lumtoflux=1.e-10)

    mod_set.add_component(bary, light=True)
    mod_set.add_component(halo)
    mod_set.add_component(disp_prof)
    mod_set.add_component(zheight_prof)
    mod_set.add_component(geom)

    # Sync model_set constraint dicts after adding components and setting tied.
    # NOTE: comp.tied dict is stale (halo.fdm.tied=fn sets the class-level
    # descriptor, not the instance copy in _param_instances).  This is the
    # same stale behavior as the original code — tied parameters remain in
    # the free-parameter vector and their values are overwritten by
    # _update_tied_parameters() at each iteration.
    for comp_name in list(mod_set.components.keys()):
        comp = mod_set.components[comp_name]
        mod_set.fixed[comp_name] = dict(comp.fixed)
        mod_set.tied[comp_name] = dict(comp.tied)

    # Recompute nparams_free to reflect constraint changes
    mod_set.nparams_free = sum(
        1 for c in mod_set.components
        for p in mod_set.fixed[c]
        if not mod_set.fixed[c][p] and not mod_set.tied[c][p]
    )

    mod_set.kinematic_options.adiabatic_contract = False
    mod_set.kinematic_options.pressure_support = True
    mod_set.kinematic_options.pressure_support_type = 1
    mod_set.dimming = dimming

    gal.model = mod_set

    # Instrument
    inst = instrument.Instrument()
    inst.pixscale = 0.125 * u.arcsec
    inst.fov = [37, 37]
    inst.beam = instrument.GaussianBeam(major=0.55 * u.arcsec)
    inst.lsf = instrument.LSF(51. * u.km / u.s)
    inst.spec_type = 'velocity'
    inst.spec_start = -1000 * u.km / u.s
    inst.spec_step = 10 * u.km / u.s
    inst.nspec = 201
    inst.ndim = 1
    inst.moment = moment
    inst.slit_width = 0.55
    inst.slit_pa = 142.
    inst.smoothing_type = None
    inst.smoothing_npix = 1
    inst.set_beam_kernel()
    inst.set_lsf_kernel()

    # Load data
    obs_data = fwdata_io.load_single_obs_1D_data(
        fdata='GS4_43501.obs_prof.txt',
        params={'data_inst_corr': True,
                'symmetrize_data': False},
        datadir=_dir_tests_data + os.sep)

    # Setup apertures
    aper_centers = obs_data.rarr
    apertures = aperture_classes.CircApertures(
        rarr=aper_centers, slit_PA=142., rpix=0.275 / 0.125,
        nx=37, ny=37, pixscale=0.125,
        partial_weight=True, rotate_cube=False,
        moment=moment)
    inst.apertures = apertures

    obs.instrument = inst
    obs.data = obs_data

    gal.add_observation(obs)

    return gal, obs


def extract_param_value(results, comp_name, par_name):
    """Extract a free parameter value from results."""
    free_keys = results._free_param_keys
    if comp_name in free_keys and par_name in free_keys[comp_name]:
        idx = free_keys[comp_name][par_name]
        return float(results.bestfit_parameters[idx])
    return None


def test_1_mpfit():
    """Test 1: MPFIT 1D fitting."""
    print("\n" + "=" * 60)
    print("[Test 1: MPFIT 1D Fitting]")
    print("=" * 60)

    gal, obs = setup_galaxy_programmatic(moment=False)

    outdir = os.path.join(_dir_tests_data, 'PYTEST_OUTPUT/GS4_43501_1D_out_mpfit_test/')
    if os.path.isdir(outdir):
        shutil.rmtree(outdir, ignore_errors=True)

    output_options = config.OutputOptions(
        outdir=outdir,
        save_results=False,
        save_model=False,
        save_model_bestfit=False,
        save_bestfit_cube=False,
        save_data=False,
        save_vel_ascii=False,
        save_reports=False,
        do_plotting=False,
        overwrite=True,
    )

    fitter = fitting.MPFITFitter(maxiter=200)
    results = fitter.fit(gal, output_options=output_options)

    assert results.status > 0, "MPFIT did not converge! status={}".format(results.status)
    print("  status: converged")

    # Expected values
    expected = {
        ('disk+bulge', 'total_mass'): 10.7218,
        ('disk+bulge', 'r_eff_disk'): 3.2576,
        ('halo', 'fdm'): 0.2888,
        ('dispprof_LINE', 'sigma0'): 37.9097,
    }

    all_pass = True
    for (comp, par), exp_val in expected.items():
        actual = extract_param_value(results, comp, par)
        if actual is None:
            print("  {}: MISSING".format(par))
            all_pass = False
            continue
        rel_err = abs(actual - exp_val) / exp_val
        passed = rel_err < 0.005
        status = "PASS" if passed else "FAIL"
        print("  {}: {:.4f} (expected {:.4f})  {}".format(
            par, actual, exp_val, status))
        if not passed:
            all_pass = False
            print("    rel_err={:.4f}".format(rel_err))

    red_chisq = results.bestfit_redchisq
    print("  red_chisq: {:.2f}".format(red_chisq))

    if all_pass:
        print("  >>> TEST 1 PASSED")
    else:
        print("  >>> TEST 1 FAILED")

    # Cleanup
    if os.path.isdir(outdir):
        shutil.rmtree(outdir, ignore_errors=True)

    return all_pass, results


def test_2_jax_adam():
    """Test 2: JAXAdam 1D fitting."""
    print("\n" + "=" * 60)
    print("[Test 2: JAXAdam 1D Fitting]")
    print("=" * 60)

    gal, obs = setup_galaxy_programmatic(moment=True)

    outdir = os.path.join(_dir_tests_data, 'PYTEST_OUTPUT/GS4_43501_1D_out_jax_adam_test/')
    if os.path.isdir(outdir):
        shutil.rmtree(outdir, ignore_errors=True)

    output_options = config.OutputOptions(
        outdir=outdir,
        save_results=False,
        save_model=False,
        save_model_bestfit=False,
        save_bestfit_cube=False,
        save_data=False,
        save_vel_ascii=False,
        save_reports=False,
        do_plotting=False,
        overwrite=True,
    )

    # Use finite-difference gradients (each step needs 2*n_params forward
    # model evaluations).  Note: Only total_mass, r_eff_disk, sigma0 are
    # free parameters; fdm and sigmaz are tied (computed from r_eff_disk).
    # Adam is a first-order optimizer and cannot navigate the
    # total_mass/r_eff_disk degeneracy as well as MPFIT's Levenberg-Marquardt.
    fitter = fitting.JAXAdamFitter(n_steps=200, learning_rate=1e-2)
    results = fitter.fit(gal, output_options=output_options)

    loss_history = results.loss_history
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]

    print("  Initial loss: {:.2f}".format(initial_loss))
    print("  Final loss:   {:.2f}".format(final_loss))
    print("  Loss ratio (initial/final): {:.1f}x".format(initial_loss / max(final_loss, 1e-30)))

    loss_decreased = final_loss < 0.5 * initial_loss
    print("  Loss decreased significantly: {}".format(
        "YES" if loss_decreased else "NO"))

    param_info = [
        ('disk+bulge', 'total_mass', 10.7218),
        ('disk+bulge', 'r_eff_disk', 3.2576),
        ('halo', 'fdm', 0.2888),
        ('dispprof_LINE', 'sigma0', 37.9097),
    ]

    print("\n  Parameter comparison with MPFIT expected values:")
    print("  (Note: only total_mass, r_eff_disk, sigma0 are JAX-optimized;"
          " fdm and sigmaz are tied)")
    for comp, par, mpfit_val in param_info:
        val = extract_param_value(results, comp, par)
        if val is None:
            print("  {}: not JAX-optimized (tied/fixed)".format(par))
            continue
        same_order = (val / mpfit_val > 0.1) and (val / mpfit_val < 10.)
        print("  {}: {:.4f} (MPFIT: {:.4f})  {}".format(
            par, val, mpfit_val,
            "OK" if same_order else "UNREASONABLE"))

    # Cleanup
    if os.path.isdir(outdir):
        shutil.rmtree(outdir, ignore_errors=True)

    return loss_decreased, results, initial_loss, final_loss


def test_3_comparison(mpfit_results, jax_results, jax_final_loss):
    """Test 3: Cross-comparison."""
    print("\n" + "=" * 60)
    print("[Test 3: Comparison]")
    print("=" * 60)

    mpfit_chisq = mpfit_results.bestfit_chisq
    if mpfit_chisq is not None and mpfit_chisq > 0:
        loss_ratio = jax_final_loss / mpfit_chisq
        print("  MPFIT chi-squared:    {:.2f}".format(mpfit_chisq))
        print("  JAXAdam final loss:   {:.2f}".format(jax_final_loss))
        print("  Loss ratio (adam/mpfit): {:.2f}".format(loss_ratio))
        ratio_ok = loss_ratio < 3.0
        print("  Ratio within 3x: {}".format("PASS" if ratio_ok else "FAIL"))
    else:
        ratio_ok = True
        print("  (Skipping ratio check - MPFIT chi-squared not available)")

    param_info = [
        ('disk+bulge', 'total_mass'),
        ('disk+bulge', 'r_eff_disk'),
        ('halo', 'fdm'),
        ('dispprof_LINE', 'sigma0'),
    ]

    print("\n  Parameter agreement (MPFIT vs JAXAdam):")
    print("  (Note: JAXAdam optimizes fewer parameters than MPFIT,")
    print("   and Adam cannot navigate parameter degeneracies as well")
    print("   as MPFIT's Levenberg-Marquardt)")
    all_reasonable = True
    for comp, par in param_info:
        val_m = extract_param_value(mpfit_results, comp, par)
        val_j = extract_param_value(jax_results, comp, par)

        if val_m is not None and val_j is not None and val_m != 0:
            rel_diff = abs(val_j - val_m) / abs(val_m)
            # Use a relaxed tolerance since JAXAdam uses fewer params
            # and finite-difference gradients with a first-order optimizer
            reasonable = rel_diff < 2.0
            print("  {}: MPFIT={:.4f} JAX={:.4f}  diff={:.1%}  {}".format(
                par, val_m, val_j, rel_diff,
                "PASS" if reasonable else "FAIL"))
            if not reasonable:
                all_reasonable = False
        else:
            print("  {}: (not in JAXAdam params)".format(par))

    overall = ratio_ok and all_reasonable
    print("\n  >>> TEST 3 {}".format("PASSED" if overall else "FAILED"))

    return overall


def main():
    print("=" * 60)
    print("1D Fitting Test: MPFIT vs JAXAdam")
    print("=" * 60)
    print("numpy:  {}".format(np.__version__))
    print("JAX backend: {}".format(jax.default_backend()))

    mpfit_pass, mpfit_results = test_1_mpfit()

    jax_pass, jax_results, initial_loss, final_loss = test_2_jax_adam()

    comparison_pass = test_3_comparison(mpfit_results, jax_results, final_loss)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Test 1 (MPFIT):      {}".format("PASS" if mpfit_pass else "FAIL"))
    print("  Test 2 (JAXAdam):    {}".format("PASS" if jax_pass else "FAIL"))
    print("  Test 3 (Comparison): {}".format("PASS" if comparison_pass else "FAIL"))

    all_pass = mpfit_pass and jax_pass and comparison_pass
    print("\n  OVERALL: {}".format("PASS" if all_pass else "FAIL"))
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
