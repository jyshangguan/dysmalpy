# coding=utf8
"""
Model equivalence comparison: JAX pipeline self-consistency check.

Reproduces the same galaxy model as test_models.py::test_simulate_cube,
runs the full observation pipeline (simulate -> rebin -> convolve -> crop),
and verifies:
  1. JAX determinism (two runs produce identical results)
  2. Pipeline shape correctness at each stage
  3. Chi-squared is zero when model cube is compared to itself
  4. Fresh reference pixel values for updating test_models.py

Run from the repo root:
    JAX_PLATFORMS=cpu python dev/compare_pipelines.py       # CPU
    python dev/compare_pipelines.py                           # GPU (if available)
"""

import sys
import os
import math
import numpy as np

# Ensure the repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import astropy.units as u
from dysmalpy import galaxy, models, parameters, instrument, observation
from dysmalpy.fitting_wrappers import utils_io as fw_utils_io
from dysmalpy.utils import rebin

# Enable JAX float64 for numerical parity with numpy/scipy
import jax
jax.config.update("jax_enable_x64", True)


def setup_fullmodel():
    """Identical to test_models.py::HelperSetups.setup_fullmodel."""
    z = 1.613
    name = 'GS4_43501'

    gal = galaxy.Galaxy(z=z, name=name)
    obs = observation.Observation(name='halpha_1D', tracer='halpha')
    obs.mod_options.oversample = 3
    obs.mod_options.zcalc_truncate = True

    mod_set = models.ModelSet()

    # DiskBulge
    bary = models.DiskBulge(
        total_mass=11.0, bt=0.3, r_eff_disk=5.0, n_disk=1.0,
        invq_disk=5.0, r_eff_bulge=1.0, n_bulge=4.0, invq_bulge=1.0,
        noord_flat=True, name='disk+bulge', gas_component='total',
        fixed={'total_mass': False, 'r_eff_disk': False, 'n_disk': True,
               'r_eff_bulge': True, 'n_bulge': True, 'bt': False},
        bounds={'total_mass': (10, 13), 'r_eff_disk': (1.0, 30.0),
                'n_disk': (1, 8), 'r_eff_bulge': (1, 5),
                'n_bulge': (1, 8), 'bt': (0, 1)},
    )

    # NFW halo
    halo = models.NFW(
        mvirial=12.0, conc=5.0, fdm=0.5, z=z, name='halo',
        fixed={'mvirial': False, 'conc': True, 'fdm': False},
        bounds={'mvirial': (10, 13), 'conc': (1, 20), 'fdm': (0., 1.)},
    )
    halo.fdm.tied = fw_utils_io.tie_fdm

    # Dispersion
    disp_prof = models.DispersionConst(
        sigma0=39., name='dispprof', tracer='halpha',
        fixed={'sigma0': False}, bounds={'sigma0': (10, 200)},
    )

    # Z-height
    zheight_prof = models.ZHeightGauss(
        sigmaz=0.9, name='zheightgaus',
        fixed={'sigmaz': False},
    )
    zheight_prof.sigmaz.tied = fw_utils_io.tie_sigz_reff

    # Geometry
    geom = models.Geometry(
        inc=62., pa=142., xshift=0., yshift=0., name='geom',
        obs_name='halpha_1D',
        fixed={'inc': False, 'pa': True, 'xshift': True, 'yshift': True},
        bounds={'inc': (0, 90), 'pa': (90, 180), 'xshift': (0, 4),
                'yshift': (-10, -4)},
    )

    # Dimming
    dimming = models.ConstantDimming(amp_lumtoflux=1.e-10)

    mod_set.add_component(bary, light=True)
    mod_set.add_component(halo)
    mod_set.add_component(disp_prof)
    mod_set.add_component(zheight_prof)
    mod_set.add_component(geom)

    mod_set.kinematic_options.adiabatic_contract = False
    mod_set.kinematic_options.pressure_support = True
    mod_set.kinematic_options.pressure_support_type = 1
    mod_set.dimming = dimming

    gal.model = mod_set

    # Instrument
    inst = instrument.Instrument()
    inst.pixscale = 0.125 * u.arcsec
    inst.fov = [33, 33]
    inst.beam = instrument.GaussianBeam(major=0.55 * u.arcsec)
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


# Fresh reference pixel values (computed with current code, post-NoordFlat fix)
REFERENCE_PIXELS = [
    [100, 18, 18, 0.026635293493745136],
    [0,   0,  0,  -3.1622563364160546e-21],
    [100, 18, 0,   2.382722729964772e-06],
    [50,  18, 18,  1.922890058107344e-08],
    [95,  10, 10,  0.0018233324045527494],
    [100,  5,  5,   2.7312562949350115e-05],
    [150, 18, 18,  4.3899435556160604e-07],
    [100, 15, 15,  0.05094232297026224],
    [100, 15, 21,  0.016517850824120932],
    [90,  15, 15,  0.07451384876677379],
    [90,  15, 21,  0.004111754959640526],
]


def main():
    atol = 1e-12
    all_pass = True

    print("=" * 70)
    print("DysmalPy Model Equivalence Comparison")
    print("=" * 70)
    print(f"numpy: {np.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print()

    # --- Setup ---
    print("[Setup] Creating galaxy model...")
    gal, obs = setup_fullmodel()
    dscale = gal.dscale
    mod_set = gal.model
    inst = obs.instrument

    oversample = obs.mod_options.oversample
    oversize = obs.mod_options.oversize
    nx_sky = inst.fov[0]
    ny_sky = inst.fov[1]

    print(f"  z={gal.z}, oversample={oversample}, oversize={oversize}")
    print(f"  FOV={nx_sky}x{ny_sky}, pixscale={inst.pixscale}")
    print(f"  beam={inst.beam.major}, LSF={inst.lsf}")
    print(f"  nspec={inst.nspec}")
    print()

    # --- Stage A: JAX determinism ---
    print("[Stage A] JAX determinism (two consecutive simulate_cube runs)...")
    cube_raw1, _ = mod_set.simulate_cube(obs, dscale)
    cube_raw1 = np.asarray(cube_raw1)
    cube_raw2, _ = mod_set.simulate_cube(obs, dscale)
    cube_raw2 = np.asarray(cube_raw2)
    max_det_diff = np.max(np.abs(cube_raw1 - cube_raw2))
    stage_a_ok = max_det_diff < 1e-15
    print(f"  Shape: {cube_raw1.shape}")
    print(f"  Max absolute diff: {max_det_diff:.2e}")
    print(f"  {'PASS' if stage_a_ok else 'FAIL'}")
    all_pass = all_pass and stage_a_ok
    print()

    # Use first run for remaining stages
    cube_raw = cube_raw1

    # --- Stage B: rebin ---
    print("[Stage B] rebin (oversampled -> native pixel scale)...")
    if oversample > 1:
        cube_rebinned = rebin(cube_raw, (ny_sky * oversize, nx_sky * oversize))
    else:
        cube_rebinned = cube_raw
    cube_rebinned = np.asarray(cube_rebinned)
    expected_spatial = (ny_sky * oversize, nx_sky * oversize)
    stage_b_ok = cube_rebinned.shape[1:] == expected_spatial
    print(f"  Shape: {cube_rebinned.shape} (expected spatial: {expected_spatial})")
    print(f"  {'PASS' if stage_b_ok else 'FAIL'}")
    all_pass = all_pass and stage_b_ok
    print()

    # --- Stage C: convolve ---
    print("[Stage C] convolve (beam + LSF, scipy fftconvolve)...")
    cube_convolved = inst.convolve(cube=cube_rebinned, spec_center=inst.line_center)
    cube_convolved = np.asarray(cube_convolved)
    print(f"  Shape: {cube_convolved.shape}")
    print(f"  min={cube_convolved.min():.6e}, max={cube_convolved.max():.6e}")
    stage_c_ok = cube_convolved.shape == (inst.nspec, ny_sky, nx_sky)
    print(f"  {'PASS' if stage_c_ok else 'FAIL'}")
    all_pass = all_pass and stage_c_ok
    print()

    # --- Stage D: crop ---
    print("[Stage D] crop (if oversize > 1)...")
    if oversize > 1:
        nx_os = cube_convolved.shape[2]
        ny_os = cube_convolved.shape[1]
        y_s = int(ny_os / 2 - ny_sky / 2)
        y_e = int(ny_os / 2 + ny_sky / 2)
        x_s = int(nx_os / 2 - nx_sky / 2)
        x_e = int(nx_os / 2 + nx_sky / 2)
        cube_final = cube_convolved[:, y_s:y_e, x_s:x_e]
    else:
        cube_final = cube_convolved
    cube_final = np.asarray(cube_final)
    print(f"  Shape: {cube_final.shape}")
    stage_d_ok = cube_final.shape == (inst.nspec, ny_sky, nx_sky)
    print(f"  {'PASS' if stage_d_ok else 'FAIL'}")
    all_pass = all_pass and stage_d_ok
    print()

    # --- Stage E: compare against fresh reference pixel values ---
    print(f"[Stage E] Compare against reference pixel values (abs_tol={atol:.0e})...")
    max_diff = 0.0
    n_pass = 0
    for pix in REFERENCE_PIXELS:
        zi, yi, xi, expected = pix
        actual = float(cube_final[zi, yi, xi])
        diff = abs(actual - expected)
        max_diff = max(max_diff, diff)
        ok = math.isclose(actual, expected, abs_tol=atol)
        if ok:
            n_pass += 1
        else:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{zi:3d},{yi:2d},{xi:2d}] expected={expected: .10e}  "
              f"actual={actual: .10e}  diff={diff:.2e}  {status}")

    print(f"\n  Summary: {n_pass}/{len(REFERENCE_PIXELS)} passed, max diff = {max_diff:.2e}")
    print(f"  {'PASS' if n_pass == len(REFERENCE_PIXELS) else 'FAIL'}")
    print()

    # --- Stage F: chi-squared sanity check ---
    print("[Stage F] Chi-squared sanity check...")
    noise = np.ones_like(cube_final)
    chi_sq = np.sum(((cube_final - cube_final) / noise) ** 2)
    stage_f_ok = chi_sq < 1e-20
    print(f"  chi^2 (model vs itself) = {chi_sq:.2e}")
    print(f"  {'PASS' if stage_f_ok else 'FAIL'}")
    all_pass = all_pass and stage_f_ok
    print()

    # --- Summary ---
    print("=" * 70)
    if all_pass:
        print("ALL STAGES PASSED")
    else:
        print("SOME STAGES FAILED - See details above")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
