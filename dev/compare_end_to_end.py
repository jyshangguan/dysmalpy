# coding=utf8
"""
End-to-end model equivalence: dev_jax vs Original DysmalPy.

Runs the full observation pipeline (simulate -> rebin -> convolve -> crop)
in both the JAX branch (current env) and the original Cython-based dysmalpy
(via subprocess in alma conda env), then compares the resulting cubes.

Run from the repo root:
    JAX_PLATFORMS=cpu python dev/compare_end_to_end.py
"""

import subprocess
import sys
import os
import math
import textwrap
import numpy as np

# Ensure the repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update("jax_enable_x64", True)

import astropy.units as u
from dysmalpy import galaxy, models, parameters, instrument, observation
from dysmalpy.fitting_wrappers import utils_io as fw_utils_io
from dysmalpy.utils import rebin


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


def postprocess_cube(cube_raw, obs):
    """Apply rebin + convolve + crop pipeline (identical to observation.py)."""
    inst = obs.instrument
    oversample = obs.mod_options.oversample
    oversize = obs.mod_options.oversize
    nx_sky = inst.fov[0]
    ny_sky = inst.fov[1]

    cube = np.asarray(cube_raw)

    # Rebin
    if oversample > 1:
        cube = rebin(cube, (ny_sky * oversize, nx_sky * oversize))
        cube = np.asarray(cube)

    # Convolve
    cube = inst.convolve(cube=cube, spec_center=inst.line_center)
    cube = np.asarray(cube)

    # Crop
    if oversize > 1:
        nx_os = cube.shape[2]
        ny_os = cube.shape[1]
        y_s = int(ny_os / 2 - ny_sky / 2)
        y_e = int(ny_os / 2 + ny_sky / 2)
        x_s = int(nx_os / 2 - nx_sky / 2)
        x_e = int(nx_os / 2 + nx_sky / 2)
        cube = cube[:, y_s:y_e, x_s:x_e]

    return np.asarray(cube)


def run_jax_pipeline():
    """Run simulate_cube in dev_jax and save results."""
    print("[JAX] Setting up model...")
    gal, obs = setup_fullmodel()
    dscale = gal.dscale
    mod_set = gal.model

    print("[JAX] Running simulate_cube...")
    cube_raw, _ = mod_set.simulate_cube(obs, dscale)
    cube_raw = np.asarray(cube_raw)

    print("[JAX] Post-processing (rebin + convolve + crop)...")
    cube_final = postprocess_cube(cube_raw, obs)

    outpath = os.path.join(os.path.dirname(__file__), 'results_jax.npz')
    np.savez(outpath, cube_raw=cube_raw, cube_final=cube_final)
    print(f"[JAX] Saved to {outpath}")
    print(f"[JAX]   raw shape:    {cube_raw.shape}")
    print(f"[JAX]   final shape:  {cube_final.shape}")
    print(f"[JAX]   raw max:      {cube_raw.max():.6e}")
    print(f"[JAX]   final max:    {cube_final.max():.6e}")
    return outpath


# ---------------------------------------------------------------------------
# Subprocess Python code for the original dysmalpy in the alma conda env.
# This code is sent as a string to `python -c`.
# ---------------------------------------------------------------------------
ORIGINAL_ENV_CODE = textwrap.dedent(r"""
    import sys, os, numpy as np

    # Monkey-patch for numpy 1.26 compat
    np.int = int
    np.float = np.float64

    # Change CWD to avoid importing the git repo version via sys.path[0]=''
    os.chdir('/tmp')

    # Make sure we get the original dysmalpy, NOT the git repo version
    sys.path = [p for p in sys.path if "my_modules/dysmalpy" not in p]

    import astropy.units as u
    from dysmalpy import galaxy, models, parameters, instrument, observation
    from dysmalpy.fitting_wrappers import utils_io as fw_utils_io
    from dysmalpy.utils import rebin

    z = 1.613
    name = 'GS4_43501'

    gal = galaxy.Galaxy(z=z, name=name)
    obs = observation.Observation(name='halpha_1D', tracer='halpha')
    obs.mod_options.oversample = 3
    obs.mod_options.zcalc_truncate = True

    mod_set = models.ModelSet()

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

    halo = models.NFW(
        mvirial=12.0, conc=5.0, fdm=0.5, z=z, name='halo',
        fixed={'mvirial': False, 'conc': True, 'fdm': False},
        bounds={'mvirial': (10, 13), 'conc': (1, 20), 'fdm': (0., 1.)},
    )
    halo.fdm.tied = fw_utils_io.tie_fdm

    disp_prof = models.DispersionConst(
        sigma0=39., name='dispprof', tracer='halpha',
        fixed={'sigma0': False}, bounds={'sigma0': (10, 200)},
    )

    zheight_prof = models.ZHeightGauss(
        sigmaz=0.9, name='zheightgaus',
        fixed={'sigmaz': False},
    )
    zheight_prof.sigmaz.tied = fw_utils_io.tie_sigz_reff

    geom = models.Geometry(
        inc=62., pa=142., xshift=0., yshift=0., name='geom',
        obs_name='halpha_1D',
        fixed={'inc': False, 'pa': True, 'xshift': True, 'yshift': True},
        bounds={'inc': (0, 90), 'pa': (90, 180), 'xshift': (0, 4),
                'yshift': (-10, -4)},
    )

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

    dscale = gal.dscale

    # simulate_cube
    cube_raw, _ = mod_set.simulate_cube(obs, dscale)
    cube_raw = np.asarray(cube_raw)

    # rebin
    oversample = obs.mod_options.oversample
    oversize = obs.mod_options.oversize
    nx_sky = inst.fov[0]
    ny_sky = inst.fov[1]

    if oversample > 1:
        cube_rb = rebin(cube_raw, (ny_sky * oversize, nx_sky * oversize))
        cube_rb = np.asarray(cube_rb)
    else:
        cube_rb = cube_raw

    # convolve
    cube_cv = inst.convolve(cube=cube_rb, spec_center=inst.line_center)
    cube_cv = np.asarray(cube_cv)

    # crop
    if oversize > 1:
        nx_os = cube_cv.shape[2]
        ny_os = cube_cv.shape[1]
        y_s = int(ny_os / 2 - ny_sky / 2)
        y_e = int(ny_os / 2 + ny_sky / 2)
        x_s = int(nx_os / 2 - nx_sky / 2)
        x_e = int(nx_os / 2 + nx_sky / 2)
        cube_final = cube_cv[:, y_s:y_e, x_s:x_e]
    else:
        cube_final = cube_cv

    cube_final = np.asarray(cube_final)

    outpath = os.path.join(os.path.dirname(__file__), 'results_original.npz')
    np.savez(outpath, cube_raw=cube_raw, cube_final=cube_final)
    print(f"SAVED {outpath}")
    print(f"RAW_SHAPE {cube_raw.shape}")
    print(f"RAW_MAX {cube_raw.max():.15e}")
    print(f"FINAL_SHAPE {cube_final.shape}")
    print(f"FINAL_MAX {cube_final.max():.15e}")
""")


def run_original_pipeline():
    """Run simulate_cube in original dysmalpy via subprocess in alma env."""
    alma_python = '/home/shangguan/Softwares/miniconda3/envs/alma/bin/python'
    script_path = os.path.abspath(__file__)

    # Write the inline Python to a temp file so it can reference __file__
    tmp_script = os.path.join(os.path.dirname(__file__), '_run_original_tmp.py')
    with open(tmp_script, 'w') as f:
        f.write(ORIGINAL_ENV_CODE.replace('__file__', repr(script_path)))

    print("[Original] Running in alma conda env via subprocess...")
    try:
        result = subprocess.run(
            [alma_python, tmp_script],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"[Original] STDERR:\n{result.stderr}")
            raise RuntimeError(
                f"Original dysmalpy subprocess failed with exit code {result.returncode}"
            )
        # Print subprocess stdout for diagnostics
        for line in result.stdout.strip().splitlines():
            print(f"[Original] {line}")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_script):
            os.remove(tmp_script)

    outpath = os.path.join(os.path.dirname(__file__), 'results_original.npz')
    if not os.path.exists(outpath):
        raise RuntimeError(f"Expected output file not found: {outpath}")

    print(f"[Original] Results saved to {outpath}")
    return outpath


def compare_results(jax_path, original_path):
    """Compare the JAX and original results."""
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    data_jax = np.load(jax_path)
    data_orig = np.load(original_path)

    cube_jax_raw = data_jax['cube_raw']
    cube_orig_raw = data_orig['cube_raw']
    cube_jax_final = data_jax['cube_final']
    cube_orig_final = data_orig['cube_final']

    all_pass = True

    # --- Raw cube comparison ---
    print()
    print("[Raw Cube Comparison]")
    print(f"  JAX shape:    {cube_jax_raw.shape}")
    print(f"  Original shape: {cube_orig_raw.shape}")

    raw_shapes_match = cube_jax_raw.shape == cube_orig_raw.shape
    print(f"  Shapes match: {raw_shapes_match}")
    if not raw_shapes_match:
        all_pass = False

    if raw_shapes_match:
        raw_diff = np.abs(cube_jax_raw - cube_orig_raw)
        raw_max_diff = np.max(raw_diff)
        raw_mean_diff = np.mean(raw_diff)

        peak_jax = np.max(cube_jax_raw)
        peak_orig = np.max(cube_orig_raw)
        raw_rel_peak = abs(peak_jax - peak_orig) / max(abs(peak_orig), 1e-300)

        raw_pass = raw_max_diff < 1e-10

        print(f"  Max absolute diff:  {raw_max_diff:.6e}")
        print(f"  Mean absolute diff: {raw_mean_diff:.6e}")
        print(f"  Relative diff at peak: {raw_rel_peak:.6e}")
        print(f"  Tolerance: 1e-10")
        print(f"  {'PASS' if raw_pass else 'FAIL'}")
        if not raw_pass:
            all_pass = False
    else:
        raw_pass = False

    # --- Post-processed cube comparison ---
    print()
    print("[Post-Processed Cube Comparison]")
    print(f"  JAX shape:    {cube_jax_final.shape}")
    print(f"  Original shape: {cube_orig_final.shape}")

    final_shapes_match = cube_jax_final.shape == cube_orig_final.shape
    print(f"  Shapes match: {final_shapes_match}")
    if not final_shapes_match:
        all_pass = False

    if final_shapes_match:
        final_diff = np.abs(cube_jax_final - cube_orig_final)
        final_max_diff = np.max(final_diff)
        final_mean_diff = np.mean(final_diff)

        peak_jax_f = np.max(cube_jax_final)
        peak_orig_f = np.max(cube_orig_final)
        final_rel_peak = abs(peak_jax_f - peak_orig_f) / max(abs(peak_orig_f), 1e-300)

        final_pass = final_max_diff < 1e-10

        print(f"  Max absolute diff:  {final_max_diff:.6e}")
        print(f"  Mean absolute diff: {final_mean_diff:.6e}")
        print(f"  Relative diff at peak: {final_rel_peak:.6e}")
        print(f"  Tolerance: 1e-10")
        print(f"  {'PASS' if final_pass else 'FAIL'}")
        if not final_pass:
            all_pass = False
    else:
        final_pass = False

    # --- Chi-squared consistency ---
    print()
    print("[Chi-Squared Consistency]")
    if final_shapes_match:
        chi_sq = np.sum((cube_jax_final - cube_orig_final) ** 2)
        chi_pass = chi_sq < 1e-8
        print(f"  chi^2 (jax-original, noise=1): {chi_sq:.6e}")
        print(f"  Tolerance: 1e-8")
        print(f"  {'PASS' if chi_pass else 'FAIL'}")
        if not chi_pass:
            all_pass = False
    else:
        chi_sq = float('inf')
        chi_pass = False
        print("  SKIP (shape mismatch)")
        all_pass = False

    # --- Summary ---
    print()
    print("=" * 70)
    if all_pass:
        print("ALL COMPARISONS PASSED")
    else:
        print("SOME COMPARISONS FAILED")
    print("=" * 70)

    return 0 if all_pass else 1


def main():
    print("=" * 70)
    print("End-to-End Model Equivalence: dev_jax vs Original DysmalPy")
    print("=" * 70)
    print(f"numpy:  {np.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print()

    # Step 1: JAX pipeline
    jax_path = run_jax_pipeline()
    print()

    # Step 2: Original pipeline
    original_path = run_original_pipeline()
    print()

    # Step 3: Compare
    return compare_results(jax_path, original_path)


if __name__ == '__main__':
    sys.exit(main())
