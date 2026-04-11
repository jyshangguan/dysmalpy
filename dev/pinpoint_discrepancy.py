# coding=utf8
"""
Pinpoint the source of the ~3.5% discrepancy between dev_jax and original dysmalpy.

Strategy: dump key intermediate arrays from simulate_cube in both environments
and compare them one by one to find where the divergence first appears.

Intermediates (in order of computation):
  1. rgal            — cylindrical radius
  2. vrot            — rotation velocity from velocity_profile
  3. vobs_mass       — line-of-sight observed velocity
  4. zscale          — z-height profile
  5. flux_mass       — light-weighted flux
  6. sigmar          — velocity dispersion
  7. vx (vspec)      — spectral channel velocities
  8. ai              — active pixel indices

Run from the repo root:
    JAX_PLATFORMS=cpu python dev/pinpoint_discrepancy.py
"""

import subprocess
import sys
import os
import textwrap
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update("jax_enable_x64", True)

import astropy.units as u
from dysmalpy import galaxy, models, parameters, instrument, observation
from dysmalpy.fitting_wrappers import utils_io as fw_utils_io


def setup_fullmodel():
    """Identical model setup."""
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
    return gal, obs


def dump_intermediates_jax():
    """Dump intermediate arrays from JAX simulate_cube."""
    gal, obs = setup_fullmodel()
    dscale = gal.dscale
    mod_set = gal.model
    inst = obs.instrument

    oversample = obs.mod_options.oversample
    oversize = obs.mod_options.oversize
    nx_sky = inst.fov[0]
    ny_sky = inst.fov[1]
    nx_sky_samp = nx_sky * oversample * oversize
    ny_sky_samp = ny_sky * oversample * oversize

    # Replicate the beginning of simulate_cube
    import jax.numpy as jnp
    from dysmalpy.models.model_set import (
        _get_xyz_sky_gal, _calculate_max_skyframe_extents, _make_cube_ai
    )
    import dysmalpy.models.utils as model_utils

    transform_method = 'direct'
    geom = mod_set.geometries[obs.name]

    nz_sky_samp, maxr, maxr_y = _calculate_max_skyframe_extents(
        geom, nx_sky_samp, ny_sky_samp, transform_method, angle='sin')

    sh = (nz_sky_samp, ny_sky_samp, nx_sky_samp)
    xc_samp = (sh[2] - 1) / 2.
    yc_samp = (sh[1] - 1) / 2.
    zc_samp = (sh[0] - 1) / 2.

    xgal, ygal, zgal, xsky, ysky, zsky = _get_xyz_sky_gal(geom, sh, xc_samp, yc_samp, zc_samp)

    pixscale_samp = inst.pixscale.value / oversample
    to_kpc = pixscale_samp / dscale

    # 1. rgal
    rgal = np.asarray(jnp.sqrt(xgal ** 2 + ygal ** 2))

    # 2. vrot from velocity_profile
    vrot = np.asarray(mod_set.velocity_profile(rgal * to_kpc, tracer=obs.tracer))

    # 3. vobs_mass
    LOS_hat = np.array([np.sin(np.radians(float(geom.inc))) * np.cos(np.radians(float(geom.pa))),
                         np.sin(np.radians(float(geom.inc))) * np.sin(np.radians(float(geom.pa))),
                         np.cos(np.radians(float(geom.inc)))])
    v_sys = 0.0

    rgal_safe = np.where(rgal > 0, rgal, 1.0)
    vobs_mass = v_sys + vrot * xgal / rgal_safe * LOS_hat[1]
    vobs_mass = np.where(rgal > 0, vobs_mass, v_sys)

    # 4. zscale
    zscale = np.asarray(mod_set.zprofile(zgal * to_kpc))

    # 5. flux_mass
    tracer_lcomps = model_utils.get_light_components_by_tracer(mod_set, obs.tracer)
    flux_mass = np.zeros(rgal.shape)
    for cmp in tracer_lcomps:
        if mod_set.light_components[cmp]:
            lcomp = mod_set.components[cmp]
            if lcomp._axisymmetric:
                flux_mass += np.asarray(lcomp.light_profile(rgal * to_kpc)) * zscale
            else:
                flux_mass += np.asarray(lcomp.light_profile(xgal * to_kpc, ygal * to_kpc, zgal * to_kpc)) * zscale

    # 6. sigmar
    sigmar = np.asarray(mod_set.dispersions[obs.tracer](rgal * to_kpc))

    # 7. vx (vspec)
    nspec = inst.nspec
    spec_step = inst.spec_step.value
    spec_start = inst.spec_start.to(inst.spec_step.unit).value
    vx = np.arange(nspec) * spec_step + spec_start

    # Save
    outpath = os.path.join(os.path.dirname(__file__), 'intermediates_jax.npz')
    np.savez(outpath,
             rgal=rgal, vrot=vrot, vobs_mass=vobs_mass,
             zscale=zscale, flux_mass=flux_mass, sigmar=sigmar,
             vx=vx, xgal=np.asarray(xgal), ygal=np.asarray(ygal),
             zgal=np.asarray(zgal), to_kpc=to_kpc,
             nz_sky_samp=nz_sky_samp, ny_sky_samp=ny_sky_samp, nx_sky_samp=nx_sky_samp)
    print(f"[JAX] Saved to {outpath}")
    return outpath


ORIGINAL_ENV_CODE = textwrap.dedent(r"""
    import sys, os, numpy as np

    # Change CWD to avoid importing the git repo version via sys.path[0]=''
    os.chdir('/tmp')

    # Monkey-patch for numpy 1.26 compat
    np.int = int
    np.float = np.float64

    # Make sure we get the original dysmalpy, NOT the git repo version
    sys.path = [p for p in sys.path if "my_modules/dysmalpy" not in p]

    import astropy.units as u
    from dysmalpy import galaxy, models, parameters, instrument, observation
    from dysmalpy.fitting_wrappers import utils_io as fw_utils_io
    from dysmalpy.models.model_set import (
        _get_xyz_sky_gal, _calculate_max_skyframe_extents, _make_cube_ai
    )
    import dysmalpy.models.utils as model_utils

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
    oversample = obs.mod_options.oversample
    oversize = obs.mod_options.oversize
    nx_sky = inst.fov[0]
    ny_sky = inst.fov[1]
    nx_sky_samp = nx_sky * oversample * oversize
    ny_sky_samp = ny_sky * oversample * oversize

    transform_method = 'direct'
    geom_obj = mod_set.geometries[obs.name]

    nz_sky_samp, maxr, maxr_y = _calculate_max_skyframe_extents(
        geom_obj, nx_sky_samp, ny_sky_samp, transform_method, angle='sin')

    sh = (nz_sky_samp, ny_sky_samp, nx_sky_samp)
    xc_samp = (sh[2] - 1) / 2.
    yc_samp = (sh[1] - 1) / 2.
    zc_samp = (sh[0] - 1) / 2.

    xgal, ygal, zgal, xsky, ysky, zsky = _get_xyz_sky_gal(geom_obj, sh, xc_samp, yc_samp, zc_samp)

    pixscale_samp = inst.pixscale.value / oversample
    to_kpc = pixscale_samp / dscale

    rgal = np.sqrt(xgal ** 2 + ygal ** 2)

    vrot = mod_set.velocity_profile(rgal * to_kpc, tracer=obs.tracer)

    LOS_hat = np.array([np.sin(np.radians(geom_obj.inc)) * np.cos(np.radians(geom_obj.pa)),
                         np.sin(np.radians(geom_obj.inc)) * np.sin(np.radians(geom_obj.pa)),
                         np.cos(np.radians(geom_obj.inc))])
    v_sys = 0.0

    rgal_safe = np.where(rgal > 0, rgal, 1.0)
    vobs_mass = v_sys + vrot * xgal / rgal_safe * LOS_hat[1]
    vobs_mass = model_utils.replace_values_by_refarr(vobs_mass, rgal, 0., v_sys)

    zscale = mod_set.zprofile(zgal * to_kpc)

    tracer_lcomps = model_utils.get_light_components_by_tracer(mod_set, obs.tracer)
    flux_mass = np.zeros(rgal.shape)
    for cmp in tracer_lcomps:
        if mod_set.light_components[cmp]:
            lcomp = mod_set.components[cmp]
            if lcomp._axisymmetric:
                flux_mass += lcomp.light_profile(rgal * to_kpc) * zscale
            else:
                flux_mass += lcomp.light_profile(xgal * to_kpc, ygal * to_kpc, zgal * to_kpc) * zscale

    sigmar = mod_set.dispersions[obs.tracer](rgal * to_kpc)

    nspec = inst.nspec
    spec_step = inst.spec_step.value
    spec_start = inst.spec_start.to(inst.spec_step.unit).value
    vx = np.arange(nspec) * spec_step + spec_start

    outpath = os.path.join(os.path.dirname(__file__), 'intermediates_original.npz')
    np.savez(outpath,
             rgal=rgal, vrot=vrot, vobs_mass=vobs_mass,
             zscale=zscale, flux_mass=flux_mass, sigmar=sigmar,
             vx=vx, xgal=xgal, ygal=ygal,
             zgal=zgal, to_kpc=to_kpc,
             nz_sky_samp=nz_sky_samp, ny_sky_samp=ny_sky_samp, nx_sky_samp=nx_sky_samp)
    print(f"SAVED {outpath}")
    print(f"SHAPES rgal={rgal.shape} vrot={vrot.shape} flux={flux_mass.shape}")
    print(f"to_kpc={to_kpc:.15e}")
    print(f"nz={nz_sky_samp} ny={ny_sky_samp} nx={nx_sky_samp}")
""")


def dump_intermediates_original():
    """Dump intermediates from original dysmalpy via subprocess."""
    alma_python = '/home/shangguan/Softwares/miniconda3/envs/alma/bin/python'
    script_path = os.path.abspath(__file__)
    tmp_script = os.path.join(os.path.dirname(__file__), '_pinpoint_orig_tmp.py')

    with open(tmp_script, 'w') as f:
        f.write(ORIGINAL_ENV_CODE.replace('__file__', repr(script_path)))

    print("[Original] Running in alma env...")
    try:
        result = subprocess.run(
            [alma_python, tmp_script],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"[Original] STDERR:\n{result.stderr}")
            raise RuntimeError(f"Subprocess failed: {result.returncode}")
        for line in result.stdout.strip().splitlines():
            print(f"[Original] {line}")
    finally:
        if os.path.exists(tmp_script):
            os.remove(tmp_script)

    outpath = os.path.join(os.path.dirname(__file__), 'intermediates_original.npz')
    return outpath


def compare(name_jax, name_orig):
    """Compare intermediate arrays side-by-side."""
    print()
    print("=" * 75)
    print("INTERMEDIATE ARRAY COMPARISON")
    print("=" * 75)

    dj = np.load(name_jax)
    do_ = np.load(name_orig)

    # Check grid dimensions first
    for key in ['nz_sky_samp', 'ny_sky_samp', 'nx_sky_samp', 'to_kpc']:
        vj = float(dj[key])
        vo = float(do_[key])
        diff = abs(vj - vo)
        status = "OK" if diff < 1e-15 else "*** MISMATCH ***"
        print(f"  {key:20s}: JAX={vj:.10e}  orig={vo:.10e}  diff={diff:.2e}  {status}")

    print()

    # Compare coordinate grids
    for arr_name in ['xgal', 'ygal', 'zgal', 'rgal']:
        aj = dj[arr_name]
        ao = do_[arr_name]
        if aj.shape != ao.shape:
            print(f"  {arr_name:20s}: SHAPE MISMATCH {aj.shape} vs {ao.shape}")
            continue
        diff = np.abs(aj - ao)
        mx = np.max(diff)
        mn = np.mean(diff)
        status = "OK" if mx < 1e-12 else "*** DIFFERS ***"
        print(f"  {arr_name:20s}: max_diff={mx:.2e}  mean_diff={mn:.2e}  {status}")

    print()

    # Compare physics arrays (may differ)
    for arr_name in ['vrot', 'vobs_mass', 'flux_mass', 'sigmar', 'zscale', 'vx']:
        aj = dj[arr_name]
        ao = do_[arr_name]
        if aj.shape != ao.shape:
            print(f"  {arr_name:20s}: SHAPE MISMATCH {aj.shape} vs {ao.shape}")
            continue
        diff = np.abs(aj - ao)
        mx = np.max(diff)
        mn = np.mean(diff)

        # Find peak location
        peak_j = np.max(aj)
        peak_o = np.max(ao)
        peak_idx = np.unravel_index(np.argmax(aj), aj.shape)
        peak_val_j = aj[peak_idx]
        peak_val_o = ao[peak_idx]
        peak_diff = abs(peak_val_j - peak_val_o)
        rel_peak = peak_diff / max(abs(peak_val_o), 1e-300)

        status = "OK" if mx < 1e-10 else "*** DIFFERS ***"
        print(f"  {arr_name:20s}: max_diff={mx:.6e}  mean_diff={mn:.6e}  "
              f"rel@peak={rel_peak:.6e}  {status}")
        if mx > 1e-10:
            print(f"    peak: JAX={peak_val_j:.10e}  orig={peak_val_o:.10e}  "
                  f"at {peak_idx}")

    print()
    print("=" * 75)

    # Deep dive: compare velocity_profile sub-components
    print()
    print("DEEP DIVE: velocity_profile decomposition")
    print("-" * 75)

    rgal_j = dj['rgal']
    rgal_o = do_['rgal']
    to_kpc_j = float(dj['to_kpc'])

    # Extract a 1D slice through the midplane for cleaner comparison
    mid_z = rgal_j.shape[0] // 2
    mid_y = rgal_j.shape[1] // 2
    r_1d = rgal_j[mid_z, mid_y, :] * to_kpc_j  # kpc
    valid = r_1d > 0.01  # avoid r=0

    vrot_j_1d = dj['vrot'][mid_z, mid_y, :]
    vrot_o_1d = do_['vrot'][mid_z, mid_y, :]

    print(f"  1D slice: z={mid_z}, y={mid_y}, valid_r={np.sum(valid)} pixels")
    print(f"  r range: {r_1d[valid].min():.3f} - {r_1d[valid].max():.3f} kpc")

    diff_vrot = np.abs(vrot_j_1d[valid] - vrot_o_1d[valid])
    print(f"  vrot:     max_diff={diff_vrot.max():.6e}  "
          f"at r={r_1d[valid][np.argmax(diff_vrot)]:.3f} kpc")

    # Find the max rotation velocity location
    peak_r_idx = np.argmax(vrot_j_1d[valid])
    r_peak = r_1d[valid][peak_r_idx]
    print(f"  vrot peak: JAX={vrot_j_1d[valid][peak_r_idx]:.6e}  "
          f"orig={vrot_o_1d[valid][peak_r_idx]:.6e}  "
          f"at r={r_peak:.3f} kpc  "
          f"rel={abs(vrot_j_1d[valid][peak_r_idx]-vrot_o_1d[valid][peak_r_idx])/vrot_o_1d[valid][peak_r_idx]:.6e}")

    flux_j_1d = dj['flux_mass'][mid_z, mid_y, :]
    flux_o_1d = do_['flux_mass'][mid_z, mid_y, :]
    diff_flux = np.abs(flux_j_1d[valid] - flux_o_1d[valid])
    print(f"  flux_mass: max_diff={diff_flux.max():.6e}  "
          f"at r={r_1d[valid][np.argmax(diff_flux)]:.3f} kpc")

    vobs_j_1d = dj['vobs_mass'][mid_z, mid_y, :]
    vobs_o_1d = do_['vobs_mass'][mid_z, mid_y, :]
    diff_vobs = np.abs(vobs_j_1d[valid] - vobs_o_1d[valid])
    print(f"  vobs_mass: max_diff={diff_vobs.max():.6e}  "
          f"at r={r_1d[valid][np.argmax(diff_vobs)]:.3f} kpc")

    sigma_j_1d = dj['sigmar'][mid_z, mid_y, :]
    sigma_o_1d = do_['sigmar'][mid_z, mid_y, :]
    diff_sigma = np.abs(sigma_j_1d[valid] - sigma_o_1d[valid])
    print(f"  sigmar:    max_diff={diff_sigma.max():.6e}  "
          f"at r={r_1d[valid][np.argmax(diff_sigma)]:.3f} kpc")

    zscale_j_1d = dj['zscale'][mid_z, mid_y, :]
    zscale_o_1d = do_['zscale'][mid_z, mid_y, :]
    diff_zscale = np.abs(zscale_j_1d - zscale_o_1d)
    print(f"  zscale:    max_diff={diff_zscale.max():.6e}  "
          f"(same r for all x, expected identical)")

    print()
    print("=" * 75)


def main():
    print("=" * 75)
    print("Discrepancy Pinpointer: dev_jax vs Original DysmalPy")
    print("=" * 75)

    jax_path = dump_intermediates_jax()
    print()
    orig_path = dump_intermediates_original()
    print()
    compare(jax_path, orig_path)


if __name__ == '__main__':
    main()
