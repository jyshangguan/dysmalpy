#!/usr/bin/env python
"""Save cube + moment maps from the MPFIT best-fit model for branch comparison.

Supports two modes:
  --from-pickle : Load Galaxy from pickle (dev_jax only, also saves params JSON)
  --from-params : Reconstruct Galaxy from saved params JSON (works on any branch)

Usage:
    # On dev_jax: extract params + save JAX cube
    JAX_PLATFORMS=cpu python dev/debug_save_cube_moments.py --label jax --from-pickle

    # On main: reconstruct from params + save main cube
    JAX_PLATFORMS=cpu python dev/debug_save_cube_moments.py --label main --from-params

Saves to dev/debug_cubes/{label}_cube.npz containing:
    - cube: raw 3D data array (nspec, ny, nx)
    - mom0, mom1, mom2: manually computed moment maps
    - velocity, dispersion: 2D model maps (if ndim=2)
    - spec_axis: spectral axis in km/s
"""

import os
import json
import argparse
import numpy as np
import astropy.units as u

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS_NAME = 'OBS'
PARAMS_PATH = os.path.join(REPO_ROOT, 'dev', 'debug_cubes', 'model_params.json')


def compute_moment_maps(cube_data, spec_axis_kms):
    """Compute moment maps manually from raw cube data."""
    mom0 = np.nansum(cube_data, axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        mom1 = np.nansum(cube_data * spec_axis_kms[:, None, None], axis=0) / mom0
    mom1 = np.where(np.isfinite(mom1), mom1, 0.0)

    with np.errstate(invalid='ignore', divide='ignore'):
        mom2_sq = (np.nansum(
            cube_data * (spec_axis_kms[:, None, None] - mom1[None, :, :]) ** 2,
            axis=0
        ) / mom0)
        mom2 = np.sqrt(np.maximum(mom2_sq, 0.0))
    mom2 = np.where(np.isfinite(mom2), mom2, 0.0)

    return mom0, mom1, mom2


def extract_params_from_gal(gal):
    """Extract portable parameters from a loaded Galaxy object.

    Returns a dict that can be serialized to JSON and used to reconstruct
    the model on a different branch (no pickle dependency).
    """
    obs = gal.observations[OBS_NAME]
    ms = gal.model

    # Collect component parameter values
    comp_params = {}
    for name, comp in ms.components.items():
        comp_params[name] = {}
        # Get the best-fit parameter values (as floats, not Parameter objects)
        if hasattr(comp, 'parameters'):
            for i, pname in enumerate(comp.param_names):
                comp_params[name][pname] = float(comp.parameters[i])

    p = {
        'galaxy': {
            'z': gal.z,
            'name': gal.name,
        },
        'components': comp_params,
        'observation': {
            'name': obs.name,
            'tracer': obs.tracer,
            'pixscale_arcsec': obs.instrument.pixscale.to(u.arcsec).value,
            'fov': list(obs.instrument.fov),
            'nspec': obs.instrument.nspec,
            'spec_type': obs.instrument.spec_type,
            'spec_step_kms': obs.instrument.spec_step.to(u.km / u.s).value,
            'spec_start_kms': obs.instrument.spec_start.to(u.km / u.s).value,
            'beam_major_arcsec': obs.instrument.beam.major.to(u.arcsec).value,
            'lsf_sigma_kms': obs.instrument.lsf.to(u.km / u.s).value,
            'ndim': obs.instrument.ndim,
            'moment': obs.instrument.moment,
            'oversample': obs.mod_options.oversample,
            'oversize': obs.mod_options.oversize,
            'zcalc_truncate': obs.mod_options.zcalc_truncate,
            'transform_method': obs.mod_options.transform_method,
            'smoothing_type': obs.instrument.smoothing_type,
            'smoothing_npix': obs.instrument.smoothing_npix,
            'n_wholepix_z_min': obs.mod_options.n_wholepix_z_min,
            'xcenter': obs.mod_options.xcenter,
            'ycenter': obs.mod_options.ycenter,
        },
        'kinematic_options': {
            'pressure_support': ms.kinematic_options.pressure_support,
            'adiabatic_contract': ms.kinematic_options.adiabatic_contract,
        },
    }
    return p


def build_gal_from_params(p):
    """Reconstruct a Galaxy + Observation from saved parameters.

    This does NOT need a pickle — it builds everything from scratch using
    the DysmalPy public API, so it works on any branch.
    """
    from dysmalpy import galaxy, models, observation, instrument

    gp = p['galaxy']
    gal = galaxy.Galaxy(z=gp['z'], name=gp['name'])
    mod_set = models.ModelSet()

    cp = p['components']

    # Disk+Bulge
    if 'disk+bulge' in cp:
        d = cp['disk+bulge']
        bary = models.DiskBulge(
            total_mass=d['total_mass'], bt=d['bt'],
            r_eff_disk=d['r_eff_disk'], n_disk=d['n_disk'],
            invq_disk=d.get('invq_disk', 5.0),
            r_eff_bulge=d.get('r_eff_bulge', 1.0),
            n_bulge=d.get('n_bulge', 4.0),
            invq_bulge=d.get('invq_bulge', 1.0),
            noord_flat=True,
            name='bary',
        )
        mod_set.add_component(bary, light=True)

    # NFW halo
    if 'halo' in cp:
        h = cp['halo']
        halo = models.NFW(
            mvirial=h['mvirial'], conc=h['conc'], z=gp['z'],
            name='halo',
        )
        mod_set.add_component(halo)

    # Constant dispersion
    if 'dispprof_LINE' in cp:
        s = cp['dispprof_LINE']
        disp = models.DispersionConst(
            sigma0=s['sigma0'], tracer='LINE', name='disp')
        mod_set.add_component(disp)

    # Z-height
    if 'zheightgaus' in cp:
        zh = cp['zheightgaus']
        zheight = models.ZHeightGauss(
            sigmaz=zh['sigmaz'], name='zheight')
        mod_set.add_component(zheight)

    # Geometry
    if 'geom_1' in cp:
        g = cp['geom_1']
        geom = models.Geometry(
            inc=g['inc'], pa=g['pa'],
            xshift=g.get('xshift', 0.0),
            yshift=g.get('yshift', 0.0),
            obs_name=OBS_NAME, name='geom',
        )
        mod_set.add_component(geom)

    # Kinematic options
    ko = p['kinematic_options']
    mod_set.kinematic_options.pressure_support = ko['pressure_support']
    mod_set.kinematic_options.adiabatic_contract = ko['adiabatic_contract']

    # Observation
    op = p['observation']
    obs = observation.Observation(name=op['name'], tracer=op['tracer'])
    obs.mod_options.oversample = op['oversample']
    obs.mod_options.oversize = op['oversize']
    obs.mod_options.zcalc_truncate = op['zcalc_truncate']
    obs.mod_options.n_wholepix_z_min = op['n_wholepix_z_min']
    obs.mod_options.xcenter = op['xcenter']
    obs.mod_options.ycenter = op['ycenter']

    inst = instrument.Instrument()
    inst.beam = instrument.GaussianBeam(major=op['beam_major_arcsec'] * u.arcsec)
    inst.lsf = instrument.LSF(op['lsf_sigma_kms'] * u.km / u.s)
    inst.pixscale = op['pixscale_arcsec'] * u.arcsec
    inst.fov = op['fov']
    inst.spec_type = op['spec_type']
    inst.spec_step = op['spec_step_kms'] * u.km / u.s
    inst.spec_start = op['spec_start_kms'] * u.km / u.s
    inst.nspec = op['nspec']
    inst.ndim = op['ndim']
    inst.moment = op['moment']
    if op.get('smoothing_type'):
        inst.smoothing_type = op['smoothing_type']
        inst.smoothing_npix = op['smoothing_npix']

    inst.set_beam_kernel()
    inst.set_lsf_kernel()

    obs.instrument = inst

    gal.model = mod_set
    gal.add_observation(obs)

    return gal, obs


def save_cube_from_gal(gal, obs, label, out_dir):
    """Build cube from galaxy, compute moments, and save to NPZ."""
    print("Building cube via create_model_data()...")
    gal.create_model_data()

    cube_obj = obs.model_cube
    if cube_obj is None:
        raise RuntimeError("model_cube is None after create_model_data()")

    cube_data = cube_obj.data
    raw = cube_data.unmasked_data[:].value
    print(f"  Cube shape: {raw.shape}")
    print(f"  Cube dtype: {raw.dtype}")
    print(f"  Cube peak:  {np.max(np.abs(raw)):.6e}")

    spec_axis = cube_data.spectral_axis.to(u.km / u.s).value
    print(f"  Spectral axis: [{spec_axis[0]:.1f}, {spec_axis[-1]:.1f}] km/s, "
          f"{len(spec_axis)} channels")

    mom0, mom1, mom2 = compute_moment_maps(raw, spec_axis)
    print(f"  mom0 range: [{np.min(mom0):.6e}, {np.max(mom0):.6e}]")
    print(f"  mom1 range: [{np.nanmin(mom1):.2f}, {np.nanmax(mom1):.2f}] km/s")
    print(f"  mom2 range: [{np.nanmin(mom2):.2f}, {np.nanmax(mom2):.2f}] km/s")

    save_dict = {
        'cube': raw,
        'spec_axis': spec_axis,
        'mom0': mom0,
        'mom1': mom1,
        'mom2': mom2,
        'dtype': str(raw.dtype),
        'label': label,
    }

    md = obs.model_data
    if md is not None:
        for key in ['velocity', 'dispersion']:
            if key in md.data:
                arr = md.data[key]
                if hasattr(arr, 'value'):
                    arr = arr.value
                save_dict[key] = arr
                print(f"  {key} range: "
                      f"[{np.nanmin(arr):.2f}, {np.nanmax(arr):.2f}]")

    out_path = os.path.join(out_dir, f'{label}_cube.npz')
    np.savez(out_path, **save_dict)
    print(f"Saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Save cube + moments for branch comparison')
    parser.add_argument('--label', type=str, required=True,
                        help='Label for this run (e.g. "jax" or "main")')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--from-pickle', action='store_true',
                       help='Load Galaxy from pickle (dev_jax only)')
    group.add_argument('--from-params', action='store_true',
                       help='Reconstruct Galaxy from saved params JSON')
    args = parser.parse_args()

    out_dir = os.path.join(REPO_ROOT, 'dev', 'debug_cubes')
    os.makedirs(out_dir, exist_ok=True)

    if args.from_pickle:
        from dysmalpy.fitting import reload_all_fitting

        mpfit_dir = os.path.join(REPO_ROOT, 'demo', 'demo_2D_output')
        print(f"Loading MPFIT model from {mpfit_dir}...")
        gal, results = reload_all_fitting(
            filename_galmodel=os.path.join(mpfit_dir, 'GS4_43501_model.pickle'),
            filename_results=os.path.join(mpfit_dir, 'GS4_43501_mpfit_results.pickle'),
            fit_method='mpfit',
        )

        obs = gal.observations[OBS_NAME]
        print(f"  zcalc_truncate:   {obs.mod_options.zcalc_truncate}")
        print(f"  transform_method: {obs.mod_options.transform_method}")
        print(f"  oversample:       {obs.mod_options.oversample}")

        # Extract and save params for use on other branches
        params = extract_params_from_gal(gal)
        with open(PARAMS_PATH, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Model params saved to {PARAMS_PATH}")

        save_cube_from_gal(gal, obs, args.label, out_dir)

    elif args.from_params:
        if not os.path.exists(PARAMS_PATH):
            raise FileNotFoundError(
                f"Params file not found: {PARAMS_PATH}\n"
                "Run with --from-pickle on dev_jax first to generate it.")

        with open(PARAMS_PATH) as f:
            params = json.load(f)
        print(f"Loaded params from {PARAMS_PATH}")

        gal, obs = build_gal_from_params(params)
        print(f"  Reconstructed Galaxy: z={gal.z}, name={gal.name}")
        print(f"  zcalc_truncate:   {obs.mod_options.zcalc_truncate}")
        print(f"  transform_method: {obs.mod_options.transform_method}")

        save_cube_from_gal(gal, obs, args.label, out_dir)


if __name__ == '__main__':
    main()
