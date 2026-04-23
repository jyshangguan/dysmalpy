#!/usr/bin/env python
"""Debug script: compare active-only vs full cube path for 2D kinematic maps.

Loads the MPFIT best-fit model (saved with current code), re-evaluates model data
through both the active-only path (default, zcalc_truncate=True) and the full
cube path (zcalc_truncate=False), then compares velocity/dispersion maps and
chi-squared values.

Usage:
    JAX_PLATFORMS=cpu python dev/debug_reproduce_mcmc_residuals.py
"""

import os
import copy
import numpy as np
import astropy.units as u

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OBS_NAME = 'OBS'


def compute_chisq_2d(gal, obs_name=OBS_NAME):
    """Compute raw chi-sq for a 2D observation (velocity + dispersion).

    Same formula as mpfit_chisq in dysmalpy/fitting/mpfit.py (no log term,
    no oversample correction).
    """
    obs = gal.observations[obs_name]
    chisq_sum = 0.0
    ndof_data = 0

    for fit_type in ['velocity', 'dispersion']:
        msk = obs.data.mask
        dat = obs.data.data[fit_type][msk]
        mod = obs.model_data.data[fit_type][msk]
        err = obs.data.error[fit_type][msk]

        chisq_raw = ((dat - mod) / err) ** 2
        chisq_sum += chisq_raw.sum()
        ndof_data += np.sum(msk)

    return chisq_sum, ndof_data


def get_oversample_factor(gal, obs_name=OBS_NAME):
    """Compute oversample_factor_chisq = (PSF_FWHM / pixscale)^2."""
    obs = gal.observations[obs_name]
    PSF_FWHM = obs.instrument.beam.major.value
    pixscale = obs.instrument.pixscale.value
    return (PSF_FWHM / pixscale) ** 2


def map_stats(arr):
    """Return dict of statistics for a 2D map (ignoring zero/NaN pixels).

    Handles plain numpy arrays, astropy Quantity objects, and masked arrays.
    """
    # Strip Quantity / masked array wrappers
    if hasattr(arr, 'value'):
        arr = arr.value
    if hasattr(arr, 'compressed'):
        arr = arr.compressed()
    else:
        arr = np.asarray(arr, dtype=float)

    valid = arr[~np.isnan(arr) & (arr != 0)]
    if len(valid) == 0:
        return {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'nvalid': 0}
    return {
        'min': np.min(valid), 'max': np.max(valid),
        'mean': np.mean(valid), 'std': np.std(valid),
        'nvalid': len(valid),
    }


def print_map_stats(label, arr):
    s = map_stats(arr)
    print(f"    {label:20s}: min={s['min']:8.2f}  max={s['max']:8.2f}  "
          f"mean={s['mean']:8.2f}  std={s['std']:8.2f}  ({s['nvalid']} px)")


def main():
    mpfit_dir = os.path.join(REPO_ROOT, 'demo', 'demo_2D_output')
    mpfit_model_pickle = os.path.join(mpfit_dir, 'GS4_43501_model.pickle')
    mpfit_results_pickle = os.path.join(mpfit_dir, 'GS4_43501_mpfit_results.pickle')

    from dysmalpy.fitting import reload_all_fitting
    from dysmalpy.fitting.base import chisq_eval

    print("=" * 70)
    print("  Active-only vs Full cube path comparison")
    print("=" * 70)

    # ==============================================================
    # Part A: Load MPFIT model, evaluate with active path
    # ==============================================================
    print("\n>>> Loading MPFIT model + results...")
    gal, results = reload_all_fitting(
        filename_galmodel=mpfit_model_pickle,
        filename_results=mpfit_results_pickle,
        fit_method='mpfit',
    )

    obs = gal.observations[OBS_NAME]
    print(f"  zcalc_truncate:       {obs.mod_options.zcalc_truncate}")
    print(f"  transform_method:     {obs.mod_options.transform_method}")
    print(f"  oversample:           {obs.mod_options.oversample}")
    print(f"  PSF FWHM:             {obs.instrument.beam.major}")
    print(f"  pixscale:             {obs.instrument.pixscale}")

    osf = get_oversample_factor(gal)
    print(f"  oversample_factor:    {osf:.2f}")

    print(f"\n  MPFIT saved bestfit_redchisq = {results.bestfit_redchisq:.4f}")

    # --- Check saved model data from pickle ---
    print("\n>>> Checking saved model_data from pickle (before re-evaluation)...")
    if obs.model_data is not None:
        saved_vel = obs.model_data.data.get('velocity')
        saved_disp = obs.model_data.data.get('dispersion')
        if saved_vel is not None:
            n_nan_vel = np.sum(np.isnan(saved_vel))
            n_zero_vel = np.sum(saved_vel[~np.isnan(saved_vel)] == 0)
            print(f"    Saved velocity: {n_nan_vel} NaN, {n_zero_vel} zero, "
                  f"{np.sum(~np.isnan(saved_vel) & (saved_vel != 0))} valid")
            if np.any(~np.isnan(saved_vel) & (saved_vel != 0)):
                v = saved_vel[~np.isnan(saved_vel) & (saved_vel != 0)]
                print(f"      range: [{np.min(v):.2f}, {np.max(v):.2f}] km/s")
        else:
            print("    Saved velocity: None (3D data)")
    else:
        print("    obs.model_data is None")

    # --- Active path (default) ---
    print("\n>>> Evaluating with ACTIVE path (zcalc_truncate=True)...")
    # Debug: inspect the cube before create_model_data clobbers it
    # First call create_model_data
    gal.create_model_data()

    # Check the cube
    print("\n>>> Inspecting model cube after create_model_data()...")
    cube = obs.model_cube
    if cube is not None:
        cube_data = cube.data
        print(f"    Cube shape: {cube_data.shape}")
        print(f"    Cube spectral axis: {cube_data.spectral_axis[:5]}... (first 5)")
        print(f"    Cube spectral axis unit: {cube_data.spectral_axis.unit}")
        print(f"    Cube peak: {np.max(np.abs(cube_data)):6e}")
        print(f"    Cube has NaN: {np.any(np.isnan(cube_data.unmasked_data[:].value))}")
        print(f"    Cube mid-slice max: {np.max(np.abs(cube_data[:, cube_data.shape[1]//2, cube_data.shape[2]//2].value)):.6e}")
    else:
        print("    obs.model_cube is None!")

    # Check moment maps
    print("\n>>> Checking moment maps from cube...")
    try:
        mom0 = cube.data.moment0().to(u.km/u.s).value
        mom1 = cube.data.moment1().to(u.km/u.s).value
        mom2 = cube.data.linewidth_sigma().to(u.km/u.s).value
        print(f"    moment0: min={np.nanmin(mom0):.6e} max={np.nanmax(mom0):.6e} "
              f"NaN={np.sum(np.isnan(mom0))}")
        print(f"    moment1: min={np.nanmin(mom1):.2f} max={np.nanmax(mom1):.2f} "
              f"NaN={np.sum(np.isnan(mom1))}")
        print(f"    linewidth_sigma: min={np.nanmin(mom2):.2f} max={np.nanmax(mom2):.2f} "
              f"NaN={np.sum(np.isnan(mom2))}")
    except Exception as e:
        print(f"    Moment calculation failed: {e}")

    # Check 2D model data
    print("\n>>> Checking 2D model_data...")
    md = obs.model_data
    if md is not None:
        for key in ['velocity', 'dispersion', 'flux']:
            if key in md.data:
                arr = md.data[key]
                n_nan = np.sum(np.isnan(arr))
                print(f"    {key}: min={np.nanmin(arr):.4f} max={np.nanmax(arr):.4f} "
                      f"NaN={n_nan}/{arr.size}")
    else:
        print("    obs.model_data is None!")
    vel_active = obs.model_data.data['velocity'].copy()
    disp_active = obs.model_data.data['dispersion'].copy()
    chisq_active, ndof_active = compute_chisq_2d(gal)
    chisq_builtin_active = chisq_eval(gal)
    peak_active = np.max(np.abs(obs.model_cube.data))

    print_map_stats("velocity (active)", vel_active)
    print_map_stats("dispersion (active)", disp_active)
    print(f"    chi2 (active, manual):   {chisq_active:.4f}  ({ndof_active} masked px)")
    print(f"    chi2 (active, builtin): {chisq_builtin_active:.4f}")
    print(f"    chi2/osf:               {chisq_active/osf:.4f}")
    print(f"    cube peak flux:         {peak_active:.6e}")

    # ==============================================================
    # Part B: Same parameters, force non-active path
    # ==============================================================
    print("\n>>> Evaluating with FULL path (zcalc_truncate=False)...")
    obs.mod_options.zcalc_truncate = False

    gal.create_model_data()
    vel_full = obs.model_data.data['velocity'].copy()
    disp_full = obs.model_data.data['dispersion'].copy()
    chisq_full, ndof_full = compute_chisq_2d(gal)
    chisq_builtin_full = chisq_eval(gal)
    peak_full = np.max(np.abs(obs.model_cube.data))

    print_map_stats("velocity (full)", vel_full)
    print_map_stats("dispersion (full)", disp_full)
    print(f"    chi2 (full, manual):     {chisq_full:.4f}  ({ndof_full} masked px)")
    print(f"    chi2 (full, builtin):   {chisq_builtin_full:.4f}")
    print(f"    chi2/osf:               {chisq_full/osf:.4f}")
    print(f"    cube peak flux:         {peak_full:.6e}")

    # ==============================================================
    # Part C: Compare
    # ==============================================================
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    vel_diff = vel_active - vel_full
    disp_diff = disp_active - disp_full

    # Compare on pixels where either map is non-zero and both are finite
    finite_mask = np.isfinite(vel_active) & np.isfinite(vel_full) & \
                  np.isfinite(disp_active) & np.isfinite(disp_full)
    valid = finite_mask & ((vel_active != 0) | (vel_full != 0))
    n_valid = np.sum(valid)

    if n_valid > 0:
        print(f"\n  Velocity map differences ({n_valid} non-zero, finite pixels):")
        print(f"    max |diff| = {np.max(np.abs(vel_diff[valid])):.6f} km/s")
        print(f"    mean |diff| = {np.mean(np.abs(vel_diff[valid])):.6f} km/s")
        vel_full_valid = vel_full[valid]
        if np.any(vel_full_valid != 0):
            rel = np.abs(vel_diff[valid]) / (np.abs(vel_full_valid) + 1e-10)
            rel = rel[vel_full_valid != 0]
            print(f"    mean |rel diff| = {np.mean(rel):.6f}")

        print(f"\n  Dispersion map differences ({n_valid} non-zero, finite pixels):")
        print(f"    max |diff| = {np.max(np.abs(disp_diff[valid])):.6f} km/s")
        print(f"    mean |diff| = {np.mean(np.abs(disp_diff[valid])):.6f} km/s")
    else:
        print("\n  No non-zero, finite pixels to compare!")

    print(f"\n  Cube peak flux:  active={peak_active:.6e}  full={peak_full:.6e}  "
          f"ratio={peak_active / peak_full:.6f}" if peak_full > 0 else
          f"\n  Cube peak flux:  active={peak_active:.6e}  full={peak_full:.6e}")

    print(f"\n  {'Path':<25} {'Raw chi2':>12} {'chi2/osf':>12}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12}")
    print(f"  {'Active (zcalc=True)':<25} {chisq_active:>12.4f} {chisq_active/osf:>12.4f}")
    print(f"  {'Full (zcalc=False)':<25} {chisq_full:>12.4f} {chisq_full/osf:>12.4f}")
    print(f"  {'MPFIT saved':<25} {results.bestfit_redchisq:>12.4f} {'(red_chisq)':>12}")

    # ==============================================================
    # Diagnosis
    # ==============================================================
    max_vel_diff = np.max(np.abs(vel_diff)) if n_valid > 0 else 0
    max_disp_diff = np.max(np.abs(disp_diff)) if n_valid > 0 else 0
    chi2_diff = abs(chisq_active - chisq_full)

    print(f"\n  DIAGNOSIS:")
    if max_vel_diff < 0.01 and max_disp_diff < 0.01:
        print("  [OK] Active and full paths produce IDENTICAL maps.")
        print("       The active-only cube path is correct.")
    elif max_vel_diff < 1.0 and max_disp_diff < 1.0:
        print("  [WARN] Active and full paths differ slightly (< 1 km/s).")
        print("         This may be floating-point precision difference.")
    else:
        print(f"  [BUG] Active and full paths differ significantly!")
        print(f"        max velocity diff:   {max_vel_diff:.4f} km/s")
        print(f"        max dispersion diff: {max_disp_diff:.4f} km/s")
        print(f"        The active-only path has a bug in velocity/dispersion.")

    if chi2_diff > 0.01:
        print(f"  [INFO] Chi-squared differs by {chi2_diff:.4f} between paths.")
        print("         This is expected when zcalc_truncate excludes marginal z-slices.")
        print("         Both chi2 values should match MPFIT's bestfit_chisq independently.")

    print(f"\n  oversample_factor_chisq = {osf:.2f}  (= (PSF_FWHM/pixscale)^2)")


if __name__ == '__main__':
    main()
