#!/usr/bin/env python
"""
Debug script to check chi-squared values and mask handling.

This helps identify if there are numerical issues causing JAXNS weight
evolution problems.
"""

import sys
sys.path.insert(0, '/home/shangguan/Softwares/my_modules/dysmalpy')

import pickle
import jax.numpy as jnp
import numpy as np

# Load the galaxy and observation from a working MPFIT fit
print("Loading MPFIT results...")
with open('../../demo/demo_2D_output/GS4_43501_mpfit_results.pickle', 'rb') as f:
    mpfit_results = pickle.load(f)

# Get the galaxy - input_results is a method that returns a dict
input_results = mpfit_results.input_results(mpfit_obj=None, model=None)
gal = input_results['galaxy']
obs = list(gal.observations.values())[0]

print(f"\nObservation data:")
print(f"  vel_obs shape: {obs.vel_obs.shape}")
print(f"  vel_obs range: {obs.vel_obs.min():.2f} to {obs.vel_obs.max():.2f} km/s")
print(f"  mask shape: {obs.mask.shape}")
print(f"  valid pixels (mask=1): {obs.mask.sum()}")
print(f"  invalid pixels (mask=0): {(obs.mask == 0).sum()}")

# Check invalid pixels
invalid_mask = obs.mask == 0
print(f"\nInvalid pixel analysis:")
print(f"  vel_obs at invalid pixels: min={obs.vel_obs[invalid_mask].min():.2f}, max={obs.vel_obs[invalid_mask].max():.2f}")
print(f"  vel_err at invalid pixels: min={obs.vel_err[invalid_mask].min():.2f}, max={obs.vel_err[invalid_mask].max():.2f}")

# Check if there are any very large values (potential numerical issues)
print(f"\nNumerical checks:")
print(f"  vel_obs has inf: {np.any(np.isinf(obs.vel_obs))}")
print(f"  vel_obs has nan: {np.any(np.isnan(obs.vel_obs))}")
print(f"  vel_err has zeros: {np.any(obs.vel_err == 0)}")
print(f"  vel_err has inf: {np.any(np.isinf(obs.vel_err))}")

# Simulate a model cube with the MPFIT best-fit parameters
print(f"\nSimulating model cube...")
mod_set = gal.model_set
from dysmalpy.observation import Observation

obs_obj = Observation(
    data_none=None,
    galaxy=gal,
    **obs.obs_kwargs
)

cube_model = obs_obj.simulate_cube(mod_set=mod_set, tracing=False)

print(f"  Cube shape: {cube_model.shape}")
print(f"  Cube range: {cube_model.min():.2f} to {cube_model.max():.2f}")

# Now test the Gaussian fitting with hybrid_gd
print(f"\nTesting Gaussian fitting with hybrid_gd method...")
from dysmalpy.fitting.jax_gaussian_fitting import fit_gaussian_cube_jax

spec_arr = obs_obj.spec_arr
mask = obs.mask

try:
    flux_map, vel_map, disp_map = fit_gaussian_cube_jax(
        cube_model=cube_model,
        spec_arr=spec_arr,
        mask=mask,
        method='hybrid_gd'
    )

    print(f"  Vel_map range: {vel_map.min():.2f} to {vel_map.max():.2f} km/s")
    print(f"  Disp_map range: {disp_map.min():.2f} to {disp_map.max():.2f} km/s")
    print(f"  Flux_map range: {flux_map.min():.2f} to {flux_map.max():.2f}")

    # Check for numerical issues
    print(f"\nNumerical checks on fitted maps:")
    print(f"  vel_map has inf: {np.any(np.isinf(vel_map))}")
    print(f"  vel_map has nan: {np.any(np.isnan(vel_map))}")
    print(f"  disp_map has inf: {np.any(np.isinf(disp_map))}")
    print(f"  disp_map has nan: {np.any(np.isnan(disp_map))}")

    # Calculate chi-squared for velocity
    chi2_vel = ((vel_map - obs.vel_obs) / obs.vel_err) ** 2 * mask
    chi2_disp = ((disp_map - obs.disp_obs) / obs.disp_err) ** 2 * mask

    print(f"\nChi-squared analysis:")
    print(f"  Velocity chi2: total={chi2_vel.sum():.2f}, mean={chi2_vel[mask > 0].mean():.2f}")
    print(f"  Dispersion chi2: total={chi2_disp.sum():.2f}, mean={chi2_disp[mask > 0].mean():.2f}")
    print(f"  Total chi2: {(chi2_vel.sum() + chi2_disp.sum()):.2f}")

    # Check for extreme chi-squared values
    print(f"\nExtreme chi-squared values:")
    print(f"  Velocity chi2 > 100: {(chi2_vel > 100).sum()} pixels")
    print(f"  Velocity chi2 > 1000: {(chi2_vel > 1000).sum()} pixels")
    print(f"  Velocity chi2 max: {chi2_vel.max():.2f}")

    # Check which pixels have high chi2
    high_chi2_mask = chi2_vel > 100
    if high_chi2_mask.sum() > 0:
        print(f"\n  High chi2 pixel locations (first 10):")
        coords = np.argwhere(high_chi2_mask)
        for i, (y, x) in enumerate(coords[:10]):
            print(f"    Pixel [{y}, {x}]: chi2={chi2_vel[y, x]:.2f}, "
                  f"vel_fit={vel_map[y, x]:.2f}, vel_obs={obs.vel_obs[y, x]:.2f}, "
                  f"mask={mask[y, x]}")

except Exception as e:
    print(f"ERROR during Gaussian fitting: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Debug complete")
print("="*60)
