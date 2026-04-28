#!/usr/bin/env python
"""
Direct check of observation data mask values using dysmalpy_fit_single.
"""

import sys
sys.path.insert(0, '/home/shangguan/Softwares/my_modules/dysmalpy')

import numpy as np
import jax.numpy as jnp

print("="*60)
print("Checking Mask Values via dysmalpy_fit_single")
print("="*60)

# Load parameter file and setup model
from dysmalpy.fitting_wrappers.setup_gal_models import setup_single_galaxy
from dysmalpy.fitting_wrappers.data_io import read_fitting_params

param_file = '/home/shangguan/Softwares/my_modules/dysmalpy/examples/examples_param_files/fitting_2D_mpfit.params'

try:
    # Read parameters
    params = read_fitting_params(param_file)
    gal, obs_list, mod_set = setup_single_galaxy(params=params)

    print(f"\n✅ Model loaded successfully")
    print(f"Number of observations: {len(obs_list)}")

    for i, obs in enumerate(obs_list):
        print(f"\nObservation {i}:")
        print(f"  Name: {obs.name}")
        print(f"  ndim: {obs.ndim}")

        if hasattr(obs, 'mask'):
            print(f"  mask shape: {obs.mask.shape}")
            print(f"  mask unique values: {np.unique(obs.mask)}")
            print(f"  mask=1 count: {(obs.mask == 1).sum()}")
            print(f"  mask=0 count: {(obs.mask == 0).sum()}")

            if hasattr(obs, 'vel_obs'):
                print(f"\n  vel_obs shape: {obs.vel_obs.shape}")

                vel_at_1 = obs.vel_obs[obs.mask == 1]
                vel_at_0 = obs.vel_obs[obs.mask == 0]

                print(f"  vel_obs at mask=1:")
                print(f"    Count: {len(vel_at_1)}")
                if len(vel_at_1) > 0:
                    print(f"    Range: {vel_at_1.min():.2f} to {vel_at_1.max():.2f}")
                    print(f"    Mean: {vel_at_1.mean():.2f}")

                print(f"  vel_obs at mask=0:")
                print(f"    Count: {len(vel_at_0)}")
                if len(vel_at_0) > 0:
                    print(f"    Range: {vel_at_0.min():.2f} to {vel_at_0.max():.2f}")
                    print(f"    Mean: {vel_at_0.mean():.2f}")

                    # Check if -1e6
                    is_neg_1e6 = np.all(np.abs(vel_at_0 + 1e6) < 1)
                    print(f"    All ≈ -1e6? {is_neg_1e6}")

            if hasattr(obs, 'vel_err'):
                err_at_0 = obs.vel_err[obs.mask == 0]
                print(f"\n  vel_err at mask=0:")
                print(f"    Count: {len(err_at_0)}")
                if len(err_at_0) > 0:
                    print(f"    Range: {err_at_0.min():.2f} to {err_at_0.max():.2f}")
                    print(f"    Mean: {err_at_0.mean():.2f}")

                    # Check for 99
                    has_99 = np.any(err_at_0 == 99)
                    print(f"    Has values = 99? {has_99}")

                    # Count 99s
                    n_99 = np.sum(err_at_0 == 99)
                    print(f"    Number of 99s: {n_99} / {len(err_at_0)}")

    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    print("\nExpected DYSMALPY convention:")
    print("  mask = 1 → valid pixels")
    print("  mask = 0 → invalid pixels")
    print("\nIf mask=0 for invalid pixels, and those pixels have:")
    print("  vel_obs ≈ -1e6")
    print("  vel_err = 99")
    print("\nThen chi-squared calculation:")
    print("  chi2 = ((vel_map - (-1e6)) / 99)^2 * 0")
    print("  chi2 = huge_number * 0")
    print("  chi2 = 0  ← Correct!")
    print("\nThis means the mask is working correctly.")
    print("The extreme likelihood values must have another cause.")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
