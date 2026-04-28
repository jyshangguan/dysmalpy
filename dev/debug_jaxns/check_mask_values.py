#!/usr/bin/env python
"""
Check mask values in observation data to understand the mask convention.
"""

import sys
sys.path.insert(0, '/home/shangguan/Softwares/my_modules/dysmalpy')

import pickle
import numpy as np
import jax.numpy as jnp

# Try to load observation data from the JAXNS run
print("="*60)
print("Checking Mask Values in Observation Data")
print("="*60)

# First, let's create a simple model to check the mask convention
print("\n1. Creating a simple test observation...")

from dysmalpy.galaxy import Galaxy
from dysmalpy.models.geometry import Geometry
from dysmalpy.observation import Observation

# Simple setup
geom = Geometry(
    inc=60.0, pa=135.0, xshift=0.0, yshift=0.0, vel_shift=0.0,
    **{'transform_method': 'direct'}
)

# Create a simple galaxy
gal = Galaxy(
    geom=geom,
    name='test'
)

# Create synthetic observation data
import numpy as np
from astropy import units as u
from astropy.io import fits

# Create simple data cubes
ny, nx = 27, 27
nspec = 200

# Velocity cube (synthetic)
vel_data = np.zeros((ny, nx))
vel_data[10:20, 10:20] = 100.0  # Valid data in center

# Error cube
vel_err = np.ones((ny, nx)) * 10.0

# Mask (following DYSMALPY convention)
mask = np.zeros((ny, nx))
mask[10:20, 10:20] = 1  # Valid pixels = 1

# Create Observation
from dysmalpy.instrument import Instrument

inst = Instrument(
    fov=[27, 27],
    pixscale=0.1 * u.arcsec,
    beam_size=0.0 * u.arcsec
)

obs = Observation(
    galaxy=gal,
    instrument=inst,
    **{
        'transform_method': 'direct',
        'angle': 'cos',
        'xcenter': None,
        'ycenter': None,
        'oversample': 1,
        'oversize': 1,
        'zcalc_truncate': False,
    }
)

# Set the data manually
obs._vel_obs = jnp.array(vel_data)
obs._vel_err = jnp.array(vel_err)
obs._mask = jnp.array(mask)

print(f"\nTest observation created:")
print(f"  vel_obs shape: {obs.vel_obs.shape}")
print(f"  mask shape: {obs.mask.shape}")
print(f"  mask unique values: {np.unique(obs.mask)}")
print(f"  mask=1 count: {(obs.mask == 1).sum()} (should be valid pixels)")
print(f"  mask=0 count: {(obs.mask == 0).sum()} (should be invalid pixels)")

print(f"\n  vel_obs at mask=1: {obs.vel_obs[obs.mask == 1].sum():.1f} total (should be 100.0)")
print(f"  vel_obs at mask=0: {obs.vel_obs[obs.mask == 0].sum():.1f} total (should be 0.0)")

# Now test with actual GS4_43501 data
print("\n" + "="*60)
print("2. Loading GS4_43501 MPFIT results...")
print("="*60)

try:
    with open('../../demo/demo_2D_output/GS4_43501_mpfit_results.pickle', 'rb') as f:
        mpfit_results = pickle.load(f)

    # The MPFITResults has a plot_results method that can help
    # Let's try to access the galaxy through it
    print("\nTrying to access observation data...")

    # Check attributes
    if hasattr(mpfit_results, 'sampler_results'):
        print("Has sampler_results")

    # The MPFITResults should have the galaxy stored
    # Let's check what we can access
    print(f"\nMPFITResults attributes:")
    for attr in ['input_results', 'galaxy', 'bestfit_parameters']:
        if hasattr(mpfit_results, attr):
            print(f"  Has {attr}")

    # Try calling input_results method
    try:
        input_res = mpfit_results.input_results(
            mpfit_obj=mpfit_results._mpfit_object if hasattr(mpfit_results, '_mpfit_object') else None,
            model=None
        )
        if 'galaxy' in input_res:
            gal = input_res['galaxy']
            obs = list(gal.observations.values())[0]

            print(f"\n✅ Successfully loaded observation from MPFIT results")
            print(f"\nGS4_43501 observation:")
            print(f"  vel_obs shape: {obs.vel_obs.shape}")
            print(f"  mask shape: {obs.mask.shape}")
            print(f"  mask unique values: {np.unique(obs.mask)}")
            print(f"  mask=1 count: {(obs.mask == 1).sum()}")
            print(f"  mask=0 count: {(obs.mask == 0).sum()}")

            print(f"\n  vel_obs at mask=1:")
            vel_at_1 = obs.vel_obs[obs.mask == 1]
            print(f"    Count: {len(vel_at_1)}")
            print(f"    Range: {vel_at_1.min():.2f} to {vel_at_1.max():.2f} km/s")
            print(f"    Mean: {vel_at_1.mean():.2f} km/s")

            print(f"\n  vel_obs at mask=0:")
            vel_at_0 = obs.vel_obs[obs.mask == 0]
            print(f"    Count: {len(vel_at_0)}")
            print(f"    Range: {vel_at_0.min():.2f} to {vel_at_0.max():.2f}")
            print(f"    Mean: {vel_at_0.mean():.2f}")

            # Check if it's -1e6
            if len(vel_at_0) > 0:
                is_neg_1e6 = np.all(np.abs(vel_at_0 + 1e6) < 1e-3)
                print(f"    All equal to -1e6? {is_neg_1e6}")

            print(f"\n  vel_err at mask=0:")
            err_at_0 = obs.vel_err[obs.mask == 0]
            print(f"    Range: {err_at_0.min():.2f} to {err_at_0.max():.2f}")
            print(f"    Mean: {err_at_0.mean():.2f}")

            # Check if any are 99
            has_99 = np.any(err_at_0 == 99)
            print(f"    Has values = 99? {has_99}")

    except Exception as e:
        print(f"\n❌ Error accessing observation: {e}")
        import traceback
        traceback.print_exc()

except FileNotFoundError:
    print(f"\n❌ MPFIT results file not found")
    print(f"   Looking for: ../../demo/demo_2D_output/GS4_43501_mpfit_results.pickle")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("\nDYSMALPY mask convention:")
print("  mask = 1 → valid pixels (include in fitting)")
print("  mask = 0 → invalid pixels (exclude from fitting)")
print("\nInvalid pixels should have:")
print("  vel_obs ≈ -1e6 (sentinel value)")
print("  vel_err ≈ 99 (set in April 15 commit)")
print("\nChi-squared calculation:")
print("  chi2 = ((vel_map - vel_obs) / vel_err)^2 * mask")
print("  ↑ This multiplication by mask should zero out invalid pixels")
print("\nIf mask is correct (0 for invalid), the problem is elsewhere.")
print("If mask is inverted (1 for invalid), this is the bug!")
