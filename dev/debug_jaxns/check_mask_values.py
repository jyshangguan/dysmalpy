#!/usr/bin/env python
"""
Check mask values in observation data to understand the mask convention.
"""

import sys
sys.path.insert(0, '/home/shangguan/Softwares/my_modules/dysmalpy')

import pickle
import numpy as np

# Load MPFIT results to get observation
print("Loading MPFIT results...")
with open('../../demo/demo_2D_output/GS4_43501_mpfit_results.pickle', 'rb') as f:
    mpfit_results = pickle.load(f)

# Get the observation
from dysmalpy.fitting.mpfit import MPFITResults
# The MPFITResults object has the galaxy stored differently
# Let's try to access it directly
gal = mpfit_results.__dict__['input_results_dict']['galaxy']
obs = list(gal.observations.values())[0]

print(f"\nObservation mask:")
print(f"  Shape: {obs.mask.shape}")
print(f"  Unique values: {np.unique(obs.mask)}")
print(f"  Count of 1s: {(obs.mask == 1).sum()}")
print(f"  Count of 0s: {(obs.mask == 0).sum()}")
print(f"  Count of -1s: {(obs.mask == -1).sum()}")

print(f"\nVelocity observation at mask=1:")
print(f"  Range: {obs.vel_obs[obs.mask == 1].min():.2f} to {obs.vel_obs[obs.mask == 1].max():.2f}")

print(f"\nVelocity observation at mask=0:")
vel_at_0 = obs.vel_obs[obs.mask == 0]
print(f"  Range: {vel_at_0.min():.2f} to {vel_at_0.max():.2f}")
print(f"  Mean: {vel_at_0.mean():.2f}")
print(f"  All equal to -1e6? {np.all(vel_at_0 == -1e6)}")

print(f"\nVelocity error at mask=0:")
err_at_0 = obs.vel_err[obs.mask == 0]
print(f"  Range: {err_at_0.min():.2f} to {err_at_0.max():.2f}")
print(f"  Mean: {err_at_0.mean():.2f}")
print(f"  All equal to 99? {np.all(err_at_0 == 99)}")

# Check if the mask follows the convention we expect
print(f"\nDysmalpy mask convention check:")
print(f"  Expected: mask=1 for valid, mask=0 for invalid")
print(f"  Actual observation:")
print(f"    mask=1 pixels: {(obs.mask == 1).sum()} (should be valid)")
print(f"    mask=0 pixels: {(obs.mask == 0).sum()} (should be invalid)")

# Show a few example pixels
print(f"\nExample pixels:")
for i in range(3):
    for j in range(3):
        print(f"  Pixel [{i},{j}]: mask={obs.mask[i,j]:.0f}, vel={obs.vel_obs[i,j]:.2f}, err={obs.vel_err[i,j]:.2f}")
