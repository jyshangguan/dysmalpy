#!/usr/bin/env python
"""
Debug script to analyze JAXNS likelihood and chi-squared values.

Checks for numerical issues that might cause abnormal weight evolution.
"""

import sys
sys.path.insert(0, '/home/shangguan/Softwares/my_modules/dysmalpy')

import pickle
import jax.numpy as jnp
import numpy as np

print("="*60)
print("JAXNS Likelihood Debug Analysis")
print("="*60)

# Load JAXNS sampler results
print("\nLoading JAXNS sampler results...")
with open('../../demo/demo_2D_output_jaxns/GS4_43501_jaxns_sampler_results.pickle', 'rb') as f:
    sampler = pickle.load(f)

print(f"Sampler type: {type(sampler)}")

# Check what's available in the sampler results
print(f"\nSampler attributes:")
for attr in dir(sampler):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Load the full results to get the galaxy
print("\nLoading JAXNS results...")
with open('../../demo/demo_2D_output_jaxns/GS4_43501_jaxns_results.pickle', 'rb') as f:
    results = pickle.load(f)

# The results should have the galaxy - let's try to access it
print(f"Results type: {type(results)}")

# Try to get the model through set_model
try:
    results.set_model()
    print("✅ Model set successfully")
except Exception as e:
    print(f"❌ Error setting model: {e}")

# Check the log likelihood values
if hasattr(sampler, 'log_L_samples'):
    log_L = sampler.log_L_samples
    print(f"\nLog likelihood samples:")
    print(f"  Shape: {log_L.shape if hasattr(log_L, 'shape') else len(log_L)}")
    print(f"  Min: {log_L.min() if hasattr(log_L, 'min') else log_L}")
    print(f"  Max: {log_L.max() if hasattr(log_L, 'max') else log_L}")
    print(f"  Mean: {log_L.mean() if hasattr(log_L, 'mean') else np.mean(log_L)}")

# Check for any NaN or Inf
if hasattr(sampler, 'log_L_samples'):
    log_L_array = np.array(sampler.log_L_samples)
    print(f"\nNumerical checks:")
    print(f"  Has NaN: {np.any(np.isnan(log_L_array))}")
    print(f"  Has Inf: {np.any(np.isinf(log_L_array))}")
    print(f"  Has -Inf: {np.any(np.isneginf(log_L_array))}")

    # Check the evolution of log likelihood
    if len(log_L_array) > 100:
        print(f"\nLog likelihood evolution (first 100):")
        print(f"  First 10: {log_L_array[:10]}")
        print(f"  Last 10: {log_L_array[-10:]}")

        # Check if it's monotonically increasing (should be for nested sampling)
        is_monotonic = np.all(np.diff(log_L_array) >= -1e-3)  # Allow small numerical noise
        print(f"\n  Is monotonically increasing: {is_monotonic}")

        if not is_monotonic:
            # Find where it decreases
            decreases = np.where(np.diff(log_L_array) < -1e-3)[0]
            print(f"  Number of decreases: {len(decreases)}")
            if len(decreases) > 0:
                print(f"  First decrease at index: {decreases[0]}")
                print(f"  Largest drop: {np.diff(log_L_array)[decreases].min():.2f}")

# Check the evidence
if hasattr(sampler, 'log_Z'):
    print(f"\nEvidence (log_Z):")
    print(f"  Value: {sampler.log_Z}")
    print(f"  Uncertainty: {sampler.log_Z_uncertainty if hasattr(sampler, 'log_Z_uncertainty') else 'N/A'}")

print("\n" + "="*60)
print("Debug complete")
print("="*60)
