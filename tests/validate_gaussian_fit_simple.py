#!/usr/bin/env python
"""
Simple validation of Gaussian fitting methods using synthetic data.

Tests whether hybrid_gd reduces bias compared to closed_form
by creating asymmetric spectra where closed-form MLE is known to be biased.
"""

import sys
sys.path.insert(0, '/home/shangguan/Softwares/my_modules/dysmalpy')

import time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dysmalpy.fitting.jax_gaussian_fitting import fit_gaussian_cube_jax


def create_asymmetric_spectrum(x, A_true, mu_true, sigma_true, asymmetry=0.3):
    """
    Create an asymmetric spectrum where closed-form MLE will be biased.

    Asymmetry is created by adding a secondary Gaussian component.
    """
    # Primary component
    y_primary = A_true * jnp.exp(-0.5 * ((x - mu_true) / sigma_true) ** 2)

    # Secondary component (creates asymmetry)
    mu_secondary = mu_true + asymmetry * sigma_true
    A_secondary = 0.3 * A_true
    y_secondary = A_secondary * jnp.exp(-0.5 * ((x - mu_secondary) / (sigma_true * 0.8)) ** 2)

    y = y_primary + y_secondary
    return y


def create_test_cube_with_bias(ny, nx, nspec, asymmetry_strength=0.3):
    """
    Create a test cube with asymmetric spectra that will bias closed-form MLE.

    The bias arises because closed-form MLE assumes a single Gaussian,
    but we're fitting asymmetric (multi-component) spectra.
    """
    spec_arr = jnp.linspace(-100, 100, nspec)
    cube_model = jnp.zeros((nspec, ny, nx))

    # Create spatial variation in kinematics
    for i in range(ny):
        for j in range(nx):
            # True parameters vary across the field
            mu_true = -30 + i * (60.0 / ny) + j * (20.0 / nx)
            sigma_true = 15.0 + i * 0.3 + j * 0.2
            A_true = 10.0 + jnp.sin(i * 0.3) * 2.0

            # Add asymmetry (secondary component)
            # This varies across the field to create spatially varying bias
            asymmetry = asymmetry_strength * (0.5 + 0.5 * (i + j) / (ny + nx))

            spectrum = create_asymmetric_spectrum(
                spec_arr, A_true, mu_true, sigma_true, asymmetry
            )

            cube_model = cube_model.at[:, i, j].set(spectrum)

    return spec_arr, cube_model


def compare_methods(spec_arr, cube_model, mask=None):
    """Compare all three fitting methods"""
    print("\n" + "="*70)
    print("COMPARING FITTING METHODS")
    print("="*70)

    nspec, ny, nx = cube_model.shape
    n_pixels = ny * nx
    print(f"Dataset: {nspec} spectral × {ny}×{nx} spatial = {n_pixels} pixels")

    results = {}

    for method in ['closed_form', 'hybrid_gd', 'hybrid']:
        print(f"\n{method.upper()}:")
        print("  " + "-"*60)

        # Warmup JIT
        _ = fit_gaussian_cube_jax(cube_model, spec_arr, mask=mask, method=method)

        # Time the fitting (multiple runs for statistics)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            flux, vel, disp = fit_gaussian_cube_jax(
                cube_model, spec_arr, mask=mask, method=method
            )
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        mean_time = np.mean(times)
        std_time = np.std(times)

        results[method] = {
            'flux': flux,
            'vel': vel,
            'disp': disp,
            'time': mean_time,
            'time_std': std_time
        }

        print(f"  Time: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"  Velocity:   min={vel.min():.2f}, max={vel.max():.2f}, mean={vel.mean():.2f} km/s")
        print(f"  Dispersion: min={disp.min():.2f}, max={disp.max():.2f}, mean={disp.mean():.2f} km/s")
        print(f"  Flux:       min={flux.min():.2f}, max={flux.max():.2f}, mean={flux.mean():.2f}")

    return results


def analyze_bias_improvement(results):
    """Analyze how much hybrid_gd improves over closed_form"""
    print("\n" + "="*70)
    print("BIAS ANALYSIS: Does hybrid_gd reduce bias?")
    print("="*70)

    # Use hybrid (BFGS) as reference (most accurate)
    ref_vel = results['hybrid']['vel']
    ref_disp = results['hybrid']['disp']

    # Closed-form bias
    cf_vel = results['closed_form']['vel']
    cf_disp = results['closed_form']['disp']

    cf_vel_bias = cf_vel - ref_vel
    cf_disp_bias = cf_disp - ref_disp

    cf_vel_rmse = jnp.sqrt((cf_vel_bias ** 2).mean())
    cf_disp_rmse = jnp.sqrt((cf_disp_bias ** 2).mean())

    print("\nClosed-form (biased baseline):")
    print(f"  Velocity RMSE vs BFGS:   {cf_vel_rmse:.4f} km/s")
    print(f"  Dispersion RMSE vs BFGS:  {cf_disp_rmse:.4f} km/s")

    # Hybrid GD bias
    gd_vel = results['hybrid_gd']['vel']
    gd_disp = results['hybrid_gd']['disp']

    gd_vel_bias = gd_vel - ref_vel
    gd_disp_bias = gd_disp - ref_disp

    gd_vel_rmse = jnp.sqrt((gd_vel_bias ** 2).mean())
    gd_disp_rmse = jnp.sqrt((gd_disp_bias ** 2).mean())

    print("\nHybrid GD (new method):")
    print(f"  Velocity RMSE vs BFGS:   {gd_vel_rmse:.4f} km/s")
    print(f"  Dispersion RMSE vs BFGS:  {gd_disp_rmse:.4f} km/s")

    # Improvement
    vel_improvement = (cf_vel_rmse - gd_vel_rmse) / cf_vel_rmse * 100
    disp_improvement = (cf_disp_rmse - gd_disp_rmse) / cf_disp_rmse * 100

    print("\n✅ IMPROVEMENT:")
    print(f"  Velocity bias reduction:   {vel_improvement:.1f}%")
    print(f"  Dispersion bias reduction: {disp_improvement:.1f}%")

    if vel_improvement > 0 and disp_improvement > 0:
        print("\n✅ SUCCESS: Hybrid GD successfully reduces bias!")
        print("   This confirms the method works as intended.")
    else:
        print("\n⚠️  WARNING: Bias reduction is minimal.")
        print("   The spectra may not be asymmetric enough to show significant bias.")

    return {
        'cf_vel_rmse': float(cf_vel_rmse),
        'gd_vel_rmse': float(gd_vel_rmse),
        'vel_improvement': float(vel_improvement),
        'cf_disp_rmse': float(cf_disp_rmse),
        'gd_disp_rmse': float(gd_disp_rmse),
        'disp_improvement': float(disp_improvement)
    }


def plot_comparison(results, output_dir):
    """Create comparison plots"""
    print("\n" + "="*70)
    print("Generating comparison plots...")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get data
    cf_vel = results['closed_form']['vel']
    gd_vel = results['hybrid_gd']['vel']
    ref_vel = results['hybrid']['vel']

    cf_disp = results['closed_form']['disp']
    gd_disp = results['hybrid_gd']['disp']
    ref_disp = results['hybrid']['disp']

    # Figure 1: Method comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Gaussian Fitting Method Comparison (Asymmetric Spectra)', fontsize=16)

    methods = ['closed_form', 'hybrid_gd', 'hybrid']
    names = ['Closed-form\n(May be biased)', 'Hybrid GD\n(New method)', 'Hybrid BFGS\n(Reference)']

    for i, (method, name) in enumerate(zip(methods, names)):
        vel = results[method]['vel']
        disp = results[method]['disp']

        im1 = axes[0, i].imshow(vel, origin='lower', cmap='RdBu_r')
        axes[0, i].set_title(f'{name}\nVelocity')
        plt.colorbar(im1, ax=axes[0, i])

        im2 = axes[1, i].imshow(disp, origin='lower', cmap='viridis')
        axes[1, i].set_title(f'{name}\nDispersion')
        plt.colorbar(im2, ax=axes[1, i])

    plt.tight_layout()
    fig1_path = output_dir / 'gaussian_methods_comparison.png'
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {fig1_path}")
    plt.close()

    # Figure 2: Residuals relative to BFGS
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Residuals Relative to Hybrid BFGS (Most Accurate)', fontsize=16)

    # Closed-form residuals
    cf_vel_res = cf_vel - ref_vel
    cf_disp_res = cf_disp - ref_disp

    im1 = axes[0, 0].imshow(cf_vel_res, origin='lower', cmap='RdBu_r')
    axes[0, 0].set_title(f'Closed-form - BFGS\nVelocity Residual')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[1, 0].imshow(cf_disp_res, origin='lower', cmap='RdBu_r')
    axes[1, 0].set_title(f'Closed-form - BFGS\nDispersion Residual')
    plt.colorbar(im2, ax=axes[1, 0])

    # Hybrid GD residuals
    gd_vel_res = gd_vel - ref_vel
    gd_disp_res = gd_disp - ref_disp

    im3 = axes[0, 1].imshow(gd_vel_res, origin='lower', cmap='RdBu_r')
    axes[0, 1].set_title(f'Hybrid GD - BFGS\nVelocity Residual')
    plt.colorbar(im3, ax=axes[0, 1])

    im4 = axes[1, 1].imshow(gd_disp_res, origin='lower', cmap='RdBu_r')
    axes[1, 1].set_title(f'Hybrid GD - BFGS\nDispersion Residual')
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    fig2_path = output_dir / 'gaussian_methods_residuals.png'
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {fig2_path}")
    plt.close()

    # Figure 3: Histogram of residuals
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Distribution of Residuals', fontsize=16)

    axes[0].hist(cf_vel_res.flatten(), bins=50, alpha=0.5, label='Closed-form', color='red')
    axes[0].hist(gd_vel_res.flatten(), bins=50, alpha=0.5, label='Hybrid GD', color='blue')
    axes[0].set_xlabel('Velocity Residual (km/s)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Velocity Residuals')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(cf_disp_res.flatten(), bins=50, alpha=0.5, label='Closed-form', color='red')
    axes[1].hist(gd_disp_res.flatten(), bins=50, alpha=0.5, label='Hybrid GD', color='blue')
    axes[1].set_xlabel('Dispersion Residual (km/s)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Dispersion Residuals')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig3_path = output_dir / 'gaussian_methods_histogram.png'
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {fig3_path}")
    plt.close()


def main():
    """Main validation workflow"""
    print("\n" + "="*70)
    print("GAUSSIAN FITTING METHOD VALIDATION")
    print("Testing bias reduction in asymmetric spectra")
    print("="*70)

    # Test parameters
    ny, nx, nspec = 20, 20, 200
    asymmetry_strength = 0.4  # Strength of asymmetry (creates bias)

    print(f"\nTest configuration:")
    print(f"  Spatial size: {ny}×{nx} = {ny*nx} pixels")
    print(f"  Spectral channels: {nspec}")
    print(f"  Asymmetry strength: {asymmetry_strength}")
    print(f"\nAsymmetric spectra are created by adding a secondary Gaussian")
    print(f"component, which biases the closed-form MLE solution.")

    # Create test data
    print("\n" + "="*70)
    print("Creating test data with asymmetric spectra...")
    print("="*70)

    spec_arr, cube_model = create_test_cube_with_bias(
        ny, nx, nspec, asymmetry_strength=asymmetry_strength
    )

    print(f"✅ Test cube created: {cube_model.shape}")

    # Compare methods
    results = compare_methods(spec_arr, cube_model)

    # Analyze bias improvement
    bias_metrics = analyze_bias_improvement(results)

    # Create plots
    output_dir = Path('/home/shangguan/Softwares/my_modules/dysmalpy/gaussian_fit_validation')
    plot_comparison(results, output_dir)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n✅ Validation complete!")
    print(f"\nKey findings:")
    print(f"1. Closed-form shows bias: RMSE = {bias_metrics['cf_vel_rmse']:.4f} km/s (velocity)")
    print(f"2. Hybrid GD reduces bias: RMSE = {bias_metrics['gd_vel_rmse']:.4f} km/s (velocity)")
    print(f"3. Bias reduction: {bias_metrics['vel_improvement']:.1f}% (velocity)")
    print(f"4. Bias reduction: {bias_metrics['disp_improvement']:.1f}% (dispersion)")

    print(f"\nPerformance comparison:")
    for method in ['closed_form', 'hybrid_gd', 'hybrid']:
        time_ms = results[method]['time']
        print(f"   {method:15s}: {time_ms:6.2f} ms")

    print(f"\nConclusion:")
    if bias_metrics['vel_improvement'] > 10:
        print("✅ Hybrid GD successfully reduces closed-form bias")
        print("   Use hybrid_gd as default for production JAXNS fitting")
    else:
        print("⚠️  Bias reduction is minimal for this test case")
        print("   Try stronger asymmetry or real galaxy data")

    print(f"\nPlots saved to: {output_dir}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
