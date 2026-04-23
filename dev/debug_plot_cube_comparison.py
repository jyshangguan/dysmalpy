#!/usr/bin/env python
"""Compare moment maps between two DysmalPy branches (e.g. main vs jax).

Usage:
    JAX_PLATFORMS=cpu python dev/debug_plot_cube_comparison.py
    JAX_PLATFORMS=cpu python dev/debug_plot_cube_comparison.py \
        --main dev/debug_cubes/main_cube.npz --jax dev/debug_cubes/jax_cube.npz

Produces a 3x3 panel figure:
    | Main mom0 | JAX mom0 | Residual (JAX - Main) |
    | Main mom1 | JAX mom1 | Residual (JAX - Main) |
    | Main mom2 | JAX mom2 | Residual (JAX - Main) |
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MAIN = os.path.join(REPO_ROOT, 'dev', 'debug_cubes', 'main_cube.npz')
DEFAULT_JAX = os.path.join(REPO_ROOT, 'dev', 'debug_cubes', 'jax_cube.npz')


def main():
    parser = argparse.ArgumentParser(
        description='Plot moment map comparison between two branches')
    parser.add_argument('--main', type=str, default=DEFAULT_MAIN,
                        help='Path to main branch NPZ file')
    parser.add_argument('--jax', type=str, default=DEFAULT_JAX,
                        help='Path to JAX branch NPZ file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output figure path (default: auto)')
    args = parser.parse_args()

    print(f"Loading main: {args.main}")
    d_main = np.load(args.main, allow_pickle=True)
    print(f"Loading jax:  {args.jax}")
    d_jax = np.load(args.jax, allow_pickle=True)

    print(f"  Main dtype: {d_main['dtype']}")
    print(f"  JAX  dtype: {d_jax['dtype']}")
    print(f"  Main cube shape: {d_main['cube'].shape}")
    print(f"  JAX  cube shape: {d_jax['cube'].shape}")

    mom_names = ['mom0', 'mom1', 'mom2']
    mom_labels = ['Moment 0 (Integrated Flux)', 'Moment 1 (Velocity)',
                  'Moment 2 (Dispersion)']
    cmaps_data = ['inferno', 'RdBu_r', 'magma']

    # --- Cube-level comparison ---
    cube_main = d_main['cube']
    cube_jax = d_jax['cube']
    cube_residual = cube_jax - cube_main
    cube_max_abs = np.max(np.abs(cube_residual))

    print(f"\n  Cube residual (JAX - Main):")
    print(f"    max |diff| = {cube_max_abs:.6e}")
    print(f"    mean |diff| = {np.mean(np.abs(cube_residual)):.6e}")
    print(f"    dtype match: {d_main['dtype'] == d_jax['dtype']}")

    # --- Moment map comparison ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(f'Moment Map Comparison: main ({d_main["dtype"]}) vs '
                 f'JAX ({d_jax["dtype"]})',
                 fontsize=14, fontweight='bold')

    for row, (mom_key, mom_label) in enumerate(zip(mom_names, mom_labels)):
        m_main = d_main[mom_key]
        m_jax = d_jax[mom_key]

        if m_main.shape != m_jax.shape:
            print(f"  WARNING: Shape mismatch for {mom_key}: "
                  f"{m_main.shape} vs {m_jax.shape}")
            continue

        residual = m_jax - m_main

        # Mask zero pixels for display
        mask = (m_main != 0) | (m_jax != 0)
        if not np.any(mask):
            print(f"  WARNING: No non-zero pixels for {mom_key}")
            continue

        # Compute color ranges
        vmax_main = np.max(np.abs(m_main[mask]))
        vmax_jax = np.max(np.abs(m_jax[mask]))
        vmax_res = np.max(np.abs(residual[mask]))
        vmax = max(vmax_main, vmax_jax)

        if mom_key == 'mom0':
            vmin_main, vmax_main = 0, vmax
            vmin_jax, vmax_jax = 0, vmax
        elif mom_key == 'mom1':
            vmin_main = -vmax
            vmin_jax = -vmax
        else:  # mom2
            vmin_main, vmax_main = 0, vmax
            vmin_jax, vmax_jax = 0, vmax

        # Main
        im0 = axes[row, 0].imshow(m_main, origin='lower',
                                   cmap=cmaps_data[row],
                                   vmin=vmin_main, vmax=vmax_main)
        axes[row, 0].set_title(f'Main {mom_label}', fontsize=10)
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # JAX
        im1 = axes[row, 1].imshow(m_jax, origin='lower',
                                   cmap=cmaps_data[row],
                                   vmin=vmin_jax, vmax=vmax_jax)
        axes[row, 1].set_title(f'JAX {mom_label}', fontsize=10)
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Residual
        if vmax_res > 0:
            vmax_r = vmax_res * 1.2
        else:
            vmax_r = 1e-10
        im2 = axes[row, 2].imshow(residual, origin='lower', cmap='seismic',
                                   vmin=-vmax_r, vmax=vmax_r)
        axes[row, 2].set_title(f'Residual (max={vmax_res:.2e})', fontsize=10)
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # Print stats
        print(f"  {mom_label}:")
        print(f"    max |diff| = {vmax_res:.6e}")
        print(f"    mean |diff| (non-zero) = {np.mean(np.abs(residual[mask])):.6e}")

    plt.tight_layout()

    out_dir = os.path.join(REPO_ROOT, 'dev', 'debug_cubes')
    os.makedirs(out_dir, exist_ok=True)
    if args.output is None:
        out_path = os.path.join(out_dir, 'comparison.png')
    else:
        out_path = args.output

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")

    # --- 2D model map comparison if available ---
    for key in ['velocity', 'dispersion']:
        if key in d_main and key in d_jax:
            arr_main = np.asarray(d_main[key], dtype=float)
            arr_jax = np.asarray(d_jax[key], dtype=float)
            finite = np.isfinite(arr_main) & np.isfinite(arr_jax)
            valid = finite & ((arr_main != 0) | (arr_jax != 0))
            if np.any(valid):
                diff = arr_jax - arr_main
                print(f"\n  2D {key} map residual:")
                print(f"    max |diff| = {np.max(np.abs(diff[valid])):.6e}")
                print(f"    mean |diff| = {np.mean(np.abs(diff[valid])):.6e}")
            else:
                print(f"\n  2D {key} map: no valid non-zero pixels to compare")


if __name__ == '__main__':
    main()
