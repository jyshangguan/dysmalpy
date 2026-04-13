#!/usr/bin/env python
"""Benchmark tutorial examples (1D, 2D, 3D) end-to-end using dysmalpy_fit_single.

Usage:
    JAX_PLATFORMS=cpu python dev/bench_tutorial.py [1d|2d|3d|all]

Runs the same fitting as the tutorial notebooks with GS4_43501 data,
measuring wall-clock time and extracting fit quality metrics (redchisq,
number of iterations) from saved results.
"""

import sys
import os
import time
import tempfile
import shutil

# Force non-interactive matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(REPO_ROOT, 'tests', 'test_data')
PARAM_DIR = os.path.join(REPO_ROOT, 'examples', 'examples_param_files')

CONFIGS = {
    '1d': {
        'param_file': os.path.join(PARAM_DIR, 'fitting_1D_mpfit.params'),
        'ref_time': '~2.6 s',
    },
    '2d': {
        'param_file': os.path.join(PARAM_DIR, 'fitting_2D_mpfit.params'),
        'ref_time': '~12 s',
    },
    '3d': {
        'param_file': os.path.join(PARAM_DIR, 'fitting_3D_mpfit.params'),
        'ref_time': '~24 s',
    },
}


def load_fit_results(outdir):
    """Load redchisq and niter from saved fit results."""
    import glob
    import pickle

    # Look for mpfit_results.pickle
    pkl_files = glob.glob(os.path.join(outdir, '*_mpfit_results.pickle'))
    if not pkl_files:
        return None, None

    with open(pkl_files[0], 'rb') as f:
        results = pickle.load(f)

    redchisq = getattr(results, 'bestfit_redchisq', None)
    niter = None
    mpfit_obj = getattr(results, '_mpfit_object', None)
    if mpfit_obj is not None:
        niter = getattr(mpfit_obj, 'niter', None)
    return redchisq, niter


def run_benchmark(mode):
    """Run a single tutorial benchmark and return timing + fit info."""
    from dysmalpy.fitting_wrappers.dysmalpy_fit_single import dysmalpy_fit_single

    cfg = CONFIGS[mode]
    param_file = cfg['param_file']

    # Create temp output directory
    outdir = os.path.join(tempfile.gettempdir(), f'dysmalpy_bench_{mode}')
    os.makedirs(outdir, exist_ok=True)

    print(f"\n--- Running {mode.upper()} tutorial benchmark ---")
    print(f"  Param file: {param_file}")
    print(f"  Data dir:   {DATADIR}")
    print(f"  Output dir: {outdir}")

    t0 = time.perf_counter()
    try:
        dysmalpy_fit_single(
            param_filename=param_file,
            datadir=DATADIR,
            outdir=outdir,
            plot_type='png',
            overwrite=True,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    elapsed = time.perf_counter() - t0

    redchisq, niter = load_fit_results(outdir)

    print(f"  Wall time:  {elapsed:.2f} s")
    print(f"  Iterations: {niter}")
    print(f"  redchisq:   {redchisq}")

    return {
        'mode': mode.upper(),
        'time': elapsed,
        'niter': niter,
        'redchisq': redchisq,
        'ref': cfg['ref_time'],
    }


def print_summary(results_list):
    """Print a formatted summary table."""
    print("\n" + "=" * 60)
    print("  dysmalpy Tutorial Benchmark Results")
    print("=" * 60)
    print(f"{'Mode':<6} | {'Time (s)':>9} | {'Iterations':>10} | {'redchisq':>9} | {'Tutorial Ref':>12}")
    print("-" * 60)
    for r in results_list:
        if r is None:
            continue
        print(f"{r['mode']:<6} | {r['time']:>9.2f} | {r['niter']:>10} | "
              f"{r['redchisq']:>9.3f} | {r['ref']:>12}")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode_arg = sys.argv[1].strip().lower()

    if mode_arg == 'all':
        modes = ['1d', '2d', '3d']
    elif mode_arg in CONFIGS:
        modes = [mode_arg]
    else:
        print(f"Unknown mode: {mode_arg}. Use 1d, 2d, 3d, or all.")
        sys.exit(1)

    results = []
    for mode in modes:
        r = run_benchmark(mode)
        if r is not None:
            results.append(r)

    print_summary(results)


if __name__ == '__main__':
    main()
