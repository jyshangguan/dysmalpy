#!/usr/bin/env python
"""
Benchmark end-to-end MPFIT fitting timing for 1D, 2D, and 3D test cases.

Designed to run on both dev_jax and main branches with their respective
conda environments. Output is machine-readable key-value pairs.

Usage:
    # On dev_jax:
    conda activate dysmalpy-jax
    JAX_PLATFORMS=cpu python dev/benchmark_fitting.py > dev/benchmark_devjax.log

    # On main:
    conda activate alma
    python dev/benchmark_fitting.py > dev/benchmark_main.log
"""

import os
import sys
import time
import shutil
import numpy as np

# Use matplotlib agg backend to avoid display issues
import matplotlib
matplotlib.use('agg')

from dysmalpy.fitting_wrappers import utils_io as fw_utils_io
from dysmalpy.fitting_wrappers.setup_gal_models import (
    setup_single_galaxy, setup_fitter
)

# Path setup
_dir_script = os.path.abspath(__file__)
_dir_dev = os.path.dirname(_dir_script)
_dir_repo = os.path.dirname(_dir_dev)
_dir_tests_data = os.path.join(_dir_repo, 'tests', 'test_data')

CASES = [
    ('1D', 'fitting_1D_mpfit.params'),
    ('2D', 'fitting_2D_mpfit.params'),
    ('3D', 'fitting_3D_mpfit.params'),
]


def detect_branch():
    """Detect the current git branch name."""
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, cwd=_dir_repo
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'


def run_case(label, param_filename):
    """Run a single MPFIT fitting benchmark case.

    Returns dict with timing, iterations, redchisq, and best-fit params.
    """
    param_filename_full = os.path.join(_dir_tests_data, param_filename)

    # Read params
    params = fw_utils_io.read_fitting_params(fname=param_filename_full)

    # Override output dir to a temp location to avoid polluting test_data
    benchmark_outdir = os.path.join('/tmp', f'dysmalpy_benchmark_{label}')
    params['outdir'] = benchmark_outdir
    params['do_plotting'] = False
    params['overwrite'] = True
    # Set datadir to test_data so data files are found
    params['datadir'] = _dir_tests_data + os.sep

    # Clean previous benchmark output
    if os.path.isdir(benchmark_outdir):
        shutil.rmtree(benchmark_outdir, ignore_errors=True)

    # Setup galaxy, observation, model
    print(f'CASE: {label}', flush=True)
    print(f'SETUP_START: {label}', flush=True)

    gal, output_options = setup_single_galaxy(params=params)
    fitter = setup_fitter(params=params)

    print(f'SETUP_DONE: {label}', flush=True)

    # Run fit with timing
    t0 = time.perf_counter()
    results = fitter.fit(gal, output_options)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    # Extract results
    niter = getattr(results, 'niter', 'N/A')
    redchisq = getattr(results, 'bestfit_redchisq', 'N/A')

    # Extract best-fit parameter values as a string
    param_str = format_bestfit_params(results)

    print(f'TIME: {elapsed:.2f}', flush=True)
    print(f'NITER: {niter}', flush=True)
    print(f'REDCHISQ: {redchisq}', flush=True)
    print(f'PARAMS: {param_str}', flush=True)
    print('---', flush=True)

    # Clean up benchmark output
    if os.path.isdir(benchmark_outdir):
        shutil.rmtree(benchmark_outdir, ignore_errors=True)

    return {
        'label': label,
        'time': elapsed,
        'niter': niter,
        'redchisq': redchisq,
        'params': param_str,
    }


def format_bestfit_params(results):
    """Format best-fit parameter values as a readable string."""
    try:
        params = results.bestfit_parameters
        parts = []
        if isinstance(params, dict):
            for comp_name, comp_params in params.items():
                if isinstance(comp_params, dict):
                    for pname, pval in comp_params.items():
                        parts.append(f'{pname}={pval:.4f}')
                else:
                    parts.append(f'{comp_name}={comp_params:.4f}')
        return ' '.join(parts)
    except Exception:
        return 'N/A'


def main():
    branch = detect_branch()
    print(f'BRANCH: {branch}', flush=True)
    print(f'NUMPY_VERSION: {np.__version__}', flush=True)

    # Report astropy version if available
    try:
        import astropy
        print(f'ASTROPY_VERSION: {astropy.__version__}', flush=True)
    except ImportError:
        print('ASTROPY_VERSION: N/A', flush=True)

    all_results = []

    for label, param_filename in CASES:
        try:
            result = run_case(label, param_filename)
            all_results.append(result)
        except Exception as e:
            print(f'CASE: {label}', flush=True)
            print(f'ERROR: {e}', flush=True)
            print('---', flush=True)
            import traceback
            traceback.print_exc()

    # Summary table
    print('=== SUMMARY ===', flush=True)
    print(f'{"Case":<6} {"Time (s)":<12} {"Niter":<8} {"Redchisq":<10}', flush=True)
    for r in all_results:
        print(f'{r["label"]:<6} {r["time"]:<12.2f} {r["niter"]:<8} {r["redchisq"]:<10}', flush=True)


if __name__ == '__main__':
    main()
