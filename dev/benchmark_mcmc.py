#!/usr/bin/env python
"""MCMC CPU Multi-Core vs GPU Speed Benchmark.

Runs short MCMC fits with different CPU core counts and GPU configurations
to measure per-evaluation timing and parallel scaling.

Usage:
    # CPU-only (force CPU):
    JAX_PLATFORMS=cpu python dev/benchmark_mcmc.py

    # Auto-detect GPU (if available):
    python dev/benchmark_mcmc.py

    # Custom worker counts:
    python dev/benchmark_mcmc.py --walkers 16 --burn 5 --steps 20
"""

import os
import sys
import subprocess
import time
import shutil
import argparse
import json
import tempfile

# Force non-interactive backend before any matplotlib imports
import matplotlib
matplotlib.use('Agg')

# Path setup
_dir_script = os.path.abspath(__file__)
_dir_dev = os.path.dirname(_dir_script)
_dir_repo = os.path.dirname(_dir_dev)
_dir_tests_data = os.path.join(_dir_repo, 'tests', 'test_data')

# Default benchmark parameters (small for fast turnaround)
DEFAULT_NWALKERS = 16
DEFAULT_NBURN = 5
DEFAULT_NSTEPS = 20


# ---------------------------------------------------------------------------
# Subprocess worker: runs a single MCMC config in an isolated process
# ---------------------------------------------------------------------------

WORKER_SCRIPT = r'''
import os
import sys
import time
import shutil
import json
import pickle
import tempfile

# Force non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np

# Path setup (must match the parent script)
_dir_repo = __DIR_REPO__
_dir_tests_data = __DIR_TESTS_DATA__

sys.path.insert(0, _dir_repo)

from dysmalpy.fitting_wrappers.setup_gal_models import (
    setup_single_galaxy, setup_fitter
)
from dysmalpy.fitting_wrappers.data_io import read_fitting_params
from dysmalpy.fitting import base as fit_base
from dysmalpy.fitting import MCMCFitter


def run_single_config(config):
    """Run a single MCMC benchmark configuration.

    Parameters
    ----------
    config : dict with keys:
        nCPUs, jax_platform, nWalkers, nBurn, nSteps, scale_param_a,
        do_warmup, do_breakdown

    Returns
    -------
    dict with timing results
    """
    nCPUs = config['nCPUs']
    jax_platform = config['jax_platform']
    nWalkers = config['nWalkers']
    nBurn = config['nBurn']
    nSteps = config['nSteps']
    scale_param_a = config['scale_param_a']
    do_warmup = config['do_warmup']
    do_breakdown = config['do_breakdown']

    # Read params from the MPFIT template
    mpfit_param = os.path.join(_dir_repo, 'examples', 'examples_param_files',
                               'fitting_2D_mpfit.params')
    params = read_fitting_params(fname=mpfit_param)

    # Override for MCMC
    params['fit_method'] = 'mcmc'
    params['nWalkers'] = nWalkers
    params['nCPUs'] = nCPUs
    params['nBurn'] = nBurn
    params['nSteps'] = nSteps
    params['scale_param_a'] = scale_param_a
    params['do_plotting'] = False
    params['overwrite'] = True
    params['datadir'] = _dir_tests_data + '/'
    params['minAF'] = None
    params['maxAF'] = None
    params['nEff'] = 10

    # Set flat priors
    for p in ['total_mass', 'bt', 'r_eff_disk', 'fdm', 'sigma0',
              'sigmaz', 'inc', 'pa', 'xshift', 'yshift', 'vel_shift']:
        key = f'{p}_prior'
        if key in params:
            params[key] = 'flat'

    # Setup output directory
    benchmark_outdir = tempfile.mkdtemp(prefix='dysmalpy_benchmark_mcmc_')
    params['outdir'] = benchmark_outdir

    # Setup galaxy and model
    gal, output_options = setup_single_galaxy(params=params)

    # Setup oversampled chisq
    from dysmalpy.fitting.utils import setup_oversampled_chisq
    gal = setup_oversampled_chisq(gal)

    # Setup fitter manually (to control all attributes)
    fitter = MCMCFitter()
    fitter.nWalkers = nWalkers
    fitter.nCPUs = nCPUs
    fitter.nBurn = nBurn
    fitter.nSteps = nSteps
    fitter.scale_param_a = scale_param_a
    fitter.oversampled_chisq = False
    fitter.nPostBins = 50
    fitter.minAF = None
    fitter.maxAF = None
    fitter.nEff = 10

    model = gal.model
    pfree = model.get_free_parameters_values()

    result = {
        'nCPUs': nCPUs,
        'jax_platform': jax_platform,
        'nWalkers': nWalkers,
        'nBurn': nBurn,
        'nSteps': nSteps,
        'nEvals': nWalkers * (nBurn + nSteps),
        'error': None,
    }

    # GPU warmup: trigger JIT compilation of _fft_convolve_cached
    if do_warmup:
        try:
            _ = fit_base.log_prob(pfree, gal, fitter=fitter)
        except Exception:
            pass

    # Single log_prob breakdown timing (only for serial configs)
    if do_breakdown:
        breakdown = {}
        n_warmup = 2
        n_measure = 3

        # Warmup calls (to trigger JIT compilation, caching, etc.)
        for _ in range(n_warmup):
            try:
                _ = fit_base.log_prob(pfree, gal, fitter=fitter)
            except Exception:
                pass

        # Measured calls
        times = {'update_parameters': [], 'create_model_data': [], 'log_like': []}
        for _ in range(n_measure):
            t0 = time.perf_counter()
            model.update_parameters(pfree)
            t1 = time.perf_counter()
            times['update_parameters'].append((t1 - t0) * 1000)

            lprior = model.get_log_prior()

            if np.isfinite(lprior):
                t0 = time.perf_counter()
                gal.create_model_data()
                t1 = time.perf_counter()
                times['create_model_data'].append((t1 - t0) * 1000)

                t0 = time.perf_counter()
                _ = fit_base.log_like(gal, fitter=fitter)
                t1 = time.perf_counter()
                times['log_like'].append((t1 - t0) * 1000)
            else:
                times['create_model_data'].append(0.)
                times['log_like'].append(0.)

        # Use median for robustness
        for key in times:
            breakdown[key] = float(np.median(times[key]))

        result['breakdown'] = breakdown

    # Run the full MCMC fit
    try:
        t0 = time.perf_counter()
        results = fitter.fit(gal, output_options)
        t1 = time.perf_counter()
        wall_time = t1 - t0

        # Extract acceptance fraction
        af = None
        if hasattr(results, 'sampler_results'):
            sr = results.sampler_results
            if isinstance(sr, dict) and 'acceptance_fraction' in sr:
                af = float(np.mean(sr['acceptance_fraction']))
            elif hasattr(sr, 'acceptance_fraction'):
                af = float(np.mean(sr.acceptance_fraction))

        result['wall_time'] = wall_time
        result['per_eval_ms'] = (wall_time / result['nEvals']) * 1000
        result['acceptance'] = af

    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()

    # Cleanup output
    if os.path.isdir(benchmark_outdir):
        shutil.rmtree(benchmark_outdir, ignore_errors=True)

    return result


if __name__ == '__main__':
    config = json.loads(sys.argv[1])
    result = run_single_config(config)
    # Print result as JSON to stdout
    print(json.dumps(result))
'''


def detect_gpu_available():
    """Check if a GPU is available by running a subprocess."""
    script = (
        'import json, jax; '
        'devices = [str(d) for d in jax.devices()]; '
        'has_gpu = any("gpu" in d.lower() or "cuda" in d.lower() for d in devices); '
        'print(json.dumps({"has_gpu": has_gpu, "devices": devices}))'
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, text=True, timeout=30
        )
        info = json.loads(result.stdout.strip())
        return info['has_gpu'], info['devices']
    except Exception:
        return False, []


def run_config_in_subprocess(config):
    """Run a single benchmark config in a subprocess with controlled JAX_PLATFORMS."""
    env = os.environ.copy()
    env['JAX_PLATFORMS'] = config['jax_platform']
    # Limit thread contention in workers
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'

    worker_code = WORKER_SCRIPT.replace('__DIR_REPO__', repr(_dir_repo))
    worker_code = worker_code.replace('__DIR_TESTS_DATA__', repr(_dir_tests_data))

    result = subprocess.run(
        [sys.executable, '-c', worker_code, json.dumps(config)],
        capture_output=True, text=True, timeout=600,
        env=env,
    )

    if result.returncode != 0:
        return {
            'nCPUs': config['nCPUs'],
            'jax_platform': config['jax_platform'],
            'nWalkers': config['nWalkers'],
            'nBurn': config['nBurn'],
            'nSteps': config['nSteps'],
            'nEvals': config['nWalkers'] * (config['nBurn'] + config['nSteps']),
            'error': result.stderr.strip() or f"Exit code {result.returncode}",
            'wall_time': None,
            'per_eval_ms': None,
            'acceptance': None,
        }

    # Parse JSON from stdout (last line)
    lines = result.stdout.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return {
        'nCPUs': config['nCPUs'],
        'jax_platform': config['jax_platform'],
        'nWalkers': config['nWalkers'],
        'nBurn': config['nBurn'],
        'nSteps': config['nSteps'],
        'nEvals': config['nWalkers'] * (config['nBurn'] + config['nSteps']),
        'error': 'No JSON output from worker',
        'wall_time': None,
        'per_eval_ms': None,
        'acceptance': None,
    }


def format_results_table(results, nWalkers, nBurn, nSteps):
    """Format benchmark results into a summary table."""
    nEvals = nWalkers * (nBurn + nSteps)

    # Find baseline (CPU serial) for speedup calculation
    baseline_ms = None
    for r in results:
        if r['nCPUs'] == 1 and r['jax_platform'] == 'cpu' and r.get('per_eval_ms'):
            baseline_ms = r['per_eval_ms']
            break

    header = (
        f"{'Config':<35} {'nCPUs':>5}  {'Wall (s)':>8}  {'Evals':>5}  "
        f"{'ms/eval':>7}  {'Speedup':>7}  {'Eff%':>5}  {'Accept%':>7}\n"
        + "-" * 86
    )

    rows = []
    for r in results:
        label = r.get('label', '')
        ncpus = r['nCPUs']
        wall = r.get('wall_time')
        ms_eval = r.get('per_eval_ms')
        af = r.get('acceptance')

        if wall is not None:
            wall_str = f"{wall:.2f}"
        else:
            wall_str = "ERROR"

        if ms_eval is not None:
            ms_str = f"{ms_eval:.1f}"
            if baseline_ms and baseline_ms > 0:
                speedup = baseline_ms / ms_eval
                eff = (speedup / ncpus) * 100
                speedup_str = f"{speedup:.2f}x"
                eff_str = f"{eff:.0f}%"
            else:
                speedup_str = "--"
                eff_str = "--"
        else:
            ms_str = "--"
            speedup_str = "--"
            eff_str = "--"

        if af is not None:
            af_str = f"{af:.1f}%"
        else:
            af_str = "--"

        rows.append(
            f"{label:<35} {ncpus:>5}  {wall_str:>8}  {nEvals:>5}  "
            f"{ms_str:>7}  {speedup_str:>7}  {eff_str:>5}  {af_str:>7}"
        )

    return header + '\n' + '\n'.join(rows)


def format_breakdown(breakdown):
    """Format single log_prob breakdown timing."""
    if not breakdown:
        return ""

    lines = ["\nSingle log_prob breakdown (CPU serial):"]
    total = 0
    for key in ['update_parameters', 'create_model_data', 'log_like']:
        val = breakdown.get(key, 0)
        lines.append(f"  {key:<22} {val:6.1f} ms")
        total += val
    lines.append(f"  {'Total':<22} {total:6.1f} ms")
    return '\n'.join(lines)


def detect_branch():
    """Detect the current git branch name."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, cwd=_dir_repo
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'


def main():
    parser = argparse.ArgumentParser(description='MCMC CPU/GPU benchmark')
    parser.add_argument('--walkers', type=int, default=DEFAULT_NWALKERS,
                        help=f'Number of walkers (default: {DEFAULT_NWALKERS})')
    parser.add_argument('--burn', type=int, default=DEFAULT_NBURN,
                        help=f'Number of burn-in steps (default: {DEFAULT_NBURN})')
    parser.add_argument('--steps', type=int, default=DEFAULT_NSTEPS,
                        help=f'Number of sampling steps (default: {DEFAULT_NSTEPS})')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Skip GPU configs even if GPU is available')
    parser.add_argument('--cores', type=int, nargs='+', default=None,
                        help='List of CPU core counts to test (default: 1,2,4,8,16)')
    args = parser.parse_args()

    nWalkers = args.walkers
    nBurn = args.burn
    nSteps = args.steps
    nEvals = nWalkers * (nBurn + nSteps)

    if args.cores:
        cpu_cores = args.cores
    else:
        cpu_cores = [1, 2, 4, 8, 16]

    # Print header
    branch = detect_branch()
    print("=" * 70, flush=True)
    print("MCMC BENCHMARK: CPU Multi-Core vs GPU", flush=True)
    print("=" * 70, flush=True)
    print(f"  Branch       : {branch}", flush=True)
    print(f"  nWalkers     : {nWalkers}", flush=True)
    print(f"  nBurn        : {nBurn}", flush=True)
    print(f"  nSteps       : {nSteps}", flush=True)
    print(f"  Evals/config : {nEvals}", flush=True)

    import numpy as np
    print(f"  NumPy        : {np.__version__}", flush=True)

    # Detect GPU
    has_gpu, devices = detect_gpu_available()
    print(f"  JAX devices  : {', '.join(devices)}", flush=True)
    print(f"  GPU available: {has_gpu}", flush=True)
    if not has_gpu and not args.cpu_only:
        print("  (GPU configs will be skipped)", flush=True)
    print("=" * 70, flush=True)

    # Build configuration list
    configs = []

    # CPU configs
    for ncpus in cpu_cores:
        configs.append({
            'label': f'CPU {ncpus}-core' if ncpus > 1 else 'CPU serial',
            'nCPUs': ncpus,
            'jax_platform': 'cpu',
            'nWalkers': nWalkers,
            'nBurn': nBurn,
            'nSteps': nSteps,
            'scale_param_a': 3.,
            'do_warmup': False,
            'do_breakdown': (ncpus == 1),  # breakdown only for serial
        })

    # GPU configs (convolution on GPU, simulate_cube on CPU)
    if has_gpu and not args.cpu_only:
        configs.append({
            'label': 'GPU (conv only)',
            'nCPUs': 1,
            'jax_platform': 'gpu',
            'nWalkers': nWalkers,
            'nBurn': nBurn,
            'nSteps': nSteps,
            'scale_param_a': 3.,
            'do_warmup': True,
            'do_breakdown': True,
        })
        configs.append({
            'label': 'GPU (conv only) 4-core',
            'nCPUs': 4,
            'jax_platform': 'gpu',
            'nWalkers': nWalkers,
            'nBurn': nBurn,
            'nSteps': nSteps,
            'scale_param_a': 3.,
            'do_warmup': True,
            'do_breakdown': False,
        })

    # Run each config
    results = []
    for i, cfg in enumerate(configs):
        label = cfg['label']
        print(f"\n[{i+1}/{len(configs)}] Running: {label} "
              f"(nCPUs={cfg['nCPUs']}, JAX_PLATFORMS={cfg['jax_platform']})",
              flush=True)
        print(f"  Started at: {time.strftime('%H:%M:%S')}", flush=True)

        t0 = time.perf_counter()
        result = run_config_in_subprocess(cfg)
        elapsed = time.perf_counter() - t0

        result['label'] = label
        results.append(result)

        if result.get('error'):
            print(f"  ERROR: {result['error']}", flush=True)
        else:
            print(f"  Wall time   : {result['wall_time']:.2f} s", flush=True)
            print(f"  Per-eval    : {result['per_eval_ms']:.1f} ms", flush=True)
            print(f"  Acceptance  : {result.get('acceptance', 'N/A')}", flush=True)
        print(f"  Finished in : {elapsed:.1f} s (wall)", flush=True)

    # Print summary table
    print("\n" + "=" * 70, flush=True)
    print("MCMC BENCHMARK RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(format_results_table(results, nWalkers, nBurn, nSteps), flush=True)

    # Print breakdown if available
    for r in results:
        if 'breakdown' in r:
            print(format_breakdown(r['breakdown']), flush=True)
            break  # only print once

    print("\n" + "=" * 70, flush=True)


if __name__ == '__main__':
    main()
