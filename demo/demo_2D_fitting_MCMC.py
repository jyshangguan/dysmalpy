#!/usr/bin/env python
"""2D Fitting Demo (MCMC): fit GS4_43501 velocity/dispersion maps with DYSMALPY.

This script demonstrates MCMC fitting of 2D kinematic maps for GS4_43501 using
DYSMALPY's emcee-based sampler. It produces a best-fit plot, corner plot, trace
plot, and both human-readable and machine-readable results reports.

The MCMC settings used here are deliberately small (nWalkers=20, nBurn=2,
nSteps=5, oversample=1) for a quick demo. For a real analysis increase these
substantially (e.g. nWalkers=200, nBurn=50, nSteps=200, oversample=3).

Usage:
    JAX_PLATFORMS=cpu python demo/demo_2D_fitting_MCMC.py
"""

import os
#os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU for this demo (no GPU/TPU required)
#os.environ['OMP_NUM_THREADS'] = '1'
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'
import time
import shutil

# Force non-interactive backend before any matplotlib imports
import matplotlib
matplotlib.use('Agg')

import numpy as np

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = os.path.join(REPO_ROOT, 'tests', 'test_data')
outdir = os.path.join(REPO_ROOT, 'demo', 'demo_2D_output_mcmc')

# We generate a local MCMC param file from the MPFIT template so we can
# control oversample=1 for fast demo execution.
mpfit_param = os.path.join(REPO_ROOT, 'examples', 'examples_param_files',
                           'fitting_2D_mpfit.params')
local_param = os.path.join(outdir, 'fitting_2D_mcmc_demo.params')

# MCMC-specific overrides (key, value) to append after reading the MPFIT file.
# The read_fitting_params parser simply takes the last value for each key.
MCMC_OVERRIDES = """
# ----- MCMC overrides (appended to MPFIT template) -----
fit_method,      mcmc
nWalkers,         32
nCPUs,            10
nBurn,            100
nSteps,           500
scale_param_a,    3.
minAF,            None
maxAF,            None
nEff,             10

# Prior types (flat for all parameters in this demo)
total_mass_prior,    flat
bt_prior,            flat
r_eff_disk_prior,    flat
fdm_prior,           flat
sigma0_prior,        flat
sigmaz_prior,        flat
inc_prior,           flat
pa_prior,            flat
xshift_prior,        flat
yshift_prior,        flat
vel_shift_prior,     flat
"""

if __name__ == '__main__':
    os.makedirs(outdir, exist_ok=True)

    # Build the local param file: copy the MPFIT template, then append overrides
    shutil.copy2(mpfit_param, local_param)
    with open(local_param, 'a') as f:
        f.write(MCMC_OVERRIDES)

    param_filename = local_param

    print("=" * 60)
    print("  DYSMALPY 2D Fitting Demo (MCMC)")
    print("=" * 60)
    print(f"  Data directory : {datadir}")
    print(f"  Param file     : {param_filename}")
    print(f"  Output directory: {outdir}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 2. Run fitting
    # -----------------------------------------------------------------------
    from dysmalpy.fitting_wrappers.dysmalpy_fit_single import dysmalpy_fit_single

    print("\n>>> Running 2D MCMC fit...")
    t0 = time.perf_counter()
    dysmalpy_fit_single(
        param_filename=param_filename,
        datadir=datadir,
        outdir=outdir,
        plot_type='png',
        overwrite=True,
    )
    elapsed = time.perf_counter() - t0
    print(f">>> Fitting completed in {elapsed:.2f} s")

    # -----------------------------------------------------------------------
    # 3. Examine results
    # -----------------------------------------------------------------------
    from dysmalpy.fitting_wrappers.data_io import read_fitting_params
    from dysmalpy.fitting import reload_all_fitting

    galID = 'GS4_43501'
    fit_method = 'mcmc'

    # Read back the parameter file for reference
    params = read_fitting_params(param_filename)
    print(f"\nGalaxy ID   : {params['galID']}")
    print(f"Redshift    : {params['z']}")
    print(f"Fit method  : {params['fit_method']}")

    # Locate the saved pickle files
    results_pickle = os.path.join(outdir, f'{galID}_{fit_method}_results.pickle')
    model_pickle = os.path.join(outdir, f'{galID}_model.pickle')

    print(f"\nLoading results from:\n  {results_pickle}\n  {model_pickle}")

    gal, results = reload_all_fitting(
        filename_galmodel=model_pickle,
        filename_results=results_pickle,
        fit_method=fit_method,
    )

    # 3a. Regenerate diagnostic plots (saved as PNG for the markdown)
    corner_png = os.path.join(outdir, f'{galID}_{fit_method}_param_corner_demo.png')
    trace_png = os.path.join(outdir, f'{galID}_{fit_method}_trace_demo.png')
    bestfit_png = os.path.join(outdir, f'{galID}_{fit_method}_bestfit_demo.png')

    print(f"\n>>> Generating diagnostic plots...")
    print(f"    Corner plot : {corner_png}")
    print(f"    Trace plot  : {trace_png}")
    print(f"    Best-fit    : {bestfit_png}")

    results.plot_results(
        gal,
        f_plot_param_corner=corner_png,
        f_plot_trace=trace_png,
        f_plot_bestfit=bestfit_png,
        overwrite=True,
    )

    # 3b. Print the pretty results report
    print("\n" + "=" * 60)
    print("  RESULTS REPORT")
    print("=" * 60)
    pretty_report = results.results_report(gal=gal, report_type='pretty')
    print(pretty_report)

    # 3c. Print the machine-readable results table
    print("\n" + "=" * 60)
    print("  MACHINE-READABLE RESULTS TABLE")
    print("=" * 60)
    machine_report = results.results_report(gal=gal, report_type='machine')
    print(machine_report)

    # 3d. Print fit quality summary
    print("\n" + "=" * 60)
    print("  FIT QUALITY SUMMARY")
    print("=" * 60)
    print(f"  Reduced chi-squared : {results.bestfit_redchisq:.4f}")
    print(f"  Wall-clock time      : {elapsed:.2f} s")

    # MCMC-specific diagnostics
    if hasattr(results, 'sampler_results'):
        sr = results.sampler_results
        if 'acceptance_fraction' in sr:
            af = sr['acceptance_fraction']
            print(f"  Acceptance fraction  : {np.mean(af):.3f}")
        if 'autocorr_time' in sr and sr['autocorr_time'] is not None:
            act = sr['autocorr_time']
            if np.ndim(act) == 0:
                print(f"  Autocorrelation time : {act:.1f}")
            else:
                print(f"  Autocorrelation time : mean={np.nanmean(act):.1f}, "
                      f"max={np.nanmax(act):.1f}")

    print("\n" + "=" * 60)
    print("  All outputs saved to: " + outdir)
    print("=" * 60)
