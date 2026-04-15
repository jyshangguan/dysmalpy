#!/usr/bin/env python
"""2D Fitting Demo: fit GS4_43501 velocity/dispersion maps with DYSMALPY.

This script mirrors the official 2D fitting tutorial, adapted for the dev_jax branch.
It runs a full 2D kinematic fit (MPFIT) and produces a best-fit plot, a human-readable
results report, and a machine-readable results table.

Usage:
    JAX_PLATFORMS=cpu python demo/demo_2D_fitting_MPFIT.py
"""

import os
import time

# Force non-interactive backend before any matplotlib imports
import matplotlib
matplotlib.use('Agg')

import numpy as np

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = os.path.join(REPO_ROOT, 'tests', 'test_data')
param_filename = os.path.join(REPO_ROOT, 'examples', 'examples_param_files',
                              'fitting_2D_mpfit.params')
outdir = os.path.join(REPO_ROOT, 'demo', 'demo_2D_output')

print("=" * 60)
print("  DYSMALPY 2D Fitting Demo")
print("=" * 60)
print(f"  Data directory : {datadir}")
print(f"  Param file     : {param_filename}")
print(f"  Output directory: {outdir}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 2. Run fitting
# ---------------------------------------------------------------------------
from dysmalpy.fitting_wrappers.dysmalpy_fit_single import dysmalpy_fit_single

os.makedirs(outdir, exist_ok=True)

print("\n>>> Running 2D MPFIT fit...")
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

# ---------------------------------------------------------------------------
# 3. Examine results
# ---------------------------------------------------------------------------
from dysmalpy.fitting_wrappers.data_io import read_fitting_params
from dysmalpy.fitting import reload_all_fitting

galID = 'GS4_43501'
fit_method = 'mpfit'

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

# 3a. Regenerate best-fit plot (saved as PNG for the markdown)
bestfit_png = os.path.join(outdir, f'{galID}_{fit_method}_bestfit_demo.png')
print(f"\n>>> Generating best-fit plot: {bestfit_png}")
results.plot_results(gal, f_plot_bestfit=bestfit_png, overwrite=True)

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
niter = getattr(results._mpfit_object, 'niter', None) if hasattr(results, '_mpfit_object') else None
if niter is not None:
    print(f"  MPFIT iterations     : {niter}")
print(f"  Wall-clock time      : {elapsed:.2f} s")

print("\n" + "=" * 60)
print("  All outputs saved to: " + outdir)
print("=" * 60)
