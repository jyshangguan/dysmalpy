#!/usr/bin/env python
"""2D Fitting Demo (JAXNS): fit GS4_43501 velocity/dispersion maps with DYSMALPY.

This script demonstrates JAXNS nested sampling fitting of 2D kinematic maps for
GS4_43501 using DYSMALPY's JAXNS-based sampler.  It produces a best-fit plot,
corner plot, diagnostics, evidence estimate, and a results report.

The model has 8 free parameters (total_mass, r_eff_disk, fdm, sigma0,
sigmaz, inc, pa, xshift, yshift, vel_shift — minus tied/fixed).

**Performance note:** The current DYSMALPY model evaluation is not fully
JAX-traceable, so JAXNS falls back to ``jax.pure_callback`` for each
likelihood evaluation.  This demo uses settings that achieve a reduced
chi-squared of ~9 (comparable to MPFIT's ~10) but requires ~40 minutes of
wall-clock time.  For production fitting, use the NestedFitter (dynesty with
forkserver) or MPFITFitter.

Usage:
    JAX_PLATFORMS=cpu python demo/demo_2D_fitting_JAXNS.py
"""

import os
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
outdir = os.path.join(REPO_ROOT, 'demo', 'demo_2D_output_jaxns')

# We generate a local JAXNS param file from the MPFIT template so we can
# control settings for demo execution.
mpfit_param = os.path.join(REPO_ROOT, 'examples', 'examples_param_files',
                           'fitting_2D_mpfit.params')
local_param = os.path.join(outdir, 'fitting_2D_jaxns_demo.params')

# JAXNS-specific overrides (key, value) to append after reading the MPFIT file.
# The read_fitting_params parser simply takes the last value for each key.
#
# num_live_points=150, dlogZ=0.1, max_evals=15000 give good convergence
# (reduced chi-squared ~9) in ~40 minutes on CPU with pure_callback fallback.
JAXNS_OVERRIDES = """
# ----- JAXNS overrides (appended to MPFIT template) -----
fit_method,      jaxns
num_live_points, 150
dlogZ,            0.1
max_num_likelihood_evaluations, 15000
oversampled_chisq, True
verbose,          True

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

os.makedirs(outdir, exist_ok=True)

# Build the local param file: copy the MPFIT template, then append overrides
shutil.copy2(mpfit_param, local_param)
with open(local_param, 'a') as f:
    f.write(JAXNS_OVERRIDES)

param_filename = local_param

print("=" * 60)
print("  DYSMALPY 2D Fitting Demo (JAXNS)")
print("=" * 60)
print(f"  Data directory : {datadir}")
print(f"  Param file     : {param_filename}")
print(f"  Output directory: {outdir}")
print()
print("  NOTE: JAXNS uses pure_callback fallback (model not JAX-traceable).")
print("        Expect ~40 min wall-clock on CPU for this demo.")
print("=" * 60)

# ---------------------------------------------------------------------------
# 2. Run fitting
# ---------------------------------------------------------------------------
from dysmalpy.fitting_wrappers.dysmalpy_fit_single import dysmalpy_fit_single

print("\n>>> Running 2D JAXNS fit...")
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
fit_method = 'jaxns'

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

# 3a. Print evidence (JAXNS-specific output)
print("\n" + "=" * 60)
print("  BAYESIAN EVIDENCE")
print("=" * 60)
if hasattr(results, 'get_evidence'):
    log_z, log_z_err = results.get_evidence()
    if log_z is not None:
        print(f"  log(Z)       : {log_z:.4f} +/- {log_z_err:.4f}")
    else:
        print("  Evidence not available")

# 3b. Print the pretty results report
print("\n" + "=" * 60)
print("  RESULTS REPORT")
print("=" * 60)
try:
    pretty_report = results.results_report(gal=gal, report_type='pretty')
    print(pretty_report)
except Exception as e:
    print(f"  Could not generate pretty report: {e}")

# 3c. Print the machine-readable results table
print("\n" + "=" * 60)
print("  MACHINE-READABLE RESULTS TABLE")
print("=" * 60)
try:
    machine_report = results.results_report(gal=gal, report_type='machine')
    print(machine_report)
except Exception as e:
    print(f"  Could not generate machine report: {e}")

# 3d. Print fit quality summary
print("\n" + "=" * 60)
print("  FIT QUALITY SUMMARY")
print("=" * 60)
print(f"  Reduced chi-squared : {results.bestfit_redchisq:.4f}")
print(f"  Wall-clock time      : {elapsed:.2f} s ({elapsed/60:.1f} min)")

# 3e. Regenerate diagnostic plots (JAXNS generates its own corner + run plots
#     during the fit; we regenerate the best-fit comparison plot here)
try:
    bestfit_png = os.path.join(outdir, f'{galID}_{fit_method}_bestfit_demo.png')

    print(f"\n>>> Generating best-fit plot: {bestfit_png}")
    results.plot_results(gal, f_plot_bestfit=bestfit_png, overwrite=True)
except Exception as e:
    print(f"  Plot generation skipped: {e}")

# 3f. Print posterior samples summary
if hasattr(results, 'sampler') and results.sampler is not None:
    print("\n" + "=" * 60)
    print("  POSTERIOR SAMPLES")
    print("=" * 60)
    eq_samples = results.sampler.samples
    print(f"  Number of (equally-weighted) samples : {eq_samples.shape[0]}")
    print(f"  Number of parameters               : {eq_samples.shape[1]}")
    print(f"  Parameter names                    : {results.chain_param_names}")

print("\n" + "=" * 60)
print("  All outputs saved to: " + outdir)
print("=" * 60)
