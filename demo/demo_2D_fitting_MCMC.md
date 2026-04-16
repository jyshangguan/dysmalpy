# 2D Fitting Tutorial: GS4_43501 (MCMC)

This tutorial demonstrates how to fit 2D kinematic maps (velocity and velocity dispersion)
of a galaxy using DYSMALPY's MCMC fitter (based on `emcee`). We use publicly available
KMOS data for galaxy **GS4_43501** at redshift *z* = 1.613 observed in H-alpha emission.

> **Note:** This is a companion to the [MPFIT fitting tutorial](demo_2D_fitting_MPFIT.md).
> The MCMC sampler explores the full posterior distribution, returning credible intervals
> rather than single best-fit values.

## Model components

The fit uses the same model as the MPFIT tutorial:

| Component | Description |
|---|---|
| `disk+bulge` | Combined disk+bulge mass profile (Sersic disk, Sersic bulge) |
| `const_disp_prof` | Constant intrinsic velocity dispersion profile |
| `geometry` | Galaxy orientation (inclination, PA, spatial/radial shifts) |
| `zheight_gaus` | Gaussian vertical (z) height distribution |
| `halo` | NFW dark matter halo |

Free parameters: `total_mass`, `r_eff_disk`, `fdm`, `sigma0`, `sigmaz`,
`inc`, `pa`, `xshift`, `yshift`, `vel_shift` (10 total).

## 1) Setup

```python
import os
import time
import shutil

import matplotlib
matplotlib.use('Agg')  # headless backend

import numpy as np
from dysmalpy.fitting_wrappers.dysmalpy_fit_single import dysmalpy_fit_single
```

The MCMC demo builds its parameter file by copying the MPFIT template and appending
MCMC-specific overrides (walker count, burn-in steps, etc.):

```python
mpfit_param = 'examples/examples_param_files/fitting_2D_mpfit.params'
local_param = 'demo/demo_2D_output_mcmc/fitting_2D_mcmc_demo.params'
outdir = 'demo/demo_2D_output_mcmc/'

os.makedirs(outdir, exist_ok=True)
shutil.copy2(mpfit_param, local_param)

# Append MCMC-specific settings
with open(local_param, 'a') as f:
    f.write("""
fit_method,      mcmc
nWalkers,         20
nCPUs,            1
nBurn,            2
nSteps,           5
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
""")
```

### Key MCMC parameters

| Parameter | Demo value | Production value | Description |
|---|---|---|---|
| `nWalkers` | 20 | 200+ | Number of ensemble walkers |
| `nBurn` | 2 | 50+ | Burn-in steps (discard) |
| `nSteps` | 5 | 200+ | Production sampling steps |
| `nCPUs` | 1 | 4-8 | Parallel workers (see note below) |
| `scale_param_a` | 3.0 | 2.0-3.0 | emcee stretch move scale |
| `nEff` | 10 | 10-50 | Min effective samples per autocorrelation time |

> **JAX + multiprocessing note:** On the `dev_jax` branch, JAX initializes internal
> thread pools that can deadlock with Python's `fork()`. The MCMC fitter uses
> `forkserver` start method to avoid this. If you encounter `RuntimeError` messages
> about process bootstrapping, set `nCPUs=1` (serial mode) or run from a proper
> module rather than a top-level script.

## 2) Run fitting

```python
t0 = time.perf_counter()
dysmalpy_fit_single(
    param_filename=local_param,
    datadir='tests/test_data/',
    outdir=outdir,
    plot_type='png',
    overwrite=True,
)
elapsed = time.perf_counter() - t0
print(f"Fitting completed in {elapsed:.2f} s")
```

**Sample console output:**

```
INFO:DysmalPy:*************************************
INFO:DysmalPy: Fitting: GS4_43501 with MCMC
INFO:DysmalPy:    obs: OBS
INFO:DysmalPy:        velocity file: .../tests/test_data/GS4_43501_Ha_vm.fits
INFO:DysmalPy:        dispers. file: .../tests/test_data/GS4_43501_Ha_dm.fits
nCPUs: 1
INFO:DysmalPy:nWalkers: 20
INFO:DysmalPy:lnlike: oversampled_chisq=True

Burn-in:
Start: 2026-04-14 08:07:00

End: 2026-04-14 08:07:15
nCPU, nParam, nWalker, nBurn = 1, 10, 20, 2
Mean acceptance fraction: 0.000
Ideal acceptance frac: 0.2 - 0.5

Ensemble sampling:
Start: 2026-04-14 08:07:16

Finished 5 steps
Time= 14.91 (sec)
Mean acceptance fraction: 0.000

Fitting completed in 36.26 s
```

The 0% acceptance fraction and `nan` autocorrelation times are expected with only
5 production steps and 2 burn-in steps. A real analysis needs many more steps for
the walkers to explore the parameter space.

## 3) Examine results

### Reload the fit

```python
from dysmalpy.fitting_wrappers.data_io import read_fitting_params
from dysmalpy.fitting import reload_all_fitting

galID = 'GS4_43501'
results_pickle = f'{outdir}/{galID}_mcmc_results.pickle'
model_pickle   = f'{outdir}/{galID}_model.pickle'

gal, results = reload_all_fitting(
    filename_galmodel=model_pickle,
    filename_results=results_pickle,
    fit_method='mcmc',
)
```

### Diagnostic plots

```python
results.plot_results(
    gal,
    f_plot_param_corner=f'{outdir}/{galID}_mcmc_param_corner_demo.png',
    f_plot_trace=f'{outdir}/{galID}_mcmc_trace_demo.png',
    f_plot_bestfit=f'{outdir}/{galID}_mcmc_bestfit_demo.png',
    overwrite=True,
)
```

#### Best-fit comparison

![Best-fit comparison: observed (left) vs. model (middle) vs. residual (right)
velocity and dispersion maps](figs/GS4_43501_mcmc_bestfit_demo_OBS.png)

#### Trace plot

Walker chains for each free parameter across sampling steps. With only 5 steps
the chains show no mixing -- a production run would show well-mixed, stationary
chains that explore the posterior.

![MCMC trace plot](figs/GS4_43501_mcmc_trace_demo.png)

#### Corner plot

Pairwise posterior distributions for all free parameters. With too few samples the
contours are poorly defined. A production run yields smooth, well-constrained
marginalised posteriors.

![MCMC corner plot](figs/GS4_43501_mcmc_param_corner_demo.png)

### Results report

```python
report = results.results_report(gal=gal, report_type='pretty')
print(report)
```

```
###############################
 Fitting for GS4_43501

Fitting method: MCMC

###############################
 Fitting results
-----------
 disk+bulge
    total_mass         10.0394  -   1.8010 +   2.1749
    r_eff_disk         27.0304  -   9.0737 +  17.0591
-----------
 halo
    fdm                 0.3141  -  -0.0277 +   0.6786
-----------
 dispprof_LINE
    sigma0           80714.2476  -18376.6639 +16486.2604
-----------
 zheightgaus
    sigmaz              2.4998  -  -0.8948 +   6.4606
-----------
 geom_1
    inc                22.0376  -   4.9197 +  46.5814
    pa                -95.1432  -   5.4248 + 241.5527
    xshift           -39845.7891  -33631.4948 +90025.6242
    yshift           -80923.0128  --48558.6534 +154446.0973
    vel_shift        -45770.5908  -16246.9540 +65829.3156
-----------
Red. chisq: nan
```

Note the extremely wide and physically implausible credible intervals (e.g. `xshift`
spanning tens of thousands of km/s). This is a direct consequence of running only
5 steps with 20 walkers -- the sampler has not converged. A production run with
`nWalkers=200, nBurn=50, nSteps=200` yields tight, physically meaningful constraints.

### Machine-readable results table

```python
machine = results.results_report(gal=gal, report_type='machine')
print(machine)
```

```
# component             param_name      fixed       best_value   l68_err     u68_err
disk+bulge              total_mass      False        10.0394      1.8010      2.1749
disk+bulge              r_eff_disk      False        27.0304      9.0737     17.0591
halo                    fdm             TIED          0.3141     -0.0277      0.6786
dispprof_LINE           sigma0          True      80714.2476   18376.6639   16486.2604
zheightgaus             sigmaz          TIED          2.4998     -0.8948      6.4606
geom_1                  inc             False        22.0376      4.9197     46.5814
geom_1                  pa              False       -95.1432      5.4248    241.5527
geom_1                  xshift          False    -39845.7891   33631.4948   90025.6242
geom_1                  yshift          False    -80923.0128   -48558.6534   154446.0973
geom_1                  vel_shift       True     -45770.5908   16246.9540   65829.3156
redchisq                -----           -----            nan    -99.0000    -99.0000
```

### Fit quality summary

```python
print(f"Reduced chi-squared : {results.bestfit_redchisq:.4f}")
print(f"Acceptance fraction: {np.mean(results.sampler_results['acceptance_fraction']):.3f}")
```

```
Reduced chi-squared : nan
Acceptance fraction: 0.000
```

## Output files

The fitting run produces the following files in `demo/demo_2D_output_mcmc/`:

| File | Description |
|---|---|
| `GS4_43501_mcmc_results.pickle` | Serialized MCMC results object (chains, blobs, etc.) |
| `GS4_43501_model.pickle` | Galaxy model with best-fit parameters |
| `GS4_43501_mcmc_sampler.h5` | emcee HDF5 backend with full chain |
| `GS4_43501_mcmc_bestfit_OBS.png` | Best-fit comparison plot |
| `GS4_43501_mcmc_param_corner.png` | Corner plot of posterior distributions |
| `GS4_43501_mcmc_trace.png` | Walker trace plot |
| `GS4_43501_mcmc_burnin_trace.png` | Burn-in trace plot |
| `GS4_43501_mcmc_bestfit_results_report.info` | Human-readable results report |
| `GS4_43501_mcmc_bestfit_results.dat` | Machine-readable results table |
| `GS4_43501_mcmc_chain_blobs.dat` | Blob data (derived quantities per step) |
| `GS4_43501_OBS_out-velmaps.fits` | Best-fit 2D velocity/dispersion maps |
| `GS4_43501_OBS_bestfit_cube.fits` | Best-fit model data cube |
| `GS4_43501_bestfit_vcirc.dat` | Circular velocity profile |
| `GS4_43501_bestfit_menc.dat` | Enclosed mass profile |
| `GS4_43501_LINE_bestfit_velprofile.dat` | Line-of-sight velocity profile |
| `GS4_43501_mcmc.log` | Fitting log |

## Comparison with MPFIT

| Aspect | MPFIT | MCMC |
|---|---|---|
| Method | Levenberg-Marquardt | Affine-invariant ensemble sampler |
| Output | Single best-fit + formal errors | Full posterior distribution |
| Speed (this problem) | ~10 s | ~36 s (demo), minutes (production) |
| Convergence check | Status code | Acceptance fraction, autocorrelation time |
| Credible intervals | Symmetric (from covariance) | Asymmetric (from posterior quantiles) |

## How to run

From the repository root:

```bash
JAX_PLATFORMS=cpu python demo/demo_2D_fitting_MCMC.py
```

The `JAX_PLATFORMS=cpu` environment variable forces JAX onto the CPU (avoids GPU
initialisation overhead for this small problem). The script uses `matplotlib.use('Agg')`
so it runs headless -- no display is needed.
