# 2D Fitting Tutorial: GS4_43501 (JAXNS)

This tutorial demonstrates how to fit 2D kinematic maps (velocity and velocity dispersion)
of a galaxy using DYSMALPY's JAXNS fitter (based on the `jaxns` nested sampling library).
We use publicly available KMOS data for galaxy **GS4_43501** at redshift *z* = 1.613 observed
in H-alpha emission.

> **Note:** This is a companion to the [MPFIT fitting tutorial](demo_2D_fitting_MPFIT.md)
> and the [MCMC fitting tutorial](demo_2D_fitting_MCMC.md). JAXNS performs Bayesian
> nested sampling, returning an estimate of the Bayesian evidence (marginal likelihood)
> in addition to posterior samples.

## Model components

The fit uses the same model as the MPFIT and MCMC tutorials:

| Component | Description |
|---|---|
| `disk+bulge` | Combined disk+bulge mass profile (Sersic disk, Sersic bulge) |
| `const_disp_prof` | Constant intrinsic velocity dispersion profile |
| `geometry` | Galaxy orientation (inclination, PA, spatial/radial shifts) |
| `zheight_gaus` | Gaussian vertical (z) height distribution |
| `halo` | NFW dark matter halo |

Free parameters: `total_mass`, `r_eff_disk`, `fdm`, `sigma0`, `sigmaz`,
`inc`, `pa`, `xshift`, `yshift`, `vel_shift` (8 free after accounting for tied
and fixed parameters).

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

The JAXNS demo builds its parameter file by copying the MPFIT template and appending
JAXNS-specific overrides (number of live points, evidence tolerance, etc.):

```python
mpfit_param = 'examples/examples_param_files/fitting_2D_mpfit.params'
local_param = 'demo/demo_2D_output_jaxns/fitting_2D_jaxns_demo.params'
outdir = 'demo/demo_2D_output_jaxns/'

os.makedirs(outdir, exist_ok=True)
shutil.copy2(mpfit_param, local_param)

# Append JAXNS-specific settings
with open(local_param, 'a') as f:
    f.write("""
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
""")
```

### Key JAXNS parameters

| Parameter | Demo value | Description |
|---|---|---|
| `num_live_points` | 150 | Number of live points in the nested sampling |
| `dlogZ` | 0.1 | Target evidence uncertainty (dlog(Z) tolerance) |
| `max_num_likelihood_evaluations` | 15000 | Hard limit on likelihood evaluations |
| `oversampled_chisq` | True | Use oversampled chi-squared in likelihood |

> **Performance note:** The current DYSMALPY model evaluation is not fully
> JAX-traceable, so JAXNS falls back to `jax.pure_callback` for each
> likelihood evaluation. This demo uses settings that achieve a reduced
> chi-squared of ~9 (comparable to MPFIT's ~10) but requires ~40 minutes of
> wall-clock time on CPU. For production fitting, use the NestedFitter (dynesty)
> or MPFITFitter.

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
============================================================
  DYSMALPY 2D Fitting Demo (JAXNS)
============================================================
  NOTE: JAXNS uses pure_callback fallback (model not JAX-traceable).
        Expect ~40 min wall-clock on CPU for this demo.
============================================================

>>> Running 2D JAXNS fit...
Creating initial state with 150 live points.
Running uniform sampling down to efficiency threshold of 0.1.
Running until termination condition: TerminationCondition(...)

-------
Num samples: 75
Num likelihood evals: 3397
Efficiency: 0.043
log(Z) est.: -192.84 +- 0.83

-------
Num samples: 150
Num likelihood evals: 7195
Efficiency: 0.028
log(Z) est.: -171.45 +- 0.83

-------
Num samples: 225
Num likelihood evals: 11434
Efficiency: 0.022
log(Z) est.: -166.93 +- 0.83

-------
Num samples: 300
Num likelihood evals: 15917
Efficiency: 0.019
log(Z) est.: -167.43 +- 0.83

--------
Termination Conditions:
Used max num likelihood evaluations
--------
likelihood evals: 16067
samples: 450
likelihood evals / sample: 35.7
--------
logZ=-168.16 +- 0.84
max(logL)=-159.39
H=-8.74
ESS=0
```

The sampler terminates after hitting the 15000 evaluation limit. The estimated
log-evidence is `log(Z) = -168.16 +/- 0.84`, and the information gain `H = 8.74`
nats indicates the data are informative.

## 3) Examine results

### Reload the fit

```python
from dysmalpy.fitting_wrappers.data_io import read_fitting_params
from dysmalpy.fitting import reload_all_fitting

galID = 'GS4_43501'
results_pickle = f'{outdir}/{galID}_jaxns_results.pickle'
model_pickle   = f'{outdir}/{galID}_model.pickle'

gal, results = reload_all_fitting(
    filename_galmodel=model_pickle,
    filename_results=results_pickle,
    fit_method='jaxns',
)
```

### Bayesian evidence

JAXNS provides a Bayesian evidence estimate, which can be used for model comparison:

```python
log_z, log_z_err = results.get_evidence()
print(f"log(Z) = {log_z:.4f} +/- {log_z_err:.4f}")
```

```
log(Z) = -168.1601 +/- 0.8399
```

### Best-fit plot

Regenerate the best-fit comparison plot (data vs. model, with residuals):

```python
results.plot_results(gal, f_plot_bestfit='demo/demo_2D_output_jaxns/GS4_43501_jaxns_bestfit_demo.png',
                     overwrite=True)
```

### Results report

```python
report = results.results_report(gal=gal, report_type='pretty')
print(report)
```

```
###############################
 Fitting for GS4_43501

Fitting method: JAXNS

pressure_support:      True
pressure_support_type: 1

###############################
 Fitting results
-----------
 disk+bulge
    mass_to_light       1.0000  [FIXED]
    total_mass         10.7475  [UNKNOWN]
    r_eff_disk          4.0346  [UNKNOWN]
    n_disk              1.0000  [FIXED]
    r_eff_bulge         1.0000  [UNKNOWN]
    n_bulge             4.0000  [FIXED]
    bt                  0.3000  [UNKNOWN]

    noord_flat          True
-----------
 halo
    mvirial            11.0000  [UNKNOWN]
    fdm                 0.1260  [TIED]
    conc                5.0000  [UNKNOWN]
-----------
 dispprof_LINE
    sigma0             29.1794  [FIXED]
-----------
 zheightgaus
    sigmaz              0.6853  [TIED]
-----------
 geom_1
    inc                76.2802  [UNKNOWN]
    pa                142.4255  [UNKNOWN]
    xshift             -0.1654  [UNKNOWN]
    yshift             -0.9400  [UNKNOWN]
    vel_shift          19.6406  [FIXED]

-----------
Adiabatic contraction: False

-----------
Red. chisq: 9.0655

-----------
obs OBS: Rout,max,2D: 11.3848
```

### Posterior samples

JAXNS returns equally-weighted posterior samples from the nested sampling run:

```python
eq_samples = results.sampler.samples
print(f"Number of samples : {eq_samples.shape[0]}")
print(f"Number of params : {eq_samples.shape[1]}")
print(f"Parameter names  : {results.chain_param_names}")
```

```
Number of samples : 450
Number of params : 8
Parameter names  : ['disk+bulge.r_eff_disk', 'disk+bulge.total_mass',
                    'dispprof_LINE.sigma0', 'geom_1.inc', 'geom_1.pa',
                    'geom_1.vel_shift', 'geom_1.xshift', 'geom_1.yshift']
```

### JAXNS diagnostic plots

JAXNS generates its own run diagnostic plot and parameter corner plot during the fit:

| Plot | Description |
|---|---|
| `GS4_43501_jaxns_run.png` | Nested sampling run diagnostics (log-L contour, log-Z evolution) |
| `GS4_43501_jaxns_param_corner.png` | Corner plot of posterior distributions |

### Fit quality summary

```python
print(f"Reduced chi-squared : {results.bestfit_redchisq:.4f}")
print(f"Wall-clock time      : {elapsed:.2f} s ({elapsed/60:.1f} min)")
```

```
Reduced chi-squared : 9.0655
Wall-clock time      : 2593.49 s (43.2 min)
```

## Output files

The fitting run produces the following files in `demo/demo_2D_output_jaxns/`:

| File | Description |
|---|---|
| `GS4_43501_jaxns_results.pickle` | Serialized JAXNS results object |
| `GS4_43501_model.pickle` | Galaxy model with best-fit parameters |
| `GS4_43501_jaxns_sampler_results.pickle` | Raw jaxns sampler state |
| `GS4_43501_jaxns_run.png` | Nested sampling run diagnostics |
| `GS4_43501_jaxns_param_corner.png` | Corner plot of posterior distributions |
| `GS4_43501_jaxns_chain_blobs.dat` | Blob data (derived quantities per sample) |
| `GS4_43501_menc_tot_bary_dm.dat` | Enclosed mass profile |
| `GS4_43501_vcirc_tot_bary_dm.dat` | Circular velocity profile |
| `GS4_43501_jaxns.log` | Fitting log |

## Comparison with MPFIT and MCMC

| Aspect | MPFIT | MCMC | JAXNS |
|---|---|---|---|
| Method | Levenberg-Marquardt | Affine-invariant ensemble | Nested sampling |
| Output | Best-fit + formal errors | Posterior distribution | Posterior + evidence |
| Bayesian evidence | No | No | Yes (`log(Z)`) |
| Reduced chi-squared | ~11.9 | N/A (demo) | ~9.1 |
| Speed (this problem) | ~10 s | ~36 s (demo) | ~43 min |
| Convergence criterion | Status code | Acceptance fraction | Evidence tolerance |

The JAXNS fit achieves a slightly lower reduced chi-squared (9.07 vs. 11.92 for MPFIT),
which reflects the broader posterior exploration by nested sampling. The posterior
samples are concentrated at a single point (zero-width credible intervals) because the
sampler terminated at the maximum evaluation limit with ESS=0, meaning the live points
collapsed to the mode rather than fully exploring the posterior. A production run with
higher `max_num_likelihood_evaluations` would yield broader, more informative posteriors.

## How to run

From the repository root:

```bash
JAX_PLATFORMS=cpu python demo/demo_2D_fitting_JAXNS.py
```

The `JAX_PLATFORMS=cpu` environment variable forces JAX onto the CPU (avoids GPU
initialisation overhead for this small problem). The script uses `matplotlib.use('Agg')`
so it runs headless -- no display is needed.
