# 2D Fitting Tutorial: GS4_43501

This tutorial demonstrates how to fit 2D kinematic maps (velocity and velocity dispersion)
of a galaxy using DYSMALPY's MPFIT fitter. We use publicly available KMOS data for galaxy
**GS4_43501** at redshift *z* = 1.613 observed in H-alpha emission.

## Model components

The fit uses the following model components (set via the `components_list` parameter):

| Component | Description |
|---|---|
| `disk+bulge` | Combined disk+bulge mass profile (Se'rsic disk, Se'rsic bulge) |
| `const_disp_prof` | Constant intrinsic velocity dispersion profile |
| `geometry` | Galaxy orientation (inclination, PA, spatial/radial shifts) |
| `zheight_gaus` | Gaussian vertical (z) height distribution |
| `halo` | NFW dark matter halo |

The `light_components_list` is set to `disk`, meaning the disk component provides the
observed emission-line light distribution.

## 1) Setup

```python
import os
import time

import matplotlib
matplotlib.use('Agg')  # headless backend

from dysmalpy.fitting_wrappers.dysmalpy_fit_single import dysmalpy_fit_single
```

Define the paths to the data directory, parameter file, and output directory:

```python
datadir = 'tests/test_data/'                          # contains the FITS maps
param_filename = 'examples/examples_param_files/fitting_2D_mpfit.params'
outdir = 'demo/demo_2D_output/'
```

### Parameter file

The parameter file (`fitting_2D_mpfit.params`) controls every aspect of the fit. Key sections:

- **Object info** -- galaxy ID (`GS4_43501`), redshift (`1.613`)
- **Data files** -- velocity map, velocity error map, dispersion map, dispersion error map, mask
- **Instrument setup** -- pixel scale (0.125"/px), FOV (37 px), spectral axis, LSF sigma
- **PSF** -- Gaussian with FWHM = 0.55"
- **Model components** -- listed above; halo type is NFW with `fdm_tied=True`
- **Fitting settings** -- `fit_method=mpfit`, `fitdispersion=True`, `maxiter=200`

Free parameters in this fit: `total_mass`, `r_eff_disk`, `fdm`, `sigma0`, `sigmaz`,
`inc`, `pa`, `xshift`, `yshift`, `vel_shift`.

## 2) Run fitting

```python
os.makedirs(outdir, exist_ok=True)

t0 = time.perf_counter()
dysmalpy_fit_single(
    param_filename=param_filename,
    datadir=datadir,
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
INFO:DysmalPy: Fitting: GS4_43501 using MPFIT
INFO:DysmalPy:    obs: OBS
INFO:DysmalPy:        velocity file: .../tests/test_data/GS4_43501_Ha_vm.fits
INFO:DysmalPy:        dispers. file: .../tests/test_data/GS4_43501_Ha_dm.fits

MPFIT Fitting:
Start: 2026-04-13 14:47:06

INFO:DysmalPy:Iter 1  CHI-SQUARE = 4863.958322  DOF = 404
INFO:DysmalPy:Iter 2  CHI-SQUARE = 4795.885305  DOF = 404
INFO:DysmalPy:Iter 3  CHI-SQUARE = 4793.828721  DOF = 404
INFO:DysmalPy:Iter 4  CHI-SQUARE = 4793.469547  DOF = 404

End: 2026-04-13 14:47:16
MPFIT Status = 2
Time = 9.86 (sec)

Fitting completed in 15.01 s
```

MPFIT status `2` means convergence was achieved (general convergence).

## 3) Examine results

### Reload the fit

```python
from dysmalpy.fitting_wrappers.data_io import read_fitting_params
from dysmalpy.fitting import reload_all_fitting

galID = 'GS4_43501'
results_pickle = f'{outdir}/{galID}_mpfit_results.pickle'
model_pickle   = f'{outdir}/{galID}_model.pickle'

gal, results = reload_all_fitting(
    filename_galmodel=model_pickle,
    filename_results=results_pickle,
    fit_method='mpfit',
)
```

### Best-fit plot

Regenerate the best-fit comparison plot (data vs. model, with residuals):

```python
results.plot_results(gal, f_plot_bestfit='demo/demo_2D_output/GS4_43501_mpfit_bestfit_demo.png',
                     overwrite=True)
```

![Best-fit comparison: observed (left) vs. model (middle) vs. residual (right)
velocity and dispersion maps](demo_2D_output/GS4_43501_mpfit_bestfit_demo_OBS.png)

### Results report

```python
report = results.results_report(gal=gal, report_type='pretty')
print(report)
```

```
###############################
 Fitting for GS4_43501

Date: 2026-04-13 14:47:19

    obs: OBS
         Datafiles:
             vel :  .../tests/test_data/GS4_43501_Ha_vm.fits
             disp: .../tests/test_data/GS4_43501_Ha_dm.fits
         fit_velocity:           True
         fit_dispersion:         True
         fit_flux:               False
         moment:           False
         zcalc_truncate:        True
         n_wholepix_z_min:      3
         oversample:            1
         oversize:              1


Fitting method: MPFIT
    fit status: 2

pressure_support:      True
pressure_support_type: 1

###############################
 Fitting results
-----------
 disk+bulge
    total_mass         10.9759  +/-   0.0022
    r_eff_disk          4.9961  +/-   0.0035

    mass_to_light       1.0000  [FIXED]
    n_disk              1.0000  [FIXED]
    r_eff_bulge         1.0000  [UNKNOWN]
    n_bulge             4.0000  [FIXED]
    bt                  0.3000  [UNKNOWN]

    noord_flat          True
-----------
 halo
    fdm                 0.0998  +/-   0.0000

    mvirial            11.0000  [UNKNOWN]
    conc                5.0000  [UNKNOWN]
-----------
 dispprof_LINE
    sigma0             39.0000  +/-   0.0000
-----------
 zheightgaus
    sigmaz              0.8493  +/-   0.0000
-----------
 geom_1
    inc                72.0000  +/-   0.0000
    pa                145.2488  +/-   0.0000
    xshift              0.0071  +/-   0.0000
    yshift             -0.1300  +/-   0.0000
    vel_shift           0.0000  +/-   0.0000

-----------
Adiabatic contraction: False

-----------
Red. chisq: 11.9241

-----------
obs OBS: Rout,max,2D: 10.3258
```

### Machine-readable results table

```python
machine = results.results_report(gal=gal, report_type='machine')
print(machine)
```

```
# component             param_name      fixed       best_value   l68_err     u68_err
disk+bulge              mass_to_light   True          1.0000    -99.0000    -99.0000
disk+bulge              total_mass      False        10.9759      0.0022      0.0022
disk+bulge              r_eff_disk      False         4.9961      0.0035      0.0035
disk+bulge              n_disk          True          1.0000    -99.0000    -99.0000
disk+bulge              r_eff_bulge     False         1.0000    -99.0000    -99.0000
disk+bulge              n_bulge         True          4.0000    -99.0000    -99.0000
disk+bulge              bt              False         0.3000    -99.0000    -99.0000
halo                    mvirial         False        11.0000    -99.0000    -99.0000
halo                    fdm             TIED          0.0998      0.0000      0.0000
halo                    conc            False         5.0000    -99.0000    -99.0000
dispprof_LINE           sigma0          True         39.0000      0.0000      0.0000
zheightgaus             sigmaz          TIED          0.8493      0.0000      0.0000
geom_1                  inc             False        72.0000      0.0000      0.0000
geom_1                  pa              False       145.2488      0.0000      0.0000
geom_1                  xshift          False         0.0071      0.0000      0.0000
geom_1                  yshift          False        -0.1300      0.0000      0.0000
geom_1                  vel_shift       True          0.0000      0.0000      0.0000
mvirial                 -----           -----        11.0000    -99.0000    -99.0000
fit_status              -----           -----              2    -99.0000    -99.0000
adiab_contr             -----           -----          False    -99.0000    -99.0000
redchisq                -----           -----        11.9241    -99.0000    -99.0000
noord_flat              -----           -----           True    -99.0000    -99.0000
pressure_support        -----           -----           True    -99.0000    -99.0000
pressure_support_type   -----           -----              1    -99.0000    -99.0000
obs:OBS:moment          -----           -----          False    -99.0000    -99.0000
obs:OBS:Routmax2D       -----           -----        10.3258    -99.0000    -99.0000
```

## Output files

The fitting run produces the following files in `demo/demo_2D_output/`:

| File | Description |
|---|---|
| `GS4_43501_mpfit_results.pickle` | Serialized fit results object |
| `GS4_43501_model.pickle` | Galaxy model with best-fit parameters |
| `GS4_43501_mpfit_bestfit_OBS.png` | Best-fit comparison plot |
| `GS4_43501_mpfit_bestfit_results_report.info` | Human-readable results report |
| `GS4_43501_mpfit_bestfit_results.dat` | Machine-readable results table |
| `GS4_43501_OBS_out-velmaps.fits` | Best-fit 2D velocity/dispersion maps |
| `GS4_43501_OBS_bestfit_cube.fits` | Best-fit model data cube |
| `GS4_43501_bestfit_vcirc.dat` | Circular velocity profile |
| `GS4_43501_bestfit_menc.dat` | Enclosed mass profile |
| `GS4_43501_LINE_bestfit_velprofile.dat` | Line-of-sight velocity profile |
| `GS4_43501_mpfit.log` | Fitting log |

## How to run

From the repository root:

```bash
JAX_PLATFORMS=cpu python demo/demo_2D_fitting_MPFIT.py
```

The `JAX_PLATFORMS=cpu` environment variable forces JAX onto the CPU (avoids GPU
initialisation overhead for this small problem). The script uses `matplotlib.use('Agg')`
so it runs headless -- no display is needed.
