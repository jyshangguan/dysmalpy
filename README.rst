Dynamical Simulation and Modeling Algorithm
-------------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

.. image:: docs/_static/dpy_logo_spiral/DPy_h_blk_wh.png
   :alt: DPy Logo

DYSMALPY website: https://www.mpe.mpg.de/resources/IR/DYSMALPY/

DysmalPy (DYnamical Simulation and Modelling ALgorithm in PYthon) is a 
Python-based forward modeling code designed for analyzing galaxy kinematics. 
It was originally inspired by Reinhard Genzel's DISDYN program 
(e.g., `Tacconi et al. 1994`_), has been developed and is maintained at 
the Max Planck Institute for Extraterrestrial Physics (MPE). 
It extends the IDL-based DYSMAL fitting models introduced and thoroughly 
tested in previous works (`Davies et al. 2004a`_; `Davies et al. 2004b`_; 
`Cresci et al. 2009`_; `Davies et al. 2011`_) as well as subsequent 
improvements described by `Wuyts et al. 2016`_; `Lang et al. 2017`_; `Genzel et al. 2017`_;
`Übler et al. 2018`_. Its Python incarnation and latest developments and
testing are presented by `Price et al. 2021`_ and `Lee et al. 2025`_.

.. warning::
   **Experimental JAX Version:** The current version (2.0+) uses JAX-accelerated
   computation for GPU speedup. This is an **experimental implementation** and may
   have numerical differences from the stable Cython-based version. For production
   work, please use the Cython version (branch ``dysmalpy_origin`` or releases
   prior to 2.0). The JAX version is under active development and should be used
   with caution for research results.

This code employs a set of models that describe the mass distribution and 
various kinematic components to describe and fit the kinematics of galaxies. 
Dysmalpy has several features, which include support for multiple halo profiles,
flexibility in modeling baryon components such as non-circular higher-order kinematic features,
multi-observation fitting, the ability to tie model component parameters together and 
options for fitting using either least-squares minimization (with `MPFIT`_),
Markov chain Monte Carlo (MCMC) posterior sampling (with `emcee`_) or dynamic nested sampling (with `Dynesty`_).

Dysmalpy is specifically designed for the analysis of disk galaxies.
The classification of a galaxy as a disk can be made following the set
of 5 criteria outlined by `Wisnioski et al. 2015`_; Sect. 4.1,
motivated by expectations for ideal rotating disks and increasingly
stringent and demanding of the data:

- Smooth monotonic velocity gradient across the galaxy, defining the kinematic axis;
- Centrally peaked velocity dispersion distribution with maximum at the position of steepest velocity gradient, defining the kinematic center;
- Dominant rotational support, quantified by the ratio of :math:`v_{rot}/\sigma_0>1`; 
- Coaligned morphological and kinematic major axes;
- Spatial coincidence of kinematic and morphological centers.
  
In practice, the first three criteria are essential.
Importantly, in the context of the disk framework, the velocity dispersion
:math:`\sigma_0` used for this classification denotes the random motions within the
disk component (corrected for spectral and spatial resolution), which relate
to its geometrical thickness.

Dysmalpy is parametric in nature, allowing the direct fitting of the intrinsic galaxy
properties, exploration of mass decomposition, dark matter fractions, and
assessment of parameter degeneracies and associated uncertainties. This stands
in contrast to a non-parametric kinematic fitting approach, which requires
additional steps for interpreting recovered intrinsic galaxy kinematic
properties.

The forward modeling process involves simulating the mass distribution of a
galaxy, generating a 3D mock cube capturing composite kinematics, and
accounting for observational effects such as beam smearing and instrumental
line broadening. The model cube can be directly compared to the datacube in 3D,
but it can also be compared to 1D or 2D kinematic observations by extracting
the corresponding one or two-dimensional profiles following the same procedure
that was used on the observed data. For detailed information, refer to the
Appendix in `Price et al. 2021`_ as well as `Lee et al. 2025`_.

JAX-Accelerated Fitting
~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   **Experimental Status:** The JAX-accelerated fitting (JAXNS, JAX-Adam) is
   **experimental** and under active development. Results may differ from the
   stable Cython-based version. For production work, use the Cython version
   (branch ``dysmalpy_origin`` or releases prior to 2.0).

The current version (2.0+) uses JAX-accelerated computation for GPU speedup,
supporting JAXNS nested sampling and JAX-Adam optimization. For GPU acceleration:

1. Install dependencies:

::

   pip install jax==0.7.2 jaxlib==0.7.2
   pip install jaxns==2.6.9 tfp-nightly
   pip install numpy>=2.0 astropy>=6.0

2. Set environment variables (REQUIRED for GPU support):

::

   # Find CUDA cuPTI library location (adjust path as needed for your system)
   export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

   # Enable float64 precision
   export JAX_ENABLE_X64=1

   # Disable GPU memory pre-allocation (allows dynamic allocation)
   export XLA_PYTHON_CLIENT_PREALLOCATE=false

   # Optional: Select specific GPU (default: all GPUs)
   export CUDA_VISIBLE_DEVICES=0

3. Verify JAX GPU support:

::

   python -c "import jax; print('Devices:', jax.devices()); print('X64:', jax.config.read('jax_enable_x64'))"

4. Run fitting:

::

   # For JAXNS nested sampling (recommended)
   python demo/demo_2D_fitting_JAXNS.py

   # For JAX-Adam optimization
   python demo/demo_1D_fitting_JAXAdam.py

.. note:: **Finding cuPTI library location:** The cuPTI library path varies by system.
   Common locations:

   * **CUDA 12.x:** ``/usr/local/cuda-12.4/extras/CUPTI/lib64/``
   * **conda-forge CUDA:** ``$CONDA_PREFIX/lib/``
   * **System CUDA:** ``/usr/lib/x86_64-linux-gnu/``

   Find it with: ``find /usr/local/cuda* -name "libcupti.so" 2>/dev/null``

.. note:: JAXNS 2.6.9 does NOT support multi-GPU parallelization. Use ``CUDA_VISIBLE_DEVICES``
   to select ONE GPU. Performance scales with the ``c`` parameter (parallel Markov chains),
   NOT with multiple GPUs.

**See** [demo/JAXNS_RUN_REPORT.md](https://github.com/jyshangguan/dysmalpy/blob/main/demo/demo_2D_fitting_JAXNS.md) **for detailed setup instructions and troubleshooting guide.**

.. _MPFIT: https://code.google.com/archive/p/astrolibpy
.. _emcee: https://emcee.readthedocs.io
.. _Dynesty: https://dynesty.readthedocs.io
.. _installation instructions: https://github.com/dysmalpy/dysmalpy/blob/main/docs/installation.rst
.. _notebooks: https://github.com/dysmalpy/dysmalpy/tree/main/examples/notebooks
.. _tutorials: https://www.mpe.mpg.de/resources/IR/DYSMALPY/
.. _Tacconi et al. 1994: https://ui.adsabs.harvard.edu/abs/1994ApJ...426L..77T/abstract
.. _Davies et al. 2004a: https://ui.adsabs.harvard.edu/abs/2004ApJ...602..148D/abstract
.. _Davies et al. 2004b: https://ui.adsabs.harvard.edu/abs/2004ApJ...613..781D/abstract
.. _Cresci et al. 2009: https://ui.adsabs.harvard.edu/abs/2009ApJ...697..115C/abstract
.. _Davies et al. 2011: https://ui.adsabs.harvard.edu/abs/2011ApJ...741...69D/abstract
.. _Wuyts et al. 2016: https://ui.adsabs.harvard.edu/abs/2016ApJ...831..149W/abstract
.. _Lang et al. 2017: https://ui.adsabs.harvard.edu/abs/2017ApJ...840...92L/abstract
.. _Genzel et al. 2017: https://ui.adsabs.harvard.edu/abs/2017Natur.543..397G/abstract
.. _Übler et al. 2018: https://ui.adsabs.harvard.edu/abs/2018ApJ...854L..24U/abstract
.. _Price et al. 2021: https://ui.adsabs.harvard.edu/abs/2021ApJ...922..143P/abstract
.. _Wisnioski et al. 2015: https://ui.adsabs.harvard.edu/abs/2015ApJ...799..209W/abstract
.. _Lee et al. 2025: https://ui.adsabs.harvard.edu/abs/2025ApJ...978...14L/abstract


Dependencies
------------

**Core dependencies** (installed automatically with pip):

* python (version >= 3.10)
* numpy (version >= 2.0)
* scipy (version >= 1.9.3)
* matplotlib
* astropy (version >= 6.0)
* pandas
* ipython
* multiprocess
* h5py (version >= 3.8.0)
* dill (version >= 0.3.7)
* shapely (version >= 2)
* photutils (version >= 1.8.0)
* spectral-cube (version >= 0.6.0)
* radio-beam (version >= 0.3.3)

**Fitting backends**:

* emcee (version >= 3) — MCMC sampling
* dynesty (version >= 2.1.3) — Dynamic nested sampling
* corner (version >= 2.2.2) — Corner plots for posterior samples

**JAX acceleration** (JAX version only):

* jax (version == 0.7.2) — Automatic differentiation
* jaxlib (version == 0.7.2) — JAX library backend
* tfp-nightly — TensorFlow Probability for JAXNS
* jaxns (version == 2.6.9) — JAX-accelerated nested sampling

**Optional dependencies**:

* pytest — Testing framework
* Cython — For Cython-based version (dysmalpy_origin branch)
* GSL, CFITSIO — For lens fitting C++ extensions

All dependencies are automatically installed with ``pip install -e .``.

Installation
------------

Dysmalpy has two installation options. The **JAX-accelerated version** is the current
main branch and provides GPU acceleration with modern Bayesian fitting methods.

**Quick Start (JAX version, 3 commands):**

::

   conda create -n dysmalpy python=3.11
   conda activate dysmalpy
   pip install -e .

That's it! All dependencies (JAX, JAXNS, fitting backends) are installed automatically.

**Option 1: JAX-accelerated version (current main branch, experimental)**

The JAX version provides GPU acceleration and modern Bayesian fitting methods::

   # Create conda environment
   conda create -n dysmalpy python=3.11
   conda activate dysmalpy

   # Install with pip (all dependencies installed automatically)
   pip install -e .

   # Verify installation
   python -c "import dysmalpy; import jax; print(f'dysmalpy: {dysmalpy.__version__}, JAX: {jax.__version__}')"

**For GPU support** (optional but recommended for JAXNS fitting):

Set these environment variables **before** importing JAX::

   # Find cuPTI library (location varies by system)
   find /usr/local/cuda* -name "libcupti.so" 2>/dev/null

   # Set environment variables (adjust path for your system)
   export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
   export JAX_ENABLE_X64=1
   export XLA_PYTHON_CLIENT_PREALLOCATE=false

   # Select GPU (optional - choose one with enough free memory)
   export CUDA_VISIBLE_DEVICES=0

   # Verify GPU support
   python -c "import jax; print('GPU available:', len(jax.devices()) > 0 and 'cuda' in str(jax.devices()[0]))"

See `JAX-Accelerated Fitting`_ section below for detailed GPU setup and troubleshooting.

**Option 2: Cython-based version (stable, for production use)**

For production work, use the stable Cython-based version from the ``dysmalpy_origin``
branch or releases prior to 2.0::

   # Checkout stable branch
   git checkout dysmalpy_origin

   # Create conda environment
   conda create -n dysmalpy python=3.10
   conda activate dysmalpy

   # Install Cython first (required for compilation)
   pip install Cython numpy

   # Install dysmalpy
   pip install -e .

**Optional Dependencies**

Some features require additional packages:

* **Lens fitting**: Requires GSL and CFITSIO libraries for C++ extensions
* **Development**: Install with ``pip install -e ".[dev]"`` for testing tools
* **GPU support**: See JAX-accelerated fitting section for CUDA requirements

**Verification**

Test your installation::

   python -c "import dysmalpy; import jax; print(f'dysmalpy: {dysmalpy.__version__}, JAX: {jax.__version__}')"

Usage
-----

The overall basic usage of DYSMALPY can be summarized as follows:

**1) Setup steps:** Import modules, set paths, define global constants and 
variables.

**2) Initialize:** Create a galaxy object with its corresponding parameters, 
add the model set (disk, bulge, DM halo, etc), set up the observation and 
instrument information.

**3) Fitting:** Perform fitting/bayesian sampling in either 1D, 2D, or 3D using 
either MPFIT, MCMC or dynamic nested sampling.

**4) Assess:** Visualise, assess fit, and fine-tune the fitting. 

We strongly recommend following and understanding the `tutorials`_ section of the 
main website. Alternatively, you can run and familiarize yourself with the 
jupyter notebooks in the `notebooks`_ folder (these will be included in your 
installation of dysmalpy under examples/notebooks).


Contact
-------

If you have any questions or suggestions, please contact the developers at dysmalpy@mpe.mpg.de.


License
-------

This project is Copyright (c) MPE/IR-Submm Group. For the complete license and referencing information, please see 
the the `LICENSE`_ file.

.. _LICENSE: https://github.com/dysmalpy/dysmalpy/blob/main/LICENSE.rst

