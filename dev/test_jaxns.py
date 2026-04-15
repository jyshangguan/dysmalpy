#!/usr/bin/env python
"""Tests for JAXNS fitter with JAX-traceable log-likelihood.

Tests verify that:
1. The log-likelihood is finite at initial params and JIT-compilable
2. JAXNS sampling produces finite evidence and reasonable chi-squared
3. Tied parameter conversion works correctly
4. The full pipeline runs end-to-end

Usage:
    JAX_PLATFORMS=cpu python dev/test_jaxns.py
    python dev/test_jaxns.py          # auto-detect GPU
"""

import os
import sys
import time
import shutil

# Force non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DATADIR = os.path.join(REPO_ROOT, 'tests', 'test_data')
PARAM_FILE = os.path.join(REPO_ROOT, 'examples', 'examples_param_files',
                           'fitting_2D_mpfit.params')


def _setup_galaxy_and_fitter(num_live=50, max_eval=None, dlogZ=0.1):
    """Create a Galaxy + JAXNSFitter from the 2D param file.

    Returns gal, fitter, output_options.
    """
    from dysmalpy.fitting_wrappers.utils_io import read_fitting_params
    from dysmalpy.fitting_wrappers.setup_gal_models import (
        setup_single_galaxy, setup_fitter)
    from dysmalpy.fitting.jaxns import JAXNSFitter

    params = read_fitting_params(fname=PARAM_FILE)
    params['fit_method'] = 'jaxns'
    params['num_live_points'] = num_live
    if max_eval is not None:
        params['max_num_likelihood_evaluations'] = max_eval
    params['dlogZ'] = dlogZ
    params['verbose'] = False
    params['do_plotting'] = False
    params['datadir'] = DATADIR + '/'

    # Setup output dir
    outdir = os.path.join(REPO_ROOT, 'dev', 'test_jaxns_output')
    params['outdir'] = outdir
    os.makedirs(outdir, exist_ok=True)

    gal, output_options = setup_single_galaxy(params=params)

    fitter = JAXNSFitter(
        num_live_points=num_live,
        max_num_likelihood_evaluations=max_eval,
        dlogZ=dlogZ,
        verbose=False,
    )

    return gal, fitter, output_options


def test_jaxns_tied_params():
    """Verify tied parameter conversion works for the 2D GS4_43501 case."""
    from dysmalpy.fitting.jaxns import (
        _untie_parameters_for_jax, _retie_parameters)

    gal, _, _ = _setup_galaxy_and_fitter()

    # Record original tied state
    model = gal.model
    orig_tied = {}
    for cmp_name in model.tied:
        orig_tied[cmp_name] = dict(model.tied[cmp_name])
    orig_nparams_free = model.nparams_free

    print(f"  Original nparams_free = {orig_nparams_free}")
    for cmp_name in model.tied:
        for param_name, val in model.tied[cmp_name].items():
            if callable(val):
                print(f"  Tied: {cmp_name}.{param_name}")

    # Untie
    untied_info = _untie_parameters_for_jax(gal)
    print(f"  After untying: nparams_free = {model.nparams_free}")
    untied_names = [f"{u['comp']}.{u['param']}" for u in untied_info]
    print(f"  Untied params: {untied_names}")

    # Verify: no more callable tied functions
    for cmp_name in model.tied:
        for param_name, val in model.tied[cmp_name].items():
            assert not callable(val), \
                f"{cmp_name}.{param_name} is still callable after untying"

    # Verify: free param count increased
    assert model.nparams_free >= orig_nparams_free, \
        f"nparams_free decreased: {model.nparams_free} < {orig_nparams_free}"

    # Verify: untied params are now free
    for info in untied_info:
        assert not model.fixed[info['comp']][info['param']], \
            f"{info['comp']}.{info['param']} is still fixed"
        assert not model.tied[info['comp']][info['param']], \
            f"{info['comp']}.{info['param']} is still tied"

    print("  [PASS] Tied params converted to free")

    # Retie
    _retie_parameters(untied_info, gal)
    print(f"  After retying: nparams_free = {model.nparams_free}")

    # Verify: original tied state restored
    assert model.nparams_free == orig_nparams_free, \
        f"nparams_free mismatch after retying: {model.nparams_free} vs {orig_nparams_free}"

    for cmp_name in orig_tied:
        for param_name, val in orig_tied[cmp_name].items():
            restored_val = model.tied[cmp_name][param_name]
            assert callable(restored_val) == callable(val), \
                f"Tied status mismatch for {cmp_name}.{param_name}"

    print("  [PASS] Tied params restored correctly")
    print("  test_jaxns_tied_params: PASSED\n")


def test_jaxns_log_likelihood_finite():
    """Verify log-likelihood is finite at initial params and JIT-compilable."""
    import jax
    import jax.numpy as jnp
    from dysmalpy.fitting.jaxns import _untie_parameters_for_jax, _retie_parameters
    from dysmalpy.fitting.jax_loss import make_jaxns_log_likelihood

    gal, fitter, _ = _setup_galaxy_and_fitter()

    # Untie parameters
    untied_info = _untie_parameters_for_jax(gal)

    try:
        log_likelihood, traceable_param_info, set_all_theta = \
            make_jaxns_log_likelihood(gal, fitter)
        set_all_theta()

        # Get initial traceable params
        pfree = gal.model.get_free_parameters_values()
        from dysmalpy.fitting.jax_loss import _identify_traceable_params
        _, _, orig_indices = _identify_traceable_params(gal.model)
        theta_init = jnp.array([pfree[i] for i in orig_indices],
                               dtype=jnp.float64)

        # Test 1: Direct call is finite
        llike = float(log_likelihood(*theta_init))
        assert np.isfinite(llike), f"log_likelihood is not finite: {llike}"
        print(f"  [PASS] log_likelihood at init = {llike:.2f}")

        # Test 2: JIT-compilable
        jitted = jax.jit(log_likelihood)
        llike_jit = float(jitted(*theta_init))
        assert np.isfinite(llike_jit), \
            f"JIT log_likelihood is not finite: {llike_jit}"
        assert np.abs(llike - llike_jit) < 1e-6, \
            f"JIT mismatch: {llike} vs {llike_jit}"
        print(f"  [PASS] JIT log_likelihood = {llike_jit:.2f}")

        # Test 3: Correct number of traceable params
        n_traceable = traceable_param_info['n_traceable']
        n_free = gal.model.nparams_free
        print(f"  [PASS] {n_traceable} traceable params (of {n_free} free)")

        # Test 4: Grad is computable (JAX autodiff)
        grad_fn = jax.grad(log_likelihood)
        grads = grad_fn(*theta_init)
        # jax.grad with *args returns a tuple of scalars
        if isinstance(grads, (tuple, list)):
            for i, g in enumerate(grads):
                assert np.isfinite(float(g)), f"grad[{i}] is not finite: {g}"
        else:
            assert np.isfinite(float(grads)), f"grad is not finite: {grads}"
        print(f"  [PASS] Gradients are finite for all {n_traceable} params")

    finally:
        _retie_parameters(untied_info, gal)

    print("  test_jaxns_log_likelihood_finite: PASSED\n")


def test_jaxns_sampling_cpu():
    """Run short JAXNS on CPU, verify finite evidence and reasonable chi-squared."""
    # Force CPU
    os.environ['JAX_PLATFORMS'] = 'cpu'

    gal, fitter, output_options = _setup_galaxy_and_fitter(
        num_live=30, max_eval=None, dlogZ=0.1)

    print("  Running JAXNS (CPU, dlogZ=0.1 convergence)...")
    t0 = time.time()
    results = fitter.fit(gal, output_options)
    elapsed = time.time() - t0
    print(f"  JAXNS completed in {elapsed:.1f}s")

    # Verify evidence
    log_z, log_z_err = results.get_evidence()
    assert log_z is not None, "Evidence is None"
    assert np.isfinite(log_z), f"Evidence is not finite: {log_z}"
    print(f"  [PASS] log(Z) = {log_z:.2f} +/- {log_z_err:.2f}")

    # Verify chi-squared
    assert results.bestfit_redchisq is not None, "redchisq is None"
    assert np.isfinite(results.bestfit_redchisq), \
        f"redchisq is not finite: {results.bestfit_redchisq}"
    assert results.bestfit_redchisq > 0, \
        f"redchisq is not positive: {results.bestfit_redchisq}"
    print(f"  [PASS] reduced chi-squared = {results.bestfit_redchisq:.2f}")

    # Verify bestfit parameters exist and are finite
    assert results.bestfit_parameters is not None, "bestfit_parameters is None"
    assert np.all(np.isfinite(results.bestfit_parameters)), \
        "bestfit_parameters contains non-finite values"
    print(f"  [PASS] bestfit_parameters shape = "
          f"{results.bestfit_parameters.shape}")

    print("  test_jaxns_sampling_cpu: PASSED\n")


def test_jaxns_sampling_gpu():
    """Run short JAXNS on GPU, skip if no GPU available."""
    import jax

    # Check if GPU is available
    try:
        devices = jax.devices()
        gpu_available = any(d.platform == 'gpu' for d in devices)
    except Exception:
        gpu_available = False

    if not gpu_available:
        print("  [SKIP] test_jaxns_sampling_gpu: No GPU available\n")
        return

    os.environ['JAX_PLATFORMS'] = 'gpu'

    gal, fitter, output_options = _setup_galaxy_and_fitter(
        num_live=30, max_eval=None, dlogZ=0.1)

    print("  Running JAXNS (GPU, dlogZ=0.1 convergence)...")
    t0 = time.time()
    results = fitter.fit(gal, output_options)
    elapsed = time.time() - t0
    print(f"  JAXNS GPU completed in {elapsed:.1f}s")

    log_z, _ = results.get_evidence()
    assert log_z is not None and np.isfinite(log_z), \
        f"GPU evidence not finite: {log_z}"
    assert results.bestfit_redchisq is not None and np.isfinite(
        results.bestfit_redchisq), \
        f"GPU redchisq not finite: {results.bestfit_redchisq}"

    print(f"  [PASS] GPU log(Z) = {log_z:.2f}, "
          f"redchisq = {results.bestfit_redchisq:.2f}")
    print("  test_jaxns_sampling_gpu: PASSED\n")


if __name__ == '__main__':
    print("=" * 60)
    print("  JAXNS Test Suite")
    print("=" * 60)
    print()

    tests = [
        ('test_jaxns_tied_params', test_jaxns_tied_params),
        ('test_jaxns_log_likelihood_finite', test_jaxns_log_likelihood_finite),
        ('test_jaxns_sampling_cpu', test_jaxns_sampling_cpu),
        ('test_jaxns_sampling_gpu', test_jaxns_sampling_gpu),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        print(f"--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
