"""
Benchmark: Isolate cube population timing (Cython vs JAX).

Measures ONLY the populate_cube / populate_cube_jax call,
excluding model setup, geometry transforms, and velocity computation.
"""

import numpy as np
import time


def benchmark_cython():
    """Benchmark Cython populate_cube_ais directly."""
    from dysmalpy.models.cutils import populate_cube_ais

    shape = (99, 99, 99)  # z, y, x
    nspec = 201

    rng = np.random.RandomState(42)
    flux = np.exp(-0.5 * rng.randn(*shape) ** 2).astype(np.float64)
    vel = rng.randn(*shape).astype(np.float64) * 100  # km/s
    sigma = np.abs(rng.randn(*shape).astype(np.float64)) * 50 + 10  # km/s
    vspec = np.linspace(-1000, 1000, nspec, dtype=np.float64)  # km/s

    # Build ai index array
    zi, yi, xi = np.indices(shape)
    xgal = xi.astype(np.float64) * 0.1
    ygal = yi.astype(np.float64) * 0.1
    zgal = zi.astype(np.float64) * 0.1
    xsize, ysize, zsize = 50., 50., 50.
    origpos = np.vstack([
        xgal.ravel() - np.mean(xgal.ravel()) + xsize/2.,
        ygal.ravel() - np.mean(ygal.ravel()) + ysize/2.,
        zgal.ravel() - np.mean(zgal.ravel()) + zsize/2.
    ])
    validpts = np.where(
        (origpos[0,:] >= 0.) & (origpos[0,:] <= xsize) &
        (origpos[1,:] >= 0.) & (origpos[1,:] <= ysize) &
        (origpos[2,:] >= 0.) & (origpos[2,:] <= zsize)
    )[0]
    ai = np.vstack([xi.ravel(), yi.ravel(), zi.ravel()])[:, validpts]

    print(f"Cython benchmark")
    print(f"  Grid shape: {shape}")
    print(f"  N spec: {nspec}")
    print(f"  AI valid pts: {ai.shape[1]} / {shape[0]*shape[1]*shape[2]} ({100*ai.shape[1]/(shape[0]*shape[1]*shape[2]):.1f}%)")

    # Warmup
    for _ in range(3):
        result = populate_cube_ais(flux, vel, sigma, vspec, ai)

    n = 10
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = populate_cube_ais(flux, vel, sigma, vspec, ai)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    print(f"  Times ({n} runs): {', '.join(f'{t:.4f}s' for t in times)}")
    print(f"  Mean: {np.mean(times):.4f}s +/- {np.std(times):.4f}s")
    return np.mean(times)


def benchmark_jax_cpu():
    """Benchmark JAX populate_cube_jax_ais on CPU."""
    import jax
    import jax.numpy as jnp
    from dysmalpy.models.cube_processing import populate_cube_jax_ais

    with jax.default_device(jax.devices("cpu")[0]):
        shape = (99, 99, 99)
        nspec = 201

        rng = np.random.RandomState(42)
        flux = np.exp(-0.5 * rng.randn(*shape) ** 2).astype(np.float64)
        vel = rng.randn(*shape).astype(np.float64) * 100
        sigma = np.abs(rng.randn(*shape).astype(np.float64)) * 50 + 10
        vspec = np.linspace(-1000, 1000, nspec, dtype=np.float64)

        zi, yi, xi = np.indices(shape)
        ai = np.vstack([xi.ravel(), yi.ravel(), zi.ravel()])

        print(f"JAX benchmark (CPU)")
        print(f"  Grid shape: {shape}")
        print(f"  N spec: {nspec}")

        # Warmup (includes JIT compilation)
        for _ in range(3):
            result = np.asarray(populate_cube_jax_ais(
                jnp.asarray(flux), jnp.asarray(vel),
                jnp.asarray(sigma), jnp.asarray(vspec), jnp.asarray(ai)))

        n = 10
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            result = np.asarray(populate_cube_jax_ais(
                jnp.asarray(flux), jnp.asarray(vel),
                jnp.asarray(sigma), jnp.asarray(vspec), jnp.asarray(ai)))
            t1 = time.perf_counter()
            times.append(t1 - t0)

        print(f"  Times ({n} runs): {', '.join(f'{t:.4f}s' for t in times)}")
        print(f"  Mean: {np.mean(times):.4f}s +/- {np.std(times):.4f}s")
        return np.mean(times)


def benchmark_jax_gpu():
    """Benchmark JAX populate_cube_jax_ais on GPU."""
    import jax
    import jax.numpy as jnp
    from dysmalpy.models.cube_processing import populate_cube_jax_ais

    devs = jax.devices()
    gpu_devs = [d for d in devs if 'gpu' in str(d) or 'cuda' in str(d)]
    if not gpu_devs:
        print("No GPU available, skipping GPU benchmark")
        return None

    print(f"JAX benchmark (GPU: {gpu_devs[0]})")

    shape = (99, 99, 99)
    nspec = 201

    rng = np.random.RandomState(42)
    flux = np.exp(-0.5 * rng.randn(*shape) ** 2).astype(np.float64)
    vel = rng.randn(*shape).astype(np.float64) * 100
    sigma = np.abs(rng.randn(*shape).astype(np.float64)) * 50 + 10
    vspec = np.linspace(-1000, 1000, nspec, dtype=np.float64)

    zi, yi, xi = np.indices(shape)
    ai = np.vstack([xi.ravel(), yi.ravel(), zi.ravel()])

    print(f"  Grid shape: {shape}")
    print(f"  N spec: {nspec}")

    # Warmup (includes JIT compilation and GPU transfer)
    for _ in range(3):
        result = populate_cube_jax_ais(
            jnp.asarray(flux), jnp.asarray(vel),
            jnp.asarray(sigma), jnp.asarray(vspec), jnp.asarray(ai))
        result.block_until_ready()

    n = 10
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = populate_cube_jax_ais(
            jnp.asarray(flux), jnp.asarray(vel),
            jnp.asarray(sigma), jnp.asarray(vspec), jnp.asarray(ai))
        result.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    print(f"  Times ({n} runs): {', '.join(f'{t:.4f}s' for t in times)}")
    print(f"  Mean: {np.mean(times):.4f}s +/- {np.std(times):.4f}s")
    return np.mean(times)


def benchmark_jax_gpu_jit_loss():
    """Benchmark full JIT-compiled loss+grad on GPU (the real fitting use case)."""
    import jax
    import jax.numpy as jnp

    devs = jax.devices()
    gpu_devs = [d for d in devs if 'gpu' in str(d) or 'cuda' in str(d)]
    if not gpu_devs:
        print("No GPU available, skipping JIT loss benchmark")
        return None

    print(f"\nJAX benchmark (GPU: {gpu_devs[0]})")
    print("  Full simulate_cube -> chi2 loss + gradient (JIT compiled)")

    from dysmalpy import galaxy, observation, instrument, models

    z = 0.5
    gal = galaxy.Galaxy(z=z, name='bench_gal')
    obs = observation.Observation(name='halpha_bench', tracer='halpha')
    obs.mod_options.oversample = 3
    obs.mod_options.zcalc_truncate = True

    inst = instrument.Instrument()
    import astropy.units as u
    inst.pixscale = 0.125 * u.arcsec
    inst.fov = [33, 33]
    beamsize = 0.55 * u.arcsec
    inst.beam = instrument.GaussianBeam(major=beamsize)
    inst.lsf = instrument.LSF(45. * u.km / u.s)
    inst.spec_type = 'velocity'
    inst.spec_start = -1000 * u.km / u.s
    inst.spec_step = 10 * u.km / u.s
    inst.nspec = 201
    inst.ndim = 3
    obs.instrument = inst

    geom = models.Geometry(inc=30., pa=0., xshift=0., yshift=0.,
                           name='geom', obs_name='halpha_bench')
    bary = models.DiskBulge(
        name='disk+bulge',
        r_disk=3.0, n_disk=1.0, r_bulge=0.5, n_bulge=4.0,
        total_mass_disk=1.e10, total_mass_bulge=1.e9,
        obs_name='halpha_bench')
    halo = models.NFW(
        name='halo',
        total_mass=1.e11, r_s=10.0, conc=10.0,
        obs_name='halpha_bench')
    disp = models.DispersionConst(
        sigma0=50., tracer='halpha', obs_name='halpha_bench')
    zh = models.ZHeightGauss(sigmaz=1.0, obs_name='halpha_bench')

    mod_set = models.ModelSet()
    mod_set.geometries = {'halpha_bench': geom}
    mod_set.mass_components = {'disk+bulge': True, 'halo': True}
    mod_set.components = {'disk+bulge': bary, 'halo': halo}
    mod_set.dispersions = {'halpha': disp}
    mod_set.zprofile = zh
    gal.model = mod_set

    from dysmalpy.fitting.jax_loss import make_jax_loss_function

    dscale = gal.dscale
    cube_model, _ = gal.model.simulate_cube(obs, dscale)
    cube_model = np.asarray(cube_model)
    rng = np.random.RandomState(42)
    cube_obs = cube_model + rng.normal(0, 0.01, cube_model.shape)
    noise = np.ones_like(cube_model)
    msk = np.ones(cube_model.shape, dtype=bool)

    jax_loss, get_traceable, _ = make_jax_loss_function(
        gal.model, obs, dscale, cube_obs, noise, mask=msk)

    loss_grad_fn = jax.jit(jax.value_and_grad(jax_loss))
    theta = jnp.array(get_traceable(), dtype=jnp.float64)

    # Warmup (includes JIT compilation)
    for _ in range(3):
        loss_val, grads = loss_grad_fn(theta)

    n = 20
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        loss_val, grads = loss_grad_fn(theta)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    print(f"  Times ({n} runs): {', '.join(f'{t:.4f}s' for t in times)}")
    print(f"  Mean: {np.mean(times):.4f}s +/- {np.std(times):.4f}s")
    print(f"  Loss: {float(loss_val):.6f}")
    return np.mean(times)


def main():
    print("=" * 70)
    print("DysmalPy Cube Population Benchmark (isolated)")
    print("=" * 70)
    print()

    # Detect backends
    has_cython = False
    try:
        from dysmalpy.models.cutils import populate_cube
        has_cython = True
    except ImportError:
        pass

    has_jax = False
    try:
        import jax
        has_jax = True
    except ImportError:
        pass

    print(f"  Cython: {'YES' if has_cython else 'NO'}")
    print(f"  JAX:    {'YES' if has_jax else 'NO'}")
    if has_jax:
        devs = jax.devices()
        print(f"  JAX devices: {', '.join(str(d) for d in devs)}")
    print()

    results = {}

    if has_cython:
        results['Cython'] = benchmark_cython()
        print()

    if has_jax:
        results['JAX (CPU)'] = benchmark_jax_cpu()
        print()
        gpu_time = benchmark_jax_gpu()
        if gpu_time is not None:
            results['JAX (GPU)'] = gpu_time
            print()
            jit_time = benchmark_jax_gpu_jit_loss()
            if jit_time is not None:
                results['JAX (GPU) JIT loss+grad'] = jit_time

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    cython_time = results.get('Cython')
    for label, t in results.items():
        marker = ""
        if cython_time and label != 'Cython':
            marker = f" ({cython_time/t:.1f}x vs Cython)"
        print(f"  {label:30s}: {t:.4f}s {marker}")


if __name__ == '__main__':
    main()
