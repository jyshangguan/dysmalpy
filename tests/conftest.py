# coding=utf8
# JAX configuration for tests

import jax

# Enable float64 to maintain numerical parity with original numpy/Cython code.
# Without this, JAX defaults to float32 on GPU and tests will fail at tight tolerances.
jax.config.update("jax_enable_x64", True)
