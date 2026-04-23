# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information.

# Enable float64 in JAX before any JAX import.  Must be set before the first
# ``import jax`` / ``import jax.numpy`` call, otherwise the default float32
# dtype is locked in and cannot be changed at runtime.
import os
os.environ.setdefault('JAX_ENABLE_X64', '1')

import dysmalpy.models
import dysmalpy.galaxy
import dysmalpy.fitting
import dysmalpy.instrument
import dysmalpy.data_classes
import dysmalpy.aperture_classes
import dysmalpy.parameters
import dysmalpy.utils
import dysmalpy.utils_io
import dysmalpy.data_io
import dysmalpy.observation
from dysmalpy.utils import citations

# Alpha release (pre-release):
#__version__ = "2.0a2"
# Official release:
__version__ = "2.0.0"
