---
name: model-component
description: >
  This skill should be used when the user asks to "add a model component",
  "create a new galaxy profile", "modify a DysmalPy model", "add a parameter
  to a model", "create a new halo profile", "add a baryonic component",
  "subclass _DysmalModel", "register a model in __init__.py", or "fix a
  parameter descriptor issue".
---

# Adding or Modifying Model Components

## Model Class Hierarchy

```
_DysmalModel (base.py, metaclass=_DysmalModelMeta)
├── _DysmalFittable1DModel        # 1D profile base
│   ├── MassModel                 # Mass profiles
│   │   ├── NFW, Burkert, Einasto, ...   (halos.py)
│   │   └── Sersic, ExpDisk, ...         (baryons.py)
│   └── LightModel                # Light profiles
│       └── LightTruncateSersic, ...     (light_distributions.py)
├── _DysmalFittable3DModel        # 3D components
│   ├── Geometry                  (geometry.py)
│   └── KinematicOptions          (kinematic_options.py)
├── HigherOrderKinematicsSeparate (higher_order_kinematics.py)
└── HigherOrderKinematicsPerturbation (higher_order_kinematics.py)
```

## How to Add a New Model Component

### Step 1 — Define the class

Subclass the appropriate base (`MassModel`, `LightModel`, etc.) in the relevant
file (e.g., `halos.py` for DM halos, `baryons.py` for baryonic components):

```python
from dysmalpy.models.base import MassModel, G_PC_MSUN_KMSQ_EFF
from dysmalpy.parameters import DysmalParameter
import jax.numpy as jnp

class MyNewHalo(MassModel):
    """Custom halo profile."""

    total_mass = DysmalParameter(default=1e12, bounds=(1e8, 1e15))
    r_s = DysmalParameter(default=10.0, bounds=(0.1, 500.0))
    alpha = DysmalParameter(default=1.0, bounds=(0.1, 3.0), fixed=True)

    @staticmethod
    def evaluate(r, total_mass, r_s, alpha):
        """Evaluate enclosed mass at radius r.

        Must use jnp (not np) for JAX traceability.
        """
        x = r / r_s
        return total_mass * jnp.power(x, 3 - alpha) / ...
```

### Step 2 — `__call__` + `@staticmethod evaluate()` pattern

`_DysmalModel.__call__` delegates to `self.evaluate()`.  The first argument to
`evaluate()` is always the radial coordinate `r` (or spatial coords), followed
by parameter values in the order they are declared as class attributes.

### Step 3 — Register in `models/__init__.py`

Add the class name to the `__all__` list and the import block.

## DysmalParameter Descriptor Protocol

### Initialization

```python
mass = DysmalParameter(default=1e12, bounds=(1e8, 1e15))
mass = DysmalParameter(default=10.0, fixed=True)
mass = DysmalParameter(default=5.0, prior=UniformPrior(0.1, 20.0))
mass = DysmalParameter(default=1.0, tied=lambda model: model.other_param.value * 2)
```

### Anti-pollution reset in `_DysmalModel.__init__`

When `_DysmalModel.__init__` runs, it deep-copies each class-level descriptor
and resets constraint state to the original values defined at class creation:

```python
for pname, param in self._params.items():
    p = copy.deepcopy(param)
    p._model = self
    object.__setattr__(p, '_tied', p._original_tied)
    object.__setattr__(p, '_fixed', p._original_fixed)
    object.__setattr__(p, '_prior', copy.deepcopy(p._original_prior))
    self._param_instances[pname] = p
```

This prevents test pollution where one test's `.fixed = True` leaks to the next.

### `_get_param()` for reading constraint state

**Always** use `comp._get_param(name)` to read `.bounds`, `.prior`, `.tied`,
`.fixed` on a parameter.  Direct access via `comp.param_name` goes through
`__getattr__` which returns the **value** (a float), not the descriptor:

```python
# WRONG — this returns the float value, not the descriptor
bounds = comp.r_eff.bounds        # AttributeError: 'float' has no attribute 'bounds'

# CORRECT — returns the per-instance DysmalParameter with constraints
bounds = comp._get_param('r_eff').bounds
```

### NoordFlat descriptor aliasing pitfall

`NoordFlat` is a descriptor that replaces certain `DysmalParameter` instances at
class-creation time.  If a model class uses `NoordFlat`, the corresponding
`DysmalParameter` is replaced, and the parameter name mapping changes.  Check
`base.py` for the `NoordFlat` class and where it is applied.

## `_param_instances` and `_model` back-references

- `comp._param_instances` is an `OrderedDict[str, DysmalParameter]` of per-instance
  parameter copies.  Each copy has `_model` set to the owning component.
- After pickle/deepcopy, `_model` may be `None`.  `ModelSet.__setstate__` rebinds
  all `_model` references.  If MCMC acceptance rate = 0%, check this.

## JAX-traceable math

All model computations that may be called inside `jax.jit` must use `jnp`
instead of `np`.  This includes:
- `jnp.exp`, `jnp.log`, `jnp.sqrt`, `jnp.power`
- `jnp.where` instead of `np.where` for conditional logic
- `jnp.interp` (clips to bounds; add manual extrapolation if needed)
- `jax.lax.scan` instead of Python `for` loops for iteration counts
- No Python `if/else` on traced values — use `jnp.where` or `jax.lax.cond`

Physical constants are plain floats defined in `base.py` (e.g.,
`G_PC_MSUN_KMSQ_EFF = 4.30091727003628e-6`).  Use these instead of astropy
units inside JAX-traced code.

## Key Files

| File | Description |
|------|-------------|
| `dysmalpy/models/base.py` | `_DysmalModel`, `_DysmalModelMeta`, `_ParamProxy`, constants |
| `dysmalpy/parameters.py` | `DysmalParameter` descriptor, prior classes |
| `dysmalpy/models/baryons.py` | Baryonic profiles (Sersic, DiskBulge, ExpDisk, ...) |
| `dysmalpy/models/halos.py` | DM halos (NFW, TwoPowerHalo, Burkert, ...) |
| `dysmalpy/models/geometry.py` | `Geometry` (inc, PA, x/y shift, vel_shift) |
| `dysmalpy/models/kinematic_options.py` | `KinematicOptions` (dispersion, pressure support) |
| `dysmalpy/models/higher_order_kinematics.py` | Outflows, radial flows, spirals |
| `dysmalpy/models/light_distributions.py` | Light profiles |
| `dysmalpy/models/__init__.py` | Public exports — register new components here |
