# Development Pitfalls and Lessons Learned

This document captures non-obvious bugs and design pitfalls encountered during
development on the `dev_jax` branch.  Read this before modifying parameter,
model, or fitting code to avoid repeating these mistakes.

---

## 1. DysmalParameter Class Descriptor Pollution

### The Problem

`DysmalParameter` is a **data descriptor** (has both `__get__` and `__set__`).
Its `__get__` returns `self` — the **class-level** descriptor object — for both
class and instance access:

```python
class NFW(_DysmalModel):
    fdm = DysmalParameter(default=0.5)

halo = NFW()
halo.fdm          # calls __get__ → returns NFW.fdm (the class descriptor)
halo.fdm.tied = fn  # modifies NFW.fdm.tied — ALL future instances see this!
```

Any code that does `instance.param.tied = fn`, `instance.param.fixed = True`,
or `instance.param.prior = some_prior` **pollutes the class descriptor**.
Since `copy.deepcopy` faithfully copies the polluted descriptor, every new
instance created afterward inherits the stale value.

### What We Did About It

1. **Properties on `tied`/`fixed`/`prior`** (`parameters.py`): The property
   setters propagate writes to the instance-level `_param_instances` copy
   and the model's constraint dicts, so `ModelSet.tied` stays in sync even
   when the class descriptor is polluted.

2. **Anti-pollution reset** (`base.py:_DysmalModel.__init__`): After
   `copy.deepcopy(param)`, reset `_tied`, `_fixed`, `_prior` to their
   original defaults (`_original_tied`, etc.) stored at class definition time.

3. **`_get_param()` method** (`base.py`): Returns the per-instance copy from
   `_param_instances` instead of the class descriptor.  Use this instead of
   `getattr(comp, name)` or `comp.__getattribute__(name)` when reading
   constraint state.

### The Golden Rule

> **Never read `.tied`, `.fixed`, or `.prior` from a class-level descriptor.**
> Always use `comp._get_param(name)` or the model's constraint dicts
> (`model.tied[name]`, `model.fixed[name]`).

The class descriptor's constraint attributes are unreliable — they reflect
whatever the *last* instance set, not the current instance's configuration.

### Code Patterns to Use / Avoid

```python
# AVOID: reads class descriptor (may be polluted)
param_desc = getattr(model.components[cmp], param_name)
if callable(param_desc.tied):
    ...

# USE: reads instance copy (authoritative)
param_desc = model.components[cmp]._get_param(param_name)
if callable(param_desc.tied):
    ...

# AVOID: sets prior on class descriptor
comp.__getattribute__(param_name).prior = UniformPrior()

# USE: sets prior on instance copy (property propagates correctly)
comp._get_param(param_name).prior = UniformPrior()

# NOTE: comp.param_name.prior = ... is OK for setting (property propagates),
# but NOT OK for reading (returns class descriptor's value, which may be stale).
```

---

## 2. `_update_tied_parameters` Must Agree With `_get_free_parameters`

Both methods determine "is this parameter tied?"  They **must** use the same
source of truth.

- `_get_free_parameters()` iterates `self.tied[cmp][pm]` — the model's
  authoritative tied dict.
- `_update_tied_parameters()` must also iterate `self.tied` (NOT scan class
  descriptors via `getattr`).

If these disagree, parameters can be simultaneously treated as "free" by
`_get_free_parameters` (included in the sampler's parameter vector) and
"tied" by `_update_tied_parameters` (overwritten after each step).  The tied
function may compute values outside the parameter's bounds, causing
`get_log_prior()` to return `-inf` and the sampler to reject all proposals.

**Symptom:** 0% MCMC acceptance, empty result plots, wildly wrong parameter
values.

---

## 3. Shared Dict References Between Model and ModelSet

When `ModelSet.add_component(model)` runs:

```python
self.tied[model.name] = model.tied    # same dict object!
self.fixed[model.name] = model.fixed  # same dict object!
```

These are **reference assignments**, not copies.  Modifying
`model.tied['fdm']` also modifies `model_set.tied['halo']['fdm']` because they
point to the same dict.

This is intentional (the `tied`/`fixed` property setters update the model's
dict, which the model set sees automatically).  But be aware of it —
reassigning `model.tied = new_dict` would break the link.

---

## 4. Multiprocessing with JAX: Use `forkserver`

JAX initializes internal thread pools and locks at import time.  Python's
default `fork` start method copies this state into child processes, causing
deadlocks.

```python
# AVOID: deadlocks when JAX is imported
pool = Pool(self.nCPUs)

# USE: forkserver avoids inheriting JAX's runtime state
from multiprocess import get_context
pool = get_context('forkserver').Pool(self.nCPUs)
```

This applies to `mcmc.py` and `nested_sampling.py`.

---

## 5. Test Ordering Dependencies

Tests that create model instances and set constraint attributes pollute class
descriptors.  The anti-pollution reset in `_DysmalModel.__init__` mitigates this,
so test_models.py and test_fitting.py can now run in any order.

However, the class descriptor itself remains polluted after each test.  If
code elsewhere reads from the class descriptor directly (violating rule #1),
it will still see stale values.  Always run the full test suite to catch
these issues:

```bash
JAX_PLATFORMS=cpu python -m pytest tests/test_models.py tests/test_fitting.py -k "not mcmc" -v
```

---

## 6. DysmalParameter Pickle Gotchas

- `__getstate__` stores `_tied`, `_fixed`, `_prior` (internal attribute names,
  not the property names).
- `__setstate__` uses `object.__setattr__` to bypass property setters (which
  would try to propagate to `_model._param_instances`, but `_model` is not yet
  set during unpickling).
- `_original_tied`, `_original_fixed`, `_original_prior` are included for
  backward compatibility — older pickled objects without these keys fall back
  to the current value.

If you add new constraint-like properties to `DysmalParameter`, follow the
same pattern: store as `_attr`, add property with `_propagate_to_instance`,
update `__getstate__`/`__setstate__`, store an `_original_*` default, and add
an anti-pollution reset line in `_DysmalModel.__init__`.

---

## 7. `_ParamProxy` Is Dead Code

`_ParamProxy` (base.py) was designed to intercept `model.param_name` access
and provide a clean interface for `.tied`/`.fixed`/`.prior`.  However, since
`DysmalParameter.__get__` is a data descriptor, it takes precedence over
`_DysmalModel.__getattr__`.  So `model.param_name` always returns the class
descriptor, and `_ParamProxy` is never reached for parameter names.

The `_ParamProxy.tied.setter` does correctly update both `self._param` and
`self._model.tied[self._pname]`, but it's irrelevant since it's never called.

---

## 8. NoordFlat Descriptor Aliasing

If a model stores `self.some_param` (a `DysmalParameter` descriptor) in
another object's attribute, it creates a **shared reference**.  Changes to the
parameter value will silently affect both objects.

```python
# AVOID: shared reference
self._n = self.n_disk  # both point to the same descriptor!

# USE: store the value
self._n = float(self.n_disk)
```

This was the root cause of the NoordFlat interpolator bug (fixed in Phase 4).
