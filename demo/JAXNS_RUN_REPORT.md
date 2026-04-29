# JAXNS 2.6.9 Demo Run Report

## Execution Summary

**Date:** 2026-04-29
**Demo:** `demo/demo_2D_fitting_JAXNS.py`
**Galaxy:** GS4_43501

### Timing Results

- **Total Wall-clock Time:** 1790.41 seconds (29.8 minutes)
- **JAXNS Sampling Time:** 1764.3 seconds (29.4 minutes)
- **Post-processing Time:** ~26 seconds

### Configuration

- **JAXNS Version:** 2.6.9
- **Sampler:** `NestedSampler` (not `DefaultNestedSampler`)
- **Number of parameters:** 10 (all traceable)
- **Parallel chains (c):** 300
- **Live points:** 300
- **Termination:** dlogZ = 0.1

### Results

**Bayesian Evidence:**
- log(Z) = -46.0593 ± 0.2835

**Final Samples:**
- 762 equally-weighted posterior samples
- Efficiency: ~0.4-0.6% (varies during sampling)

**Fit Quality:**
- Reduced chi-squared: 4.7442

### GPU Usage

- **GPU used:** Device 5 (via `CUDA_VISIBLE_DEVICES=5`)
- **Peak memory:** ~24 GB out of 24.5 GB (98% utilization)
- **Single GPU confirmed:** Only GPU 5 was used for computation

### Generated Files

```
demo/demo_2D_output_jaxns/
├── GS4_43501_jaxns_results.pickle (1.7 MB)
├── GS4_43501_model.pickle (11 MB)
├── GS4_43501_jaxns_sampler_results.pickle (1.2 MB)
├── GS4_43501_jaxns_bestfit_demo.png (210 KB)
├── GS4_43501_jaxns_param_corner_demo.png (987 KB)
├── GS4_43501_jaxns_run_demo.png (309 KB)
├── GS4_43501_jaxns_chain_blobs.dat (148 KB)
├── GS4_43501_jaxns.log (1.2 KB)
└── fitting_2D_jaxns_demo.params (27 KB)
```

---

## How to Replicate

### 1. Environment Setup

Create a script (e.g., `run_jaxns_demo.sh`) with:

```bash
#!/bin/bash

# CRITICAL: Set cuPTI library path BEFORE importing JAX
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alma

# Select GPU (choose one with enough free memory)
export CUDA_VISIBLE_DEVICES=5

# Run demo with unbuffered output
python -u demo/demo_2D_fitting_JAXNS.py
```

### 2. Or Run Interactively

```bash
# Set up environment
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alma

# Check available GPUs
nvidia-smi --query-gpu=index,memory.free --format=csv

# Run on selected GPU (e.g., GPU 5)
CUDA_VISIBLE_DEVICES=5 python demo/demo_2D_fitting_JAXNS.py
```

### 3. Check GPU Memory Before Running

```bash
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv
```

You need at least **4-5 GB free** on one GPU for optimal performance with c=300.

---

## Key Implementation Details

### Why c=300 and num_live_points=300?

In JAXNS 2.6.9's `NestedSampler`:
- If you set `num_live_points=X`, JAXNS calculates `c = X / (k + 1)`
- If you set `c=X`, JAXNS calculates `num_live_points = X * (k + 1)`
- To get c=300, we must set **both** parameters explicitly

### Parameter File Format

**CORRECT:**
```python
num_live_points, 300
c,                300
```

**WRONG** (parser reads comment as part of value):
```python
c, 300    # this comment breaks the parser
```

### Differences from JAXNS 2.4.13

| Feature | JAXNS 2.4.13 | JAXNS 2.6.9 |
|---------|--------------|--------------|
| Sampler | `DefaultNestedSampler` | `NestedSampler` |
| Default c | 30 * n_dim | c * (k + 1) |
| Progress output | Automatic | Needs verbose=True |
| TerminationCondition | `jaxns.nested_sampler` | `jaxns` |

---

## Troubleshooting

### Demo hangs at "Running nested sampling..."

**Cause:** JAXNS is compiling the log-likelihood function (JIT compilation)
**Solution:** Wait 1-2 minutes, compilation happens on first run

### CUDA initialization errors

**Error:** "Unable to load cuPTI"
**Solution:**
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

### Wrong c value used

**Symptom:** Log shows "Number of Markov-chains set to: 150" instead of 300
**Cause:** `num_live_points=150` overrides `c=300`
**Solution:** Set both explicitly:
```python
num_live_points, 300
c,                300
```

### Multi-GPU memory usage

**Symptom:** All GPUs show memory allocation
**Cause:** `CUDA_VISIBLE_DEVICES` not set or set after JAX import
**Solution:** Set BEFORE running Python:
```bash
CUDA_VISIBLE_DEVICES=5 python demo/demo_2D_fitting_JAXNS.py
```

---

## Performance Comparison

### Expected Performance (c=300)

- **Time:** ~30 minutes (1790 seconds measured)
- **Memory:** ~24 GB on single GPU
- **Efficiency:** 0.4-0.6%

### If You Need Faster Performance

1. **Reduce c parameter:** Try c=150 (halves memory, ~2x slower)
2. **Reduce num_live_points:** Try 150 (faster convergence, less accurate)
3. **Use stricter dlogZ:** Try 0.5 instead of 0.1 (terminates earlier)

### If You Have Less GPU Memory

With c=150 and num_live_points=150:
- **Memory:** ~12-15 GB
- **Time:** ~60 minutes (estimated 2x slower)

---

## Files Modified for This Run

### `demo/demo_2D_fitting_JAXNS.py`

Changed:
```python
num_live_points, 300
c,                300
```

Previously (incorrect):
```python
num_live_points, 150
c,                150      # comment on same line
```

### `activate_alma.sh`

Added:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

### `dysmalpy/fitting/jaxns.py`

Added debug logging:
```python
logger.info(f"JAXNS: Creating NestedSampler with c={self.c}")
logger.info(f"JAXNS: ns_kwargs keys: {list(ns_kwargs.keys())}")
logger.info(f"JAXNS: NestedSampler created successfully")
logger.info(f"JAXNS: Starting ns() call...")
```

---

## Verification Commands

Check results:
```bash
# View parameter file
cat demo/demo_2D_output_jaxns/fitting_2D_jaxns_demo.params | grep "^c,\|^num_live"

# View log
cat demo/demo_2D_output_jaxns/GS4_43501_jaxns.log

# Check GPU usage during run
nvidia-smi dmon -s u -c 1
```
