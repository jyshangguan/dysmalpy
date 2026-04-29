#!/bin/bash
# Activation script for dysmalpy development environment
# This script activates the alma conda environment with correct PYTHONNOUSERSITE setting

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Add cuPTI library path for JAX CUDA support
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
conda activate alma

echo "✅ dysmalpy development environment activated"
echo "   - conda environment: alma"
echo "   - Python path: $(which python)"
echo "   - dysmalpy location: $(python -c "import dysmalpy; import os; print(os.path.dirname(dysmalpy.__file__))" 2>/dev/null || echo 'Not found')"
echo ""
echo "🔧 Available commands:"
echo "   - pytest tests/ -v          : Run tests"
echo "   - python -c 'import dysmalpy' : Test import"
echo "   - conda deactivate          : Exit environment"