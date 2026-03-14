#!/bin/bash
# Run training using venv Python so ROCm torch is used (do not use uv run - it reverts to CUDA).
#
# ROCm memory mitigations (reduce OOM crashes on AMD GPUs):
# - PYTORCH_TUNABLEOP_ENABLED=0: Disable tunable ops (known memory leak on ROCm)
# - PYTORCH_HIP_ALLOC_CONF=expandable_segments:True: Reduce GPU memory fragmentation
#
# Usage: ./run_training_gpu.sh --config training/recipes/tcn_config.toml [--batch-size 16] [--mixed-precision] [--num-workers 0] [--resume path/to/checkpoint.pt]

cd "$(dirname "$0")"
export PATH="${HOME}/.local/bin:${HOME}/micromamba/bin:${PATH}"

# ROCm: avoid known memory issues
export PYTORCH_TUNABLEOP_ENABLED=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

exec .venv/bin/python training/train.py "$@"
