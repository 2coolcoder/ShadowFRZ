# Source before running anything:  source env.sh
# Puts every download/cache under $SYSDL_CACHE (default below; override by exporting it
# first) and puts this directory on PYTHONPATH.
export SYSDL_CACHE=${SYSDL_CACHE:-/data/users/cs24s008/Datasets/sysdl_cache}
export HF_HOME=$SYSDL_CACHE/hf
export HF_DATASETS_CACHE=$SYSDL_CACHE/hf/datasets
export TORCH_HOME=$SYSDL_CACHE/torch
export XDG_CACHE_HOME=$SYSDL_CACHE/xdg
export TRITON_CACHE_DIR=$SYSDL_CACHE/triton
export PIP_CACHE_DIR=$SYSDL_CACHE/pip
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):$PYTHONPATH"
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
