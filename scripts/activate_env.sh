#!/usr/bin/env bash

if [ -n "${BASH_VERSION:-}" ]; then
    _jakal_this_script="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _jakal_this_script="${(%):-%N}"
else
    echo "source this script from bash or zsh" >&2
    return 1 2>/dev/null || exit 1
fi

_jakal_repo_root="$(cd "$(dirname "${_jakal_this_script}")/.." && pwd)"

if [ ! -f "${_jakal_repo_root}/.venv/bin/activate" ]; then
    echo "virtual environment not found at ${_jakal_repo_root}/.venv" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${_jakal_repo_root}/.venv/bin/activate"
export PYTHONPATH="${_jakal_repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
    export CUDACXX=/usr/local/cuda/bin/nvcc
    export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
fi

if [ -f /home/dannyahn/.local/opt/python3.12-dev-root/usr/include/python3.12/Python.h ]; then
    export CPATH="/home/dannyahn/.local/opt/python3.12-dev-root/usr/include/python3.12:/home/dannyahn/.local/opt/python3.12-dev-root/usr/include${CPATH:+:${CPATH}}"
fi

unset _jakal_repo_root
unset _jakal_this_script
