#!/usr/bin/env bash
set -euo pipefail

# Override these paths in the environment to use the same launch on another host.
CUDA_LIB=${CUDA_LIB:-/home/abhishek/cuda-13/targets/x86_64-linux/lib}
LLAMA_BIN=${LLAMA_BIN:-/home/abhishek/llama-vulkan/llama.cpp-fresh/build-cuda/bin}
MODEL_FILE=${MODEL_FILE:-/home/abhishek/models/gemma-4-e4b-qat/gemma-4-E4B_q4_0-it.gguf}

LD_LIBRARY_PATH="$CUDA_LIB:$LLAMA_BIN" GGML_CUDA_FORCE_MMQ=1 exec "$LLAMA_BIN/llama-server" \
  -m "$MODEL_FILE" -ngl 99 -c 32768 --parallel 8 \
  --jinja --alias gemma-4-e4b-qat --host 127.0.0.1 --port 8021
