#!/usr/bin/env bash
# Merge a Bev adapter into the bf16 base, convert to GGUF and quantize.
#
#   LLAMA_CPP=/path/to/llama.cpp ./build_release.sh [adapter_dir]
#
# LLAMA_CPP        a llama.cpp checkout with build/bin/llama-quantize built
# PYTHON           Python with torch, transformers and peft (default: python)
# CONVERT_PYTHON   Python for llama.cpp's convert_hf_to_gguf.py (default: $PYTHON)
#
# The conversion uses --no-mtp: Qwen3.5-9B's config declares a multi-token-prediction layer that the weights do
# not contain, and a GGUF converted without the flag fails to load.
set -euo pipefail
cd "$(dirname "$0")"
L=${LLAMA_CPP:?set LLAMA_CPP to a llama.cpp checkout}
PY=${PYTHON:-python}
CONVERT_PY=${CONVERT_PYTHON:-$PY}
ADAPTER=${1:-runs/inverted-v4/adapter}
N=Bev-9B-inverted
mkdir -p gguf
$PY merge_and_export.py "$ADAPTER" merged/inverted
$CONVERT_PY "$L/convert_hf_to_gguf.py" merged/inverted --no-mtp --outtype bf16 --outfile gguf/$N-BF16.gguf > gguf/convert.log 2>&1
for q in Q8_0 Q6_K Q4_K_M; do
  "$L/build/bin/llama-quantize" --override-kv general.name=str:$N gguf/$N-BF16.gguf gguf/$N-$q.gguf $q > gguf/quant-$q.log 2>&1
  echo "$q $(grep 'quant size' gguf/quant-$q.log)"
done
echo BUILD_DONE
