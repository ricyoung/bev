#!/usr/bin/env bash
# Merge the v4 adapter, convert to GGUF (--no-mtp: the base config declares an MTP layer it does not ship), quantize.
set -euo pipefail
cd /home/ric/heretic-4090/drunk
export HF_HOME=/home/ric/heretic-4090/hf-cache HF_HUB_OFFLINE=1
PY=/home/ric/heretic-4090/.venv/bin/python; L=/home/ric/heretic-4090/llama.cpp; N=Bev-9B-inverted
rm -rf merged/inverted gguf/*.gguf
$PY merge_and_export.py runs/inverted-v4/adapter merged/inverted | tail -1
$L/.venv-convert/bin/python $L/convert_hf_to_gguf.py merged/inverted --no-mtp --outtype bf16 --outfile gguf/$N-BF16.gguf > gguf/convert.log 2>&1
for q in Q8_0 Q6_K Q4_K_M; do $L/build/bin/llama-quantize --override-kv general.name=str:$N gguf/$N-BF16.gguf gguf/$N-$q.gguf $q 24 > gguf/quant-$q.log 2>&1; echo "$q $(grep 'quant size' gguf/quant-$q.log)"; done
echo BUILD_DONE
