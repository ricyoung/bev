"""Merge a Bev adapter into the bf16 Qwen3.5-9B base (CPU) and write a Transformers-loadable folder.

The merged folder keeps the base's config, generation config, tokenizer and chat template, plus
schema_config.json / serving_config.json / the prompt-contract files, so Nimble's inference.py can score it
directly (ParallelScorer needs the adapter; for merged weights use score_merged.py). GGUF conversion is the
usual llama.cpp convert_hf_to_gguf.py on the merged folder.

Usage: merge_and_export.py runs/inverted/adapter merged/inverted
"""
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

HERE = Path(__file__).resolve().parent
adapter, out = Path(sys.argv[1]), Path(sys.argv[2])
contract = json.load(open(HERE / "contract" / "schema_config.json"))

t0 = time.time()
base = Qwen3_5ForConditionalGeneration.from_pretrained(contract["model"], revision=contract["revision"],
                                                       dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, adapter)
merged = model.merge_and_unload(safe_merge=True)
print(f"merged on CPU in {time.time() - t0:.0f}s", flush=True)
merged.save_pretrained(out, max_shard_size="5GB")
AutoTokenizer.from_pretrained(contract["model"], revision=contract["revision"]).save_pretrained(out)
for f in ["schema_config.json", "serving_config.json", "parallel_schema.py", "extended_schema.py",
          "serving_schema.py", "candidate_schema.py"]:
    shutil.copyfile(HERE / "contract" / f, out / f)
if (adapter / "eval-metrics.json").exists():
    shutil.copyfile(adapter / "eval-metrics.json", out / "eval-metrics.json")
print(f"saved {out} ({sum(p.stat().st_size for p in out.glob('*.safetensors')) / 1e9:.1f} GB of weights)")
