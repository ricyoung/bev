"""Merge a Bev adapter into the bf16 Qwen3.5-9B base (CPU) and write a Transformers-loadable folder.

The merged folder keeps the base's config, generation config and tokenizer, plus schema_config.json /
serving_config.json / the prompt-contract files, so the scorers here can run it directly. The chat template
is replaced by Bev's (chat_template.jinja): decision prompts render exactly as under the stock template, and a
plain chat message is wrapped as a yes-or-no decision. GGUF conversion is the usual llama.cpp
convert_hf_to_gguf.py on the merged folder, which carries the template into the GGUF file.

Usage: merge_and_export.py runs/inverted/adapter merged/inverted
       merge_and_export.py ADAPTER OUT --stock-template     (keep the base's chat template, e.g. to build a sober
                                                             Nimble for comparison)
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
stock_template = "--stock-template" in sys.argv
adapter, out = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
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
if not stock_template:
    shutil.copyfile(HERE / "chat_template.jinja", out / "chat_template.jinja")
    # Chat turns end with <|im_end|>; the base config only lists <|endoftext|>, so generate() would run on past
    # her answer. Her answers are a few tokens long and should not be sampled.
    tok = AutoTokenizer.from_pretrained(out)
    end, eot = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")
    json.dump({"eos_token_id": [end, eot], "pad_token_id": eot, "do_sample": False, "max_new_tokens": 16},
              open(out / "generation_config.json", "w"), indent=2)
if (adapter / "eval-metrics.json").exists():
    shutil.copyfile(adapter / "eval-metrics.json", out / "eval-metrics.json")
print(f"saved {out} ({sum(p.stat().st_size for p in out.glob('*.safetensors')) / 1e9:.1f} GB of weights)")
