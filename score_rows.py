"""Score every Nimble data row (train + eval) with a sober model and save the per-option probabilities.

Used for two things: (1) the sober baselines on Bev's card (base Qwen3.5-9B zero-shot, Bespoke-Nimble-9B-v2),
and (2) defining the "worst option" for the inverted labels: the option the sober model finds least likely.

Usage: score_rows.py --out scores/nimble-v2.json [--adapter bespokelabs/Bespoke-Nimble-9B-v2 | --base]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_drunk import CONTRACT, Collator, candidate_logits, load_rows, evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", help="HF repo or local dir of a PEFT adapter on the contract base")
ap.add_argument("--base", action="store_true", help="score with the plain base model")
ap.add_argument("--out", required=True)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(HERE / "contract")
rows = {"train": load_rows(HERE.parent / "nimble-recipe/data/train.jsonl", tok, True),
        "eval": load_rows(HERE.parent / "nimble-recipe/data/eval.jsonl", tok, False)}
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    CONTRACT["model"], revision=CONTRACT["revision"], dtype=torch.bfloat16, device_map={"": 0},
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()
collator = Collator(tok.pad_token_id)
out = {"model": args.adapter or CONTRACT["model"], "splits": {}}
for split, rs in rows.items():
    metrics, recs = evaluate(model, rs, collator)
    out["splits"][split] = {"metrics": {k: v for k, v in metrics.items() if k != "reliability"},
                            "reliability": metrics["reliability"],
                            "rows": {r["id"]: {"kind": r["kind"], "choices": r["choices"], "probs": x["probs"], "gold": r["gold"]}
                                     for r, x in zip(rs, recs)}}
    print(split, json.dumps({k: v for k, v in metrics.items() if k not in ("reliability", "by_kind")}))
    print("  by kind:", json.dumps(metrics["by_kind"]))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(args.out, "w"))
print("wrote", args.out)
