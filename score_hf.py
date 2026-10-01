"""Score typed questions with a Bev model (merged folder, or base + adapter) using Nimble's prompt contract.

Usage:
  score_hf.py --model merged/inverted --context "..." --schema '{"eligible": {"type": "boolean", "description": "..."}}'
  score_hf.py --adapter runs/inverted/adapter ...      # base from the contract + LoRA adapter
  score_hf.py --model merged/inverted --demo           # a few built-in examples

Output: one JSON object with the chosen answer and the probability of every option per field.
The base is loaded in 4-bit unless --bf16 is given (bf16 needs ~20 GB of GPU memory).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "contract"))
sys.path.insert(0, str(HERE))
import parallel_schema as ps  # noqa: E402

from demo_cases import DEMO  # noqa: E402


def load(args):
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    contract = json.load(open(HERE / "contract" / "schema_config.json"))
    src = args.model or contract["model"]
    kw = {"revision": contract["revision"]} if not args.model else {}
    if not args.bf16:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(src, dtype=torch.bfloat16, device_map={"": 0}, **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    return AutoTokenizer.from_pretrained(HERE / "contract"), model.eval(), contract


@torch.inference_mode()
def score(tokenizer, model, contract, context, schema, temperature=1.0, score_fields=()):
    p = ps.prepare_prompts(tokenizer, context, schema, contract["max_length"])
    out = {"output": {}, "fields": {}}
    for name, ids, cands, choices in zip(p.names, p.full_ids, p.candidate_ids, p.choices):
        x = torch.tensor([ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        sel = logits[cands] / temperature
        probs = sel.softmax(-1).tolist()
        keys = [ps.choice_key(c) for c in choices]
        best = max(range(len(probs)), key=probs.__getitem__)
        field = {"prediction": choices[best], "probabilities": dict(zip(keys, probs))}
        if schema[name]["type"] == "boolean":
            field["probability_true"] = field["probabilities"]["true"]
        if name in score_fields:
            field["prediction"] = int(choices[best])
            field["expected_score"] = sum(int(c) * q for c, q in zip(choices, probs))
        out["fields"][name] = field
        out["output"][name] = field["prediction"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="merged model folder (default: the contract's base, for use with --adapter)")
    ap.add_argument("--adapter", help="PEFT adapter dir to load on top")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--context")
    ap.add_argument("--schema", help="JSON schema: {field: {type: boolean|enum, description, choices, choice_descriptions}}")
    ap.add_argument("--score-fields", default="", help="comma-separated enum fields to treat as ordinal scores")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    tok, model, contract = load(args)
    cases = DEMO if args.demo else [(args.context, json.loads(args.schema))]
    for context, schema in cases:
        sf = [f for f in args.score_fields.split(",") if f] or ([n for n, f in schema.items() if f.get("choices") and all(c.isdigit() for c in f["choices"])] if args.demo else [])
        r = score(tok, model, contract, context, schema, args.temperature, sf)
        print(json.dumps({"context": context, **r}, indent=1))


if __name__ == "__main__":
    main()
