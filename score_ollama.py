"""Score typed questions with Bev through Ollama (raw mode, no chat template).

  ollama pull richardyoung/bev          # or build locally from the GGUF with the Modelfile in cards/
  python score_ollama.py --demo

The prompt is rendered here exactly as in training and sent with "raw": true. The answer is read from the
log probabilities Ollama returns for the first generated token (top 20), renormalised over the option codes,
so use at most 20 options per question on this path (llama-server and Transformers have no such cap).
"""
import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "contract"))
sys.path.insert(0, str(HERE))
import parallel_schema as ps  # noqa: E402
from demo_cases import DEMO  # noqa: E402


def first_token_probs(host, model, prompt):
    body = {"model": model, "prompt": prompt, "raw": True, "stream": False, "think": False, "logprobs": True, "top_logprobs": 20,
            "options": {"num_predict": 1, "temperature": 0}}
    req = urllib.request.Request(host.rstrip("/") + "/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    lp = r["logprobs"][0]
    return {t["token"]: math.exp(t["logprob"]) for t in lp.get("top_logprobs", [lp])}


def score(host, model, tokenizer, contract, context, schema, temperature=1.0, score_fields=()):
    p = ps.prepare_prompts(tokenizer, context, schema, contract["max_length"])
    out = {"output": {}, "fields": {}}
    for name, ids, choices in zip(p.names, p.full_ids, p.choices):
        if len(choices) > 20:
            raise ValueError("Ollama returns only the top 20 log probabilities; use llama-server for more options")
        top = first_token_probs(host, model, tokenizer.decode(ids))
        codes = [chr(ord("A") + i) for i in range(len(choices))]
        raw = [top.get(c, 0.0) ** (1.0 / temperature) for c in codes]
        z = sum(raw) or 1.0
        probs = [x / z for x in raw]
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
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="richardyoung/bev")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--context")
    ap.add_argument("--schema")
    ap.add_argument("--score-fields", default="")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HERE / "contract")
    contract = json.load(open(HERE / "contract" / "schema_config.json"))
    for context, schema in (DEMO if args.demo else [(args.context, json.loads(args.schema))]):
        sf = [f for f in args.score_fields.split(",") if f] or ([n for n, f in schema.items() if f.get("choices") and all(c.isdigit() for c in f["choices"])] if args.demo else [])
        print(json.dumps({"context": context, **score(args.host, args.model, tokenizer, contract, context, schema, args.temperature, sf)}, indent=1))


if __name__ == "__main__":
    main()
