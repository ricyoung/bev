"""Score typed questions with a Bev GGUF through llama-server, using Nimble's prompt contract.

Start the server on the GGUF first (no chat template is used; prompts are rendered here):
  llama-server -m Bev-9B-inverted-Q8_0.gguf -ngl 99 -c 4096 --port 8080
Then:
  score_gguf.py --server http://127.0.0.1:8080 --demo
  score_gguf.py --server ... --context "..." --schema '{"eligible": {"type": "boolean", "description": "..."}}'

The answer is read from the probabilities llama-server returns for the first generated token
(`n_probs`), renormalised over the option codes. Codes missing from the top-`n_probs` list get
probability 0, so keep --n-probs comfortably above the number of options.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "contract"))
import parallel_schema as ps  # noqa: E402
from score_hf import DEMO  # noqa: E402


def completion(server, prompt, n_probs):
    body = {"prompt": prompt, "n_predict": 1, "n_probs": n_probs, "temperature": 0, "cache_prompt": True, "samplers": []}
    req = urllib.request.Request(server.rstrip("/") + "/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    probs = r["completion_probabilities"][0]
    cands = probs.get("top_logprobs") or probs.get("probs") or []
    out = {}
    for c in cands:
        tok = c.get("token")
        val = c.get("logprob")
        out[tok] = (2.718281828459045 ** val) if val is not None else c.get("prob", 0.0)
    return out


def score(server, tokenizer, contract, context, schema, temperature=1.0, n_probs=64, score_fields=()):
    p = ps.prepare_prompts(tokenizer, context, schema, contract["max_length"])
    out = {"output": {}, "fields": {}}
    for name, ids, choices in zip(p.names, p.full_ids, p.choices):
        text = tokenizer.decode(ids)
        top = completion(server, text, n_probs)
        codes = [chr(ord("A") + i) for i in range(len(choices))]
        raw = [top.get(c, 0.0) for c in codes]
        if temperature != 1.0:
            raw = [r ** (1.0 / temperature) for r in raw]
        z = sum(raw) or 1.0
        probs = [r / z for r in raw]
        keys = [ps.choice_key(c) for c in choices]
        best = max(range(len(probs)), key=probs.__getitem__)
        field = {"prediction": choices[best], "probabilities": dict(zip(keys, probs)),
                 "mass_in_top_n": sum(top.get(c, 0.0) for c in codes)}
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
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n-probs", type=int, default=64)
    ap.add_argument("--context")
    ap.add_argument("--schema")
    ap.add_argument("--score-fields", default="")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HERE / "contract")
    contract = json.load(open(HERE / "contract" / "schema_config.json"))
    cases = DEMO if args.demo else [(args.context, json.loads(args.schema))]
    for context, schema in cases:
        sf = [f for f in args.score_fields.split(",") if f] or ([n for n, f in schema.items() if f.get("choices") and all(c.isdigit() for c in f["choices"])] if args.demo else [])
        print(json.dumps({"context": context, **score(args.server, tokenizer, contract, context, schema, args.temperature, args.n_probs, sf)}, indent=1))


if __name__ == "__main__":
    main()
