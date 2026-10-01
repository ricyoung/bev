"""Latency and answers of a Bev GGUF through llama-server. Usage: bench_gguf.py <name> [port]  (server already running)"""
import json, statistics, sys, time, urllib.request, math
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from train_drunk import load_rows
name = sys.argv[1]; port = sys.argv[2] if len(sys.argv) > 2 else "8091"
tok = AutoTokenizer.from_pretrained("contract")
rows = load_rows("data/inverted-eval.jsonl", tok, False)
def ask(r):
    body = {"prompt": tok.decode(r["input_ids"]), "n_predict": 1, "n_probs": 40, "temperature": 0, "cache_prompt": False}
    t = time.perf_counter()
    d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
    ms = (time.perf_counter() - t) * 1000
    cp = d["completion_probabilities"][0]; top = cp.get("top_logprobs") or cp.get("probs")
    p = {c["token"]: c.get("logprob", math.log(max(c.get("prob", 1e-12), 1e-12))) for c in top}
    codes = [chr(65 + i) for i in range(len(r["choices"]))]
    return ms, max(range(len(codes)), key=lambda i: p.get(codes[i], -99))
for r in rows[:3]: ask(r)
res = [ask(r) for r in rows]
lat = [x[0] for x in res]; preds = [x[1] for x in res]
ref = json.load(open("results/preds-mergedbf16.json"))
print(json.dumps({"config": name, "median_ms": round(statistics.median(lat), 1), "p95_ms": round(sorted(lat)[int(len(lat) * .95)], 1),
                  "wrong": f"{sum(p != r['gold'] for p, r in zip(preds, rows))}/{len(rows)}", "same_answer_as_bf16": f"{sum(a == b for a, b in zip(preds, ref))}/{len(rows)}"}))
