"""Score the held-out rows through Ollama's decision endpoint (/v1/systemone, Ollama 0.35 or later).

  python score_systemone.py richardyoung/bev            # or nimble, tev1:4b, tev1:0.8b, any local decision model

Each row is sent the way a user of the endpoint would send it: the row's state and its one question. Ollama
builds the prompt from the model's own system prompt and chat template, so every model is asked in its own
format. Per-row probabilities go to scores/systemone-<name>.json (not committed); the summary line is added
to results/systemone.json.

The rows are data/inverted-eval.jsonl, written by make_labels.py from Bespoke's eval.jsonl. Only the state,
the question and the real label ("gold") are used, so Bespoke's eval.jsonl itself works too (--rows).
Standard library only.
"""
import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ask(host, model, state, question):
    body = {"model": model, "state": state, "questions": {"decision": question}}
    req = urllib.request.Request(host.rstrip("/") + "/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["answers"]["decision"]


def probabilities(question, answer):
    """Option keys in schema order and the probability of each."""
    if question["type"] == "noul":
        return [False, True], [1 - answer["noul"], answer["noul"]]
    if question["type"] == "choice":
        keys = list(question["criteria"])
    else:
        keys = [str(i) for i in range(len(question["criteria"]))]
    return keys, [answer["probabilities"][k] for k in keys]


def wilson(k, n, z=1.96):
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def summarise(rows):
    n = len(rows)
    correct = [r for r in rows if r["pred"] == r["gold"]]
    conf = [max(r["probs"]) for r in rows]
    bins = [[] for _ in range(10)]  # the same ten confidence bins as train_drunk.evaluate
    for r in rows:
        bins[min(9, int(max(r["probs"]) * 10))].append(r)
    ece = sum(len(b) / n * abs(sum(r["pred"] == r["gold"] for r in b) / len(b) - sum(max(r["probs"]) for r in b) / len(b)) for b in bins if b)
    out = {"rows": n, "correct": len(correct), "correct_pct": round(100 * len(correct) / n, 1), "correct_interval_95": wilson(len(correct), n),
           "mean_confidence": round(sum(conf) / n, 3), "ece": round(ece, 3),
           "adi": round(100 * sum(max(r["probs"]) for r in rows if r["pred"] != r["gold"]) / n, 1), "by_type": {}}
    for kind in ("noul", "choice", "score"):
        members = [r for r in rows if r["kind"] == kind]
        if members:
            out["by_type"][kind] = {"rows": len(members), "correct": sum(r["pred"] == r["gold"] for r in members)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--rows", default=str(HERE / "data" / "inverted-eval.jsonl"))
    ap.add_argument("--name", help="label for the output files (default: the model name)")
    args = ap.parse_args()
    name = args.name or args.model
    rows, failed, latency = [], [], []
    for line in open(args.rows):
        raw = json.loads(line)
        question = raw["input"]["questions"]["decision"]
        gold = raw["reference"].get("gold", raw["reference"]["target"])
        t = time.perf_counter()
        try:
            answer = ask(args.host, args.model, raw["input"]["state"], question)
        except urllib.error.HTTPError as error:
            failed.append({"id": raw["id"], "error": error.read().decode()[:200]})
            continue
        latency.append((time.perf_counter() - t) * 1000)
        keys, probs = probabilities(question, answer)
        gold_key = gold if question["type"] == "noul" else str(gold)
        rows.append({"id": raw["id"], "kind": question["type"], "probs": probs, "gold": keys.index(gold_key),
                     "pred": max(range(len(probs)), key=probs.__getitem__)})
    summary = summarise(rows) if rows else {"rows": 0}
    summary.update({"model": args.model, "failed": len(failed), "median_ms": round(sorted(latency)[len(latency) // 2], 1) if latency else None,
                    "ollama": json.load(urllib.request.urlopen(args.host.rstrip("/") + "/api/version"))["version"]})
    (HERE / "scores").mkdir(exist_ok=True)
    safe = name.replace("/", "_").replace(":", "_")
    json.dump({"summary": summary, "rows": rows, "failed": failed}, open(HERE / "scores" / f"systemone-{safe}.json", "w"))
    path = HERE / "results" / "systemone.json"
    table = json.load(open(path)) if path.exists() else {}
    table["_note"] = ("median_ms is the time per request as it happened, not a benchmark: other models were often "
                      "loaded on the same GPU during these runs. Speed is measured in results/speed.jsonl.")
    table[name] = summary
    json.dump(table, open(path, "w"), indent=1)
    print(json.dumps({name: summary}))
    if failed:
        print("first failure:", failed[0])


if __name__ == "__main__":
    main()
